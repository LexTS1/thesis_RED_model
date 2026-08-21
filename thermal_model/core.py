"""Deterministic ISO 13790:2008 5R1C residential thermal core.

The implementation follows the simple hourly method in Annex C.  It keeps the
single dynamic mass state explicit, treats HVAC demand as an ideal signed load
at the air node, and exposes small pure functions for equation-level testing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import (
    DEFAULT_ASSUMPTIONS_PATH,
    ORIENTATIONS,
    ArchetypeStateInput,
    AssumptionContract,
    ContractError,
    PreparedArchetype,
    SimulationDiagnostics,
    SimulationInput,
    SimulationResult,
    load_assumption_contract,
    validate_archetype_state,
    validate_prepared_archetype,
    validate_simulation_input,
    validate_simulation_result,
)


class ThermalCoreError(ContractError):
    """Raised when a physically or numerically valid 5R1C solution is impossible."""


@dataclass(frozen=True)
class BoundaryCorrection:
    """TABULA correction for an opaque construction boundary."""

    additional_resistance_m2K_W: float
    temperature_adjustment_factor: float


@dataclass(frozen=True)
class GainAllocation:
    """Hourly ISO allocation to the air, mass, and surface nodes."""

    Phi_ia_W: float
    Phi_m_W: float
    Phi_st_W: float


@dataclass(frozen=True)
class VentilationState:
    """Hourly ventilation flow and effective heat-transfer state."""

    H_ve_W_K: float
    infiltration_airflow_m3_h: float
    ventilation_airflow_m3_h: float
    effective_hrv_efficiency: float
    bypass_active: bool


@dataclass(frozen=True)
class NodeSolution:
    """One 5R1C evaluation for a prescribed signed air-node HVAC load."""

    theta_mass_end_C: float
    theta_mass_C: float
    theta_surface_C: float
    theta_air_C: float
    signed_hvac_load_W: float
    max_abs_energy_balance_residual_W: float


@dataclass(frozen=True)
class HourSolution:
    """Controlled solution for one hour, including diagnostic intermediate values."""

    theta_mass_end_C: float
    theta_mass_C: float
    theta_surface_C: float
    theta_air_C: float
    theta_operative_C: float
    signed_hvac_load_W: float
    heating_demand_W: float
    cooling_demand_W: float
    Phi_solar_W: float
    gains: GainAllocation
    ventilation: VentilationState
    free_running_air_C: float
    test_load_air_C: float | None
    max_abs_energy_balance_residual_W: float


_BOUNDARY_PATTERN = re.compile(
    r"R_add\s*=\s*([-+]?\d+(?:\.\d+)?)\s*;\s*b_tr\s*=\s*([-+]?\d+(?:\.\d+)?)"
)
_GLAZING_U_PATTERN = re.compile(r"U_window\s*=\s*([-+]?\d+(?:\.\d+)?)")


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalCoreError(f"{name} must be numeric; got {value!r}.") from exc
    if not np.isfinite(number):
        raise ThermalCoreError(f"{name} must be finite; got {value!r}.")
    return number


def _contract_or_default(
    assumptions: AssumptionContract | None,
) -> AssumptionContract:
    return assumptions if assumptions is not None else load_assumption_contract(DEFAULT_ASSUMPTIONS_PATH)


def _boundary_correction(
    assumptions: AssumptionContract, assumption_id: str
) -> BoundaryCorrection:
    text = assumptions.text(assumption_id)
    match = _BOUNDARY_PATTERN.search(text)
    if not match:
        raise ThermalCoreError(
            f"Assumption {assumption_id!r} does not contain parseable R_add and b_tr values."
        )
    correction = BoundaryCorrection(float(match.group(1)), float(match.group(2)))
    if correction.additional_resistance_m2K_W < 0.0:
        raise ThermalCoreError(f"{assumption_id} has negative additional resistance.")
    if not 0.0 <= correction.temperature_adjustment_factor <= 1.0:
        raise ThermalCoreError(f"{assumption_id} has b_tr outside [0, 1].")
    return correction


def _effective_opaque_conductance(
    area_m2: float,
    u_value_W_m2K: float,
    correction: BoundaryCorrection,
) -> float:
    """Return ``A * b_tr / (1/U + R_add)`` in W/K."""

    if area_m2 < 0.0 or u_value_W_m2K <= 0.0:
        raise ThermalCoreError("Opaque area must be non-negative and U-value positive.")
    denominator = 1.0 / u_value_W_m2K + correction.additional_resistance_m2K_W
    if denominator <= 0.0:
        raise ThermalCoreError("Opaque effective-U denominator must be positive.")
    return area_m2 * correction.temperature_adjustment_factor / denominator


def _glazing_g_value(u_window_W_m2K: float, assumptions: AssumptionContract) -> float:
    candidates: list[tuple[float, float]] = []
    rows = assumptions.frame.loc[
        assumptions.frame["assumption_id"].astype(str).str.startswith("solar.glazing_")
    ]
    for row in rows.itertuples(index=False):
        match = _GLAZING_U_PATTERN.search(str(row.scope_or_key))
        if match and not pd.isna(row.value_numeric):
            candidates.append((float(match.group(1)), float(row.value_numeric)))
    matches = [
        g_value
        for supported_u, g_value in candidates
        if np.isclose(u_window_W_m2K, supported_u, rtol=0.0, atol=1e-9)
    ]
    if len(matches) != 1:
        supported = sorted(value for value, _ in candidates)
        raise ThermalCoreError(
            f"U_window={u_window_W_m2K} W/m2K has no unique documented glazing "
            f"mapping; supported values are {supported}."
        )
    g_value = matches[0]
    if not 0.0 <= g_value <= 1.0:
        raise ThermalCoreError(f"Mapped glazing g-value {g_value} is outside [0, 1].")
    return g_value


def _ventilation_rate(ventilation_system: str, assumptions: AssumptionContract) -> float:
    ids = {
        "existing_unspecified": "ventilation.use_rate_existing",
        "exhaust_air_ventilation": "ventilation.use_rate_exhaust",
        "balanced_mechanical_HRV": "ventilation.use_rate_balanced",
    }
    try:
        assumption_id = ids[ventilation_system]
    except KeyError as exc:
        raise ThermalCoreError(f"Unsupported ventilation system {ventilation_system!r}.") from exc
    rate = assumptions.number(assumption_id)
    if rate < 0.0:
        raise ThermalCoreError(f"{assumption_id} must be non-negative.")
    return rate


def preprocess_archetype(
    state: ArchetypeStateInput | Mapping[str, Any],
    assumptions: AssumptionContract | None = None,
) -> PreparedArchetype:
    """Convert a validated physical archetype/state into 5R1C parameters.

    The precomputed stock-layer screening conductance is intentionally absent:
    all transparent and corrected opaque conductances are rebuilt from the
    component geometry, U-values, and documented boundary corrections.
    """

    contract = _contract_or_default(assumptions)
    if isinstance(state, ArchetypeStateInput):
        validated_state = validate_archetype_state(asdict(state))
    else:
        validated_state = validate_archetype_state(state)

    exterior = _boundary_correction(contract, "boundary.exterior")
    unheated_room = _boundary_correction(contract, "boundary.unheated_room")
    cellar = _boundary_correction(contract, "boundary.unheated_cellar")
    soil = _boundary_correction(contract, "boundary.soil")

    H_tr_w = validated_state.U_window_W_m2K * validated_state.windows_total_m2
    H_tr_op = sum(
        (
            _effective_opaque_conductance(
                validated_state.exterior_wall_area_m2,
                validated_state.U_facade_W_m2K,
                exterior,
            ),
            _effective_opaque_conductance(
                validated_state.roof_area_m2,
                validated_state.U_roof_W_m2K,
                exterior,
            ),
            _effective_opaque_conductance(
                validated_state.doors_area_m2,
                validated_state.U_door_W_m2K,
                exterior,
            ),
            _effective_opaque_conductance(
                validated_state.exterior_wall_bordering_unheated_neighboring_spaces_m2,
                validated_state.U_facade_W_m2K,
                unheated_room,
            ),
            _effective_opaque_conductance(
                validated_state.floor_bordering_unheated_neighboring_spaces_m2,
                validated_state.U_floor_W_m2K,
                cellar,
            ),
            _effective_opaque_conductance(
                validated_state.floor_on_soil_m2,
                validated_state.U_floor_W_m2K,
                soil,
            ),
        )
    )
    if not np.isclose(
        contract.number("boundary.thermal_bridges"), 0.0, rtol=0.0, atol=1e-12
    ):
        raise ThermalCoreError(
            "This core implements the documented zero-thermal-bridge scope only."
        )

    A_f = validated_state.floor_surface_area_m2
    A_t = contract.number("network.total_surface_ratio") * A_f
    A_m = contract.number("network.effective_mass_area_ratio") * A_f
    C_m = contract.number("network.mass_capacitance_ratio") * A_f
    H_tr_is = contract.number("network.air_surface_coefficient") * A_t
    H_tr_ms = contract.number("network.mass_surface_coefficient") * A_m
    if H_tr_op <= 0.0 or H_tr_ms <= 0.0 or H_tr_op >= H_tr_ms:
        raise ThermalCoreError(
            "Cannot construct H_tr,em: require 0 < H_tr,op < H_tr,ms; "
            f"got H_tr,op={H_tr_op}, H_tr,ms={H_tr_ms}."
        )
    reciprocal_difference = 1.0 / H_tr_op - 1.0 / H_tr_ms
    if reciprocal_difference <= 0.0:
        raise ThermalCoreError("Equivalent exterior-to-mass conductance is not positive.")
    H_tr_em = 1.0 / reciprocal_difference

    prepared = PreparedArchetype(
        archetype_id=validated_state.archetype_id,
        dwelling_type=validated_state.dwelling_type,
        construction_period=validated_state.construction_period,
        state_id=validated_state.state_id,
        floor_area_m2=A_f,
        zone_volume_m3=validated_state.protected_volume_m3,
        window_area_north_m2=validated_state.windows_north_m2,
        window_area_east_m2=validated_state.windows_east_m2,
        window_area_south_m2=validated_state.windows_south_m2,
        window_area_west_m2=validated_state.windows_west_m2,
        glazing_g_value=_glazing_g_value(validated_state.U_window_W_m2K, contract),
        window_frame_fraction=contract.number("solar.frame_fraction"),
        non_normal_irradiance_factor=contract.number("solar.non_normal_factor"),
        vertical_shading_factor=contract.number("solar.external_shading_vertical"),
        A_t_m2=A_t,
        A_m_m2=A_m,
        C_m_J_K=C_m,
        H_tr_w_W_K=H_tr_w,
        H_tr_op_W_K=H_tr_op,
        H_tr_is_W_K=H_tr_is,
        H_tr_ms_W_K=H_tr_ms,
        H_tr_em_W_K=H_tr_em,
        infiltration_airflow_m3_h=validated_state.infiltration_airflow_normal_m3_h,
        ventilation_ach_h_1=_ventilation_rate(
            validated_state.ventilation_system, contract
        ),
        ventilation_system=validated_state.ventilation_system,
        hrv_efficiency=validated_state.hrv_eta,
        summer_bypass=validated_state.summer_bypass,
        air_density_kg_m3=contract.number("air.rho"),
        air_specific_heat_J_kgK=contract.number("air.cp"),
        assumptions_sha256=contract.sha256,
    )
    validate_prepared_archetype(prepared)
    return prepared


def ventilation_conductance(
    archetype: PreparedArchetype,
    *,
    bypass_active: bool = False,
    _validate_archetype: bool = True,
) -> VentilationState:
    """Calculate hourly ``H_ve`` while never recovering infiltration heat."""

    if _validate_archetype:
        validate_prepared_archetype(archetype)
    if bypass_active and not archetype.summer_bypass:
        raise ThermalCoreError("Cannot activate bypass for an archetype without summer bypass.")
    eta_effective = 0.0 if bypass_active else archetype.hrv_efficiency
    ventilation_airflow = archetype.ventilation_ach_h_1 * archetype.zone_volume_m3
    effective_airflow = archetype.infiltration_airflow_m3_h + (
        1.0 - eta_effective
    ) * ventilation_airflow
    H_ve = (
        archetype.air_density_kg_m3
        * archetype.air_specific_heat_J_kgK
        * effective_airflow
        / 3600.0
    )
    if H_ve < 0.0 or not np.isfinite(H_ve):
        raise ThermalCoreError(f"Calculated invalid H_ve={H_ve} W/K.")
    return VentilationState(
        H_ve_W_K=H_ve,
        infiltration_airflow_m3_h=archetype.infiltration_airflow_m3_h,
        ventilation_airflow_m3_h=ventilation_airflow,
        effective_hrv_efficiency=eta_effective,
        bypass_active=bool(bypass_active),
    )


def solar_gains(
    archetype: PreparedArchetype,
    irradiance_W_m2: Mapping[str, Any],
    *,
    _validate_archetype: bool = True,
) -> float:
    """Calculate transmitted solar gains from already-transposed façade inputs."""

    if _validate_archetype:
        validate_prepared_archetype(archetype)
    area_factor = (
        (1.0 - archetype.window_frame_fraction)
        * archetype.glazing_g_value
        * archetype.non_normal_irradiance_factor
        * archetype.vertical_shading_factor
    )
    total = 0.0
    for orientation in ORIENTATIONS:
        column = f"I_{orientation}_W_m2"
        if column not in irradiance_W_m2:
            raise ThermalCoreError(f"Missing façade irradiance {column!r}.")
        irradiance = _finite(irradiance_W_m2[column], column)
        if irradiance < 0.0:
            raise ThermalCoreError(f"{column} must be non-negative.")
        area = getattr(archetype, f"window_area_{orientation}_m2")
        total += irradiance * area * area_factor
    if total < 0.0 or not np.isfinite(total):
        raise ThermalCoreError(f"Calculated invalid solar gain {total} W.")
    return total


def split_gains(
    archetype: PreparedArchetype,
    Phi_int_W: float,
    Phi_solar_W: float,
    *,
    internal_air_fraction: float = 0.5,
    _validate_archetype: bool = True,
) -> GainAllocation:
    """Apply ISO Annex C gain allocation without a second wrapper-side split."""

    if _validate_archetype:
        validate_prepared_archetype(archetype)
    internal = _finite(Phi_int_W, "Phi_int_W")
    solar = _finite(Phi_solar_W, "Phi_solar_W")
    if internal < 0.0 or solar < 0.0:
        raise ThermalCoreError("Internal and solar gains must be non-negative.")
    if not 0.0 <= internal_air_fraction <= 1.0:
        raise ThermalCoreError("internal_air_fraction must be between zero and one.")

    surface_and_mass_input = (1.0 - internal_air_fraction) * internal + solar
    mass_fraction = archetype.A_m_m2 / archetype.A_t_m2
    h_ms_W_m2K = archetype.H_tr_ms_W_K / archetype.A_m_m2
    surface_fraction = (
        1.0
        - mass_fraction
        - archetype.H_tr_w_W_K / (h_ms_W_m2K * archetype.A_t_m2)
    )
    if surface_fraction < -1e-12:
        raise ThermalCoreError(
            f"ISO surface-gain fraction is negative ({surface_fraction})."
        )
    surface_fraction = max(surface_fraction, 0.0)
    return GainAllocation(
        Phi_ia_W=internal_air_fraction * internal,
        Phi_m_W=mass_fraction * surface_and_mass_input,
        Phi_st_W=surface_fraction * surface_and_mass_input,
    )


def evaluate_5r1c_hour(
    archetype: PreparedArchetype,
    *,
    theta_mass_previous_C: float,
    theta_out_C: float,
    gains: GainAllocation,
    ventilation: VentilationState,
    signed_hvac_load_W: float = 0.0,
    timestep_seconds: float = 3600.0,
    _validate_archetype: bool = True,
) -> NodeSolution:
    """Solve one hour for a prescribed signed HVAC load using Crank–Nicolson.

    Positive HVAC load heats the air node and negative load cools it. The
    returned ``theta_mass_C`` is the timestep-mean mass temperature; the end
    state is returned separately for propagation to the next hour.
    """

    if _validate_archetype:
        validate_prepared_archetype(archetype)
    theta_m_prev = _finite(theta_mass_previous_C, "theta_mass_previous_C")
    theta_e = _finite(theta_out_C, "theta_out_C")
    phi_hc = _finite(signed_hvac_load_W, "signed_hvac_load_W")
    dt = _finite(timestep_seconds, "timestep_seconds")
    if dt <= 0.0:
        raise ThermalCoreError("timestep_seconds must be positive.")

    H_ve = ventilation.H_ve_W_K
    H_is = archetype.H_tr_is_W_K
    H_w = archetype.H_tr_w_W_K
    H_ms = archetype.H_tr_ms_W_K
    H_em = archetype.H_tr_em_W_K
    if H_ve < 0.0:
        raise ThermalCoreError("H_ve must be non-negative.")

    # Stable forms of the Annex C reduced conductances. They avoid divisions by
    # H_ve or H_tr,2 and remain defined for a hypothetical zero-airflow case.
    H_1 = H_ve * H_is / (H_ve + H_is)
    H_2 = H_1 + H_w
    H_3 = H_2 * H_ms / (H_2 + H_ms)
    phi_air = gains.Phi_ia_W + phi_hc
    air_forcing_at_surface = H_1 * theta_e + H_is / (H_ve + H_is) * phi_air
    surface_forcing = gains.Phi_st_W + H_w * theta_e + air_forcing_at_surface
    phi_m_total = (
        gains.Phi_m_W
        + H_em * theta_e
        + H_ms / (H_ms + H_2) * surface_forcing
    )

    storage_conductance = archetype.C_m_J_K / dt
    dynamic_conductance = H_3 + H_em
    denominator = storage_conductance + 0.5 * dynamic_conductance
    if denominator <= 0.0:
        raise ThermalCoreError("Mass-state update denominator is not positive.")
    theta_m_end = (
        theta_m_prev * (storage_conductance - 0.5 * dynamic_conductance)
        + phi_m_total
    ) / denominator
    theta_m = 0.5 * (theta_m_end + theta_m_prev)
    theta_s = (H_ms * theta_m + surface_forcing) / (H_ms + H_2)
    theta_air = (H_is * theta_s + H_ve * theta_e + phi_air) / (H_is + H_ve)

    air_residual = phi_air + H_ve * (theta_e - theta_air) + H_is * (
        theta_s - theta_air
    )
    surface_residual = (
        gains.Phi_st_W
        + H_ms * (theta_m - theta_s)
        + H_w * (theta_e - theta_s)
        + H_is * (theta_air - theta_s)
    )
    mass_storage = archetype.C_m_J_K * (theta_m_end - theta_m_prev) / dt
    mass_net_input = (
        gains.Phi_m_W
        + H_em * (theta_e - theta_m)
        + H_ms * (theta_s - theta_m)
    )
    max_residual = max(
        abs(air_residual),
        abs(surface_residual),
        abs(mass_storage - mass_net_input),
    )
    values = np.array([theta_m_end, theta_m, theta_s, theta_air, max_residual])
    if not np.isfinite(values).all():
        raise ThermalCoreError("5R1C node solution contains a non-finite value.")
    return NodeSolution(
        theta_mass_end_C=theta_m_end,
        theta_mass_C=theta_m,
        theta_surface_C=theta_s,
        theta_air_C=theta_air,
        signed_hvac_load_W=phi_hc,
        max_abs_energy_balance_residual_W=max_residual,
    )


def solve_ideal_hour(
    archetype: PreparedArchetype,
    *,
    theta_mass_previous_C: float,
    theta_out_C: float,
    irradiance_W_m2: Mapping[str, Any],
    Phi_int_W: float,
    theta_set_heat_C: float,
    theta_set_cool_C: float,
    test_load_W_m2: float = 10.0,
    timestep_seconds: float = 3600.0,
    internal_air_fraction: float = 0.5,
    operative_air_weight: float = 0.3,
    _validate_archetype: bool = True,
) -> HourSolution:
    """Solve one hour with ISO free-run/test-load ideal control."""

    if _validate_archetype:
        validate_prepared_archetype(archetype)
    theta_e = _finite(theta_out_C, "theta_out_C")
    set_heat = _finite(theta_set_heat_C, "theta_set_heat_C")
    set_cool = _finite(theta_set_cool_C, "theta_set_cool_C")
    if set_heat > set_cool:
        raise ThermalCoreError("Heating setpoint exceeds cooling setpoint.")
    test_density = _finite(test_load_W_m2, "test_load_W_m2")
    if test_density <= 0.0:
        raise ThermalCoreError("test_load_W_m2 must be positive.")
    if not 0.0 <= operative_air_weight <= 1.0:
        raise ThermalCoreError("operative_air_weight must be between zero and one.")

    phi_solar = solar_gains(
        archetype, irradiance_W_m2, _validate_archetype=False
    )
    gains = split_gains(
        archetype,
        Phi_int_W,
        phi_solar,
        internal_air_fraction=internal_air_fraction,
        _validate_archetype=False,
    )

    nominal_ventilation = ventilation_conductance(
        archetype, bypass_active=False, _validate_archetype=False
    )
    nominal_free = evaluate_5r1c_hour(
        archetype,
        theta_mass_previous_C=theta_mass_previous_C,
        theta_out_C=theta_e,
        gains=gains,
        ventilation=nominal_ventilation,
        signed_hvac_load_W=0.0,
        timestep_seconds=timestep_seconds,
        _validate_archetype=False,
    )
    bypass_active = bool(
        archetype.summer_bypass
        and nominal_free.theta_air_C > set_cool
        and theta_e < nominal_free.theta_air_C
    )
    ventilation = (
        ventilation_conductance(
            archetype, bypass_active=True, _validate_archetype=False
        )
        if bypass_active
        else nominal_ventilation
    )
    free = (
        evaluate_5r1c_hour(
            archetype,
            theta_mass_previous_C=theta_mass_previous_C,
            theta_out_C=theta_e,
            gains=gains,
            ventilation=ventilation,
            signed_hvac_load_W=0.0,
            timestep_seconds=timestep_seconds,
            _validate_archetype=False,
        )
        if bypass_active
        else nominal_free
    )

    signed_load = 0.0
    test_solution: NodeSolution | None = None
    target: float | None = None
    if free.theta_air_C < set_heat:
        test_load = test_density * archetype.floor_area_m2
        target = set_heat
    elif free.theta_air_C > set_cool:
        test_load = -test_density * archetype.floor_area_m2
        target = set_cool
    else:
        test_load = 0.0

    if target is not None:
        test_solution = evaluate_5r1c_hour(
            archetype,
            theta_mass_previous_C=theta_mass_previous_C,
            theta_out_C=theta_e,
            gains=gains,
            ventilation=ventilation,
            signed_hvac_load_W=test_load,
            timestep_seconds=timestep_seconds,
            _validate_archetype=False,
        )
        response = test_solution.theta_air_C - free.theta_air_C
        if abs(response) <= 1e-12:
            raise ThermalCoreError("ISO test load produces no resolvable air-temperature response.")
        signed_load = test_load * (target - free.theta_air_C) / response
        if target == set_heat and signed_load <= 0.0:
            raise ThermalCoreError("Ideal heating interpolation produced a non-positive load.")
        if target == set_cool and signed_load >= 0.0:
            raise ThermalCoreError("Ideal cooling interpolation produced a non-negative load.")
        final = evaluate_5r1c_hour(
            archetype,
            theta_mass_previous_C=theta_mass_previous_C,
            theta_out_C=theta_e,
            gains=gains,
            ventilation=ventilation,
            signed_hvac_load_W=signed_load,
            timestep_seconds=timestep_seconds,
            _validate_archetype=False,
        )
        if not np.isclose(final.theta_air_C, target, rtol=0.0, atol=1e-8):
            raise ThermalCoreError(
                f"Ideal controller missed setpoint: {final.theta_air_C} != {target}."
            )
    else:
        final = free

    surface_weight = 1.0 - operative_air_weight
    theta_operative = (
        operative_air_weight * final.theta_air_C
        + surface_weight * final.theta_surface_C
    )
    return HourSolution(
        theta_mass_end_C=final.theta_mass_end_C,
        theta_mass_C=final.theta_mass_C,
        theta_surface_C=final.theta_surface_C,
        theta_air_C=final.theta_air_C,
        theta_operative_C=theta_operative,
        signed_hvac_load_W=signed_load,
        heating_demand_W=max(signed_load, 0.0),
        cooling_demand_W=max(-signed_load, 0.0),
        Phi_solar_W=phi_solar,
        gains=gains,
        ventilation=ventilation,
        free_running_air_C=free.theta_air_C,
        test_load_air_C=(
            test_solution.theta_air_C if test_solution is not None else None
        ),
        max_abs_energy_balance_residual_W=final.max_abs_energy_balance_residual_W,
    )


def _simulate_one_year(
    request: SimulationInput,
    *,
    initial_mass_temperature_C: float,
    timestep_seconds: float,
    test_load_W_m2: float,
    internal_air_fraction: float,
    operative_air_weight: float,
) -> tuple[list[HourSolution], float]:
    weather = request.weather
    schedules = request.schedules
    theta_m_previous = initial_mass_temperature_C
    solutions: list[HourSolution] = []
    irradiance_columns = [f"I_{orientation}_W_m2" for orientation in ORIENTATIONS]
    irradiance = weather[irradiance_columns].to_numpy(dtype=float)
    theta_out = weather["T_out_C"].to_numpy(dtype=float)
    internal_gains = schedules["Phi_int_W"].to_numpy(dtype=float)
    set_heat = schedules["theta_set_heat_C"].to_numpy(dtype=float)
    set_cool = schedules["theta_set_cool_C"].to_numpy(dtype=float)

    for hour in range(len(weather)):
        facade_values = {
            column: irradiance[hour, index]
            for index, column in enumerate(irradiance_columns)
        }
        solution = solve_ideal_hour(
            request.archetype,
            theta_mass_previous_C=theta_m_previous,
            theta_out_C=theta_out[hour],
            irradiance_W_m2=facade_values,
            Phi_int_W=internal_gains[hour],
            theta_set_heat_C=set_heat[hour],
            theta_set_cool_C=set_cool[hour],
            test_load_W_m2=test_load_W_m2,
            timestep_seconds=timestep_seconds,
            internal_air_fraction=internal_air_fraction,
            operative_air_weight=operative_air_weight,
            _validate_archetype=False,
        )
        solutions.append(solution)
        theta_m_previous = solution.theta_mass_end_C
    return solutions, theta_m_previous


def simulate(
    request: SimulationInput,
    assumptions: AssumptionContract | None = None,
) -> SimulationResult:
    """Simulate the converged deterministic hourly heating/cooling demand year."""

    contract = _contract_or_default(assumptions)
    validated = validate_simulation_input(request)
    if validated.archetype.assumptions_sha256 != contract.sha256:
        raise ThermalCoreError(
            "Prepared archetype assumptions checksum does not match the simulation contract."
        )
    timestep = contract.number("solver.timestep")
    if not np.isclose(timestep, 3600.0, rtol=0.0, atol=1e-12):
        raise ThermalCoreError("This ISO simple-hourly implementation requires a 3600 s timestep.")
    if contract.text("solver.integration") != "Crank-Nicolson":
        raise ThermalCoreError("This core implements the Crank-Nicolson contract only.")
    if contract.text("solver.controller") != "ideal unlimited load":
        raise ThermalCoreError("This core implements the ideal unlimited-load controller only.")

    theta_initial = contract.number("solver.warmup_initial")
    tolerance = contract.number("solver.warmup_tolerance")
    max_cycles = int(contract.number("solver.warmup_max_cycles"))
    test_load = contract.number("solver.test_load")
    internal_air_fraction = contract.number("network.gain_air_fraction")
    operative_air_weight = contract.number("output.operative_weight_air")
    operative_surface_weight = contract.number("output.operative_weight_surface")
    if not np.isclose(
        operative_air_weight + operative_surface_weight, 1.0, rtol=0.0, atol=1e-12
    ):
        raise ThermalCoreError("Operative-temperature weights must sum to one.")

    accepted_solutions: list[HourSolution] | None = None
    warmup_cycles = 0
    last_delta = np.inf
    for cycle in range(1, max_cycles + 1):
        cycle_initial = theta_initial
        solutions, theta_end = _simulate_one_year(
            validated,
            initial_mass_temperature_C=cycle_initial,
            timestep_seconds=timestep,
            test_load_W_m2=test_load,
            internal_air_fraction=internal_air_fraction,
            operative_air_weight=operative_air_weight,
        )
        last_delta = abs(theta_end - cycle_initial)
        warmup_cycles = cycle
        if last_delta <= tolerance:
            accepted_solutions = solutions
            break
        theta_initial = theta_end
    if accepted_solutions is None:
        raise ThermalCoreError(
            "Periodic mass-temperature convergence failed after "
            f"{max_cycles} complete-year cycles; final mismatch={last_delta:.6g} K, "
            f"tolerance={tolerance:.6g} K."
        )

    hourly = pd.DataFrame(
        {
            "timestamp_utc": validated.weather["timestamp_utc"].copy(),
            "T_out_C": validated.weather["T_out_C"].to_numpy(dtype=float),
            "theta_air_C": [solution.theta_air_C for solution in accepted_solutions],
            "theta_surface_C": [
                solution.theta_surface_C for solution in accepted_solutions
            ],
            "theta_mass_C": [solution.theta_mass_C for solution in accepted_solutions],
            "theta_operative_C": [
                solution.theta_operative_C for solution in accepted_solutions
            ],
            # This is the ISO no-load air temperature evaluated at the start of
            # the ideal-load procedure for the accepted, periodically converged
            # year.  It is retained so Gate 5 can audit pre-control overheating.
            "theta_air_free_running_C": [
                solution.free_running_air_C for solution in accepted_solutions
            ],
            "Phi_internal_W": validated.schedules["Phi_int_W"].to_numpy(dtype=float),
            "Phi_solar_W": [solution.Phi_solar_W for solution in accepted_solutions],
            "heating_demand_W": [
                solution.heating_demand_W for solution in accepted_solutions
            ],
            "cooling_demand_W": [
                solution.cooling_demand_W for solution in accepted_solutions
            ],
            "theta_set_heat_C": validated.schedules["theta_set_heat_C"].to_numpy(
                dtype=float
            ),
            "theta_set_cool_C": validated.schedules["theta_set_cool_C"].to_numpy(
                dtype=float
            ),
            "H_ve_W_K": [
                solution.ventilation.H_ve_W_K for solution in accepted_solutions
            ],
            "hrv_bypass_active": [
                solution.ventilation.bypass_active for solution in accepted_solutions
            ],
        }
    )
    annual_heating = float(hourly["heating_demand_W"].sum()) / 1000.0
    annual_cooling = float(hourly["cooling_demand_W"].sum()) / 1000.0
    diagnostics = SimulationDiagnostics(
        archetype_id=validated.archetype.archetype_id,
        state_id=validated.archetype.state_id,
        weather_member_id=validated.weather_member_id,
        occupant_seed=validated.occupant_seed,
        model_scenario=validated.model_scenario,
        assumptions_sha256=contract.sha256,
        annual_heating_kWh=annual_heating,
        annual_cooling_kWh=annual_cooling,
        heating_intensity_kWh_m2=annual_heating
        / validated.archetype.floor_area_m2,
        cooling_intensity_kWh_m2=annual_cooling
        / validated.archetype.floor_area_m2,
        peak_heating_W=float(hourly["heating_demand_W"].max()),
        peak_cooling_W=float(hourly["cooling_demand_W"].max()),
        max_abs_energy_balance_residual_W=max(
            solution.max_abs_energy_balance_residual_W
            for solution in accepted_solutions
        ),
        warmup_cycles=warmup_cycles,
    )
    return validate_simulation_result(
        SimulationResult(hourly=hourly, diagnostics=diagnostics), validated
    )
