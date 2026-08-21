from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.contracts import (
    ContractError,
    PreparedArchetype,
    SimulationInput,
    assemble_archetype_state,
    load_assumption_contract,
    validate_archetype_state,
    validate_schedule_frame,
)
from thermal_model.core import (
    GainAllocation,
    VentilationState,
    evaluate_5r1c_hour,
    preprocess_archetype,
    simulate,
    solar_gains,
    solve_ideal_hour,
    split_gains,
    ventilation_conductance,
)
from thermal_model.validation import load_unique_archetype_states


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_MATRIX = (
    PROJECT_ROOT
    / "BE_building_stock/data/matrices/national/base_physical_archetype_matrix.csv"
)
STATE_MATRIX = (
    PROJECT_ROOT
    / "BE_building_stock/data/scenarios/renovation/"
    "archetype_matrix_2050_renovation_scenarios.csv"
)


@pytest.fixture(scope="module")
def contract():
    return load_assumption_contract()


@pytest.fixture(scope="module")
def synthetic(contract) -> PreparedArchetype:
    return PreparedArchetype(
        archetype_id="BE_TABULA_TEST",
        dwelling_type="Detached house",
        construction_period="pre-1946",
        state_id="TABULA_existing",
        floor_area_m2=100.0,
        zone_volume_m3=275.0,
        window_area_north_m2=5.0,
        window_area_east_m2=5.0,
        window_area_south_m2=10.0,
        window_area_west_m2=5.0,
        glazing_g_value=0.85,
        window_frame_fraction=0.3,
        non_normal_irradiance_factor=0.9,
        vertical_shading_factor=0.6,
        A_t_m2=450.0,
        A_m_m2=250.0,
        C_m_J_K=16_500_000.0,
        H_tr_w_W_K=50.0,
        H_tr_op_W_K=100.0,
        H_tr_is_W_K=1552.5,
        H_tr_ms_W_K=2275.0,
        H_tr_em_W_K=104.59770114942529,
        infiltration_airflow_m3_h=100.0,
        ventilation_ach_h_1=0.4,
        ventilation_system="existing_unspecified",
        hrv_efficiency=0.0,
        summer_bypass=False,
        air_density_kg_m3=1.2,
        air_specific_heat_J_kgK=1005.0,
        assumptions_sha256=contract.sha256,
    )


def zero_facades() -> dict[str, float]:
    return {
        "I_north_W_m2": 0.0,
        "I_east_W_m2": 0.0,
        "I_south_W_m2": 0.0,
        "I_west_W_m2": 0.0,
    }


def constant_year(
    archetype: PreparedArchetype,
    *,
    theta_out_C: float = 5.0,
    internal_gain_W: float = 0.0,
    heat_C: float = 20.0,
    cool_C: float = 26.0,
) -> SimulationInput:
    timestamps = pd.date_range(
        "2021-01-01T00:00:00Z", "2021-12-31T23:00:00Z", freq="h"
    )
    count = len(timestamps)
    weather = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "T_out_C": np.full(count, theta_out_C),
            **{column: np.zeros(count) for column in zero_facades()},
        }
    )
    schedules = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "Phi_int_W": np.full(count, internal_gain_W),
            "theta_set_heat_C": np.full(count, heat_C),
            "theta_set_cool_C": np.full(count, cool_C),
        }
    )
    return SimulationInput(archetype, weather, schedules, "constant_2021", 0)


def test_input_contract_stops_invalid_physics_instead_of_repairing() -> None:
    base = pd.read_csv(BASE_MATRIX).iloc[0]
    states = pd.read_csv(STATE_MATRIX)
    existing = states.loc[
        (states["archetype_id"] == base["archetype_id"])
        & (states["state_id"] == "TABULA_existing")
    ].iloc[0]
    advanced = states.loc[
        (states["archetype_id"] == base["archetype_id"])
        & (states["state_id"] == "TABULA_advanced_A_proxy")
    ].iloc[0]

    zero_floor = base.copy()
    zero_floor["floor_surface_area_m2"] = 0.0
    with pytest.raises(ContractError, match="floor_surface_area_m2 must be greater"):
        assemble_archetype_state(zero_floor, existing)

    negative_u = existing.copy()
    negative_u["U_roof_W_m2K"] = -0.1
    with pytest.raises(ContractError, match="U_roof_W_m2K must be greater"):
        assemble_archetype_state(base, negative_u)

    negative_infiltration = existing.copy()
    negative_infiltration["infiltration_airflow_normal_m3_h"] = -1.0
    with pytest.raises(ContractError, match="infiltration_airflow_normal_m3_h"):
        assemble_archetype_state(base, negative_infiltration)

    invalid_hrv = advanced.copy()
    invalid_hrv["hrv_eta"] = 1.01
    with pytest.raises(ContractError, match="hrv_eta must be <= 1.0"):
        assemble_archetype_state(base, invalid_hrv)

    bad_windows = base.copy()
    bad_windows["windows_total_m2"] += 1.0
    with pytest.raises(ContractError, match="Oriented window areas do not sum"):
        assemble_archetype_state(bad_windows, existing)

    bad_envelope = base.copy()
    bad_envelope["total_building_envelope_area_m2"] += 1.0
    with pytest.raises(ContractError, match="0.25 m2 source-rounding tolerance"):
        assemble_archetype_state(bad_envelope, existing)


def test_negative_internal_gains_are_rejected(contract, synthetic) -> None:
    request = constant_year(synthetic)
    schedules = request.schedules.copy()
    schedules.loc[0, "Phi_int_W"] = -0.01
    with pytest.raises(ContractError, match="below 0.0 W"):
        validate_schedule_frame(schedules)


def test_zero_temperature_difference_has_zero_branch_heat_flow(synthetic) -> None:
    solution = evaluate_5r1c_hour(
        synthetic,
        theta_mass_previous_C=20.0,
        theta_out_C=20.0,
        gains=split_gains(synthetic, 0.0, 0.0),
        ventilation=ventilation_conductance(synthetic),
    )
    assert solution.theta_mass_end_C == pytest.approx(20.0, abs=1e-13)
    assert solution.theta_surface_C == pytest.approx(20.0, abs=1e-13)
    assert solution.theta_air_C == pytest.approx(20.0, abs=1e-13)
    assert solution.max_abs_energy_balance_residual_W < 1e-10


def test_zero_irradiance_and_zero_window_area_each_remove_solar_gain(synthetic) -> None:
    assert solar_gains(synthetic, zero_facades()) == 0.0
    windowless = replace(
        synthetic,
        window_area_north_m2=0.0,
        window_area_east_m2=0.0,
        window_area_south_m2=0.0,
        window_area_west_m2=0.0,
        H_tr_w_W_K=0.0,
    )
    nonzero = {column: 500.0 for column in zero_facades()}
    assert solar_gains(windowless, nonzero) == 0.0


def test_perfect_hrv_removes_only_recoverable_ventilation(synthetic) -> None:
    balanced = replace(
        synthetic,
        ventilation_system="balanced_mechanical_HRV",
        hrv_efficiency=1.0,
        summer_bypass=True,
    )
    state = ventilation_conductance(balanced)
    infiltration_only = (
        synthetic.air_density_kg_m3
        * synthetic.air_specific_heat_J_kgK
        * synthetic.infiltration_airflow_m3_h
        / 3600.0
    )
    assert state.H_ve_W_K == pytest.approx(infiltration_only, abs=1e-12)
    assert state.ventilation_airflow_m3_h > 0.0


def test_summer_bypass_requires_both_temperature_conditions(synthetic) -> None:
    balanced = replace(
        synthetic,
        ventilation_system="balanced_mechanical_HRV",
        hrv_efficiency=0.8,
        summer_bypass=True,
    )
    below_cooling = solve_ideal_hour(
        balanced,
        theta_mass_previous_C=23.0,
        theta_out_C=18.0,
        irradiance_W_m2=zero_facades(),
        Phi_int_W=0.0,
        theta_set_heat_C=20.0,
        theta_set_cool_C=26.0,
    )
    assert below_cooling.ventilation.bypass_active is False

    hot_outside = solve_ideal_hour(
        balanced,
        theta_mass_previous_C=30.0,
        theta_out_C=35.0,
        irradiance_W_m2=zero_facades(),
        Phi_int_W=2000.0,
        theta_set_heat_C=20.0,
        theta_set_cool_C=26.0,
    )
    assert hot_outside.ventilation.bypass_active is False

    eligible = solve_ideal_hour(
        balanced,
        theta_mass_previous_C=30.0,
        theta_out_C=20.0,
        irradiance_W_m2=zero_facades(),
        Phi_int_W=2000.0,
        theta_set_heat_C=20.0,
        theta_set_cool_C=26.0,
    )
    assert eligible.ventilation.bypass_active is True


def test_removing_door_changes_opaque_conductance_by_exact_branch(contract) -> None:
    base = pd.read_csv(BASE_MATRIX).iloc[0]
    states = pd.read_csv(STATE_MATRIX)
    state_row = states.loc[
        (states["archetype_id"] == base["archetype_id"])
        & (states["state_id"] == "TABULA_existing")
    ].iloc[0]
    state = assemble_archetype_state(base, state_row)
    full = preprocess_archetype(state, contract)
    without_door = replace(
        state,
        total_building_envelope_area_m2=(
            state.total_building_envelope_area_m2 - state.doors_area_m2
        ),
        doors_area_m2=0.0,
    )
    reduced = preprocess_archetype(without_door, contract)
    expected = state.doors_area_m2 * state.U_door_W_m2K
    assert full.H_tr_op_W_K - reduced.H_tr_op_W_K == pytest.approx(
        expected, abs=1e-11
    )


def test_iso_gain_allocation_satisfies_corrected_balance_not_naive_sum(synthetic) -> None:
    allocation = split_gains(synthetic, 300.0, 500.0)
    X = 0.5 * 300.0 + 500.0
    omitted_window_term = (
        synthetic.H_tr_w_W_K
        / ((synthetic.H_tr_ms_W_K / synthetic.A_m_m2) * synthetic.A_t_m2)
        * X
    )
    allocated = allocation.Phi_ia_W + allocation.Phi_m_W + allocation.Phi_st_W
    assert 300.0 + 500.0 - allocated == pytest.approx(
        omitted_window_term, abs=1e-12
    )


def test_free_decay_is_monotonic_and_matches_multi_hour_closed_form(synthetic) -> None:
    ventilation = ventilation_conductance(synthetic)
    gains = split_gains(synthetic, 0.0, 0.0)
    H_1 = ventilation.H_ve_W_K * synthetic.H_tr_is_W_K / (
        ventilation.H_ve_W_K + synthetic.H_tr_is_W_K
    )
    H_2 = H_1 + synthetic.H_tr_w_W_K
    H_3 = H_2 * synthetic.H_tr_ms_W_K / (H_2 + synthetic.H_tr_ms_W_K)
    dynamic = H_3 + synthetic.H_tr_em_W_K
    storage = synthetic.C_m_J_K / 3600.0
    ratio = (storage - 0.5 * dynamic) / (storage + 0.5 * dynamic)

    previous = 20.0
    end_temperatures: list[float] = []
    for _ in range(240):
        solution = evaluate_5r1c_hour(
            synthetic,
            theta_mass_previous_C=previous,
            theta_out_C=5.0,
            gains=gains,
            ventilation=ventilation,
        )
        end_temperatures.append(solution.theta_mass_end_C)
        previous = solution.theta_mass_end_C
    assert np.all(np.diff(end_temperatures) < 0.0)
    assert end_temperatures[-1] > 5.0
    expected = 5.0 + ratio ** 240 * (20.0 - 5.0)
    assert end_temperatures[-1] == pytest.approx(expected, abs=1e-11)


def test_adiabatic_zone_conserves_prescribed_gain_energy(synthetic) -> None:
    adiabatic = replace(
        synthetic,
        window_area_north_m2=0.0,
        window_area_east_m2=0.0,
        window_area_south_m2=0.0,
        window_area_west_m2=0.0,
        H_tr_w_W_K=0.0,
        H_tr_em_W_K=0.0,
        infiltration_airflow_m3_h=0.0,
        ventilation_ach_h_1=0.0,
    )
    ventilation = VentilationState(0.0, 0.0, 0.0, 0.0, False)
    gains = split_gains(
        adiabatic, 100.0, 0.0, _validate_archetype=False
    )
    previous = 20.0
    for _ in range(24):
        solution = evaluate_5r1c_hour(
            adiabatic,
            theta_mass_previous_C=previous,
            theta_out_C=5.0,
            gains=gains,
            ventilation=ventilation,
            _validate_archetype=False,
        )
        stored_power = (
            adiabatic.C_m_J_K
            * (solution.theta_mass_end_C - previous)
            / 3600.0
        )
        assert stored_power == pytest.approx(100.0, abs=1e-9)
        assert solution.max_abs_energy_balance_residual_W < 1e-8
        previous = solution.theta_mass_end_C


def _static_heating_oracle(
    archetype: PreparedArchetype,
    *,
    theta_out_C: float,
    theta_air_C: float,
    internal_gain_W: float,
) -> float:
    gains = split_gains(archetype, internal_gain_W, 0.0)
    ventilation = ventilation_conductance(archetype)
    # Independent steady two-equation solution for surface and mass temperatures.
    matrix = np.array(
        [
            [
                archetype.H_tr_ms_W_K + archetype.H_tr_w_W_K + archetype.H_tr_is_W_K,
                -archetype.H_tr_ms_W_K,
            ],
            [-archetype.H_tr_ms_W_K, archetype.H_tr_em_W_K + archetype.H_tr_ms_W_K],
        ]
    )
    forcing = np.array(
        [
            gains.Phi_st_W
            + archetype.H_tr_w_W_K * theta_out_C
            + archetype.H_tr_is_W_K * theta_air_C,
            gains.Phi_m_W + archetype.H_tr_em_W_K * theta_out_C,
        ]
    )
    theta_surface, _ = np.linalg.solve(matrix, forcing)
    return (
        ventilation.H_ve_W_K * (theta_air_C - theta_out_C)
        + archetype.H_tr_is_W_K * (theta_air_C - theta_surface)
        - gains.Phi_ia_W
    )


def test_constant_internal_gain_reduction_matches_multinode_static_oracle(
    synthetic, contract
) -> None:
    powers = []
    for gain in (0.0, 500.0, 1000.0):
        result = simulate(constant_year(synthetic, internal_gain_W=gain), contract)
        simulated = float(result.hourly["heating_demand_W"].iloc[0])
        oracle = _static_heating_oracle(
            synthetic,
            theta_out_C=5.0,
            theta_air_C=20.0,
            internal_gain_W=gain,
        )
        assert simulated == pytest.approx(oracle, abs=1e-8)
        powers.append(simulated)
    first_reduction = powers[0] - powers[1]
    second_reduction = powers[1] - powers[2]
    assert first_reduction == pytest.approx(second_reduction, abs=1e-8)
    assert 0.0 < first_reduction <= 500.0


@pytest.mark.parametrize("orientation", ["north", "east", "south", "west"])
def test_solar_orientation_mapping_is_exclusive(synthetic, orientation) -> None:
    window_values = {
        "window_area_north_m2": 0.0,
        "window_area_east_m2": 0.0,
        "window_area_south_m2": 0.0,
        "window_area_west_m2": 0.0,
    }
    window_values[f"window_area_{orientation}_m2"] = 10.0
    one_orientation = replace(synthetic, **window_values)
    forcing = zero_facades()
    forcing[f"I_{orientation}_W_m2"] = 500.0
    expected = 500.0 * 10.0 * 0.7 * 0.85 * 0.9 * 0.6
    assert solar_gains(one_orientation, forcing) == pytest.approx(expected)

    wrong_orientation = zero_facades()
    other = next(item for item in ("north", "east", "south", "west") if item != orientation)
    wrong_orientation[f"I_{other}_W_m2"] = 500.0
    assert solar_gains(one_orientation, wrong_orientation) == 0.0


def test_periodic_solution_is_independent_of_first_pass_temperature(
    synthetic, contract
) -> None:
    cold_frame = contract.frame.copy()
    hot_frame = contract.frame.copy()
    cold_frame.loc[
        cold_frame["assumption_id"] == "solver.warmup_initial", "value_numeric"
    ] = 5.0
    hot_frame.loc[
        hot_frame["assumption_id"] == "solver.warmup_initial", "value_numeric"
    ] = 35.0
    cold = simulate(constant_year(synthetic), replace(contract, frame=cold_frame))
    hot = simulate(constant_year(synthetic), replace(contract, frame=hot_frame))
    np.testing.assert_allclose(
        cold.hourly["heating_demand_W"],
        hot.hourly["heating_demand_W"],
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cold.hourly["theta_mass_C"], hot.hourly["theta_mass_C"], rtol=0.0, atol=1e-8
    )


def test_identical_inputs_are_bit_reproducible(synthetic, contract) -> None:
    request = constant_year(synthetic, internal_gain_W=300.0)
    first = simulate(request, contract)
    second = simulate(request, contract)
    pd.testing.assert_frame_equal(first.hourly, second.hourly, check_exact=True)
    assert asdict(first.diagnostics) == asdict(second.diagnostics)


def test_highly_insulated_and_leaky_real_states_remain_stable(contract) -> None:
    prepared = [
        preprocess_archetype(state, contract) for state in load_unique_archetype_states()
    ]
    ranked = sorted(
        prepared,
        key=lambda item: (
            item.H_tr_op_W_K
            + item.H_tr_w_W_K
            + ventilation_conductance(item).H_ve_W_K
        )
        / item.floor_area_m2,
    )
    for archetype in (ranked[0], ranked[-1]):
        result = simulate(constant_year(archetype, internal_gain_W=0.0), contract)
        assert np.isfinite(
            result.hourly[
                [
                    "theta_air_C",
                    "theta_surface_C",
                    "theta_mass_C",
                    "heating_demand_W",
                    "cooling_demand_W",
                ]
            ].to_numpy(dtype=float)
        ).all()
        assert result.diagnostics.max_abs_energy_balance_residual_W < 1e-6


def test_heating_to_cooling_transitions_track_setpoints_without_overlap(
    synthetic, contract
) -> None:
    timestamps = pd.date_range(
        "2021-01-01T00:00:00Z", "2021-12-31T23:00:00Z", freq="h"
    )
    phase = np.arange(len(timestamps), dtype=float) / len(timestamps)
    theta_out = 21.0 + 16.0 * np.sin(2.0 * np.pi * phase - np.pi / 2.0)
    weather = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "T_out_C": theta_out,
            **{column: np.zeros(len(timestamps)) for column in zero_facades()},
        }
    )
    schedules = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "Phi_int_W": np.zeros(len(timestamps)),
            "theta_set_heat_C": np.full(len(timestamps), 20.0),
            "theta_set_cool_C": np.full(len(timestamps), 24.0),
        }
    )
    result = simulate(
        SimulationInput(synthetic, weather, schedules, "transition_2021", 0), contract
    )
    heating = result.hourly["heating_demand_W"] > 1e-9
    cooling = result.hourly["cooling_demand_W"] > 1e-9
    assert heating.any() and cooling.any()
    assert not (heating & cooling).any()
    np.testing.assert_allclose(
        result.hourly.loc[heating, "theta_air_C"], 20.0, rtol=0.0, atol=1e-8
    )
    np.testing.assert_allclose(
        result.hourly.loc[cooling, "theta_air_C"], 24.0, rtol=0.0, atol=1e-8
    )
