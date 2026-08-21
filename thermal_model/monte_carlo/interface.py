"""Stable, side-effect-free Gate-5 dwelling simulation interface."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from thermal_model.behaviour import (
    BehaviourRequest,
    BehaviourResult,
    dwelling_class,
    generate_behaviour,
    load_behaviour_assumptions,
)
from thermal_model.behaviour.contracts import weather_forcing_sha256
from thermal_model.contracts import (
    ArchetypeStateInput,
    AssumptionContract,
    SimulationInput,
    load_assumption_contract,
    validate_archetype_state,
)
from thermal_model.core import preprocess_archetype, simulate as simulate_thermal_core

from .contracts import (
    MODEL_CONTRACT_VERSION,
    ModelScenario,
    MonteCarloContractError,
    MonteCarloDiagnostics,
    MonteCarloResult,
    WeatherMember,
    archetype_identity,
    archetype_state_sha256,
    canonical_sha256,
    validate_weather_member,
)
from .scenarios import (
    apply_archetype_scenario,
    effective_assumption_contract,
    model_scenario_sha256,
    resolve_model_scenario,
)
from .weather import load_weather_member


CONTROL_TOLERANCE_W = 1.0e-9
TEMPERATURE_TOLERANCE_K = 1.0e-9


def _validated_state(
    value: ArchetypeStateInput | Mapping[str, Any],
) -> ArchetypeStateInput:
    if isinstance(value, ArchetypeStateInput):
        return validate_archetype_state(value.__dict__)
    if isinstance(value, Mapping):
        return validate_archetype_state(value)
    raise MonteCarloContractError(
        "archetype_state must be ArchetypeStateInput or a complete field mapping."
    )


def _validated_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MonteCarloContractError("occupant_seed must be an integer.")
    seed = int(value)
    if not 0 <= seed <= 2**32 - 1:
        raise MonteCarloContractError("occupant_seed must be between 0 and 2**32-1.")
    return seed


def _setpoint_histogram(values: pd.Series) -> tuple[tuple[float, int], ...]:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    counts = numeric.value_counts(sort=False).sort_index()
    return tuple((float(value), int(count)) for value, count in counts.items())


def _full_load_hours(annual_energy_kWh: float, peak_power_W: float) -> float:
    return (
        1000.0 * float(annual_energy_kWh) / float(peak_power_W)
        if peak_power_W > 0.0
        else 0.0
    )


def _run_id(
    state: ArchetypeStateInput,
    weather: WeatherMember,
    seed: int,
    scenario: ModelScenario,
    *,
    effective_thermal_sha256: str,
    behaviour_assumptions_sha256: str,
    occupant_distribution_sha256: str,
) -> str:
    return "mc_" + canonical_sha256(
        {
            "model_contract_version": MODEL_CONTRACT_VERSION,
            **archetype_identity(state),
            "archetype_state_sha256": archetype_state_sha256(state),
            "climate_scenario_id": weather.climate_scenario_id,
            "weather_member_id": weather.member_id,
            "weather_contract_sha256": weather.weather_contract_sha256,
            "weather_forcing_sha256": weather.forcing_sha256,
            "occupant_seed": seed,
            "model_scenario_id": scenario.scenario_id,
            "effective_thermal_assumptions_sha256": effective_thermal_sha256,
            "behaviour_assumptions_sha256": behaviour_assumptions_sha256,
            "occupant_distribution_sha256": occupant_distribution_sha256,
        }
    )


def _simulate_with_behaviour(
    state: ArchetypeStateInput,
    weather: WeatherMember,
    seed: int,
    scenario: ModelScenario,
    behaviour: BehaviourResult,
    central_thermal_contract: AssumptionContract,
) -> MonteCarloResult:
    """Compose a previously generated class-equivalent behaviour realization."""

    actual_class = dwelling_class(state.dwelling_type)
    behaviour_diagnostics = behaviour.diagnostics
    if behaviour_diagnostics.dwelling_class != actual_class:
        raise MonteCarloContractError(
            "Cached behavioural dwelling class does not match the archetype."
        )
    if behaviour_diagnostics.seed != seed:
        raise MonteCarloContractError("Cached behavioural seed does not match the run.")
    if behaviour_diagnostics.weather_member_id != weather.member_id:
        raise MonteCarloContractError(
            "Cached behavioural weather member does not match the run."
        )
    expected_horizontal_hash = weather_forcing_sha256(weather.frame)
    if behaviour_diagnostics.weather_forcing_sha256 != expected_horizontal_hash:
        raise MonteCarloContractError(
            "Behaviour profile was not generated from this weather forcing."
        )

    transformed_state = apply_archetype_scenario(state, scenario)
    effective_contract = effective_assumption_contract(
        central_thermal_contract, scenario
    )
    prepared = preprocess_archetype(transformed_state, effective_contract)
    thermal = simulate_thermal_core(
        SimulationInput(
            archetype=prepared,
            weather=weather.frame.copy(deep=True),
            schedules=behaviour.schedules,
            weather_member_id=weather.member_id,
            occupant_seed=seed,
            model_scenario=scenario.scenario_id,
        ),
        effective_contract,
    )
    hourly = thermal.hourly.copy(deep=True)
    if not pd.DatetimeIndex(hourly["timestamp_utc"]).equals(
        pd.DatetimeIndex(weather.frame["timestamp_utc"])
    ):
        raise MonteCarloContractError("Gate-5 output timestamps changed during coupling.")

    heating_setpoint_hours = _setpoint_histogram(hourly["theta_set_heat_C"])
    cooling_setpoint_hours = _setpoint_histogram(hourly["theta_set_cool_C"])
    if sum(count for _, count in heating_setpoint_hours) != len(hourly):
        raise MonteCarloContractError("Heating setpoint-hour histogram is incomplete.")
    if sum(count for _, count in cooling_setpoint_hours) != len(hourly):
        raise MonteCarloContractError("Cooling setpoint-hour histogram is incomplete.")

    heating_active = hourly["heating_demand_W"].to_numpy(dtype=float) > CONTROL_TOLERANCE_W
    cooling_active = hourly["cooling_demand_W"].to_numpy(dtype=float) > CONTROL_TOLERANCE_W
    free_air = hourly["theta_air_free_running_C"].to_numpy(dtype=float)
    set_heat = hourly["theta_set_heat_C"].to_numpy(dtype=float)
    set_cool = hourly["theta_set_cool_C"].to_numpy(dtype=float)
    thermal_diagnostics = thermal.diagnostics
    scenario_sha = model_scenario_sha256(scenario, central_thermal_contract.sha256)
    run_id = _run_id(
        state,
        weather,
        seed,
        scenario,
        effective_thermal_sha256=effective_contract.sha256,
        behaviour_assumptions_sha256=(
            behaviour_diagnostics.behaviour_assumptions_sha256
        ),
        occupant_distribution_sha256=(
            behaviour_diagnostics.occupant_distribution_sha256
        ),
    )
    diagnostics = MonteCarloDiagnostics(
        run_id=run_id,
        archetype_id=state.archetype_id,
        dwelling_type=state.dwelling_type,
        dwelling_class=actual_class,
        construction_period=state.construction_period,
        state_id=state.state_id,
        archetype_state_sha256=archetype_state_sha256(state),
        floor_area_m2=prepared.floor_area_m2,
        climate_scenario_id=weather.climate_scenario_id,
        weather_member_id=weather.member_id,
        weather_pair_id=weather.weather_pair_id,
        observed_pvgis_year=weather.observed_pvgis_year,
        climate_target=weather.climate_target,
        occupant_seed=seed,
        occupant_count=behaviour_diagnostics.occupant_count,
        richardson_seed=behaviour_diagnostics.richardson_seed,
        model_scenario_id=scenario.scenario_id,
        model_scenario_axis=scenario.axis,
        annual_heating_kWh=thermal_diagnostics.annual_heating_kWh,
        annual_cooling_kWh=thermal_diagnostics.annual_cooling_kWh,
        heating_intensity_kWh_m2=thermal_diagnostics.heating_intensity_kWh_m2,
        cooling_intensity_kWh_m2=thermal_diagnostics.cooling_intensity_kWh_m2,
        peak_heating_W=thermal_diagnostics.peak_heating_W,
        peak_cooling_W=thermal_diagnostics.peak_cooling_W,
        heating_full_load_equivalent_hours=_full_load_hours(
            thermal_diagnostics.annual_heating_kWh,
            thermal_diagnostics.peak_heating_W,
        ),
        cooling_full_load_equivalent_hours=_full_load_hours(
            thermal_diagnostics.annual_cooling_kWh,
            thermal_diagnostics.peak_cooling_W,
        ),
        heating_controlled_hours=int(heating_active.sum()),
        cooling_controlled_hours=int(cooling_active.sum()),
        heating_setpoint_hours=heating_setpoint_hours,
        cooling_setpoint_hours=cooling_setpoint_hours,
        iso_no_load_trial_above_cooling_setpoint_hours=int(
            (free_air > set_cool + TEMPERATURE_TOLERANCE_K).sum()
        ),
        iso_no_load_trial_below_heating_setpoint_hours=int(
            (free_air < set_heat - TEMPERATURE_TOLERANCE_K).sum()
        ),
        hrv_bypass_hours=int(hourly["hrv_bypass_active"].astype(bool).sum()),
        annual_internal_gains_kWh=float(hourly["Phi_internal_W"].sum()) / 1000.0,
        annual_household_electricity_kWh=(
            behaviour_diagnostics.annual_total_electricity_kWh
        ),
        max_abs_energy_balance_residual_W=(
            thermal_diagnostics.max_abs_energy_balance_residual_W
        ),
        warmup_cycles=thermal_diagnostics.warmup_cycles,
        model_contract_version=MODEL_CONTRACT_VERSION,
        central_thermal_assumptions_sha256=central_thermal_contract.sha256,
        effective_thermal_assumptions_sha256=effective_contract.sha256,
        behaviour_assumptions_sha256=(
            behaviour_diagnostics.behaviour_assumptions_sha256
        ),
        occupant_distribution_sha256=(
            behaviour_diagnostics.occupant_distribution_sha256
        ),
        model_scenario_sha256=scenario_sha,
        member_sha256=weather.member_sha256,
        metadata_sha256=weather.metadata_sha256,
        climate_manifest_sha256=weather.manifest_sha256,
        morph_contract_sha256=weather.morph_contract_sha256,
        facade_source_sha256=weather.facade_source_sha256,
        weather_contract_sha256=weather.weather_contract_sha256,
        weather_forcing_sha256=weather.forcing_sha256,
    )
    return MonteCarloResult(hourly=hourly, diagnostics=diagnostics)


def simulate(
    archetype_state: ArchetypeStateInput | Mapping[str, Any],
    weather_member: WeatherMember | str,
    occupant_seed: int,
    model_scenario: ModelScenario | str,
) -> MonteCarloResult:
    """Run one reproducible dwelling realization with no external writes.

    ``weather_member`` may be a validated :class:`WeatherMember` or its
    authoritative manifest identifier.  A string is resolved read-only through
    the climate layer.  Structural assumptions are allow-listed and applied to
    in-memory copies; they are never mixed into ``occupant_seed``.
    """

    state = _validated_state(archetype_state)
    seed = _validated_seed(occupant_seed)
    scenario = resolve_model_scenario(model_scenario)
    weather = (
        load_weather_member(weather_member)
        if isinstance(weather_member, str)
        else validate_weather_member(weather_member)
    )
    behaviour_contract = load_behaviour_assumptions()
    behaviour = generate_behaviour(
        BehaviourRequest(
            dwelling_type=state.dwelling_type,
            weather=weather.frame.copy(deep=True),
            weather_member_id=weather.member_id,
            seed=seed,
        ),
        behaviour_contract,
    )
    return _simulate_with_behaviour(
        state,
        weather,
        seed,
        scenario,
        behaviour,
        load_assumption_contract(),
    )
