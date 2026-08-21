"""Deterministic Gate-4 coupling audit for behaviour and the 5R1C core.

The module owns orchestration only.  RichardsonPy produces an hourly boundary
schedule, while :mod:`thermal_model.core` continues to know only internal gains
and heating/cooling setpoints.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from thermal_model.contracts import (
    DEFAULT_ASSUMPTIONS_PATH,
    ArchetypeStateInput,
    AssumptionContract,
    PreparedArchetype,
    SimulationInput,
    load_assumption_contract,
)
from thermal_model.core import preprocess_archetype, simulate
from thermal_model.validation import (
    ENERGY_BALANCE_TOLERANCE_W,
    SETPOINT_TRACKING_TOLERANCE_K,
    STATE_ORDER,
    build_reference_schedules,
    load_reference_weather,
    load_unique_archetype_states,
)

from .contracts import (
    BehaviourAssumptionContract,
    BehaviourRequest,
    BehaviourResult,
    load_behaviour_assumptions,
)
from .wrapper import dwelling_class, generate_behaviour


DEFAULT_COUPLING_SEED = 20250805
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data/behaviour"
REPRESENTATIVE_ARCHETYPES = {
    "SFH": "BE_TABULA_11",  # Detached house, 1971–1990
    "MFH": "BE_TABULA_14",  # Enclosed apartment, 1971–1990
}
REPRESENTATIVE_STATE = "TABULA_existing"


class CouplingError(ValueError):
    """Raised when the deterministic behaviour-coupling audit is inconsistent."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _simulate_schedules(
    prepared: PreparedArchetype,
    weather: pd.DataFrame,
    schedules: pd.DataFrame,
    assumptions: AssumptionContract,
    *,
    weather_member_id: str,
    seed: int,
    scenario: str,
):
    return simulate(
        SimulationInput(
            archetype=prepared,
            weather=weather,
            schedules=schedules,
            weather_member_id=weather_member_id,
            occupant_seed=seed,
            model_scenario=scenario,
        ),
        assumptions,
    )


def _schedule_with(
    behaviour: BehaviourResult,
    *,
    internal_gains_W: np.ndarray | float | None = None,
    heating_setpoint_C: np.ndarray | float | None = None,
    cooling_setpoint_C: np.ndarray | float | None = None,
) -> pd.DataFrame:
    """Return a schedule variant without changing the behavioural result."""

    schedules = behaviour.schedules
    count = len(schedules)
    replacements = {
        "Phi_int_W": internal_gains_W,
        "theta_set_heat_C": heating_setpoint_C,
        "theta_set_cool_C": cooling_setpoint_C,
    }
    for column, value in replacements.items():
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(count, float(array), dtype=float)
        if array.shape != (count,):
            raise CouplingError(f"{column} replacement must contain {count} values.")
        schedules[column] = array
    return schedules


def flatten_lighting_gain(behaviour: BehaviourResult) -> pd.DataFrame:
    """Flatten lighting heat only, preserving its annual energy exactly."""

    hourly = behaviour.hourly
    flattened_lighting = np.full(
        len(hourly), float(hourly["lighting_sensible_gain_W"].mean())
    )
    total_internal = (
        hourly["occupant_sensible_gain_W"].to_numpy(dtype=float)
        + hourly["appliance_sensible_gain_W"].to_numpy(dtype=float)
        + flattened_lighting
    )
    if not np.isclose(
        flattened_lighting.sum(),
        hourly["lighting_sensible_gain_W"].sum(),
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise CouplingError("Flattening lighting failed to conserve annual sensible heat.")
    return _schedule_with(behaviour, internal_gains_W=total_internal)


def _tracking_errors(hourly: pd.DataFrame) -> tuple[float, float]:
    heating = hourly["heating_demand_W"] > 1.0e-9
    cooling = hourly["cooling_demand_W"] > 1.0e-9
    heating_error = (
        float(
            (
                hourly.loc[heating, "theta_air_C"]
                - hourly.loc[heating, "theta_set_heat_C"]
            )
            .abs()
            .max()
        )
        if heating.any()
        else 0.0
    )
    cooling_error = (
        float(
            (
                hourly.loc[cooling, "theta_air_C"]
                - hourly.loc[cooling, "theta_set_cool_C"]
            )
            .abs()
            .max()
        )
        if cooling.any()
        else 0.0
    )
    return heating_error, cooling_error


def _profile_for_class(
    household_class: str,
    weather: pd.DataFrame,
    weather_member_id: str,
    seed: int,
    assumptions: BehaviourAssumptionContract,
) -> BehaviourResult:
    representative_type = (
        "Detached house" if household_class == "SFH" else "Apartment, enclosed"
    )
    return generate_behaviour(
        BehaviourRequest(
            dwelling_type=representative_type,
            weather=weather,
            weather_member_id=weather_member_id,
            seed=seed,
        ),
        assumptions,
    )


def _summarize_pair(
    state: ArchetypeStateInput,
    prepared: PreparedArchetype,
    baseline,
    behavioural,
    profile: BehaviourResult,
) -> dict[str, Any]:
    baseline_heat = baseline.diagnostics.heating_intensity_kWh_m2
    behavioural_heat = behavioural.diagnostics.heating_intensity_kWh_m2
    baseline_cool = baseline.diagnostics.cooling_intensity_kWh_m2
    behavioural_cool = behavioural.diagnostics.cooling_intensity_kWh_m2
    heat_error, cool_error = _tracking_errors(behavioural.hourly)
    simultaneous = bool(
        (
            (behavioural.hourly["heating_demand_W"] > 1.0e-9)
            & (behavioural.hourly["cooling_demand_W"] > 1.0e-9)
        ).any()
    )
    gain_kWh_m2 = float(profile.hourly["Phi_int_W"].sum()) / (
        1000.0 * prepared.floor_area_m2
    )
    return {
        "archetype_id": state.archetype_id,
        "dwelling_type": state.dwelling_type,
        "dwelling_class": dwelling_class(state.dwelling_type),
        "construction_period": state.construction_period,
        "state_id": state.state_id,
        "floor_area_m2": prepared.floor_area_m2,
        "occupant_count": profile.diagnostics.occupant_count,
        "active_occupancy_hours": profile.diagnostics.active_occupancy_hours,
        "annual_internal_gains_kWh_m2": gain_kWh_m2,
        "constant_heating_kWh_m2": baseline_heat,
        "behavioural_heating_kWh_m2": behavioural_heat,
        "heating_change_kWh_m2": behavioural_heat - baseline_heat,
        "heating_change_percent": (
            100.0 * (behavioural_heat - baseline_heat) / baseline_heat
            if baseline_heat > 0.0
            else np.nan
        ),
        "constant_cooling_kWh_m2": baseline_cool,
        "behavioural_cooling_kWh_m2": behavioural_cool,
        "cooling_change_kWh_m2": behavioural_cool - baseline_cool,
        "cooling_change_percent": (
            100.0 * (behavioural_cool - baseline_cool) / baseline_cool
            if baseline_cool > 0.0
            else np.nan
        ),
        "constant_peak_heating_W": baseline.diagnostics.peak_heating_W,
        "behavioural_peak_heating_W": behavioural.diagnostics.peak_heating_W,
        "constant_peak_cooling_W": baseline.diagnostics.peak_cooling_W,
        "behavioural_peak_cooling_W": behavioural.diagnostics.peak_cooling_W,
        "max_energy_balance_residual_W": (
            behavioural.diagnostics.max_abs_energy_balance_residual_W
        ),
        "max_heating_setpoint_error_K": heat_error,
        "max_cooling_setpoint_error_K": cool_error,
        "simultaneous_heating_cooling": simultaneous,
        "warmup_cycles": behavioural.diagnostics.warmup_cycles,
        "weather_member_id": behavioural.diagnostics.weather_member_id,
        "occupant_seed": behavioural.diagnostics.occupant_seed,
        "thermal_assumptions_sha256": behavioural.diagnostics.assumptions_sha256,
        "behaviour_assumptions_sha256": (
            profile.diagnostics.behaviour_assumptions_sha256
        ),
        "occupant_distribution_sha256": (
            profile.diagnostics.occupant_distribution_sha256
        ),
        "weather_forcing_sha256": profile.diagnostics.weather_forcing_sha256,
    }


def _effect_decomposition(
    states: Sequence[ArchetypeStateInput],
    profiles: Mapping[str, BehaviourResult],
    weather: pd.DataFrame,
    weather_member_id: str,
    thermal_assumptions: AssumptionContract,
    behaviour_assumptions: BehaviourAssumptionContract,
    seed: int,
) -> pd.DataFrame:
    active_heat = behaviour_assumptions.number("control.heating_active_C")
    cooling = behaviour_assumptions.number("control.cooling_C")
    records: list[dict[str, Any]] = []
    by_key = {(state.archetype_id, state.state_id): state for state in states}

    for household_class, archetype_id in REPRESENTATIVE_ARCHETYPES.items():
        try:
            state = by_key[(archetype_id, REPRESENTATIVE_STATE)]
        except KeyError as exc:
            raise CouplingError(
                f"Missing representative {archetype_id}/{REPRESENTATIVE_STATE}."
            ) from exc
        prepared = preprocess_archetype(state, thermal_assumptions)
        profile = profiles[household_class]
        constant = build_reference_schedules(
            weather["timestamp_utc"], prepared.floor_area_m2, thermal_assumptions
        )
        mean_gain = _schedule_with(
            profile,
            internal_gains_W=float(profile.hourly["Phi_int_W"].mean()),
            heating_setpoint_C=active_heat,
            cooling_setpoint_C=cooling,
        )
        dynamic_gain = _schedule_with(
            profile,
            heating_setpoint_C=active_heat,
            cooling_setpoint_C=cooling,
        )
        full = profile.schedules
        flattened_lighting = flatten_lighting_gain(profile)
        scenarios = {
            "A_constant_3_W_m2_20_26": constant,
            "B_mean_behavioural_gain_20_26": mean_gain,
            "C_dynamic_behavioural_gain_20_26": dynamic_gain,
            "D_full_behavioural": full,
            "E_full_flat_lighting": flattened_lighting,
        }
        values: dict[str, tuple[float, float]] = {}
        for scenario, schedules in scenarios.items():
            result = _simulate_schedules(
                prepared,
                weather,
                schedules,
                thermal_assumptions,
                weather_member_id=weather_member_id,
                seed=seed,
                scenario=f"gate4_decomposition_{scenario}",
            )
            values[scenario] = (
                result.diagnostics.heating_intensity_kWh_m2,
                result.diagnostics.cooling_intensity_kWh_m2,
            )
            records.append(
                {
                    "archetype_id": archetype_id,
                    "dwelling_type": state.dwelling_type,
                    "dwelling_class": household_class,
                    "state_id": state.state_id,
                    "occupant_count": profile.diagnostics.occupant_count,
                    "scenario": scenario,
                    "heating_kWh_m2": values[scenario][0],
                    "cooling_kWh_m2": values[scenario][1],
                    "heating_effect_vs_preceding_kWh_m2": np.nan,
                    "cooling_effect_vs_preceding_kWh_m2": np.nan,
                    "interpretation": "scenario total",
                }
            )

        comparisons = (
            (
                "gain_magnitude",
                "B_mean_behavioural_gain_20_26",
                "A_constant_3_W_m2_20_26",
                "mean behavioural gain minus constant 3 W/m2",
            ),
            (
                "gain_timing",
                "C_dynamic_behavioural_gain_20_26",
                "B_mean_behavioural_gain_20_26",
                "dynamic minus time-constant gains at equal annual gain",
            ),
            (
                "setpoint_setback",
                "D_full_behavioural",
                "C_dynamic_behavioural_gain_20_26",
                "18/20 C occupancy schedule minus constant 20 C",
            ),
            (
                "lighting_weather_timing",
                "D_full_behavioural",
                "E_full_flat_lighting",
                "weather-driven lighting minus equal-energy flat lighting",
            ),
        )
        for effect, left, right, interpretation in comparisons:
            records.append(
                {
                    "archetype_id": archetype_id,
                    "dwelling_type": state.dwelling_type,
                    "dwelling_class": household_class,
                    "state_id": state.state_id,
                    "occupant_count": profile.diagnostics.occupant_count,
                    "scenario": f"effect_{effect}",
                    "heating_kWh_m2": np.nan,
                    "cooling_kWh_m2": np.nan,
                    "heating_effect_vs_preceding_kWh_m2": (
                        values[left][0] - values[right][0]
                    ),
                    "cooling_effect_vs_preceding_kWh_m2": (
                        values[left][1] - values[right][1]
                    ),
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame.from_records(records)


def run_deterministic_coupling(
    *,
    seed: int = DEFAULT_COUPLING_SEED,
    reference_year: int | None = None,
    states: Sequence[ArchetypeStateInput] | None = None,
    thermal_assumptions: AssumptionContract | None = None,
    behaviour_assumptions: BehaviourAssumptionContract | None = None,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare one fixed behavioural realization with Gate-3 conditions."""

    thermal_contract = thermal_assumptions or load_assumption_contract(
        DEFAULT_ASSUMPTIONS_PATH
    )
    behaviour_contract = behaviour_assumptions or load_behaviour_assumptions()
    weather, weather_metadata = load_reference_weather(reference_year)
    weather_member_id = str(weather_metadata["weather_member_id"])
    unique_states = list(states) if states is not None else load_unique_archetype_states()
    if len({(state.archetype_id, state.state_id) for state in unique_states}) != len(
        unique_states
    ):
        raise CouplingError("Archetype-state inputs must be unique.")

    profiles = {
        household_class: _profile_for_class(
            household_class,
            weather,
            weather_member_id,
            seed,
            behaviour_contract,
        )
        for household_class in ("SFH", "MFH")
    }

    records: list[dict[str, Any]] = []
    for state in unique_states:
        prepared = preprocess_archetype(state, thermal_contract)
        household_class = dwelling_class(state.dwelling_type)
        profile = profiles[household_class]
        baseline_schedules = build_reference_schedules(
            weather["timestamp_utc"], prepared.floor_area_m2, thermal_contract
        )
        baseline = _simulate_schedules(
            prepared,
            weather,
            baseline_schedules,
            thermal_contract,
            weather_member_id=weather_member_id,
            seed=seed,
            scenario="gate4_constant_gain_reference",
        )
        behavioural = _simulate_schedules(
            prepared,
            weather,
            profile.schedules,
            thermal_contract,
            weather_member_id=weather_member_id,
            seed=seed,
            scenario="gate4_fixed_behaviour",
        )
        records.append(
            _summarize_pair(state, prepared, baseline, behavioural, profile)
        )

    results = pd.DataFrame.from_records(records)
    state_rank = {state: rank for rank, state in enumerate(STATE_ORDER)}
    results["_state_rank"] = results["state_id"].map(state_rank)
    results["_type_number"] = results["archetype_id"].str[-2:].astype(int)
    results = (
        results.sort_values(["_type_number", "_state_rank"], kind="stable")
        .drop(columns=["_type_number", "_state_rank"])
        .reset_index(drop=True)
    )
    decomposition = _effect_decomposition(
        unique_states,
        profiles,
        weather,
        weather_member_id,
        thermal_contract,
        behaviour_contract,
        seed,
    )

    numerical_pass = bool(
        (
            results["max_energy_balance_residual_W"]
            <= ENERGY_BALANCE_TOLERANCE_W
        ).all()
        and (
            results[
                ["max_heating_setpoint_error_K", "max_cooling_setpoint_error_K"]
            ].max(axis=1)
            <= SETPOINT_TRACKING_TOLERANCE_K
        ).all()
        and (~results["simultaneous_heating_cooling"]).all()
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "gate": "Gate 4 deterministic occupant-behaviour coupling",
        "verification_status": "PASS" if numerical_pass else "FAIL",
        "cell_count": len(results),
        "fixed_seed": seed,
        "reference_weather": weather_metadata,
        "profile_policy": (
            "one fixed realization per SFH/MFH class, reused across all 75 cells"
        ),
        "profiles": {
            household_class: asdict(profile.diagnostics)
            for household_class, profile in profiles.items()
        },
        "comparison": {
            "median_heating_change_kWh_m2": float(
                results["heating_change_kWh_m2"].median()
            ),
            "minimum_heating_change_kWh_m2": float(
                results["heating_change_kWh_m2"].min()
            ),
            "maximum_heating_change_kWh_m2": float(
                results["heating_change_kWh_m2"].max()
            ),
            "median_cooling_change_kWh_m2": float(
                results["cooling_change_kWh_m2"].median()
            ),
        },
        "numerical_checks": {
            "energy_balance_tolerance_W": ENERGY_BALANCE_TOLERANCE_W,
            "setpoint_tracking_tolerance_K": SETPOINT_TRACKING_TOLERANCE_K,
            "maximum_energy_balance_residual_W": float(
                results["max_energy_balance_residual_W"].max()
            ),
            "maximum_setpoint_error_K": float(
                results[
                    [
                        "max_heating_setpoint_error_K",
                        "max_cooling_setpoint_error_K",
                    ]
                ].to_numpy(dtype=float).max()
            ),
            "simultaneous_heating_cooling_cells": int(
                results["simultaneous_heating_cooling"].sum()
            ),
        },
        "provenance": {
            "thermal_assumptions_sha256": thermal_contract.sha256,
            "behaviour_assumptions_sha256": behaviour_contract.sha256,
        },
    }

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        results.to_csv(destination / "deterministic_coupling_comparison.csv", index=False)
        decomposition.to_csv(destination / "coupling_effect_decomposition.csv", index=False)
        for household_class, profile in profiles.items():
            profile.hourly.to_csv(
                destination / f"fixed_profile_{household_class.lower()}.csv",
                index=False,
            )
        pd.DataFrame(
            [asdict(profile.diagnostics) for profile in profiles.values()]
        ).to_csv(destination / "fixed_profile_diagnostics.csv", index=False)
        _write_json(destination / "coupling_summary.json", summary)
    return results, decomposition, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_COUPLING_SEED)
    parser.add_argument("--reference-year", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    _, _, summary = run_deterministic_coupling(
        seed=args.seed,
        reference_year=args.reference_year,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return 0 if summary["verification_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
