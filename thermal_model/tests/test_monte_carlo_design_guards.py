from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from thermal_model.monte_carlo import (
    ConvergenceRule,
    MonteCarloContractError,
    PROSPECTIVE_N160_CONVERGENCE_RULE,
    build_balanced_manifest,
    distribution_summary,
    load_weather_member,
    paired_model_scenario_deltas,
    simulate,
)
from thermal_model.validation import load_unique_archetype_states


def test_prospective_n160_rule_only_extends_the_immutable_checkpoint_contract() -> None:
    original = ConvergenceRule()
    prospective = PROSPECTIVE_N160_CONVERGENCE_RULE

    assert original.checkpoints == (5, 10, 20, 40, 80)
    assert prospective.checkpoints == (*original.checkpoints, 160)
    assert prospective.relative_tolerance == original.relative_tolerance
    assert (
        prospective.required_consecutive_expansions
        == original.required_consecutive_expansions
        == 2
    )
    assert prospective.metrics_and_absolute_floors == original.metrics_and_absolute_floors
    assert prospective.statistics == original.statistics
    with pytest.raises(FrozenInstanceError):
        prospective.checkpoints = original.checkpoints  # type: ignore[misc]


def _metric_record(*, weather: str, seed: int, scenario: str = "central") -> dict:
    offset = (0.5 if weather == "w2" else 0.0) + 0.1 * seed
    return {
        "archetype_id": "a1",
        "state_id": "s1",
        "climate_scenario_id": "rcp_4_5",
        "weather_member_id": weather,
        "weather_pair_id": "pvgis_2010" if weather == "w1" else "pvgis_2011",
        "observed_pvgis_year": 2010 if weather == "w1" else 2011,
        "occupant_seed": seed,
        "model_scenario_id": scenario,
        "model_scenario_axis": "central" if scenario == "central" else "thermal_mass",
        "annual_heating_kWh": 1000.0 + offset,
        "annual_cooling_kWh": 100.0 + offset,
        "heating_intensity_kWh_m2": 10.0 + offset,
        "cooling_intensity_kWh_m2": 1.0 + offset,
        "peak_heating_W": 5000.0 + offset,
        "peak_cooling_W": 1000.0 + offset,
        "heating_full_load_equivalent_hours": 200.0 + offset,
        "cooling_full_load_equivalent_hours": 100.0 + offset,
    }


def test_balanced_manifest_rejects_unpaired_rcp_weather_years() -> None:
    state = load_unique_archetype_states()[0]
    mismatched = [
        load_weather_member("weather_2050_rcp_2_6_pvgis_2010"),
        load_weather_member("weather_2050_rcp_4_5_pvgis_2015"),
    ]
    with pytest.raises(MonteCarloContractError, match="identical PVGIS weather-pair"):
        build_balanced_manifest([state], mismatched, [7])

    paired = [
        mismatched[0],
        load_weather_member("weather_2050_rcp_4_5_pvgis_2010"),
    ]
    manifest = build_balanced_manifest([state], paired, [7])
    assert len(manifest) == 2
    assert set(manifest["weather_pair_id"]) == {"pvgis_2010"}
    with pytest.raises(MonteCarloContractError, match="must be integers"):
        build_balanced_manifest([state], paired[:1], [7.5])


def test_leap_member_runs_through_public_interface_without_dropping_hours() -> None:
    state = load_unique_archetype_states()[0]
    member = load_weather_member("weather_2050_rcp_4_5_pvgis_2020")
    result = simulate(state, member, 20200808, "central")
    assert member.is_leap_year
    assert len(result.hourly) == 8784
    assert sum(count for _, count in result.diagnostics.heating_setpoint_hours) == 8784
    assert sum(count for _, count in result.diagnostics.cooling_setpoint_hours) == 8784


def test_distribution_summary_requires_complete_unique_weather_seed_grid() -> None:
    complete = pd.DataFrame.from_records(
        [_metric_record(weather=weather, seed=seed) for weather in ("w1", "w2") for seed in (1, 2)]
    )
    summary = distribution_summary(complete)
    assert set(summary["sample_count"]) == {4}

    with pytest.raises(MonteCarloContractError, match="not a complete"):
        distribution_summary(complete.iloc[:-1].copy())
    duplicate = pd.concat([complete, complete.iloc[[0]]], ignore_index=True)
    with pytest.raises(MonteCarloContractError, match="duplicate"):
        distribution_summary(duplicate)


def test_model_scenario_deltas_are_paired_on_weather_and_seed() -> None:
    central = pd.DataFrame.from_records(
        [_metric_record(weather="w1", seed=seed) for seed in (1, 2)]
    )
    heavy = central.copy(deep=True)
    heavy["model_scenario_id"] = "mass_heavy"
    heavy["model_scenario_axis"] = "thermal_mass"
    heavy["heating_intensity_kWh_m2"] -= 2.5
    paired = paired_model_scenario_deltas(
        pd.concat([central, heavy], ignore_index=True)
    )
    selected = paired.loc[paired["metric"] == "heating_intensity_kWh_m2"]
    assert len(selected) == 2
    assert selected["delta"].tolist() == pytest.approx([-2.5, -2.5])

    with pytest.raises(MonteCarloContractError, match="Missing paired model scenario"):
        paired_model_scenario_deltas(
            pd.concat([central, heavy.iloc[[0]]], ignore_index=True)
        )
