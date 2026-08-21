from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.validation import (
    DEFAULT_OUTPUT_DIR,
    TABULA_PROVENANCE_PATH,
    TABULA_TARGET_PATH,
    ValidationError,
    load_reference_weather,
    load_tabula_targets,
    load_unique_archetype_states,
    select_reference_weather_year,
    summarize_qualitative_patterns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tabula_reference_matrix_is_complete_and_matches_frozen_source() -> None:
    targets = load_tabula_targets()
    provenance = json.loads(TABULA_PROVENANCE_PATH.read_text(encoding="utf-8"))
    source = PROJECT_ROOT / provenance["source_path"]

    assert len(targets) == 75
    assert not targets.duplicated(["archetype_id", "state_id"]).any()
    assert set(targets["state_id"]) == {
        "TABULA_existing",
        "TABULA_standard_B_proxy",
        "TABULA_advanced_A_proxy",
    }
    assert source.exists()
    assert _sha256(source) == provenance["source_sha256"]
    assert _sha256(TABULA_TARGET_PATH) == (
        "6fc04de2393e97e843687222ad136ef5f913edb0e30dd4054699dc4b7145528c"
    )


def test_tabula_loader_fails_on_incomplete_source(tmp_path: Path) -> None:
    incomplete = pd.read_csv(TABULA_TARGET_PATH).iloc[:-1]
    path = tmp_path / "incomplete.csv"
    incomplete.to_csv(path, index=False)
    with pytest.raises(ValidationError, match="25 unique archetypes"):
        load_tabula_targets(path)


def test_reference_weather_selection_and_alignment_are_frozen() -> None:
    selection = select_reference_weather_year()
    weather, metadata = load_reference_weather()

    assert selection["selected_year"] == 2015
    assert metadata["weather_member_id"] == "pvgis_sarah3_observed_2015_reference"
    assert len(weather) == 8760
    assert weather["timestamp_utc"].iloc[0] == pd.Timestamp(
        "2015-01-01T00:00:00Z"
    )
    assert weather["timestamp_utc"].iloc[-1] == pd.Timestamp(
        "2015-12-31T23:00:00Z"
    )
    assert weather["timestamp_utc"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    irradiance = weather[
        [
            "I_north_W_m2",
            "I_east_W_m2",
            "I_south_W_m2",
            "I_west_W_m2",
        ]
    ]
    assert np.isfinite(weather.select_dtypes(include="number")).all().all()
    assert (irradiance >= 0.0).all().all()
    assert weather["I_solar_W_m2"].to_numpy() == pytest.approx(
        (
            weather["I_beam_horizontal_W_m2"]
            + weather["I_diffuse_horizontal_W_m2"]
        ).to_numpy(),
        abs=1.0e-6,
    )
    assert metadata["horizontal_annual_kWh_m2"] == pytest.approx(
        weather["I_solar_W_m2"].sum() / 1000.0,
        abs=1.0e-10,
    )
    assert metadata["facade_annual_kWh_m2"] == pytest.approx(
        {
            "north": 303.84355,
            "east": 709.42022,
            "south": 976.05416,
            "west": 689.67995,
        },
        abs=1.0e-8,
    )


def test_stock_validation_scope_is_exactly_twenty_five_by_three() -> None:
    states = load_unique_archetype_states()
    identities = {(state.archetype_id, state.state_id) for state in states}
    assert len(states) == 75
    assert len(identities) == 75
    assert len({state.archetype_id for state in states}) == 25


def test_persisted_gate3_results_reconcile_with_summary() -> None:
    results = pd.read_csv(DEFAULT_OUTPUT_DIR / "deterministic_archetype_validation.csv")
    summary = json.loads(
        (DEFAULT_OUTPUT_DIR / "validation_summary.json").read_text(encoding="utf-8")
    )
    patterns = summarize_qualitative_patterns(results)

    assert len(results) == summary["cell_count"] == 75
    assert summary["verification_status"] == "PASS"
    assert summary["validation_status"] == "PASS"
    assert int(results["within_predeclared_tabula_band"].sum()) == 68
    assert summary["tabula_comparison"]["pass_rate"] == pytest.approx(68.0 / 75.0)
    for name, persisted in summary["qualitative_patterns"].items():
        calculated = patterns[name]
        assert calculated["passed"] == persisted["passed"]
        assert calculated["total"] == persisted["total"]
        assert calculated["rate"] == pytest.approx(persisted["rate"])
    assert results["simultaneous_heating_cooling"].eq(False).all()
    assert results["max_energy_balance_residual_W"].max() <= 1.0e-6
    assert results[
        ["max_heating_setpoint_error_K", "max_cooling_setpoint_error_K"]
    ].to_numpy().max() <= 1.0e-8


def test_sensitivity_artifacts_cover_each_predeclared_axis() -> None:
    results = pd.read_csv(DEFAULT_OUTPUT_DIR / "sensitivity_results.csv")
    summary = json.loads(
        (DEFAULT_OUTPUT_DIR / "sensitivity_summary.json").read_text(encoding="utf-8")
    )
    expected_axes = {
        "thermal_mass",
        "fixed_shading",
        "window_frame_fraction",
        "infiltration",
        "ventilation_rate",
        "heating_setpoint",
        "cooling_setpoint",
        "internal_gains",
        "boundary_treatment",
        "hrv",
    }
    assert len(results) == summary["case_count"] == 19
    assert expected_axes.issubset(set(results["axis"]))
    assert all(summary["directional_checks"].values())
    assert results["max_energy_balance_residual_W"].max() <= 1.0e-6
