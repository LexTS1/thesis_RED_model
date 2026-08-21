from __future__ import annotations

import json

import pandas as pd
import pytest

from thermal_model.behaviour.coupling import DEFAULT_OUTPUT_DIR


def test_persisted_gate4_coupling_artifacts_reconcile() -> None:
    results = pd.read_csv(DEFAULT_OUTPUT_DIR / "deterministic_coupling_comparison.csv")
    decomposition = pd.read_csv(
        DEFAULT_OUTPUT_DIR / "coupling_effect_decomposition.csv"
    )
    diagnostics = pd.read_csv(DEFAULT_OUTPUT_DIR / "fixed_profile_diagnostics.csv")
    summary = json.loads(
        (DEFAULT_OUTPUT_DIR / "coupling_summary.json").read_text(encoding="utf-8")
    )

    assert len(results) == summary["cell_count"] == 75
    assert results[["archetype_id", "state_id"]].duplicated().sum() == 0
    assert summary["verification_status"] == "PASS"
    assert results["simultaneous_heating_cooling"].eq(False).all()
    assert results["max_energy_balance_residual_W"].max() <= 1.0e-6
    assert results[
        ["max_heating_setpoint_error_K", "max_cooling_setpoint_error_K"]
    ].to_numpy().max() <= 1.0e-8

    assert set(diagnostics["dwelling_class"]) == {"SFH", "MFH"}
    assert diagnostics["annual_total_electricity_kWh"].to_numpy() == pytest.approx(
        [3500.0, 3500.0], abs=1.0e-6
    )
    assert set(decomposition["archetype_id"]) == {"BE_TABULA_11", "BE_TABULA_14"}
    assert len(decomposition) == 18
    assert decomposition["scenario"].str.startswith("effect_").sum() == 8


def test_persisted_profiles_are_complete_and_match_diagnostics() -> None:
    diagnostics = pd.read_csv(DEFAULT_OUTPUT_DIR / "fixed_profile_diagnostics.csv").set_index(
        "dwelling_class"
    )

    for household_class in ("SFH", "MFH"):
        profile = pd.read_csv(
            DEFAULT_OUTPUT_DIR / f"fixed_profile_{household_class.lower()}.csv"
        )
        assert len(profile) == 8760
        assert profile["timestamp_utc"].nunique() == 8760
        assert (profile[["total_electricity_W", "Phi_int_W"]] >= 0.0).all().all()
        assert profile["total_electricity_W"].sum() / 1000.0 == pytest.approx(
            diagnostics.loc[household_class, "annual_total_electricity_kWh"],
            abs=1.0e-6,
        )
