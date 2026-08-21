from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from thermal_model.monte_carlo.contracts import (
    MonteCarloContractError,
    canonical_sha256,
)
from thermal_model.monte_carlo.design import make_seed_bank, ordered_seed_bank_sha256
from thermal_model.monte_carlo.runner import main as monte_carlo_main
from thermal_model.monte_carlo import sensitivity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weather_records() -> list[dict]:
    records = []
    for scenario in sensitivity.SENSITIVITY_CLIMATE_SCENARIOS:
        for year in sensitivity.SENSITIVITY_WEATHER_YEARS:
            records.append(
                {
                    "weather_member_id": f"weather_2050_{scenario}_pvgis_{year}",
                    "climate_scenario_id": scenario,
                    "weather_pair_id": f"pvgis_{year}",
                    "observed_pvgis_year": year,
                    "climate_target": "2041-2060",
                    "member_sha256": "1" * 64,
                    "metadata_sha256": "2" * 64,
                    "manifest_sha256": "3" * 64,
                    "morph_contract_sha256": "4" * 64,
                    "facade_source_sha256_json": '{"east":"test"}',
                    "weather_contract_sha256": "5" * 64,
                    "weather_forcing_sha256": "6" * 64,
                }
            )
    return records


def _minimal_design(stage: int) -> dict:
    seeds = make_seed_bank(stage, master_seed=sensitivity.SENSITIVITY_MASTER_SEED)
    design = {
        "streaming_stock_contract_version": "test_streaming_v1",
        "model_contract_version": "test_model_v1",
        "central_thermal_assumptions_sha256": "a" * 64,
        "behaviour_assumptions_sha256": "b" * 64,
        "occupant_distribution_sha256": "c" * 64,
        "archetype_states": [
            {
                "archetype_id": f"A{index:02d}",
                "state_id": "state",
                "archetype_state_sha256": f"{index % 10}" * 64,
            }
            for index in range(75)
        ],
        "weather_members": _weather_records(),
        "occupant_seeds": list(seeds),
        "occupant_seed_bank_sha256": ordered_seed_bank_sha256(seeds),
        "convergence_evidence": {"status": "NOT_VERIFIED_BY_RUNNER"},
        "model_scenarios": [
            sensitivity.resolve_model_scenario(item).definition()
            for item in sorted(sensitivity.SENSITIVITY_MODEL_SCENARIOS)
        ],
        "stock_weights_sha256": "d" * 64,
        "stock_weights_source_sha256": "e" * 64,
        "require_full_stock": True,
        "partition_specs": [
            {
                "partition_id": f"partition_{index:03d}",
                "weather_member_id": _weather_records()[index // 6]["weather_member_id"],
                "model_scenario_id": sorted(sensitivity.SENSITIVITY_MODEL_SCENARIOS)[
                    index % 6
                ],
            }
            for index in range(108)
        ],
        "expected_run_count": 75 * 18 * stage * 6,
    }
    design["design_sha256"] = canonical_sha256(design)
    return design


def _write_contract(root: Path, stage: int = 40, *, previous_stage: dict | None = None) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    design = _minimal_design(stage)
    design_path = root / "streaming_design_contract.json"
    _write_json(design_path, design)
    seeds = make_seed_bank(stage, master_seed=sensitivity.SENSITIVITY_MASTER_SEED)
    audit_payload = {
        "selection_basis_climate_scenario_id": "rcp_4_5",
        "candidate_years": list(range(2006, 2024)),
        "candidate_records": [],
        "selected_role_to_year": {
            item["role"]: item["observed_pvgis_year"]
            for item in sensitivity.SENSITIVITY_WEATHER_STRATA
        },
    }
    audit = {
        **audit_payload,
        "weather_selection_audit_sha256": canonical_sha256(audit_payload),
    }
    payload = {
        "structural_sensitivity_contract_version": (
            sensitivity.STRUCTURAL_SENSITIVITY_CONTRACT_VERSION
        ),
        "selection_contract_sha256": sensitivity.SENSITIVITY_SELECTION_SHA256,
        "target_seed_count": stage,
        "active_seed_checkpoints": list(sensitivity._active_checkpoints(stage)),
        "occupant_seeds": list(seeds),
        "occupant_seed_bank_sha256": ordered_seed_bank_sha256(seeds),
        "weather_selection_audit": audit,
        "weather_members": design["weather_members"],
        "streaming_design_sha256": design["design_sha256"],
        "streaming_design_contract_file_sha256": _sha256(design_path),
        "streaming_design_basis_sha256": sensitivity._design_basis_sha256(design),
        "previous_stage": previous_stage,
    }
    contract = {
        **payload,
        "structural_sensitivity_contract_sha256": canonical_sha256(payload),
    }
    _write_json(root / sensitivity.SENSITIVITY_CONTRACT_FILENAME, contract)
    return contract


def _panel_deltas(stage: int, *, unstable: bool = False) -> pd.DataFrame:
    seeds = make_seed_bank(stage, master_seed=sensitivity.SENSITIVITY_MASTER_SEED)
    records = []
    for cell in sensitivity.SENSITIVITY_PANEL_CELLS:
        for climate in sensitivity.SENSITIVITY_CLIMATE_SCENARIOS:
            for year in sensitivity.SENSITIVITY_WEATHER_YEARS:
                member_id = f"weather_2050_{climate}_pvgis_{year}"
                for seed_rank, seed in enumerate(seeds, start=1):
                    for scenario_id in sensitivity.SENSITIVITY_COMPARISON_SCENARIOS:
                        axis = sensitivity.resolve_model_scenario(scenario_id).axis
                        for metric, floor in sensitivity.SENSITIVITY_METRIC_FLOORS.items():
                            delta = 2.0 * floor
                            if unstable and cell == sensitivity.SENSITIVITY_PANEL_CELLS[0]:
                                delta *= 1.0 if seed_rank <= 10 else (3.0 if seed_rank <= 20 else 8.0)
                            baseline = 100.0 * floor
                            records.append(
                                {
                                    "archetype_id": cell["archetype_id"],
                                    "state_id": cell["state_id"],
                                    "climate_scenario_id": climate,
                                    "weather_member_id": member_id,
                                    "weather_pair_id": f"pvgis_{year}",
                                    "observed_pvgis_year": year,
                                    "occupant_seed": seed,
                                    "baseline_model_scenario_id": "central",
                                    "comparison_model_scenario_id": scenario_id,
                                    "comparison_model_scenario_axis": axis,
                                    "metric": metric,
                                    "baseline_value": baseline,
                                    "comparison_value": baseline + delta,
                                    "delta": delta,
                                }
                            )
    return pd.DataFrame.from_records(records)


def test_frozen_selection_and_seed_prefix_are_exact() -> None:
    assert sensitivity.SENSITIVITY_WEATHER_YEARS == (2010, 2013, 2015, 2019, 2020, 2022)
    assert sensitivity.SENSITIVITY_MODEL_SCENARIOS == (
        "central",
        "infiltration_half",
        "infiltration_one_and_half",
        "mass_light",
        "mass_heavy",
        "shading_unshaded",
    )
    seeds = make_seed_bank(160, master_seed=20250808)
    assert seeds[0] == 1203443498
    assert seeds[39] == 1182233951
    assert ordered_seed_bank_sha256(seeds[:40]) == (
        "d9c1d0f471fb97f0efab0c100463285cd55e4de7718f288f7be26835b3adb0fe"
    )
    assert len(sensitivity.SENSITIVITY_SELECTION_SHA256) == 64


def test_prepare_persists_exact_workflow_only_contract(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "screen"
    fake_states = [
        SimpleNamespace(archetype_id=f"A{index:02d}", state_id="state")
        for index in range(75)
    ]
    monkeypatch.setattr(sensitivity, "load_unique_archetype_states", lambda: fake_states)
    monkeypatch.setattr(
        sensitivity,
        "_selected_member_ids",
        lambda: tuple(item["weather_member_id"] for item in _weather_records()),
    )
    monkeypatch.setattr(
        sensitivity,
        "load_weather_members",
        lambda identifiers: tuple(SimpleNamespace(member_id=item) for item in identifiers),
    )
    audit_payload = {
        "selection_basis_climate_scenario_id": "rcp_4_5",
        "candidate_years": list(range(2006, 2024)),
        "candidate_records": [],
        "selected_role_to_year": {
            item["role"]: item["observed_pvgis_year"]
            for item in sensitivity.SENSITIVITY_WEATHER_STRATA
        },
    }
    monkeypatch.setattr(
        sensitivity,
        "_audit_weather_selection",
        lambda _: {
            **audit_payload,
            "weather_selection_audit_sha256": canonical_sha256(audit_payload),
        },
    )
    calls = []

    def fake_execute(states, members, seeds, scenarios, **kwargs):
        calls.append((states, members, seeds, scenarios, kwargs))
        design = _minimal_design(len(seeds))
        _write_json(root / "streaming_design_contract.json", design)
        return {"status": "PREPARED", "design_sha256": design["design_sha256"]}

    monkeypatch.setattr(sensitivity, "execute_streaming_stock_design", fake_execute)
    result = sensitivity.prepare_structural_sensitivity_screen(
        root, target_seed_count=40, max_workers=3
    )
    assert result["status"] == "PREPARED"
    assert result["production_qualification"] == "NOT_A_PRODUCTION_PASS"
    assert result["expected_run_count"] == 324_000
    assert len(calls) == 1
    assert calls[0][-1]["require_full_stock"] is True
    assert calls[0][-1]["require_convergence_evidence"] is False
    assert calls[0][-1]["prepare_only"] is True
    contract = sensitivity._load_contract(root)
    assert contract["coverage"] == {
        "occupant_seed_prefix": "N40_OF_160",
        "stock_weight_coverage": "AUTHORITATIVE_2050_WEIGHTS",
        "structural_scenario_coverage": "ALL_SIX_DECLARED_SCENARIOS",
        "weather_coverage": "STRATIFIED_6_OF_18_PER_RCP",
    }
    sensitivity.prepare_structural_sensitivity_screen(
        root, target_seed_count=40, max_workers=3
    )
    assert len(calls) == 2


def test_delta_stability_selects_n40_after_two_passing_expansions(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "screen"
    contract = _write_contract(root, 40)
    post = {"postprocess_contract_version": "test_post_v1"}
    _write_json(root / sensitivity.POSTPROCESS_SUMMARY_FILENAME, post)
    monkeypatch.setattr(
        sensitivity,
        "streaming_stock_status",
        lambda _: {"status": "WORKFLOW_CHECK_ONLY"},
    )
    monkeypatch.setattr(
        sensitivity,
        "postprocessing_status",
        lambda _: {"status": "PASS"},
    )
    monkeypatch.setattr(
        sensitivity,
        "_read_panel_deltas",
        lambda _root, _contract: _panel_deltas(40),
    )
    summary = sensitivity.evaluate_structural_sensitivity_screen(root)
    assert summary["status"] == "STRUCTURAL_SENSITIVITY_SCREEN_STABLE_AT_N40"
    assert summary["stable_at_seed_count"] == 40
    assert summary["next_stage_target_seed_count"] is None
    assert summary["production_qualification"] == "NOT_A_PRODUCTION_PASS"
    results = pd.read_csv(root / sensitivity.SENSITIVITY_RESULTS_FILENAME)
    assert not results.loc[results["seed_count"] == 10, "criterion_pass"].any()
    assert results.loc[results["seed_count"] == 20, "panel_all_groups_pass"].all()
    assert results.loc[results["seed_count"] == 40, "panel_stable_at_checkpoint"].all()
    assert summary["structural_sensitivity_contract_sha256"] == contract[
        "structural_sensitivity_contract_sha256"
    ]


def test_delta_nonstabilisation_requests_only_prospective_n80_stage(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "screen"
    _write_contract(root, 40)
    _write_json(
        root / sensitivity.POSTPROCESS_SUMMARY_FILENAME,
        {"postprocess_contract_version": "test_post_v1"},
    )
    monkeypatch.setattr(
        sensitivity,
        "streaming_stock_status",
        lambda _: {"status": "WORKFLOW_CHECK_ONLY"},
    )
    monkeypatch.setattr(
        sensitivity,
        "postprocessing_status",
        lambda _: {"status": "PASS"},
    )
    monkeypatch.setattr(
        sensitivity,
        "_read_panel_deltas",
        lambda _root, _contract: _panel_deltas(40, unstable=True),
    )
    summary = sensitivity.evaluate_structural_sensitivity_screen(root)
    assert summary["status"] == "STRUCTURAL_SENSITIVITY_SCREEN_NOT_STABLE_AT_N40"
    assert summary["decision"] == "EXTEND_TO_N80"
    assert summary["next_stage_target_seed_count"] == 80
    assert summary["stable_at_seed_count"] is None


def test_invalid_stage_and_cli_status_are_bounded(tmp_path: Path, capsys) -> None:
    with pytest.raises(MonteCarloContractError, match="prospectively restricted"):
        sensitivity.default_structural_sensitivity_output_dir(41)
    assert monte_carlo_main(
        [
            "sensitivity",
            "--status",
            "--stage",
            "40",
            "--output-dir",
            str(tmp_path / "absent"),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "NOT_PREPARED"
    assert payload["production_qualification"] == "NOT_A_PRODUCTION_PASS"
