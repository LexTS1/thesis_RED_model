from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from thermal_model.monte_carlo.aggregation import validate_stock_weights
from thermal_model.monte_carlo.contracts import MonteCarloContractError, canonical_sha256
from thermal_model.monte_carlo.postprocess import (
    postprocess_production_results,
    postprocessing_status,
)
from thermal_model.monte_carlo.runner import main as monte_carlo_main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _test_weights() -> pd.DataFrame:
    counts = {
        "TABULA_existing": 50.0,
        "TABULA_standard_B_proxy": 30.0,
        "TABULA_advanced_A_proxy": 20.0,
    }
    records = []
    for state_id, count in counts.items():
        records.append(
            {
                "scenario": "central",
                "target_year": 2050,
                "region": "Test Region",
                "archetype_id": "A1",
                "dwelling_type": "Detached house",
                "construction_period": "1971-1990",
                "state_id": state_id,
                "renovation_state": state_id,
                "state_dwellings": count,
                "state_dwellings_2050": count,
                "state_share_within_region_2050": count / 100.0,
                "regional_number_of_dwellings": 100.0,
                "regional_modelled_stock_dwellings": 100.0,
            }
        )
    return validate_stock_weights(
        pd.DataFrame.from_records(records), require_authoritative_shape=False
    )


def _make_completed_design(root: Path) -> pd.DataFrame:
    root.mkdir()
    partitions = root / "partitions"
    partitions.mkdir()
    weights = _test_weights()
    state_ids = [
        "TABULA_existing",
        "TABULA_standard_B_proxy",
        "TABULA_advanced_A_proxy",
    ]
    state_hashes = {
        state_id: character * 64
        for state_id, character in zip(state_ids, ("a", "b", "c"))
    }
    weather_records = [
        {
            "weather_member_id": f"weather_{year}",
            "climate_scenario_id": "rcp_4_5",
            "weather_pair_id": f"pvgis_{year}",
            "observed_pvgis_year": year,
            "climate_target": "2041-2060",
            "member_sha256": "d" * 64,
            "metadata_sha256": "e" * 64,
            "manifest_sha256": "f" * 64,
            "morph_contract_sha256": "1" * 64,
            "facade_source_sha256_json": '{"east":"test"}',
            "weather_contract_sha256": str(year)[-1] * 64,
            "weather_forcing_sha256": str(year - 1)[-1] * 64,
        }
        for year in (2010, 2011)
    ]
    scenarios = [
        {"scenario_id": "central", "axis": "central", "description": "central"},
        {
            "scenario_id": "mass_heavy",
            "axis": "thermal_mass",
            "description": "heavy",
        },
    ]
    partition_specs = [
        {
            "partition_id": f"part_{weather['observed_pvgis_year']}_{scenario['scenario_id']}",
            "weather_member_id": weather["weather_member_id"],
            "model_scenario_id": scenario["scenario_id"],
        }
        for weather in weather_records
        for scenario in scenarios
    ]
    design = {
        "streaming_stock_contract_version": "test_streaming_v1",
        "model_contract_version": "test_model_v1",
        "central_thermal_assumptions_sha256": "2" * 64,
        "behaviour_assumptions_sha256": "3" * 64,
        "occupant_distribution_sha256": "4" * 64,
        "archetype_states": [
            {
                "archetype_id": "A1",
                "state_id": state_id,
                "archetype_state_sha256": state_hashes[state_id],
            }
            for state_id in state_ids
        ],
        "weather_members": weather_records,
        "occupant_seeds": [11, 22],
        "occupant_seed_bank_sha256": "5" * 64,
        "convergence_evidence": {"status": "TEST"},
        "model_scenarios": scenarios,
        "stock_weights_sha256": str(weights["stock_weights_sha256"].iloc[0]),
        "stock_weights_source_sha256": str(
            weights["stock_weights_source_sha256"].iloc[0]
        ),
        "require_full_stock": False,
        "partition_specs": partition_specs,
        "expected_run_count": 24,
    }
    design["design_sha256"] = canonical_sha256(design)
    _write_json(root / "streaming_design_contract.json", design)

    index_records = []
    scenario_hashes = {"central": "6" * 64, "mass_heavy": "7" * 64}
    effective_hashes = {"central": "8" * 64, "mass_heavy": "9" * 64}
    for spec in partition_specs:
        weather = next(
            item
            for item in weather_records
            if item["weather_member_id"] == spec["weather_member_id"]
        )
        scenario = next(
            item for item in scenarios if item["scenario_id"] == spec["model_scenario_id"]
        )
        partition_dir = partitions / spec["partition_id"]
        partition_dir.mkdir()
        manifests = []
        diagnostics = []
        for state_rank, state_id in enumerate(state_ids):
            for seed_rank, seed in enumerate((11, 22), start=1):
                run_id = f"{spec['partition_id']}_{state_rank}_{seed}"
                manifest = {
                    "run_id": run_id,
                    "archetype_id": "A1",
                    "dwelling_type": "Detached house",
                    "construction_period": "1971-1990",
                    "state_id": state_id,
                    "archetype_state_sha256": state_hashes[state_id],
                    "climate_scenario_id": "rcp_4_5",
                    "weather_member_id": weather["weather_member_id"],
                    "weather_pair_id": weather["weather_pair_id"],
                    "observed_pvgis_year": weather["observed_pvgis_year"],
                    "occupant_seed": seed,
                    "occupant_seed_rank": seed_rank,
                    "model_scenario_id": scenario["scenario_id"],
                    "model_scenario_axis": scenario["axis"],
                    "weather_contract_sha256": weather["weather_contract_sha256"],
                    "model_scenario_sha256": scenario_hashes[scenario["scenario_id"]],
                    "weather_forcing_sha256": weather["weather_forcing_sha256"],
                    "effective_thermal_assumptions_sha256": effective_hashes[
                        scenario["scenario_id"]
                    ],
                    "behaviour_assumptions_sha256": design[
                        "behaviour_assumptions_sha256"
                    ],
                    "occupant_distribution_sha256": design[
                        "occupant_distribution_sha256"
                    ],
                }
                manifests.append(manifest)
                offset = (
                    state_rank * 10.0
                    + (weather["observed_pvgis_year"] - 2010) * 2.0
                    + seed_rank
                    + (5.0 if scenario["scenario_id"] == "mass_heavy" else 0.0)
                )
                diagnostics.append(
                    {
                        **{
                            key: value
                            for key, value in manifest.items()
                            if key != "occupant_seed_rank"
                        },
                        "dwelling_class": "SFH",
                        "floor_area_m2": 100.0,
                        "climate_target": weather["climate_target"],
                        "annual_heating_kWh": 1000.0 + offset,
                        "annual_cooling_kWh": 100.0 + offset,
                        "heating_intensity_kWh_m2": 10.0 + offset / 100.0,
                        "cooling_intensity_kWh_m2": 1.0 + offset / 100.0,
                        "peak_heating_W": 5000.0 + offset,
                        "peak_cooling_W": 1000.0 + offset,
                        "heating_full_load_equivalent_hours": 200.0 + offset,
                        "cooling_full_load_equivalent_hours": 100.0 + offset,
                        "model_contract_version": design["model_contract_version"],
                        "central_thermal_assumptions_sha256": design[
                            "central_thermal_assumptions_sha256"
                        ],
                        "member_sha256": weather["member_sha256"],
                        "metadata_sha256": weather["metadata_sha256"],
                        "climate_manifest_sha256": weather["manifest_sha256"],
                        "morph_contract_sha256": weather["morph_contract_sha256"],
                        "facade_source_sha256_json": weather[
                            "facade_source_sha256_json"
                        ],
                    }
                )
        manifest_path = partition_dir / "run_manifest.csv"
        diagnostics_path = partition_dir / "run_diagnostics.csv"
        pd.DataFrame.from_records(manifests).to_csv(manifest_path, index=False)
        pd.DataFrame.from_records(diagnostics).to_csv(diagnostics_path, index=False)
        artifacts = {
            "run_manifest.csv": {"sha256": _sha256(manifest_path), "row_count": 6},
            "run_diagnostics.csv": {
                "sha256": _sha256(diagnostics_path),
                "row_count": 6,
            },
        }
        complete = {
            "status": "PASS",
            "design_sha256": design["design_sha256"],
            "partition_id": spec["partition_id"],
            "run_count": 6,
            "artifacts": artifacts,
        }
        complete_path = partition_dir / "partition_complete.json"
        _write_json(complete_path, complete)
        index_records.append(
            {
                "partition_id": spec["partition_id"],
                "weather_member_id": weather["weather_member_id"],
                "climate_scenario_id": weather["climate_scenario_id"],
                "model_scenario_id": scenario["scenario_id"],
                "run_count": 6,
                "run_diagnostics_path": str(diagnostics_path.relative_to(root)),
                "run_diagnostics_sha256": artifacts["run_diagnostics.csv"]["sha256"],
                "partition_complete_sha256": _sha256(complete_path),
            }
        )
    index_path = root / "partition_index.csv"
    pd.DataFrame.from_records(index_records).to_csv(index_path, index=False)
    summary = {
        "status": "PARTIAL_STOCK_WORKFLOW",
        "design_sha256": design["design_sha256"],
        "partition_count": 4,
        "completed_run_count": 24,
        "expected_run_count": 24,
        "artifact_sha256": {"partition_index.csv": _sha256(index_path)},
    }
    _write_json(root / "monte_carlo_summary.json", summary)
    return weights


def test_postprocessor_authenticates_streams_and_is_idempotent(tmp_path: Path) -> None:
    production = tmp_path / "production"
    weights = _make_completed_design(production)

    summary = postprocess_production_results(
        production,
        stock_weights=weights,
        chunk_rows=2,
    )
    assert summary["status"] == "PASS"
    assert summary["processed_run_count"] == 24
    assert summary["dwelling_hour_files_read"] == 0
    assert summary["maximum_in_memory_analysis_group_rows"] == 4
    assert summary["maximum_in_memory_weighted_group_rows"] == 12
    assert summary["maximum_in_memory_paired_renovation_group_rows"] == 12
    assert summary["maximum_in_memory_paired_model_scenario_group_rows"] == 8
    assert set(summary["output_artifacts"]) == {
        "distribution_summary.csv",
        "variance_contributions.csv",
        "paired_renovation_deltas.csv",
        "paired_model_scenario_deltas.csv",
        "stock_weighted_distribution_summary.csv",
    }

    assert len(pd.read_csv(production / "distribution_summary.csv")) == 48
    assert len(pd.read_csv(production / "variance_contributions.csv")) == 72
    assert len(pd.read_csv(production / "paired_renovation_deltas.csv")) == 96
    assert len(pd.read_csv(production / "paired_model_scenario_deltas.csv")) == 72
    weighted = pd.read_csv(production / "stock_weighted_distribution_summary.csv")
    assert len(weighted) == 32
    assert set(weighted["weight_basis"]) == {"2050 dwelling count"}
    assert set(weighted["total_weight_dwellings"]) == {100.0}
    assert weighted["interpretation"].str.contains("distinct from the unweighted").all()
    assert postprocessing_status(production)["status"] == "PASS"

    repeated = postprocess_production_results(
        production,
        stock_weights=weights,
        chunk_rows=2,
    )
    assert repeated == summary
    for filename, metadata in summary["output_artifacts"].items():
        assert _sha256(production / filename) == metadata["sha256"]


def test_postprocessor_rejects_tampered_partition_and_status_detects_output_tamper(
    tmp_path: Path,
) -> None:
    production = tmp_path / "production"
    weights = _make_completed_design(production)
    postprocess_production_results(production, stock_weights=weights, chunk_rows=3)

    output = production / "distribution_summary.csv"
    output.write_text(output.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    status = postprocessing_status(production)
    assert status["status"] == "INVALID"
    assert "checksum mismatch" in status["reason"]

    diagnostics = next((production / "partitions").glob("*/run_diagnostics.csv"))
    diagnostics.write_text(
        diagnostics.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(MonteCarloContractError, match="checksum mismatch"):
        postprocess_production_results(production, stock_weights=weights, chunk_rows=3)


def test_postprocessing_status_reports_not_run(tmp_path: Path, capsys) -> None:
    assert postprocessing_status(tmp_path)["status"] == "NOT_RUN"
    assert (
        monte_carlo_main(
            ["postprocess", "--output-dir", str(tmp_path), "--status"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "NOT_RUN"
