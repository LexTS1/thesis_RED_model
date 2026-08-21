from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from thermal_model.monte_carlo.aggregation import (
    STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION,
)
from thermal_model.monte_carlo.contracts import (
    MonteCarloContractError,
    canonical_sha256,
)
from thermal_model.monte_carlo.design import ordered_seed_bank_sha256
from thermal_model.monte_carlo.postprocess import (
    POSTPROCESS_CONTRACT_VERSION,
    POSTPROCESS_SUMMARY_FILENAME,
)
from thermal_model.monte_carlo.reporting import (
    FIGURE_PROVENANCE_FILENAME,
    FIGURE_BASENAMES,
    REPORTING_SUMMARY_FILENAME,
    authenticate_reporting_inputs,
    generate_production_report,
    reporting_status,
)
from thermal_model.monte_carlo.runner import (
    STREAMING_STOCK_CONTRACT_VERSION,
    main as monte_carlo_main,
)
from thermal_model.monte_carlo.stock_streaming import BELGIUM_REGION_ID


RCPS = ("rcp_2_6", "rcp_4_5", "rcp_8_5")
REGIONS = (
    "Flemish Region",
    "Walloon Region",
    "Brussels-Capital Region",
)
STATES = (
    "TABULA_existing",
    "TABULA_standard_B_proxy",
    "TABULA_advanced_A_proxy",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, records: list[dict]) -> None:
    pd.DataFrame.from_records(records).to_csv(
        path, index=False, float_format="%.17g", lineterminator="\n"
    )


def _rows(path: Path) -> int:
    return max(len(path.read_text(encoding="utf-8").splitlines()) - 1, 0)


def _ledger(path: Path) -> dict:
    return {"sha256": _sha(path), "row_count": _rows(path)}


def _distribution_records(stock: pd.DataFrame) -> list[dict]:
    metrics = (
        "annual_heating_GWh",
        "annual_potential_sensible_cooling_GWh",
        "coincident_peak_heating_MW",
        "coincident_peak_potential_cooling_MW",
    )
    records = []
    for (rcp, region), group in stock.groupby(
        ["climate_scenario_id", "region"], sort=True
    ):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            records.append(
                {
                    "climate_scenario_id": rcp,
                    "model_scenario_id": "central",
                    "model_scenario_axis": "central",
                    "stock_scenario_id": "central",
                    "target_year": 2050,
                    "region": region,
                    "weather_member_count": len(values),
                    "metric": metric,
                    "minimum": float(np.min(values)),
                    "p05": float(np.quantile(values, 0.05)),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "maximum": float(np.max(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "interval_interpretation": (
                        "descriptive empirical interval over included paired weather "
                        "members; not a complete prediction interval"
                    ),
                }
            )
    return records


def _completed_reporting_fixture(root: Path) -> None:
    root.mkdir()
    partitions = root / "partitions"
    partitions.mkdir()
    convergence_path = root / "convergence_results.csv"
    _write_csv(convergence_path, [{"status": "CONVERGED", "seed_count": 1}])
    archetype_states = [
        {
            "archetype_id": f"A{archetype:02d}",
            "state_id": state,
            "archetype_state_sha256": f"{archetype:064x}"[-64:],
        }
        for archetype in range(1, 26)
        for state in STATES
    ]
    weather = []
    partition_specs = []
    for rcp_index, rcp in enumerate(RCPS):
        for weather_index in range(18):
            member_id = f"weather_{rcp}_{weather_index + 1:02d}"
            partition_id = f"stock_{rcp_index}_{weather_index:02d}"
            weather.append(
                {
                    "weather_member_id": member_id,
                    "member_id": member_id,
                    "climate_scenario_id": rcp,
                    "weather_pair_id": f"pvgis_{2006 + weather_index}",
                    "observed_pvgis_year": 2006 + weather_index,
                }
            )
            partition_specs.append(
                {
                    "partition_id": partition_id,
                    "weather_member_id": member_id,
                    "model_scenario_id": "central",
                }
            )
    seeds = [11]
    expected_runs = len(partition_specs) * len(archetype_states) * len(seeds)
    design = {
        "streaming_stock_contract_version": STREAMING_STOCK_CONTRACT_VERSION,
        "stock_partition_provenance_contract_version": (
            STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
        ),
        "model_contract_version": "test_model_v1",
        "central_thermal_assumptions_sha256": "1" * 64,
        "behaviour_assumptions_sha256": "2" * 64,
        "occupant_distribution_sha256": "3" * 64,
        "stock_weights_sha256": "4" * 64,
        "stock_weights_source_sha256": "5" * 64,
        "require_full_stock": True,
        "archetype_states": archetype_states,
        "weather_members": weather,
        "occupant_seeds": seeds,
        "occupant_seed_bank_sha256": ordered_seed_bank_sha256(tuple(seeds)),
        "convergence_evidence": {
            "status": "VERIFIED",
            "selected_seed_count": 1,
            "convergence_results_sha256": _sha(convergence_path),
        },
        "model_scenarios": [
            {"scenario_id": "central", "axis": "central", "description": "central"}
        ],
        "partition_specs": partition_specs,
        "expected_run_count": expected_runs,
    }
    design["design_sha256"] = canonical_sha256(design)
    design_path = root / "streaming_design_contract.json"
    _write_json(design_path, design)

    partition_index = []
    partition_input_ledger = []
    for spec in partition_specs:
        partition_dir = partitions / spec["partition_id"]
        partition_dir.mkdir()
        run_ids = [
            f"{spec['partition_id']}_{item['archetype_id']}_{item['state_id']}_11"
            for item in archetype_states
        ]
        _write_csv(partition_dir / "run_manifest.csv", [{"run_id": value} for value in run_ids])
        _write_csv(partition_dir / "run_diagnostics.csv", [{"run_id": value} for value in run_ids])
        _write_csv(partition_dir / "stock_aggregation.csv", [{"row": value} for value in range(4)])
        _write_csv(partition_dir / "stock_contributions.csv", [{"row": value} for value in range(16)])
        _write_csv(partition_dir / "stock_hourly.csv", [{"row": 1}])
        artifacts = {
            filename: _ledger(partition_dir / filename)
            for filename in (
                "run_manifest.csv",
                "run_diagnostics.csv",
                "stock_aggregation.csv",
                "stock_contributions.csv",
                "stock_hourly.csv",
            )
        }
        complete = {
            "status": "PASS",
            "stock_coverage_status": "AUTHORITATIVE_FULL_STOCK",
            "streaming_stock_contract_version": STREAMING_STOCK_CONTRACT_VERSION,
            "stock_partition_provenance_contract_version": (
                STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
            ),
            "design_sha256": design["design_sha256"],
            "partition_id": spec["partition_id"],
            "weather_member_id": spec["weather_member_id"],
            "model_scenario_id": "central",
            "expected_run_id_sha256": canonical_sha256(
                {"run_ids": sorted(run_ids)}
            ),
            "run_count": len(run_ids),
            "occupant_seeds": seeds,
            "artifacts": artifacts,
        }
        complete_path = partition_dir / "partition_complete.json"
        _write_json(complete_path, complete)
        partition_input_ledger.append(
            {
                "partition_id": spec["partition_id"],
                "run_count": len(run_ids),
                "partition_complete_sha256": _sha(complete_path),
                "run_manifest_sha256": artifacts["run_manifest.csv"]["sha256"],
                "run_diagnostics_sha256": artifacts["run_diagnostics.csv"]["sha256"],
            }
        )
        weather_record = next(
            item
            for item in weather
            if item["weather_member_id"] == spec["weather_member_id"]
        )
        partition_index.append(
            {
                "partition_id": spec["partition_id"],
                "weather_member_id": spec["weather_member_id"],
                "climate_scenario_id": weather_record["climate_scenario_id"],
                "model_scenario_id": "central",
                "run_count": len(run_ids),
                "run_diagnostics_path": str(
                    (partition_dir / "run_diagnostics.csv").relative_to(root)
                ),
                "run_diagnostics_sha256": artifacts["run_diagnostics.csv"]["sha256"],
                "stock_hourly_path": str(
                    (partition_dir / "stock_hourly.csv").relative_to(root)
                ),
                "stock_hourly_row_count": artifacts["stock_hourly.csv"]["row_count"],
                "stock_hourly_sha256": artifacts["stock_hourly.csv"]["sha256"],
                "partition_complete_sha256": _sha(complete_path),
            }
        )
    index_path = root / "partition_index.csv"
    _write_csv(index_path, partition_index)

    stock_records = []
    contribution_records = []
    dimension_shares = {
        "region": dict(zip(REGIONS, (0.58, 0.34, 0.08))),
        "dwelling_type": {
            "Detached house": 0.30,
            "Semi-detached house": 0.22,
            "Terraced house": 0.20,
            "Apartment, enclosed": 0.12,
            "Apartment, exposed": 0.16,
        },
        "construction_period": {
            "pre-1946": 0.28,
            "1946-1970": 0.25,
            "1971-1990": 0.22,
            "1991-2005": 0.15,
            "post-2005": 0.10,
        },
        "state_id": dict(zip(STATES, (0.55, 0.30, 0.15))),
    }
    for rcp_index, rcp in enumerate(RCPS):
        for weather_index in range(18):
            member_id = f"weather_{rcp}_{weather_index + 1:02d}"
            national_heat = 80_000.0 + rcp_index * 5_000.0 + weather_index * 300.0
            national_cool = 7_000.0 + rcp_index * 2_000.0 + weather_index * 120.0
            national_peak = 25_000.0 + rcp_index * 1_200.0 + weather_index * 90.0
            national_cool_peak = 5_000.0 + rcp_index * 800.0 + weather_index * 50.0
            for region, share in (*zip(REGIONS, (0.58, 0.34, 0.08)), (BELGIUM_REGION_ID, 1.0)):
                stock_records.append(
                    {
                        "climate_scenario_id": rcp,
                        "weather_member_id": member_id,
                        "model_scenario_id": "central",
                        "model_scenario_axis": "central",
                        "stock_scenario_id": "central",
                        "target_year": 2050,
                        "region": region,
                        "modelled_dwellings": 5_537_385.0 if share == 1.0 else 5_537_385.0 * share,
                        "annual_heating_GWh": national_heat * share,
                        "annual_potential_sensible_cooling_GWh": national_cool * share,
                        "coincident_peak_heating_MW": national_peak * share,
                        "coincident_peak_potential_cooling_MW": national_cool_peak * share,
                        "stock_coverage": "R1-R4 modelled stock; R5-R6 residual excluded",
                    }
                )
            for dimension, shares in dimension_shares.items():
                for value, share in shares.items():
                    contribution_records.append(
                        {
                            "climate_scenario_id": rcp,
                            "weather_member_id": member_id,
                            "model_scenario_id": "central",
                            "contribution_dimension": dimension,
                            "contribution_value": value,
                            "annual_heating_GWh": national_heat * share,
                            "annual_potential_sensible_cooling_GWh": national_cool * share,
                            "share_of_stock_heating": share,
                            "share_of_stock_potential_cooling": share,
                        }
                    )
    stock = pd.DataFrame.from_records(stock_records)
    stock_path = root / "stock_aggregation.csv"
    stock.to_csv(stock_path, index=False, float_format="%.17g", lineterminator="\n")
    contributions_path = root / "stock_contributions.csv"
    _write_csv(contributions_path, contribution_records)
    distributions_path = root / "stock_distribution_summary.csv"
    _write_csv(distributions_path, _distribution_records(stock))
    root_artifacts = {
        path.name: _sha(path)
        for path in (
            index_path,
            stock_path,
            contributions_path,
            distributions_path,
            convergence_path,
        )
    }
    summary = {
        "status": "PASS",
        "scope": "complete authoritative bounded-memory streaming stock design",
        "stock_coverage_status": "AUTHORITATIVE_FULL_STOCK",
        "require_full_stock": True,
        "streaming_stock_contract_version": STREAMING_STOCK_CONTRACT_VERSION,
        "stock_partition_provenance_contract_version": (
            STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
        ),
        "design_sha256": design["design_sha256"],
        "partition_count": 54,
        "completed_run_count": expected_runs,
        "expected_run_count": expected_runs,
        "occupant_seeds": seeds,
        "occupant_seed_bank_sha256": design["occupant_seed_bank_sha256"],
        "stock_weights_sha256": design["stock_weights_sha256"],
        "stock_weights_source_sha256": design["stock_weights_source_sha256"],
        "interval_interpretation": (
            "descriptive empirical intervals; not complete prediction intervals"
        ),
        "convergence_evidence": {
            **design["convergence_evidence"],
            "persisted_path": "convergence_results.csv",
            "persisted_sha256": root_artifacts["convergence_results.csv"],
        },
        "artifact_sha256": root_artifacts,
    }
    summary_path = root / "monte_carlo_summary.json"
    _write_json(summary_path, summary)

    variance_records = []
    for archetype in ("A01", "A02"):
        for state in STATES:
            for rcp in RCPS:
                for component, ss in (
                    ("weather_year", 70.0),
                    ("occupant_seed", 20.0),
                    ("weather_seed_interaction", 10.0),
                ):
                    variance_records.append(
                        {
                            "archetype_id": archetype,
                            "state_id": state,
                            "climate_scenario_id": rcp,
                            "model_scenario_id": "central",
                            "metric": "heating_intensity_kWh_m2",
                            "component": component,
                            "weather_member_count": 18,
                            "occupant_seed_count": 1,
                            "sum_of_squares": ss,
                            "total_sum_of_squares": 100.0,
                            "sum_of_squares_share": ss / 100.0,
                            "interpretation": "balanced ANOVA share within included empirical ensemble",
                        }
                    )
    paired_records = []
    for archetype_index, archetype in enumerate(("A01", "A02")):
        for rcp_index, rcp in enumerate(RCPS):
            for weather_index in range(2):
                baseline = 150.0 + 5.0 * archetype_index + rcp_index + weather_index
                for state, fraction in zip(STATES[1:], (0.65, 0.38)):
                    comparison = baseline * fraction
                    paired_records.append(
                        {
                            "archetype_id": archetype,
                            "climate_scenario_id": rcp,
                            "weather_member_id": f"weather_{rcp}_{weather_index + 1:02d}",
                            "occupant_seed": 11,
                            "model_scenario_id": "central",
                            "baseline_state_id": STATES[0],
                            "comparison_state_id": state,
                            "metric": "heating_intensity_kWh_m2",
                            "baseline_value": baseline,
                            "comparison_value": comparison,
                            "delta": comparison - baseline,
                        }
                    )
    postprocess_paths = {
        "distribution_summary.csv": [{"metric": "heating_intensity_kWh_m2", "median": 100.0}],
        "variance_contributions.csv": variance_records,
        "paired_renovation_deltas.csv": paired_records,
        "stock_weighted_distribution_summary.csv": [{"metric": "heating_intensity_kWh_m2", "median": 100.0}],
    }
    post_ledger = {}
    for filename, records in postprocess_paths.items():
        path = root / filename
        _write_csv(path, records)
        post_ledger[filename] = _ledger(path)
    post_summary = {
        "status": "PASS",
        "postprocess_contract_version": POSTPROCESS_CONTRACT_VERSION,
        "source_execution_status": "PASS",
        "source_design_sha256": design["design_sha256"],
        "source_design_file_sha256": _sha(design_path),
        "source_monte_carlo_summary_sha256": _sha(summary_path),
        "source_partition_index_sha256": _sha(index_path),
        "source_partition_count": 54,
        "dwelling_hour_files_read": 0,
        "source_partition_input_ledger_sha256": canonical_sha256(
            partition_input_ledger
        ),
        "processed_run_count": expected_runs,
        "expected_run_count": expected_runs,
        "stock_weights_sha256": design["stock_weights_sha256"],
        "stock_weights_source_sha256": design["stock_weights_source_sha256"],
        "interval_interpretation": (
            "empirical intervals over included axes; not complete prediction intervals"
        ),
        "output_artifacts": post_ledger,
    }
    _write_json(root / POSTPROCESS_SUMMARY_FILENAME, post_summary)


def test_authenticated_report_writes_nine_independent_png_pdf_figures(
    tmp_path: Path, capsys
) -> None:
    production = tmp_path / "production"
    figures = tmp_path / "figures"
    report = tmp_path / "report" / "RESULTS.md"
    _completed_reporting_fixture(production)

    inputs = authenticate_reporting_inputs(production)
    assert inputs.stock_summary["status"] == "PASS"
    assert len(inputs.source_artifacts) == 54 * 6 + 12

    summary = generate_production_report(
        production, figure_dir=figures, report_path=report
    )
    assert summary["status"] == "PASS"
    assert summary["figure_count"] == 9
    figure_provenance = json.loads(
        (figures / FIGURE_PROVENANCE_FILENAME).read_text(encoding="utf-8")
    )
    assert figure_provenance["source_artifacts"] == inputs.source_artifacts
    assert len(figure_provenance["figure_artifacts"]) == 18
    assert set(FIGURE_BASENAMES) == {
        path.stem for path in figures.glob("*.png")
    }
    for basename in FIGURE_BASENAMES:
        png = figures / f"{basename}.png"
        pdf = figures / f"{basename}.pdf"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert pdf.read_bytes().startswith(b"%PDF-")
        with Image.open(png) as image:
            assert image.width >= 1800
            assert image.height >= 1100
    text = report.read_text(encoding="utf-8")
    assert "ideal useful space-heating demand" in text
    assert "R1–R4 only" in text
    assert "same Brussels-area weather member" in text
    assert "not complete prediction intervals" in text
    assert "Potential sensible cooling" in text
    assert reporting_status(production)["status"] == "PASS"
    repeated = generate_production_report(
        production, figure_dir=figures, report_path=report
    )
    assert repeated == summary
    assert (
        monte_carlo_main(
            [
                "report",
                "--output-dir",
                str(production),
                "--figure-dir",
                str(figures),
                "--report-path",
                str(report),
            ]
        )
        == 0
    )
    cli = json.loads(capsys.readouterr().out)
    assert cli["status"] == "PASS"
    assert cli["figure_count"] == 9
    assert cli["report_path"] == str(report.resolve())


def test_reporting_rejects_tampered_source_and_status_detects_output_tamper(
    tmp_path: Path,
) -> None:
    production = tmp_path / "production"
    figures = tmp_path / "figures"
    _completed_reporting_fixture(production)
    generate_production_report(production, figure_dir=figures)

    output = figures / "mc_national_annual_heating.png"
    output.write_bytes(output.read_bytes() + b"tampered")
    status = reporting_status(production)
    assert status["status"] == "INVALID"
    assert "checksum mismatch" in status["reason"]

    source = production / "stock_contributions.csv"
    source.write_text(source.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(MonteCarloContractError, match="checksum mismatch"):
        authenticate_reporting_inputs(production)


def test_reporting_rejects_nonproduction_status_and_cli_status_is_read_only(
    tmp_path: Path, capsys
) -> None:
    production = tmp_path / "production"
    _completed_reporting_fixture(production)
    summary_path = production / "monte_carlo_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "WORKFLOW_CHECK_ONLY"
    _write_json(summary_path, summary)
    post_path = production / POSTPROCESS_SUMMARY_FILENAME
    post = json.loads(post_path.read_text(encoding="utf-8"))
    post["source_monte_carlo_summary_sha256"] = _sha(summary_path)
    _write_json(post_path, post)
    with pytest.raises(MonteCarloContractError, match="production PASS"):
        authenticate_reporting_inputs(production)

    empty = tmp_path / "empty"
    assert monte_carlo_main(["report", "--output-dir", str(empty), "--status"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "NOT_RUN"


def test_reporting_status_detects_summary_tamper(tmp_path: Path) -> None:
    production = tmp_path / "production"
    _completed_reporting_fixture(production)
    generate_production_report(production, figure_dir=tmp_path / "figures")
    path = production / REPORTING_SUMMARY_FILENAME
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["figure_count"] = 8
    _write_json(path, summary)
    status = reporting_status(production)
    assert status["status"] == "INVALID"
    assert "figure count" in status["reason"]


@pytest.mark.parametrize(
    "relative_path",
    (
        Path("partitions/stock_0_00/stock_hourly.csv"),
        Path("variance_contributions.csv"),
    ),
)
def test_reporting_authentication_rejects_partition_or_postprocess_tamper(
    tmp_path: Path, relative_path: Path
) -> None:
    production = tmp_path / "production"
    _completed_reporting_fixture(production)
    target = production / relative_path
    target.write_text(target.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(MonteCarloContractError, match="checksum mismatch"):
        authenticate_reporting_inputs(production)


def test_reporting_recomputes_postprocess_partition_input_ledger(tmp_path: Path) -> None:
    production = tmp_path / "production"
    _completed_reporting_fixture(production)
    path = production / POSTPROCESS_SUMMARY_FILENAME
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["source_partition_input_ledger_sha256"] = "0" * 64
    _write_json(path, summary)
    with pytest.raises(MonteCarloContractError, match="Post-processing summary"):
        authenticate_reporting_inputs(production)
