from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from thermal_model.monte_carlo import convergence_extension as extension
from thermal_model.monte_carlo.contracts import (
    MonteCarloContractError,
    canonical_sha256,
)
from thermal_model.monte_carlo.convergence_runner import (
    _frame_csv_sha256,
    _read_json,
    _sha256_file,
)
from thermal_model.monte_carlo.design import (
    ConvergenceRule,
    PROSPECTIVE_N160_CONVERGENCE_RULE,
    make_seed_bank,
    ordered_seed_bank_sha256,
)


BASE_OUTPUT = extension.DEFAULT_CONVERGENCE_OUTPUT_DIR


def _source_snapshot(paths: list[Path]) -> dict[str, tuple[int, int, str]]:
    return {
        str(path.relative_to(BASE_OUTPUT)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            _sha256_file(path),
        )
        for path in paths
    }


def _source_root_paths() -> list[Path]:
    return [
        BASE_OUTPUT / "convergence_execution_contract.json",
        BASE_OUTPUT / "convergence_summary.json",
        BASE_OUTPUT / "run_manifest.csv",
        BASE_OUTPUT / "run_diagnostics.csv",
        BASE_OUTPUT / "convergence_results.csv",
        BASE_OUTPUT / "checkpoints/n080/checkpoint_summary.json",
    ]


def test_prepare_authenticates_real_base_and_does_not_mutate_it(
    tmp_path: Path,
) -> None:
    before = _source_snapshot(_source_root_paths())

    contract = extension.prepare_convergence_extension(tmp_path)

    assert _source_snapshot(_source_root_paths()) == before
    unsigned = {
        key: value for key, value in contract.items() if key != "design_sha256"
    }
    assert contract["design_sha256"] == canonical_sha256(unsigned)
    assert contract["convergence_extension_contract_version"] == (
        extension.CONVERGENCE_EXTENSION_CONTRACT_VERSION
    )
    assert contract["source_mutation_permitted"] is False
    assert contract["intermediate_stopping_permitted"] is False
    assert contract["expected_imported_run_count"] == 3 * 54 * 80
    assert contract["expected_new_run_count"] == 3 * 54 * 80
    assert contract["expected_total_run_count_at_n160"] == 3 * 54 * 160

    receipt = contract["base_experiment"]
    source_contract = _read_json(
        BASE_OUTPUT / "convergence_execution_contract.json"
    )
    assert receipt["base_terminal_status"] == "NOT_CONVERGED_AT_N80"
    assert receipt["base_design_sha256"] == source_contract["design_sha256"]
    assert receipt["base_contract_file_sha256"] == _sha256_file(
        BASE_OUTPUT / "convergence_execution_contract.json"
    )
    assert receipt["base_n080_all_criteria_pass"] is True
    assert receipt["base_n080_panel_consecutive_passing_expansions"] == 1
    assert receipt["base_artifact_sha256"] == {
        name: _sha256_file(BASE_OUTPUT / name)
        for name in (
            "run_manifest.csv",
            "run_diagnostics.csv",
            "convergence_results.csv",
        )
    }
    assert extension.prepare_convergence_extension(tmp_path) == contract


def test_prepare_rejects_source_destination_tree_overlap() -> None:
    nested_destination = BASE_OUTPUT / "forbidden_extension_child"

    with pytest.raises(MonteCarloContractError, match="non-overlapping"):
        extension.prepare_convergence_extension(nested_destination)

    assert not nested_destination.exists()


def test_prospective_rule_and_seed_bank_are_exact_nested_extensions(
    tmp_path: Path,
) -> None:
    contract = extension.prepare_convergence_extension(tmp_path)
    original = asdict(ConvergenceRule())
    prospective = asdict(PROSPECTIVE_N160_CONVERGENCE_RULE)
    original_checkpoints = tuple(original.pop("checkpoints"))
    prospective_checkpoints = tuple(prospective.pop("checkpoints"))

    assert prospective == original
    assert prospective_checkpoints == (*original_checkpoints, 160)
    seeds = tuple(contract["occupant_seeds"])
    source_seeds = tuple(
        _read_json(BASE_OUTPUT / "convergence_execution_contract.json")[
            "occupant_seeds"
        ]
    )
    assert seeds == make_seed_bank(160, master_seed=extension.MASTER_SEED)
    assert seeds[:80] == source_seeds
    assert contract["imported_seed_prefix_sha256"] == ordered_seed_bank_sha256(
        source_seeds
    )
    assert contract["occupant_seed_bank_sha256"] == ordered_seed_bank_sha256(
        seeds
    )


def test_one_partition_import_is_authenticated_and_exactly_restartable(
    tmp_path: Path,
) -> None:
    contract = extension.prepare_convergence_extension(tmp_path)
    member_id = str(contract["weather_members"][0]["member_id"])
    seed_bank = tuple(int(value) for value in contract["occupant_seeds"])
    source_partition = BASE_OUTPUT / "partitions" / member_id
    source_progress = _read_json(source_partition / "progress.json")
    source_active = source_partition / str(
        source_progress["active_diagnostics_slot"]
    )
    source_before = _source_snapshot(
        [
            source_partition / "partition_contract.json",
            source_partition / "run_manifest.csv",
            source_partition / "progress.json",
            source_active,
        ]
    )

    first = extension._bootstrap_extension_partition(
        member_id,
        seed_bank,
        str(tmp_path),
        str(contract["design_sha256"]),
    )
    partition = tmp_path / "partitions" / member_id
    first_progress_sha = _sha256_file(partition / "progress.json")
    first_receipt_sha = _sha256_file(partition / "import_receipt.json")
    progress = _read_json(partition / "progress.json")
    first_active_sha = _sha256_file(
        partition / str(progress["active_diagnostics_slot"])
    )

    second = extension._bootstrap_extension_partition(
        member_id,
        seed_bank,
        str(tmp_path),
        str(contract["design_sha256"]),
    )

    assert first == second == {
        "weather_member_id": member_id,
        "completed_seed_count": 80,
        "imported_run_count": 240,
    }
    assert _sha256_file(partition / "progress.json") == first_progress_sha
    assert _sha256_file(partition / "import_receipt.json") == first_receipt_sha
    assert _sha256_file(
        partition / str(progress["active_diagnostics_slot"])
    ) == first_active_sha
    assert _read_json(partition / "import_receipt.json")["status"] == (
        "IMPORTED_AND_VERIFIED"
    )
    assert _source_snapshot(
        [
            source_partition / "partition_contract.json",
            source_partition / "run_manifest.csv",
            source_partition / "progress.json",
            source_active,
        ]
    ) == source_before


def test_partition_import_rejects_tampered_source_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "extension"
    contract = extension.prepare_convergence_extension(destination)
    member_id = str(contract["weather_members"][0]["member_id"])
    seed_bank = tuple(int(value) for value in contract["occupant_seeds"])
    fake_base = tmp_path / "fake_base"
    fake_partition = fake_base / "partitions" / member_id
    shutil.copytree(BASE_OUTPUT / "partitions" / member_id, fake_partition)
    progress = _read_json(fake_partition / "progress.json")
    active = fake_partition / str(progress["active_diagnostics_slot"])
    with active.open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    monkeypatch.setattr(
        extension, "_base_dir_from_contract", lambda contract: fake_base
    )

    with pytest.raises(MonteCarloContractError, match="checksum mismatch"):
        extension._bootstrap_extension_partition(
            member_id,
            seed_bank,
            str(destination),
            str(contract["design_sha256"]),
        )


def test_historical_decision_tampering_is_rejected() -> None:
    historical = pd.read_csv(
        BASE_OUTPUT / "convergence_results.csv", float_precision="round_trip"
    )
    extension._verify_historical_decisions(historical.copy(), BASE_OUTPUT)
    tampered = historical.copy()
    tampered.loc[tampered.index[0], "value"] = (
        float(tampered.loc[tampered.index[0], "value"]) + 0.01
    )

    with pytest.raises(MonteCarloContractError, match="retroactively changed"):
        extension._verify_historical_decisions(tampered, BASE_OUTPUT)


def test_terminal_selection_loader_is_contract_rule_and_checksum_bound(
    tmp_path: Path,
) -> None:
    contract = extension.prepare_convergence_extension(tmp_path)
    evidence = tmp_path / "convergence_results.csv"
    evidence.write_text(
        "seed_count,panel_converged_at_checkpoint\n160,true\n",
        encoding="utf-8",
    )
    (tmp_path / "run_manifest.csv").write_text("run_id\nrun-a\n", encoding="utf-8")
    (tmp_path / "run_diagnostics.csv").write_text(
        "run_id,occupant_seed\nrun-a,1\n", encoding="utf-8"
    )
    artifact_sha256 = {
        name: _sha256_file(tmp_path / name)
        for name in extension.FINAL_EXTENSION_ARTIFACTS
    }
    evidence_sha = artifact_sha256["convergence_results.csv"]
    summary = {
        "status": "CONVERGED",
        "design_sha256": contract["design_sha256"],
        "selected_seed_count": 160,
        "selected_occupant_seeds": contract["occupant_seeds"],
        "first_panel_converged_checkpoint": 160,
        "evaluated_seed_count": 160,
        "convergence_rule": contract["convergence_rule"],
        "artifact_sha256": artifact_sha256,
    }
    extension._atomic_json(
        summary, tmp_path / "convergence_extension_summary.json"
    )

    selection = extension.load_convergence_extension_selection(tmp_path)
    assert selection.occupant_seeds == tuple(contract["occupant_seeds"])
    assert selection.convergence_results_path == evidence.resolve()
    assert selection.convergence_results_sha256 == evidence_sha
    assert selection.design_sha256 == contract["design_sha256"]
    assert selection.convergence_rule == PROSPECTIVE_N160_CONVERGENCE_RULE

    evidence.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(MonteCarloContractError, match="artifact changed"):
        extension.load_convergence_extension_selection(tmp_path)


def test_terminal_selection_loader_normalizes_malformed_summary_error(
    tmp_path: Path,
) -> None:
    contract = extension.prepare_convergence_extension(tmp_path)
    extension._atomic_json(
        {
            "status": "CONVERGED",
            "design_sha256": contract["design_sha256"],
            "convergence_rule": contract["convergence_rule"],
        },
        tmp_path / "convergence_extension_summary.json",
    )

    with pytest.raises(MonteCarloContractError, match="exact three-artifact ledger"):
        extension.load_convergence_extension_selection(tmp_path)


def test_status_labels_obsolete_extension_contract_as_stale(
    tmp_path: Path,
) -> None:
    extension._atomic_json(
        {
            "convergence_extension_contract_version": "obsolete-v0",
            "design_sha256": "a" * 64,
        },
        tmp_path / "convergence_extension_contract.json",
    )

    status = extension.convergence_extension_status(tmp_path)

    assert status == {
        "status": "STALE_CONTRACT",
        "output_dir": str(tmp_path.resolve()),
        "persisted_contract_version": "obsolete-v0",
        "required_contract_version": (
            extension.CONVERGENCE_EXTENSION_CONTRACT_VERSION
        ),
        "design_sha256": "a" * 64,
    }


@pytest.mark.parametrize(
    ("converged", "expected_status"),
    [(True, "CONVERGED"), (False, "NOT_CONVERGED_AT_N160")],
)
def test_mocked_terminal_n160_promotion_and_nonpromotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    converged: bool,
    expected_status: str,
) -> None:
    seeds = make_seed_bank(160, master_seed=extension.MASTER_SEED)
    contract = {
        "design_sha256": "d" * 64,
        "occupant_seeds": list(seeds),
        "base_experiment": {"base_design_sha256": "b" * 64},
    }
    weather = pd.DataFrame({"member_id": ["weather-a", "weather-b"]})
    calls: list[str] = []
    monkeypatch.setattr(
        extension,
        "prepare_convergence_extension",
        lambda destination, base_output_dir: contract,
    )
    monkeypatch.setattr(extension, "_weather_selection", lambda: weather.copy())
    monkeypatch.setattr(
        extension,
        "_bootstrap_all_partitions",
        lambda *args: calls.append("bootstrap"),
    )
    monkeypatch.setattr(
        extension,
        "_verify_imported_aggregate",
        lambda *args: calls.append("verify-import"),
    )
    monkeypatch.setattr(
        extension,
        "_advance_all_extension_partitions",
        lambda *args: calls.append("advance"),
    )
    monkeypatch.setattr(
        extension,
        "_verify_historical_decisions",
        lambda *args: calls.append("verify-history"),
    )
    manifest = pd.DataFrame({"run_id": ["run-a"]})
    diagnostics = pd.DataFrame(
        {"run_id": ["run-a"], "occupant_seed": [seeds[-1]]}
    )
    convergence = pd.DataFrame(
        {
            "seed_count": [160],
            "panel_converged_at_checkpoint": [converged],
        }
    )
    monkeypatch.setattr(
        extension,
        "_collect_extension_checkpoint",
        lambda *args: (manifest, diagnostics, convergence, converged),
    )

    summary = extension._run_convergence_extension_unlocked(
        tmp_path, base_output_dir=BASE_OUTPUT, max_workers=1
    )

    assert calls == ["bootstrap", "verify-import", "advance", "verify-history"]
    assert summary["status"] == expected_status
    assert summary["evaluated_seed_count"] == 160
    assert summary["selected_seed_count"] == (160 if converged else None)
    assert summary["selected_occupant_seeds"] == (
        list(seeds) if converged else None
    )
    assert summary["first_panel_converged_checkpoint"] == (
        160 if converged else None
    )
    assert (
        _read_json(tmp_path / "checkpoints/n160/checkpoint_summary.json")[
            "status"
        ]
        == ("CONVERGED" if converged else "NOT_YET_CONVERGED")
    )
    for filename, digest in summary["artifact_sha256"].items():
        assert _sha256_file(tmp_path / filename) == digest
