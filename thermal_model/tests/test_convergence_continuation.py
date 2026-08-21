from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil

import pandas as pd
import pytest

from thermal_model.monte_carlo import convergence_continuation as continuation
from thermal_model.monte_carlo.contracts import (
    MonteCarloContractError,
    canonical_sha256,
)
from thermal_model.monte_carlo.convergence_runner import (
    _read_json,
    _sha256_file,
)
from thermal_model.monte_carlo.design import (
    PROSPECTIVE_N160_CONVERGENCE_RULE,
    PROSPECTIVE_N320_N640_CONVERGENCE_RULE,
    make_seed_bank,
    ordered_seed_bank_sha256,
)


SOURCE_OUTPUT = continuation.DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, int, str]]:
    return {
        str(path.relative_to(SOURCE_OUTPUT)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            _sha256_file(path),
        )
        for path in paths
    }


def _source_root_paths() -> list[Path]:
    return [
        SOURCE_OUTPUT / "convergence_extension_contract.json",
        SOURCE_OUTPUT / "convergence_extension_summary.json",
        SOURCE_OUTPUT / "run_manifest.csv",
        SOURCE_OUTPUT / "run_diagnostics.csv",
        SOURCE_OUTPUT / "convergence_results.csv",
        SOURCE_OUTPUT / "checkpoints/n160/checkpoint_summary.json",
    ]


def test_prepare_authenticates_real_n160_source_without_mutating_it(
    tmp_path: Path,
) -> None:
    before = _snapshot(_source_root_paths())

    contract = continuation.prepare_convergence_continuation(tmp_path)

    assert _snapshot(_source_root_paths()) == before
    unsigned = {
        key: value for key, value in contract.items() if key != "design_sha256"
    }
    assert contract["design_sha256"] == canonical_sha256(unsigned)
    assert contract["source_mutation_permitted"] is False
    assert contract["intermediate_stopping_permitted"] is False
    assert contract["n320_selection_permitted"] is False
    assert contract["selection_requires_both_new_checkpoints_pass"] is True
    assert contract["imported_seed_count"] == 160
    assert contract["new_seed_count"] == 480
    assert contract["continuation_checkpoints"] == [320, 640]
    assert contract["continuation_seed_prefix_sha256"] == {
        "n320": "0bce047ba8fc8c0eb0a2531effd23d3260ff97ef07883f64292513091b921f4e",
        "n640": "658d6245c4af6148e863c80d498be1748cdc6e9ed181c0c34c2144e9dba61430",
    }
    assert contract["expected_imported_run_count"] == 3 * 54 * 160
    assert contract["expected_new_run_count_161_to_320"] == 3 * 54 * 160
    assert contract["expected_new_run_count_321_to_640"] == 3 * 54 * 320
    assert contract["expected_new_run_count"] == 3 * 54 * 480
    assert contract["expected_total_run_count_at_n320"] == 3 * 54 * 320
    assert contract["expected_total_run_count_at_n640"] == 3 * 54 * 640
    assert contract["source_experiment"]["source_terminal_status"] == (
        "NOT_CONVERGED_AT_N160"
    )
    assert contract["source_experiment"]["source_n160_all_criteria_pass"] is False
    assert continuation.prepare_convergence_continuation(tmp_path) == contract


def test_rule_appends_both_checkpoints_without_changing_any_criterion() -> None:
    source = asdict(PROSPECTIVE_N160_CONVERGENCE_RULE)
    prospective = asdict(PROSPECTIVE_N320_N640_CONVERGENCE_RULE)
    source_checkpoints = tuple(source.pop("checkpoints"))
    prospective_checkpoints = tuple(prospective.pop("checkpoints"))

    assert prospective == source
    assert prospective_checkpoints == (*source_checkpoints, 320, 640)
    assert prospective["relative_tolerance"] == 0.02
    assert prospective["required_consecutive_expansions"] == 2


def test_prepare_rejects_overlap_with_both_immutable_predecessor_trees() -> None:
    source_contract = _read_json(
        SOURCE_OUTPUT / "convergence_extension_contract.json"
    )
    original_base = continuation.PROJECT_ROOT / source_contract["base_experiment"][
        "base_output_relative_path"
    ]

    with pytest.raises(MonteCarloContractError, match="non-overlapping"):
        continuation.prepare_convergence_continuation(SOURCE_OUTPUT)
    with pytest.raises(MonteCarloContractError, match="immutable original n=80"):
        continuation.prepare_convergence_continuation(original_base)


def test_one_partition_import_is_exact_restartable_and_source_immutable(
    tmp_path: Path,
) -> None:
    contract = continuation.prepare_convergence_continuation(tmp_path)
    member_id = str(contract["weather_members"][0]["member_id"])
    seed_bank = tuple(int(value) for value in contract["occupant_seeds"])
    source_partition = SOURCE_OUTPUT / "partitions" / member_id
    source_progress = _read_json(source_partition / "progress.json")
    source_paths = [
        source_partition / "partition_contract.json",
        source_partition / "run_manifest.csv",
        source_partition / "progress.json",
        source_partition / str(source_progress["active_diagnostics_slot"]),
    ]
    before = _snapshot(source_paths)

    first = continuation._bootstrap_continuation_partition(
        member_id,
        seed_bank,
        str(tmp_path),
        str(contract["design_sha256"]),
    )
    partition = tmp_path / "partitions" / member_id
    progress_sha = _sha256_file(partition / "progress.json")
    receipt_sha = _sha256_file(partition / "import_receipt.json")
    progress = _read_json(partition / "progress.json")
    active_sha = _sha256_file(
        partition / str(progress["active_diagnostics_slot"])
    )
    second = continuation._bootstrap_continuation_partition(
        member_id,
        seed_bank,
        str(tmp_path),
        str(contract["design_sha256"]),
    )

    assert first == second == {
        "weather_member_id": member_id,
        "completed_seed_count": 160,
        "imported_run_count": 480,
    }
    assert _sha256_file(partition / "progress.json") == progress_sha
    assert _sha256_file(partition / "import_receipt.json") == receipt_sha
    assert _sha256_file(
        partition / str(progress["active_diagnostics_slot"])
    ) == active_sha
    assert _snapshot(source_paths) == before


def test_partition_execution_starts_at_seed_161_and_restart_skips_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = continuation.prepare_convergence_continuation(tmp_path)
    member_id = str(contract["weather_members"][0]["member_id"])
    seed_bank = tuple(int(value) for value in contract["occupant_seeds"])
    continuation._bootstrap_continuation_partition(
        member_id,
        seed_bank,
        str(tmp_path),
        str(contract["design_sha256"]),
    )
    full_manifest = pd.read_csv(
        tmp_path / "partitions" / member_id / "run_manifest.csv",
        float_precision="round_trip",
    )
    calls: list[int] = []

    def fake_execute(states, members, seeds, scenarios):
        seed = int(tuple(seeds)[0])
        calls.append(seed)
        selected = full_manifest.loc[
            full_manifest["occupant_seed"] == seed
        ].copy()
        diagnostics = selected[["run_id", "occupant_seed"]].copy()
        return selected, diagnostics, {}

    monkeypatch.setattr(continuation, "execute_balanced_design", fake_execute)
    first = continuation._advance_continuation_partition(
        member_id,
        161,
        seed_bank,
        str(tmp_path),
        str(contract["design_sha256"]),
    )
    second = continuation._advance_continuation_partition(
        member_id,
        161,
        seed_bank,
        str(tmp_path),
        str(contract["design_sha256"]),
    )

    assert calls == [seed_bank[160]]
    assert first["completed_seed_count"] == 161
    assert second["completed_seed_count"] == 161


def test_n320_collection_extracts_exact_prefix_from_partitions_already_beyond_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = make_seed_bank(640, master_seed=continuation.MASTER_SEED)
    member_id = "weather-a"
    records = [
        {
            "run_id": f"run-{rank:03d}-{cell}",
            "occupant_seed": seed,
            "occupant_seed_rank": rank,
        }
        for rank, seed in enumerate(seeds, start=1)
        for cell in range(3)
    ]
    manifest = pd.DataFrame.from_records(records)
    diagnostics = manifest[["run_id", "occupant_seed"]].copy()
    partition = tmp_path / "partitions" / member_id
    continuation._atomic_csv(manifest, partition / "run_manifest.csv")
    monkeypatch.setattr(
        continuation,
        "_restore_partition_diagnostics",
        lambda *args, **kwargs: (diagnostics.copy(), 640, "run_diagnostics.slot_a.csv"),
    )

    def fake_evaluate(selected, *, seed_order, rule):
        assert len(selected) == 3 * 320
        assert tuple(seed_order) == seeds[:320]
        assert set(selected["occupant_seed"]) == set(seeds[:320])
        return pd.DataFrame(
            {
                "seed_count": [320],
                "criterion_pass": [True],
                "panel_converged_at_checkpoint": [False],
            }
        )

    monkeypatch.setattr(continuation, "evaluate_seed_convergence", fake_evaluate)
    collected_manifest, collected_diagnostics, _, panel_pass, converged = (
        continuation._collect_continuation_checkpoint(
            tmp_path,
            pd.DataFrame({"member_id": [member_id]}),
            seeds,
            "d" * 64,
            320,
        )
    )

    assert len(collected_manifest) == len(collected_diagnostics) == 3 * 320
    assert collected_manifest["occupant_seed_rank"].max() == 320
    assert set(collected_diagnostics["occupant_seed"]) == set(seeds[:320])
    assert panel_pass is True
    assert converged is False


def test_source_partition_requires_exact_n160_protocol_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "continuation"
    contract = continuation.prepare_convergence_continuation(destination)
    member_id = str(contract["weather_members"][0]["member_id"])
    seed_bank = tuple(int(value) for value in contract["occupant_seeds"])
    fake_source = tmp_path / "fake_source"
    fake_partition = fake_source / "partitions" / member_id
    shutil.copytree(SOURCE_OUTPUT / "partitions" / member_id, fake_partition)
    source_contract_path = fake_partition / "partition_contract.json"
    source_contract = _read_json(source_contract_path)
    continuation._atomic_json(
        {
            **source_contract,
            "convergence_extension_partition_version": "wrong-protocol-v0",
        },
        source_contract_path,
    )
    monkeypatch.setattr(
        continuation, "_source_dir_from_contract", lambda prepared: fake_source
    )

    with pytest.raises(MonteCarloContractError, match="source partition contract"):
        continuation._bootstrap_continuation_partition(
            member_id,
            seed_bank,
            str(destination),
            str(contract["design_sha256"]),
        )


@pytest.mark.parametrize("tamper", ["diagnostic_seed", "manifest_rank"])
def test_checkpoint_authentication_rejects_run_seed_or_rank_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    seeds = make_seed_bank(640, master_seed=continuation.MASTER_SEED)
    checkpoint = tmp_path / "checkpoints/n320"
    manifest = pd.DataFrame(
        {
            "run_id": [f"run-{rank:03d}" for rank in range(1, 321)],
            "occupant_seed": list(seeds[:320]),
            "occupant_seed_rank": list(range(1, 321)),
        }
    )
    diagnostics = manifest[["run_id", "occupant_seed"]].copy()
    if tamper == "diagnostic_seed":
        diagnostics.loc[0, "occupant_seed"] = seeds[1]
    else:
        manifest.loc[0, "occupant_seed_rank"] = 2
    evidence = pd.DataFrame({"seed_count": [320]})
    for filename, frame in {
        "run_manifest.csv": manifest,
        "run_diagnostics.csv": diagnostics,
        "convergence_results.csv": evidence,
    }.items():
        continuation._atomic_csv(frame, checkpoint / filename)
    continuation._atomic_json(
        {
            "artifact_sha256": {
                filename: _sha256_file(checkpoint / filename)
                for filename in continuation.FINAL_CONTINUATION_ARTIFACTS
            }
        },
        checkpoint / "checkpoint_summary.json",
    )
    contract = {
        "panel": [{}],
        "weather_members": [{}],
        "occupant_seeds": list(seeds),
    }

    with pytest.raises(
        MonteCarloContractError,
        match="run-ID occupant-seed identity|occupant-seed rank",
    ):
        continuation._authenticate_checkpoint(
            tmp_path, contract, 320, expected_authorized=False
        )


@pytest.mark.parametrize(
    ("n320_pass", "n640_pass", "expected_status"),
    [
        (True, True, "CONVERGED"),
        (True, False, "NOT_CONVERGED_AT_N640"),
        (False, True, "NOT_CONVERGED_AT_N640"),
        (False, False, "NOT_CONVERGED_AT_N640"),
    ],
)
def test_orchestrator_always_runs_both_stages_and_only_joint_pass_selects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n320_pass: bool,
    n640_pass: bool,
    expected_status: str,
) -> None:
    seeds = make_seed_bank(640, master_seed=continuation.MASTER_SEED)
    contract = {
        "design_sha256": "d" * 64,
        "occupant_seeds": list(seeds),
        "source_experiment": {"source_design_sha256": "s" * 64},
    }
    weather = pd.DataFrame({"member_id": ["weather-a", "weather-b"]})
    advances: list[int] = []
    monkeypatch.setattr(
        continuation,
        "prepare_convergence_continuation",
        lambda destination, source_output_dir: contract,
    )
    monkeypatch.setattr(continuation, "_weather_selection", lambda: weather.copy())
    monkeypatch.setattr(continuation, "_bootstrap_all_partitions", lambda *args: None)
    monkeypatch.setattr(continuation, "_verify_imported_aggregate", lambda *args: None)
    monkeypatch.setattr(
        continuation,
        "_advance_all_partitions",
        lambda member_ids, target, *args: advances.append(target),
    )
    monkeypatch.setattr(
        continuation, "_verify_historical_decisions", lambda *args: None
    )

    def fake_collect(destination, selected_weather, seed_bank, design, target):
        passed = n320_pass if target == 320 else n640_pass
        evaluator = False if target == 320 else n320_pass and n640_pass
        manifest = pd.DataFrame({"run_id": [f"run-{target}"]})
        diagnostics = pd.DataFrame(
            {"run_id": [f"run-{target}"], "occupant_seed": [seeds[target - 1]]}
        )
        evidence = pd.DataFrame(
            {
                "seed_count": [target],
                "criterion_pass": [passed],
                "panel_converged_at_checkpoint": [evaluator],
            }
        )
        return manifest, diagnostics, evidence, passed, evaluator

    monkeypatch.setattr(
        continuation, "_collect_continuation_checkpoint", fake_collect
    )
    monkeypatch.setattr(
        continuation,
        "_validate_terminal_summary_semantics",
        lambda destination, prepared, summary: summary["status"] == "CONVERGED",
    )

    summary = continuation._run_convergence_continuation_unlocked(
        tmp_path,
        source_output_dir=SOURCE_OUTPUT,
        max_workers=1,
    )

    assert advances == [320, 640]
    assert summary["status"] == expected_status
    assert summary["evaluated_seed_count"] == 640
    assert summary["selected_seed_count"] == (
        640 if expected_status == "CONVERGED" else None
    )
    assert summary["selected_occupant_seeds"] == (
        list(seeds) if expected_status == "CONVERGED" else None
    )
    assert summary["new_checkpoint_decisions"]["n320"][
        "selection_permitted"
    ] is False
    assert _read_json(tmp_path / "checkpoints/n320/checkpoint_summary.json")[
        "status"
    ] == "NOT_YET_CONVERGED"


def test_n320_checkpoint_writer_rejects_selection(tmp_path: Path) -> None:
    frame = pd.DataFrame({"run_id": ["run-a"]})
    with pytest.raises(MonteCarloContractError, match="never a selectable"):
        continuation._write_continuation_checkpoint(
            tmp_path,
            320,
            frame,
            frame,
            pd.DataFrame(
                {
                    "seed_count": [320],
                    "criterion_pass": [True],
                    "panel_converged_at_checkpoint": [False],
                }
            ),
            panel_pass=True,
            selection_authorized=True,
        )


def test_selection_loader_is_exact_seed_and_checksum_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = make_seed_bank(640, master_seed=continuation.MASTER_SEED)
    payload = {
        "convergence_continuation_contract_version": (
            continuation.CONVERGENCE_CONTINUATION_CONTRACT_VERSION
        ),
        "source_experiment": {
            "source_output_relative_path": "thermal_model/data/monte_carlo/"
            "convergence_panel_n160_extension"
        },
        "occupant_seeds": list(seeds),
        "occupant_seed_bank_sha256": ordered_seed_bank_sha256(seeds),
    }
    contract = {**payload, "design_sha256": canonical_sha256(payload)}
    continuation._atomic_json(
        contract, tmp_path / "convergence_continuation_contract.json"
    )
    evidence = tmp_path / "convergence_results.csv"
    evidence.write_text("seed_count\n640\n", encoding="utf-8")
    evidence_sha = _sha256_file(evidence)
    continuation._atomic_json(
        {
            "status": "CONVERGED",
            "selected_occupant_seeds": list(seeds),
            "artifact_sha256": {"convergence_results.csv": evidence_sha},
        },
        tmp_path / "convergence_continuation_summary.json",
    )
    monkeypatch.setattr(
        continuation, "_validate_completed_source", lambda source: {}
    )
    monkeypatch.setattr(
        continuation, "_validate_terminal_summary_semantics", lambda *args: True
    )

    selection = continuation.load_convergence_continuation_selection(tmp_path)
    assert selection.occupant_seeds == seeds
    assert selection.convergence_results_sha256 == evidence_sha
    assert selection.convergence_rule == PROSPECTIVE_N320_N640_CONVERGENCE_RULE

    evidence.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(MonteCarloContractError, match="selected checksum"):
        continuation.load_convergence_continuation_selection(tmp_path)


def test_status_labels_obsolete_contract_as_stale(tmp_path: Path) -> None:
    continuation._atomic_json(
        {
            "convergence_continuation_contract_version": "obsolete-v0",
            "design_sha256": "a" * 64,
        },
        tmp_path / "convergence_continuation_contract.json",
    )

    status = continuation.convergence_continuation_status(tmp_path)

    assert status == {
        "status": "STALE_CONTRACT",
        "output_dir": str(tmp_path.resolve()),
        "persisted_contract_version": "obsolete-v0",
        "required_contract_version": (
            continuation.CONVERGENCE_CONTINUATION_CONTRACT_VERSION
        ),
        "design_sha256": "a" * 64,
    }
