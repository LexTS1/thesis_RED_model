from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from thermal_model.monte_carlo import convergence_runner as runner
from thermal_model.monte_carlo.contracts import (
    MonteCarloContractError,
    archetype_state_sha256,
    canonical_sha256,
)


def _checkpoint_manifest(seed_bank: tuple[int, ...]) -> pd.DataFrame:
    records: list[dict[str, int | str]] = []
    for rank, seed in enumerate(seed_bank, start=1):
        for panel_cell in range(3):
            records.append(
                {
                    "run_id": f"run-{rank}-{panel_cell}",
                    "occupant_seed": seed,
                    "occupant_seed_rank": rank,
                }
            )
    return pd.DataFrame.from_records(records)


def _write_diagnostics(
    path: Path,
    manifest: pd.DataFrame,
    *,
    seed_count: int,
) -> pd.DataFrame:
    diagnostics = manifest.loc[
        manifest["occupant_seed_rank"] <= seed_count,
        ["run_id", "occupant_seed"],
    ].copy()
    diagnostics.to_csv(path, index=False)
    return diagnostics


def test_frozen_convergence_panel_revalidates_gate3_source_and_stock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states, panel = runner.load_convergence_panel()

    assert len(states) == len(runner.PANEL_SPECS) == 3
    assert panel["demand_role"].tolist() == ["low", "medium", "high"]
    assert set(panel["state_id"]) == {
        "TABULA_existing",
        "TABULA_standard_B_proxy",
        "TABULA_advanced_A_proxy",
    }
    assert panel["positive_weight_dwellings_2050"].gt(0.0).all()
    assert panel["validation_source_sha256"].eq(
        runner.EXPECTED_VALIDATION_SOURCE_SHA256
    ).all()
    assert panel["archetype_state_sha256"].tolist() == [
        archetype_state_sha256(state) for state in states
    ]

    tampered_source = tmp_path / "deterministic_archetype_validation.csv"
    tampered_source.write_bytes(
        runner.VALIDATION_SOURCE_PATH.read_bytes() + b"\n"
    )
    monkeypatch.setattr(runner, "VALIDATION_SOURCE_PATH", tampered_source)
    with pytest.raises(MonteCarloContractError, match="source changed"):
        runner.load_convergence_panel()


def test_prepare_convergence_experiment_freezes_complete_inventory(
    tmp_path: Path,
) -> None:
    contract = runner.prepare_convergence_experiment(tmp_path)

    unsigned = {
        key: value for key, value in contract.items() if key != "design_sha256"
    }
    assert contract["design_sha256"] == canonical_sha256(unsigned)
    assert contract["convergence_execution_contract_version"] == (
        runner.CONVERGENCE_EXECUTION_CONTRACT_VERSION
    )
    assert contract["partition_checkpoint_protocol_version"] == (
        runner.PARTITION_CHECKPOINT_PROTOCOL_VERSION
    )
    assert len(contract["panel"]) == 3
    assert len(contract["weather_members"]) == 54
    assert len(contract["occupant_seeds"]) == runner.MAX_SEED_COUNT == 80
    assert len(set(contract["occupant_seeds"])) == 80
    assert contract["convergence_rule"]["checkpoints"] == [5, 10, 20, 40, 80]
    assert contract["expected_maximum_run_count"] == 3 * 54 * 80
    assert contract["model_scenario"]["scenario_id"] == "central"
    assert contract["adaptive_stopping"] is True
    assert contract["selection_artifact_sha256"] == {
        filename: runner._sha256_file(tmp_path / filename)
        for filename in ("panel_selection.csv", "weather_selection.csv")
    }

    persisted = json.loads(
        (tmp_path / "convergence_execution_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == contract
    assert len(pd.read_csv(tmp_path / "panel_selection.csv")) == 3
    assert len(pd.read_csv(tmp_path / "weather_selection.csv")) == 54
    assert runner.prepare_convergence_experiment(tmp_path) == contract

    persisted["master_seed"] += 1
    (tmp_path / "convergence_execution_contract.json").write_text(
        json.dumps(persisted), encoding="utf-8"
    )
    with pytest.raises(MonteCarloContractError, match="different design"):
        runner.prepare_convergence_experiment(tmp_path)


@pytest.mark.parametrize(
    "filename", ["panel_selection.csv", "weather_selection.csv"]
)
def test_prepare_rejects_tampered_selection_artifacts(
    tmp_path: Path,
    filename: str,
) -> None:
    runner.prepare_convergence_experiment(tmp_path)
    with (tmp_path / filename).open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")

    with pytest.raises(MonteCarloContractError, match="checksum mismatch"):
        runner.prepare_convergence_experiment(tmp_path)


def test_partition_checkpoint_accepts_only_an_exact_contiguous_seed_prefix(
    tmp_path: Path,
) -> None:
    seed_bank = (11, 22, 33)
    manifest = _checkpoint_manifest(seed_bank)
    diagnostics_path = tmp_path / "run_diagnostics.csv"
    expected = _write_diagnostics(
        diagnostics_path, manifest, seed_count=2
    )

    restored, completed = runner._read_partition_diagnostics(
        diagnostics_path, manifest, seed_bank
    )
    pd.testing.assert_frame_equal(restored, expected)
    assert completed == 2

    gapped = manifest.loc[
        manifest["occupant_seed_rank"].isin([1, 3]),
        ["run_id", "occupant_seed"],
    ]
    gapped.to_csv(diagnostics_path, index=False)
    with pytest.raises(MonteCarloContractError, match="contiguous seed prefix"):
        runner._read_partition_diagnostics(
            diagnostics_path, manifest, seed_bank
        )

    partial = expected.iloc[:-1]
    partial.to_csv(diagnostics_path, index=False)
    with pytest.raises(MonteCarloContractError, match="exact run-ID prefix"):
        runner._read_partition_diagnostics(
            diagnostics_path, manifest, seed_bank
        )

    foreign = expected.copy()
    foreign.loc[0, "occupant_seed"] = 999
    foreign.to_csv(diagnostics_path, index=False)
    with pytest.raises(MonteCarloContractError, match="foreign occupant seed"):
        runner._read_partition_diagnostics(
            diagnostics_path, manifest, seed_bank
        )


def test_partition_checkpoint_rejects_run_ids_reassigned_between_valid_seeds(
    tmp_path: Path,
) -> None:
    seed_bank = (11, 22)
    manifest = _checkpoint_manifest(seed_bank)
    diagnostics_path = tmp_path / "run_diagnostics.csv"
    diagnostics = _write_diagnostics(
        diagnostics_path, manifest, seed_count=2
    )
    first_seed_row = diagnostics.index[diagnostics["occupant_seed"] == 11][0]
    second_seed_row = diagnostics.index[diagnostics["occupant_seed"] == 22][0]
    diagnostics.loc[first_seed_row, "occupant_seed"] = 22
    diagnostics.loc[second_seed_row, "occupant_seed"] = 11
    diagnostics.to_csv(diagnostics_path, index=False)

    with pytest.raises(MonteCarloContractError, match="occupant-seed identity"):
        runner._read_partition_diagnostics(
            diagnostics_path, manifest, seed_bank
        )


def test_alternating_slots_reject_tampering_and_ignore_uncommitted_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_bank = (11, 22)
    manifest = _checkpoint_manifest(seed_bank)
    first = manifest.loc[
        manifest["occupant_seed_rank"] <= 1,
        ["run_id", "occupant_seed"],
    ].copy()
    first["annual_heating_kWh"] = 100.0
    pointer_one = runner._commit_partition_diagnostics(
        tmp_path,
        first,
        manifest,
        seed_bank,
        completed_seed_count=1,
        active_slot=None,
        design_sha256="d" * 64,
        member_id="weather-a",
    )

    second = manifest.loc[
        manifest["occupant_seed_rank"] <= 2,
        ["run_id", "occupant_seed"],
    ].copy()
    second["annual_heating_kWh"] = 200.0
    real_atomic_json = runner._atomic_json

    def interrupt_pointer_commit(payload: dict, path: Path) -> None:
        if path.name == "progress.json":
            raise RuntimeError("interrupted before pointer commit")
        real_atomic_json(payload, path)

    monkeypatch.setattr(runner, "_atomic_json", interrupt_pointer_commit)
    with pytest.raises(RuntimeError, match="before pointer commit"):
        runner._commit_partition_diagnostics(
            tmp_path,
            second,
            manifest,
            seed_bank,
            completed_seed_count=2,
            active_slot=str(pointer_one["active_diagnostics_slot"]),
            design_sha256="d" * 64,
            member_id="weather-a",
        )

    restored, completed, active_slot = runner._restore_partition_diagnostics(
        tmp_path,
        manifest,
        seed_bank,
        design_sha256="d" * 64,
        member_id="weather-a",
    )
    assert completed == 1
    assert active_slot == pointer_one["active_diagnostics_slot"]
    assert set(restored["occupant_seed"]) == {11}

    monkeypatch.setattr(runner, "_atomic_json", real_atomic_json)
    pointer_two = runner._commit_partition_diagnostics(
        tmp_path,
        second,
        manifest,
        seed_bank,
        completed_seed_count=2,
        active_slot=active_slot,
        design_sha256="d" * 64,
        member_id="weather-a",
    )
    assert pointer_two["active_diagnostics_slot"] != active_slot
    assert all((tmp_path / filename).is_file() for filename in runner.DIAGNOSTICS_SLOT_FILENAMES)

    active_path = tmp_path / str(pointer_two["active_diagnostics_slot"])
    tampered = pd.read_csv(active_path)
    tampered.loc[0, "annual_heating_kWh"] += 1.0
    tampered.to_csv(active_path, index=False)
    with pytest.raises(MonteCarloContractError, match="slot checksum mismatch"):
        runner._restore_partition_diagnostics(
            tmp_path,
            manifest,
            seed_bank,
            design_sha256="d" * 64,
            member_id="weather-a",
        )


def test_execution_lock_never_removes_empty_malformed_or_recent_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_lock = empty_dir / "execution.lock"
    empty_lock.write_bytes(b"")
    with pytest.raises(MonteCarloContractError, match="malformed or still being"):
        runner._acquire_execution_lock(empty_dir)
    assert empty_lock.exists()

    recent_dir = tmp_path / "recent"
    recent_dir.mkdir()
    recent_lock = recent_dir / "execution.lock"
    recent_lock.write_text(
        json.dumps({"pid": 999_999, "started_at_utc": runner._utc_now()}),
        encoding="utf-8",
    )

    def missing_process(pid: int, signal: int) -> None:
        del pid, signal
        raise ProcessLookupError

    monkeypatch.setattr(runner.os, "kill", missing_process)
    with pytest.raises(MonteCarloContractError, match="too recent"):
        runner._acquire_execution_lock(recent_dir)
    assert recent_lock.exists()

    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    (stale_dir / "execution.lock").write_text(
        json.dumps({"pid": 999_999, "started_at_utc": "2000-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    acquired = runner._acquire_execution_lock(stale_dir)
    try:
        assert json.loads(acquired.read_text(encoding="utf-8"))["pid"] == runner.os.getpid()
    finally:
        runner._release_execution_lock(acquired)


def test_completed_partition_early_return_retires_stale_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_bank = (11,)
    manifest = _checkpoint_manifest(seed_bank)
    member = SimpleNamespace(
        member_id="weather-a",
        climate_scenario_id="rcp_4_5",
        weather_contract_sha256="a" * 64,
        forcing_sha256="b" * 64,
    )
    monkeypatch.setattr(
        runner,
        "load_convergence_panel",
        lambda: ((object(), object(), object()), pd.DataFrame()),
    )
    monkeypatch.setattr(runner, "load_weather_member", lambda member_id: member)
    monkeypatch.setattr(
        runner,
        "build_balanced_manifest",
        lambda states, weather, seeds, scenarios: manifest.copy(),
    )

    def fake_execute(*args: object) -> tuple[pd.DataFrame, pd.DataFrame, None]:
        diagnostics = manifest[["run_id", "occupant_seed"]].copy()
        diagnostics["annual_heating_kWh"] = 100.0
        return manifest.copy(), diagnostics, None

    monkeypatch.setattr(runner, "execute_balanced_design", fake_execute)
    runner._advance_weather_partition(
        "weather-a", 1, seed_bank, str(tmp_path), "d" * 64
    )
    failure_path = tmp_path / "partitions/weather-a/last_failure.json"
    runner._atomic_json(
        {
            "status": "FAILED",
            "exception_type": "RuntimeError",
            "exception_message": "stale failure",
        },
        failure_path,
    )
    monkeypatch.setattr(
        runner,
        "execute_balanced_design",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not rerun")),
    )

    resumed = runner._advance_weather_partition(
        "weather-a", 1, seed_bank, str(tmp_path), "d" * 64
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert resumed["completed_seed_count"] == 1
    assert failure["status"] == "RECOVERED"
    assert failure["recovered_at_seed_count"] == 1
    assert failure["exception_message"] == "stale failure"


def test_adaptive_runner_promotes_only_the_first_converged_checkpoint_to_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_bank = tuple(range(100, 180))
    contract = {"design_sha256": "d" * 64, "occupant_seeds": seed_bank}
    weather = pd.DataFrame({"member_id": ["weather-a", "weather-b"]})
    advanced: list[int] = []

    monkeypatch.setattr(
        runner, "prepare_convergence_experiment", lambda destination: contract
    )
    monkeypatch.setattr(runner, "_weather_selection", lambda: weather.copy())

    def fake_advance(
        member_ids: tuple[str, ...],
        checkpoint: int,
        seeds: tuple[int, ...],
        destination: Path,
        design_sha256: str,
        workers: int,
    ) -> None:
        assert member_ids == ("weather-a", "weather-b")
        assert seeds == seed_bank
        assert destination == tmp_path.resolve()
        assert design_sha256 == contract["design_sha256"]
        assert workers == 1
        advanced.append(checkpoint)

    def fake_collect(
        destination: Path,
        selected_weather: pd.DataFrame,
        seeds: tuple[int, ...],
        checkpoint: int,
        rule: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
        del destination, selected_weather, seeds, rule
        converged = checkpoint == 10
        manifest = pd.DataFrame(
            {"run_id": [f"manifest-at-{checkpoint}"]}
        )
        diagnostics = pd.DataFrame(
            {
                "run_id": [f"diagnostics-at-{checkpoint}"],
                "occupant_seed": [seed_bank[checkpoint - 1]],
            }
        )
        convergence = pd.DataFrame(
            {
                "seed_count": [checkpoint],
                "panel_converged_at_checkpoint": [converged],
            }
        )
        return manifest, diagnostics, convergence, converged

    monkeypatch.setattr(runner, "_advance_all_weather", fake_advance)
    monkeypatch.setattr(runner, "_collect_checkpoint", fake_collect)
    runner._atomic_json(
        {
            "status": "FAILED",
            "design_sha256": contract["design_sha256"],
            "started_at_utc": "2026-01-01T00:00:00+00:00",
        },
        tmp_path / "convergence_summary.json",
    )

    summary = runner.run_convergence_experiment(tmp_path, max_workers=1)

    assert advanced == [5, 10]
    assert summary["status"] == "CONVERGED"
    assert summary["selected_seed_count"] == 10
    assert summary["evaluated_seed_count"] == 10
    assert summary["selected_occupant_seeds"] == list(seed_bank[:10])
    assert summary["weather_member_count"] == 2
    assert summary["panel_cell_count"] == 3
    assert (
        json.loads(
            (tmp_path / "checkpoints/n005/checkpoint_summary.json").read_text()
        )["status"]
        == "NOT_YET_CONVERGED"
    )
    assert (
        json.loads(
            (tmp_path / "checkpoints/n010/checkpoint_summary.json").read_text()
        )["status"]
        == "CONVERGED"
    )
    for filename, digest in summary["artifact_sha256"].items():
        assert runner._sha256_file(tmp_path / filename) == digest

    advanced.clear()
    assert runner.run_convergence_experiment(tmp_path, max_workers=1) == summary
    assert advanced == []

    with (tmp_path / "run_diagnostics.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(MonteCarloContractError, match="artifact changed"):
        runner.run_convergence_experiment(tmp_path, max_workers=1)


def test_adaptive_runner_reports_no_selected_seed_at_final_failed_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_bank = tuple(range(100, 180))
    contract = {"design_sha256": "e" * 64, "occupant_seeds": seed_bank}
    weather = pd.DataFrame({"member_id": ["weather-a"]})
    advanced: list[int] = []

    monkeypatch.setattr(
        runner, "prepare_convergence_experiment", lambda destination: contract
    )
    monkeypatch.setattr(runner, "_weather_selection", lambda: weather.copy())
    monkeypatch.setattr(
        runner,
        "_advance_all_weather",
        lambda member_ids, checkpoint, seeds, destination, design_sha256, workers: (
            advanced.append(checkpoint)
        ),
    )

    def never_converged(
        destination: Path,
        selected_weather: pd.DataFrame,
        seeds: tuple[int, ...],
        checkpoint: int,
        rule: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
        del destination, selected_weather, seeds, rule
        return (
            pd.DataFrame({"run_id": [f"manifest-at-{checkpoint}"]}),
            pd.DataFrame(
                {
                    "run_id": [f"diagnostics-at-{checkpoint}"],
                    "occupant_seed": [seed_bank[checkpoint - 1]],
                }
            ),
            pd.DataFrame(
                {
                    "seed_count": [checkpoint],
                    "panel_converged_at_checkpoint": [False],
                }
            ),
            False,
        )

    monkeypatch.setattr(runner, "_collect_checkpoint", never_converged)

    summary = runner.run_convergence_experiment(tmp_path, max_workers=1)

    assert advanced == [5, 10, 20, 40, 80]
    assert summary["status"] == "NOT_CONVERGED_AT_N80"
    assert summary["selected_seed_count"] is None
    assert summary["selected_occupant_seeds"] is None
    assert summary["evaluated_seed_count"] == 80
    assert "No production seed count was selected" in summary[
        "production_interpretation"
    ]


def test_coordinator_failure_is_persisted_and_lock_is_released(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_run_convergence_experiment_unlocked",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("coordinator interrupted")
        ),
    )

    with pytest.raises(RuntimeError, match="coordinator interrupted"):
        runner.run_convergence_experiment(tmp_path, max_workers=1)

    summary = json.loads(
        (tmp_path / "convergence_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "FAILED"
    assert summary["coordinator_failure"]["exception_type"] == "RuntimeError"
    assert summary["coordinator_failure"]["exception_message"] == (
        "coordinator interrupted"
    )
    assert not (tmp_path / "execution.lock").exists()

    monkeypatch.setattr(
        runner,
        "_run_convergence_experiment_unlocked",
        lambda *args, **kwargs: {"status": "RESUMED"},
    )
    assert runner.run_convergence_experiment(tmp_path, max_workers=1) == {
        "status": "RESUMED"
    }


def test_status_labels_an_obsolete_execution_contract_as_stale(
    tmp_path: Path,
) -> None:
    runner._atomic_json(
        {
            "convergence_execution_contract_version": "obsolete-v1",
            "design_sha256": "a" * 64,
        },
        tmp_path / "convergence_execution_contract.json",
    )

    status = runner.convergence_status(tmp_path)

    assert status["status"] == "STALE_CONTRACT"
    assert status["persisted_contract_version"] == "obsolete-v1"
    assert status["required_contract_version"] == (
        runner.CONVERGENCE_EXECUTION_CONTRACT_VERSION
    )
    assert status["design_sha256"] == "a" * 64


def test_load_convergence_selection_returns_checksum_bound_runner_arguments(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "convergence_results.csv"
    evidence.write_text("seed_count,panel_converged_at_checkpoint\n20,true\n")
    evidence_sha256 = runner._sha256_file(evidence)
    seeds = tuple(range(100, 180))
    design_sha256 = "d" * 64
    runner._atomic_json(
        {
            "convergence_execution_contract_version": (
                runner.CONVERGENCE_EXECUTION_CONTRACT_VERSION
            ),
            "design_sha256": design_sha256,
            "occupant_seeds": seeds,
        },
        tmp_path / "convergence_execution_contract.json",
    )
    runner._atomic_json(
        {
            "status": "CONVERGED",
            "design_sha256": design_sha256,
            "selected_seed_count": 20,
            "first_panel_converged_checkpoint": 20,
            "selected_occupant_seeds": seeds[:20],
            "artifact_sha256": {
                "convergence_results.csv": evidence_sha256,
            },
        },
        tmp_path / "convergence_summary.json",
    )

    selection = runner.load_convergence_selection(tmp_path)

    assert selection.occupant_seeds == seeds[:20]
    assert selection.convergence_results_path == evidence.resolve()
    assert selection.convergence_results_sha256 == evidence_sha256
    assert selection.design_sha256 == design_sha256

    evidence.write_text("tampered\n")
    with pytest.raises(MonteCarloContractError, match="evidence"):
        runner.load_convergence_selection(tmp_path)


def test_load_convergence_selection_rejects_nonconverged_summary(
    tmp_path: Path,
) -> None:
    runner._atomic_json(
        {
            "convergence_execution_contract_version": (
                runner.CONVERGENCE_EXECUTION_CONTRACT_VERSION
            ),
            "design_sha256": "d" * 64,
            "occupant_seeds": [11, 22, 33],
        },
        tmp_path / "convergence_execution_contract.json",
    )
    runner._atomic_json(
        {"status": "NOT_CONVERGED_AT_N80", "design_sha256": "d" * 64},
        tmp_path / "convergence_summary.json",
    )

    with pytest.raises(MonteCarloContractError, match="only after convergence"):
        runner.load_convergence_selection(tmp_path)
