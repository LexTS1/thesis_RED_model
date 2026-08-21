from __future__ import annotations

from concurrent.futures import Future
import json

import pandas as pd
import pytest

from thermal_model.monte_carlo import runner
from thermal_model.monte_carlo.aggregation import (
    AUTHORITATIVE_STOCK_CONTENT_SHA256,
    AUTHORITATIVE_STOCK_SOURCE_SHA256,
    DEFAULT_STOCK_WEIGHTS_PATH,
    load_stock_weights,
    validate_stock_weights,
)
from thermal_model.monte_carlo.contracts import (
    MonteCarloContractError,
    canonical_sha256,
)


def test_authoritative_stock_identity_is_pinned() -> None:
    authoritative = load_stock_weights()
    assert authoritative["stock_weights_sha256"].eq(
        AUTHORITATIVE_STOCK_CONTENT_SHA256
    ).all()
    assert authoritative["stock_weights_source_sha256"].eq(
        AUTHORITATIVE_STOCK_SOURCE_SHA256
    ).all()

    raw = pd.read_csv(DEFAULT_STOCK_WEIGHTS_PATH)
    with pytest.raises(MonteCarloContractError, match="pinned 2050 stock"):
        validate_stock_weights(raw, require_authoritative_shape=True)

    relabelled = raw.copy(deep=True)
    relabelled["region"] = relabelled["region"].map(
        {
            "Flemish Region": "Region X",
            "Walloon Region": "Region Y",
            "Brussels-Capital Region": "Region Z",
        }
    )
    with pytest.raises(MonteCarloContractError, match="pinned 2050 stock"):
        validate_stock_weights(
            relabelled,
            source_sha256=AUTHORITATIVE_STOCK_SOURCE_SHA256,
            require_authoritative_shape=True,
        )


def test_streaming_execution_lock_is_single_writer(tmp_path) -> None:
    destination = tmp_path / "production"
    lock = runner._acquire_streaming_execution_lock(destination)
    try:
        with pytest.raises(MonteCarloContractError, match="already running"):
            runner._acquire_streaming_execution_lock(destination)
    finally:
        runner._release_streaming_execution_lock(lock)
    assert not lock.exists()


def test_public_wrapper_releases_lock_and_removes_empty_failed_preflight(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "failed_preflight"

    def fail(*args, **kwargs):
        raise MonteCarloContractError("preflight failed")

    monkeypatch.setattr(runner, "_execute_streaming_stock_design_unlocked", fail)
    with pytest.raises(MonteCarloContractError, match="preflight failed"):
        runner.execute_streaming_stock_design(
            (), (), (), output_dir=destination
        )
    assert not destination.exists()


def test_parallel_partition_coordinator_assigns_each_partition_once(
    tmp_path, monkeypatch
) -> None:
    calls: list[str] = []

    def execute(partition_id, *args):
        calls.append(partition_id)
        return {"partition_id": partition_id}

    class ImmediateExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args):
            future = Future()
            try:
                future.set_result(function(*args))
            except Exception as exc:  # pragma: no cover - exercised by production failures
                future.set_exception(exc)
            return future

    monkeypatch.setattr(runner, "_execute_streaming_stock_partition", execute)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ImmediateExecutor)
    specs = [
        {
            "partition_id": f"p{index}",
            "weather_member_id": f"w{index}",
            "model_scenario_id": "central",
        }
        for index in range(3)
    ]
    runner._advance_streaming_stock_partitions(
        specs,
        (),
        (1,),
        pd.DataFrame(),
        tmp_path,
        "d" * 64,
        True,
        2,
    )
    assert sorted(calls) == ["p0", "p1", "p2"]


def test_streaming_status_treats_committed_progress_as_authoritative(
    tmp_path,
) -> None:
    destination = tmp_path / "production"
    unsigned = {
        "streaming_stock_contract_version": runner.STREAMING_STOCK_CONTRACT_VERSION,
        "occupant_seeds": [11, 22],
        "expected_run_count": 4,
        "model_scenarios": [{"scenario_id": "central"}],
        "partition_specs": [
            {
                "partition_id": "p1",
                "weather_member_id": "w1",
                "model_scenario_id": "central",
            },
            {
                "partition_id": "p2",
                "weather_member_id": "w2",
                "model_scenario_id": "central",
            },
        ],
    }
    design = {**unsigned, "design_sha256": canonical_sha256(unsigned)}
    runner._atomic_json(design, destination / "streaming_design_contract.json")
    p1 = destination / "partitions/p1"
    runner._atomic_json(
        {
            "streaming_stock_contract_version": runner.STREAMING_STOCK_CONTRACT_VERSION,
            "design_sha256": design["design_sha256"],
            "partition_id": "p1",
            "completed_seed_count": 2,
        },
        p1 / "progress.json",
    )
    # This marker is stale: its failed rank is already in the committed prefix.
    runner._atomic_json(
        {"status": "FAILED", "occupant_seed_rank": 2},
        p1 / "last_failure.json",
    )
    p2 = destination / "partitions/p2"
    runner._atomic_json(
        {
            "streaming_stock_contract_version": runner.STREAMING_STOCK_CONTRACT_VERSION,
            "design_sha256": design["design_sha256"],
            "partition_id": "p2",
            "completed_seed_count": 1,
        },
        p2 / "progress.json",
    )
    runner._atomic_json(
        {"status": "FAILED", "occupant_seed_rank": 2},
        p2 / "last_failure.json",
    )

    status = runner.streaming_stock_status(destination)
    assert status["status"] == "INTERRUPTED"
    assert status["started_partition_count"] == 2
    assert status["partition_seed_count_histogram"] == {"1": 1, "2": 1}
    assert status["active_failure_count"] == 1
    assert status["progress_validation"] == "OBSERVATIONAL_POINTER_COUNTS_ONLY"


def test_stock_status_cli_is_read_only(tmp_path, capsys) -> None:
    destination = tmp_path / "not_started"
    assert runner.main(
        ["stock", "--status", "--output-dir", str(destination)]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "NOT_PREPARED"
    assert not destination.exists()
