"""Prospective n=320/n=640 continuation of Gate-5 seed convergence.

The completed n=160 extension is an immutable source.  This module imports and
authenticates its exact 160-seed prefix in a separate output tree, simulates
only seeds 161--640, and evaluates the two prospectively frozen checkpoints
n=320 and n=640.  n=320 is never a stopping point: the exact 640-seed prefix is
selected only when the complete panel passes at both new checkpoints.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .contracts import MonteCarloContractError, canonical_sha256
from .convergence_extension import (
    CONVERGENCE_EXTENSION_CONTRACT_VERSION,
    CONVERGENCE_EXTENSION_PARTITION_VERSION,
    DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
    FINAL_EXTENSION_ARTIFACTS,
    _strict_bool,
    _validate_terminal_summary_semantics as _validate_n160_terminal_summary,
    prepare_convergence_extension,
)
from .convergence_runner import (
    MASTER_SEED,
    MODEL_SCENARIO_ID,
    PANEL_SPECS,
    PARTITION_CHECKPOINT_PROTOCOL_VERSION,
    PROJECT_ROOT,
    ConvergenceSelection,
    _acquire_execution_lock,
    _atomic_csv,
    _atomic_json,
    _commit_partition_diagnostics,
    _copy_atomic,
    _frame_csv_sha256,
    _json_ready,
    _mark_failure_recovered,
    _read_json,
    _release_execution_lock,
    _restore_partition_diagnostics,
    _rule_payload,
    _sha256_file,
    _utc_now,
    _weather_selection,
    _write_checkpoint,
    load_convergence_panel,
)
from .design import (
    PROSPECTIVE_N160_CONVERGENCE_RULE,
    PROSPECTIVE_N320_N640_CONVERGENCE_RULE,
    build_balanced_manifest,
    evaluate_seed_convergence,
    make_seed_bank,
    ordered_seed_bank_sha256,
)
from .runner import execute_balanced_design
from .weather import load_weather_member


DEFAULT_CONVERGENCE_CONTINUATION_OUTPUT_DIR = (
    PROJECT_ROOT
    / "thermal_model/data/monte_carlo/convergence_panel_n640_continuation"
)
CONVERGENCE_CONTINUATION_CONTRACT_VERSION = (
    "gate5_convergence_n320_n640_continuation_v1"
)
CONVERGENCE_CONTINUATION_PARTITION_VERSION = (
    "gate5_convergence_n320_n640_continuation_partition_v1"
)
SOURCE_SEED_COUNT = 160
CONTINUATION_CHECKPOINTS = (320, 640)
MAXIMUM_SEED_COUNT = CONTINUATION_CHECKPOINTS[-1]
PROSPECTIVE_DECLARATION_DATE = "2026-08-10"
FINAL_CONTINUATION_ARTIFACTS = FINAL_EXTENSION_ARTIFACTS
TERMINAL_STATUSES = {"CONVERGED", "NOT_CONVERGED_AT_N640"}


def _relative_project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError as exc:
        raise MonteCarloContractError(
            "The n=160 convergence source must be inside the thesis project root."
        ) from exc


def _source_dir_from_contract(contract: Mapping[str, Any]) -> Path:
    relative = str(contract["source_experiment"]["source_output_relative_path"])
    source = (PROJECT_ROOT / relative).resolve()
    if source == DEFAULT_CONVERGENCE_CONTINUATION_OUTPUT_DIR.resolve():
        raise MonteCarloContractError(
            "Continuation source path resolves to its default destination."
        )
    return source


def _assert_rule_is_exact_continuation() -> None:
    source = _rule_payload(PROSPECTIVE_N160_CONVERGENCE_RULE)
    continuation = _rule_payload(PROSPECTIVE_N320_N640_CONVERGENCE_RULE)
    if (
        continuation["checkpoints"][:-2] != source["checkpoints"]
        or tuple(continuation["checkpoints"][-2:]) != CONTINUATION_CHECKPOINTS
        or {key: value for key, value in continuation.items() if key != "checkpoints"}
        != {key: value for key, value in source.items() if key != "checkpoints"}
    ):
        raise MonteCarloContractError(
            "The continuation rule must append only n=320 and n=640 to the "
            "unchanged n=160 protocol."
        )


def _validate_completed_source(
    source_output_dir: str | Path = DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
) -> dict[str, Any]:
    """Authenticate the exact terminal NOT_CONVERGED_AT_N160 source."""

    source = Path(source_output_dir).resolve()
    contract_path = source / "convergence_extension_contract.json"
    summary_path = source / "convergence_extension_summary.json"
    checkpoint_summary_path = source / "checkpoints/n160/checkpoint_summary.json"
    contract = _read_json(contract_path)
    summary = _read_json(summary_path)
    checkpoint_summary = _read_json(checkpoint_summary_path)

    if (
        contract.get("convergence_extension_contract_version")
        != CONVERGENCE_EXTENSION_CONTRACT_VERSION
    ):
        raise MonteCarloContractError(
            "The continuation source uses an unsupported n=160 contract version."
        )
    unsigned = {key: value for key, value in contract.items() if key != "design_sha256"}
    if canonical_sha256(unsigned) != contract.get("design_sha256"):
        raise MonteCarloContractError(
            "The n=160 source design checksum cannot be reconstructed."
        )

    # Re-run all source/base/selection-contract checks without changing the
    # existing tree.  The function only writes when the source contract is
    # absent; here its presence and exact equality are required first.
    prepared = prepare_convergence_extension(
        source,
        base_output_dir=(
            PROJECT_ROOT
            / str(contract["base_experiment"]["base_output_relative_path"])
        ),
    )
    if _json_ready(prepared) != contract:
        raise MonteCarloContractError(
            "The n=160 source no longer reproduces its frozen contract."
        )
    if _validate_n160_terminal_summary(source, contract, summary):
        raise MonteCarloContractError(
            "The n=320/n=640 continuation is permitted only after a non-converged "
            "n=160 source."
        )
    if (
        summary.get("status") != "NOT_CONVERGED_AT_N160"
        or int(summary.get("evaluated_seed_count", -1)) != SOURCE_SEED_COUNT
        or summary.get("selected_seed_count") is not None
        or summary.get("selected_occupant_seeds") is not None
        or summary.get("first_panel_converged_checkpoint") is not None
    ):
        raise MonteCarloContractError(
            "The continuation source must be exact terminal NOT_CONVERGED_AT_N160."
        )
    if (
        checkpoint_summary.get("status") != "NOT_YET_CONVERGED"
        or int(checkpoint_summary.get("seed_count", -1)) != SOURCE_SEED_COUNT
        or bool(checkpoint_summary.get("panel_converged_at_checkpoint"))
    ):
        raise MonteCarloContractError(
            "The source n=160 checkpoint is not the required non-converged result."
        )
    if (
        contract.get("convergence_rule")
        != _rule_payload(PROSPECTIVE_N160_CONVERGENCE_RULE)
        or summary.get("convergence_rule")
        != _rule_payload(PROSPECTIVE_N160_CONVERGENCE_RULE)
    ):
        raise MonteCarloContractError(
            "The n=160 source does not use the authorized unchanged rule."
        )

    source_seeds = tuple(int(value) for value in contract.get("occupant_seeds", ()))
    generated = make_seed_bank(MAXIMUM_SEED_COUNT, master_seed=MASTER_SEED)
    if (
        int(contract.get("master_seed", -1)) != MASTER_SEED
        or len(source_seeds) != SOURCE_SEED_COUNT
        or source_seeds != generated[:SOURCE_SEED_COUNT]
    ):
        raise MonteCarloContractError(
            "The source seed bank is not the exact prefix of the frozen 640-seed bank."
        )

    artifact_sha256 = summary.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping) or set(artifact_sha256) != set(
        FINAL_CONTINUATION_ARTIFACTS
    ):
        raise MonteCarloContractError(
            "The n=160 source lacks its exact terminal artifact ledger."
        )
    checkpoint_hashes = checkpoint_summary.get("artifact_sha256")
    if not isinstance(checkpoint_hashes, Mapping) or set(checkpoint_hashes) != set(
        FINAL_CONTINUATION_ARTIFACTS
    ):
        raise MonteCarloContractError(
            "The n=160 source checkpoint lacks its exact artifact ledger."
        )
    for filename in FINAL_CONTINUATION_ARTIFACTS:
        root_path = source / filename
        checkpoint_path = source / "checkpoints/n160" / filename
        root_sha = _sha256_file(root_path)
        if (
            root_sha != str(artifact_sha256[filename])
            or _sha256_file(checkpoint_path) != str(checkpoint_hashes[filename])
            or root_sha != _sha256_file(checkpoint_path)
        ):
            raise MonteCarloContractError(
                f"The n=160 source artifact changed: {filename}."
            )

    manifest = pd.read_csv(source / "run_manifest.csv", float_precision="round_trip")
    diagnostics = pd.read_csv(
        source / "run_diagnostics.csv", float_precision="round_trip"
    )
    expected_runs = len(PANEL_SPECS) * len(contract["weather_members"]) * SOURCE_SEED_COUNT
    if (
        len(manifest) != expected_runs
        or len(diagnostics) != expected_runs
        or manifest["run_id"].duplicated().any()
        or diagnostics["run_id"].duplicated().any()
        or set(manifest["run_id"].astype(str))
        != set(diagnostics["run_id"].astype(str))
    ):
        raise MonteCarloContractError(
            "The n=160 source manifest/diagnostics are incomplete or duplicated."
        )
    recomputed = evaluate_seed_convergence(
        diagnostics,
        seed_order=source_seeds,
        rule=PROSPECTIVE_N160_CONVERGENCE_RULE,
    )
    if _frame_csv_sha256(recomputed) != str(
        artifact_sha256["convergence_results.csv"]
    ):
        raise MonteCarloContractError(
            "The n=160 source decision cannot be reproduced from raw diagnostics."
        )
    current = recomputed.loc[recomputed["seed_count"] == SOURCE_SEED_COUNT]
    current_pass = _strict_bool(
        current["criterion_pass"], label="source n=160 criterion pass"
    )
    current_converged = _strict_bool(
        current["panel_converged_at_checkpoint"],
        label="source n=160 panel convergence",
    )
    if current.empty or bool(current_pass.all()) or bool(current_converged.any()):
        raise MonteCarloContractError(
            "The frozen continuation requires the authenticated failed n=160 "
            "expansion, not merely an unselected source."
        )

    return {
        "source_output_relative_path": _relative_project_path(source),
        "source_convergence_extension_contract_version": (
            CONVERGENCE_EXTENSION_CONTRACT_VERSION
        ),
        "source_design_sha256": str(contract["design_sha256"]),
        "source_contract_file_sha256": _sha256_file(contract_path),
        "source_summary_file_sha256": _sha256_file(summary_path),
        "source_n160_checkpoint_summary_sha256": _sha256_file(
            checkpoint_summary_path
        ),
        "source_terminal_status": str(summary["status"]),
        "source_completed_at_utc": str(summary["completed_at_utc"]),
        "source_evaluated_seed_count": SOURCE_SEED_COUNT,
        "source_run_count": expected_runs,
        "source_rule": _rule_payload(PROSPECTIVE_N160_CONVERGENCE_RULE),
        "source_rule_sha256": canonical_sha256(
            _rule_payload(PROSPECTIVE_N160_CONVERGENCE_RULE)
        ),
        "source_seed_bank_sha256": ordered_seed_bank_sha256(source_seeds),
        "source_artifact_sha256": {
            filename: str(artifact_sha256[filename])
            for filename in FINAL_CONTINUATION_ARTIFACTS
        },
        "source_n160_all_criteria_pass": False,
        "source_n160_panel_converged": False,
    }


def prepare_convergence_continuation(
    output_dir: str | Path = DEFAULT_CONVERGENCE_CONTINUATION_OUTPUT_DIR,
    *,
    source_output_dir: str | Path = DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
) -> dict[str, Any]:
    """Freeze the separate n=320/n=640 design without simulating."""

    destination = Path(output_dir).resolve()
    source = Path(source_output_dir).resolve()
    if (
        destination == source
        or source in destination.parents
        or destination in source.parents
    ):
        raise MonteCarloContractError(
            "The continuation and n=160 source must be separate, non-overlapping "
            "directory trees."
        )
    _assert_rule_is_exact_continuation()
    source_receipt = _validate_completed_source(source)
    source_contract = _read_json(source / "convergence_extension_contract.json")
    original_base = (
        PROJECT_ROOT
        / str(source_contract["base_experiment"]["base_output_relative_path"])
    ).resolve()
    if (
        destination == original_base
        or original_base in destination.parents
        or destination in original_base.parents
    ):
        raise MonteCarloContractError(
            "The continuation must not overlap the immutable original n=80 "
            "experiment tree."
        )
    states, panel = load_convergence_panel()
    weather = _weather_selection()
    seed_bank = make_seed_bank(MAXIMUM_SEED_COUNT, master_seed=MASTER_SEED)
    selection_sha256 = {
        "panel_selection.csv": _frame_csv_sha256(panel),
        "weather_selection.csv": _frame_csv_sha256(weather),
    }
    if (
        source_contract.get("panel") != panel.to_dict(orient="records")
        or source_contract.get("weather_members")
        != weather.to_dict(orient="records")
        or source_contract.get("selection_artifact_sha256") != selection_sha256
    ):
        raise MonteCarloContractError(
            "The current panel/weather inventory differs from the n=160 source."
        )

    runs_per_seed = len(states) * len(weather)
    payload = {
        "convergence_continuation_contract_version": (
            CONVERGENCE_CONTINUATION_CONTRACT_VERSION
        ),
        "partition_checkpoint_protocol_version": (
            PARTITION_CHECKPOINT_PROTOCOL_VERSION
        ),
        "prospective_declaration_date": PROSPECTIVE_DECLARATION_DATE,
        "methodological_rationale": (
            "The authenticated n=160 expansion failed the complete-panel "
            "criterion. Checkpoints n=320 and n=640 are therefore declared "
            "together so two entirely new consecutive passes are required."
        ),
        "source_experiment": source_receipt,
        "model_contract_version": source_contract["model_contract_version"],
        "central_thermal_assumptions_sha256": source_contract[
            "central_thermal_assumptions_sha256"
        ],
        "behaviour_assumptions_sha256": source_contract[
            "behaviour_assumptions_sha256"
        ],
        "occupant_distribution_sha256": source_contract[
            "occupant_distribution_sha256"
        ],
        "model_scenario": source_contract["model_scenario"],
        "model_scenario_sha256": source_contract["model_scenario_sha256"],
        "panel": panel.to_dict(orient="records"),
        "weather_members": weather.to_dict(orient="records"),
        "selection_artifact_sha256": selection_sha256,
        "master_seed": MASTER_SEED,
        "occupant_seeds": list(seed_bank),
        "occupant_seed_bank_sha256": ordered_seed_bank_sha256(seed_bank),
        "imported_seed_count": SOURCE_SEED_COUNT,
        "imported_seed_prefix_sha256": ordered_seed_bank_sha256(
            seed_bank[:SOURCE_SEED_COUNT]
        ),
        "new_seed_count": MAXIMUM_SEED_COUNT - SOURCE_SEED_COUNT,
        "continuation_seed_prefix_sha256": {
            f"n{checkpoint}": ordered_seed_bank_sha256(seed_bank[:checkpoint])
            for checkpoint in CONTINUATION_CHECKPOINTS
        },
        "convergence_rule": _rule_payload(
            PROSPECTIVE_N320_N640_CONVERGENCE_RULE
        ),
        "continuation_checkpoints": list(CONTINUATION_CHECKPOINTS),
        "selection_checkpoint": MAXIMUM_SEED_COUNT,
        "selection_requires_both_new_checkpoints_pass": True,
        "expected_imported_run_count": runs_per_seed * SOURCE_SEED_COUNT,
        "expected_new_run_count_161_to_320": runs_per_seed * (320 - 160),
        "expected_new_run_count_321_to_640": runs_per_seed * (640 - 320),
        "expected_new_run_count": runs_per_seed * (
            MAXIMUM_SEED_COUNT - SOURCE_SEED_COUNT
        ),
        "expected_total_run_count_at_n320": runs_per_seed * 320,
        "expected_total_run_count_at_n640": runs_per_seed * 640,
        "source_mutation_permitted": False,
        "intermediate_stopping_permitted": False,
        "n320_selection_permitted": False,
    }
    design_sha256 = canonical_sha256(payload)
    contract = {**payload, "design_sha256": design_sha256}

    contract_path = destination / "convergence_continuation_contract.json"
    panel_path = destination / "panel_selection.csv"
    weather_path = destination / "weather_selection.csv"
    if contract_path.exists():
        if _read_json(contract_path) != _json_ready(contract):
            raise MonteCarloContractError(
                "Convergence-continuation output belongs to a different design."
            )
    else:
        _atomic_csv(panel, panel_path)
        _atomic_csv(weather, weather_path)
        _atomic_json(contract, contract_path)
    for path in (panel_path, weather_path):
        if (
            not path.is_file()
            or _sha256_file(path) != selection_sha256[path.name]
        ):
            raise MonteCarloContractError(
                f"Continuation selection artifact changed: {path.name}."
            )
    return contract


def _continuation_partition_context(
    member_id: str,
    seed_bank: tuple[int, ...],
    destination: Path,
    design_sha256: str,
) -> tuple[tuple[Any, ...], Any, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    contract = _read_json(destination / "convergence_continuation_contract.json")
    unsigned = {key: value for key, value in contract.items() if key != "design_sha256"}
    if (
        contract.get("design_sha256") != design_sha256
        or canonical_sha256(unsigned) != design_sha256
    ):
        raise MonteCarloContractError(
            f"Continuation design changed while preparing {member_id}."
        )
    if tuple(int(value) for value in contract.get("occupant_seeds", ())) != seed_bank:
        raise MonteCarloContractError(
            f"Continuation seed bank changed while preparing {member_id}."
        )

    states, _ = load_convergence_panel()
    member = load_weather_member(member_id)
    manifest = build_balanced_manifest(
        states, [member], seed_bank, (MODEL_SCENARIO_ID,)
    )
    source_dir = _source_dir_from_contract(contract)
    source_partition = source_dir / "partitions" / member_id
    source_manifest_path = source_partition / "run_manifest.csv"
    source_contract_path = source_partition / "partition_contract.json"
    source_progress_path = source_partition / "progress.json"
    source_manifest = pd.read_csv(
        source_manifest_path, float_precision="round_trip"
    )
    source_partition_contract = _read_json(source_contract_path)
    source_progress = _read_json(source_progress_path)
    source_design_sha256 = str(
        contract["source_experiment"]["source_design_sha256"]
    )
    if (
        source_partition_contract.get("convergence_extension_partition_version")
        != CONVERGENCE_EXTENSION_PARTITION_VERSION
        or source_partition_contract.get("design_sha256")
        != source_design_sha256
        or source_partition_contract.get("weather_member_id") != member_id
        or _sha256_file(source_manifest_path)
        != str(source_partition_contract.get("run_manifest_sha256", ""))
    ):
        raise MonteCarloContractError(
            f"The n=160 source partition contract changed for {member_id}."
        )
    prefix_manifest = manifest.loc[
        manifest["occupant_seed_rank"] <= SOURCE_SEED_COUNT,
        source_manifest.columns,
    ]
    if _frame_csv_sha256(prefix_manifest) != _sha256_file(source_manifest_path):
        raise MonteCarloContractError(
            f"The 640-seed manifest does not preserve source run identities for "
            f"{member_id}."
        )
    source_diagnostics, source_completed, _ = _restore_partition_diagnostics(
        source_partition,
        source_manifest,
        seed_bank[:SOURCE_SEED_COUNT],
        design_sha256=source_design_sha256,
        member_id=member_id,
    )
    if source_completed != SOURCE_SEED_COUNT:
        raise MonteCarloContractError(
            f"Source partition {member_id} is not complete at n=160."
        )
    source_active_slot = str(source_progress.get("active_diagnostics_slot", ""))
    source_active_path = source_partition / source_active_slot
    source_active_sha256 = _sha256_file(source_active_path)
    if source_active_sha256 != str(
        source_progress.get("active_diagnostics_sha256", "")
    ):
        raise MonteCarloContractError(
            f"Source diagnostics checksum changed for {member_id}."
        )

    partition_dir = destination / "partitions" / member_id
    manifest_path = partition_dir / "run_manifest.csv"
    if not manifest_path.exists():
        _atomic_csv(manifest, manifest_path)
    elif _sha256_file(manifest_path) != _frame_csv_sha256(manifest):
        raise MonteCarloContractError(
            f"Continuation manifest checksum mismatch for {member_id}."
        )
    partition_contract = {
        "convergence_continuation_partition_version": (
            CONVERGENCE_CONTINUATION_PARTITION_VERSION
        ),
        "checkpoint_protocol_version": PARTITION_CHECKPOINT_PROTOCOL_VERSION,
        "design_sha256": design_sha256,
        "source_design_sha256": source_design_sha256,
        "weather_member_id": member.member_id,
        "climate_scenario_id": member.climate_scenario_id,
        "weather_contract_sha256": member.weather_contract_sha256,
        "weather_forcing_sha256": member.forcing_sha256,
        "imported_seed_count": SOURCE_SEED_COUNT,
        "continuation_checkpoints": list(CONTINUATION_CHECKPOINTS),
        "maximum_seed_count": MAXIMUM_SEED_COUNT,
        "expected_imported_run_count": len(source_manifest),
        "expected_run_count_at_n320": len(states) * 320,
        "expected_run_count_at_n640": len(manifest),
        "expected_imported_run_id_sha256": canonical_sha256(
            {"run_ids": source_manifest["run_id"].astype(str).tolist()}
        ),
        "expected_full_run_id_sha256": canonical_sha256(
            {"run_ids": manifest["run_id"].astype(str).tolist()}
        ),
        "run_manifest_sha256": _sha256_file(manifest_path),
        "source_partition_contract_sha256": _sha256_file(source_contract_path),
        "source_run_manifest_sha256": _sha256_file(source_manifest_path),
        "source_progress_sha256": _sha256_file(source_progress_path),
        "source_active_diagnostics_slot": source_active_slot,
        "source_active_diagnostics_sha256": source_active_sha256,
    }
    partition_contract_path = partition_dir / "partition_contract.json"
    if partition_contract_path.exists():
        if _read_json(partition_contract_path) != _json_ready(partition_contract):
            raise MonteCarloContractError(
                f"Continuation partition {member_id} belongs to a different design."
            )
    else:
        _atomic_json(partition_contract, partition_contract_path)
    return states, member, manifest, partition_contract, source_diagnostics


def _bootstrap_continuation_partition(
    member_id: str,
    seed_bank: tuple[int, ...],
    output_dir: str,
    design_sha256: str,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    _, _, manifest, partition_contract, source_diagnostics = (
        _continuation_partition_context(
            member_id, seed_bank, destination, design_sha256
        )
    )
    partition_dir = destination / "partitions" / member_id
    diagnostics, completed, active_slot = _restore_partition_diagnostics(
        partition_dir,
        manifest,
        seed_bank,
        design_sha256=design_sha256,
        member_id=member_id,
    )
    if completed == 0:
        pointer = _commit_partition_diagnostics(
            partition_dir,
            source_diagnostics,
            manifest,
            seed_bank,
            completed_seed_count=SOURCE_SEED_COUNT,
            active_slot=None,
            design_sha256=design_sha256,
            member_id=member_id,
        )
        diagnostics = source_diagnostics
        completed = SOURCE_SEED_COUNT
        active_slot = str(pointer["active_diagnostics_slot"])
    if completed < SOURCE_SEED_COUNT or active_slot is None:
        raise MonteCarloContractError(
            f"Continuation partition {member_id} lacks its complete n=160 import."
        )
    imported = diagnostics.loc[
        diagnostics["occupant_seed"].isin(seed_bank[:SOURCE_SEED_COUNT])
    ].copy()
    if (
        len(imported) != len(source_diagnostics)
        or _frame_csv_sha256(imported)
        != str(partition_contract["source_active_diagnostics_sha256"])
    ):
        raise MonteCarloContractError(
            f"Imported n=160 diagnostics changed for {member_id}."
        )
    receipt = {
        "status": "IMPORTED_AND_VERIFIED",
        "design_sha256": design_sha256,
        "source_design_sha256": partition_contract["source_design_sha256"],
        "weather_member_id": member_id,
        "imported_seed_count": SOURCE_SEED_COUNT,
        "imported_run_count": len(imported),
        "imported_diagnostics_sha256": _frame_csv_sha256(imported),
        "source_partition_contract_sha256": partition_contract[
            "source_partition_contract_sha256"
        ],
        "source_run_manifest_sha256": partition_contract[
            "source_run_manifest_sha256"
        ],
        "source_progress_sha256": partition_contract["source_progress_sha256"],
        "verified_at_utc": _utc_now(),
    }
    receipt_path = partition_dir / "import_receipt.json"
    if receipt_path.exists():
        existing = _read_json(receipt_path)
        def without_time(value: Mapping[str, Any]) -> dict[str, Any]:
            return {
                key: item
                for key, item in value.items()
                if key != "verified_at_utc"
            }

        if without_time(existing) != without_time(receipt):
            raise MonteCarloContractError(
                f"Continuation import receipt changed for {member_id}."
            )
    else:
        _atomic_json(receipt, receipt_path)
    return {
        "weather_member_id": member_id,
        "completed_seed_count": completed,
        "imported_run_count": len(imported),
    }


def _bootstrap_all_partitions(
    member_ids: Sequence[str],
    seed_bank: tuple[int, ...],
    destination: Path,
    design_sha256: str,
) -> None:
    for index, member_id in enumerate(member_ids, start=1):
        _bootstrap_continuation_partition(
            member_id, seed_bank, str(destination), design_sha256
        )
        if index % 9 == 0 or index == len(member_ids):
            print(
                f"[{_utc_now()}] authenticated n=160 import: "
                f"{index}/{len(member_ids)} weather partitions",
                flush=True,
            )


def _verify_imported_aggregate(
    destination: Path,
    weather: pd.DataFrame,
    seed_bank: tuple[int, ...],
    contract: Mapping[str, Any],
) -> None:
    manifests: list[pd.DataFrame] = []
    diagnostics_frames: list[pd.DataFrame] = []
    prefix = set(seed_bank[:SOURCE_SEED_COUNT])
    for member_id in weather["member_id"].astype(str):
        partition_dir = destination / "partitions" / member_id
        manifest = pd.read_csv(
            partition_dir / "run_manifest.csv", float_precision="round_trip"
        )
        diagnostics, completed, _ = _restore_partition_diagnostics(
            partition_dir,
            manifest,
            seed_bank,
            design_sha256=str(contract["design_sha256"]),
            member_id=member_id,
        )
        if completed < SOURCE_SEED_COUNT:
            raise MonteCarloContractError(
                f"Continuation import is incomplete for {member_id}."
            )
        manifests.append(
            manifest.loc[
                manifest["occupant_seed_rank"] <= SOURCE_SEED_COUNT
            ].copy()
        )
        diagnostics_frames.append(
            diagnostics.loc[diagnostics["occupant_seed"].isin(prefix)].copy()
        )
    manifest = pd.concat(manifests, ignore_index=True)
    diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
    source_hashes = contract["source_experiment"]["source_artifact_sha256"]
    if (
        _frame_csv_sha256(manifest) != source_hashes["run_manifest.csv"]
        or _frame_csv_sha256(diagnostics) != source_hashes["run_diagnostics.csv"]
    ):
        raise MonteCarloContractError(
            "The imported n=160 aggregate differs from the authenticated source."
        )
    reproduced = evaluate_seed_convergence(
        diagnostics,
        seed_order=seed_bank[:SOURCE_SEED_COUNT],
        rule=PROSPECTIVE_N160_CONVERGENCE_RULE,
    )
    if _frame_csv_sha256(reproduced) != source_hashes["convergence_results.csv"]:
        raise MonteCarloContractError(
            "The imported n=160 convergence evidence differs from the source."
        )


def _advance_continuation_partition(
    member_id: str,
    target_seed_count: int,
    seed_bank: tuple[int, ...],
    output_dir: str,
    design_sha256: str,
) -> dict[str, Any]:
    if not SOURCE_SEED_COUNT <= target_seed_count <= MAXIMUM_SEED_COUNT:
        raise MonteCarloContractError(
            "Continuation partition target must be between 160 and 640 seeds."
        )
    destination = Path(output_dir).resolve()
    states, member, manifest, _, _ = _continuation_partition_context(
        member_id, seed_bank, destination, design_sha256
    )
    partition_dir = destination / "partitions" / member_id
    failure_path = partition_dir / "last_failure.json"
    diagnostics, completed, active_slot = _restore_partition_diagnostics(
        partition_dir,
        manifest,
        seed_bank,
        design_sha256=design_sha256,
        member_id=member_id,
    )
    if completed < SOURCE_SEED_COUNT:
        raise MonteCarloContractError(
            f"Continuation partition {member_id} was not imported before execution."
        )
    if completed >= target_seed_count:
        _mark_failure_recovered(failure_path, completed)
        return {
            "weather_member_id": member_id,
            "completed_seed_count": completed,
            "run_count": len(diagnostics),
        }

    records = diagnostics.to_dict(orient="records")
    run_order = {
        run_id: index for index, run_id in enumerate(manifest["run_id"].astype(str))
    }
    for rank in range(completed + 1, target_seed_count + 1):
        if rank <= SOURCE_SEED_COUNT:
            raise MonteCarloContractError(
                "Continuation attempted to resimulate an imported seed."
            )
        seed = int(seed_bank[rank - 1])
        try:
            seed_manifest, seed_diagnostics, _ = execute_balanced_design(
                states, [member], [seed], (MODEL_SCENARIO_ID,)
            )
            expected = set(
                manifest.loc[
                    manifest["occupant_seed_rank"] == rank, "run_id"
                ].astype(str)
            )
            if (
                set(seed_manifest["run_id"].astype(str)) != expected
                or set(seed_diagnostics["run_id"].astype(str)) != expected
            ):
                raise MonteCarloContractError(
                    f"Executed continuation seed {rank} differs from its manifest."
                )
            records.extend(seed_diagnostics.to_dict(orient="records"))
            committed = pd.DataFrame.from_records(records)
            committed["_run_order"] = committed["run_id"].map(run_order)
            if committed["_run_order"].isna().any():
                raise MonteCarloContractError(
                    f"Foreign run ID in continuation partition {member_id}."
                )
            committed = committed.sort_values("_run_order", kind="stable").drop(
                columns="_run_order"
            )
            pointer = _commit_partition_diagnostics(
                partition_dir,
                committed,
                manifest,
                seed_bank,
                completed_seed_count=rank,
                active_slot=active_slot,
                design_sha256=design_sha256,
                member_id=member_id,
            )
            active_slot = str(pointer["active_diagnostics_slot"])
        except Exception as exc:
            _atomic_json(
                {
                    "status": "FAILED",
                    "design_sha256": design_sha256,
                    "weather_member_id": member_id,
                    "occupant_seed_rank": rank,
                    "occupant_seed": seed,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "failed_at_utc": _utc_now(),
                },
                failure_path,
            )
            raise
    _mark_failure_recovered(failure_path, target_seed_count)
    return {
        "weather_member_id": member_id,
        "completed_seed_count": target_seed_count,
        "run_count": len(records),
    }


def _advance_all_partitions(
    member_ids: Sequence[str],
    target_seed_count: int,
    seed_bank: tuple[int, ...],
    destination: Path,
    design_sha256: str,
    max_workers: int,
) -> None:
    print(
        f"[{_utc_now()}] advancing {len(member_ids)} weather partitions to "
        f"n={target_seed_count} with {max_workers} worker(s)",
        flush=True,
    )
    if max_workers == 1:
        for index, member_id in enumerate(member_ids, start=1):
            _advance_continuation_partition(
                member_id,
                target_seed_count,
                seed_bank,
                str(destination),
                design_sha256,
            )
            print(
                f"[{_utc_now()}] n={target_seed_count}: {index}/{len(member_ids)} "
                "weather partitions complete",
                flush=True,
            )
        return
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _advance_continuation_partition,
                member_id,
                target_seed_count,
                seed_bank,
                str(destination),
                design_sha256,
            ): member_id
            for member_id in member_ids
        }
        completed = 0
        for future in as_completed(futures):
            member_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise MonteCarloContractError(
                    f"Continuation weather partition {member_id} failed at "
                    f"n={target_seed_count}."
                ) from exc
            completed += 1
            print(
                f"[{_utc_now()}] n={target_seed_count}: "
                f"{completed}/{len(member_ids)} weather partitions complete",
                flush=True,
            )


def _collect_continuation_checkpoint(
    destination: Path,
    weather: pd.DataFrame,
    seed_bank: tuple[int, ...],
    design_sha256: str,
    target_seed_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, bool]:
    if target_seed_count not in CONTINUATION_CHECKPOINTS:
        raise MonteCarloContractError(
            "Only the prospectively declared n=320 and n=640 checkpoints may "
            "be collected."
        )
    manifests: list[pd.DataFrame] = []
    diagnostics_frames: list[pd.DataFrame] = []
    prefix = set(seed_bank[:target_seed_count])
    for member_id in weather["member_id"].astype(str):
        partition_dir = destination / "partitions" / member_id
        manifest = pd.read_csv(
            partition_dir / "run_manifest.csv", float_precision="round_trip"
        )
        diagnostics, completed, _ = _restore_partition_diagnostics(
            partition_dir,
            manifest,
            seed_bank,
            design_sha256=design_sha256,
            member_id=member_id,
        )
        if completed < target_seed_count:
            raise MonteCarloContractError(
                f"Continuation partition {member_id} ended at n={completed}, "
                f"below n={target_seed_count}."
            )
        manifests.append(
            manifest.loc[
                manifest["occupant_seed_rank"] <= target_seed_count
            ].copy()
        )
        diagnostics_frames.append(
            diagnostics.loc[diagnostics["occupant_seed"].isin(prefix)].copy()
        )
    manifest = pd.concat(manifests, ignore_index=True)
    diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
    expected_runs = len(PANEL_SPECS) * len(weather) * target_seed_count
    if (
        len(manifest) != expected_runs
        or len(diagnostics) != expected_runs
        or manifest["run_id"].duplicated().any()
        or diagnostics["run_id"].duplicated().any()
        or set(manifest["run_id"].astype(str))
        != set(diagnostics["run_id"].astype(str))
    ):
        raise MonteCarloContractError(
            f"The n={target_seed_count} continuation aggregate is incomplete "
            "or duplicated."
        )
    convergence = evaluate_seed_convergence(
        diagnostics,
        seed_order=seed_bank[:target_seed_count],
        rule=PROSPECTIVE_N320_N640_CONVERGENCE_RULE,
    )
    current = convergence.loc[convergence["seed_count"] == target_seed_count]
    panel_pass = bool(
        not current.empty
        and _strict_bool(
            current["criterion_pass"],
            label=f"n={target_seed_count} criterion pass",
        ).all()
    )
    evaluator_converged = bool(
        not current.empty
        and _strict_bool(
            current["panel_converged_at_checkpoint"],
            label=f"n={target_seed_count} panel convergence",
        ).all()
    )
    return manifest, diagnostics, convergence, panel_pass, evaluator_converged


def _verify_historical_decisions(
    convergence: pd.DataFrame,
    source_output_dir: Path,
) -> None:
    source = pd.read_csv(
        source_output_dir / "convergence_results.csv", float_precision="round_trip"
    )
    historical = convergence.loc[
        convergence["seed_count"] <= SOURCE_SEED_COUNT
    ].copy()
    ignored = {"occupant_seed_bank_count", "occupant_seed_bank_sha256"}
    columns = [column for column in source.columns if column not in ignored]
    sort_columns = [
        "archetype_id",
        "state_id",
        "climate_scenario_id",
        "model_scenario_id",
        "seed_count",
        "metric",
        "statistic",
    ]
    left = source[columns].sort_values(sort_columns, kind="stable").reset_index(
        drop=True
    )
    right = historical[columns].sort_values(
        sort_columns, kind="stable"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    except AssertionError as exc:
        raise MonteCarloContractError(
            "The n=320/n=640 evaluation retroactively changed source evidence."
        ) from exc


def _write_continuation_checkpoint(
    destination: Path,
    target_seed_count: int,
    manifest: pd.DataFrame,
    diagnostics: pd.DataFrame,
    convergence: pd.DataFrame,
    *,
    panel_pass: bool,
    selection_authorized: bool,
) -> dict[str, Any]:
    if target_seed_count == 320 and selection_authorized:
        raise MonteCarloContractError("n=320 is never a selectable checkpoint.")
    summary = _write_checkpoint(
        destination,
        target_seed_count,
        manifest,
        diagnostics,
        convergence,
        selection_authorized,
    )
    summary = {
        **summary,
        "panel_all_groups_statistics_pass": panel_pass,
        "selection_eligible_checkpoint": target_seed_count == MAXIMUM_SEED_COUNT,
        "selection_authorized": selection_authorized,
        "selection_requires_both_n320_and_n640_pass": True,
    }
    _atomic_json(
        summary,
        destination
        / "checkpoints"
        / f"n{target_seed_count:03d}"
        / "checkpoint_summary.json",
    )
    return summary


def _checkpoint_pass(convergence: pd.DataFrame, checkpoint: int) -> bool:
    rows = convergence.loc[convergence["seed_count"] == checkpoint]
    return bool(
        not rows.empty
        and _strict_bool(
            rows["criterion_pass"], label=f"terminal n={checkpoint} criterion pass"
        ).all()
    )


def _authenticate_checkpoint(
    destination: Path,
    contract: Mapping[str, Any],
    target_seed_count: int,
    *,
    expected_authorized: bool,
) -> tuple[bool, pd.DataFrame, dict[str, Any]]:
    checkpoint_dir = destination / "checkpoints" / f"n{target_seed_count:03d}"
    summary = _read_json(checkpoint_dir / "checkpoint_summary.json")
    artifact_sha256 = summary.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping) or set(artifact_sha256) != set(
        FINAL_CONTINUATION_ARTIFACTS
    ):
        raise MonteCarloContractError(
            f"n={target_seed_count} checkpoint lacks its exact artifact ledger."
        )
    for filename in FINAL_CONTINUATION_ARTIFACTS:
        path = checkpoint_dir / filename
        if not path.is_file() or _sha256_file(path) != str(artifact_sha256[filename]):
            raise MonteCarloContractError(
                f"n={target_seed_count} checkpoint artifact changed: {filename}."
            )
    manifest = pd.read_csv(
        checkpoint_dir / "run_manifest.csv", float_precision="round_trip"
    )
    diagnostics = pd.read_csv(
        checkpoint_dir / "run_diagnostics.csv", float_precision="round_trip"
    )
    evidence = pd.read_csv(
        checkpoint_dir / "convergence_results.csv", float_precision="round_trip"
    )
    expected_runs = (
        len(contract["panel"])
        * len(contract["weather_members"])
        * target_seed_count
    )
    if (
        len(manifest) != expected_runs
        or len(diagnostics) != expected_runs
        or manifest["run_id"].duplicated().any()
        or diagnostics["run_id"].duplicated().any()
        or set(manifest["run_id"].astype(str))
        != set(diagnostics["run_id"].astype(str))
    ):
        raise MonteCarloContractError(
            f"n={target_seed_count} checkpoint run identities are incomplete."
        )
    manifest_seed_by_run = {
        str(row.run_id): int(row.occupant_seed)
        for row in manifest[["run_id", "occupant_seed"]].itertuples(index=False)
    }
    if any(
        int(row.occupant_seed) != manifest_seed_by_run[str(row.run_id)]
        for row in diagnostics[["run_id", "occupant_seed"]].itertuples(index=False)
    ):
        raise MonteCarloContractError(
            f"n={target_seed_count} diagnostics changed a run-ID occupant-seed "
            "identity."
        )
    expected_seeds = tuple(int(value) for value in contract["occupant_seeds"])
    rank_by_seed = {seed: rank for rank, seed in enumerate(expected_seeds, start=1)}
    if any(
        int(row.occupant_seed_rank) != rank_by_seed[int(row.occupant_seed)]
        for row in manifest[
            ["occupant_seed", "occupant_seed_rank"]
        ].itertuples(index=False)
    ):
        raise MonteCarloContractError(
            f"n={target_seed_count} manifest changed an occupant-seed rank."
        )
    recomputed = evaluate_seed_convergence(
        diagnostics,
        seed_order=expected_seeds[:target_seed_count],
        rule=PROSPECTIVE_N320_N640_CONVERGENCE_RULE,
    )
    if _frame_csv_sha256(recomputed) != _frame_csv_sha256(evidence):
        raise MonteCarloContractError(
            f"n={target_seed_count} checkpoint evidence cannot be reproduced."
        )
    _verify_historical_decisions(
        evidence, _source_dir_from_contract(contract)
    )
    panel_pass = _checkpoint_pass(evidence, target_seed_count)
    if (
        int(summary.get("seed_count", -1)) != target_seed_count
        or int(summary.get("run_count", -1)) != expected_runs
        or bool(summary.get("panel_all_groups_statistics_pass")) != panel_pass
        or bool(summary.get("selection_authorized")) != expected_authorized
        or bool(summary.get("panel_converged_at_checkpoint"))
        != expected_authorized
        or summary.get("status")
        != ("CONVERGED" if expected_authorized else "NOT_YET_CONVERGED")
        or bool(summary.get("selection_eligible_checkpoint"))
        != (target_seed_count == MAXIMUM_SEED_COUNT)
        or not bool(summary.get("selection_requires_both_n320_and_n640_pass"))
    ):
        raise MonteCarloContractError(
            f"n={target_seed_count} checkpoint summary contradicts its evidence."
        )
    return panel_pass, evidence, summary


def _validate_terminal_summary_semantics(
    destination: Path,
    contract: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> bool:
    """Authenticate terminal artifacts and the explicit two-new-pass decision."""

    source_receipt = _validate_completed_source(_source_dir_from_contract(contract))
    if _json_ready(source_receipt) != contract.get("source_experiment"):
        raise MonteCarloContractError(
            "The authenticated n=160 source no longer matches the frozen "
            "continuation lineage receipt."
        )
    if summary.get("status") not in TERMINAL_STATUSES:
        raise MonteCarloContractError("Continuation summary is not terminal.")
    if summary.get("design_sha256") != contract.get("design_sha256"):
        raise MonteCarloContractError(
            "Terminal continuation summary belongs to a different design."
        )
    expected_rule = _rule_payload(PROSPECTIVE_N320_N640_CONVERGENCE_RULE)
    if (
        contract.get("convergence_rule") != expected_rule
        or summary.get("convergence_rule") != expected_rule
    ):
        raise MonteCarloContractError(
            "Terminal continuation does not use the frozen n=320/n=640 rule."
        )

    artifact_sha256 = summary.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping) or set(artifact_sha256) != set(
        FINAL_CONTINUATION_ARTIFACTS
    ):
        raise MonteCarloContractError(
            "Terminal continuation must contain the exact three-artifact ledger."
        )
    for filename in FINAL_CONTINUATION_ARTIFACTS:
        path = destination / filename
        if not path.is_file() or _sha256_file(path) != str(artifact_sha256[filename]):
            raise MonteCarloContractError(
                f"Completed continuation artifact changed: {filename}."
            )

    n320_pass, _, checkpoint_320_summary = _authenticate_checkpoint(
        destination, contract, 320, expected_authorized=False
    )
    n640_dir = destination / "checkpoints/n640"
    n640_evidence = pd.read_csv(
        n640_dir / "convergence_results.csv", float_precision="round_trip"
    )
    n640_pass = _checkpoint_pass(n640_evidence, 640)
    selected = n320_pass and n640_pass
    authenticated_n640_pass, evidence, checkpoint_640_summary = (
        _authenticate_checkpoint(
            destination, contract, 640, expected_authorized=selected
        )
    )
    if authenticated_n640_pass != n640_pass:
        raise MonteCarloContractError("n=640 checkpoint pass authentication changed.")
    for filename in FINAL_CONTINUATION_ARTIFACTS:
        if _sha256_file(destination / filename) != _sha256_file(n640_dir / filename):
            raise MonteCarloContractError(
                f"Terminal root is not the exact n=640 checkpoint: {filename}."
            )

    n320_rows = evidence.loc[evidence["seed_count"] == 320]
    if bool(
        _strict_bool(
            n320_rows["panel_converged_at_checkpoint"],
            label="terminal n=320 panel convergence",
        ).any()
    ):
        raise MonteCarloContractError(
            "n=320 cannot be a converged/selected checkpoint in this continuation."
        )
    n640_rows = evidence.loc[evidence["seed_count"] == 640]
    evaluator_selected = bool(
        not n640_rows.empty
        and _strict_bool(
            n640_rows["panel_converged_at_checkpoint"],
            label="terminal n=640 panel convergence",
        ).all()
    )
    if evaluator_selected != selected:
        raise MonteCarloContractError(
            "The two-new-pass decision conflicts with independently recomputed evidence."
        )

    expected_run_count = (
        len(contract["panel"])
        * len(contract["weather_members"])
        * MAXIMUM_SEED_COUNT
    )
    if (
        int(summary.get("run_count", -1)) != expected_run_count
        or int(summary.get("imported_seed_count", -1)) != SOURCE_SEED_COUNT
        or int(summary.get("new_seed_count", -1))
        != MAXIMUM_SEED_COUNT - SOURCE_SEED_COUNT
        or int(summary.get("panel_cell_count", -1)) != len(contract["panel"])
        or int(summary.get("weather_member_count", -1))
        != len(contract["weather_members"])
        or summary.get("source_design_sha256")
        != contract["source_experiment"]["source_design_sha256"]
        or summary.get("checkpoint_n320") != checkpoint_320_summary
        or summary.get("latest_checkpoint") != checkpoint_640_summary
    ):
        raise MonteCarloContractError(
            "Terminal continuation counts, lineage, or checkpoint ledgers "
            "contradict authenticated artifacts."
        )

    decisions = summary.get("new_checkpoint_decisions")
    if not isinstance(decisions, Mapping) or decisions != {
        "n320": {
            "panel_all_groups_statistics_pass": n320_pass,
            "selection_permitted": False,
        },
        "n640": {
            "panel_all_groups_statistics_pass": n640_pass,
            "both_new_checkpoints_pass": selected,
            "selection_permitted": True,
        },
    }:
        raise MonteCarloContractError(
            "Terminal continuation decision ledger contradicts its evidence."
        )
    try:
        evaluated = int(summary["evaluated_seed_count"])
        selected_count = summary.get("selected_seed_count")
        selected_seeds = summary.get("selected_occupant_seeds")
        first_checkpoint = summary.get("first_panel_converged_checkpoint")
    except (KeyError, TypeError, ValueError) as exc:
        raise MonteCarloContractError(
            "Terminal continuation contains invalid decision fields."
        ) from exc
    if evaluated != MAXIMUM_SEED_COUNT:
        raise MonteCarloContractError(
            "Terminal continuation did not evaluate exactly 640 seeds."
        )
    if selected:
        try:
            selected_count_value = int(selected_count)
            selected_seed_values = tuple(int(value) for value in selected_seeds)
            first_checkpoint_value = int(first_checkpoint)
        except (TypeError, ValueError) as exc:
            raise MonteCarloContractError(
                "Converged continuation has invalid selected seeds."
            ) from exc
        if (
            summary.get("status") != "CONVERGED"
            or selected_count_value != MAXIMUM_SEED_COUNT
            or first_checkpoint_value != MAXIMUM_SEED_COUNT
            or selected_seed_values
            != tuple(int(value) for value in contract["occupant_seeds"])
        ):
            raise MonteCarloContractError(
                "Terminal continuation selection is not the exact 640-seed prefix."
            )
    elif (
        summary.get("status") != "NOT_CONVERGED_AT_N640"
        or selected_count is not None
        or selected_seeds is not None
        or first_checkpoint is not None
    ):
        raise MonteCarloContractError(
            "Non-converged n=640 continuation contains a production selection."
        )
    return selected


def _run_convergence_continuation_unlocked(
    output_dir: str | Path = DEFAULT_CONVERGENCE_CONTINUATION_OUTPUT_DIR,
    *,
    source_output_dir: str | Path = DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
    max_workers: int | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    workers = min(4, os.cpu_count() or 1) if max_workers is None else max_workers
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 8
    ):
        raise MonteCarloContractError(
            "Convergence-continuation max_workers must be an integer from 1 to 8."
        )
    contract = prepare_convergence_continuation(
        destination, source_output_dir=source_output_dir
    )
    weather = _weather_selection()
    member_ids = tuple(weather["member_id"].astype(str))
    seed_bank = tuple(int(value) for value in contract["occupant_seeds"])
    summary_path = destination / "convergence_continuation_summary.json"
    if summary_path.exists():
        existing = _read_json(summary_path)
        if existing.get("design_sha256") not in {None, contract["design_sha256"]}:
            raise MonteCarloContractError(
                "Existing continuation summary belongs to a different design."
            )
        if existing.get("status") in TERMINAL_STATUSES:
            _validate_terminal_summary_semantics(destination, contract, existing)
            return existing
        started_at = str(existing.get("started_at_utc", _utc_now()))
    else:
        started_at = _utc_now()

    def write_progress(phase: str, active_checkpoint: int) -> None:
        _atomic_json(
            {
                "status": "IN_PROGRESS",
                "phase": phase,
                "design_sha256": contract["design_sha256"],
                "started_at_utc": started_at,
                "imported_seed_count": SOURCE_SEED_COUNT,
                "active_checkpoint": active_checkpoint,
                "max_workers": workers,
                "updated_at_utc": _utc_now(),
            },
            summary_path,
        )

    write_progress("AUTHENTICATING_N160_IMPORT", 320)
    _bootstrap_all_partitions(
        member_ids,
        seed_bank,
        destination,
        str(contract["design_sha256"]),
    )
    _verify_imported_aggregate(destination, weather, seed_bank, contract)

    write_progress("SIMULATING_SEEDS_161_TO_320", 320)
    _advance_all_partitions(
        member_ids,
        320,
        seed_bank,
        destination,
        str(contract["design_sha256"]),
        workers,
    )
    manifest_320, diagnostics_320, convergence_320, n320_pass, evaluator_320 = (
        _collect_continuation_checkpoint(
            destination,
            weather,
            seed_bank,
            str(contract["design_sha256"]),
            320,
        )
    )
    _verify_historical_decisions(
        convergence_320, Path(source_output_dir).resolve()
    )
    if evaluator_320:
        raise MonteCarloContractError(
            "n=320 unexpectedly satisfies the two-pass rule despite the "
            "authenticated failed n=160 source."
        )
    checkpoint_320 = _write_continuation_checkpoint(
        destination,
        320,
        manifest_320,
        diagnostics_320,
        convergence_320,
        panel_pass=n320_pass,
        selection_authorized=False,
    )
    print(
        f"[{_utc_now()}] continuation checkpoint n=320: "
        f"{'first new pass; confirmation still required' if n320_pass else 'failed'}",
        flush=True,
    )

    # n=640 is always evaluated.  This is deliberately not an adaptive stop at
    # n=320: two new prospective expansions are part of one frozen protocol.
    write_progress("SIMULATING_SEEDS_321_TO_640", 640)
    _advance_all_partitions(
        member_ids,
        640,
        seed_bank,
        destination,
        str(contract["design_sha256"]),
        workers,
    )
    manifest, diagnostics, convergence, n640_pass, evaluator_640 = (
        _collect_continuation_checkpoint(
            destination,
            weather,
            seed_bank,
            str(contract["design_sha256"]),
            640,
        )
    )
    _verify_historical_decisions(convergence, Path(source_output_dir).resolve())
    selected = n320_pass and n640_pass
    if evaluator_640 != selected:
        raise MonteCarloContractError(
            "The evaluator and explicit n=320+n=640 two-pass decision disagree."
        )
    checkpoint_640 = _write_continuation_checkpoint(
        destination,
        640,
        manifest,
        diagnostics,
        convergence,
        panel_pass=n640_pass,
        selection_authorized=selected,
    )
    checkpoint_dir = destination / "checkpoints/n640"
    for filename in FINAL_CONTINUATION_ARTIFACTS:
        _copy_atomic(checkpoint_dir / filename, destination / filename)

    final = {
        "status": "CONVERGED" if selected else "NOT_CONVERGED_AT_N640",
        "scope": "prospective n=320/n=640 occupant-seed continuation",
        "design_sha256": contract["design_sha256"],
        "source_design_sha256": contract["source_experiment"][
            "source_design_sha256"
        ],
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "imported_seed_count": SOURCE_SEED_COUNT,
        "new_seed_count": MAXIMUM_SEED_COUNT - SOURCE_SEED_COUNT,
        "evaluated_seed_count": MAXIMUM_SEED_COUNT,
        "selected_seed_count": MAXIMUM_SEED_COUNT if selected else None,
        "selected_occupant_seeds": list(seed_bank) if selected else None,
        "first_panel_converged_checkpoint": (
            MAXIMUM_SEED_COUNT if selected else None
        ),
        "panel_cell_count": len(PANEL_SPECS),
        "weather_member_count": len(weather),
        "run_count": len(diagnostics),
        "convergence_rule": _rule_payload(
            PROSPECTIVE_N320_N640_CONVERGENCE_RULE
        ),
        "new_checkpoint_decisions": {
            "n320": {
                "panel_all_groups_statistics_pass": n320_pass,
                "selection_permitted": False,
            },
            "n640": {
                "panel_all_groups_statistics_pass": n640_pass,
                "both_new_checkpoints_pass": selected,
                "selection_permitted": True,
            },
        },
        "checkpoint_n320": checkpoint_320,
        "latest_checkpoint": checkpoint_640,
        "artifact_sha256": {
            filename: _sha256_file(destination / filename)
            for filename in FINAL_CONTINUATION_ARTIFACTS
        },
        "production_interpretation": (
            "The exact 640-seed prefix is authorized for full-stock execution "
            "because both prospectively declared new checkpoints passed."
            if selected
            else "No seed count was selected because n=320 and n=640 did not "
            "both pass. No authoritative stock run is authorized."
        ),
    }
    _atomic_json(final, summary_path)
    _validate_terminal_summary_semantics(destination, contract, final)
    return final


def run_convergence_continuation(
    output_dir: str | Path = DEFAULT_CONVERGENCE_CONTINUATION_OUTPUT_DIR,
    *,
    source_output_dir: str | Path = DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run or resume the separately contracted n=320/n=640 continuation."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_execution_lock(destination)
    try:
        return _run_convergence_continuation_unlocked(
            destination,
            source_output_dir=source_output_dir,
            max_workers=max_workers,
        )
    except Exception as exc:
        summary_path = destination / "convergence_continuation_summary.json"
        try:
            existing = _read_json(summary_path) if summary_path.exists() else {}
            contract_path = destination / "convergence_continuation_contract.json"
            contract = _read_json(contract_path) if contract_path.exists() else {}
            _atomic_json(
                {
                    **existing,
                    "status": "FAILED",
                    "design_sha256": existing.get("design_sha256")
                    or contract.get("design_sha256"),
                    "coordinator_failure": {
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "failed_at_utc": _utc_now(),
                    },
                    "updated_at_utc": _utc_now(),
                },
                summary_path,
            )
        except Exception:
            pass
        raise
    finally:
        _release_execution_lock(lock_path)


def convergence_continuation_status(
    output_dir: str | Path = DEFAULT_CONVERGENCE_CONTINUATION_OUTPUT_DIR,
) -> dict[str, Any]:
    """Return a read-only progress snapshot; authenticate terminal outputs."""

    destination = Path(output_dir).resolve()
    contract_path = destination / "convergence_continuation_contract.json"
    if not contract_path.exists():
        return {"status": "NOT_PREPARED", "output_dir": str(destination)}
    contract = _read_json(contract_path)
    version = contract.get("convergence_continuation_contract_version")
    if version != CONVERGENCE_CONTINUATION_CONTRACT_VERSION:
        return {
            "status": "STALE_CONTRACT",
            "output_dir": str(destination),
            "persisted_contract_version": version,
            "required_contract_version": CONVERGENCE_CONTINUATION_CONTRACT_VERSION,
            "design_sha256": contract.get("design_sha256"),
        }
    unsigned = {key: value for key, value in contract.items() if key != "design_sha256"}
    if canonical_sha256(unsigned) != contract.get("design_sha256"):
        return {
            "status": "CORRUPT_CONTRACT",
            "output_dir": str(destination),
            "design_sha256": contract.get("design_sha256"),
        }
    summary_path = destination / "convergence_continuation_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {
        "status": "PREPARED"
    }
    terminal_authenticated = False
    if summary.get("status") in TERMINAL_STATUSES:
        _validate_terminal_summary_semantics(destination, contract, summary)
        terminal_authenticated = True

    counts: list[int] = []
    failures = 0
    partitions_dir = destination / "partitions"
    if partitions_dir.exists():
        for partition_dir in partitions_dir.iterdir():
            if not partition_dir.is_dir():
                continue
            progress_path = partition_dir / "progress.json"
            if progress_path.exists():
                counts.append(int(_read_json(progress_path)["completed_seed_count"]))
            failure_path = partition_dir / "last_failure.json"
            if (
                failure_path.exists()
                and _read_json(failure_path).get("status") == "FAILED"
            ):
                failures += 1
    return {
        "status": summary.get("status", "UNKNOWN"),
        "phase": summary.get("phase"),
        "output_dir": str(destination),
        "design_sha256": contract.get("design_sha256"),
        "source_design_sha256": contract["source_experiment"][
            "source_design_sha256"
        ],
        "active_checkpoint": summary.get("active_checkpoint"),
        "selected_seed_count": summary.get("selected_seed_count"),
        "weather_partitions_started": len(counts),
        "weather_partition_seed_count_histogram": {
            str(value): counts.count(value) for value in sorted(set(counts))
        },
        "active_failure_count": failures,
        "progress_validation": (
            "FULL_TERMINAL_AUTHENTICATION"
            if terminal_authenticated
            else "OBSERVATIONAL_POINTER_COUNTS_ONLY"
        ),
        "terminal_artifacts_authenticated": terminal_authenticated,
        "execution_lock_present": (destination / "execution.lock").exists(),
        "updated_at_utc": summary.get("updated_at_utc"),
    }


def load_convergence_continuation_selection(
    output_dir: str | Path = DEFAULT_CONVERGENCE_CONTINUATION_OUTPUT_DIR,
) -> ConvergenceSelection:
    """Load the exact 640-seed prefix only after full terminal authentication."""

    destination = Path(output_dir).resolve()
    contract = _read_json(destination / "convergence_continuation_contract.json")
    if (
        contract.get("convergence_continuation_contract_version")
        != CONVERGENCE_CONTINUATION_CONTRACT_VERSION
    ):
        raise MonteCarloContractError(
            "Cannot load production seeds from a stale continuation contract."
        )
    unsigned = {key: value for key, value in contract.items() if key != "design_sha256"}
    if canonical_sha256(unsigned) != contract.get("design_sha256"):
        raise MonteCarloContractError(
            "The continuation design checksum cannot be reconstructed."
        )
    summary = _read_json(destination / "convergence_continuation_summary.json")
    if summary.get("status") != "CONVERGED":
        raise MonteCarloContractError(
            "Production occupant seeds are available only after joint n=320 and "
            "n=640 convergence."
        )
    if not _validate_terminal_summary_semantics(destination, contract, summary):
        raise MonteCarloContractError(
            "Continuation evidence does not authorize an n=640 selection."
        )
    seeds = tuple(int(value) for value in summary["selected_occupant_seeds"])
    if (
        len(seeds) != MAXIMUM_SEED_COUNT
        or seeds != tuple(int(value) for value in contract["occupant_seeds"])
        or ordered_seed_bank_sha256(seeds)
        != str(contract["occupant_seed_bank_sha256"])
    ):
        raise MonteCarloContractError(
            "Continuation summary does not select the exact frozen 640-seed bank."
        )
    evidence_path = destination / "convergence_results.csv"
    evidence_sha256 = str(
        summary["artifact_sha256"]["convergence_results.csv"]
    )
    if _sha256_file(evidence_path) != evidence_sha256:
        raise MonteCarloContractError(
            "Continuation convergence evidence differs from its selected checksum."
        )
    return ConvergenceSelection(
        occupant_seeds=seeds,
        convergence_results_path=evidence_path,
        convergence_results_sha256=evidence_sha256,
        design_sha256=str(contract["design_sha256"]),
        convergence_rule=PROSPECTIVE_N320_N640_CONVERGENCE_RULE,
    )
