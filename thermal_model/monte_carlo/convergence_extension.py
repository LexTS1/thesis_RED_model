"""Prospective n=160 confirmation of the completed Gate-5 seed experiment.

The original n=80 experiment is an immutable source.  This module authenticates
and copies its exact seed prefix into a separate execution contract, appends
seeds 81--160, and re-evaluates the unchanged stopping rule over all nested
checkpoints.  It never edits the source experiment.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .contracts import MonteCarloContractError, canonical_sha256
from .convergence_runner import (
    CONVERGENCE_EXECUTION_CONTRACT_VERSION,
    DEFAULT_CONVERGENCE_OUTPUT_DIR,
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
    _restore_partition_diagnostics,
    _rule_payload,
    _sha256_file,
    _utc_now,
    _weather_selection,
    _write_checkpoint,
    load_convergence_panel,
)
from .design import (
    ConvergenceRule,
    PROSPECTIVE_N160_CONVERGENCE_RULE,
    build_balanced_manifest,
    evaluate_seed_convergence,
    make_seed_bank,
    ordered_seed_bank_sha256,
)
from .runner import execute_balanced_design
from .weather import load_weather_member


DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR = (
    PROJECT_ROOT / "thermal_model/data/monte_carlo/convergence_panel_n160_extension"
)
CONVERGENCE_EXTENSION_CONTRACT_VERSION = "gate5_convergence_n160_extension_v1"
CONVERGENCE_EXTENSION_PARTITION_VERSION = (
    "gate5_convergence_n160_extension_partition_v1"
)
BASE_SEED_COUNT = 80
EXTENSION_SEED_COUNT = 160
EXTENSION_CHECKPOINT = 160
PROSPECTIVE_DECLARATION_DATE = "2026-08-09"
FINAL_EXTENSION_ARTIFACTS = (
    "run_manifest.csv",
    "run_diagnostics.csv",
    "convergence_results.csv",
)


def _strict_bool(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise MonteCarloContractError(f"{label} must contain strict booleans.")
    return normalized.eq("true")


def _relative_project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError as exc:
        raise MonteCarloContractError(
            "The convergence source must be inside the thesis project root."
        ) from exc


def _validate_completed_base(
    base_output_dir: str | Path = DEFAULT_CONVERGENCE_OUTPUT_DIR,
) -> dict[str, Any]:
    """Authenticate the terminal n=80 experiment and return a frozen receipt."""

    base = Path(base_output_dir).resolve()
    contract_path = base / "convergence_execution_contract.json"
    summary_path = base / "convergence_summary.json"
    checkpoint_summary_path = base / "checkpoints/n080/checkpoint_summary.json"
    contract = _read_json(contract_path)
    summary = _read_json(summary_path)
    checkpoint_summary = _read_json(checkpoint_summary_path)

    if (
        contract.get("convergence_execution_contract_version")
        != CONVERGENCE_EXECUTION_CONTRACT_VERSION
    ):
        raise MonteCarloContractError(
            "The source convergence experiment uses an unsupported contract version."
        )
    unsigned_contract = {
        key: value for key, value in contract.items() if key != "design_sha256"
    }
    if canonical_sha256(unsigned_contract) != contract.get("design_sha256"):
        raise MonteCarloContractError(
            "The source convergence design checksum cannot be reconstructed."
        )
    if (
        summary.get("status") != "NOT_CONVERGED_AT_N80"
        or int(summary.get("evaluated_seed_count", -1)) != BASE_SEED_COUNT
        or summary.get("selected_seed_count") is not None
        or summary.get("selected_occupant_seeds") is not None
        or summary.get("design_sha256") != contract.get("design_sha256")
    ):
        raise MonteCarloContractError(
            "The source must be the completed, non-selected n=80 experiment."
        )
    if (
        checkpoint_summary.get("status") != "NOT_YET_CONVERGED"
        or int(checkpoint_summary.get("seed_count", -1)) != BASE_SEED_COUNT
        or bool(checkpoint_summary.get("panel_converged_at_checkpoint"))
    ):
        raise MonteCarloContractError(
            "The source n=80 checkpoint does not record the required first pass."
        )

    if contract.get("convergence_rule") != _rule_payload(ConvergenceRule()):
        raise MonteCarloContractError(
            "The source convergence rule differs from the original n=80 protocol."
        )
    source_seeds = tuple(int(value) for value in contract.get("occupant_seeds", ()))
    generated = make_seed_bank(EXTENSION_SEED_COUNT, master_seed=MASTER_SEED)
    if (
        int(contract.get("master_seed", -1)) != MASTER_SEED
        or len(source_seeds) != BASE_SEED_COUNT
        or source_seeds != generated[:BASE_SEED_COUNT]
    ):
        raise MonteCarloContractError(
            "The source seed bank is not the exact prefix of the n=160 bank."
        )

    artifact_sha256 = summary.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping):
        raise MonteCarloContractError("The source summary lacks artifact checksums.")
    required_root_files = (
        "run_manifest.csv",
        "run_diagnostics.csv",
        "convergence_results.csv",
    )
    for filename in required_root_files:
        path = base / filename
        if (
            not path.is_file()
            or _sha256_file(path) != str(artifact_sha256.get(filename, ""))
        ):
            raise MonteCarloContractError(
                f"The source convergence artifact changed: {filename}."
            )

    manifest = pd.read_csv(base / "run_manifest.csv", float_precision="round_trip")
    diagnostics = pd.read_csv(
        base / "run_diagnostics.csv", float_precision="round_trip"
    )
    expected_runs = len(PANEL_SPECS) * 54 * BASE_SEED_COUNT
    if (
        len(manifest) != expected_runs
        or len(diagnostics) != expected_runs
        or manifest["run_id"].duplicated().any()
        or diagnostics["run_id"].duplicated().any()
        or set(manifest["run_id"].astype(str))
        != set(diagnostics["run_id"].astype(str))
    ):
        raise MonteCarloContractError(
            "The source root manifest/diagnostics are incomplete or duplicated."
        )
    recomputed = evaluate_seed_convergence(
        diagnostics,
        seed_order=source_seeds,
        rule=ConvergenceRule(),
    )
    if _frame_csv_sha256(recomputed) != str(
        artifact_sha256["convergence_results.csv"]
    ):
        raise MonteCarloContractError(
            "The source convergence decision cannot be reproduced from diagnostics."
        )
    at_40 = recomputed.loc[recomputed["seed_count"] == 40]
    at_80 = recomputed.loc[recomputed["seed_count"] == BASE_SEED_COUNT]
    at_40_pass = _strict_bool(at_40["criterion_pass"], label="source n=40 pass")
    at_80_pass = _strict_bool(at_80["criterion_pass"], label="source n=80 pass")
    if (
        at_40.empty
        or bool(at_40_pass.all())
        or at_80.empty
        or not bool(at_80_pass.all())
        or not bool(
            _strict_bool(
                at_80["panel_all_groups_statistics_pass"],
                label="source n=80 panel pass",
            ).all()
        )
        or not at_80["panel_consecutive_passing_expansions"].eq(1).all()
        or bool(
            _strict_bool(
                at_80["panel_converged_at_checkpoint"],
                label="source n=80 convergence",
            ).any()
        )
    ):
        raise MonteCarloContractError(
            "The source evidence is not the expected one-pass-at-n=80 outcome."
        )

    return {
        "base_output_relative_path": _relative_project_path(base),
        "base_convergence_execution_contract_version": (
            CONVERGENCE_EXECUTION_CONTRACT_VERSION
        ),
        "base_design_sha256": str(contract["design_sha256"]),
        "base_contract_file_sha256": _sha256_file(contract_path),
        "base_summary_file_sha256": _sha256_file(summary_path),
        "base_n080_checkpoint_summary_sha256": _sha256_file(
            checkpoint_summary_path
        ),
        "base_terminal_status": str(summary["status"]),
        "base_completed_at_utc": str(summary["completed_at_utc"]),
        "base_evaluated_seed_count": BASE_SEED_COUNT,
        "base_run_count": expected_runs,
        "base_rule": _rule_payload(ConvergenceRule()),
        "base_rule_sha256": canonical_sha256(_rule_payload(ConvergenceRule())),
        "base_seed_bank_sha256": ordered_seed_bank_sha256(source_seeds),
        "base_artifact_sha256": {
            filename: str(artifact_sha256[filename])
            for filename in required_root_files
        },
        "base_n080_all_criteria_pass": True,
        "base_n080_panel_consecutive_passing_expansions": 1,
    }


def prepare_convergence_extension(
    output_dir: str | Path = DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
    *,
    base_output_dir: str | Path = DEFAULT_CONVERGENCE_OUTPUT_DIR,
) -> dict[str, Any]:
    """Freeze the append-only n=160 design without starting simulations."""

    destination = Path(output_dir).resolve()
    source = Path(base_output_dir).resolve()
    if (
        destination == source
        or source in destination.parents
        or destination in source.parents
    ):
        raise MonteCarloContractError(
            "The prospective extension and source experiment must be separate, "
            "non-overlapping directory trees."
        )
    base_receipt = _validate_completed_base(base_output_dir)
    states, panel = load_convergence_panel()
    weather = _weather_selection()
    seed_bank = make_seed_bank(EXTENSION_SEED_COUNT, master_seed=MASTER_SEED)
    selection_sha256 = {
        "panel_selection.csv": _frame_csv_sha256(panel),
        "weather_selection.csv": _frame_csv_sha256(weather),
    }

    base_contract = _read_json(
        Path(base_output_dir).resolve() / "convergence_execution_contract.json"
    )
    if (
        base_contract.get("panel") != panel.to_dict(orient="records")
        or base_contract.get("weather_members") != weather.to_dict(orient="records")
        or base_contract.get("selection_artifact_sha256") != selection_sha256
    ):
        raise MonteCarloContractError(
            "The current panel/weather inventory differs from the source experiment."
        )

    extension_rule = _rule_payload(PROSPECTIVE_N160_CONVERGENCE_RULE)
    original_rule = _rule_payload(ConvergenceRule())
    if (
        extension_rule["checkpoints"][:-1] != original_rule["checkpoints"]
        or extension_rule["checkpoints"][-1] != EXTENSION_CHECKPOINT
        or {
            key: extension_rule[key]
            for key in extension_rule
            if key != "checkpoints"
        }
        != {
            key: original_rule[key]
            for key in original_rule
            if key != "checkpoints"
        }
    ):
        raise MonteCarloContractError(
            "The prospective rule must only append checkpoint n=160."
        )

    payload = {
        "convergence_extension_contract_version": (
            CONVERGENCE_EXTENSION_CONTRACT_VERSION
        ),
        "partition_checkpoint_protocol_version": (
            PARTITION_CHECKPOINT_PROTOCOL_VERSION
        ),
        "prospective_declaration_date": PROSPECTIVE_DECLARATION_DATE,
        "methodological_rationale": (
            "The original n=80 experiment ended with one complete-panel pass; "
            "n=160 is declared prospectively to test the unchanged requirement "
            "for two consecutive passing expansions."
        ),
        "base_experiment": base_receipt,
        "model_contract_version": base_contract["model_contract_version"],
        "central_thermal_assumptions_sha256": base_contract[
            "central_thermal_assumptions_sha256"
        ],
        "behaviour_assumptions_sha256": base_contract[
            "behaviour_assumptions_sha256"
        ],
        "occupant_distribution_sha256": base_contract[
            "occupant_distribution_sha256"
        ],
        "model_scenario": base_contract["model_scenario"],
        "model_scenario_sha256": base_contract["model_scenario_sha256"],
        "panel": panel.to_dict(orient="records"),
        "weather_members": weather.to_dict(orient="records"),
        "selection_artifact_sha256": selection_sha256,
        "master_seed": MASTER_SEED,
        "occupant_seeds": list(seed_bank),
        "occupant_seed_bank_sha256": ordered_seed_bank_sha256(seed_bank),
        "imported_seed_count": BASE_SEED_COUNT,
        "imported_seed_prefix_sha256": ordered_seed_bank_sha256(
            seed_bank[:BASE_SEED_COUNT]
        ),
        "new_seed_count": EXTENSION_SEED_COUNT - BASE_SEED_COUNT,
        "convergence_rule": extension_rule,
        "extension_checkpoint": EXTENSION_CHECKPOINT,
        "expected_total_run_count_at_n160": (
            len(states) * len(weather) * EXTENSION_SEED_COUNT
        ),
        "expected_imported_run_count": (
            len(states) * len(weather) * BASE_SEED_COUNT
        ),
        "expected_new_run_count": (
            len(states) * len(weather) * (EXTENSION_SEED_COUNT - BASE_SEED_COUNT)
        ),
        "source_mutation_permitted": False,
        "intermediate_stopping_permitted": False,
    }
    design_sha256 = canonical_sha256(payload)
    contract = {**payload, "design_sha256": design_sha256}

    contract_path = destination / "convergence_extension_contract.json"
    panel_path = destination / "panel_selection.csv"
    weather_path = destination / "weather_selection.csv"
    if contract_path.exists():
        if _read_json(contract_path) != _json_ready(contract):
            raise MonteCarloContractError(
                "Convergence-extension output belongs to a different design."
            )
    else:
        _atomic_csv(panel, panel_path)
        _atomic_csv(weather, weather_path)
        _atomic_json(contract, contract_path)
    for path in (panel_path, weather_path):
        expected = selection_sha256[path.name]
        if not path.is_file() or _sha256_file(path) != expected:
            raise MonteCarloContractError(
                f"Convergence-extension selection artifact changed: {path.name}."
            )
    return contract


def _base_dir_from_contract(contract: Mapping[str, Any]) -> Path:
    relative = str(contract["base_experiment"]["base_output_relative_path"])
    resolved = (PROJECT_ROOT / relative).resolve()
    if resolved == Path(DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR).resolve():
        raise MonteCarloContractError("Extension source path resolves to its destination.")
    return resolved


def _extension_partition_context(
    member_id: str,
    seed_bank: tuple[int, ...],
    destination: Path,
    design_sha256: str,
) -> tuple[tuple[Any, ...], Any, pd.DataFrame, dict[str, Any], Path]:
    contract = _read_json(destination / "convergence_extension_contract.json")
    unsigned_contract = {
        key: value for key, value in contract.items() if key != "design_sha256"
    }
    if (
        contract.get("design_sha256") != design_sha256
        or canonical_sha256(unsigned_contract) != design_sha256
    ):
        raise MonteCarloContractError(
            f"Extension design changed while preparing {member_id}."
        )
    states, _ = load_convergence_panel()
    member = load_weather_member(member_id)
    manifest = build_balanced_manifest(
        states, [member], seed_bank, (MODEL_SCENARIO_ID,)
    )
    base_dir = _base_dir_from_contract(contract)
    base_partition = base_dir / "partitions" / member_id
    base_manifest_path = base_partition / "run_manifest.csv"
    base_partition_contract_path = base_partition / "partition_contract.json"
    base_progress_path = base_partition / "progress.json"
    base_manifest = pd.read_csv(base_manifest_path, float_precision="round_trip")
    base_partition_contract = _read_json(base_partition_contract_path)
    base_progress = _read_json(base_progress_path)
    base_design = str(contract["base_experiment"]["base_design_sha256"])
    if (
        base_partition_contract.get("design_sha256") != base_design
        or base_partition_contract.get("weather_member_id") != member_id
        or _sha256_file(base_manifest_path)
        != str(base_partition_contract.get("run_manifest_sha256", ""))
    ):
        raise MonteCarloContractError(
            f"Source partition contract changed for {member_id}."
        )
    prefix_manifest = manifest.loc[
        manifest["occupant_seed_rank"] <= BASE_SEED_COUNT,
        base_manifest.columns,
    ]
    if _frame_csv_sha256(prefix_manifest) != _sha256_file(base_manifest_path):
        raise MonteCarloContractError(
            f"Extended manifest does not preserve source run identities for {member_id}."
        )
    base_diagnostics, base_completed, _ = _restore_partition_diagnostics(
        base_partition,
        base_manifest,
        seed_bank[:BASE_SEED_COUNT],
        design_sha256=base_design,
        member_id=member_id,
    )
    if base_completed != BASE_SEED_COUNT:
        raise MonteCarloContractError(
            f"Source partition {member_id} is not complete at n=80."
        )
    active_base_slot = str(base_progress["active_diagnostics_slot"])
    base_active_path = base_partition / active_base_slot
    base_active_sha256 = _sha256_file(base_active_path)
    if base_active_sha256 != str(base_progress["active_diagnostics_sha256"]):
        raise MonteCarloContractError(
            f"Source diagnostics checksum changed for {member_id}."
        )

    partition_dir = destination / "partitions" / member_id
    manifest_path = partition_dir / "run_manifest.csv"
    if not manifest_path.exists():
        _atomic_csv(manifest, manifest_path)
    elif _sha256_file(manifest_path) != _frame_csv_sha256(manifest):
        raise MonteCarloContractError(
            f"Extension manifest checksum mismatch for {member_id}."
        )
    partition_contract = {
        "convergence_extension_partition_version": (
            CONVERGENCE_EXTENSION_PARTITION_VERSION
        ),
        "checkpoint_protocol_version": PARTITION_CHECKPOINT_PROTOCOL_VERSION,
        "design_sha256": design_sha256,
        "base_design_sha256": base_design,
        "weather_member_id": member.member_id,
        "climate_scenario_id": member.climate_scenario_id,
        "weather_contract_sha256": member.weather_contract_sha256,
        "weather_forcing_sha256": member.forcing_sha256,
        "imported_seed_count": BASE_SEED_COUNT,
        "maximum_seed_count": EXTENSION_SEED_COUNT,
        "expected_imported_run_count": len(base_manifest),
        "expected_run_count_at_n160": len(manifest),
        "expected_imported_run_id_sha256": canonical_sha256(
            {"run_ids": base_manifest["run_id"].astype(str).tolist()}
        ),
        "expected_full_run_id_sha256": canonical_sha256(
            {"run_ids": manifest["run_id"].astype(str).tolist()}
        ),
        "run_manifest_sha256": _sha256_file(manifest_path),
        "source_partition_contract_sha256": _sha256_file(
            base_partition_contract_path
        ),
        "source_run_manifest_sha256": _sha256_file(base_manifest_path),
        "source_progress_sha256": _sha256_file(base_progress_path),
        "source_active_diagnostics_slot": active_base_slot,
        "source_active_diagnostics_sha256": base_active_sha256,
    }
    partition_contract_path = partition_dir / "partition_contract.json"
    if partition_contract_path.exists():
        if _read_json(partition_contract_path) != _json_ready(partition_contract):
            raise MonteCarloContractError(
                f"Extension partition {member_id} belongs to a different design."
            )
    else:
        _atomic_json(partition_contract, partition_contract_path)
    return states, member, manifest, partition_contract, base_diagnostics


def _bootstrap_extension_partition(
    member_id: str,
    seed_bank: tuple[int, ...],
    output_dir: str,
    design_sha256: str,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    _, _, manifest, partition_contract, source_diagnostics = (
        _extension_partition_context(
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
            completed_seed_count=BASE_SEED_COUNT,
            active_slot=None,
            design_sha256=design_sha256,
            member_id=member_id,
        )
        diagnostics = source_diagnostics
        completed = BASE_SEED_COUNT
        active_slot = str(pointer["active_diagnostics_slot"])
    if completed < BASE_SEED_COUNT or active_slot is None:
        raise MonteCarloContractError(
            f"Extension partition {member_id} lacks its complete n=80 import."
        )
    imported = diagnostics.loc[
        diagnostics["occupant_seed"].isin(seed_bank[:BASE_SEED_COUNT])
    ].copy()
    if (
        len(imported) != len(source_diagnostics)
        or _frame_csv_sha256(imported)
        != str(partition_contract["source_active_diagnostics_sha256"])
    ):
        raise MonteCarloContractError(
            f"Imported n=80 diagnostics changed for {member_id}."
        )
    receipt = {
        "status": "IMPORTED_AND_VERIFIED",
        "design_sha256": design_sha256,
        "weather_member_id": member_id,
        "imported_seed_count": BASE_SEED_COUNT,
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
        comparable = {
            key: value for key, value in existing.items() if key != "verified_at_utc"
        }
        if comparable != {
            key: value for key, value in receipt.items() if key != "verified_at_utc"
        }:
            raise MonteCarloContractError(
                f"Extension import receipt changed for {member_id}."
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
        _bootstrap_extension_partition(
            member_id, seed_bank, str(destination), design_sha256
        )
        if index % 9 == 0 or index == len(member_ids):
            print(
                f"[{_utc_now()}] authenticated n=80 import: "
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
    prefix = set(seed_bank[:BASE_SEED_COUNT])
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
        if completed < BASE_SEED_COUNT:
            raise MonteCarloContractError(
                f"Extension import is incomplete for {member_id}."
            )
        manifests.append(
            manifest.loc[manifest["occupant_seed_rank"] <= BASE_SEED_COUNT].copy()
        )
        diagnostics_frames.append(
            diagnostics.loc[diagnostics["occupant_seed"].isin(prefix)].copy()
        )
    manifest = pd.concat(manifests, ignore_index=True)
    diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
    base_hashes = contract["base_experiment"]["base_artifact_sha256"]
    if (
        _frame_csv_sha256(manifest) != base_hashes["run_manifest.csv"]
        or _frame_csv_sha256(diagnostics) != base_hashes["run_diagnostics.csv"]
    ):
        raise MonteCarloContractError(
            "The imported n=80 aggregate differs from the authenticated source."
        )
    reproduced = evaluate_seed_convergence(
        diagnostics,
        seed_order=seed_bank[:BASE_SEED_COUNT],
        rule=ConvergenceRule(),
    )
    if _frame_csv_sha256(reproduced) != base_hashes["convergence_results.csv"]:
        raise MonteCarloContractError(
            "The imported n=80 convergence result differs from the source."
        )


def _advance_extension_partition(
    member_id: str,
    target_seed_count: int,
    seed_bank: tuple[int, ...],
    output_dir: str,
    design_sha256: str,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    states, member, manifest, _, _ = _extension_partition_context(
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
    if completed < BASE_SEED_COUNT:
        raise MonteCarloContractError(
            f"Extension partition {member_id} was not imported before execution."
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
                    f"Executed extension seed {rank} differs from its manifest."
                )
            records.extend(seed_diagnostics.to_dict(orient="records"))
            committed = pd.DataFrame.from_records(records)
            committed["_run_order"] = committed["run_id"].map(run_order)
            if committed["_run_order"].isna().any():
                raise MonteCarloContractError(
                    f"Foreign run ID in extension partition {member_id}."
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


def _advance_all_extension_partitions(
    member_ids: Sequence[str],
    seed_bank: tuple[int, ...],
    destination: Path,
    design_sha256: str,
    max_workers: int,
) -> None:
    print(
        f"[{_utc_now()}] advancing 54 weather partitions from n=80 to n=160 "
        f"with {max_workers} worker(s)",
        flush=True,
    )
    if max_workers == 1:
        for index, member_id in enumerate(member_ids, start=1):
            _advance_extension_partition(
                member_id,
                EXTENSION_CHECKPOINT,
                seed_bank,
                str(destination),
                design_sha256,
            )
            print(
                f"[{_utc_now()}] n=160: {index}/{len(member_ids)} "
                "weather partitions complete",
                flush=True,
            )
        return
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _advance_extension_partition,
                member_id,
                EXTENSION_CHECKPOINT,
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
                    f"Extension weather partition {member_id} failed."
                ) from exc
            completed += 1
            print(
                f"[{_utc_now()}] n=160: {completed}/{len(member_ids)} "
                "weather partitions complete",
                flush=True,
            )


def _collect_extension_checkpoint(
    destination: Path,
    weather: pd.DataFrame,
    seed_bank: tuple[int, ...],
    design_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    manifests: list[pd.DataFrame] = []
    diagnostics_frames: list[pd.DataFrame] = []
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
        if completed != EXTENSION_CHECKPOINT:
            raise MonteCarloContractError(
                f"Extension partition {member_id} ended at n={completed}, not n=160."
            )
        manifests.append(manifest.copy())
        diagnostics_frames.append(diagnostics.copy())
    manifest = pd.concat(manifests, ignore_index=True)
    diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
    expected_runs = len(PANEL_SPECS) * len(weather) * EXTENSION_CHECKPOINT
    if (
        len(manifest) != expected_runs
        or len(diagnostics) != expected_runs
        or manifest["run_id"].duplicated().any()
        or diagnostics["run_id"].duplicated().any()
        or set(manifest["run_id"].astype(str))
        != set(diagnostics["run_id"].astype(str))
    ):
        raise MonteCarloContractError(
            "The n=160 extension aggregate is incomplete or duplicated."
        )
    convergence = evaluate_seed_convergence(
        diagnostics,
        seed_order=seed_bank,
        rule=PROSPECTIVE_N160_CONVERGENCE_RULE,
    )
    current = convergence.loc[
        convergence["seed_count"] == EXTENSION_CHECKPOINT
    ]
    converged = bool(
        not current.empty
        and _strict_bool(
            current["panel_converged_at_checkpoint"], label="n=160 convergence"
        ).all()
    )
    return manifest, diagnostics, convergence, converged


def _verify_historical_decisions(
    convergence: pd.DataFrame,
    base_output_dir: Path,
) -> None:
    base = pd.read_csv(
        base_output_dir / "convergence_results.csv", float_precision="round_trip"
    )
    historical = convergence.loc[
        convergence["seed_count"] <= BASE_SEED_COUNT
    ].copy()
    ignored = {"occupant_seed_bank_count", "occupant_seed_bank_sha256"}
    columns = [column for column in base.columns if column not in ignored]
    sort_columns = [
        "archetype_id",
        "state_id",
        "climate_scenario_id",
        "model_scenario_id",
        "seed_count",
        "metric",
        "statistic",
    ]
    left = base[columns].sort_values(sort_columns, kind="stable").reset_index(drop=True)
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
            "The n=160 evaluation retroactively changed a source checkpoint."
        ) from exc


def _validate_terminal_summary_semantics(
    destination: Path,
    contract: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> bool:
    """Reconcile a terminal summary with its exact artifact ledger and n=160 flag."""

    if summary.get("status") not in {"CONVERGED", "NOT_CONVERGED_AT_N160"}:
        raise MonteCarloContractError("Extension summary is not terminal.")
    if summary.get("design_sha256") != contract.get("design_sha256"):
        raise MonteCarloContractError(
            "Terminal extension summary belongs to a different design."
        )
    artifact_sha256 = summary.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping) or set(artifact_sha256) != set(
        FINAL_EXTENSION_ARTIFACTS
    ):
        raise MonteCarloContractError(
            "Terminal extension summary must contain the exact three-artifact ledger."
        )
    for filename in FINAL_EXTENSION_ARTIFACTS:
        path = destination / filename
        if not path.is_file() or _sha256_file(path) != str(artifact_sha256[filename]):
            raise MonteCarloContractError(
                f"Completed extension artifact changed: {filename}."
            )
    try:
        evidence = pd.read_csv(
            destination / "convergence_results.csv", float_precision="round_trip"
        )
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MonteCarloContractError(
            "Cannot parse terminal n=160 convergence evidence."
        ) from exc
    required = {"seed_count", "panel_converged_at_checkpoint"}
    if evidence.empty or not required.issubset(evidence.columns):
        raise MonteCarloContractError(
            "Terminal extension evidence lacks the n=160 decision fields."
        )
    current = evidence.loc[
        pd.to_numeric(evidence["seed_count"], errors="coerce")
        == EXTENSION_CHECKPOINT
    ]
    if current.empty:
        raise MonteCarloContractError(
            "Terminal extension evidence contains no n=160 checkpoint."
        )
    converged = bool(
        _strict_bool(
            current["panel_converged_at_checkpoint"],
            label="terminal n=160 panel convergence",
        ).all()
    )
    try:
        evaluated = int(summary["evaluated_seed_count"])
        selected = summary.get("selected_seed_count")
        selected_seeds = summary.get("selected_occupant_seeds")
        first_checkpoint = summary.get("first_panel_converged_checkpoint")
    except (TypeError, ValueError) as exc:
        raise MonteCarloContractError(
            "Terminal extension summary contains invalid decision fields."
        ) from exc
    if evaluated != EXTENSION_SEED_COUNT:
        raise MonteCarloContractError(
            "Terminal extension summary did not evaluate exactly 160 seeds."
        )
    if converged:
        try:
            selected_values = tuple(int(value) for value in selected_seeds)
            selected_count = int(selected)
            first_selected_checkpoint = int(first_checkpoint)
        except (TypeError, ValueError) as exc:
            raise MonteCarloContractError(
                "Converged extension summary has invalid selected seeds."
            ) from exc
        if (
            summary.get("status") != "CONVERGED"
            or selected_count != EXTENSION_SEED_COUNT
            or first_selected_checkpoint != EXTENSION_SEED_COUNT
            or selected_values
            != tuple(int(value) for value in contract["occupant_seeds"])
        ):
            raise MonteCarloContractError(
                "Terminal extension selection contradicts its n=160 evidence."
            )
    elif (
        summary.get("status") != "NOT_CONVERGED_AT_N160"
        or selected is not None
        or selected_seeds is not None
        or first_checkpoint is not None
    ):
        raise MonteCarloContractError(
            "Non-converged extension summary contains a production selection."
        )
    return converged


def _run_convergence_extension_unlocked(
    output_dir: str | Path = DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
    *,
    base_output_dir: str | Path = DEFAULT_CONVERGENCE_OUTPUT_DIR,
    max_workers: int | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    workers = min(4, os.cpu_count() or 1) if max_workers is None else max_workers
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise MonteCarloContractError(
            "Convergence-extension max_workers must be an integer from 1 to 8."
        )
    contract = prepare_convergence_extension(
        destination, base_output_dir=base_output_dir
    )
    weather = _weather_selection()
    member_ids = tuple(weather["member_id"].astype(str))
    seed_bank = tuple(int(value) for value in contract["occupant_seeds"])
    summary_path = destination / "convergence_extension_summary.json"
    if summary_path.exists():
        existing = _read_json(summary_path)
        if existing.get("design_sha256") != contract["design_sha256"]:
            raise MonteCarloContractError(
                "Existing extension summary belongs to a different design."
            )
        if existing.get("status") in {"CONVERGED", "NOT_CONVERGED_AT_N160"}:
            _validate_terminal_summary_semantics(destination, contract, existing)
            return existing
        started_at = str(existing.get("started_at_utc", _utc_now()))
    else:
        started_at = _utc_now()

    _atomic_json(
        {
            "status": "IN_PROGRESS",
            "phase": "AUTHENTICATING_N80_IMPORT",
            "design_sha256": contract["design_sha256"],
            "started_at_utc": started_at,
            "imported_seed_count": BASE_SEED_COUNT,
            "active_checkpoint": EXTENSION_CHECKPOINT,
            "max_workers": workers,
            "updated_at_utc": _utc_now(),
        },
        summary_path,
    )
    _bootstrap_all_partitions(
        member_ids,
        seed_bank,
        destination,
        str(contract["design_sha256"]),
    )
    _verify_imported_aggregate(destination, weather, seed_bank, contract)
    _atomic_json(
        {
            "status": "IN_PROGRESS",
            "phase": "SIMULATING_SEEDS_81_TO_160",
            "design_sha256": contract["design_sha256"],
            "started_at_utc": started_at,
            "imported_seed_count": BASE_SEED_COUNT,
            "active_checkpoint": EXTENSION_CHECKPOINT,
            "max_workers": workers,
            "updated_at_utc": _utc_now(),
        },
        summary_path,
    )
    _advance_all_extension_partitions(
        member_ids,
        seed_bank,
        destination,
        str(contract["design_sha256"]),
        workers,
    )
    manifest, diagnostics, convergence, converged = _collect_extension_checkpoint(
        destination,
        weather,
        seed_bank,
        str(contract["design_sha256"]),
    )
    _verify_historical_decisions(convergence, Path(base_output_dir).resolve())
    checkpoint = _write_checkpoint(
        destination,
        EXTENSION_CHECKPOINT,
        manifest,
        diagnostics,
        convergence,
        converged,
    )
    final_files = (
        "run_manifest.csv",
        "run_diagnostics.csv",
        "convergence_results.csv",
    )
    checkpoint_dir = destination / "checkpoints/n160"
    for filename in final_files:
        _copy_atomic(checkpoint_dir / filename, destination / filename)
    final_status = "CONVERGED" if converged else "NOT_CONVERGED_AT_N160"
    final = {
        "status": final_status,
        "scope": "prospective n=160 occupant-seed confirmation",
        "design_sha256": contract["design_sha256"],
        "base_design_sha256": contract["base_experiment"]["base_design_sha256"],
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "imported_seed_count": BASE_SEED_COUNT,
        "new_seed_count": EXTENSION_SEED_COUNT - BASE_SEED_COUNT,
        "evaluated_seed_count": EXTENSION_SEED_COUNT,
        "selected_seed_count": EXTENSION_SEED_COUNT if converged else None,
        "selected_occupant_seeds": list(seed_bank) if converged else None,
        "first_panel_converged_checkpoint": (
            EXTENSION_SEED_COUNT if converged else None
        ),
        "panel_cell_count": len(PANEL_SPECS),
        "weather_member_count": len(weather),
        "run_count": len(diagnostics),
        "convergence_rule": _rule_payload(PROSPECTIVE_N160_CONVERGENCE_RULE),
        "latest_checkpoint": checkpoint,
        "artifact_sha256": {
            filename: _sha256_file(destination / filename)
            for filename in final_files
        },
        "production_interpretation": (
            "The exact 160-seed prefix is authorized for full-stock execution "
            "with the prospective n=160 convergence rule."
            if converged
            else "No seed count was selected. A further protocol must be declared "
            "before any authoritative stock run."
        ),
    }
    _atomic_json(final, summary_path)
    return final


def run_convergence_extension(
    output_dir: str | Path = DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
    *,
    base_output_dir: str | Path = DEFAULT_CONVERGENCE_OUTPUT_DIR,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run or resume the separately contracted n=160 extension."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_execution_lock(destination)
    try:
        return _run_convergence_extension_unlocked(
            destination,
            base_output_dir=base_output_dir,
            max_workers=max_workers,
        )
    except Exception as exc:
        summary_path = destination / "convergence_extension_summary.json"
        try:
            existing = _read_json(summary_path) if summary_path.exists() else {}
            contract_path = destination / "convergence_extension_contract.json"
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
        from .convergence_runner import _release_execution_lock

        _release_execution_lock(lock_path)


def convergence_extension_status(
    output_dir: str | Path = DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    contract_path = destination / "convergence_extension_contract.json"
    if not contract_path.exists():
        return {"status": "NOT_PREPARED", "output_dir": str(destination)}
    contract = _read_json(contract_path)
    version = contract.get("convergence_extension_contract_version")
    if version != CONVERGENCE_EXTENSION_CONTRACT_VERSION:
        return {
            "status": "STALE_CONTRACT",
            "output_dir": str(destination),
            "persisted_contract_version": version,
            "required_contract_version": CONVERGENCE_EXTENSION_CONTRACT_VERSION,
            "design_sha256": contract.get("design_sha256"),
        }
    summary_path = destination / "convergence_extension_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {
        "status": "PREPARED"
    }
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
        "base_design_sha256": contract["base_experiment"]["base_design_sha256"],
        "active_checkpoint": summary.get("active_checkpoint"),
        "selected_seed_count": summary.get("selected_seed_count"),
        "weather_partitions_started": len(counts),
        "weather_partition_seed_count_histogram": {
            str(value): counts.count(value) for value in sorted(set(counts))
        },
        "active_failure_count": failures,
        "progress_validation": "OBSERVATIONAL_POINTER_COUNTS_ONLY",
        "execution_lock_present": (destination / "execution.lock").exists(),
        "updated_at_utc": summary.get("updated_at_utc"),
    }


def load_convergence_extension_selection(
    output_dir: str | Path = DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
) -> ConvergenceSelection:
    destination = Path(output_dir).resolve()
    contract = _read_json(destination / "convergence_extension_contract.json")
    if (
        contract.get("convergence_extension_contract_version")
        != CONVERGENCE_EXTENSION_CONTRACT_VERSION
    ):
        raise MonteCarloContractError(
            "Cannot load a production selection from a stale extension contract."
        )
    unsigned_contract = {
        key: value for key, value in contract.items() if key != "design_sha256"
    }
    if canonical_sha256(unsigned_contract) != contract.get("design_sha256"):
        raise MonteCarloContractError(
            "The extension design checksum cannot be reconstructed."
        )
    summary = _read_json(destination / "convergence_extension_summary.json")
    if summary.get("status") != "CONVERGED":
        raise MonteCarloContractError(
            "Production occupant seeds are available only after n=160 convergence."
        )
    if summary.get("design_sha256") != contract.get("design_sha256"):
        raise MonteCarloContractError(
            "Extension summary and contract design checksums differ."
        )
    expected_rule = _rule_payload(PROSPECTIVE_N160_CONVERGENCE_RULE)
    if (
        contract.get("convergence_rule") != expected_rule
        or summary.get("convergence_rule") != expected_rule
    ):
        raise MonteCarloContractError(
            "The extension selection does not use the authorized n=160 rule."
        )
    if not _validate_terminal_summary_semantics(destination, contract, summary):
        raise MonteCarloContractError(
            "The extension evidence does not authorize an n=160 selection."
        )
    try:
        seeds = tuple(int(value) for value in summary["selected_occupant_seeds"])
        selected_count = int(summary["selected_seed_count"])
        first_checkpoint = int(summary["first_panel_converged_checkpoint"])
        declared_seeds = tuple(int(value) for value in contract["occupant_seeds"])
        expected_sha256 = str(
            summary["artifact_sha256"]["convergence_results.csv"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MonteCarloContractError(
            "The converged extension summary lacks a valid production selection."
        ) from exc
    if (
        len(seeds) != EXTENSION_SEED_COUNT
        or selected_count != EXTENSION_SEED_COUNT
        or first_checkpoint != EXTENSION_SEED_COUNT
        or seeds != declared_seeds
    ):
        raise MonteCarloContractError(
            "Extension summary does not select the exact n=160 seed bank."
        )
    evidence_path = destination / "convergence_results.csv"
    if not evidence_path.is_file() or _sha256_file(evidence_path) != expected_sha256:
        raise MonteCarloContractError(
            "Extension evidence is missing or differs from its selected checksum."
        )
    return ConvergenceSelection(
        occupant_seeds=seeds,
        convergence_results_path=evidence_path,
        convergence_results_sha256=expected_sha256,
        design_sha256=str(contract["design_sha256"]),
        convergence_rule=PROSPECTIVE_N160_CONVERGENCE_RULE,
    )
