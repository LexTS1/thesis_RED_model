"""Restartable occupant-seed convergence experiment for the Gate-5 model.

The experiment advances every weather member to the same nested seed
checkpoint before evaluating the predeclared convergence rule.  Each weather
partition is atomically committed after one complete seed across the three
representative physical cells, so an interrupted process loses at most one
weather/seed block and can be restarted with the identical command.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from thermal_model.behaviour import (
    dwelling_class,
    load_behaviour_assumptions,
    load_occupant_distribution,
)
from thermal_model.contracts import (
    ArchetypeStateInput,
    load_assumption_contract,
)
from thermal_model.validation import load_unique_archetype_states

from .aggregation import load_stock_weights
from .contracts import (
    MODEL_CONTRACT_VERSION,
    MonteCarloContractError,
    archetype_state_sha256,
    canonical_sha256,
)
from .design import (
    ConvergenceRule,
    build_balanced_manifest,
    evaluate_seed_convergence,
    make_seed_bank,
)
from .runner import execute_balanced_design
from .scenarios import model_scenario_sha256, resolve_model_scenario
from .weather import load_weather_catalog, load_weather_member


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONVERGENCE_OUTPUT_DIR = (
    PROJECT_ROOT / "thermal_model/data/monte_carlo/convergence_panel"
)
VALIDATION_SOURCE_PATH = (
    PROJECT_ROOT
    / "thermal_model/data/validation/deterministic_archetype_validation.csv"
)
EXPECTED_VALIDATION_SOURCE_SHA256 = (
    "062206bb36023adea89f8eb8216a62cccc704793c8fa77b66adc7e07957fd0da"
)
CONVERGENCE_EXECUTION_CONTRACT_VERSION = "gate5_convergence_execution_v2"
PARTITION_CHECKPOINT_PROTOCOL_VERSION = "alternating_diagnostics_slots_v1"
DIAGNOSTICS_SLOT_FILENAMES = (
    "run_diagnostics.slot_a.csv",
    "run_diagnostics.slot_b.csv",
)
LOCK_STALE_AFTER_SECONDS = 300.0
MASTER_SEED = 20250808
MAX_SEED_COUNT = 80
MODEL_SCENARIO_ID = "central"

# The selection is pre-stratified on deterministic Gate-3 demand.  Keeping the
# construction period fixed avoids using age as an uncontrolled selection
# difference while spanning all three renovation states and both behaviour
# classes.
PANEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "demand_role": "low",
        "archetype_id": "BE_TABULA_14",
        "state_id": "TABULA_advanced_A_proxy",
        "expected_heating_kWh_m2": 7.6346829039,
        "deterministic_rank_of_75": 1,
        "selection_basis": (
            "global minimum heating intensity (five-way enclosed-apartment tie; "
            "middle construction-period tie-break)"
        ),
    },
    {
        "demand_role": "medium",
        "archetype_id": "BE_TABULA_13",
        "state_id": "TABULA_standard_B_proxy",
        "expected_heating_kWh_m2": 67.9223894123,
        "deterministic_rank_of_75": 38,
        "selection_basis": "exact median deterministic heating intensity",
    },
    {
        "demand_role": "high",
        "archetype_id": "BE_TABULA_11",
        "state_id": "TABULA_existing",
        "expected_heating_kWh_m2": 202.6780125480,
        "deterministic_rank_of_75": 67,
        "selection_basis": (
            "approximately 90th-percentile heating intensity and the greatest "
            "annual/peak heating demand among positive-weight 2050 cells"
        ),
    },
)


@dataclass(frozen=True)
class ConvergenceSelection:
    """Authenticated arguments selected for the full-stock runner."""

    occupant_seeds: tuple[int, ...]
    convergence_results_path: Path
    convergence_results_sha256: str
    design_sha256: str
    convergence_rule: ConvergenceRule = ConvergenceRule()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a frame once so contract and on-disk checksums are identical."""

    serializable = frame.copy(deep=True)
    for column in serializable.select_dtypes(
        include=["datetimetz", "datetime64"]
    ).columns:
        serializable[column] = serializable[column].map(
            lambda value: value.isoformat() if not pd.isna(value) else ""
        )
    stream = io.StringIO(newline="")
    serializable.to_csv(stream, index=False, float_format="%.17g")
    return stream.getvalue().encode("utf-8")


def _frame_csv_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(_csv_bytes(frame)).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_bytes(_csv_bytes(frame))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonteCarloContractError(f"Cannot read convergence JSON {path}.") from exc
    if not isinstance(payload, dict):
        raise MonteCarloContractError(f"Convergence JSON {path} must contain an object.")
    return payload


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.writing")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _acquire_execution_lock(destination: Path) -> Path:
    """Prevent two convergence coordinators from writing the same partitions."""

    lock_path = destination / "execution.lock"
    payload = {
        "pid": os.getpid(),
        "started_at_utc": _utc_now(),
        "purpose": "Gate-5 occupant-seed convergence coordinator",
    }
    for attempt in range(2):
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            try:
                existing = _read_json(lock_path)
                existing_pid = int(existing["pid"])
                started_at = datetime.fromisoformat(
                    str(existing["started_at_utc"]).replace("Z", "+00:00")
                )
                if existing_pid <= 0 or started_at.tzinfo is None:
                    raise ValueError("invalid lock identity")
            except FileNotFoundError:
                # The owner may have released the lock between O_EXCL and the
                # read.  Retry acquisition without unlinking an unknown file.
                continue
            except (KeyError, TypeError, ValueError, MonteCarloContractError) as exc:
                # In particular, an empty file can be the tiny interval between
                # another coordinator's O_EXCL creation and payload write.
                raise MonteCarloContractError(
                    "The convergence execution lock is malformed or still being "
                    "initialized; refusing to remove it automatically."
                ) from exc
            try:
                os.kill(existing_pid, 0)
            except ProcessLookupError as exc:
                age_seconds = (
                    datetime.now(timezone.utc) - started_at.astimezone(timezone.utc)
                ).total_seconds()
                if age_seconds < LOCK_STALE_AFTER_SECONDS:
                    raise MonteCarloContractError(
                        "The convergence execution lock belongs to a missing PID "
                        "but is too recent to remove safely."
                    ) from exc
                if attempt == 0:
                    lock_path.unlink(missing_ok=True)
                    continue
                raise MonteCarloContractError(
                    "Cannot replace a stale convergence execution lock."
                ) from exc
            except PermissionError as exc:
                raise MonteCarloContractError(
                    "Cannot determine whether the convergence coordinator lock is active."
                ) from exc
            raise MonteCarloContractError(
                "A convergence coordinator is already running for this output "
                f"directory (PID {existing_pid})."
            )
        else:
            try:
                os.write(
                    descriptor,
                    (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
                )
            finally:
                os.close(descriptor)
            return lock_path
    raise MonteCarloContractError("Unable to acquire convergence execution lock.")


def _release_execution_lock(lock_path: Path) -> None:
    try:
        payload = _read_json(lock_path)
    except (FileNotFoundError, MonteCarloContractError):
        return
    if int(payload.get("pid", -1)) == os.getpid():
        lock_path.unlink(missing_ok=True)


def _rule_payload(rule: ConvergenceRule) -> dict[str, Any]:
    return {
        "checkpoints": list(rule.checkpoints),
        "relative_tolerance": float(rule.relative_tolerance),
        "required_consecutive_expansions": int(
            rule.required_consecutive_expansions
        ),
        "metrics_and_absolute_floors": [
            {"metric": metric, "absolute_floor": float(floor)}
            for metric, floor in rule.metrics_and_absolute_floors
        ],
        "statistics": list(rule.statistics),
    }


def load_convergence_panel() -> tuple[tuple[ArchetypeStateInput, ...], pd.DataFrame]:
    """Load and revalidate the frozen low/medium/high convergence panel."""

    if not VALIDATION_SOURCE_PATH.is_file():
        raise FileNotFoundError(
            f"Deterministic validation source is missing: {VALIDATION_SOURCE_PATH}"
        )
    source_sha256 = _sha256_file(VALIDATION_SOURCE_PATH)
    if source_sha256 != EXPECTED_VALIDATION_SOURCE_SHA256:
        raise MonteCarloContractError(
            "Deterministic validation source changed after convergence-panel "
            f"selection: expected {EXPECTED_VALIDATION_SOURCE_SHA256}, got "
            f"{source_sha256}. Revalidate the panel before running."
        )
    validation = pd.read_csv(VALIDATION_SOURCE_PATH)
    state_lookup = {
        (item.archetype_id, item.state_id): item
        for item in load_unique_archetype_states()
    }
    stock = load_stock_weights()
    thermal_contract = load_assumption_contract()
    records: list[dict[str, Any]] = []
    states: list[ArchetypeStateInput] = []
    for spec in PANEL_SPECS:
        key = (str(spec["archetype_id"]), str(spec["state_id"]))
        selected = validation.loc[
            (validation["archetype_id"] == key[0])
            & (validation["state_id"] == key[1])
        ]
        if len(selected) != 1 or key not in state_lookup:
            raise MonteCarloContractError(
                f"Convergence panel cell {key} is missing or ambiguous."
            )
        row = selected.iloc[0]
        actual_heating = float(row["model_heating_kWh_m2"])
        if not np.isclose(
            actual_heating,
            float(spec["expected_heating_kWh_m2"]),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise MonteCarloContractError(
                f"Convergence panel demand changed for {key}: got {actual_heating}."
            )
        if not bool(row["within_predeclared_tabula_band"]):
            raise MonteCarloContractError(
                f"Convergence panel cell {key} is outside its Gate-3 validation band."
            )
        if str(row["assumptions_sha256"]) != thermal_contract.sha256:
            raise MonteCarloContractError(
                f"Convergence panel cell {key} was validated under a stale thermal contract."
            )
        active_weight = stock.loc[
            (stock["archetype_id"] == key[0]) & (stock["state_id"] == key[1]),
            "state_dwellings_2050",
        ]
        if active_weight.empty or float(active_weight.sum()) <= 0.0:
            raise MonteCarloContractError(
                f"Convergence panel cell {key} has no positive 2050 stock weight."
            )
        state = state_lookup[key]
        states.append(state)
        records.append(
            {
                **spec,
                "dwelling_type": state.dwelling_type,
                "construction_period": state.construction_period,
                "floor_area_m2": float(row["floor_area_m2"]),
                "model_heating_kWh_m2": actual_heating,
                "model_cooling_kWh_m2": float(row["model_cooling_kWh_m2"]),
                "peak_heating_W": float(row["peak_heating_W"]),
                "peak_cooling_W": float(row["peak_cooling_W"]),
                "archetype_state_sha256": archetype_state_sha256(state),
                "positive_weight_dwellings_2050": float(active_weight.sum()),
                "validation_weather_member_id": str(row["weather_member_id"]),
                "validation_assumptions_sha256": str(row["assumptions_sha256"]),
                "validation_source_path": str(
                    VALIDATION_SOURCE_PATH.relative_to(PROJECT_ROOT)
                ),
                "validation_source_sha256": source_sha256,
            }
        )
    if {dwelling_class(item.dwelling_type) for item in states} != {"SFH", "MFH"}:
        raise MonteCarloContractError(
            "Convergence panel must cover both SFH and MFH behaviour classes."
        )
    if len({item.state_id for item in states}) != 3:
        raise MonteCarloContractError(
            "Convergence panel must cover all three physical renovation states."
        )
    return tuple(states), pd.DataFrame.from_records(records)


def _weather_selection() -> pd.DataFrame:
    catalog = load_weather_catalog()
    if len(catalog) != 54:
        raise MonteCarloContractError(
            "The convergence experiment requires all 54 authoritative weather members."
        )
    columns = [
        "member_id",
        "scenario",
        "weather_pair_id",
        "observed_pvgis_year",
        "climate_target",
        "row_count",
        "member_sha256",
        "metadata_sha256",
        "manifest_sha256",
        "morph_contract_sha256",
        "facade_source_sha256_json",
        "weather_contract_sha256",
    ]
    selected = catalog.loc[:, columns].sort_values(
        ["scenario", "observed_pvgis_year"], kind="stable"
    )
    pairs = selected.groupby("scenario")["weather_pair_id"].apply(tuple)
    if pairs.nunique() != 1:
        raise MonteCarloContractError(
            "Convergence weather pathways do not share one paired-year set."
        )
    return selected.reset_index(drop=True)


def prepare_convergence_experiment(
    output_dir: str | Path = DEFAULT_CONVERGENCE_OUTPUT_DIR,
) -> dict[str, Any]:
    """Freeze the panel, weather, seed and contract inventory without simulating."""

    destination = Path(output_dir).resolve()
    states, panel = load_convergence_panel()
    weather = _weather_selection()
    seeds = make_seed_bank(MAX_SEED_COUNT, master_seed=MASTER_SEED)
    rule = ConvergenceRule()
    thermal = load_assumption_contract()
    behaviour = load_behaviour_assumptions()
    _, occupant_distribution_sha256 = load_occupant_distribution()
    scenario = resolve_model_scenario(MODEL_SCENARIO_ID)
    selection_artifact_sha256 = {
        "panel_selection.csv": _frame_csv_sha256(panel),
        "weather_selection.csv": _frame_csv_sha256(weather),
    }
    payload = {
        "convergence_execution_contract_version": (
            CONVERGENCE_EXECUTION_CONTRACT_VERSION
        ),
        "partition_checkpoint_protocol_version": (
            PARTITION_CHECKPOINT_PROTOCOL_VERSION
        ),
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "central_thermal_assumptions_sha256": thermal.sha256,
        "behaviour_assumptions_sha256": behaviour.sha256,
        "occupant_distribution_sha256": occupant_distribution_sha256,
        "model_scenario": scenario.definition(),
        "model_scenario_sha256": model_scenario_sha256(
            scenario, thermal.sha256
        ),
        "panel": panel.to_dict(orient="records"),
        "weather_members": weather.to_dict(orient="records"),
        "selection_artifact_sha256": selection_artifact_sha256,
        "master_seed": MASTER_SEED,
        "occupant_seeds": list(seeds),
        "convergence_rule": _rule_payload(rule),
        "expected_maximum_run_count": len(states) * len(weather) * len(seeds),
        "adaptive_stopping": True,
        "checkpoint_commit_boundary": (
            "one complete occupant seed across three panel cells within one "
            "weather partition"
        ),
    }
    design_sha256 = canonical_sha256(payload)
    contract = {**payload, "design_sha256": design_sha256}
    contract_path = destination / "convergence_execution_contract.json"
    panel_path = destination / "panel_selection.csv"
    weather_path = destination / "weather_selection.csv"
    if contract_path.exists():
        if _read_json(contract_path) != _json_ready(contract):
            raise MonteCarloContractError(
                "Convergence output directory belongs to a different design."
            )
    else:
        _atomic_csv(panel, panel_path)
        _atomic_csv(weather, weather_path)
        _atomic_json(contract, contract_path)
    for path in (panel_path, weather_path):
        if not path.is_file():
            raise MonteCarloContractError(
                f"Convergence design artifact is missing: {path}."
            )
        expected_sha256 = str(selection_artifact_sha256[path.name])
        if _sha256_file(path) != expected_sha256:
            raise MonteCarloContractError(
                f"Convergence selection artifact checksum mismatch: {path.name}."
            )
    return contract


def _read_partition_diagnostics(
    path: Path,
    manifest: pd.DataFrame,
    seed_bank: Sequence[int],
) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        return pd.DataFrame(), 0
    try:
        diagnostics = pd.read_csv(path, float_precision="round_trip")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MonteCarloContractError(
            f"Cannot restore convergence diagnostics {path}."
        ) from exc
    if diagnostics.empty or "run_id" not in diagnostics or "occupant_seed" not in diagnostics:
        raise MonteCarloContractError(
            f"Convergence checkpoint {path} is empty or lacks run identities."
        )
    if diagnostics["run_id"].duplicated().any():
        raise MonteCarloContractError(
            f"Convergence checkpoint {path} contains duplicate run IDs."
        )
    required_manifest_columns = {
        "run_id",
        "occupant_seed",
        "occupant_seed_rank",
    }
    missing_manifest_columns = sorted(
        required_manifest_columns.difference(manifest.columns)
    )
    if missing_manifest_columns or manifest["run_id"].duplicated().any():
        raise MonteCarloContractError(
            f"Convergence manifest for {path} is missing identity columns or "
            "contains duplicate run IDs."
        )
    rank_by_seed = {int(seed): index + 1 for index, seed in enumerate(seed_bank)}
    try:
        ranks = sorted(
            {rank_by_seed[int(value)] for value in diagnostics["occupant_seed"]}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MonteCarloContractError(
            f"Convergence checkpoint {path} contains a foreign occupant seed."
        ) from exc
    completed = max(ranks)
    if ranks != list(range(1, completed + 1)):
        raise MonteCarloContractError(
            f"Convergence checkpoint {path} is not a contiguous seed prefix."
        )
    expected_ids = set(
        manifest.loc[
            manifest["occupant_seed_rank"] <= completed, "run_id"
        ].astype(str)
    )
    if set(diagnostics["run_id"].astype(str)) != expected_ids:
        raise MonteCarloContractError(
            f"Convergence checkpoint {path} does not contain its exact run-ID prefix."
        )
    expected_seed_by_run_id = {
        str(row.run_id): int(row.occupant_seed)
        for row in manifest[["run_id", "occupant_seed"]].itertuples(index=False)
    }
    mismatched_seed_identities = [
        str(row.run_id)
        for row in diagnostics[["run_id", "occupant_seed"]].itertuples(index=False)
        if int(row.occupant_seed)
        != expected_seed_by_run_id[str(row.run_id)]
    ]
    if mismatched_seed_identities:
        raise MonteCarloContractError(
            f"Convergence checkpoint {path} changed the occupant-seed identity "
            "of one or more run IDs."
        )
    return diagnostics, completed


def _restore_partition_diagnostics(
    partition_dir: Path,
    manifest: pd.DataFrame,
    seed_bank: Sequence[int],
    *,
    design_sha256: str,
    member_id: str,
) -> tuple[pd.DataFrame, int, str | None]:
    """Restore only the checksum-verified diagnostics slot named by progress."""

    progress_path = partition_dir / "progress.json"
    if not progress_path.exists():
        # Slot files without a pointer were written before an interrupted
        # pointer commit and are deliberately uncommitted.
        return pd.DataFrame(), 0, None
    progress = _read_json(progress_path)
    if (
        progress.get("checkpoint_protocol_version")
        != PARTITION_CHECKPOINT_PROTOCOL_VERSION
        or progress.get("design_sha256") != design_sha256
        or progress.get("weather_member_id") != member_id
    ):
        raise MonteCarloContractError(
            f"Convergence progress pointer is incompatible for {member_id}."
        )
    active_slot = str(progress.get("active_diagnostics_slot", ""))
    if active_slot not in DIAGNOSTICS_SLOT_FILENAMES:
        raise MonteCarloContractError(
            f"Convergence progress pointer has an invalid diagnostics slot for "
            f"{member_id}."
        )
    active_path = partition_dir / active_slot
    if not active_path.is_file():
        raise MonteCarloContractError(
            f"Convergence progress pointer references a missing diagnostics slot "
            f"for {member_id}."
        )
    expected_sha256 = str(progress.get("active_diagnostics_sha256", ""))
    if (
        _sha256_file(active_path) != expected_sha256
        or str(progress.get("run_diagnostics_sha256", "")) != expected_sha256
    ):
        raise MonteCarloContractError(
            f"Convergence diagnostics slot checksum mismatch for {member_id}."
        )
    diagnostics, completed = _read_partition_diagnostics(
        active_path, manifest, seed_bank
    )
    try:
        pointer_completed = int(progress["completed_seed_count"])
        pointer_run_count = int(progress["completed_run_count"])
        pointer_seeds = [int(value) for value in progress["completed_occupant_seeds"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise MonteCarloContractError(
            f"Convergence progress pointer is malformed for {member_id}."
        ) from exc
    if (
        pointer_completed != completed
        or pointer_run_count != len(diagnostics)
        or pointer_seeds != list(seed_bank[:completed])
    ):
        raise MonteCarloContractError(
            f"Convergence progress pointer does not match its diagnostics slot "
            f"for {member_id}."
        )
    return diagnostics, completed, active_slot


def _commit_partition_diagnostics(
    partition_dir: Path,
    diagnostics: pd.DataFrame,
    manifest: pd.DataFrame,
    seed_bank: Sequence[int],
    *,
    completed_seed_count: int,
    active_slot: str | None,
    design_sha256: str,
    member_id: str,
) -> dict[str, Any]:
    """Commit diagnostics to the inactive slot, then atomically switch pointer."""

    if active_slot is not None and active_slot not in DIAGNOSTICS_SLOT_FILENAMES:
        raise MonteCarloContractError(
            f"Cannot commit from an invalid diagnostics slot for {member_id}."
        )
    inactive_slot = (
        DIAGNOSTICS_SLOT_FILENAMES[1]
        if active_slot == DIAGNOSTICS_SLOT_FILENAMES[0]
        else DIAGNOSTICS_SLOT_FILENAMES[0]
    )
    inactive_path = partition_dir / inactive_slot
    _atomic_csv(diagnostics, inactive_path)
    restored, restored_count = _read_partition_diagnostics(
        inactive_path, manifest, seed_bank
    )
    if restored_count != completed_seed_count or len(restored) != len(diagnostics):
        raise MonteCarloContractError(
            f"New convergence diagnostics slot failed verification for {member_id}."
        )
    diagnostics_sha256 = _sha256_file(inactive_path)
    pointer = {
        "status": "IN_PROGRESS",
        "checkpoint_protocol_version": PARTITION_CHECKPOINT_PROTOCOL_VERSION,
        "design_sha256": design_sha256,
        "weather_member_id": member_id,
        "active_diagnostics_slot": inactive_slot,
        "active_diagnostics_sha256": diagnostics_sha256,
        # Retained for consumers of the v1 progress snapshot; in v2 it is the
        # checksum of the active slot named above.
        "run_diagnostics_sha256": diagnostics_sha256,
        "completed_seed_count": completed_seed_count,
        "completed_occupant_seeds": list(seed_bank[:completed_seed_count]),
        "completed_run_count": len(diagnostics),
        "updated_at_utc": _utc_now(),
    }
    _atomic_json(pointer, partition_dir / "progress.json")
    return pointer


def _mark_failure_recovered(failure_path: Path, completed_seed_count: int) -> None:
    """Retire a stale failure marker once its seed prefix is committed."""

    if not failure_path.exists():
        return
    failure = _read_json(failure_path)
    if failure.get("status") != "FAILED":
        return
    _atomic_json(
        {
            **failure,
            "status": "RECOVERED",
            "recovered_at_seed_count": completed_seed_count,
            "updated_at_utc": _utc_now(),
        },
        failure_path,
    )


def _advance_weather_partition(
    member_id: str,
    target_seed_count: int,
    seed_bank: tuple[int, ...],
    output_dir: str,
    design_sha256: str,
) -> dict[str, Any]:
    """Advance one weather partition atomically to a common seed checkpoint."""

    destination = Path(output_dir).resolve()
    states, _ = load_convergence_panel()
    member = load_weather_member(member_id)
    manifest = build_balanced_manifest(
        states, [member], seed_bank, (MODEL_SCENARIO_ID,)
    )
    partition_dir = destination / "partitions" / member_id
    manifest_path = partition_dir / "run_manifest.csv"
    failure_path = partition_dir / "last_failure.json"
    partition_contract = {
        "convergence_execution_contract_version": (
            CONVERGENCE_EXECUTION_CONTRACT_VERSION
        ),
        "partition_checkpoint_protocol_version": (
            PARTITION_CHECKPOINT_PROTOCOL_VERSION
        ),
        "design_sha256": design_sha256,
        "weather_member_id": member.member_id,
        "climate_scenario_id": member.climate_scenario_id,
        "weather_contract_sha256": member.weather_contract_sha256,
        "weather_forcing_sha256": member.forcing_sha256,
        "expected_run_count_at_n80": len(manifest),
        "expected_run_id_sha256": canonical_sha256(
            {"run_ids": manifest["run_id"].astype(str).tolist()}
        ),
    }
    partition_contract_path = partition_dir / "partition_contract.json"
    if not manifest_path.exists():
        _atomic_csv(manifest, manifest_path)
    else:
        persisted_manifest = pd.read_csv(manifest_path)
        if (
            len(persisted_manifest) != len(manifest)
            or set(persisted_manifest["run_id"].astype(str))
            != set(manifest["run_id"].astype(str))
        ):
            raise MonteCarloContractError(
                f"Convergence manifest content changed for {member_id}."
            )
    partition_contract = {
        **partition_contract,
        "run_manifest_sha256": _sha256_file(manifest_path),
    }
    if partition_contract_path.exists():
        if _read_json(partition_contract_path) != _json_ready(partition_contract):
            raise MonteCarloContractError(
                f"Convergence partition {member_id} belongs to a different design."
            )
    else:
        _atomic_json(partition_contract, partition_contract_path)
    persisted_contract = _read_json(partition_contract_path)
    if not manifest_path.is_file() or _sha256_file(manifest_path) != str(
        persisted_contract.get("run_manifest_sha256")
    ):
        raise MonteCarloContractError(
            f"Convergence manifest checksum mismatch for {member_id}."
        )
    diagnostics, completed, active_slot = _restore_partition_diagnostics(
        partition_dir,
        manifest,
        seed_bank,
        design_sha256=design_sha256,
        member_id=member_id,
    )
    if completed >= target_seed_count:
        _mark_failure_recovered(failure_path, completed)
        if active_slot is None:
            raise MonteCarloContractError(
                f"Completed convergence partition {member_id} lacks an active slot."
            )
        diagnostics_path = partition_dir / active_slot
        return {
            "weather_member_id": member_id,
            "completed_seed_count": completed,
            "run_count": len(diagnostics),
            "diagnostics_sha256": _sha256_file(diagnostics_path),
        }

    records = diagnostics.to_dict(orient="records") if not diagnostics.empty else []
    run_order = {
        run_id: index for index, run_id in enumerate(manifest["run_id"].astype(str))
    }
    for rank in range(completed + 1, target_seed_count + 1):
        seed = int(seed_bank[rank - 1])
        try:
            seed_manifest, seed_diagnostics, _ = execute_balanced_design(
                states,
                [member],
                [seed],
                (MODEL_SCENARIO_ID,),
            )
            expected = set(
                manifest.loc[
                    manifest["occupant_seed_rank"] == rank, "run_id"
                ].astype(str)
            )
            if set(seed_manifest["run_id"].astype(str)) != expected or set(
                seed_diagnostics["run_id"].astype(str)
            ) != expected:
                raise MonteCarloContractError(
                    f"Executed convergence seed {rank} differs from its manifest."
                )
            records.extend(seed_diagnostics.to_dict(orient="records"))
            committed = pd.DataFrame.from_records(records)
            committed["_run_order"] = committed["run_id"].map(run_order)
            if committed["_run_order"].isna().any():
                raise MonteCarloContractError(
                    f"Foreign run ID encountered in convergence partition {member_id}."
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
    if active_slot is None:
        raise MonteCarloContractError(
            f"Completed convergence partition {member_id} lacks an active slot."
        )
    diagnostics_path = partition_dir / active_slot
    return {
        "weather_member_id": member_id,
        "completed_seed_count": target_seed_count,
        "run_count": len(records),
        "diagnostics_sha256": _sha256_file(diagnostics_path),
    }


def _advance_all_weather(
    member_ids: Sequence[str],
    target_seed_count: int,
    seed_bank: tuple[int, ...],
    destination: Path,
    design_sha256: str,
    max_workers: int,
) -> None:
    print(
        f"[{_utc_now()}] advancing 54 weather partitions to n={target_seed_count} "
        f"with {max_workers} worker(s)",
        flush=True,
    )
    if max_workers == 1:
        completed = 0
        for member_id in member_ids:
            _advance_weather_partition(
                member_id,
                target_seed_count,
                seed_bank,
                str(destination),
                design_sha256,
            )
            completed += 1
            print(
                f"[{_utc_now()}] n={target_seed_count}: {completed}/{len(member_ids)} "
                "weather partitions complete",
                flush=True,
            )
        return
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _advance_weather_partition,
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
                    f"Convergence weather partition {member_id} failed."
                ) from exc
            completed += 1
            print(
                f"[{_utc_now()}] n={target_seed_count}: {completed}/{len(member_ids)} "
                "weather partitions complete",
                flush=True,
            )


def _collect_checkpoint(
    destination: Path,
    weather: pd.DataFrame,
    seed_bank: tuple[int, ...],
    target_seed_count: int,
    rule: ConvergenceRule,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    diagnostics_frames: list[pd.DataFrame] = []
    manifest_frames: list[pd.DataFrame] = []
    prefix = set(seed_bank[:target_seed_count])
    for member_id in weather["member_id"].astype(str):
        partition_dir = destination / "partitions" / member_id
        manifest = pd.read_csv(partition_dir / "run_manifest.csv")
        partition_contract = _read_json(
            partition_dir / "partition_contract.json"
        )
        diagnostics, completed, _ = _restore_partition_diagnostics(
            partition_dir,
            manifest,
            seed_bank,
            design_sha256=str(partition_contract.get("design_sha256", "")),
            member_id=member_id,
        )
        if completed < target_seed_count:
            raise MonteCarloContractError(
                f"Weather partition {member_id} has only {completed} seeds at "
                f"checkpoint {target_seed_count}."
            )
        diagnostics_frames.append(
            diagnostics.loc[diagnostics["occupant_seed"].isin(prefix)].copy()
        )
        manifest_frames.append(
            manifest.loc[manifest["occupant_seed_rank"] <= target_seed_count].copy()
        )
    diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
    manifest = pd.concat(manifest_frames, ignore_index=True)
    expected_runs = len(PANEL_SPECS) * len(weather) * target_seed_count
    if (
        len(diagnostics) != expected_runs
        or len(manifest) != expected_runs
        or diagnostics["run_id"].duplicated().any()
        or manifest["run_id"].duplicated().any()
        or set(diagnostics["run_id"].astype(str))
        != set(manifest["run_id"].astype(str))
    ):
        raise MonteCarloContractError(
            f"Convergence checkpoint n={target_seed_count} is incomplete or duplicated."
        )
    convergence = evaluate_seed_convergence(
        diagnostics,
        seed_order=seed_bank[:target_seed_count],
        rule=rule,
    )
    current = convergence.loc[convergence["seed_count"] == target_seed_count]
    converged = bool(
        not current.empty and current["panel_converged_at_checkpoint"].all()
    )
    return manifest, diagnostics, convergence, converged


def _write_checkpoint(
    destination: Path,
    target_seed_count: int,
    manifest: pd.DataFrame,
    diagnostics: pd.DataFrame,
    convergence: pd.DataFrame,
    converged: bool,
) -> dict[str, Any]:
    checkpoint_dir = destination / "checkpoints" / f"n{target_seed_count:03d}"
    artifacts = {
        "run_manifest.csv": manifest,
        "run_diagnostics.csv": diagnostics,
        "convergence_results.csv": convergence,
    }
    for filename, frame in artifacts.items():
        _atomic_csv(frame, checkpoint_dir / filename)
    summary = {
        "status": "CONVERGED" if converged else "NOT_YET_CONVERGED",
        "seed_count": target_seed_count,
        "run_count": len(diagnostics),
        "panel_converged_at_checkpoint": converged,
        "artifact_sha256": {
            filename: _sha256_file(checkpoint_dir / filename)
            for filename in artifacts
        },
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(summary, checkpoint_dir / "checkpoint_summary.json")
    return summary


def _run_convergence_experiment_unlocked(
    output_dir: str | Path = DEFAULT_CONVERGENCE_OUTPUT_DIR,
    *,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run or resume the adaptive 5/10/20/40/80 convergence experiment."""

    destination = Path(output_dir).resolve()
    workers = min(4, os.cpu_count() or 1) if max_workers is None else max_workers
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise MonteCarloContractError("Convergence max_workers must be an integer from 1 to 8.")
    contract = prepare_convergence_experiment(destination)
    weather = _weather_selection()
    member_ids = tuple(weather["member_id"].astype(str))
    seed_bank = tuple(int(value) for value in contract["occupant_seeds"])
    rule = ConvergenceRule()
    summary_path = destination / "convergence_summary.json"
    if summary_path.exists():
        existing = _read_json(summary_path)
        existing_design_sha256 = existing.get("design_sha256")
        if existing_design_sha256 not in {None, contract["design_sha256"]}:
            raise MonteCarloContractError(
                "Existing convergence summary belongs to a different design."
            )
        if existing.get("status") in {"CONVERGED", "NOT_CONVERGED_AT_N80"}:
            for filename, expected in existing.get("artifact_sha256", {}).items():
                path = destination / filename
                if not path.is_file() or _sha256_file(path) != str(expected):
                    raise MonteCarloContractError(
                        f"Completed convergence artifact changed: {filename}."
                    )
            return existing
        started_at = str(existing.get("started_at_utc", _utc_now()))
    else:
        started_at = _utc_now()

    latest_checkpoint_summary: dict[str, Any] | None = None
    for checkpoint in rule.checkpoints:
        _atomic_json(
            {
                "status": "IN_PROGRESS",
                "design_sha256": contract["design_sha256"],
                "started_at_utc": started_at,
                "active_checkpoint": checkpoint,
                "max_workers": workers,
                "updated_at_utc": _utc_now(),
            },
            summary_path,
        )
        _advance_all_weather(
            member_ids,
            checkpoint,
            seed_bank,
            destination,
            str(contract["design_sha256"]),
            workers,
        )
        manifest, diagnostics, convergence, converged = _collect_checkpoint(
            destination, weather, seed_bank, checkpoint, rule
        )
        latest_checkpoint_summary = _write_checkpoint(
            destination,
            checkpoint,
            manifest,
            diagnostics,
            convergence,
            converged,
        )
        print(
            f"[{_utc_now()}] convergence checkpoint n={checkpoint}: "
            f"{'PASS' if converged else 'not yet stable'}",
            flush=True,
        )
        if converged or checkpoint == rule.checkpoints[-1]:
            final_status = "CONVERGED" if converged else "NOT_CONVERGED_AT_N80"
            checkpoint_dir = destination / "checkpoints" / f"n{checkpoint:03d}"
            final_files = (
                "run_manifest.csv",
                "run_diagnostics.csv",
                "convergence_results.csv",
            )
            for filename in final_files:
                _copy_atomic(checkpoint_dir / filename, destination / filename)
            final = {
                "status": final_status,
                "scope": "production occupant-seed convergence panel",
                "design_sha256": contract["design_sha256"],
                "started_at_utc": started_at,
                "completed_at_utc": _utc_now(),
                "selected_seed_count": checkpoint if converged else None,
                "evaluated_seed_count": checkpoint,
                "selected_occupant_seeds": (
                    list(seed_bank[:checkpoint]) if converged else None
                ),
                "panel_cell_count": len(PANEL_SPECS),
                "weather_member_count": len(weather),
                "run_count": len(diagnostics),
                "first_panel_converged_checkpoint": (
                    checkpoint if converged else None
                ),
                "latest_checkpoint": latest_checkpoint_summary,
                "artifact_sha256": {
                    filename: _sha256_file(destination / filename)
                    for filename in final_files
                },
                "production_interpretation": (
                    "The selected ordered seed prefix may be supplied to the "
                    "full-stock runner with this convergence_results.csv checksum."
                    if converged
                    else "No production seed count was selected; extend or revise the "
                    "predeclared experiment before running the stock."
                ),
            }
            _atomic_json(final, summary_path)
            return final
    raise AssertionError("Convergence checkpoint loop ended unexpectedly.")


def run_convergence_experiment(
    output_dir: str | Path = DEFAULT_CONVERGENCE_OUTPUT_DIR,
    *,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run or resume convergence while holding one output-directory lock."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_execution_lock(destination)
    try:
        return _run_convergence_experiment_unlocked(
            destination, max_workers=max_workers
        )
    except Exception as exc:
        summary_path = destination / "convergence_summary.json"
        try:
            existing = _read_json(summary_path) if summary_path.exists() else {}
            contract_path = destination / "convergence_execution_contract.json"
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
            # Never replace the original coordinator failure with a secondary
            # status-write error; the execution lock is still released below.
            pass
        raise
    finally:
        _release_execution_lock(lock_path)


def convergence_status(
    output_dir: str | Path = DEFAULT_CONVERGENCE_OUTPUT_DIR,
) -> dict[str, Any]:
    """Return a read-only progress snapshot without acquiring the execution lock."""

    destination = Path(output_dir).resolve()
    contract_path = destination / "convergence_execution_contract.json"
    if not contract_path.exists():
        return {"status": "NOT_PREPARED", "output_dir": str(destination)}
    contract = _read_json(contract_path)
    persisted_version = contract.get("convergence_execution_contract_version")
    if persisted_version != CONVERGENCE_EXECUTION_CONTRACT_VERSION:
        return {
            "status": "STALE_CONTRACT",
            "output_dir": str(destination),
            "design_sha256": contract.get("design_sha256"),
            "persisted_contract_version": persisted_version,
            "required_contract_version": CONVERGENCE_EXECUTION_CONTRACT_VERSION,
            "execution_lock_present": (destination / "execution.lock").exists(),
        }
    summary_path = destination / "convergence_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {
        "status": "PREPARED"
    }
    completed_counts: list[int] = []
    failure_count = 0
    partitions_dir = destination / "partitions"
    if partitions_dir.exists():
        for partition_dir in partitions_dir.iterdir():
            if not partition_dir.is_dir():
                continue
            progress_path = partition_dir / "progress.json"
            if progress_path.exists():
                completed_counts.append(
                    int(_read_json(progress_path).get("completed_seed_count", 0))
                )
            failure_path = partition_dir / "last_failure.json"
            if failure_path.exists() and _read_json(failure_path).get("status") == "FAILED":
                failure_count += 1
    histogram = {
        str(value): completed_counts.count(value)
        for value in sorted(set(completed_counts))
    }
    return {
        "status": summary.get("status", "UNKNOWN"),
        "output_dir": str(destination),
        "design_sha256": contract.get("design_sha256"),
        "active_checkpoint": summary.get("active_checkpoint"),
        "selected_seed_count": summary.get("selected_seed_count"),
        "weather_partitions_started": len(completed_counts),
        "weather_partition_seed_count_histogram": histogram,
        "active_failure_count": failure_count,
        "updated_at_utc": summary.get("updated_at_utc"),
        "execution_lock_present": (destination / "execution.lock").exists(),
    }


def load_convergence_selection(
    output_dir: str | Path = DEFAULT_CONVERGENCE_OUTPUT_DIR,
) -> ConvergenceSelection:
    """Load the exact seed/evidence tuple authorized by a converged experiment."""

    destination = Path(output_dir).resolve()
    contract = _read_json(destination / "convergence_execution_contract.json")
    if (
        contract.get("convergence_execution_contract_version")
        != CONVERGENCE_EXECUTION_CONTRACT_VERSION
    ):
        raise MonteCarloContractError(
            "Cannot load a production selection from a stale convergence contract."
        )
    summary = _read_json(destination / "convergence_summary.json")
    if summary.get("status") != "CONVERGED":
        raise MonteCarloContractError(
            "Production occupant seeds are available only after convergence status "
            "is CONVERGED."
        )
    if summary.get("design_sha256") != contract.get("design_sha256"):
        raise MonteCarloContractError(
            "Convergence summary and execution contract design checksums differ."
        )
    try:
        seeds = tuple(int(value) for value in summary["selected_occupant_seeds"])
        selected_count = int(summary["selected_seed_count"])
        first_checkpoint = int(summary["first_panel_converged_checkpoint"])
        expected_sha256 = str(
            summary["artifact_sha256"]["convergence_results.csv"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MonteCarloContractError(
            "Converged summary lacks a valid production seed/evidence selection."
        ) from exc
    declared_bank = tuple(int(value) for value in contract["occupant_seeds"])
    if (
        not seeds
        or len(seeds) != selected_count
        or first_checkpoint != selected_count
        or seeds != declared_bank[:selected_count]
    ):
        raise MonteCarloContractError(
            "Converged summary does not select the exact declared occupant-seed prefix."
        )
    evidence_path = destination / "convergence_results.csv"
    if not evidence_path.is_file() or _sha256_file(evidence_path) != expected_sha256:
        raise MonteCarloContractError(
            "Convergence evidence is missing or differs from its selected checksum."
        )
    return ConvergenceSelection(
        occupant_seeds=seeds,
        convergence_results_path=evidence_path,
        convergence_results_sha256=expected_sha256,
        design_sha256=str(contract["design_sha256"]),
        convergence_rule=ConvergenceRule(),
    )
