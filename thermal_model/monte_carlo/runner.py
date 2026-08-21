"""Reproducible orchestration and a deliberately small Gate-5 pilot run."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from thermal_model.behaviour import (
    BehaviourRequest,
    dwelling_class,
    generate_behaviour,
    load_behaviour_assumptions,
    load_occupant_distribution,
)
from thermal_model.contracts import (
    ArchetypeStateInput,
    load_assumption_contract,
    validate_archetype_state,
)
from thermal_model.validation import load_unique_archetype_states

from .contracts import (
    MODEL_CONTRACT_VERSION,
    ModelScenario,
    MonteCarloContractError,
    WeatherMember,
    archetype_state_sha256,
    canonical_sha256,
    diagnostics_to_record,
    validate_weather_member,
)
from .design import (
    ConvergenceRule,
    DEFAULT_CONVERGENCE_CHECKPOINTS,
    PROSPECTIVE_N160_CONVERGENCE_RULE,
    PROSPECTIVE_N320_N640_CONVERGENCE_RULE,
    build_balanced_manifest,
    convergence_weather_panel_sha256,
    distribution_summary,
    make_seed_bank,
    ordered_seed_bank_sha256,
    paired_model_scenario_deltas,
    paired_renovation_deltas,
    variance_contributions,
)
from .interface import _simulate_with_behaviour
from .aggregation import (
    STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION,
    load_stock_weights,
    stock_distribution_summary,
    validate_stock_weights,
)
from .scenarios import model_scenario_sha256, resolve_model_scenario, scenario_catalog
from .stock_streaming import StreamingStockAccumulator
from .weather import load_weather_catalog, load_weather_members

if TYPE_CHECKING:
    from .supervisor_results import SupervisorResultsSelection


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data/monte_carlo/pilot"
DEFAULT_PRODUCTION_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "data/monte_carlo/production"
)
STREAMING_STOCK_CONTRACT_VERSION = "gate5_streaming_stock_v4"
STREAMING_LOCK_STALE_AFTER_SECONDS = 300.0
PILOT_ARCHETYPE_ID = "BE_TABULA_11"
PILOT_STATE_IDS = ("TABULA_existing", "TABULA_advanced_A_proxy")
PILOT_WEATHER_MEMBER_IDS = (
    "weather_2050_rcp_4_5_pvgis_2010",
    "weather_2050_rcp_4_5_pvgis_2015",
)
PILOT_MODEL_SCENARIOS = ("central", "mass_heavy")
PILOT_SEED_COUNT = 3
PILOT_MASTER_SEED = 20250808


def _resolve_convergence_rule_authorization(
    convergence_rule: ConvergenceRule | None,
    *,
    require_full_stock: bool,
) -> tuple[ConvergenceRule, str]:
    """Resolve a rule and attach its explicit production authorization provenance."""

    default_rule = ConvergenceRule()
    declared_rule = default_rule if convergence_rule is None else convergence_rule
    if convergence_rule is None:
        source = "production_original_default_n80_implicit"
    elif declared_rule == default_rule:
        source = "production_original_default_n80_explicit"
    elif declared_rule == PROSPECTIVE_N160_CONVERGENCE_RULE:
        source = "production_authorized_prospective_n160_confirmation"
    elif declared_rule == PROSPECTIVE_N320_N640_CONVERGENCE_RULE:
        source = "production_authorized_prospective_n320_n640_continuation"
    else:
        source = "explicit_custom_partial_workflow"

    if (
        require_full_stock
        and declared_rule != default_rule
        and declared_rule != PROSPECTIVE_N160_CONVERGENCE_RULE
        and declared_rule != PROSPECTIVE_N320_N640_CONVERGENCE_RULE
    ):
        raise MonteCarloContractError(
            "Full-stock production execution requires either the predeclared default "
            "ConvergenceRule (5/10/20/40/80 checkpoints) or the authorized prospective "
            "n=160 confirmation rule (5/10/20/40/80/160 checkpoints), or the "
            "authorized prospective n=320/n=640 continuation rule. All retain the "
            "2% tolerance, original absolute floors/statistics, and two consecutive "
            "passing expansions. Other custom rules are restricted to partial workflow "
            "or sensitivity runs."
        )
    return declared_rule, source


def _streaming_design_qualification(
    *,
    require_full_stock: bool,
    convergence_verified: bool,
    supervisor_results_authenticated: bool = False,
) -> tuple[str, str, str]:
    """Qualify a streaming result without promoting a subset to production PASS."""

    if supervisor_results_authenticated:
        if not require_full_stock or convergence_verified:
            raise MonteCarloContractError(
                "The supervisor-results qualification requires complete building-stock "
                "coverage and a distinct non-convergence fixed-budget authorization."
            )
        return (
            "PRELIMINARY_REPRESENTATIVE_WEATHER_STOCK_COMPLETE",
            "complete weighted 75-state stock under one paired representative "
            "2015 weather member per RCP; within-RCP weather variability excluded",
            "AUTHORITATIVE_BUILDING_STOCK_REPRESENTATIVE_WEATHER_ONLY",
        )
    if not require_full_stock:
        return (
            "PARTIAL_STOCK_WORKFLOW",
            "partial requested stock subset; workflow/verification artifact only",
            "PARTIAL_SUBSET",
        )
    if not convergence_verified:
        return (
            "WORKFLOW_CHECK_ONLY",
            "full-stock streaming workflow without verified seed convergence",
            "AUTHORITATIVE_FULL_STOCK",
        )
    return (
        "PASS",
        "complete authoritative bounded-memory streaming stock design",
        "AUTHORITATIVE_FULL_STOCK",
    )


def _streaming_partition_coverage_status(
    *,
    require_full_stock: bool,
    supervisor_results_authenticated: bool,
) -> str:
    """Return the precise coverage claim permitted in partition metadata."""

    if supervisor_results_authenticated:
        if not require_full_stock:
            raise MonteCarloContractError(
                "Representative-weather supervisor partitions require all weighted "
                "building-stock states."
            )
        return "AUTHORITATIVE_BUILDING_STOCK_REPRESENTATIVE_WEATHER_ONLY"
    return "AUTHORITATIVE_FULL_STOCK" if require_full_stock else "PARTIAL_SUBSET"


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
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


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    serializable = frame.copy(deep=True)
    for column in serializable.select_dtypes(include=["datetimetz", "datetime64"]).columns:
        serializable[column] = serializable[column].map(
            lambda value: value.isoformat() if not pd.isna(value) else ""
        )
    serializable.to_csv(
        temporary,
        index=False,
        # Seventeen significant digits round-trip every IEEE-754 binary64
        # value, so a resumed accumulator reconstructs the same scalar sums as
        # an uninterrupted run.
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(arrays: Mapping[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonteCarloContractError(f"Cannot read checkpoint JSON {path}.") from exc
    if not isinstance(payload, dict):
        raise MonteCarloContractError(f"Checkpoint JSON {path} must contain an object.")
    return payload


def _verify_file(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise MonteCarloContractError(f"Completed {label} artifact is missing: {path}.")
    actual = _sha256_file(path)
    if actual != str(expected_sha256):
        raise MonteCarloContractError(
            f"Completed {label} checksum mismatch: expected {expected_sha256}, got {actual}."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _acquire_streaming_execution_lock(
    destination: Path,
    *,
    purpose: str = "Gate-5 full-stock Monte Carlo coordinator",
) -> Path:
    """Create the single-writer lock for one production output directory."""

    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / "execution.lock"
    payload = {
        "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
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
                continue
            except (KeyError, TypeError, ValueError, MonteCarloContractError) as exc:
                raise MonteCarloContractError(
                    "The stock execution lock is malformed or still being initialized; "
                    "refusing to remove it automatically."
                ) from exc
            try:
                os.kill(existing_pid, 0)
            except ProcessLookupError as exc:
                age_seconds = (
                    datetime.now(timezone.utc) - started_at.astimezone(timezone.utc)
                ).total_seconds()
                if age_seconds < STREAMING_LOCK_STALE_AFTER_SECONDS:
                    raise MonteCarloContractError(
                        "The stock execution lock belongs to a missing PID but is too "
                        "recent to remove safely."
                    ) from exc
                if attempt == 0:
                    lock_path.unlink(missing_ok=True)
                    continue
                raise MonteCarloContractError(
                    "Cannot replace a stale stock execution lock."
                ) from exc
            except PermissionError as exc:
                raise MonteCarloContractError(
                    "Cannot determine whether the stock coordinator lock is active."
                ) from exc
            raise MonteCarloContractError(
                "A full-stock coordinator is already running for this output "
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
    raise MonteCarloContractError("Unable to acquire the stock execution lock.")


def _release_streaming_execution_lock(lock_path: Path) -> None:
    """Remove only the lock owned by this coordinator process."""

    try:
        payload = _read_json(lock_path)
    except (FileNotFoundError, MonteCarloContractError):
        return
    if int(payload.get("pid", -1)) == os.getpid():
        lock_path.unlink(missing_ok=True)


def _mark_streaming_failure_recovered(
    failure_path: Path, completed_seed_count: int
) -> None:
    """Retire a stale failure marker after its seed prefix is authoritative."""

    if not failure_path.exists():
        return
    failure = _read_json(failure_path)
    if failure.get("status") != "FAILED":
        return
    _atomic_json(
        {
            "status": "RECOVERED",
            "recovered_at_completed_seed_count": completed_seed_count,
            "failure": failure,
        },
        failure_path,
    )


def _strict_boolean(series: pd.Series, *, column: str) -> pd.Series:
    """Parse a persisted CSV Boolean column without truthy-string coercion."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true": True, "false": False}
    if not normalized.isin(allowed).all():
        invalid = sorted(series.loc[~normalized.isin(allowed)].astype(str).unique())
        raise MonteCarloContractError(
            f"Convergence evidence column {column!r} contains invalid Boolean "
            f"values: {invalid}."
        )
    return normalized.map(allowed).astype(bool)


_CONVERGENCE_GROUP_COLUMNS = (
    "archetype_id",
    "state_id",
    "climate_scenario_id",
    "model_scenario_id",
)
_CONVERGENCE_CONTRACT_COLUMNS = (
    "model_contract_version",
    "central_thermal_assumptions_sha256",
    "behaviour_assumptions_sha256",
    "occupant_distribution_sha256",
)
_CONVERGENCE_GROUP_HASH_COLUMNS = (
    "archetype_state_sha256",
    "model_scenario_sha256",
    "weather_panel_sha256",
)


def _normalise_convergence_rule(rule: ConvergenceRule) -> dict[str, Any]:
    """Validate and canonicalise the rule against which evidence is audited."""

    if not isinstance(rule, ConvergenceRule):
        raise MonteCarloContractError(
            "convergence_rule must be a ConvergenceRule instance."
        )
    raw_checkpoints = tuple(rule.checkpoints)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_checkpoints
    ):
        raise MonteCarloContractError(
            "Declared convergence checkpoints must be integers."
        )
    checkpoints = tuple(int(value) for value in raw_checkpoints)
    if (
        not checkpoints
        or checkpoints[0] <= 0
        or tuple(sorted(set(checkpoints))) != checkpoints
    ):
        raise MonteCarloContractError(
            "Declared convergence checkpoints must be unique, positive and increasing."
        )
    tolerance = float(rule.relative_tolerance)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise MonteCarloContractError(
            "Declared convergence tolerance must be finite and within (0, 1)."
        )
    expansions = rule.required_consecutive_expansions
    if (
        isinstance(expansions, (bool, np.bool_))
        or not isinstance(expansions, (int, np.integer))
        or int(expansions) <= 0
    ):
        raise MonteCarloContractError(
            "Declared required consecutive expansions must be a positive integer."
        )
    raw_metrics = tuple(rule.metrics_and_absolute_floors)
    if any(
        not isinstance(item, (tuple, list)) or len(item) != 2
        for item in raw_metrics
    ):
        raise MonteCarloContractError(
            "Declared convergence metrics must be non-empty metric/floor pairs."
        )
    metric_names = [item[0] for item in raw_metrics]
    if not metric_names or any(
        not isinstance(name, str) or not name.strip() for name in metric_names
    ):
        raise MonteCarloContractError(
            "Declared convergence metrics must be non-empty metric/floor pairs."
        )
    if len(set(metric_names)) != len(metric_names):
        raise MonteCarloContractError("Declared convergence metrics must be unique.")
    metrics: list[dict[str, Any]] = []
    for name, raw_floor in raw_metrics:
        floor = float(raw_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise MonteCarloContractError(
                f"Declared convergence floor for {name!r} must be finite and positive."
            )
        metrics.append({"metric": name, "absolute_floor": floor})
    statistics = tuple(rule.statistics)
    if (
        not statistics
        or any(not isinstance(item, str) for item in statistics)
        or len(set(statistics)) != len(statistics)
        or any(item not in {"mean", "median", "p95"} for item in statistics)
    ):
        raise MonteCarloContractError(
            "Declared convergence statistics must be unique supported values "
            "('mean', 'median', 'p95')."
        )
    return {
        "checkpoints": list(checkpoints),
        "relative_tolerance": tolerance,
        "required_consecutive_expansions": int(expansions),
        "metrics_and_absolute_floors": metrics,
        "statistics": list(statistics),
    }


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise MonteCarloContractError(f"{label} must be a valid SHA-256 digest.")
    return digest


def _float_equal(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=1.0e-12, atol=1.0e-12))


def _validate_convergence_evidence(
    seeds: tuple[int, ...],
    *,
    convergence_results_path: str | Path | None,
    convergence_results_sha256: str | None,
    require_convergence_evidence: bool,
    convergence_rule: ConvergenceRule,
    convergence_rule_source: str,
    expected_climate_scenario_ids: Sequence[str],
    expected_contract_provenance: Mapping[str, str],
    expected_archetype_state_sha256: Mapping[tuple[str, str], str],
    expected_model_scenario_sha256: Mapping[str, str],
    expected_weather_panel_sha256: Mapping[str, str],
    require_panel_matches_execution: bool,
) -> tuple[dict[str, Any], bytes | None]:
    """Authenticate and independently reconstruct a seed-count decision."""

    rule = _normalise_convergence_rule(convergence_rule)
    rule_sha256 = canonical_sha256(rule)
    expected_rcps = tuple(sorted(set(str(item) for item in expected_climate_scenario_ids)))
    if not expected_rcps or any(not item.strip() for item in expected_rcps):
        raise MonteCarloContractError(
            "Selected climate-scenario identifiers are required for convergence audit."
        )
    if set(expected_contract_provenance) != set(_CONVERGENCE_CONTRACT_COLUMNS):
        raise MonteCarloContractError(
            "Expected convergence contract provenance must contain exactly "
            f"{list(_CONVERGENCE_CONTRACT_COLUMNS)}."
        )
    expected_contract = {
        column: str(expected_contract_provenance[column]).strip()
        for column in _CONVERGENCE_CONTRACT_COLUMNS
    }
    if not expected_contract["model_contract_version"]:
        raise MonteCarloContractError("Expected model contract version must not be blank.")
    for column in _CONVERGENCE_CONTRACT_COLUMNS[1:]:
        expected_contract[column] = _require_sha256(
            expected_contract[column], label=f"expected {column}"
        )

    supplied = convergence_results_path is not None or convergence_results_sha256 is not None
    if not supplied:
        if require_convergence_evidence:
            raise MonteCarloContractError(
                "Production streaming execution requires convergence evidence: supply "
                "convergence_results_path and convergence_results_sha256. Use "
                "require_convergence_evidence=False only for an explicitly labelled "
                "workflow/test run."
            )
        return (
            {
                "status": "NOT_VERIFIED_BY_RUNNER",
                "required": False,
                "selected_occupant_seed_count": None,
                "selected_occupant_seed_bank_sha256": None,
                "convergence_results_sha256": None,
                "convergence_rule": rule,
                "convergence_rule_sha256": rule_sha256,
                "convergence_rule_source": convergence_rule_source,
                "expected_contract_provenance": expected_contract,
                "panel_matches_execution_required": bool(
                    require_panel_matches_execution
                ),
            },
            None,
        )
    if convergence_results_path is None or convergence_results_sha256 is None:
        raise MonteCarloContractError(
            "Convergence evidence is incomplete; both convergence_results_path and "
            "convergence_results_sha256 are required."
        )

    evidence_path = Path(convergence_results_path).resolve()
    expected_sha256 = _require_sha256(
        convergence_results_sha256, label="convergence_results_sha256"
    )
    _verify_file(evidence_path, expected_sha256, label="convergence-results")
    payload = evidence_path.read_bytes()
    try:
        evidence = pd.read_csv(evidence_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MonteCarloContractError(
            f"Cannot parse convergence evidence CSV {evidence_path}."
        ) from exc
    required_columns = {
        *_CONVERGENCE_GROUP_COLUMNS,
        *_CONVERGENCE_CONTRACT_COLUMNS,
        *_CONVERGENCE_GROUP_HASH_COLUMNS,
        "occupant_seed_bank_count",
        "occupant_seed_bank_sha256",
        "occupant_seed_prefix_sha256",
        "seed_count",
        "previous_seed_count",
        "metric",
        "statistic",
        "value",
        "previous_value",
        "absolute_floor",
        "relative_change",
        "relative_tolerance",
        "criterion_pass",
        "all_statistics_pass",
        "consecutive_passing_expansions",
        "required_consecutive_expansions",
        "converged_at_checkpoint",
        "panel_all_groups_statistics_pass",
        "panel_consecutive_passing_expansions",
        "panel_converged_at_checkpoint",
    }
    missing = sorted(required_columns.difference(evidence.columns))
    if missing:
        raise MonteCarloContractError(
            f"Convergence evidence is missing provenance/decision columns: {missing}."
        )
    if evidence.empty:
        raise MonteCarloContractError("Convergence evidence contains no rows.")

    evidence = evidence.copy(deep=True)
    for column in _CONVERGENCE_GROUP_COLUMNS:
        if evidence[column].isna().any():
            raise MonteCarloContractError(
                f"Convergence panel identity column {column!r} contains missing values."
            )
        evidence[column] = evidence[column].astype(str).str.strip()
        if evidence[column].eq("").any():
            raise MonteCarloContractError(
                f"Convergence panel identity column {column!r} contains blank values."
            )
    for column in _CONVERGENCE_CONTRACT_COLUMNS:
        if evidence[column].isna().any():
            raise MonteCarloContractError(
                f"Convergence contract column {column!r} contains missing values."
            )
        observed = evidence[column].astype(str).str.strip()
        if observed.nunique() != 1 or str(observed.iloc[0]) != expected_contract[column]:
            raise MonteCarloContractError(
                f"Convergence evidence {column!r} does not match the current model "
                "contract provenance."
            )
        evidence[column] = observed
    for column in _CONVERGENCE_GROUP_HASH_COLUMNS:
        if evidence[column].isna().any():
            raise MonteCarloContractError(
                f"Convergence panel provenance column {column!r} contains missing values."
            )
        evidence[column] = evidence[column].map(
            lambda value: _require_sha256(value, label=column)
        )

    integer_columns = (
        "occupant_seed_bank_count",
        "seed_count",
        "consecutive_passing_expansions",
        "required_consecutive_expansions",
        "panel_consecutive_passing_expansions",
    )
    for column in integer_columns:
        values = pd.to_numeric(evidence[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise MonteCarloContractError(
                f"Convergence evidence column {column!r} must contain integers."
            )
        evidence[column] = values.astype(int)
    previous_seed_count = pd.to_numeric(
        evidence["previous_seed_count"], errors="coerce"
    )
    evidence["previous_seed_count"] = previous_seed_count
    for column in (
        "value",
        "previous_value",
        "absolute_floor",
        "relative_change",
        "relative_tolerance",
    ):
        evidence[column] = pd.to_numeric(evidence[column], errors="coerce")
    for column in (
        "criterion_pass",
        "all_statistics_pass",
        "converged_at_checkpoint",
        "panel_all_groups_statistics_pass",
        "panel_converged_at_checkpoint",
    ):
        evidence[column] = _strict_boolean(evidence[column], column=column)

    bank_counts = evidence["occupant_seed_bank_count"]
    if bank_counts.nunique() != 1 or int(bank_counts.iloc[0]) <= 0:
        raise MonteCarloContractError(
            "Convergence evidence has invalid or inconsistent seed-bank counts."
        )
    bank_count = int(bank_counts.iloc[0])
    expected_checkpoints = tuple(
        value for value in rule["checkpoints"] if value <= bank_count
    )
    observed_checkpoints = tuple(sorted(evidence["seed_count"].unique()))
    if observed_checkpoints != expected_checkpoints:
        raise MonteCarloContractError(
            "Convergence evidence checkpoints do not exactly match the declared "
            f"active rule: got {observed_checkpoints}, expected {expected_checkpoints}."
        )

    declared_metrics = {
        item["metric"]: float(item["absolute_floor"])
        for item in rule["metrics_and_absolute_floors"]
    }
    declared_statistics = tuple(rule["statistics"])
    expected_pairs = {
        (metric, statistic)
        for metric in declared_metrics
        for statistic in declared_statistics
    }
    evidence["metric"] = evidence["metric"].astype(str)
    evidence["statistic"] = evidence["statistic"].astype(str)
    if set(zip(evidence["metric"], evidence["statistic"])) != expected_pairs:
        raise MonteCarloContractError(
            "Convergence evidence does not contain the exact declared metric-by-"
            "statistic set."
        )
    duplicate_key = [
        *_CONVERGENCE_GROUP_COLUMNS,
        "seed_count",
        "metric",
        "statistic",
    ]
    if evidence.duplicated(duplicate_key).any():
        raise MonteCarloContractError(
            "Convergence evidence contains duplicate panel/checkpoint/statistic rows."
        )

    group_provenance_columns = [
        *_CONVERGENCE_GROUP_COLUMNS,
        *_CONVERGENCE_GROUP_HASH_COLUMNS,
    ]
    panel_groups = evidence[group_provenance_columns].drop_duplicates()
    if panel_groups.duplicated(list(_CONVERGENCE_GROUP_COLUMNS)).any():
        raise MonteCarloContractError(
            "Convergence evidence has conflicting physical/model/weather hashes for "
            "one panel group."
        )
    physical_cells = panel_groups[
        ["archetype_id", "state_id", "archetype_state_sha256"]
    ].drop_duplicates()
    if physical_cells.duplicated(["archetype_id", "state_id"]).any():
        raise MonteCarloContractError(
            "Convergence evidence has conflicting archetype-state hashes."
        )
    if len(physical_cells) < 3:
        raise MonteCarloContractError(
            "Convergence evidence must cover a representative multi-cell physical "
            "panel (at least three archetype/state cells for low/medium/high demand)."
        )
    observed_rcps = tuple(sorted(panel_groups["climate_scenario_id"].unique()))
    if observed_rcps != expected_rcps:
        raise MonteCarloContractError(
            "Convergence evidence must cover exactly every climate scenario selected "
            f"for execution: got {observed_rcps}, expected {expected_rcps}."
        )
    coverage_without_rcp = ("archetype_id", "state_id", "model_scenario_id")
    coverage_by_rcp = {
        rcp: frozenset(
            map(
                tuple,
                panel_groups.loc[
                    panel_groups["climate_scenario_id"] == rcp,
                    list(coverage_without_rcp),
                ].to_numpy(),
            )
        )
        for rcp in observed_rcps
    }
    if len(set(coverage_by_rcp.values())) != 1:
        raise MonteCarloContractError(
            "The representative convergence panel must use identical physical/model "
            "cells for every selected climate scenario."
        )
    weather_hashes_by_rcp = panel_groups.groupby("climate_scenario_id")[
        "weather_panel_sha256"
    ].nunique()
    if not weather_hashes_by_rcp.eq(1).all():
        raise MonteCarloContractError(
            "All physical/model groups within an RCP must share one canonical "
            "weather-panel hash."
        )
    state_hashes = panel_groups.groupby(["archetype_id", "state_id"])[
        "archetype_state_sha256"
    ].nunique()
    if not state_hashes.eq(1).all():
        raise MonteCarloContractError(
            "Archetype-state hashes change across convergence panel groups."
        )
    scenario_hashes = panel_groups.groupby("model_scenario_id")[
        "model_scenario_sha256"
    ].nunique()
    if not scenario_hashes.eq(1).all():
        raise MonteCarloContractError(
            "Model-scenario hashes change across convergence panel groups."
        )
    if require_panel_matches_execution:
        for row in panel_groups.itertuples(index=False):
            state_key = (str(row.archetype_id), str(row.state_id))
            if state_key not in expected_archetype_state_sha256 or (
                str(row.archetype_state_sha256)
                != _require_sha256(
                    expected_archetype_state_sha256[state_key],
                    label=f"expected archetype-state hash for {state_key}",
                )
            ):
                raise MonteCarloContractError(
                    f"Convergence panel physical cell {state_key} is absent from or "
                    "inconsistent with the full-stock execution inputs."
                )
            scenario_id = str(row.model_scenario_id)
            if scenario_id not in expected_model_scenario_sha256 or (
                str(row.model_scenario_sha256)
                != _require_sha256(
                    expected_model_scenario_sha256[scenario_id],
                    label=f"expected model-scenario hash for {scenario_id}",
                )
            ):
                raise MonteCarloContractError(
                    f"Convergence panel model scenario {scenario_id!r} is absent from "
                    "or inconsistent with the full-stock execution inputs."
                )
            rcp = str(row.climate_scenario_id)
            if rcp not in expected_weather_panel_sha256 or (
                str(row.weather_panel_sha256)
                != _require_sha256(
                    expected_weather_panel_sha256[rcp],
                    label=f"expected weather-panel hash for {rcp}",
                )
            ):
                raise MonteCarloContractError(
                    f"Convergence weather panel for {rcp!r} differs from the selected "
                    "full-stock weather members."
                )
    group_records = panel_groups.sort_values(
        list(_CONVERGENCE_GROUP_COLUMNS), kind="stable"
    ).to_dict(orient="records")
    panel_group_sha256 = canonical_sha256(group_records)
    physical_records = physical_cells.sort_values(
        ["archetype_id", "state_id"], kind="stable"
    ).to_dict(orient="records")
    panel_physical_cell_sha256 = canonical_sha256(physical_records)
    panel_contract_sha256 = canonical_sha256(
        {
            "group_columns": list(_CONVERGENCE_GROUP_COLUMNS),
            "groups": group_records,
            "contract_provenance": expected_contract,
            "convergence_rule_sha256": rule_sha256,
        }
    )

    for checkpoint in expected_checkpoints:
        checkpoint_rows = evidence.loc[evidence["seed_count"] == checkpoint]
        checkpoint_groups = checkpoint_rows[group_provenance_columns].drop_duplicates()
        if canonical_sha256(
            checkpoint_groups.sort_values(
                list(_CONVERGENCE_GROUP_COLUMNS), kind="stable"
            ).to_dict(orient="records")
        ) != panel_group_sha256:
            raise MonteCarloContractError(
                f"Convergence panel coverage changes at checkpoint {checkpoint}."
            )
        expected_row_count = len(panel_groups) * len(expected_pairs)
        if len(checkpoint_rows) != expected_row_count:
            raise MonteCarloContractError(
                f"Convergence checkpoint {checkpoint} is incomplete: got "
                f"{len(checkpoint_rows)} rows, expected {expected_row_count}."
            )
        per_group_pairs = [
            set(zip(item["metric"], item["statistic"]))
            for _, item in checkpoint_rows.groupby(
                list(_CONVERGENCE_GROUP_COLUMNS), sort=False
            )
        ]
        if any(pairs != expected_pairs for pairs in per_group_pairs):
            raise MonteCarloContractError(
                f"Convergence checkpoint {checkpoint} lacks the declared metric/"
                "statistic cross-product in at least one panel group."
            )

    if not evidence["value"].map(np.isfinite).all():
        raise MonteCarloContractError("Convergence evidence values must be finite.")
    for metric, floor in declared_metrics.items():
        observed_floors = evidence.loc[
            evidence["metric"] == metric, "absolute_floor"
        ]
        if not observed_floors.map(
            lambda value: np.isfinite(value) and _float_equal(float(value), floor)
        ).all():
            raise MonteCarloContractError(
                f"Convergence evidence floor for {metric!r} differs from the declared rule."
            )
    if not evidence["relative_tolerance"].map(
        lambda value: np.isfinite(value)
        and _float_equal(float(value), float(rule["relative_tolerance"]))
    ).all():
        raise MonteCarloContractError(
            "Convergence evidence tolerance differs from the declared rule."
        )
    if not evidence["required_consecutive_expansions"].eq(
        int(rule["required_consecutive_expansions"])
    ).all():
        raise MonteCarloContractError(
            "Convergence evidence expansion count differs from the declared rule."
        )

    full_hashes = evidence["occupant_seed_bank_sha256"].map(
        lambda value: _require_sha256(value, label="occupant_seed_bank_sha256")
    )
    if full_hashes.nunique() != 1:
        raise MonteCarloContractError(
            "Convergence evidence has inconsistent ordered seed-bank hashes."
        )
    full_hash = str(full_hashes.iloc[0])
    prefix_hash_by_checkpoint: dict[int, str] = {}
    for checkpoint in expected_checkpoints:
        hashes = evidence.loc[
            evidence["seed_count"] == checkpoint, "occupant_seed_prefix_sha256"
        ].map(lambda value: _require_sha256(value, label="occupant_seed_prefix_sha256"))
        if hashes.nunique() != 1:
            raise MonteCarloContractError(
                f"Convergence checkpoint {checkpoint} has inconsistent prefix hashes."
            )
        prefix_hash_by_checkpoint[checkpoint] = str(hashes.iloc[0])
    if bank_count in prefix_hash_by_checkpoint and (
        prefix_hash_by_checkpoint[bank_count] != full_hash
    ):
        raise MonteCarloContractError(
            "The full convergence seed-bank hash differs from its final prefix hash."
        )

    group_convergence: dict[tuple[str, ...], int] = {}
    for group_key, group in evidence.groupby(
        list(_CONVERGENCE_GROUP_COLUMNS), sort=True
    ):
        key = tuple(str(item) for item in group_key)
        consecutive = 0
        previous_values: dict[tuple[str, str], float] = {}
        for checkpoint_index, checkpoint in enumerate(expected_checkpoints):
            rows = group.loc[group["seed_count"] == checkpoint]
            expected_previous_count = (
                None if checkpoint_index == 0 else expected_checkpoints[checkpoint_index - 1]
            )
            computed_passes: list[bool] = []
            for row in rows.itertuples(index=False):
                pair = (str(row.metric), str(row.statistic))
                previous_value = previous_values.get(pair)
                if expected_previous_count is None:
                    if not pd.isna(row.previous_seed_count) or not pd.isna(row.previous_value):
                        raise MonteCarloContractError(
                            "First convergence checkpoint must not declare a previous "
                            "checkpoint or statistic value."
                        )
                    expected_change = np.nan
                    expected_pass = False
                    if not pd.isna(row.relative_change):
                        raise MonteCarloContractError(
                            "First convergence checkpoint must have undefined relative change."
                        )
                else:
                    if pd.isna(row.previous_seed_count) or int(row.previous_seed_count) != (
                        expected_previous_count
                    ):
                        raise MonteCarloContractError(
                            "Convergence previous_seed_count does not follow the declared "
                            "checkpoint sequence."
                        )
                    if previous_value is None or pd.isna(row.previous_value) or not _float_equal(
                        float(row.previous_value), previous_value
                    ):
                        raise MonteCarloContractError(
                            "Convergence previous_value does not match the preceding checkpoint."
                        )
                    expected_change = abs(float(row.value) - previous_value) / max(
                        abs(float(row.value)), declared_metrics[pair[0]]
                    )
                    if pd.isna(row.relative_change) or not _float_equal(
                        float(row.relative_change), expected_change
                    ):
                        raise MonteCarloContractError(
                            "Convergence relative_change is not reproduced by its values."
                        )
                    expected_pass = bool(
                        expected_change <= float(rule["relative_tolerance"])
                    )
                if bool(row.criterion_pass) != expected_pass:
                    raise MonteCarloContractError(
                        "Convergence criterion_pass is inconsistent with the declared rule."
                    )
                computed_passes.append(expected_pass)
                previous_values[pair] = float(row.value)
            group_pass = bool(computed_passes) and all(computed_passes)
            consecutive = consecutive + 1 if group_pass else 0
            group_converged = consecutive >= int(
                rule["required_consecutive_expansions"]
            )
            for column, expected_value in (
                ("all_statistics_pass", group_pass),
                ("consecutive_passing_expansions", consecutive),
                ("converged_at_checkpoint", group_converged),
            ):
                if rows[column].nunique() != 1 or rows[column].iloc[0] != expected_value:
                    raise MonteCarloContractError(
                        f"Convergence {column} is inconsistent for panel group {key} "
                        f"at checkpoint {checkpoint}."
                    )
            if group_converged and key not in group_convergence:
                group_convergence[key] = checkpoint

    panel_consecutive = 0
    recomputed_panel_flags: dict[int, bool] = {}
    for checkpoint in expected_checkpoints:
        rows = evidence.loc[evidence["seed_count"] == checkpoint]
        panel_pass = bool(rows["criterion_pass"].all())
        panel_consecutive = panel_consecutive + 1 if panel_pass else 0
        panel_converged = panel_consecutive >= int(
            rule["required_consecutive_expansions"]
        )
        recomputed_panel_flags[checkpoint] = panel_converged
        for column, expected_value in (
            ("panel_all_groups_statistics_pass", panel_pass),
            ("panel_consecutive_passing_expansions", panel_consecutive),
            ("panel_converged_at_checkpoint", panel_converged),
        ):
            if rows[column].nunique() != 1 or rows[column].iloc[0] != expected_value:
                raise MonteCarloContractError(
                    f"Convergence {column} is inconsistent with the reconstructed "
                    f"panel decision at checkpoint {checkpoint}."
                )
    converged_checkpoints = [
        checkpoint
        for checkpoint, converged in recomputed_panel_flags.items()
        if converged
    ]
    if not converged_checkpoints:
        raise MonteCarloContractError(
            "Convergence evidence does not satisfy the declared panel stopping rule."
        )
    selected_count = int(min(converged_checkpoints))
    if len(seeds) != selected_count:
        raise MonteCarloContractError(
            "Streaming occupant seed count does not equal the first independently "
            f"reconstructed converged checkpoint: got {len(seeds)}, expected "
            f"{selected_count}."
        )
    expected_selected_hash = prefix_hash_by_checkpoint[selected_count]
    actual_selected_hash = ordered_seed_bank_sha256(seeds)
    if actual_selected_hash != expected_selected_hash:
        raise MonteCarloContractError(
            "Streaming occupant seeds do not match the exact ordered seed prefix "
            "selected by the convergence evidence."
        )
    return (
        {
            "status": "VERIFIED",
            "required": bool(require_convergence_evidence),
            "source_path": str(evidence_path),
            "convergence_results_sha256": expected_sha256,
            "first_panel_converged_checkpoint": selected_count,
            "selected_occupant_seed_count": selected_count,
            "selected_occupant_seed_bank_sha256": actual_selected_hash,
            "convergence_experiment_seed_bank_count": bank_count,
            "convergence_experiment_seed_bank_sha256": full_hash,
            "convergence_rule": rule,
            "convergence_rule_sha256": rule_sha256,
            "convergence_rule_source": convergence_rule_source,
            "model_contract_provenance": expected_contract,
            "panel_matches_execution_required": bool(
                require_panel_matches_execution
            ),
            "panel_execution_match_status": (
                "VERIFIED"
                if require_panel_matches_execution
                else "NOT_REQUIRED_PARTIAL_WORKFLOW"
            ),
            "representative_panel_group_columns": list(
                _CONVERGENCE_GROUP_COLUMNS
            ),
            "representative_panel_group_count": len(panel_groups),
            "representative_panel_group_sha256": panel_group_sha256,
            "representative_panel_contract_sha256": panel_contract_sha256,
            "representative_physical_cell_count": len(physical_cells),
            "representative_physical_cell_sha256": panel_physical_cell_sha256,
            "representative_panel_climate_scenario_ids": list(observed_rcps),
            "representative_panel_model_scenario_ids": sorted(
                panel_groups["model_scenario_id"].unique().tolist()
            ),
        },
        payload,
    )


def execute_balanced_design(
    archetype_states: Sequence[ArchetypeStateInput],
    weather_members: Sequence[WeatherMember],
    occupant_seeds: Sequence[int],
    model_scenarios: Sequence[str | ModelScenario] = ("central",),
    *,
    retain_hourly_run_ids: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Execute a complete small design while caching class-equivalent behaviour.

    The returned diagnostics contain every run.  Hourly data are retained only
    for explicitly requested run IDs, preventing accidental creation of a
    multi-billion-row artifact in production designs.
    """

    states = tuple(validate_archetype_state(item.__dict__) for item in archetype_states)
    weather = tuple(validate_weather_member(item) for item in weather_members)
    raw_seeds = tuple(occupant_seeds)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_seeds
    ):
        raise MonteCarloContractError("Occupant seeds must be integers, not coerced values.")
    seeds = tuple(int(value) for value in raw_seeds)
    scenarios = tuple(resolve_model_scenario(value) for value in model_scenarios)
    manifest = build_balanced_manifest(states, weather, seeds, scenarios)
    retain = set(str(value) for value in retain_hourly_run_ids)
    unknown_retain = retain - set(manifest["run_id"])
    if unknown_retain:
        raise MonteCarloContractError(
            f"Requested hourly run IDs are absent from the manifest: {sorted(unknown_retain)}."
        )

    state_by_key = {(item.archetype_id, item.state_id): item for item in states}
    weather_by_id = {item.member_id: item for item in weather}
    scenario_by_id = {item.scenario_id: item for item in scenarios}
    central_thermal = load_assumption_contract()
    behaviour_contract = load_behaviour_assumptions()
    representative_type = {
        "SFH": "Detached house",
        "MFH": "Apartment, enclosed",
    }
    diagnostic_records: list[dict[str, Any]] = []
    retained_hourly: dict[str, pd.DataFrame] = {}
    scheduled = manifest.assign(
        dwelling_class=manifest["dwelling_type"].map(dwelling_class)
    )
    behaviour_keys = ["dwelling_class", "weather_member_id", "occupant_seed"]
    # One profile is retained only for the duration of its complete reuse group.
    # This bounds production memory while still avoiding regeneration across
    # archetypes, renovation states, and structural scenarios.
    for behaviour_key, run_group in scheduled.groupby(
        behaviour_keys, sort=True, dropna=False
    ):
        household_class, member_id, seed_value = behaviour_key
        member = weather_by_id[member_id]
        seed = int(seed_value)
        behaviour = generate_behaviour(
            BehaviourRequest(
                dwelling_type=representative_type[household_class],
                weather=member.frame.copy(deep=True),
                weather_member_id=member.member_id,
                seed=seed,
            ),
            behaviour_contract,
        )
        for row in run_group.sort_values(
            ["archetype_id", "state_id", "model_scenario_id"], kind="stable"
        ).itertuples(index=False):
            state = state_by_key[(row.archetype_id, row.state_id)]
            scenario = scenario_by_id[row.model_scenario_id]
            result = _simulate_with_behaviour(
                state,
                member,
                seed,
                scenario,
                behaviour,
                central_thermal,
            )
            if result.diagnostics.run_id != row.run_id:
                raise MonteCarloContractError(
                    f"Executed run ID {result.diagnostics.run_id} differs from manifest {row.run_id}."
                )
            diagnostic_records.append(diagnostics_to_record(result.diagnostics))
            if row.run_id in retain:
                retained_hourly[row.run_id] = result.hourly.copy(deep=True)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    if len(diagnostics) != len(manifest) or diagnostics["run_id"].duplicated().any():
        raise MonteCarloContractError("Executed diagnostics do not complete the manifest.")
    if set(diagnostics["run_id"]) != set(manifest["run_id"]):
        raise MonteCarloContractError("Executed and planned run-ID sets differ.")
    return manifest, diagnostics, retained_hourly


def _streaming_partition_id(
    member: WeatherMember,
    scenario: ModelScenario,
    *,
    central_thermal_sha256: str,
    behaviour_assumptions_sha256: str,
    occupant_distribution_sha256: str,
) -> str:
    """Return the stable weather-by-structural-scenario partition identity."""

    return "stock_" + canonical_sha256(
        {
            "weather_member_id": member.member_id,
            "weather_contract_sha256": member.weather_contract_sha256,
            "model_scenario_id": scenario.scenario_id,
            "scenario_definition": scenario.definition(),
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "central_thermal_assumptions_sha256": central_thermal_sha256,
            "behaviour_assumptions_sha256": behaviour_assumptions_sha256,
            "occupant_distribution_sha256": occupant_distribution_sha256,
        }
    )[:24]


def _validate_completed_streaming_partition(
    partition_dir: Path,
    *,
    partition_id: str,
    member_id: str,
    scenario_id: str,
    design_sha256: str,
    manifest: pd.DataFrame,
    seeds: tuple[int, ...],
    require_full_stock: bool,
    supervisor_results_authenticated: bool,
) -> dict[str, Any]:
    """Authenticate a terminal partition before it is reused or consolidated."""

    complete_path = partition_dir / "partition_complete.json"
    complete = _read_json(complete_path)
    expected_run_ids = set(manifest["run_id"].astype(str))
    expected_run_id_sha256 = canonical_sha256(
        {"run_ids": sorted(expected_run_ids)}
    )
    expected_coverage = _streaming_partition_coverage_status(
        require_full_stock=require_full_stock,
        supervisor_results_authenticated=supervisor_results_authenticated,
    )
    if (
        complete.get("status") != "PASS"
        or complete.get("stock_coverage_status") != expected_coverage
        or complete.get("streaming_stock_contract_version")
        != STREAMING_STOCK_CONTRACT_VERSION
        or complete.get("stock_partition_provenance_contract_version")
        != STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
        or complete.get("design_sha256") != design_sha256
        or complete.get("partition_id") != partition_id
        or complete.get("weather_member_id") != member_id
        or complete.get("model_scenario_id") != scenario_id
        or complete.get("expected_run_id_sha256") != expected_run_id_sha256
        or int(complete.get("run_count", -1)) != len(manifest)
        or complete.get("occupant_seeds") != list(seeds)
    ):
        raise MonteCarloContractError(
            f"Completed partition {partition_id} does not match the requested design."
        )
    artifacts = complete.get("artifacts")
    required_artifacts = {
        "run_manifest.csv",
        "run_diagnostics.csv",
        "stock_aggregation.csv",
        "stock_contributions.csv",
        "stock_hourly.csv",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts:
        raise MonteCarloContractError(
            f"Completed partition {partition_id} has an incomplete artifact ledger."
        )
    for filename, metadata in artifacts.items():
        if not isinstance(metadata, Mapping):
            raise MonteCarloContractError(
                f"Completed partition {partition_id} has malformed {filename} metadata."
            )
        path = partition_dir / filename
        _verify_file(
            path,
            str(metadata.get("sha256", "")),
            label=f"partition {partition_id} {filename}",
        )
        try:
            expected_rows = int(metadata["row_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MonteCarloContractError(
                f"Completed partition {partition_id} has invalid {filename} row metadata."
            ) from exc
        observed_rows = sum(1 for _ in path.open("rb")) - 1
        if observed_rows != expected_rows:
            raise MonteCarloContractError(
                f"Completed partition {partition_id} {filename} row count changed."
            )
    stored_manifest = pd.read_csv(
        partition_dir / "run_manifest.csv", usecols=["run_id"]
    )
    stored_diagnostics = pd.read_csv(
        partition_dir / "run_diagnostics.csv", usecols=["run_id"]
    )
    for label, frame in (
        ("manifest", stored_manifest),
        ("diagnostics", stored_diagnostics),
    ):
        if frame["run_id"].duplicated().any() or set(
            frame["run_id"].astype(str)
        ) != expected_run_ids:
            raise MonteCarloContractError(
                f"Completed partition {partition_id} has incomplete/duplicate {label} runs."
            )
    _mark_streaming_failure_recovered(
        partition_dir / "last_failure.json", len(seeds)
    )
    return complete


def _execute_streaming_stock_partition_unlocked(
    partition_id: str,
    member_id: str,
    scenario_id: str,
    states: tuple[ArchetypeStateInput, ...],
    seeds: tuple[int, ...],
    weights: pd.DataFrame,
    output_dir: str,
    design_sha256: str,
    require_full_stock: bool,
) -> dict[str, Any]:
    """Run or resume exactly one independently writable stock partition."""

    destination = Path(output_dir).resolve()
    member = load_weather_members((member_id,))[0]
    scenario = resolve_model_scenario(scenario_id)
    central_thermal = load_assumption_contract()
    behaviour_contract = load_behaviour_assumptions()
    _, occupant_distribution_sha256 = load_occupant_distribution()
    reconstructed_partition_id = _streaming_partition_id(
        member,
        scenario,
        central_thermal_sha256=central_thermal.sha256,
        behaviour_assumptions_sha256=behaviour_contract.sha256,
        occupant_distribution_sha256=occupant_distribution_sha256,
    )
    if reconstructed_partition_id != partition_id:
        raise MonteCarloContractError(
            f"Partition identity changed before execution for {member_id}/{scenario_id}."
        )
    design_contract = _read_json(destination / "streaming_design_contract.json")
    if design_contract.get("design_sha256") != design_sha256:
        raise MonteCarloContractError(
            f"Streaming design changed before partition {partition_id}."
        )
    supervisor_results_authenticated = (
        design_contract.get("supervisor_results_provenance") is not None
    )
    partition_coverage_status = _streaming_partition_coverage_status(
        require_full_stock=require_full_stock,
        supervisor_results_authenticated=supervisor_results_authenticated,
    )

    partition_dir = destination / "partitions" / partition_id
    complete_path = partition_dir / "partition_complete.json"
    manifest = build_balanced_manifest(states, [member], seeds, [scenario])
    expected_run_ids = set(manifest["run_id"].astype(str))
    expected_run_id_sha256 = canonical_sha256(
        {"run_ids": sorted(expected_run_ids)}
    )
    if complete_path.exists():
        complete = _validate_completed_streaming_partition(
            partition_dir,
            partition_id=partition_id,
            member_id=member_id,
            scenario_id=scenario_id,
            design_sha256=design_sha256,
            manifest=manifest,
            seeds=seeds,
            require_full_stock=require_full_stock,
            supervisor_results_authenticated=supervisor_results_authenticated,
        )
        return {
            "partition_id": partition_id,
            "weather_member_id": member_id,
            "model_scenario_id": scenario_id,
            "run_count": int(complete["run_count"]),
        }

    accumulator = StreamingStockAccumulator(
        weights, seeds, require_full_stock=require_full_stock
    )
    diagnostics_records: list[dict[str, Any]] = []
    completed_seed_count = 0
    active_slot: int | None = None
    progress_path = partition_dir / "progress.json"
    if progress_path.exists():
        progress = _read_json(progress_path)
        if (
            progress.get("streaming_stock_contract_version")
            != STREAMING_STOCK_CONTRACT_VERSION
            or progress.get("design_sha256") != design_sha256
            or progress.get("partition_id") != partition_id
            or progress.get("expected_run_id_sha256") != expected_run_id_sha256
        ):
            raise MonteCarloContractError(
                f"Progress checkpoint for {partition_id} belongs to another design."
            )
        completed_seed_count = int(progress.get("completed_seed_count", -1))
        if completed_seed_count < 1 or completed_seed_count > len(seeds):
            raise MonteCarloContractError(
                f"Progress checkpoint for {partition_id} has an invalid seed prefix."
            )
        if progress.get("completed_occupant_seeds") != list(
            seeds[:completed_seed_count]
        ):
            raise MonteCarloContractError(
                f"Progress checkpoint for {partition_id} changed seed order."
            )
        active_slot = int(progress.get("active_slot", -1))
        if active_slot not in (0, 1):
            raise MonteCarloContractError(
                f"Progress checkpoint for {partition_id} has an invalid slot."
            )
        arrays_path = partition_dir / f"progress_slot_{active_slot}_arrays.npz"
        diagnostics_path = (
            partition_dir / f"progress_slot_{active_slot}_diagnostics.csv"
        )
        _verify_file(
            arrays_path,
            str(progress.get("arrays_sha256")),
            label=f"partition {partition_id} progress arrays",
        )
        _verify_file(
            diagnostics_path,
            str(progress.get("diagnostics_sha256")),
            label=f"partition {partition_id} progress diagnostics",
        )
        restored_diagnostics = pd.read_csv(
            diagnostics_path, float_precision="round_trip"
        )
        expected_prefix = set(
            manifest.loc[
                manifest["occupant_seed_rank"] <= completed_seed_count, "run_id"
            ].astype(str)
        )
        if (
            restored_diagnostics["run_id"].duplicated().any()
            or set(restored_diagnostics["run_id"].astype(str)) != expected_prefix
            or len(restored_diagnostics) != len(states) * completed_seed_count
        ):
            raise MonteCarloContractError(
                f"Progress checkpoint for {partition_id} is incomplete or duplicated."
            )
        expected_seed_by_run_id = manifest.set_index("run_id")["occupant_seed"]
        restored_seed_by_run_id = restored_diagnostics.set_index("run_id")[
            "occupant_seed"
        ]
        if not restored_seed_by_run_id.astype(int).equals(
            expected_seed_by_run_id.loc[restored_seed_by_run_id.index].astype(int)
        ):
            raise MonteCarloContractError(
                f"Progress checkpoint for {partition_id} changed run/seed identities."
            )
        with np.load(arrays_path, allow_pickle=False) as stored:
            if set(stored.files) != {"timestamp_ns", "heating_W", "cooling_W"}:
                raise MonteCarloContractError(
                    f"Progress arrays for {partition_id} have an invalid schema."
                )
            accumulator.restore(
                restored_diagnostics,
                timestamp_ns=stored["timestamp_ns"],
                heating_W=stored["heating_W"],
                cooling_W=stored["cooling_W"],
                region_order=progress.get("region_order", ()),
            )
        diagnostics_records = restored_diagnostics.to_dict("records")

    expected_by_factor = manifest.set_index(
        ["archetype_id", "state_id", "occupant_seed"]
    )["run_id"].to_dict()
    ordered_states = sorted(
        states, key=lambda item: (item.archetype_id, item.state_id)
    )
    representative_type = {"SFH": "Detached house", "MFH": "Apartment, enclosed"}
    for seed_rank in range(completed_seed_count + 1, len(seeds) + 1):
        seed = seeds[seed_rank - 1]
        behaviour_by_class: dict[str, Any] = {}
        for state in ordered_states:
            household_class = dwelling_class(state.dwelling_type)
            expected_run_id = expected_by_factor[
                (state.archetype_id, state.state_id, seed)
            ]
            try:
                if household_class not in behaviour_by_class:
                    behaviour_by_class[household_class] = generate_behaviour(
                        BehaviourRequest(
                            dwelling_type=representative_type[household_class],
                            weather=member.frame.copy(deep=True),
                            weather_member_id=member.member_id,
                            seed=seed,
                        ),
                        behaviour_contract,
                    )
                result = _simulate_with_behaviour(
                    state,
                    member,
                    seed,
                    scenario,
                    behaviour_by_class[household_class],
                    central_thermal,
                )
                if result.diagnostics.run_id != expected_run_id:
                    raise MonteCarloContractError(
                        f"Executed run ID {result.diagnostics.run_id} differs from "
                        f"partition manifest {expected_run_id}."
                    )
                accumulator.add(result)
                diagnostics_records.append(
                    diagnostics_to_record(result.diagnostics)
                )
            except Exception as exc:
                _atomic_json(
                    {
                        "status": "FAILED",
                        "design_sha256": design_sha256,
                        "partition_id": partition_id,
                        "weather_member_id": member.member_id,
                        "model_scenario_id": scenario.scenario_id,
                        "run_id": expected_run_id,
                        "archetype_id": state.archetype_id,
                        "state_id": state.state_id,
                        "occupant_seed": seed,
                        "occupant_seed_rank": seed_rank,
                        "last_completed_seed_count": seed_rank - 1,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    partition_dir / "last_failure.json",
                )
                raise

        completed = pd.DataFrame.from_records(diagnostics_records)
        expected_prefix = set(
            manifest.loc[
                manifest["occupant_seed_rank"] <= seed_rank, "run_id"
            ].astype(str)
        )
        if completed["run_id"].duplicated().any() or set(
            completed["run_id"].astype(str)
        ) != expected_prefix:
            raise MonteCarloContractError(
                f"Cannot checkpoint incomplete seed prefix {seed_rank} for {partition_id}."
            )
        next_slot = 0 if active_slot is None else 1 - active_slot
        arrays_path = partition_dir / f"progress_slot_{next_slot}_arrays.npz"
        diagnostics_path = (
            partition_dir / f"progress_slot_{next_slot}_diagnostics.csv"
        )
        _atomic_npz(accumulator.snapshot_arrays(), arrays_path)
        _atomic_csv(completed, diagnostics_path)
        progress = {
            "streaming_stock_contract_version": STREAMING_STOCK_CONTRACT_VERSION,
            "design_sha256": design_sha256,
            "partition_id": partition_id,
            "expected_run_id_sha256": expected_run_id_sha256,
            "active_slot": next_slot,
            "completed_seed_count": seed_rank,
            "completed_occupant_seeds": list(seeds[:seed_rank]),
            "completed_run_count": len(completed),
            "region_order": list(accumulator.region_order),
            "arrays_sha256": _sha256_file(arrays_path),
            "diagnostics_sha256": _sha256_file(diagnostics_path),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(progress, progress_path)
        active_slot = next_slot
        _mark_streaming_failure_recovered(
            partition_dir / "last_failure.json", seed_rank
        )

    diagnostics = pd.DataFrame.from_records(diagnostics_records)
    if diagnostics["run_id"].duplicated().any() or set(
        diagnostics["run_id"].astype(str)
    ) != expected_run_ids:
        raise MonteCarloContractError(
            f"Completed execution for {partition_id} is incomplete or duplicated."
        )
    run_order = {
        run_id: index for index, run_id in enumerate(manifest["run_id"])
    }
    diagnostics["_run_order"] = diagnostics["run_id"].map(run_order)
    diagnostics = diagnostics.sort_values("_run_order", kind="stable").drop(
        columns="_run_order"
    )
    stock_summary, stock_hourly, contributions = accumulator.finalize()
    output_frames = {
        "run_manifest.csv": manifest,
        "run_diagnostics.csv": diagnostics,
        "stock_aggregation.csv": stock_summary,
        "stock_contributions.csv": contributions,
        "stock_hourly.csv": stock_hourly,
    }
    for filename, frame in output_frames.items():
        _atomic_csv(frame, partition_dir / filename)
    artifacts = {
        filename: {
            "sha256": _sha256_file(partition_dir / filename),
            "row_count": len(frame),
        }
        for filename, frame in output_frames.items()
    }
    complete = {
        "status": "PASS",
        "stock_coverage_status": partition_coverage_status,
        "streaming_stock_contract_version": STREAMING_STOCK_CONTRACT_VERSION,
        "stock_partition_provenance_contract_version": (
            STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
        ),
        "design_sha256": design_sha256,
        "partition_id": partition_id,
        "weather_member_id": member.member_id,
        "model_scenario_id": scenario.scenario_id,
        "expected_run_id_sha256": expected_run_id_sha256,
        "run_count": len(diagnostics),
        "occupant_seeds": list(seeds),
        "artifacts": artifacts,
    }
    _atomic_json(complete, complete_path)
    _mark_streaming_failure_recovered(
        partition_dir / "last_failure.json", len(seeds)
    )
    _validate_completed_streaming_partition(
        partition_dir,
        partition_id=partition_id,
        member_id=member_id,
        scenario_id=scenario_id,
        design_sha256=design_sha256,
        manifest=manifest,
        seeds=seeds,
        require_full_stock=require_full_stock,
        supervisor_results_authenticated=supervisor_results_authenticated,
    )
    return {
        "partition_id": partition_id,
        "weather_member_id": member_id,
        "model_scenario_id": scenario_id,
        "run_count": len(diagnostics),
    }


def _execute_streaming_stock_partition(
    partition_id: str,
    member_id: str,
    scenario_id: str,
    states: tuple[ArchetypeStateInput, ...],
    seeds: tuple[int, ...],
    weights: pd.DataFrame,
    output_dir: str,
    design_sha256: str,
    require_full_stock: bool,
) -> dict[str, Any]:
    """Hold a per-partition lock around one worker's restartable execution."""

    partition_dir = Path(output_dir).resolve() / "partitions" / partition_id
    lock_path = _acquire_streaming_execution_lock(
        partition_dir,
        purpose=f"Gate-5 stock partition worker {partition_id}",
    )
    try:
        return _execute_streaming_stock_partition_unlocked(
            partition_id,
            member_id,
            scenario_id,
            states,
            seeds,
            weights,
            output_dir,
            design_sha256,
            require_full_stock,
        )
    finally:
        _release_streaming_execution_lock(lock_path)


def _advance_streaming_stock_partitions(
    partition_specs: Sequence[Mapping[str, str]],
    states: tuple[ArchetypeStateInput, ...],
    seeds: tuple[int, ...],
    weights: pd.DataFrame,
    destination: Path,
    design_sha256: str,
    require_full_stock: bool,
    max_workers: int,
) -> None:
    """Advance independent partitions with one writer assigned to each path."""

    total = len(partition_specs)
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] advancing {total} stock "
        f"partitions with {max_workers} worker(s)",
        flush=True,
    )
    arguments = [
        (
            str(spec["partition_id"]),
            str(spec["weather_member_id"]),
            str(spec["model_scenario_id"]),
            states,
            seeds,
            weights,
            str(destination),
            design_sha256,
            require_full_stock,
        )
        for spec in partition_specs
    ]
    if max_workers == 1:
        for index, arguments_for_partition in enumerate(arguments, start=1):
            _execute_streaming_stock_partition(*arguments_for_partition)
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] stock partitions: "
                f"{index}/{total} complete",
                flush=True,
            )
        return
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _execute_streaming_stock_partition, *arguments_for_partition
            ): arguments_for_partition[0]
            for arguments_for_partition in arguments
        }
        completed = 0
        for future in as_completed(futures):
            partition_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise MonteCarloContractError(
                    f"Streaming stock partition {partition_id} failed."
                ) from exc
            completed += 1
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] stock partitions: "
                f"{completed}/{total} complete",
                flush=True,
            )


def _execute_streaming_stock_design_unlocked(
    archetype_states: Sequence[ArchetypeStateInput],
    weather_members: Sequence[WeatherMember],
    occupant_seeds: Sequence[int],
    model_scenarios: Sequence[str | ModelScenario] = ("central",),
    *,
    output_dir: str | Path = DEFAULT_PRODUCTION_OUTPUT_DIR,
    stock_weights: pd.DataFrame | None = None,
    require_full_stock: bool = True,
    convergence_results_path: str | Path | None = None,
    convergence_results_sha256: str | None = None,
    require_convergence_evidence: bool = True,
    convergence_rule: ConvergenceRule | None = None,
    supervisor_results_selection: "SupervisorResultsSelection | None" = None,
    max_workers: int = 1,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Execute and persist a bounded-memory, incrementally restartable stock design.

    Work is partitioned by weather member and structural model scenario.  Only
    regional/national cumulative hourly arrays are retained.  Progress is
    committed after every complete occupant seed through alternating checkpoint
    slots; a failed seed is rerun, while all earlier verified seeds are reused.
    With ``prepare_only=True``, the exact authenticated design and convergence
    evidence are persisted, but no partition worker is started.
    """

    destination = Path(output_dir).resolve()
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= 8
    ):
        raise MonteCarloContractError(
            "Streaming-stock max_workers must be an integer from 1 to 8."
        )
    states = tuple(validate_archetype_state(item.__dict__) for item in archetype_states)
    members = tuple(validate_weather_member(item) for item in weather_members)
    raw_seeds = tuple(occupant_seeds)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_seeds
    ):
        raise MonteCarloContractError("Occupant seeds must be integers, not coerced values.")
    seeds = tuple(int(value) for value in raw_seeds)
    scenarios = tuple(resolve_model_scenario(value) for value in model_scenarios)
    if not states or not members or not seeds or not scenarios:
        raise MonteCarloContractError(
            "Streaming stock execution needs states, weather, seeds, and scenarios."
        )
    if len(set(seeds)) != len(seeds) or any(
        value < 0 or value > 2**32 - 1 for value in seeds
    ):
        raise MonteCarloContractError("Occupant seeds must be unique uint32 values.")
    supervisor_route = supervisor_results_selection is not None
    if supervisor_route:
        if not require_full_stock:
            raise MonteCarloContractError(
                "Supervisor results require all 75 weighted building-stock states."
            )
        if (
            convergence_results_path is not None
            or convergence_results_sha256 is not None
            or convergence_rule is not None
            or not require_convergence_evidence
        ):
            raise MonteCarloContractError(
                "Supervisor results use their separate authenticated fixed-budget "
                "selection; standard convergence evidence/rules cannot be mixed into it."
            )
        if seeds != tuple(supervisor_results_selection.occupant_seeds):
            raise MonteCarloContractError(
                "Supervisor stock seeds differ from the authenticated 160-seed prefix."
            )
        declared_convergence_rule = None
        convergence_rule_source = "not_applicable_fixed_budget_nonconvergence"
    else:
        declared_convergence_rule, convergence_rule_source = (
            _resolve_convergence_rule_authorization(
                convergence_rule,
                require_full_stock=require_full_stock,
            )
        )
    central_thermal = load_assumption_contract()
    behaviour_contract = load_behaviour_assumptions()
    _, occupant_distribution_sha256 = load_occupant_distribution()
    expected_state_hashes = {
        (item.archetype_id, item.state_id): archetype_state_sha256(item)
        for item in states
    }
    expected_scenario_hashes = {
        item.scenario_id: model_scenario_sha256(item, central_thermal.sha256)
        for item in scenarios
    }
    expected_weather_hashes = {
        climate_scenario_id: convergence_weather_panel_sha256(
            climate_scenario_id,
            [
                {
                    "weather_member_id": item.member_id,
                    "weather_contract_sha256": item.weather_contract_sha256,
                    "weather_forcing_sha256": item.forcing_sha256,
                }
                for item in members
                if item.climate_scenario_id == climate_scenario_id
            ],
        )
        for climate_scenario_id in sorted(
            {item.climate_scenario_id for item in members}
        )
    }
    expected_contract_provenance = {
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "central_thermal_assumptions_sha256": central_thermal.sha256,
        "behaviour_assumptions_sha256": behaviour_contract.sha256,
        "occupant_distribution_sha256": occupant_distribution_sha256,
    }
    if supervisor_route:
        convergence_evidence = {
            "status": "ORIGINAL_PROTOCOL_NOT_CONVERGED",
            "required": True,
            "original_protocol_status": "NOT_CONVERGED_AT_N160",
            "original_protocol_converged": False,
            "convergence_rule_source": convergence_rule_source,
            "production_authorization": "separate_authenticated_fixed_budget_selection",
        }
        convergence_payload = None
    else:
        convergence_evidence, convergence_payload = _validate_convergence_evidence(
            seeds,
            convergence_results_path=convergence_results_path,
            convergence_results_sha256=convergence_results_sha256,
            require_convergence_evidence=require_convergence_evidence,
            convergence_rule=declared_convergence_rule,
            convergence_rule_source=convergence_rule_source,
            expected_climate_scenario_ids=tuple(
                item.climate_scenario_id for item in members
            ),
            expected_contract_provenance=expected_contract_provenance,
            expected_archetype_state_sha256=expected_state_hashes,
            expected_model_scenario_sha256=expected_scenario_hashes,
            expected_weather_panel_sha256=expected_weather_hashes,
            require_panel_matches_execution=require_full_stock,
        )
    state_keys = [(item.archetype_id, item.state_id) for item in states]
    member_ids = [item.member_id for item in members]
    scenario_ids = [item.scenario_id for item in scenarios]
    if len(set(state_keys)) != len(state_keys):
        raise MonteCarloContractError("Streaming archetype/state inputs must be unique.")
    if len(set(member_ids)) != len(member_ids):
        raise MonteCarloContractError("Streaming weather members must be unique.")
    weather_cells = [
        (item.climate_scenario_id, item.weather_pair_id) for item in members
    ]
    if len(set(weather_cells)) != len(weather_cells):
        raise MonteCarloContractError(
            "Each selected RCP/weather-pair cell must contain exactly one member."
        )
    pairs_by_rcp = {
        scenario_id: frozenset(
            item.weather_pair_id
            for item in members
            if item.climate_scenario_id == scenario_id
        )
        for scenario_id in sorted({item.climate_scenario_id for item in members})
    }
    if len(set(pairs_by_rcp.values())) > 1:
        raise MonteCarloContractError(
            "Selected RCPs must use identical PVGIS weather-pair sets; "
            f"got {pairs_by_rcp}."
        )
    if len(set(scenario_ids)) != len(scenario_ids):
        raise MonteCarloContractError("Streaming model scenarios must be unique.")

    supplied_weights = load_stock_weights() if stock_weights is None else stock_weights
    weights = validate_stock_weights(
        supplied_weights, require_authoritative_shape=require_full_stock
    )
    weighted_cells = set(
        map(tuple, weights[["archetype_id", "state_id"]].drop_duplicates().to_numpy())
    )
    if set(state_keys) != weighted_cells:
        missing = sorted(weighted_cells - set(state_keys))
        extra = sorted(set(state_keys) - weighted_cells)
        raise MonteCarloContractError(
            f"Streaming states must exactly cover weighted cells; missing={missing}, extra={extra}."
        )
    supervisor_provenance: dict[str, Any] | None = None
    if supervisor_route:
        if tuple(member_ids) != tuple(
            supervisor_results_selection.weather_member_ids
        ):
            raise MonteCarloContractError(
                "Supervisor stock weather members differ from the authenticated "
                "paired 2015 selection or its frozen order."
            )
        from .supervisor_results import validate_supervisor_selection_for_stock

        supervisor_provenance = validate_supervisor_selection_for_stock(
            supervisor_results_selection,
            expected_contract_provenance=expected_contract_provenance,
            expected_archetype_state_sha256=expected_state_hashes,
            expected_model_scenario_sha256=expected_scenario_hashes,
            expected_weather_member_provenance={
                item.member_id: {
                    "climate_scenario_id": item.climate_scenario_id,
                    "weather_pair_id": item.weather_pair_id,
                    "weather_contract_sha256": item.weather_contract_sha256,
                    "weather_forcing_sha256": item.forcing_sha256,
                }
                for item in members
            },
            archetype_state_count=len(states),
            stock_weight_row_count=len(weights),
            stock_weights_sha256=str(weights["stock_weights_sha256"].iloc[0]),
            stock_weights_source_sha256=str(
                weights["stock_weights_source_sha256"].iloc[0]
            ),
            model_scenario_ids=scenario_ids,
        )
    partition_specs: list[dict[str, str]] = []
    for member in sorted(members, key=lambda item: item.member_id):
        for scenario in sorted(scenarios, key=lambda item: item.scenario_id):
            partition_specs.append(
                {
                    "partition_id": _streaming_partition_id(
                        member,
                        scenario,
                        central_thermal_sha256=central_thermal.sha256,
                        behaviour_assumptions_sha256=behaviour_contract.sha256,
                        occupant_distribution_sha256=occupant_distribution_sha256,
                    ),
                    "weather_member_id": member.member_id,
                    "model_scenario_id": scenario.scenario_id,
                }
            )
    design_payload = {
        "streaming_stock_contract_version": STREAMING_STOCK_CONTRACT_VERSION,
        "stock_partition_provenance_contract_version": (
            STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
        ),
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "central_thermal_assumptions_sha256": central_thermal.sha256,
        "behaviour_assumptions_sha256": behaviour_contract.sha256,
        "occupant_distribution_sha256": occupant_distribution_sha256,
        "archetype_states": [
            {
                "archetype_id": item.archetype_id,
                "state_id": item.state_id,
                "archetype_state_sha256": archetype_state_sha256(item),
            }
            for item in sorted(states, key=lambda value: (value.archetype_id, value.state_id))
        ],
        "weather_members": [
            {
                "weather_member_id": item.member_id,
                "climate_scenario_id": item.climate_scenario_id,
                "weather_pair_id": item.weather_pair_id,
                "observed_pvgis_year": item.observed_pvgis_year,
                "climate_target": item.climate_target,
                "member_sha256": item.member_sha256,
                "metadata_sha256": item.metadata_sha256,
                "manifest_sha256": item.manifest_sha256,
                "morph_contract_sha256": item.morph_contract_sha256,
                "facade_source_sha256_json": json.dumps(
                    item.facade_source_sha256,
                    separators=(",", ":"),
                ),
                "weather_contract_sha256": item.weather_contract_sha256,
                "weather_forcing_sha256": item.forcing_sha256,
            }
            for item in sorted(members, key=lambda value: value.member_id)
        ],
        "occupant_seeds": list(seeds),
        "occupant_seed_bank_sha256": ordered_seed_bank_sha256(seeds),
        "convergence_evidence": {
            key: value
            for key, value in convergence_evidence.items()
            if key != "source_path"
        },
        "supervisor_results_provenance": supervisor_provenance,
        "model_scenarios": [
            item.definition() for item in sorted(scenarios, key=lambda value: value.scenario_id)
        ],
        "stock_weights_sha256": str(weights["stock_weights_sha256"].iloc[0]),
        "stock_weights_source_sha256": str(
            weights["stock_weights_source_sha256"].iloc[0]
        ),
        "require_full_stock": bool(require_full_stock),
        "partition_specs": partition_specs,
        "expected_run_count": len(states) * len(members) * len(seeds) * len(scenarios),
    }
    design_sha256 = canonical_sha256(design_payload)
    design_contract = {**design_payload, "design_sha256": design_sha256}
    design_path = destination / "streaming_design_contract.json"
    if design_path.exists():
        existing_design = _read_json(design_path)
        if existing_design.get("design_sha256") != design_sha256 or existing_design != _json_ready(
            design_contract
        ):
            raise MonteCarloContractError(
                "Output directory belongs to a different streaming stock design."
            )
    else:
        _atomic_json(design_contract, design_path)
    persisted_design = _read_json(design_path)
    if (
        persisted_design != _json_ready(design_contract)
        or canonical_sha256(
            {key: value for key, value in persisted_design.items() if key != "design_sha256"}
        )
        != design_sha256
    ):
        raise MonteCarloContractError(
            "Persisted streaming stock design failed its identity check."
        )

    convergence_destination = destination / "convergence_results.csv"
    supervisor_evidence_destination = (
        destination / "representative_weather_n160_stability.csv"
    )
    supervisor_contract_destination = (
        destination / "deadline_n160_supervisor_contract.json"
    )
    supervisor_summary_destination = (
        destination / "deadline_n160_selection_summary.json"
    )
    if supervisor_route:
        if convergence_destination.exists():
            raise MonteCarloContractError(
                "Supervisor-results output cannot contain a convergence_results.csv "
                "that could mislabel its fixed-budget authorization."
            )
        supervisor_archive = (
            (
                supervisor_results_selection.evidence_path,
                supervisor_evidence_destination,
                supervisor_results_selection.evidence_sha256,
                "persisted representative-weather n=160 stability evidence",
            ),
            (
                supervisor_results_selection.contract_path,
                supervisor_contract_destination,
                supervisor_results_selection.contract_sha256,
                "persisted deadline n=160 supervisor contract",
            ),
            (
                supervisor_results_selection.summary_path,
                supervisor_summary_destination,
                supervisor_results_selection.summary_sha256,
                "persisted deadline n=160 selection summary",
            ),
        )
        for source, target, expected_sha256, label in supervisor_archive:
            if not target.exists():
                _atomic_bytes(source.read_bytes(), target)
            _verify_file(
                target,
                expected_sha256,
                label=label,
            )
    elif any(
        path.exists()
        for path in (
            supervisor_evidence_destination,
            supervisor_contract_destination,
            supervisor_summary_destination,
        )
    ):
        raise MonteCarloContractError(
            "A supervisor-results authorization artifact is present in a standard "
            "stock output directory."
        )
    if convergence_payload is not None:
        if convergence_destination.exists():
            _verify_file(
                convergence_destination,
                str(convergence_evidence["convergence_results_sha256"]),
                label="persisted convergence-results",
            )
        else:
            _atomic_bytes(convergence_payload, convergence_destination)
            _verify_file(
                convergence_destination,
                str(convergence_evidence["convergence_results_sha256"]),
                label="persisted convergence-results",
            )
    elif convergence_destination.exists():
        raise MonteCarloContractError(
            "An unverified convergence_results.csv is present in a workflow-only "
            "output directory. Remove it or supply its path and checksum explicitly."
        )

    partitions_dir = destination / "partitions"
    expected_partition_ids = {item["partition_id"] for item in partition_specs}
    if partitions_dir.exists():
        completed_on_disk = {
            path.parent.name
            for path in partitions_dir.glob("*/partition_complete.json")
        }
        unexpected = completed_on_disk - expected_partition_ids
        if unexpected:
            raise MonteCarloContractError(
                f"Output directory contains completed foreign partitions: {sorted(unexpected)}."
            )

    if prepare_only:
        return {
            "status": (
                "PRELIMINARY_REPRESENTATIVE_WEATHER_STOCK_PREPARED"
                if supervisor_route
                else "PREPARED"
            ),
            "scope": (
                "authenticated weighted 75-state building-stock design under one "
                "paired representative 2015 member per RCP; simulation not started"
                if supervisor_route
                else "authenticated full-stock design; simulation not started"
                if require_full_stock
                else "authenticated partial-stock workflow design; simulation not started"
            ),
            "stock_coverage_status": (
                "AUTHORITATIVE_BUILDING_STOCK_REPRESENTATIVE_WEATHER_ONLY"
                if supervisor_route
                else "AUTHORITATIVE_FULL_STOCK"
                if require_full_stock
                else "PARTIAL_SUBSET"
            ),
            "output_dir": str(destination),
            "execution_started": False,
            "design_sha256": design_sha256,
            "design_contract_path": str(design_path),
            "design_contract_file_sha256": _sha256_file(design_path),
            "archetype_state_count": len(states),
            "weather_member_count": len(members),
            "occupant_seed_count": len(seeds),
            "model_scenario_count": len(scenarios),
            "model_scenario_ids": [item.scenario_id for item in scenarios],
            "partition_count": len(partition_specs),
            "runs_per_partition": len(states) * len(seeds),
            "expected_run_count": design_payload["expected_run_count"],
            "requested_max_workers": max_workers,
            "convergence_evidence": convergence_evidence,
            "supervisor_results_provenance": supervisor_provenance,
            "stock_weights_sha256": design_payload["stock_weights_sha256"],
            "stock_weights_source_sha256": design_payload[
                "stock_weights_source_sha256"
            ],
        }

    _advance_streaming_stock_partitions(
        partition_specs,
        states,
        seeds,
        weights,
        destination,
        design_sha256,
        require_full_stock,
        max_workers,
    )

    state_by_key = {(item.archetype_id, item.state_id): item for item in states}
    member_by_id = {item.member_id: item for item in members}
    scenario_by_id = {item.scenario_id: item for item in scenarios}
    representative_type = {"SFH": "Detached house", "MFH": "Apartment, enclosed"}
    partition_coverage_status = _streaming_partition_coverage_status(
        require_full_stock=require_full_stock,
        supervisor_results_authenticated=supervisor_route,
    )
    partition_index_records: list[dict[str, Any]] = []
    stock_summaries: list[pd.DataFrame] = []
    stock_contributions: list[pd.DataFrame] = []

    for spec in partition_specs:
        partition_id = spec["partition_id"]
        member = member_by_id[spec["weather_member_id"]]
        scenario = scenario_by_id[spec["model_scenario_id"]]
        partition_dir = partitions_dir / partition_id
        complete_path = partition_dir / "partition_complete.json"
        manifest = build_balanced_manifest(states, [member], seeds, [scenario])
        expected_run_ids = set(manifest["run_id"].astype(str))
        expected_run_id_sha256 = canonical_sha256(
            {"run_ids": sorted(expected_run_ids)}
        )

        if complete_path.exists():
            complete = _read_json(complete_path)
            if (
                complete.get("status") != "PASS"
                or complete.get("stock_coverage_status")
                != partition_coverage_status
                or complete.get("design_sha256") != design_sha256
                or complete.get("partition_id") != partition_id
                or complete.get("expected_run_id_sha256") != expected_run_id_sha256
                or int(complete.get("run_count", -1)) != len(manifest)
            ):
                raise MonteCarloContractError(
                    f"Completed partition {partition_id} does not match the requested design."
                )
            artifacts = complete.get("artifacts")
            if not isinstance(artifacts, dict):
                raise MonteCarloContractError(
                    f"Completed partition {partition_id} has no artifact ledger."
                )
            required_artifacts = {
                "run_manifest.csv",
                "run_diagnostics.csv",
                "stock_aggregation.csv",
                "stock_contributions.csv",
                "stock_hourly.csv",
            }
            if set(artifacts) != required_artifacts:
                raise MonteCarloContractError(
                    f"Completed partition {partition_id} has an incomplete artifact ledger."
                )
            for filename, metadata in artifacts.items():
                _verify_file(
                    partition_dir / filename,
                    str(metadata["sha256"]),
                    label=f"partition {partition_id} {filename}",
                )
            stored_manifest = pd.read_csv(
                partition_dir / "run_manifest.csv", usecols=["run_id"]
            )
            stored_diagnostics = pd.read_csv(
                partition_dir / "run_diagnostics.csv", usecols=["run_id"]
            )
            for label, frame in (
                ("manifest", stored_manifest),
                ("diagnostics", stored_diagnostics),
            ):
                if frame["run_id"].duplicated().any() or set(
                    frame["run_id"].astype(str)
                ) != expected_run_ids:
                    raise MonteCarloContractError(
                        f"Completed partition {partition_id} has incomplete/duplicate {label} runs."
                    )
        else:
            accumulator = StreamingStockAccumulator(
                weights, seeds, require_full_stock=require_full_stock
            )
            diagnostics_records: list[dict[str, Any]] = []
            completed_seed_count = 0
            active_slot: int | None = None
            progress_path = partition_dir / "progress.json"
            if progress_path.exists():
                progress = _read_json(progress_path)
                if (
                    progress.get("design_sha256") != design_sha256
                    or progress.get("partition_id") != partition_id
                    or progress.get("expected_run_id_sha256") != expected_run_id_sha256
                ):
                    raise MonteCarloContractError(
                        f"Progress checkpoint for {partition_id} belongs to another design."
                    )
                completed_seed_count = int(progress.get("completed_seed_count", -1))
                if completed_seed_count < 1 or completed_seed_count > len(seeds):
                    raise MonteCarloContractError(
                        f"Progress checkpoint for {partition_id} has an invalid seed prefix."
                    )
                if progress.get("completed_occupant_seeds") != list(
                    seeds[:completed_seed_count]
                ):
                    raise MonteCarloContractError(
                        f"Progress checkpoint for {partition_id} changed seed order."
                    )
                active_slot = int(progress.get("active_slot", -1))
                if active_slot not in (0, 1):
                    raise MonteCarloContractError(
                        f"Progress checkpoint for {partition_id} has an invalid slot."
                    )
                arrays_path = partition_dir / f"progress_slot_{active_slot}_arrays.npz"
                diagnostics_path = (
                    partition_dir / f"progress_slot_{active_slot}_diagnostics.csv"
                )
                _verify_file(
                    arrays_path,
                    str(progress.get("arrays_sha256")),
                    label=f"partition {partition_id} progress arrays",
                )
                _verify_file(
                    diagnostics_path,
                    str(progress.get("diagnostics_sha256")),
                    label=f"partition {partition_id} progress diagnostics",
                )
                restored_diagnostics = pd.read_csv(
                    diagnostics_path,
                    float_precision="round_trip",
                )
                expected_prefix = set(
                    manifest.loc[
                        manifest["occupant_seed_rank"] <= completed_seed_count, "run_id"
                    ].astype(str)
                )
                if (
                    restored_diagnostics["run_id"].duplicated().any()
                    or set(restored_diagnostics["run_id"].astype(str)) != expected_prefix
                    or len(restored_diagnostics) != len(states) * completed_seed_count
                ):
                    raise MonteCarloContractError(
                        f"Progress checkpoint for {partition_id} is incomplete or duplicated."
                    )
                with np.load(arrays_path, allow_pickle=False) as stored:
                    if set(stored.files) != {"timestamp_ns", "heating_W", "cooling_W"}:
                        raise MonteCarloContractError(
                            f"Progress arrays for {partition_id} have an invalid schema."
                        )
                    accumulator.restore(
                        restored_diagnostics,
                        timestamp_ns=stored["timestamp_ns"],
                        heating_W=stored["heating_W"],
                        cooling_W=stored["cooling_W"],
                        region_order=progress.get("region_order", ()),
                    )
                diagnostics_records = restored_diagnostics.to_dict("records")

            expected_by_factor = manifest.set_index(
                ["archetype_id", "state_id", "occupant_seed"]
            )["run_id"].to_dict()
            ordered_states = sorted(
                states, key=lambda item: (item.archetype_id, item.state_id)
            )
            for seed_rank in range(completed_seed_count + 1, len(seeds) + 1):
                seed = seeds[seed_rank - 1]
                behaviour_by_class = {}
                for state in ordered_states:
                    household_class = dwelling_class(state.dwelling_type)
                    expected_run_id = expected_by_factor[
                        (state.archetype_id, state.state_id, seed)
                    ]
                    try:
                        if household_class not in behaviour_by_class:
                            behaviour_by_class[household_class] = generate_behaviour(
                                BehaviourRequest(
                                    dwelling_type=representative_type[household_class],
                                    weather=member.frame.copy(deep=True),
                                    weather_member_id=member.member_id,
                                    seed=seed,
                                ),
                                behaviour_contract,
                            )
                        result = _simulate_with_behaviour(
                            state,
                            member,
                            seed,
                            scenario,
                            behaviour_by_class[household_class],
                            central_thermal,
                        )
                        if result.diagnostics.run_id != expected_run_id:
                            raise MonteCarloContractError(
                                f"Executed run ID {result.diagnostics.run_id} differs from "
                                f"partition manifest {expected_run_id}."
                            )
                        accumulator.add(result)
                        diagnostics_records.append(
                            diagnostics_to_record(result.diagnostics)
                        )
                    except Exception as exc:
                        _atomic_json(
                            {
                                "status": "FAILED",
                                "design_sha256": design_sha256,
                                "partition_id": partition_id,
                                "weather_member_id": member.member_id,
                                "model_scenario_id": scenario.scenario_id,
                                "run_id": expected_run_id,
                                "archetype_id": state.archetype_id,
                                "state_id": state.state_id,
                                "occupant_seed": seed,
                                "occupant_seed_rank": seed_rank,
                                "last_completed_seed_count": seed_rank - 1,
                                "exception_type": type(exc).__name__,
                                "exception_message": str(exc),
                            },
                            partition_dir / "last_failure.json",
                        )
                        raise

                completed = pd.DataFrame.from_records(diagnostics_records)
                expected_prefix = set(
                    manifest.loc[
                        manifest["occupant_seed_rank"] <= seed_rank, "run_id"
                    ].astype(str)
                )
                if completed["run_id"].duplicated().any() or set(
                    completed["run_id"].astype(str)
                ) != expected_prefix:
                    raise MonteCarloContractError(
                        f"Cannot checkpoint incomplete seed prefix {seed_rank} for {partition_id}."
                    )
                next_slot = 0 if active_slot is None else 1 - active_slot
                arrays_path = partition_dir / f"progress_slot_{next_slot}_arrays.npz"
                diagnostics_path = (
                    partition_dir / f"progress_slot_{next_slot}_diagnostics.csv"
                )
                _atomic_npz(accumulator.snapshot_arrays(), arrays_path)
                _atomic_csv(completed, diagnostics_path)
                progress = {
                    "streaming_stock_contract_version": STREAMING_STOCK_CONTRACT_VERSION,
                    "design_sha256": design_sha256,
                    "partition_id": partition_id,
                    "expected_run_id_sha256": expected_run_id_sha256,
                    "active_slot": next_slot,
                    "completed_seed_count": seed_rank,
                    "completed_occupant_seeds": list(seeds[:seed_rank]),
                    "completed_run_count": len(completed),
                    "region_order": list(accumulator.region_order),
                    "arrays_sha256": _sha256_file(arrays_path),
                    "diagnostics_sha256": _sha256_file(diagnostics_path),
                }
                _atomic_json(progress, progress_path)
                active_slot = next_slot
                failure_path = partition_dir / "last_failure.json"
                if failure_path.exists():
                    failure = _read_json(failure_path)
                    if failure.get("status") == "FAILED" and int(
                        failure.get("occupant_seed_rank", -1)
                    ) <= seed_rank:
                        _atomic_json(
                            {
                                "status": "RECOVERED",
                                "recovered_at_completed_seed_count": seed_rank,
                                "failure": failure,
                            },
                            failure_path,
                        )

            diagnostics = pd.DataFrame.from_records(diagnostics_records)
            if diagnostics["run_id"].duplicated().any() or set(
                diagnostics["run_id"].astype(str)
            ) != expected_run_ids:
                raise MonteCarloContractError(
                    f"Completed execution for {partition_id} is incomplete or duplicated."
                )
            run_order = {run_id: index for index, run_id in enumerate(manifest["run_id"])}
            diagnostics["_run_order"] = diagnostics["run_id"].map(run_order)
            diagnostics = diagnostics.sort_values("_run_order", kind="stable").drop(
                columns="_run_order"
            )
            stock_summary, stock_hourly, contributions = accumulator.finalize()
            output_frames = {
                "run_manifest.csv": manifest,
                "run_diagnostics.csv": diagnostics,
                "stock_aggregation.csv": stock_summary,
                "stock_contributions.csv": contributions,
                "stock_hourly.csv": stock_hourly,
            }
            for filename, frame in output_frames.items():
                _atomic_csv(frame, partition_dir / filename)
            artifacts = {
                filename: {
                    "sha256": _sha256_file(partition_dir / filename),
                    "row_count": len(frame),
                }
                for filename, frame in output_frames.items()
            }
            complete = {
                "status": "PASS",
                "stock_coverage_status": partition_coverage_status,
                "streaming_stock_contract_version": STREAMING_STOCK_CONTRACT_VERSION,
                "stock_partition_provenance_contract_version": (
                    STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
                ),
                "design_sha256": design_sha256,
                "partition_id": partition_id,
                "weather_member_id": member.member_id,
                "model_scenario_id": scenario.scenario_id,
                "expected_run_id_sha256": expected_run_id_sha256,
                "run_count": len(diagnostics),
                "occupant_seeds": list(seeds),
                "artifacts": artifacts,
            }
            _atomic_json(complete, complete_path)

        complete = _read_json(complete_path)
        artifacts = complete["artifacts"]
        stock_summary = pd.read_csv(
            partition_dir / "stock_aggregation.csv",
            keep_default_na=False,
            float_precision="round_trip",
        )
        contributions = pd.read_csv(
            partition_dir / "stock_contributions.csv",
            keep_default_na=False,
            float_precision="round_trip",
        )
        stock_summaries.append(stock_summary)
        stock_contributions.append(contributions)
        partition_index_records.append(
            {
                "partition_id": partition_id,
                "weather_member_id": member.member_id,
                "climate_scenario_id": member.climate_scenario_id,
                "weather_pair_id": member.weather_pair_id,
                "observed_pvgis_year": member.observed_pvgis_year,
                "climate_target": member.climate_target,
                "model_scenario_id": scenario.scenario_id,
                "supervisor_results_contract_sha256": (
                    supervisor_provenance["contract_sha256"]
                    if supervisor_provenance is not None
                    else ""
                ),
                "weather_scope": (
                    "paired_2015_representative_weather_only"
                    if supervisor_route
                    else "declared_weather_design"
                ),
                "run_count": int(complete["run_count"]),
                "run_diagnostics_path": str(
                    (partition_dir / "run_diagnostics.csv").relative_to(destination)
                ),
                "run_diagnostics_sha256": artifacts["run_diagnostics.csv"]["sha256"],
                "stock_hourly_path": str(
                    (partition_dir / "stock_hourly.csv").relative_to(destination)
                ),
                "stock_hourly_row_count": artifacts["stock_hourly.csv"]["row_count"],
                "stock_hourly_sha256": artifacts["stock_hourly.csv"]["sha256"],
                "partition_complete_sha256": _sha256_file(complete_path),
            }
        )

    combined_stock = pd.concat(stock_summaries, ignore_index=True)
    combined_contributions = pd.concat(stock_contributions, ignore_index=True)
    stock_key = [
        "weather_member_id",
        "model_scenario_id",
        "stock_scenario_id",
        "region",
    ]
    if combined_stock.duplicated(stock_key).any():
        raise MonteCarloContractError("Consolidated stock results contain duplicate partitions.")
    distributions = stock_distribution_summary(combined_stock)
    partition_index = pd.DataFrame.from_records(partition_index_records)
    consolidated = {
        "partition_index.csv": partition_index,
        "stock_aggregation.csv": combined_stock,
        "stock_contributions.csv": combined_contributions,
        "stock_distribution_summary.csv": distributions,
    }
    for filename, frame in consolidated.items():
        _atomic_csv(frame, destination / filename)
    artifact_sha256 = {
        filename: _sha256_file(destination / filename) for filename in consolidated
    }
    if convergence_payload is not None:
        artifact_sha256["convergence_results.csv"] = _sha256_file(
            convergence_destination
        )
    if supervisor_route:
        for path in (
            supervisor_evidence_destination,
            supervisor_contract_destination,
            supervisor_summary_destination,
        ):
            artifact_sha256[path.name] = _sha256_file(path)
    convergence_verified = convergence_evidence["status"] == "VERIFIED"
    status, scope, stock_coverage_status = _streaming_design_qualification(
        require_full_stock=require_full_stock,
        convergence_verified=convergence_verified,
        supervisor_results_authenticated=supervisor_route,
    )
    summary = {
        "status": status,
        "scope": scope,
        "stock_coverage_status": stock_coverage_status,
        "require_full_stock": bool(require_full_stock),
        "streaming_stock_contract_version": STREAMING_STOCK_CONTRACT_VERSION,
        "stock_partition_provenance_contract_version": (
            STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
        ),
        "design_sha256": design_sha256,
        "partition_count": len(partition_specs),
        "last_coordinator_max_workers": max_workers,
        "completed_run_count": int(sum(item["run_count"] for item in partition_index_records)),
        "expected_run_count": design_payload["expected_run_count"],
        "occupant_seeds": list(seeds),
        "occupant_seed_bank_sha256": design_payload["occupant_seed_bank_sha256"],
        "convergence_evidence": {
            **convergence_evidence,
            "persisted_path": (
                "convergence_results.csv" if convergence_verified else None
            ),
            "persisted_sha256": (
                artifact_sha256.get("convergence_results.csv")
                if convergence_verified
                else None
            ),
        },
        "supervisor_results_provenance": supervisor_provenance,
        "stock_weights_sha256": design_payload["stock_weights_sha256"],
        "stock_weights_source_sha256": design_payload["stock_weights_source_sha256"],
        "stock_hourly_storage": (
            "one checksum-indexed regional/national file per weather/scenario partition"
        ),
        "restart_boundary": "last checksum-verified complete occupant seed per partition",
        "interval_interpretation": (
            "occupant-seed variability conditional on one paired representative 2015 "
            "weather member per RCP; within-RCP weather variability and structural "
            "uncertainty are excluded; not complete prediction intervals"
            if supervisor_route
            else "descriptive empirical intervals over explicitly included uncertainty "
            "axes; not complete prediction intervals"
        ),
        "claims_not_supported": (
            [
                "weather-variance attribution",
                "extreme-weather-year demand",
                "system-reliability adequacy",
                "complete prediction intervals",
            ]
            if supervisor_route
            else []
        ),
        "artifact_sha256": artifact_sha256,
    }
    _atomic_json(summary, destination / "monte_carlo_summary.json")
    return summary


def execute_streaming_stock_design(
    archetype_states: Sequence[ArchetypeStateInput],
    weather_members: Sequence[WeatherMember],
    occupant_seeds: Sequence[int],
    model_scenarios: Sequence[str | ModelScenario] = ("central",),
    *,
    output_dir: str | Path = DEFAULT_PRODUCTION_OUTPUT_DIR,
    stock_weights: pd.DataFrame | None = None,
    require_full_stock: bool = True,
    convergence_results_path: str | Path | None = None,
    convergence_results_sha256: str | None = None,
    require_convergence_evidence: bool = True,
    convergence_rule: ConvergenceRule | None = None,
    supervisor_results_selection: "SupervisorResultsSelection | None" = None,
    max_workers: int = 1,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Prepare or run one locked stock design; partitions may execute in parallel."""

    destination = Path(output_dir).resolve()
    destination_preexisted = destination.exists()
    lock_path = _acquire_streaming_execution_lock(destination)
    try:
        return _execute_streaming_stock_design_unlocked(
            archetype_states,
            weather_members,
            occupant_seeds,
            model_scenarios,
            output_dir=destination,
            stock_weights=stock_weights,
            require_full_stock=require_full_stock,
            convergence_results_path=convergence_results_path,
            convergence_results_sha256=convergence_results_sha256,
            require_convergence_evidence=require_convergence_evidence,
            convergence_rule=convergence_rule,
            supervisor_results_selection=supervisor_results_selection,
            max_workers=max_workers,
            prepare_only=prepare_only,
        )
    finally:
        _release_streaming_execution_lock(lock_path)
        if not destination_preexisted:
            try:
                destination.rmdir()
            except OSError:
                # Successful preparation/execution leaves artifacts behind;
                # failed preflight leaves no empty output-directory side effect.
                pass


def streaming_stock_status(
    output_dir: str | Path = DEFAULT_PRODUCTION_OUTPUT_DIR,
) -> dict[str, Any]:
    """Return an observational, read-only snapshot of production progress."""

    destination = Path(output_dir).resolve()
    lock_present = (destination / "execution.lock").exists()
    design_path = destination / "streaming_design_contract.json"
    if not design_path.exists():
        return {
            "status": "INITIALIZING" if lock_present else "NOT_PREPARED",
            "output_dir": str(destination),
            "execution_lock_present": lock_present,
            "progress_validation": "OBSERVATIONAL_POINTER_COUNTS_ONLY",
        }
    design = _read_json(design_path)
    if design.get("streaming_stock_contract_version") != STREAMING_STOCK_CONTRACT_VERSION:
        return {
            "status": "STALE_CONTRACT",
            "output_dir": str(destination),
            "design_sha256": design.get("design_sha256"),
            "persisted_contract_version": design.get(
                "streaming_stock_contract_version"
            ),
            "required_contract_version": STREAMING_STOCK_CONTRACT_VERSION,
            "execution_lock_present": lock_present,
            "progress_validation": "OBSERVATIONAL_POINTER_COUNTS_ONLY",
        }
    unsigned_design = {
        key: value for key, value in design.items() if key != "design_sha256"
    }
    if canonical_sha256(unsigned_design) != design.get("design_sha256"):
        return {
            "status": "CORRUPT_CONTRACT",
            "output_dir": str(destination),
            "design_sha256": design.get("design_sha256"),
            "execution_lock_present": lock_present,
            "progress_validation": "OBSERVATIONAL_POINTER_COUNTS_ONLY",
        }
    specs = design.get("partition_specs", ())
    if not isinstance(specs, list):
        raise MonteCarloContractError(
            "Streaming design partition_specs must be a list."
        )
    seed_count = len(design.get("occupant_seeds", ()))
    completed_partitions = 0
    started_partitions = 0
    active_partition_locks = 0
    counts: list[int] = []
    active_failures = 0
    for spec in specs:
        partition_id = str(spec["partition_id"])
        partition_dir = destination / "partitions" / partition_id
        active_partition_locks += int((partition_dir / "execution.lock").exists())
        complete_path = partition_dir / "partition_complete.json"
        progress_path = partition_dir / "progress.json"
        completed_count = 0
        if complete_path.exists():
            complete = _read_json(complete_path)
            if (
                complete.get("design_sha256") == design.get("design_sha256")
                and complete.get("partition_id") == partition_id
                and complete.get("status") == "PASS"
            ):
                completed_partitions += 1
                started_partitions += 1
                completed_count = seed_count
                counts.append(seed_count)
        elif progress_path.exists():
            progress = _read_json(progress_path)
            if (
                progress.get("design_sha256") == design.get("design_sha256")
                and progress.get("partition_id") == partition_id
            ):
                completed_count = int(progress.get("completed_seed_count", 0))
                started_partitions += 1
                counts.append(completed_count)
        failure_path = partition_dir / "last_failure.json"
        if failure_path.exists():
            failure = _read_json(failure_path)
            failed_rank = int(failure.get("occupant_seed_rank", -1))
            if failure.get("status") == "FAILED" and failed_rank > completed_count:
                active_failures += 1

    summary_path = destination / "monte_carlo_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    if (
        completed_partitions == len(specs)
        and len(specs) > 0
        and summary.get("design_sha256") == design.get("design_sha256")
        and summary.get("status") in {
            "PASS",
            "WORKFLOW_CHECK_ONLY",
            "PARTIAL_STOCK_WORKFLOW",
            "PRELIMINARY_REPRESENTATIVE_WEATHER_STOCK_COMPLETE",
        }
    ):
        status = str(summary["status"])
    elif lock_present:
        status = "IN_PROGRESS"
    elif started_partitions or completed_partitions:
        status = "INTERRUPTED"
    else:
        status = (
            "PRELIMINARY_REPRESENTATIVE_WEATHER_STOCK_PREPARED"
            if design.get("supervisor_results_provenance") is not None
            else "PREPARED"
        )
    return {
        "status": status,
        "output_dir": str(destination),
        "design_sha256": design.get("design_sha256"),
        "design_contract_file_sha256": _sha256_file(design_path),
        "expected_run_count": design.get("expected_run_count"),
        "model_scenario_ids": [
            str(item["scenario_id"]) for item in design.get("model_scenarios", ())
        ],
        "occupant_seed_count": seed_count,
        "supervisor_results_provenance": design.get(
            "supervisor_results_provenance"
        ),
        "partition_count": len(specs),
        "completed_partition_count": completed_partitions,
        "started_partition_count": started_partitions,
        "partition_seed_count_histogram": {
            str(value): counts.count(value) for value in sorted(set(counts))
        },
        "active_failure_count": active_failures,
        "active_partition_lock_count": active_partition_locks,
        "execution_lock_present": lock_present,
        "progress_validation": "OBSERVATIONAL_POINTER_COUNTS_ONLY",
    }


def run_pilot(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Run a bounded workflow check; it is not the final seed-convergence study."""

    destination = Path(output_dir).resolve()
    states = [
        item
        for item in load_unique_archetype_states()
        if item.archetype_id == PILOT_ARCHETYPE_ID and item.state_id in PILOT_STATE_IDS
    ]
    if {(item.archetype_id, item.state_id) for item in states} != {
        (PILOT_ARCHETYPE_ID, state_id) for state_id in PILOT_STATE_IDS
    }:
        raise MonteCarloContractError("Pilot archetype/state selection is incomplete.")
    members = load_weather_members(PILOT_WEATHER_MEMBER_IDS)
    seeds = make_seed_bank(PILOT_SEED_COUNT, master_seed=PILOT_MASTER_SEED)
    provisional_manifest = build_balanced_manifest(
        states, members, seeds, PILOT_MODEL_SCENARIOS
    )
    representative = provisional_manifest.loc[
        (provisional_manifest["state_id"] == "TABULA_existing")
        & (provisional_manifest["weather_member_id"] == PILOT_WEATHER_MEMBER_IDS[-1])
        & (provisional_manifest["occupant_seed_rank"] == 1)
        & (provisional_manifest["model_scenario_id"] == "central"),
        "run_id",
    ]
    if len(representative) != 1:
        raise MonteCarloContractError("Pilot representative run selection is ambiguous.")
    manifest, diagnostics, retained = execute_balanced_design(
        states,
        members,
        seeds,
        PILOT_MODEL_SCENARIOS,
        retain_hourly_run_ids=[representative.iloc[0]],
    )
    distributions = distribution_summary(diagnostics)
    variance = variance_contributions(diagnostics)
    paired = paired_renovation_deltas(diagnostics)
    paired_scenarios = paired_model_scenario_deltas(diagnostics)

    _atomic_csv(manifest, destination / "run_manifest.csv")
    _atomic_csv(diagnostics, destination / "run_diagnostics.csv")
    _atomic_csv(distributions, destination / "distribution_summary.csv")
    _atomic_csv(variance, destination / "variance_contributions.csv")
    _atomic_csv(paired, destination / "paired_renovation_deltas.csv")
    _atomic_csv(
        paired_scenarios,
        destination / "paired_model_scenario_deltas.csv",
    )
    _atomic_csv(
        retained[representative.iloc[0]],
        destination / "representative_hourly.csv",
    )
    scenarios = pd.DataFrame.from_records(
        [item.definition() for item in scenario_catalog()]
    )
    _atomic_csv(scenarios, destination / "model_scenario_catalog.csv")
    selected_weather = load_weather_catalog().loc[
        lambda frame: frame["member_id"].isin(PILOT_WEATHER_MEMBER_IDS),
        [
            "member_id",
            "scenario",
            "weather_pair_id",
            "observed_pvgis_year",
            "row_count",
            "member_sha256",
            "metadata_sha256",
            "weather_contract_sha256",
        ],
    ]
    _atomic_csv(selected_weather, destination / "weather_selection.csv")
    summary_path = destination / "monte_carlo_summary.json"
    artifact_paths = sorted(
        path for path in destination.glob("*") if path != summary_path and path.is_file()
    )
    summary = {
        "status": "PASS",
        "scope": "bounded Gate-5 workflow pilot; not the final Monte Carlo experiment",
        "run_count": len(diagnostics),
        "archetype_id": PILOT_ARCHETYPE_ID,
        "state_ids": list(PILOT_STATE_IDS),
        "weather_member_ids": list(PILOT_WEATHER_MEMBER_IDS),
        "climate_scenario_ids": sorted(diagnostics["climate_scenario_id"].unique()),
        "occupant_seeds_in_nested_order": list(seeds),
        "model_scenario_ids": list(PILOT_MODEL_SCENARIOS),
        "representative_run_id": representative.iloc[0],
        "behaviour_profile_cache_entries": (
            len({
                (row.dwelling_class, row.weather_member_id, row.occupant_seed)
                for row in diagnostics.itertuples(index=False)
            })
        ),
        "seed_convergence": {
            "status": "NOT_EVALUATED",
            "reason": (
                f"pilot has {len(seeds)} seeds; first predeclared production checkpoint "
                f"is {DEFAULT_CONVERGENCE_CHECKPOINTS[0]}"
            ),
            "production_checkpoints": list(DEFAULT_CONVERGENCE_CHECKPOINTS),
        },
        "stock_aggregation": {
            "status": "NOT_EVALUATED",
            "reason": "pilot covers 2 of 75 physics cells",
        },
        "interval_interpretation": (
            "pilot summaries are workflow checks, not confidence or prediction intervals"
        ),
        "output_files": sorted([*(path.name for path in artifact_paths), summary_path.name]),
        "artifact_sha256": {
            path.name: _sha256_file(path) for path in artifact_paths
        },
    }
    _atomic_json(summary, summary_path)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot", help="run the bounded Gate-5 workflow pilot")
    pilot.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    convergence = subparsers.add_parser(
        "convergence",
        help="run or resume the production occupant-seed convergence panel",
    )
    convergence.add_argument("--output-dir", type=Path, default=None)
    convergence.add_argument("--workers", type=int, default=None)
    convergence.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate and persist the frozen design without simulating",
    )
    convergence.add_argument(
        "--status",
        action="store_true",
        help="print a read-only convergence progress snapshot",
    )
    extension = subparsers.add_parser(
        "convergence-extension",
        help="run or resume the prospective n=160 seed confirmation",
    )
    extension.add_argument("--output-dir", type=Path, default=None)
    extension.add_argument("--base-output-dir", type=Path, default=None)
    extension.add_argument("--workers", type=int, default=None)
    extension.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate and persist the n=160 extension contract without simulating",
    )
    extension.add_argument(
        "--status",
        action="store_true",
        help="print a read-only extension progress snapshot",
    )
    continuation = subparsers.add_parser(
        "convergence-continuation",
        help="run or resume the prospective n=320/n=640 seed continuation",
    )
    continuation.add_argument("--output-dir", type=Path, default=None)
    continuation.add_argument("--source-output-dir", type=Path, default=None)
    continuation.add_argument("--workers", type=int, default=None)
    continuation.add_argument(
        "--prepare-only",
        action="store_true",
        help="persist/authenticate the n=320/n=640 contract without simulating",
    )
    continuation.add_argument(
        "--status",
        action="store_true",
        help="print a read-only continuation progress snapshot",
    )
    stock = subparsers.add_parser(
        "stock",
        help="run/resume the authoritative partitioned 2050 stock simulation",
    )
    stock.add_argument("--output-dir", type=Path, default=None)
    stock.add_argument("--workers", type=int, default=4)
    stock.add_argument(
        "--model-scenarios",
        nargs="+",
        choices=sorted(item.scenario_id for item in scenario_catalog()),
        default=["central"],
        help="registered structural scenarios; central is required by convergence evidence",
    )
    stock.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate and persist the exact authenticated design without simulating",
    )
    stock.add_argument(
        "--status",
        action="store_true",
        help="print a read-only production progress snapshot",
    )
    postprocess = subparsers.add_parser(
        "postprocess",
        help="authenticate and summarize completed stock partition diagnostics",
    )
    postprocess.add_argument("--output-dir", type=Path, default=None)
    postprocess.add_argument(
        "--chunk-rows",
        type=int,
        default=2_000,
        help="maximum partition-diagnostic rows read at once",
    )
    postprocess.add_argument(
        "--status",
        action="store_true",
        help="verify the committed post-processing artifact set without rerunning it",
    )
    sensitivity = subparsers.add_parser(
        "sensitivity",
        help="prepare, run/resume, or inspect the frozen structural-sensitivity screen",
    )
    sensitivity.add_argument("--output-dir", type=Path, default=None)
    sensitivity.add_argument(
        "--stage",
        type=int,
        choices=(40, 80, 160),
        default=40,
        help="prospectively allowed nested occupant-seed stage",
    )
    sensitivity.add_argument(
        "--previous-stage-dir",
        type=Path,
        default=None,
        help="authenticated predecessor required when first preparing n=80 or n=160",
    )
    sensitivity.add_argument("--workers", type=int, default=4)
    sensitivity.add_argument(
        "--chunk-rows",
        type=int,
        default=2_000,
        help="maximum partition-diagnostic rows read per post-processing chunk",
    )
    sensitivity.add_argument(
        "--prepare-only",
        action="store_true",
        help="persist the exact screen contract without launching simulations",
    )
    sensitivity.add_argument(
        "--status",
        action="store_true",
        help="print a read-only structural-screen progress/decision snapshot",
    )
    report = subparsers.add_parser(
        "report",
        help="authenticate completed central results and publish the Gate-5 report",
    )
    report.add_argument("--output-dir", type=Path, default=None)
    report.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="destination for independent PNG/PDF figures",
    )
    report.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Markdown destination (defaults to RESULTS.md in the production directory)",
    )
    report.add_argument(
        "--status",
        action="store_true",
        help="reauthenticate the committed report and figures without rewriting them",
    )
    subparsers.add_parser("catalog", help="print the declared weather/scenario inventory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "pilot":
        print(json.dumps(_json_ready(run_pilot(args.output_dir)), indent=2, sort_keys=True))
        return 0
    if args.command == "convergence":
        from .convergence_runner import (
            DEFAULT_CONVERGENCE_OUTPUT_DIR,
            convergence_status,
            prepare_convergence_experiment,
            run_convergence_experiment,
        )

        output_dir = (
            DEFAULT_CONVERGENCE_OUTPUT_DIR
            if args.output_dir is None
            else args.output_dir
        )
        if args.prepare_only and args.status:
            parser.error("convergence --prepare-only and --status are mutually exclusive")
        if args.status:
            result = convergence_status(output_dir)
        elif args.prepare_only:
            prepared = prepare_convergence_experiment(output_dir)
            result = {
                "status": "PREPARED",
                "output_dir": str(Path(output_dir).resolve()),
                "design_sha256": prepared["design_sha256"],
                "panel_cell_count": len(prepared["panel"]),
                "weather_member_count": len(prepared["weather_members"]),
                "maximum_seed_count": len(prepared["occupant_seeds"]),
                "expected_maximum_run_count": prepared[
                    "expected_maximum_run_count"
                ],
            }
        else:
            result = run_convergence_experiment(
                output_dir, max_workers=args.workers
            )
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0
    if args.command == "convergence-extension":
        from .convergence_extension import (
            DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
            convergence_extension_status,
            prepare_convergence_extension,
            run_convergence_extension,
        )
        from .convergence_runner import DEFAULT_CONVERGENCE_OUTPUT_DIR

        output_dir = (
            DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR
            if args.output_dir is None
            else args.output_dir
        )
        base_output_dir = (
            DEFAULT_CONVERGENCE_OUTPUT_DIR
            if args.base_output_dir is None
            else args.base_output_dir
        )
        if args.prepare_only and args.status:
            parser.error(
                "convergence-extension --prepare-only and --status are mutually exclusive"
            )
        if args.status:
            result = convergence_extension_status(output_dir)
        elif args.prepare_only:
            prepared = prepare_convergence_extension(
                output_dir, base_output_dir=base_output_dir
            )
            result = {
                "status": "PREPARED",
                "output_dir": str(Path(output_dir).resolve()),
                "design_sha256": prepared["design_sha256"],
                "base_design_sha256": prepared["base_experiment"][
                    "base_design_sha256"
                ],
                "imported_seed_count": prepared["imported_seed_count"],
                "target_seed_count": prepared["extension_checkpoint"],
                "expected_new_run_count": prepared["expected_new_run_count"],
            }
        else:
            result = run_convergence_extension(
                output_dir,
                base_output_dir=base_output_dir,
                max_workers=args.workers,
            )
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0
    if args.command == "convergence-continuation":
        from .convergence_continuation import (
            DEFAULT_CONVERGENCE_CONTINUATION_OUTPUT_DIR,
            convergence_continuation_status,
            prepare_convergence_continuation,
            run_convergence_continuation,
        )
        from .convergence_extension import (
            DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR,
        )

        output_dir = (
            DEFAULT_CONVERGENCE_CONTINUATION_OUTPUT_DIR
            if args.output_dir is None
            else args.output_dir
        )
        source_output_dir = (
            DEFAULT_CONVERGENCE_EXTENSION_OUTPUT_DIR
            if args.source_output_dir is None
            else args.source_output_dir
        )
        if args.prepare_only and args.status:
            parser.error(
                "convergence-continuation --prepare-only and --status are "
                "mutually exclusive"
            )
        if args.status:
            result = convergence_continuation_status(output_dir)
        elif args.prepare_only:
            prepared = prepare_convergence_continuation(
                output_dir, source_output_dir=source_output_dir
            )
            result = {
                "status": "PREPARED",
                "output_dir": str(Path(output_dir).resolve()),
                "design_sha256": prepared["design_sha256"],
                "source_design_sha256": prepared["source_experiment"][
                    "source_design_sha256"
                ],
                "imported_seed_count": prepared["imported_seed_count"],
                "continuation_checkpoints": prepared[
                    "continuation_checkpoints"
                ],
                "expected_new_run_count": prepared["expected_new_run_count"],
                "expected_total_run_count_at_n640": prepared[
                    "expected_total_run_count_at_n640"
                ],
            }
        else:
            result = run_convergence_continuation(
                output_dir,
                source_output_dir=source_output_dir,
                max_workers=args.workers,
            )
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0
    if args.command == "stock":
        output_dir = (
            DEFAULT_PRODUCTION_OUTPUT_DIR
            if args.output_dir is None
            else args.output_dir
        )
        if args.prepare_only and args.status:
            parser.error("stock --prepare-only and --status are mutually exclusive")
        if args.status:
            result = streaming_stock_status(output_dir)
        else:
            from .convergence_continuation import (
                load_convergence_continuation_selection,
            )

            selection = load_convergence_continuation_selection()
            states = tuple(load_unique_archetype_states())
            weather_catalog = load_weather_catalog()
            members = load_weather_members(
                tuple(weather_catalog["member_id"].astype(str))
            )
            result = execute_streaming_stock_design(
                states,
                members,
                selection.occupant_seeds,
                tuple(args.model_scenarios),
                output_dir=output_dir,
                convergence_results_path=selection.convergence_results_path,
                convergence_results_sha256=(
                    selection.convergence_results_sha256
                ),
                convergence_rule=selection.convergence_rule,
                max_workers=args.workers,
                prepare_only=args.prepare_only,
            )
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0
    if args.command == "postprocess":
        from .postprocess import (
            postprocess_production_results,
            postprocessing_status,
        )

        output_dir = (
            DEFAULT_PRODUCTION_OUTPUT_DIR
            if args.output_dir is None
            else args.output_dir
        )
        if args.status:
            result = postprocessing_status(output_dir)
        else:
            result = postprocess_production_results(
                output_dir,
                chunk_rows=args.chunk_rows,
            )
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0
    if args.command == "sensitivity":
        from .sensitivity import (
            default_structural_sensitivity_output_dir,
            run_structural_sensitivity_screen,
            structural_sensitivity_status,
        )

        output_dir = (
            default_structural_sensitivity_output_dir(args.stage)
            if args.output_dir is None
            else args.output_dir
        )
        if args.prepare_only and args.status:
            parser.error("sensitivity --prepare-only and --status are mutually exclusive")
        if args.status:
            result = structural_sensitivity_status(
                output_dir,
                target_seed_count=args.stage,
            )
        else:
            result = run_structural_sensitivity_screen(
                output_dir,
                target_seed_count=args.stage,
                previous_stage_dir=args.previous_stage_dir,
                max_workers=args.workers,
                postprocess_chunk_rows=args.chunk_rows,
                prepare_only=args.prepare_only,
            )
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0
    if args.command == "report":
        from .reporting import (
            DEFAULT_FIGURE_DIR,
            REPORTING_SUMMARY_FILENAME,
            generate_production_report,
            reporting_status,
        )

        output_dir = (
            DEFAULT_PRODUCTION_OUTPUT_DIR
            if args.output_dir is None
            else args.output_dir
        )
        if args.status:
            result = reporting_status(output_dir)
        else:
            generated = generate_production_report(
                output_dir,
                figure_dir=(
                    DEFAULT_FIGURE_DIR
                    if args.figure_dir is None
                    else args.figure_dir
                ),
                report_path=args.report_path,
            )
            result = {
                "status": generated["status"],
                "reporting_contract_version": generated[
                    "reporting_contract_version"
                ],
                "source_design_sha256": generated["source_design_sha256"],
                "figure_count": generated["figure_count"],
                "report_path": generated["output_artifacts"]["report"]["path"],
                "figure_dir": str(
                    (
                        DEFAULT_FIGURE_DIR
                        if args.figure_dir is None
                        else args.figure_dir
                    ).resolve()
                ),
                "reporting_summary_path": str(
                    Path(output_dir).resolve() / REPORTING_SUMMARY_FILENAME
                ),
                "reporting_sha256": generated["reporting_sha256"],
            }
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0
    if args.command == "catalog":
        weather = load_weather_catalog()
        payload = {
            "weather_members": len(weather),
            "weather_by_rcp": weather.groupby("scenario").size().to_dict(),
            "paired_years": sorted(weather["observed_pvgis_year"].unique().tolist()),
            "model_scenarios": [item.definition() for item in scenario_catalog()],
        }
        print(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command {args.command!r}.")
