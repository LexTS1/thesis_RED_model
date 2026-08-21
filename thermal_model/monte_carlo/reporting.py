"""Authenticated Gate-5 production reporting and publication figures.

The reporter is intentionally downstream-only: it never launches simulations or
post-processing.  It accepts only a committed, authoritative, central-scenario
stock run with a valid post-processing commit, authenticates the complete hash
chain, cross-checks the reporting tables, and then writes one chart per file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from .aggregation import STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
from .contracts import CLIMATE_SCENARIOS, MonteCarloContractError, canonical_sha256
from .design import ordered_seed_bank_sha256
from .postprocess import (
    MODEL_SCENARIO_OUTPUT_FILENAME,
    POSTPROCESS_CONTRACT_VERSION,
    POSTPROCESS_SUMMARY_FILENAME,
    RENOVATION_OUTPUT_FILENAME,
    UNWEIGHTED_OUTPUT_FILENAME,
    VARIANCE_OUTPUT_FILENAME,
    WEIGHTED_OUTPUT_FILENAME,
    postprocessing_status,
)
from .runner import DEFAULT_PRODUCTION_OUTPUT_DIR, STREAMING_STOCK_CONTRACT_VERSION
from .stock_streaming import BELGIUM_REGION_ID


REPORTING_CONTRACT_VERSION = "gate5_results_reporting_v1"
REPORT_FILENAME = "RESULTS.md"
REPORTING_SUMMARY_FILENAME = "results_reporting_summary.json"
FIGURE_PROVENANCE_FILENAME = "mc_results_figure_provenance.json"
MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = MODULE_ROOT / "figures"

RCP_ORDER = tuple(CLIMATE_SCENARIOS)
RCP_LABELS = {
    "rcp_2_6": "RCP 2.6",
    "rcp_4_5": "RCP 4.5",
    "rcp_8_5": "RCP 8.5",
}
EXPECTED_WEATHER_MEMBERS_PER_RCP = 18
EXPECTED_PHYSICS_CELLS = 75
MODELLED_STOCK_DWELLINGS = 5_537_385.0
NATIONAL_REGION = BELGIUM_REGION_ID
REGION_ORDER = (
    "Flemish Region",
    "Walloon Region",
    "Brussels-Capital Region",
)
DWELLING_ORDER = (
    "Detached house",
    "Semi-detached house",
    "Terraced house",
    "Apartment, enclosed",
    "Apartment, exposed",
)
PERIOD_ORDER = ("pre-1946", "1946-1970", "1971-1990", "1991-2005", "post-2005")
STATE_ORDER = (
    "TABULA_existing",
    "TABULA_standard_B_proxy",
    "TABULA_advanced_A_proxy",
)
STATE_LABELS = {
    "TABULA_existing": "Existing",
    "TABULA_standard_B_proxy": "Standard (B proxy)",
    "TABULA_advanced_A_proxy": "Advanced (A proxy)",
}

FIGURE_BASENAMES = (
    "mc_national_annual_heating",
    "mc_national_potential_cooling",
    "mc_national_coincident_heating_peak",
    "mc_heating_contribution_region",
    "mc_heating_contribution_dwelling_type",
    "mc_heating_contribution_construction_period",
    "mc_heating_contribution_renovation_state",
    "mc_heating_variance_weather_occupant",
    "mc_paired_renovation_heating_effect",
)

_ROOT_STOCK_ARTIFACTS = {
    "partition_index.csv",
    "stock_aggregation.csv",
    "stock_contributions.csv",
    "stock_distribution_summary.csv",
    "convergence_results.csv",
}
_PARTITION_ARTIFACTS = {
    "run_manifest.csv",
    "run_diagnostics.csv",
    "stock_aggregation.csv",
    "stock_contributions.csv",
    "stock_hourly.csv",
}
_POSTPROCESS_ARTIFACTS = {
    UNWEIGHTED_OUTPUT_FILENAME,
    VARIANCE_OUTPUT_FILENAME,
    RENOVATION_OUTPUT_FILENAME,
    WEIGHTED_OUTPUT_FILENAME,
}


@dataclass(frozen=True)
class AuthenticatedReportingInputs:
    """Fully authenticated, validated tables consumed by the reporter."""

    production_dir: Path
    design: Mapping[str, Any]
    stock_summary: Mapping[str, Any]
    postprocess_summary: Mapping[str, Any]
    stock_aggregation: pd.DataFrame
    stock_contributions: pd.DataFrame
    stock_distribution: pd.DataFrame
    variance_contributions: pd.DataFrame
    paired_renovation_deltas: pd.DataFrame
    source_artifacts: Mapping[str, Mapping[str, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonteCarloContractError(f"Cannot read {label} JSON {path}.") from exc
    if not isinstance(payload, dict):
        raise MonteCarloContractError(f"{label} JSON must contain an object: {path}.")
    return payload


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MonteCarloContractError(f"Cannot read {label} CSV {path}.") from exc


def _read_run_ids(path: Path, *, label: str) -> pd.Series:
    """Read only the identity column from a potentially wide partition table."""

    try:
        frame = pd.read_csv(path, usecols=["run_id"], keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MonteCarloContractError(
            f"Cannot read the run_id column from {label} CSV {path}."
        ) from exc
    return frame["run_id"].astype(str)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise MonteCarloContractError(f"{label} is missing columns: {missing}.")


def _valid_sha256(value: Any, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MonteCarloContractError(f"{label} is not a valid SHA-256 digest.")
    return digest


def _safe_path(root: Path, value: Any, *, label: str) -> Path:
    raw = Path(str(value))
    path = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MonteCarloContractError(f"{label} escapes the production directory.") from exc
    return path


def _verify_file(
    path: Path,
    expected_sha256: Any,
    *,
    label: str,
    row_count: Any | None = None,
) -> dict[str, Any]:
    expected = _valid_sha256(expected_sha256, label=f"{label} checksum")
    if not path.is_file():
        raise MonteCarloContractError(f"{label} is missing: {path}.")
    observed_rows: int | None = None
    if row_count is None:
        actual = _sha256_file(path)
    else:
        digest = hashlib.sha256()
        newline_count = 0
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    newline_count += chunk.count(b"\n")
        except OSError as exc:
            raise MonteCarloContractError(f"Cannot authenticate {label}: {path}.") from exc
        actual = digest.hexdigest()
        observed_rows = max(newline_count - 1, 0)
    if actual != expected:
        raise MonteCarloContractError(
            f"{label} checksum mismatch: expected {expected}, got {actual}."
        )
    record: dict[str, Any] = {"sha256": actual}
    if row_count is not None:
        try:
            expected_rows = int(row_count)
        except (TypeError, ValueError) as exc:
            raise MonteCarloContractError(f"{label} has invalid row-count metadata.") from exc
        assert observed_rows is not None
        if expected_rows < 0 or observed_rows != expected_rows:
            raise MonteCarloContractError(
                f"{label} row count changed: expected {expected_rows}, got {observed_rows}."
            )
        record["row_count"] = observed_rows
    return record


def _record_source(
    ledger: dict[str, dict[str, Any]], root: Path, path: Path, metadata: Mapping[str, Any]
) -> None:
    relative = str(path.resolve().relative_to(root))
    ledger[relative] = {"sha256": metadata["sha256"]}
    if "row_count" in metadata:
        ledger[relative]["row_count"] = int(metadata["row_count"])


def _number_column(frame: pd.DataFrame, column: str, *, label: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise MonteCarloContractError(f"{label} column {column!r} is non-finite.")
    return values


def _validate_stock_tables(
    aggregation: pd.DataFrame,
    contributions: pd.DataFrame,
    distributions: pd.DataFrame,
) -> None:
    stock_metrics = (
        "annual_heating_GWh",
        "annual_potential_sensible_cooling_GWh",
        "coincident_peak_heating_MW",
        "coincident_peak_potential_cooling_MW",
    )
    stock_keys = (
        "climate_scenario_id",
        "weather_member_id",
        "model_scenario_id",
        "region",
    )
    _require_columns(
        aggregation,
        {*stock_keys, "model_scenario_axis", "stock_coverage", "modelled_dwellings", *stock_metrics},
        label="stock aggregation",
    )
    if aggregation.empty or aggregation.duplicated(list(stock_keys)).any():
        raise MonteCarloContractError("Stock aggregation is empty or has duplicate reporting keys.")
    if set(aggregation["model_scenario_id"].astype(str)) != {"central"}:
        raise MonteCarloContractError("Reporting accepts the completed central model scenario only.")
    if set(aggregation["model_scenario_axis"].astype(str)) != {"central"}:
        raise MonteCarloContractError("Stock aggregation carries a non-central scenario axis.")
    if set(aggregation["climate_scenario_id"].astype(str)) != set(RCP_ORDER):
        raise MonteCarloContractError("Stock aggregation must retain all three RCPs separately.")
    if not aggregation["stock_coverage"].astype(str).eq(
        "R1-R4 modelled stock; R5-R6 residual excluded"
    ).all():
        raise MonteCarloContractError("Stock aggregation does not declare the frozen R1-R4 scope.")
    for metric in stock_metrics:
        if (_number_column(aggregation, metric, label="stock aggregation") < 0.0).any():
            raise MonteCarloContractError(f"Stock aggregation metric {metric!r} is negative.")
    counts = (
        aggregation.loc[aggregation["region"].astype(str) == NATIONAL_REGION]
        .groupby("climate_scenario_id")["weather_member_id"]
        .nunique()
    )
    if counts.to_dict() != {rcp: EXPECTED_WEATHER_MEMBERS_PER_RCP for rcp in RCP_ORDER}:
        raise MonteCarloContractError("National stock results require 18 weather members per RCP.")
    national_dwellings = pd.to_numeric(
        aggregation.loc[
            aggregation["region"].astype(str) == NATIONAL_REGION,
            "modelled_dwellings",
        ],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(national_dwellings).all() or not np.allclose(
        national_dwellings,
        MODELLED_STOCK_DWELLINGS,
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise MonteCarloContractError("National stock results do not reconstruct 5,537,385 R1-R4 dwellings.")

    distribution_keys = (
        "climate_scenario_id",
        "model_scenario_id",
        "region",
        "metric",
    )
    statistics = ("minimum", "p05", "median", "mean", "p95", "maximum")
    _require_columns(
        distributions,
        {*distribution_keys, "weather_member_count", "interval_interpretation", *statistics},
        label="stock distribution summary",
    )
    if distributions.empty or distributions.duplicated(list(distribution_keys)).any():
        raise MonteCarloContractError("Stock distribution summary has duplicate or absent groups.")
    if set(distributions["metric"].astype(str)) != set(stock_metrics):
        raise MonteCarloContractError("Stock distribution summary has an unexpected metric set.")
    if not distributions["interval_interpretation"].astype(str).str.contains(
        "not a complete prediction interval", regex=False
    ).all():
        raise MonteCarloContractError("Stock distributions lack the empirical-interval limitation.")
    for row in distributions.itertuples(index=False):
        source = aggregation.loc[
            (aggregation["climate_scenario_id"].astype(str) == str(row.climate_scenario_id))
            & (aggregation["model_scenario_id"].astype(str) == str(row.model_scenario_id))
            & (aggregation["region"].astype(str) == str(row.region)),
            str(row.metric),
        ]
        values = pd.to_numeric(source, errors="coerce").to_numpy(dtype=float)
        if len(values) != int(row.weather_member_count) or len(values) != EXPECTED_WEATHER_MEMBERS_PER_RCP:
            raise MonteCarloContractError("Stock distribution weather-member count is inconsistent.")
        expected = {
            "minimum": float(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }
        for statistic, expected_value in expected.items():
            if not np.isclose(float(getattr(row, statistic)), expected_value, rtol=1.0e-10, atol=1.0e-8):
                raise MonteCarloContractError(
                    f"Stock distribution {statistic} does not reproduce stock aggregation."
                )

    contribution_dimensions = {
        "region": set(REGION_ORDER),
        "dwelling_type": set(DWELLING_ORDER),
        "construction_period": set(PERIOD_ORDER),
        "state_id": set(STATE_ORDER),
    }
    contribution_keys = (
        "climate_scenario_id",
        "weather_member_id",
        "model_scenario_id",
        "contribution_dimension",
        "contribution_value",
    )
    _require_columns(
        contributions,
        {
            *contribution_keys,
            "annual_heating_GWh",
            "annual_potential_sensible_cooling_GWh",
            "share_of_stock_heating",
            "share_of_stock_potential_cooling",
        },
        label="stock contributions",
    )
    if contributions.empty or contributions.duplicated(list(contribution_keys)).any():
        raise MonteCarloContractError("Stock contributions have duplicate or absent keys.")
    if set(contributions["contribution_dimension"].astype(str)) != set(contribution_dimensions):
        raise MonteCarloContractError("Stock contributions do not contain the four frozen dimensions.")
    for dimension, expected_values in contribution_dimensions.items():
        observed = set(
            contributions.loc[
                contributions["contribution_dimension"].astype(str) == dimension,
                "contribution_value",
            ].astype(str)
        )
        if observed != expected_values:
            raise MonteCarloContractError(
                f"Stock contribution dimension {dimension!r} has unexpected categories."
            )
    for column in (
        "annual_heating_GWh",
        "annual_potential_sensible_cooling_GWh",
        "share_of_stock_heating",
        "share_of_stock_potential_cooling",
    ):
        if (_number_column(contributions, column, label="stock contributions") < -1.0e-12).any():
            raise MonteCarloContractError(f"Stock contribution column {column!r} is negative.")
    national = aggregation.loc[
        aggregation["region"].astype(str) == NATIONAL_REGION,
        ["climate_scenario_id", "weather_member_id", "model_scenario_id", "annual_heating_GWh", "annual_potential_sensible_cooling_GWh"],
    ]
    for keys, group in contributions.groupby(
        ["climate_scenario_id", "weather_member_id", "model_scenario_id", "contribution_dimension"],
        sort=False,
    ):
        matching = national.loc[
            (national["climate_scenario_id"].astype(str) == str(keys[0]))
            & (national["weather_member_id"].astype(str) == str(keys[1]))
            & (national["model_scenario_id"].astype(str) == str(keys[2]))
        ]
        if len(matching) != 1:
            raise MonteCarloContractError("A contribution group lacks one national stock total.")
        for metric, national_metric, share in (
            ("annual_heating_GWh", "annual_heating_GWh", "share_of_stock_heating"),
            (
                "annual_potential_sensible_cooling_GWh",
                "annual_potential_sensible_cooling_GWh",
                "share_of_stock_potential_cooling",
            ),
        ):
            total = float(pd.to_numeric(group[metric], errors="raise").sum())
            target = float(matching[national_metric].iloc[0])
            if not np.isclose(total, target, rtol=1.0e-10, atol=1.0e-7):
                raise MonteCarloContractError("Stock contribution energy does not reconcile nationally.")
            observed_share = float(pd.to_numeric(group[share], errors="raise").sum())
            expected_share = 1.0 if target > 0.0 else 0.0
            if not np.isclose(observed_share, expected_share, rtol=0.0, atol=1.0e-9):
                raise MonteCarloContractError("Stock contribution shares do not close.")


def _validate_postprocess_tables(variance: pd.DataFrame, paired: pd.DataFrame) -> None:
    variance_keys = (
        "archetype_id",
        "state_id",
        "climate_scenario_id",
        "model_scenario_id",
        "metric",
    )
    _require_columns(
        variance,
        {
            *variance_keys,
            "component",
            "sum_of_squares",
            "total_sum_of_squares",
            "sum_of_squares_share",
            "interpretation",
        },
        label="variance contributions",
    )
    if variance.empty or variance.duplicated([*variance_keys, "component"]).any():
        raise MonteCarloContractError("Variance contributions are empty or duplicated.")
    if set(variance["model_scenario_id"].astype(str)) != {"central"}:
        raise MonteCarloContractError("Variance reporting accepts the central scenario only.")
    components = {"weather_year", "occupant_seed", "weather_seed_interaction"}
    for _, group in variance.groupby(list(variance_keys), sort=False):
        if set(group["component"].astype(str)) != components:
            raise MonteCarloContractError("A variance group has an incomplete component set.")
        ss = _number_column(group, "sum_of_squares", label="variance contributions")
        totals = _number_column(group, "total_sum_of_squares", label="variance contributions")
        shares = _number_column(group, "sum_of_squares_share", label="variance contributions")
        if (ss < -1.0e-12).any() or (shares < -1.0e-12).any() or len(np.unique(totals)) != 1:
            raise MonteCarloContractError("A variance group has invalid sums of squares.")
        total = float(totals[0])
        if not np.isclose(float(ss.sum()), total, rtol=1.0e-9, atol=1.0e-8):
            raise MonteCarloContractError("Variance sums of squares do not close.")
        expected_share = 1.0 if total > 0.0 else 0.0
        if not np.isclose(float(shares.sum()), expected_share, rtol=0.0, atol=1.0e-9):
            raise MonteCarloContractError("Variance shares do not close.")

    pair_keys = (
        "archetype_id",
        "climate_scenario_id",
        "weather_member_id",
        "occupant_seed",
        "model_scenario_id",
        "metric",
    )
    _require_columns(
        paired,
        {
            *pair_keys,
            "baseline_state_id",
            "comparison_state_id",
            "baseline_value",
            "comparison_value",
            "delta",
        },
        label="paired renovation deltas",
    )
    if paired.empty or paired.duplicated([*pair_keys, "comparison_state_id"]).any():
        raise MonteCarloContractError("Paired renovation deltas are empty or duplicated.")
    if set(paired["model_scenario_id"].astype(str)) != {"central"} or set(
        paired["baseline_state_id"].astype(str)
    ) != {STATE_ORDER[0]}:
        raise MonteCarloContractError("Paired renovation results use unexpected scenarios/states.")
    if set(paired["comparison_state_id"].astype(str)) != set(STATE_ORDER[1:]):
        raise MonteCarloContractError("Paired renovation results omit a frozen renovation state.")
    baseline = _number_column(paired, "baseline_value", label="paired renovation deltas")
    comparison = _number_column(paired, "comparison_value", label="paired renovation deltas")
    delta = _number_column(paired, "delta", label="paired renovation deltas")
    if not np.allclose(comparison - baseline, delta, rtol=1.0e-10, atol=1.0e-8):
        raise MonteCarloContractError("Paired renovation deltas fail their arithmetic identity.")
    for _, group in paired.groupby(list(pair_keys), sort=False):
        if set(group["comparison_state_id"].astype(str)) != set(STATE_ORDER[1:]):
            raise MonteCarloContractError("A common weather/seed pair lacks a renovation state.")


def _authenticate_reporting_inputs_unchecked(
    production_dir: str | Path = DEFAULT_PRODUCTION_OUTPUT_DIR,
) -> AuthenticatedReportingInputs:
    """Authenticate a complete central production and post-processing commit."""

    root = Path(production_dir).resolve()
    source_ledger: dict[str, dict[str, Any]] = {}
    design_path = root / "streaming_design_contract.json"
    summary_path = root / "monte_carlo_summary.json"
    post_path = root / POSTPROCESS_SUMMARY_FILENAME
    design = _read_json(design_path, label="streaming design contract")
    stock_summary = _read_json(summary_path, label="Monte Carlo summary")
    post_summary = _read_json(post_path, label="post-processing summary")

    design_sha = _valid_sha256(design.get("design_sha256"), label="design_sha256")
    unsigned_design = {key: value for key, value in design.items() if key != "design_sha256"}
    if canonical_sha256(unsigned_design) != design_sha:
        raise MonteCarloContractError("Streaming design content does not reproduce design_sha256.")
    for key in (
        "central_thermal_assumptions_sha256",
        "behaviour_assumptions_sha256",
        "occupant_distribution_sha256",
        "stock_weights_sha256",
        "stock_weights_source_sha256",
    ):
        _valid_sha256(design.get(key), label=key)
    if design.get("streaming_stock_contract_version") != STREAMING_STOCK_CONTRACT_VERSION:
        raise MonteCarloContractError("Reporting requires the current streaming-stock contract.")
    if (
        design.get("stock_partition_provenance_contract_version")
        != STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
    ):
        raise MonteCarloContractError(
            "Reporting requires the current stock-partition provenance contract."
        )
    scenario_ids = {str(item.get("scenario_id")) for item in design.get("model_scenarios", ())}
    if (
        scenario_ids != {"central"}
        or len(design.get("model_scenarios", ())) != 1
        or str(design["model_scenarios"][0].get("axis")) != "central"
    ):
        raise MonteCarloContractError("Reporting requires a central-only production design.")
    if not bool(design.get("require_full_stock")) or len(design.get("archetype_states", ())) != EXPECTED_PHYSICS_CELLS:
        raise MonteCarloContractError("Reporting requires all 75 authoritative physics cells.")
    design_convergence = design.get("convergence_evidence")
    if not isinstance(design_convergence, dict) or design_convergence.get("status") != "VERIFIED":
        raise MonteCarloContractError("The production design lacks verified seed convergence.")
    weather = design.get("weather_members", ())
    weather_counts = pd.Series(
        [str(item.get("climate_scenario_id")) for item in weather], dtype="object"
    ).value_counts().to_dict()
    if weather_counts != {rcp: EXPECTED_WEATHER_MEMBERS_PER_RCP for rcp in RCP_ORDER}:
        raise MonteCarloContractError("Reporting requires the complete 54-member weather design.")
    seeds = design.get("occupant_seeds", ())
    if not seeds or len({int(seed) for seed in seeds}) != len(seeds):
        raise MonteCarloContractError("The production seed bank is empty or duplicated.")
    if design.get("occupant_seed_bank_sha256") != ordered_seed_bank_sha256(
        tuple(int(seed) for seed in seeds)
    ):
        raise MonteCarloContractError("The production seed-bank checksum is invalid.")

    summary_convergence = stock_summary.get("convergence_evidence")
    if (
        stock_summary.get("status") != "PASS"
        or stock_summary.get("stock_coverage_status") != "AUTHORITATIVE_FULL_STOCK"
        or stock_summary.get("require_full_stock") is not True
        or stock_summary.get("streaming_stock_contract_version") != STREAMING_STOCK_CONTRACT_VERSION
        or stock_summary.get("stock_partition_provenance_contract_version")
        != STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
        or stock_summary.get("design_sha256") != design_sha
        or not isinstance(summary_convergence, dict)
        or summary_convergence.get("status") != "VERIFIED"
    ):
        raise MonteCarloContractError("Stock execution is not an authenticated production PASS.")
    if (
        stock_summary.get("occupant_seeds") != list(seeds)
        or stock_summary.get("occupant_seed_bank_sha256")
        != design.get("occupant_seed_bank_sha256")
        or stock_summary.get("stock_weights_sha256")
        != design.get("stock_weights_sha256")
        or stock_summary.get("stock_weights_source_sha256")
        != design.get("stock_weights_source_sha256")
    ):
        raise MonteCarloContractError(
            "Stock execution changes the seed bank or stock-weight provenance."
        )
    if "not complete prediction intervals" not in str(
        stock_summary.get("interval_interpretation", "")
    ):
        raise MonteCarloContractError(
            "Stock summary lacks the empirical-interval limitation."
        )
    for key, value in design_convergence.items():
        if summary_convergence.get(key) != value:
            raise MonteCarloContractError(
                "Stock execution changes the design's convergence evidence."
            )
    expected_runs = int(design.get("expected_run_count", -1))
    if (
        expected_runs <= 0
        or int(stock_summary.get("expected_run_count", -2)) != expected_runs
        or int(stock_summary.get("completed_run_count", -3)) != expected_runs
    ):
        raise MonteCarloContractError("Stock execution counts do not match its design.")

    root_artifacts = stock_summary.get("artifact_sha256")
    if not isinstance(root_artifacts, dict) or not _ROOT_STOCK_ARTIFACTS.issubset(root_artifacts):
        raise MonteCarloContractError("Stock summary has an incomplete root artifact ledger.")
    if (
        summary_convergence.get("persisted_path") != "convergence_results.csv"
        or summary_convergence.get("persisted_sha256")
        != root_artifacts.get("convergence_results.csv")
        or design_convergence.get("convergence_results_sha256")
        != root_artifacts.get("convergence_results.csv")
    ):
        raise MonteCarloContractError(
            "Stock summary does not authenticate its persisted convergence evidence."
        )
    design_meta = _verify_file(design_path, _sha256_file(design_path), label="streaming design")
    summary_meta = _verify_file(summary_path, _sha256_file(summary_path), label="Monte Carlo summary")
    _record_source(source_ledger, root, design_path, design_meta)
    _record_source(source_ledger, root, summary_path, summary_meta)
    for filename, digest in root_artifacts.items():
        path = _safe_path(root, filename, label="stock root artifact")
        if path.parent != root:
            raise MonteCarloContractError("Stock root artifacts must be direct production children.")
        metadata = _verify_file(path, digest, label=f"stock root artifact {filename}")
        _record_source(source_ledger, root, path, metadata)

    partition_index = _read_csv(root / "partition_index.csv", label="partition index")
    _require_columns(
        partition_index,
        {
            "partition_id",
            "weather_member_id",
            "climate_scenario_id",
            "model_scenario_id",
            "run_count",
            "run_diagnostics_path",
            "run_diagnostics_sha256",
            "stock_hourly_path",
            "stock_hourly_row_count",
            "stock_hourly_sha256",
            "partition_complete_sha256",
        },
        label="partition index",
    )
    specs = {
        (str(item["partition_id"]), str(item["weather_member_id"]), str(item["model_scenario_id"]))
        for item in design.get("partition_specs", ())
    }
    observed_specs = set(
        map(
            tuple,
            partition_index[["partition_id", "weather_member_id", "model_scenario_id"]]
            .astype(str)
            .to_numpy(),
        )
    )
    if len(partition_index) != 54 or partition_index["partition_id"].duplicated().any() or observed_specs != specs:
        raise MonteCarloContractError("Partition index does not exactly cover the production design.")
    expected_partition_runs = EXPECTED_PHYSICS_CELLS * len(seeds)
    if not pd.to_numeric(partition_index["run_count"], errors="coerce").eq(expected_partition_runs).all():
        raise MonteCarloContractError("A stock partition has incomplete cell/seed coverage.")

    partition_input_ledger: list[dict[str, Any]] = []
    for row in partition_index.itertuples(index=False):
        partition_id = str(row.partition_id)
        partition_dir = _safe_path(root, Path("partitions") / partition_id, label="partition")
        complete_path = partition_dir / "partition_complete.json"
        complete_meta = _verify_file(
            complete_path,
            row.partition_complete_sha256,
            label=f"partition {partition_id} completion ledger",
        )
        _record_source(source_ledger, root, complete_path, complete_meta)
        complete = _read_json(complete_path, label=f"partition {partition_id} completion")
        if (
            complete.get("status") != "PASS"
            or complete.get("stock_coverage_status") != "AUTHORITATIVE_FULL_STOCK"
            or complete.get("streaming_stock_contract_version") != STREAMING_STOCK_CONTRACT_VERSION
            or complete.get("stock_partition_provenance_contract_version")
            != STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
            or complete.get("design_sha256") != design_sha
            or str(complete.get("partition_id")) != partition_id
            or str(complete.get("weather_member_id")) != str(row.weather_member_id)
            or str(complete.get("model_scenario_id")) != "central"
            or int(complete.get("run_count", -1)) != expected_partition_runs
            or complete.get("occupant_seeds") != list(seeds)
        ):
            raise MonteCarloContractError(f"Partition {partition_id} completion contract is invalid.")
        artifacts = complete.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != _PARTITION_ARTIFACTS:
            raise MonteCarloContractError(f"Partition {partition_id} artifact ledger is incomplete.")
        for filename, raw_metadata in artifacts.items():
            if not isinstance(raw_metadata, dict):
                raise MonteCarloContractError(f"Partition {partition_id} metadata is malformed.")
            artifact_path = partition_dir / filename
            metadata = _verify_file(
                artifact_path,
                raw_metadata.get("sha256"),
                label=f"partition {partition_id} {filename}",
                row_count=raw_metadata.get("row_count"),
            )
            _record_source(source_ledger, root, artifact_path, metadata)
        if (
            str(row.run_diagnostics_sha256) != str(artifacts["run_diagnostics.csv"]["sha256"])
            or str(row.stock_hourly_sha256) != str(artifacts["stock_hourly.csv"]["sha256"])
            or int(row.stock_hourly_row_count) != int(artifacts["stock_hourly.csv"]["row_count"])
        ):
            raise MonteCarloContractError(f"Partition index and ledger disagree for {partition_id}.")
        diagnostics_path = _safe_path(root, row.run_diagnostics_path, label="partition diagnostics")
        hourly_path = _safe_path(root, row.stock_hourly_path, label="partition stock-hourly")
        if diagnostics_path != partition_dir / "run_diagnostics.csv" or hourly_path != partition_dir / "stock_hourly.csv":
            raise MonteCarloContractError(f"Partition index paths are inconsistent for {partition_id}.")
        manifest_ids = _read_run_ids(
            partition_dir / "run_manifest.csv", label="run manifest"
        )
        diagnostic_ids = _read_run_ids(
            partition_dir / "run_diagnostics.csv", label="run diagnostics"
        )
        if (
            len(manifest_ids) != expected_partition_runs
            or manifest_ids.duplicated().any()
            or diagnostic_ids.duplicated().any()
            or set(manifest_ids) != set(diagnostic_ids)
            or complete.get("expected_run_id_sha256")
            != canonical_sha256({"run_ids": sorted(set(manifest_ids))})
        ):
            raise MonteCarloContractError(f"Partition {partition_id} run-ID ledger is invalid.")
        partition_input_ledger.append(
            {
                "partition_id": partition_id,
                "run_count": expected_partition_runs,
                "partition_complete_sha256": str(row.partition_complete_sha256),
                "run_manifest_sha256": str(
                    artifacts["run_manifest.csv"]["sha256"]
                ),
                "run_diagnostics_sha256": str(
                    artifacts["run_diagnostics.csv"]["sha256"]
                ),
            }
        )

    status = postprocessing_status(root)
    if status.get("status") != "PASS":
        raise MonteCarloContractError(
            f"Post-processing commit authentication failed: {status.get('reason', status.get('status'))}."
        )
    if (
        post_summary.get("status") != "PASS"
        or post_summary.get("postprocess_contract_version") != POSTPROCESS_CONTRACT_VERSION
        or post_summary.get("source_execution_status") != "PASS"
        or post_summary.get("source_design_sha256") != design_sha
        or int(post_summary.get("processed_run_count", -1)) != expected_runs
        or int(post_summary.get("expected_run_count", -2)) != expected_runs
        or int(post_summary.get("source_partition_count", -3)) != 54
        or int(post_summary.get("dwelling_hour_files_read", -1)) != 0
        or post_summary.get("stock_weights_sha256")
        != design.get("stock_weights_sha256")
        or post_summary.get("stock_weights_source_sha256")
        != design.get("stock_weights_source_sha256")
        or post_summary.get("source_partition_input_ledger_sha256")
        != canonical_sha256(partition_input_ledger)
    ):
        raise MonteCarloContractError("Post-processing summary does not match the completed production run.")
    if "not complete prediction intervals" not in str(
        post_summary.get("interval_interpretation", "")
    ):
        raise MonteCarloContractError(
            "Post-processing summary lacks the empirical-interval limitation."
        )
    post_meta = _verify_file(post_path, _sha256_file(post_path), label="post-processing summary")
    _record_source(source_ledger, root, post_path, post_meta)
    post_artifacts = post_summary.get("output_artifacts")
    if not isinstance(post_artifacts, dict) or set(post_artifacts) != _POSTPROCESS_ARTIFACTS:
        if isinstance(post_artifacts, dict) and MODEL_SCENARIO_OUTPUT_FILENAME in post_artifacts:
            raise MonteCarloContractError("Central-only reporting cannot consume structural-scenario deltas.")
        raise MonteCarloContractError("Post-processing output ledger is incomplete.")
    for filename, raw_metadata in post_artifacts.items():
        if not isinstance(raw_metadata, dict):
            raise MonteCarloContractError("Post-processing artifact metadata is malformed.")
        path = root / filename
        metadata = _verify_file(
            path,
            raw_metadata.get("sha256"),
            label=f"post-processing artifact {filename}",
            row_count=raw_metadata.get("row_count"),
        )
        _record_source(source_ledger, root, path, metadata)

    aggregation = _read_csv(root / "stock_aggregation.csv", label="stock aggregation")
    contributions = _read_csv(root / "stock_contributions.csv", label="stock contributions")
    distributions = _read_csv(root / "stock_distribution_summary.csv", label="stock distributions")
    variance = _read_csv(root / VARIANCE_OUTPUT_FILENAME, label="variance contributions")
    paired = _read_csv(root / RENOVATION_OUTPUT_FILENAME, label="paired renovation deltas")
    _validate_stock_tables(aggregation, contributions, distributions)
    _validate_postprocess_tables(variance, paired)
    return AuthenticatedReportingInputs(
        production_dir=root,
        design=design,
        stock_summary=stock_summary,
        postprocess_summary=post_summary,
        stock_aggregation=aggregation,
        stock_contributions=contributions,
        stock_distribution=distributions,
        variance_contributions=variance,
        paired_renovation_deltas=paired,
        source_artifacts=dict(sorted(source_ledger.items())),
    )


def authenticate_reporting_inputs(
    production_dir: str | Path = DEFAULT_PRODUCTION_OUTPUT_DIR,
) -> AuthenticatedReportingInputs:
    """Return reporting inputs or normalize malformed contracts to one error type."""

    try:
        return _authenticate_reporting_inputs_unchecked(production_dir)
    except MonteCarloContractError:
        raise
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise MonteCarloContractError(
            "Completed production artifacts contain malformed reporting metadata."
        ) from exc


def _style() -> None:
    plt = _pyplot()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "savefig.dpi": 300,
        }
    )


def _pyplot() -> Any:
    """Load the non-interactive plotting backend only for report generation."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _save_figure(figure: plt.Figure, output_dir: Path, basename: str) -> dict[str, Path]:
    plt = _pyplot()
    png = output_dir / f"{basename}.png"
    pdf = output_dir / f"{basename}.pdf"
    figure.savefig(
        png,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "thermal_model.monte_carlo.reporting"},
    )
    figure.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "thermal_model.monte_carlo.reporting",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png, "pdf": pdf}


def _footer(figure: plt.Figure, text: str) -> None:
    figure.text(0.01, 0.015, text, ha="left", va="bottom", fontsize=7.2, color="#4A4A4A")


def _national_interval_figure(
    distribution: pd.DataFrame,
    *,
    metric: str,
    scale: float,
    ylabel: str,
    title: str,
    footer: str,
) -> plt.Figure:
    plt = _pyplot()
    selected = distribution.loc[
        (distribution["region"].astype(str) == NATIONAL_REGION)
        & (distribution["model_scenario_id"].astype(str) == "central")
        & (distribution["metric"].astype(str) == metric)
    ].set_index("climate_scenario_id").loc[list(RCP_ORDER)]
    medians = selected["median"].to_numpy(dtype=float) * scale
    lower = medians - selected["p05"].to_numpy(dtype=float) * scale
    upper = selected["p95"].to_numpy(dtype=float) * scale - medians
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(RCP_ORDER))
    axis.errorbar(
        x,
        medians,
        yerr=np.vstack((lower, upper)),
        fmt="o",
        markersize=7,
        capsize=5,
        elinewidth=1.8,
        color="#0072B2",
        ecolor="#56B4E9",
    )
    for position, value in zip(x, medians):
        axis.annotate(f"{value:,.2f}", (position, value), xytext=(0, 9), textcoords="offset points", ha="center")
    axis.set_xticks(x, [RCP_LABELS[item] for item in RCP_ORDER])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="x", visible=False)
    axis.margins(x=0.18, y=0.18)
    figure.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.20)
    _footer(figure, footer)
    return figure


def _contribution_figure(
    contributions: pd.DataFrame,
    *,
    dimension: str,
    order: tuple[str, ...],
    labels: Mapping[str, str] | None,
    title: str,
) -> plt.Figure:
    plt = _pyplot()
    selected = contributions.loc[
        (contributions["contribution_dimension"].astype(str) == dimension)
        & (contributions["model_scenario_id"].astype(str) == "central")
    ].copy()
    means = (
        selected.groupby(["climate_scenario_id", "contribution_value"], sort=False)["annual_heating_GWh"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(index=list(RCP_ORDER), columns=list(order), fill_value=0.0)
        / 1000.0
    )
    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
    figure, axis = plt.subplots(figsize=(7.2, 4.9))
    x = np.arange(len(RCP_ORDER))
    bottom = np.zeros(len(RCP_ORDER), dtype=float)
    for index, category in enumerate(order):
        values = means[category].to_numpy(dtype=float)
        axis.bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            color=palette[index % len(palette)],
            label=(labels or {}).get(category, category),
            linewidth=0.25,
            edgecolor="white",
        )
        bottom += values
    for position, total in zip(x, bottom):
        axis.annotate(f"{total:,.1f}", (position, total), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    axis.set_xticks(x, [RCP_LABELS[item] for item in RCP_ORDER])
    axis.set_ylabel("Annual ideal useful heating (TWh/year)")
    axis.set_title(title)
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(3, len(order)), frameon=False)
    axis.margins(x=0.12, y=0.13)
    figure.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.30)
    _footer(
        figure,
        "Mean across 18 included weather members per RCP · R1–R4 modelled stock · common Brussels weather forcing",
    )
    return figure


def _variance_figure(variance: pd.DataFrame) -> plt.Figure:
    plt = _pyplot()
    selected = variance.loc[
        (variance["metric"].astype(str) == "heating_intensity_kWh_m2")
        & (variance["model_scenario_id"].astype(str) == "central")
    ].copy()
    pooled = selected.groupby(["climate_scenario_id", "component"], sort=False)["sum_of_squares"].sum().unstack(fill_value=0.0)
    pooled = pooled.reindex(index=list(RCP_ORDER))
    totals = pooled.sum(axis=1)
    if not np.isfinite(totals.to_numpy(dtype=float)).all() or (totals <= 0.0).any():
        raise MonteCarloContractError(
            "Heating-intensity variance is zero or absent for at least one RCP."
        )
    shares = pooled.div(totals, axis=0) * 100.0
    components = ("weather_year", "occupant_seed", "weather_seed_interaction")
    labels = {
        "weather_year": "Weather member",
        "occupant_seed": "Occupant seed",
        "weather_seed_interaction": "Weather × occupant",
    }
    colors = ("#0072B2", "#E69F00", "#999999")
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(RCP_ORDER))
    bottom = np.zeros(len(RCP_ORDER), dtype=float)
    for component, color in zip(components, colors):
        values = shares[component].to_numpy(dtype=float)
        axis.bar(x, values, bottom=bottom, width=0.62, color=color, label=labels[component])
        for position, base, value in zip(x, bottom, values):
            if value >= 4.0:
                axis.text(position, base + value / 2.0, f"{value:.1f}%", ha="center", va="center", fontsize=8, color="white")
        bottom += values
    axis.set_xticks(x, [RCP_LABELS[item] for item in RCP_ORDER])
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("Share of pooled within-cell sum of squares (%)")
    axis.set_title("Weather and occupant contributions to heating-intensity variation")
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.89, bottom=0.29)
    _footer(
        figure,
        "Balanced ANOVA across unweighted R1–R4 archetype/state cells · empirical included ensemble; not total real-world uncertainty",
    )
    return figure


def _paired_renovation_figure(paired: pd.DataFrame) -> plt.Figure:
    plt = _pyplot()
    selected = paired.loc[paired["metric"].astype(str) == "heating_intensity_kWh_m2"].copy()
    baseline = pd.to_numeric(selected["baseline_value"], errors="raise").to_numpy(dtype=float)
    if (baseline <= 0.0).any():
        raise MonteCarloContractError("Paired heating-effect percentages require positive baselines.")
    selected["relative_change_percent"] = 100.0 * pd.to_numeric(selected["delta"], errors="raise") / baseline
    summaries = (
        selected.groupby(["climate_scenario_id", "comparison_state_id"])["relative_change_percent"]
        .quantile([0.05, 0.50, 0.95])
        .unstack()
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    x = np.arange(len(RCP_ORDER), dtype=float)
    offsets = (-0.13, 0.13)
    colors = ("#E69F00", "#009E73")
    markers = ("s", "o")
    for state, offset, color, marker in zip(STATE_ORDER[1:], offsets, colors, markers):
        rows = summaries.xs(state, level="comparison_state_id").loc[list(RCP_ORDER)]
        median = rows[0.50].to_numpy(dtype=float)
        lower = median - rows[0.05].to_numpy(dtype=float)
        upper = rows[0.95].to_numpy(dtype=float) - median
        axis.errorbar(
            x + offset,
            median,
            yerr=np.vstack((lower, upper)),
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=4,
            markersize=6,
            label=STATE_LABELS[state],
        )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_xticks(x, [RCP_LABELS[item] for item in RCP_ORDER])
    axis.set_ylabel("Change from existing-state heating intensity (%)")
    axis.set_title("Paired renovation effect on ideal useful heating demand")
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    axis.margins(x=0.12, y=0.15)
    figure.subplots_adjust(left=0.14, right=0.98, top=0.89, bottom=0.29)
    _footer(
        figure,
        "Median and empirical p05–p95 over exact common weather/occupant pairs · unweighted R1–R4 archetype draws · common Brussels weather",
    )
    return figure


def _create_figures(inputs: AuthenticatedReportingInputs, output_dir: Path) -> dict[str, dict[str, Path]]:
    _style()
    figures: dict[str, dict[str, Path]] = {}
    national_specs = (
        (
            "mc_national_annual_heating",
            "annual_heating_GWh",
            1.0 / 1000.0,
            "Annual ideal useful heating (TWh/year)",
            "National modelled-stock annual heating demand",
            "Ideal useful demand · R1–R4 only · common Brussels weather · median and empirical p05–p95 across 18 members; not a prediction interval",
        ),
        (
            "mc_national_potential_cooling",
            "annual_potential_sensible_cooling_GWh",
            1.0 / 1000.0,
            "Annual potential sensible cooling (TWh/year)",
            "National modelled-stock cooling-demand potential",
            "Universal ideal 26 °C control; not AC adoption or electricity · R1–R4 · common Brussels weather · empirical p05–p95",
        ),
        (
            "mc_national_coincident_heating_peak",
            "coincident_peak_heating_MW",
            1.0 / 1000.0,
            "Coincident ideal useful heating peak (GW)",
            "National coincident heating-demand peak",
            "Peak of aggregated hourly R1–R4 demand · common Brussels weather may overstate geographic coincidence · empirical p05–p95",
        ),
    )
    for basename, metric, scale, ylabel, title, footer in national_specs:
        figures[basename] = _save_figure(
            _national_interval_figure(
                inputs.stock_distribution,
                metric=metric,
                scale=scale,
                ylabel=ylabel,
                title=title,
                footer=footer,
            ),
            output_dir,
            basename,
        )
    contribution_specs = (
        ("mc_heating_contribution_region", "region", REGION_ORDER, None, "Mean stock heating contribution by region"),
        ("mc_heating_contribution_dwelling_type", "dwelling_type", DWELLING_ORDER, None, "Mean stock heating contribution by dwelling type"),
        ("mc_heating_contribution_construction_period", "construction_period", PERIOD_ORDER, None, "Mean stock heating contribution by construction period"),
        ("mc_heating_contribution_renovation_state", "state_id", STATE_ORDER, STATE_LABELS, "Mean stock heating contribution by renovation state"),
    )
    for basename, dimension, order, labels, title in contribution_specs:
        figures[basename] = _save_figure(
            _contribution_figure(
                inputs.stock_contributions,
                dimension=dimension,
                order=order,
                labels=labels,
                title=title,
            ),
            output_dir,
            basename,
        )
    figures["mc_heating_variance_weather_occupant"] = _save_figure(
        _variance_figure(inputs.variance_contributions),
        output_dir,
        "mc_heating_variance_weather_occupant",
    )
    figures["mc_paired_renovation_heating_effect"] = _save_figure(
        _paired_renovation_figure(inputs.paired_renovation_deltas),
        output_dir,
        "mc_paired_renovation_heating_effect",
    )
    return figures


def _fmt_interval(row: pd.Series, *, scale: float, unit: str) -> str:
    return (
        f"{float(row['median']) * scale:,.2f} "
        f"[{float(row['p05']) * scale:,.2f}, {float(row['p95']) * scale:,.2f}] {unit}"
    )


def _relative_markdown_link(path: Path, parent: Path) -> str:
    return Path(os.path.relpath(path, parent)).as_posix()


def _report_markdown(
    inputs: AuthenticatedReportingInputs,
    figure_paths: Mapping[str, Mapping[str, Path]],
    report_path: Path,
) -> str:
    national = inputs.stock_distribution.loc[
        (inputs.stock_distribution["region"].astype(str) == NATIONAL_REGION)
        & (inputs.stock_distribution["model_scenario_id"].astype(str) == "central")
    ]
    national_index = national.set_index(["climate_scenario_id", "metric"])
    lines = [
        "# Gate-5 residential demand results",
        "",
        "**Authenticated reporting status: PASS.** These results are ideal useful space-heating demand and potential sensible cooling demand from the single-zone 5R1C model. They are not delivered fuel, heat-pump electricity, or billed energy.",
        "",
        "The stock scope is **R1–R4 only** (5,537,385 modelled dwellings); the R5–R6 residual is excluded. All regions use the same Brussels-area weather member. Regional totals therefore reflect stock composition, not Belgian spatial climate gradients, and the national coincident peak is conditional on geographically common weather.",
        "",
        "Intervals below are descriptive empirical p05–p95 ranges over the included ensemble dimensions. They are **not complete prediction intervals** and do not include every source of climate, building, stock, behavioural, or model uncertainty.",
        "For national stock energy and coincident peaks, occupant seeds are averaged within each physical stock cell before stock aggregation; those p05–p95 ranges therefore describe the 18 weather members within each RCP. Occupant variability is retained in the separate crossed ANOVA and paired-dwelling results.",
        "",
        "## National demand and coincident peak",
        "",
        "| Pathway | Annual heating | Potential sensible cooling | Coincident heating peak |",
        "|---|---:|---:|---:|",
    ]
    for rcp in RCP_ORDER:
        heating = national_index.loc[(rcp, "annual_heating_GWh")]
        cooling = national_index.loc[(rcp, "annual_potential_sensible_cooling_GWh")]
        peak = national_index.loc[(rcp, "coincident_peak_heating_MW")]
        lines.append(
            f"| {RCP_LABELS[rcp]} | {_fmt_interval(heating, scale=1/1000, unit='TWh/year')} | "
            f"{_fmt_interval(cooling, scale=1/1000, unit='TWh/year')} | "
            f"{_fmt_interval(peak, scale=1/1000, unit='GW')} |"
        )
    lines.extend(["", "Cooling is a technical potential under universal unlimited ideal 26 °C control; it does not represent cooling-system ownership, adoption, efficiency, or electricity demand.", ""])
    for basename in FIGURE_BASENAMES[:3]:
        path = figure_paths[basename]["png"]
        lines.append(f"![{basename}]({_relative_markdown_link(path, report_path.parent)})")
        lines.append("")

    lines.extend([
        "## Stock contributions",
        "",
        "The contribution charts use mean annual ideal useful heating across the 18 included weather members within each RCP. They retain dwelling-count stock weights and do not average archetypes equally.",
        "",
        "| Dimension | Pathway | Largest mean contributor | Mean share |",
        "|---|---|---|---:|",
    ])
    for dimension in ("region", "dwelling_type", "construction_period", "state_id"):
        selected = inputs.stock_contributions.loc[
            inputs.stock_contributions["contribution_dimension"].astype(str) == dimension
        ]
        means = selected.groupby(["climate_scenario_id", "contribution_value"])["annual_heating_GWh"].mean()
        for rcp in RCP_ORDER:
            row = means.loc[rcp]
            category = str(row.idxmax())
            share = float(row.max() / row.sum()) if float(row.sum()) > 0.0 else 0.0
            label = STATE_LABELS.get(category, category)
            lines.append(f"| {dimension.replace('_', ' ').title()} | {RCP_LABELS[rcp]} | {label} | {share:.1%} |")
    lines.append("")
    for basename in FIGURE_BASENAMES[3:7]:
        path = figure_paths[basename]["png"]
        lines.append(f"![{basename}]({_relative_markdown_link(path, report_path.parent)})")
        lines.append("")

    variance = inputs.variance_contributions.loc[
        inputs.variance_contributions["metric"].astype(str) == "heating_intensity_kWh_m2"
    ]
    pooled = variance.groupby(["climate_scenario_id", "component"])["sum_of_squares"].sum().unstack(fill_value=0.0)
    totals = pooled.sum(axis=1)
    if not np.isfinite(totals.to_numpy(dtype=float)).all() or (totals <= 0.0).any():
        raise MonteCarloContractError(
            "Heating-intensity variance is zero or absent for at least one RCP."
        )
    pooled = pooled.div(totals, axis=0)
    lines.extend([
        "## Weather and occupant variability",
        "",
        "The balanced within-cell ANOVA is a descriptive attribution for this crossed experiment, not a causal population variance decomposition.",
        "",
        "| Pathway | Weather member | Occupant seed | Interaction |",
        "|---|---:|---:|---:|",
    ])
    for rcp in RCP_ORDER:
        row = pooled.loc[rcp]
        lines.append(
            f"| {RCP_LABELS[rcp]} | {float(row['weather_year']):.1%} | "
            f"{float(row['occupant_seed']):.1%} | {float(row['weather_seed_interaction']):.1%} |"
        )
    variance_path = figure_paths["mc_heating_variance_weather_occupant"]["png"]
    lines.extend(["", f"![mc_heating_variance_weather_occupant]({_relative_markdown_link(variance_path, report_path.parent)})", ""])

    paired = inputs.paired_renovation_deltas.loc[
        inputs.paired_renovation_deltas["metric"].astype(str) == "heating_intensity_kWh_m2"
    ].copy()
    paired["relative_change_percent"] = 100.0 * pd.to_numeric(paired["delta"], errors="raise") / pd.to_numeric(paired["baseline_value"], errors="raise")
    effects = paired.groupby(["climate_scenario_id", "comparison_state_id"])["relative_change_percent"].quantile([0.05, 0.50, 0.95]).unstack()
    lines.extend([
        "## Paired renovation effects",
        "",
        "Renovated-minus-existing contrasts use exact common weather members and occupant seeds. Percentages below are unweighted across archetype draws; negative values mean lower ideal useful heating intensity.",
        "",
        "| Pathway | Renovation state | Median change | Empirical p05–p95 |",
        "|---|---|---:|---:|",
    ])
    for rcp in RCP_ORDER:
        for state in STATE_ORDER[1:]:
            row = effects.loc[(rcp, state)]
            lines.append(
                f"| {RCP_LABELS[rcp]} | {STATE_LABELS[state]} | {float(row[0.50]):.1f}% | "
                f"{float(row[0.05]):.1f}% to {float(row[0.95]):.1f}% |"
            )
    paired_path = figure_paths["mc_paired_renovation_heating_effect"]["png"]
    lines.extend(["", f"![mc_paired_renovation_heating_effect]({_relative_markdown_link(paired_path, report_path.parent)})", ""])

    design = inputs.design
    summary = inputs.stock_summary
    lines.extend([
        "## Reproducibility record",
        "",
        f"- Completed dwelling-year runs: {int(summary['completed_run_count']):,}",
        f"- Weather members: {len(design['weather_members'])} (18 per RCP)",
        f"- Occupant seeds per weather/member cell: {len(design['occupant_seeds'])}",
        f"- Seed convergence evidence: {summary['convergence_evidence']['status']}",
        "- Stock basis: 2050 renovation-state scenario on the fixed 2025 R1–R4 dwelling denominator",
        f"- Streaming design SHA-256: `{design['design_sha256']}`",
        f"- Thermal-model contract version: `{design.get('model_contract_version', 'not recorded')}`",
        f"- Stock-weight content SHA-256: `{design.get('stock_weights_sha256', 'not recorded')}`",
        f"- Stock-weight source SHA-256: `{design.get('stock_weights_source_sha256', 'not recorded')}`",
        f"- Thermal assumptions SHA-256: `{design.get('central_thermal_assumptions_sha256', 'not recorded')}`",
        f"- Behaviour assumptions SHA-256: `{design.get('behaviour_assumptions_sha256', 'not recorded')}`",
        f"- Occupant distribution SHA-256: `{design.get('occupant_distribution_sha256', 'not recorded')}`",
        f"- Complete source/output checksum ledger: `{REPORTING_SUMMARY_FILENAME}`",
        f"- Figure-specific source/output ledger: `{FIGURE_PROVENANCE_FILENAME}`",
        "",
        "The per-member climate and forcing checksums, partition checksums, post-processing checksums, and every figure checksum are retained in the reporting provenance ledger.",
        "",
    ])
    return "\n".join(lines)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_production_report(
    production_dir: str | Path = DEFAULT_PRODUCTION_OUTPUT_DIR,
    *,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate completed results and atomically publish report artifacts."""

    inputs = authenticate_reporting_inputs(production_dir)
    output_dir = Path(figure_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = (
        inputs.production_dir / REPORT_FILENAME
        if report_path is None
        else Path(report_path).resolve()
    )
    authenticated_source_paths = {
        (inputs.production_dir / relative).resolve()
        for relative in inputs.source_artifacts
    }
    if report in authenticated_source_paths or report == inputs.production_dir / REPORTING_SUMMARY_FILENAME:
        raise MonteCarloContractError(
            "The report path would overwrite an authenticated source or commit marker."
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    summary_path = inputs.production_dir / REPORTING_SUMMARY_FILENAME
    with tempfile.TemporaryDirectory(prefix=".gate5_reporting_", dir=output_dir) as staging_raw:
        staging = Path(staging_raw)
        staged_figures = _create_figures(inputs, staging)
        final_figures = {
            basename: {
                extension: output_dir / path.name for extension, path in paths.items()
            }
            for basename, paths in staged_figures.items()
        }
        markdown = _report_markdown(inputs, final_figures, report)
        staged_report = staging / REPORT_FILENAME
        staged_report.write_text(markdown, encoding="utf-8")
        for paths in staged_figures.values():
            for path in paths.values():
                path.replace(output_dir / path.name)
        report_temp = report.with_name(f".{report.name}.writing")
        staged_report.replace(report_temp)
        report_temp.replace(report)

    output_artifacts: dict[str, dict[str, Any]] = {
        "report": {"path": str(report), "sha256": _sha256_file(report)}
    }
    figure_artifacts: dict[str, dict[str, Any]] = {}
    for basename in FIGURE_BASENAMES:
        for extension in ("png", "pdf"):
            key = f"{basename}.{extension}"
            path = output_dir / key
            metadata = {"path": str(path), "sha256": _sha256_file(path)}
            output_artifacts[key] = metadata
            figure_artifacts[key] = metadata
    figure_provenance: dict[str, Any] = {
        "status": "PASS",
        "reporting_contract_version": REPORTING_CONTRACT_VERSION,
        "source_design_sha256": inputs.design["design_sha256"],
        "source_artifacts": inputs.source_artifacts,
        "figure_artifacts": figure_artifacts,
        "figure_count": len(FIGURE_BASENAMES),
        "interpretation": (
            "ideal useful demand for the R1-R4 modelled stock; common Brussels "
            "weather; empirical represented-ensemble intervals, not complete "
            "prediction intervals"
        ),
    }
    figure_provenance["figure_provenance_sha256"] = canonical_sha256(
        figure_provenance
    )
    figure_provenance_path = output_dir / FIGURE_PROVENANCE_FILENAME
    _atomic_json(figure_provenance, figure_provenance_path)
    output_artifacts["figure_provenance"] = {
        "path": str(figure_provenance_path),
        "sha256": _sha256_file(figure_provenance_path),
    }
    payload: dict[str, Any] = {
        "status": "PASS",
        "reporting_contract_version": REPORTING_CONTRACT_VERSION,
        "scope": "central 2050 R1-R4 ideal useful heating and potential sensible cooling",
        "source_design_sha256": inputs.design["design_sha256"],
        "source_artifacts": inputs.source_artifacts,
        "output_artifacts": output_artifacts,
        "figure_count": len(FIGURE_BASENAMES),
        "figure_format_policy": "one independent figure per basename, each exported as PNG and PDF",
        "weather_limitation": "all Belgian regions use the same Brussels-area weather member",
        "interval_interpretation": "empirical represented-ensemble intervals; not complete prediction intervals",
    }
    payload["reporting_sha256"] = canonical_sha256(payload)
    _atomic_json(payload, summary_path)
    return payload


def reporting_status(
    production_dir: str | Path = DEFAULT_PRODUCTION_OUTPUT_DIR,
) -> dict[str, Any]:
    """Reauthenticate sources and every committed reporting output."""

    root = Path(production_dir).resolve()
    path = root / REPORTING_SUMMARY_FILENAME
    if not path.is_file():
        return {"status": "NOT_RUN", "production_dir": str(root), "summary_path": str(path)}
    try:
        summary = _read_json(path, label="results-reporting summary")
        if summary.get("status") != "PASS" or summary.get("reporting_contract_version") != REPORTING_CONTRACT_VERSION:
            raise MonteCarloContractError("Results-reporting summary status/contract is invalid.")
        if int(summary.get("figure_count", -1)) != len(FIGURE_BASENAMES):
            raise MonteCarloContractError("Results-reporting summary has an invalid figure count.")
        declared = _valid_sha256(summary.get("reporting_sha256"), label="reporting_sha256")
        unsigned = {key: value for key, value in summary.items() if key != "reporting_sha256"}
        if canonical_sha256(unsigned) != declared:
            raise MonteCarloContractError("Results-reporting summary content hash is invalid.")
        inputs = authenticate_reporting_inputs(root)
        if summary.get("source_design_sha256") != inputs.design["design_sha256"] or summary.get("source_artifacts") != inputs.source_artifacts:
            raise MonteCarloContractError("Results-reporting source ledger changed.")
        outputs = summary.get("output_artifacts")
        expected_keys = {
            "report",
            "figure_provenance",
            *(f"{name}.{suffix}" for name in FIGURE_BASENAMES for suffix in ("png", "pdf")),
        }
        if not isinstance(outputs, dict) or set(outputs) != expected_keys:
            raise MonteCarloContractError("Results-reporting output ledger is incomplete.")
        for name, metadata in outputs.items():
            if not isinstance(metadata, dict):
                raise MonteCarloContractError("Results-reporting output metadata is malformed.")
            _verify_file(
                Path(str(metadata.get("path"))).resolve(),
                metadata.get("sha256"),
                label=f"reporting output {name}",
            )
        provenance = _read_json(
            Path(str(outputs["figure_provenance"]["path"])).resolve(),
            label="figure provenance",
        )
        provenance_sha = _valid_sha256(
            provenance.get("figure_provenance_sha256"),
            label="figure_provenance_sha256",
        )
        provenance_unsigned = {
            key: value
            for key, value in provenance.items()
            if key != "figure_provenance_sha256"
        }
        if canonical_sha256(provenance_unsigned) != provenance_sha:
            raise MonteCarloContractError("Figure-provenance content hash is invalid.")
        expected_figure_artifacts = {
            key: value
            for key, value in outputs.items()
            if key not in {"report", "figure_provenance"}
        }
        if (
            provenance.get("source_artifacts") != inputs.source_artifacts
            or provenance.get("source_design_sha256") != inputs.design["design_sha256"]
            or provenance.get("figure_artifacts") != expected_figure_artifacts
            or int(provenance.get("figure_count", -1)) != len(FIGURE_BASENAMES)
        ):
            raise MonteCarloContractError("Figure provenance does not match its sources/outputs.")
        return {
            "status": "PASS",
            "production_dir": str(root),
            "source_design_sha256": inputs.design["design_sha256"],
            "figure_count": len(FIGURE_BASENAMES),
            "report_path": outputs["report"]["path"],
            "reporting_sha256": declared,
        }
    except (
        AttributeError,
        MonteCarloContractError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        return {"status": "INVALID", "production_dir": str(root), "reason": str(exc)}


__all__ = [
    "AuthenticatedReportingInputs",
    "DEFAULT_FIGURE_DIR",
    "FIGURE_PROVENANCE_FILENAME",
    "FIGURE_BASENAMES",
    "REPORTING_CONTRACT_VERSION",
    "REPORTING_SUMMARY_FILENAME",
    "authenticate_reporting_inputs",
    "generate_production_report",
    "reporting_status",
]
