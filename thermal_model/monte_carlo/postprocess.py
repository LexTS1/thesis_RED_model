"""Authenticated bounded-memory post-processing for completed stock runs.

The stock runner deliberately keeps run diagnostics inside weather/scenario
partitions.  This module authenticates that partition ledger, streams only the
annual diagnostic columns needed for analysis, and writes the root run-level
summaries promised by the Gate-5 contract.  Dwelling-hour files are neither
opened nor required.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .aggregation import load_stock_weights, validate_stock_weights
from .contracts import MonteCarloContractError, canonical_sha256
from .design import (
    DEFAULT_ANALYSIS_GROUPS,
    DEFAULT_DISTRIBUTION_METRICS,
    distribution_summary,
    paired_model_scenario_deltas,
    paired_renovation_deltas,
    variance_contributions,
)
from .stock_streaming import BELGIUM_REGION_ID


POSTPROCESS_CONTRACT_VERSION = "gate5_production_postprocess_v1"
POSTPROCESS_SUMMARY_FILENAME = "postprocessing_summary.json"
DEFAULT_CHUNK_ROWS = 2_000
UNWEIGHTED_OUTPUT_FILENAME = "distribution_summary.csv"
VARIANCE_OUTPUT_FILENAME = "variance_contributions.csv"
RENOVATION_OUTPUT_FILENAME = "paired_renovation_deltas.csv"
MODEL_SCENARIO_OUTPUT_FILENAME = "paired_model_scenario_deltas.csv"
WEIGHTED_OUTPUT_FILENAME = "stock_weighted_distribution_summary.csv"

_GROUP_COLUMNS = tuple(DEFAULT_ANALYSIS_GROUPS)
_VARIANCE_METRICS = (
    "heating_intensity_kWh_m2",
    "cooling_intensity_kWh_m2",
    "peak_heating_W",
    "peak_cooling_W",
)
_SPOOL_COLUMNS = (
    "run_id",
    "archetype_id",
    "state_id",
    "climate_scenario_id",
    "weather_member_id",
    "weather_pair_id",
    "observed_pvgis_year",
    "occupant_seed",
    "model_scenario_id",
    "model_scenario_axis",
    *DEFAULT_DISTRIBUTION_METRICS,
)
_MANIFEST_IDENTITY_COLUMNS = (
    "archetype_id",
    "state_id",
    "archetype_state_sha256",
    "climate_scenario_id",
    "weather_member_id",
    "weather_pair_id",
    "observed_pvgis_year",
    "occupant_seed",
    "model_scenario_id",
    "model_scenario_axis",
    "weather_contract_sha256",
    "model_scenario_sha256",
    "weather_forcing_sha256",
    "effective_thermal_assumptions_sha256",
    "behaviour_assumptions_sha256",
    "occupant_distribution_sha256",
)
_GLOBAL_PROVENANCE_MAP = {
    "model_contract_version": "model_contract_version",
    "central_thermal_assumptions_sha256": "central_thermal_assumptions_sha256",
    "behaviour_assumptions_sha256": "behaviour_assumptions_sha256",
    "occupant_distribution_sha256": "occupant_distribution_sha256",
}
_WEATHER_PROVENANCE_MAP = {
    "member_sha256": "member_sha256",
    "metadata_sha256": "metadata_sha256",
    "climate_manifest_sha256": "manifest_sha256",
    "morph_contract_sha256": "morph_contract_sha256",
    "facade_source_sha256_json": "facade_source_sha256_json",
    "weather_contract_sha256": "weather_contract_sha256",
    "weather_forcing_sha256": "weather_forcing_sha256",
}
_CELL_METADATA_COLUMNS = (
    "dwelling_type",
    "dwelling_class",
    "construction_period",
    "floor_area_m2",
    "archetype_state_sha256",
)
_WEIGHTED_GROUP_COLUMNS = (
    "climate_scenario_id",
    "model_scenario_id",
    "region",
)
_MANIFEST_STRING_COLUMNS = {
    "run_id",
    *(
        column
        for column in _MANIFEST_IDENTITY_COLUMNS
        if column not in {"observed_pvgis_year", "occupant_seed"}
    ),
}
_DIAGNOSTIC_STRING_COLUMNS = {
    "run_id",
    "archetype_id",
    "state_id",
    "climate_scenario_id",
    "weather_member_id",
    "weather_pair_id",
    "model_scenario_id",
    "model_scenario_axis",
    *(
        column
        for column in _MANIFEST_IDENTITY_COLUMNS
        if column not in {"observed_pvgis_year", "occupant_seed"}
    ),
    *_GLOBAL_PROVENANCE_MAP,
    *_WEATHER_PROVENANCE_MAP,
    *(column for column in _CELL_METADATA_COLUMNS if column != "floor_area_m2"),
}


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


def _verify_file(path: Path, expected_sha256: Any, *, label: str) -> str:
    digest = str(expected_sha256).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MonteCarloContractError(f"{label} ledger contains an invalid SHA-256 digest.")
    if not path.is_file():
        raise MonteCarloContractError(f"{label} artifact is missing: {path}.")
    actual = _sha256_file(path)
    if actual != digest:
        raise MonteCarloContractError(
            f"{label} checksum mismatch: expected {digest}, got {actual}."
        )
    return actual


def _safe_relative_path(root: Path, value: Any, *, label: str) -> Path:
    raw = Path(str(value))
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MonteCarloContractError(
            f"{label} path escapes the production directory: {value!r}."
        ) from exc
    return resolved


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )


def _append_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        mode="a" if path.exists() else "w",
        header=not path.exists(),
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )


def _spool_path(directory: Path, prefix: str, key: Sequence[Any]) -> Path:
    digest = canonical_sha256({"prefix": prefix, "key": [str(value) for value in key]})
    return directory / f"{prefix}_{digest[:24]}.csv"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise MonteCarloContractError(f"{label} is missing columns: {missing}.")


def _constant_column(frame: pd.DataFrame, column: str, expected: Any, *, label: str) -> None:
    observed = frame[column]
    if observed.isna().any() or not observed.astype(str).eq(str(expected)).all():
        values = sorted(observed.dropna().astype(str).unique().tolist())[:5]
        raise MonteCarloContractError(
            f"{label} has inconsistent {column!r}; observed {values}, expected {expected!r}."
        )


def _validated_design(root: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    design_path = root / "streaming_design_contract.json"
    summary_path = root / "monte_carlo_summary.json"
    design = _read_json(design_path, label="streaming design contract")
    summary = _read_json(summary_path, label="Monte Carlo summary")
    design_sha256 = str(design.get("design_sha256", ""))
    design_payload = {key: value for key, value in design.items() if key != "design_sha256"}
    if canonical_sha256(design_payload) != design_sha256:
        raise MonteCarloContractError(
            "Streaming design contract content does not reproduce its design_sha256."
        )
    if str(summary.get("design_sha256", "")) != design_sha256:
        raise MonteCarloContractError(
            "Monte Carlo summary and streaming design contract use different designs."
        )
    completed = int(summary.get("completed_run_count", -1))
    expected = int(design.get("expected_run_count", -2))
    if (
        expected <= 0
        or completed != expected
        or int(summary.get("expected_run_count", -3)) != expected
    ):
        raise MonteCarloContractError(
            "Post-processing requires a complete stock execution with matching "
            "expected/completed run counts."
        )
    partition_index_path = root / "partition_index.csv"
    artifact_ledger = summary.get("artifact_sha256")
    if not isinstance(artifact_ledger, dict) or "partition_index.csv" not in artifact_ledger:
        raise MonteCarloContractError(
            "Monte Carlo summary does not authenticate partition_index.csv."
        )
    _verify_file(
        partition_index_path,
        artifact_ledger["partition_index.csv"],
        label="root partition index",
    )
    try:
        partition_index = pd.read_csv(partition_index_path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MonteCarloContractError("Cannot parse the root partition index.") from exc
    required_index = {
        "partition_id",
        "weather_member_id",
        "climate_scenario_id",
        "model_scenario_id",
        "run_count",
        "run_diagnostics_path",
        "run_diagnostics_sha256",
        "partition_complete_sha256",
    }
    _require_columns(partition_index, required_index, label="partition index")
    if partition_index.empty or partition_index["partition_id"].duplicated().any():
        raise MonteCarloContractError("Partition index is empty or contains duplicate partitions.")
    expected_specs = {
        (
            str(item["partition_id"]),
            str(item["weather_member_id"]),
            str(item["model_scenario_id"]),
        )
        for item in design.get("partition_specs", ())
    }
    observed_specs = set(
        map(
            tuple,
            partition_index[
                ["partition_id", "weather_member_id", "model_scenario_id"]
            ].astype(str).to_numpy(),
        )
    )
    if not expected_specs or observed_specs != expected_specs:
        raise MonteCarloContractError(
            "Partition index does not exactly cover the design partition specifications."
        )
    if int(summary.get("partition_count", -1)) != len(partition_index):
        raise MonteCarloContractError("Partition count differs between summary and index.")
    return design, summary, partition_index


def _validated_weights(
    design: Mapping[str, Any], stock_weights: pd.DataFrame | None
) -> pd.DataFrame:
    supplied = load_stock_weights() if stock_weights is None else stock_weights
    weights = validate_stock_weights(
        supplied,
        require_authoritative_shape=bool(design.get("require_full_stock")),
    )
    observed_content = str(weights["stock_weights_sha256"].iloc[0])
    observed_source = str(weights["stock_weights_source_sha256"].iloc[0])
    if observed_content != str(design.get("stock_weights_sha256", "")):
        raise MonteCarloContractError(
            "Stock weights do not match the completed streaming design content checksum."
        )
    if observed_source != str(design.get("stock_weights_source_sha256", "")):
        raise MonteCarloContractError(
            "Stock weights do not match the completed streaming design source checksum."
        )
    return weights


def _manifest_for_partition(
    path: Path,
    *,
    ledger: Mapping[str, Any],
    expected_weather_member_id: str,
    expected_scenario_id: str,
    design: Mapping[str, Any],
) -> pd.DataFrame:
    artifacts = ledger.get("artifacts")
    if not isinstance(artifacts, dict) or not {
        "run_manifest.csv",
        "run_diagnostics.csv",
    }.issubset(artifacts):
        raise MonteCarloContractError(
            f"Partition ledger {path.parent.name!r} lacks manifest/diagnostics artifacts."
        )
    manifest_metadata = artifacts["run_manifest.csv"]
    if not isinstance(manifest_metadata, dict):
        raise MonteCarloContractError("Partition manifest ledger entry must be an object.")
    _verify_file(
        path,
        manifest_metadata.get("sha256"),
        label=f"partition {path.parent.name} run manifest",
    )
    try:
        manifest = pd.read_csv(
            path,
            keep_default_na=False,
            dtype={column: str for column in _MANIFEST_STRING_COLUMNS},
        )
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MonteCarloContractError(f"Cannot parse partition manifest {path}.") from exc
    _require_columns(
        manifest,
        {"run_id", "occupant_seed_rank", *_MANIFEST_IDENTITY_COLUMNS},
        label=f"partition {path.parent.name} manifest",
    )
    expected_rows = len(design["archetype_states"]) * len(design["occupant_seeds"])
    if (
        len(manifest) != expected_rows
        or int(manifest_metadata.get("row_count", -1)) != expected_rows
        or manifest["run_id"].astype(str).duplicated().any()
    ):
        raise MonteCarloContractError(
            f"Partition {path.parent.name} manifest is incomplete or duplicated."
        )
    _constant_column(
        manifest,
        "weather_member_id",
        expected_weather_member_id,
        label=f"partition {path.parent.name} manifest",
    )
    _constant_column(
        manifest,
        "model_scenario_id",
        expected_scenario_id,
        label=f"partition {path.parent.name} manifest",
    )
    expected_cells = {
        (str(item["archetype_id"]), str(item["state_id"]))
        for item in design["archetype_states"]
    }
    observed_cells = set(
        map(tuple, manifest[["archetype_id", "state_id"]].astype(str).to_numpy())
    )
    if observed_cells != expected_cells:
        raise MonteCarloContractError(
            f"Partition {path.parent.name} manifest does not cover every design cell."
        )
    seeds = tuple(int(value) for value in design["occupant_seeds"])
    seed_rank = {seed: rank for rank, seed in enumerate(seeds, start=1)}
    observed_seeds = pd.to_numeric(manifest["occupant_seed"], errors="coerce")
    observed_ranks = pd.to_numeric(manifest["occupant_seed_rank"], errors="coerce")
    if (
        observed_seeds.isna().any()
        or observed_ranks.isna().any()
        or set(observed_seeds.astype(int)) != set(seeds)
        or not np.equal(
            observed_ranks.to_numpy(dtype=float),
            observed_seeds.astype(int).map(seed_rank).to_numpy(dtype=float),
        ).all()
    ):
        raise MonteCarloContractError(
            f"Partition {path.parent.name} manifest changed the ordered seed bank."
        )
    counts = manifest.groupby(["archetype_id", "state_id"], sort=False)[
        "occupant_seed"
    ].nunique()
    if not counts.eq(len(seeds)).all():
        raise MonteCarloContractError(
            f"Partition {path.parent.name} manifest has incomplete cell/seed coverage."
        )
    return manifest


def _validate_chunk_identity(
    chunk: pd.DataFrame,
    expected_by_run_id: pd.DataFrame,
    *,
    partition_id: str,
) -> None:
    run_ids = chunk["run_id"].astype(str)
    unknown = sorted(set(run_ids).difference(expected_by_run_id.index))
    if unknown:
        raise MonteCarloContractError(
            f"Partition {partition_id} diagnostics contain unknown run IDs: {unknown[:3]}."
        )
    expected = expected_by_run_id.loc[run_ids].reset_index(drop=True)
    observed = chunk.reset_index(drop=True)
    for column in _MANIFEST_IDENTITY_COLUMNS:
        if column in {"observed_pvgis_year", "occupant_seed"}:
            left = pd.to_numeric(observed[column], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
            matches = np.equal(left, right)
        else:
            matches = observed[column].astype(str).to_numpy() == expected[column].astype(
                str
            ).to_numpy()
        if not bool(np.all(matches)):
            raise MonteCarloContractError(
                f"Partition {partition_id} diagnostics disagree with the authenticated "
                f"manifest column {column!r}."
            )


def _validate_chunk_provenance(
    chunk: pd.DataFrame,
    *,
    design: Mapping[str, Any],
    weather_record: Mapping[str, Any],
    scenario_record: Mapping[str, Any],
    cell_hashes: Mapping[tuple[str, str], str],
    partition_id: str,
) -> None:
    label = f"partition {partition_id} diagnostics"
    for column, design_key in _GLOBAL_PROVENANCE_MAP.items():
        _constant_column(chunk, column, design[design_key], label=label)
    for column, weather_key in _WEATHER_PROVENANCE_MAP.items():
        _constant_column(chunk, column, weather_record[weather_key], label=label)
    _constant_column(
        chunk,
        "model_scenario_axis",
        scenario_record["axis"],
        label=label,
    )
    observed_hashes = chunk[["archetype_id", "state_id", "archetype_state_sha256"]]
    for row in observed_hashes.drop_duplicates().itertuples(index=False):
        key = (str(row.archetype_id), str(row.state_id))
        if key not in cell_hashes or str(row.archetype_state_sha256) != cell_hashes[key]:
            raise MonteCarloContractError(
                f"{label} changes archetype-state provenance for {key}."
            )


def _update_cell_metadata(
    chunk: pd.DataFrame,
    metadata: dict[tuple[str, str], dict[str, Any]],
    *,
    partition_id: str,
) -> None:
    for key, group in chunk.groupby(["archetype_id", "state_id"], sort=False):
        normalized_key = (str(key[0]), str(key[1]))
        record: dict[str, Any] = {}
        for column in _CELL_METADATA_COLUMNS:
            values = group[column].dropna().unique()
            if len(values) != 1:
                raise MonteCarloContractError(
                    f"Partition {partition_id} has mixed cell metadata {column!r} "
                    f"for {normalized_key}."
                )
            record[column] = values[0]
        record["floor_area_m2"] = float(record["floor_area_m2"])
        if not np.isfinite(record["floor_area_m2"]) or record["floor_area_m2"] <= 0.0:
            raise MonteCarloContractError(
                f"Partition {partition_id} has invalid floor area for {normalized_key}."
            )
        previous = metadata.get(normalized_key)
        if previous is None:
            metadata[normalized_key] = record
            continue
        for column in _CELL_METADATA_COLUMNS:
            if column == "floor_area_m2":
                equal = np.isclose(
                    float(previous[column]),
                    float(record[column]),
                    rtol=0.0,
                    atol=1.0e-10,
                )
            else:
                equal = str(previous[column]) == str(record[column])
            if not equal:
                raise MonteCarloContractError(
                    f"Cell metadata {column!r} changes across partitions for {normalized_key}."
                )


def _weight_lookup(weights: pd.DataFrame) -> pd.DataFrame:
    regional = weights[
        ["archetype_id", "state_id", "region", "state_dwellings_2050"]
    ].copy(deep=True)
    national = (
        regional.groupby(["archetype_id", "state_id"], as_index=False, sort=False)[
            "state_dwellings_2050"
        ]
        .sum()
        .assign(region=BELGIUM_REGION_ID)
    )
    return pd.concat([regional, national], ignore_index=True)


def _validate_weight_metadata(
    weights: pd.DataFrame,
    cell_metadata: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    expected_cells = set(
        map(tuple, weights[["archetype_id", "state_id"]].astype(str).to_numpy())
    )
    if set(cell_metadata) != expected_cells:
        raise MonteCarloContractError(
            "Authenticated diagnostics and stock weights cover different physics cells."
        )
    for row in weights[
        ["archetype_id", "state_id", "dwelling_type", "construction_period"]
    ].drop_duplicates().itertuples(index=False):
        key = (str(row.archetype_id), str(row.state_id))
        metadata = cell_metadata[key]
        if str(row.dwelling_type) != str(metadata["dwelling_type"]) or str(
            row.construction_period
        ) != str(metadata["construction_period"]):
            raise MonteCarloContractError(
                f"Stock metadata disagrees with authenticated diagnostics for {key}."
            )


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, dtype=float)
    threshold = float(probability) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _weighted_distribution_summary(
    frame: pd.DataFrame,
    *,
    expected_total_weight: float,
) -> pd.DataFrame:
    _require_columns(
        frame,
        {
            *_WEIGHTED_GROUP_COLUMNS,
            "archetype_id",
            "state_id",
            "weather_member_id",
            "occupant_seed",
            "sample_weight_dwellings",
            *DEFAULT_DISTRIBUTION_METRICS,
        },
        label="stock-weighted distribution spool",
    )
    if frame[list(_WEIGHTED_GROUP_COLUMNS)].drop_duplicates().shape[0] != 1:
        raise MonteCarloContractError("A weighted spool mixes analysis groups.")
    weights = pd.to_numeric(
        frame["sample_weight_dwellings"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(weights).all()
        or (weights < 0.0).any()
        or not bool((weights > 0.0).any())
    ):
        raise MonteCarloContractError("Weighted distribution contains invalid weights.")
    selected = weights > 0.0
    frame = frame.loc[selected].copy(deep=True)
    weights = weights[selected]
    total_weight = float(weights.sum())
    if not np.isclose(total_weight, expected_total_weight, rtol=0.0, atol=1.0e-5):
        raise MonteCarloContractError(
            "Weighted empirical samples do not reconstruct the declared dwelling stock: "
            f"got {total_weight}, expected {expected_total_weight}."
        )
    identity = {
        column: str(frame[column].iloc[0]) for column in _WEIGHTED_GROUP_COLUMNS
    }
    records: list[dict[str, Any]] = []
    for metric in DEFAULT_DISTRIBUTION_METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise MonteCarloContractError(
                f"Stock-weighted distribution metric {metric!r} is non-finite."
            )
        mean = float(np.average(values, weights=weights))
        variance = float(np.average(np.square(values - mean), weights=weights))
        records.append(
            {
                **identity,
                "metric": metric,
                "raw_positive_weight_run_count": len(values),
                "physics_cell_count": frame[["archetype_id", "state_id"]]
                .drop_duplicates()
                .shape[0],
                "weather_member_count": frame["weather_member_id"].nunique(),
                "occupant_seed_count": frame["occupant_seed"].nunique(),
                "total_weight_dwellings": total_weight,
                "minimum": float(np.min(values)),
                "p05": _weighted_quantile(values, weights, 0.05),
                "p25": _weighted_quantile(values, weights, 0.25),
                "median": _weighted_quantile(values, weights, 0.50),
                "mean": mean,
                "p75": _weighted_quantile(values, weights, 0.75),
                "p95": _weighted_quantile(values, weights, 0.95),
                "maximum": float(np.max(values)),
                "weighted_population_standard_deviation": float(np.sqrt(variance)),
                "weight_basis": "2050 dwelling count",
                "quantile_definition": "inverse weighted empirical CDF",
                "interpretation": (
                    "stock-weighted dwelling/weather/occupant distribution; intensity "
                    "quantiles are weighted by dwellings, not conditioned floor area; "
                    "distinct from the unweighted per-archetype distribution_summary.csv"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _analysis_group_key(frame: pd.DataFrame) -> tuple[str, ...]:
    if frame[list(_GROUP_COLUMNS)].drop_duplicates().shape[0] != 1:
        raise MonteCarloContractError("An analysis spool mixes archetype/RCP/scenario groups.")
    return tuple(str(frame[column].iloc[0]) for column in _GROUP_COLUMNS)


def _append_group_spools(
    chunk: pd.DataFrame,
    directory: Path,
    paths: dict[tuple[str, ...], Path],
) -> None:
    for raw_key, group in chunk.groupby(list(_GROUP_COLUMNS), sort=True, dropna=False):
        key = tuple(str(value) for value in raw_key)
        path = paths.setdefault(key, _spool_path(directory, "analysis", key))
        _append_csv(group.loc[:, list(_SPOOL_COLUMNS)], path)


def _append_weighted_spools(
    chunk: pd.DataFrame,
    directory: Path,
    paths: dict[tuple[str, ...], Path],
    *,
    weights: pd.DataFrame,
    weather_counts: Mapping[str, int],
    seed_count: int,
) -> None:
    columns = [
        "archetype_id",
        "state_id",
        "climate_scenario_id",
        "weather_member_id",
        "occupant_seed",
        "model_scenario_id",
        *DEFAULT_DISTRIBUTION_METRICS,
    ]
    expanded = chunk.loc[:, columns].merge(
        weights,
        on=["archetype_id", "state_id"],
        how="left",
        validate="many_to_many",
    )
    if expanded["state_dwellings_2050"].isna().any():
        raise MonteCarloContractError(
            "A diagnostic physics cell has no regional/national stock weight."
        )
    denominators = expanded["climate_scenario_id"].map(weather_counts) * seed_count
    if denominators.isna().any() or (denominators <= 0).any():
        raise MonteCarloContractError("Cannot normalize weighted weather/seed samples.")
    expanded["sample_weight_dwellings"] = (
        pd.to_numeric(expanded["state_dwellings_2050"], errors="raise")
        / denominators.astype(float)
    )
    spool_columns = [
        *_WEIGHTED_GROUP_COLUMNS,
        "archetype_id",
        "state_id",
        "weather_member_id",
        "occupant_seed",
        "sample_weight_dwellings",
        *DEFAULT_DISTRIBUTION_METRICS,
    ]
    for raw_key, group in expanded.groupby(
        list(_WEIGHTED_GROUP_COLUMNS), sort=True, dropna=False
    ):
        key = tuple(str(value) for value in raw_key)
        path = paths.setdefault(key, _spool_path(directory, "weighted", key))
        _append_csv(group.loc[:, spool_columns], path)


def _output_ledger(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            row_count = max(sum(1 for _ in stream) - 1, 0)
    except (OSError, UnicodeDecodeError) as exc:
        raise MonteCarloContractError(f"Cannot count post-processing rows in {path}.") from exc
    return {"sha256": _sha256_file(path), "row_count": row_count}


def _empty_renovation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "archetype_id",
            "climate_scenario_id",
            "weather_member_id",
            "occupant_seed",
            "model_scenario_id",
            "baseline_state_id",
            "comparison_state_id",
            "metric",
            "baseline_value",
            "comparison_value",
            "delta",
        ]
    )


def _empty_model_scenario_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "archetype_id",
            "state_id",
            "climate_scenario_id",
            "weather_member_id",
            "weather_pair_id",
            "observed_pvgis_year",
            "occupant_seed",
            "baseline_model_scenario_id",
            "comparison_model_scenario_id",
            "comparison_model_scenario_axis",
            "metric",
            "baseline_value",
            "comparison_value",
            "delta",
        ]
    )


def postprocess_production_results(
    production_dir: str | Path,
    *,
    stock_weights: pd.DataFrame | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Authenticate and summarize a completed partitioned stock execution.

    Memory is bounded by one diagnostics chunk plus one analysis group.  Exact
    empirical values are kept in temporary skinny group spools, never in one
    whole-design DataFrame.  The function is safely rerunnable: every output is
    staged and atomically promoted, and the summary commit marker is written
    last.
    """

    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise MonteCarloContractError("chunk_rows must be a positive integer.")
    root = Path(production_dir).resolve()
    if not root.is_dir():
        raise MonteCarloContractError(f"Production directory does not exist: {root}.")
    design, source_summary, partition_index = _validated_design(root)
    weights = _validated_weights(design, stock_weights)
    source_design_path = root / "streaming_design_contract.json"
    source_summary_path = root / "monte_carlo_summary.json"
    source_design_sha256 = _sha256_file(source_design_path)
    source_summary_sha256 = _sha256_file(source_summary_path)
    source_partition_index_sha256 = _sha256_file(root / "partition_index.csv")

    weather_by_id = {
        str(item["weather_member_id"]): item for item in design["weather_members"]
    }
    scenario_by_id = {
        str(item["scenario_id"]): item for item in design["model_scenarios"]
    }
    cell_hashes = {
        (str(item["archetype_id"]), str(item["state_id"])): str(
            item["archetype_state_sha256"]
        )
        for item in design["archetype_states"]
    }
    seeds = tuple(int(value) for value in design["occupant_seeds"])
    weather_counts = defaultdict(int)
    for record in weather_by_id.values():
        weather_counts[str(record["climate_scenario_id"])] += 1
    expanded_weights = _weight_lookup(weights)
    expected_weight_by_region = (
        expanded_weights.groupby("region", sort=False)["state_dwellings_2050"]
        .sum()
        .to_dict()
    )

    input_ledger_records: list[dict[str, Any]] = []
    analysis_paths: dict[tuple[str, ...], Path] = {}
    weighted_paths: dict[tuple[str, ...], Path] = {}
    cell_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    scenario_provenance: dict[tuple[str, str], str] = {}
    total_processed_rows = 0

    with tempfile.TemporaryDirectory(prefix="gate5_postprocess_spool_") as spool_raw:
        spool_dir = Path(spool_raw)
        for index_row in partition_index.sort_values("partition_id", kind="stable").itertuples(
            index=False
        ):
            partition_id = str(index_row.partition_id)
            weather_member_id = str(index_row.weather_member_id)
            scenario_id = str(index_row.model_scenario_id)
            if weather_member_id not in weather_by_id or scenario_id not in scenario_by_id:
                raise MonteCarloContractError(
                    f"Partition {partition_id} references an unknown weather/scenario."
                )
            weather_record = weather_by_id[weather_member_id]
            scenario_record = scenario_by_id[scenario_id]
            if str(index_row.climate_scenario_id) != str(
                weather_record["climate_scenario_id"]
            ):
                raise MonteCarloContractError(
                    f"Partition {partition_id} changes its climate-scenario identity."
                )
            diagnostics_path = _safe_relative_path(
                root,
                index_row.run_diagnostics_path,
                label=f"partition {partition_id} diagnostics",
            )
            if diagnostics_path.parent.name != partition_id:
                raise MonteCarloContractError(
                    f"Partition {partition_id} diagnostics are stored under another partition."
                )
            complete_path = diagnostics_path.parent / "partition_complete.json"
            _verify_file(
                complete_path,
                index_row.partition_complete_sha256,
                label=f"partition {partition_id} completion ledger",
            )
            complete = _read_json(complete_path, label=f"partition {partition_id} completion")
            if (
                complete.get("status") != "PASS"
                or str(complete.get("design_sha256")) != str(design["design_sha256"])
                or str(complete.get("partition_id")) != partition_id
                or int(complete.get("run_count", -1)) != int(index_row.run_count)
            ):
                raise MonteCarloContractError(
                    f"Partition {partition_id} completion ledger is inconsistent."
                )
            artifacts = complete.get("artifacts")
            if not isinstance(artifacts, dict):
                raise MonteCarloContractError(
                    f"Partition {partition_id} completion ledger has no artifact map."
                )
            diagnostic_metadata = artifacts.get("run_diagnostics.csv")
            if not isinstance(diagnostic_metadata, dict) or str(
                diagnostic_metadata.get("sha256")
            ) != str(index_row.run_diagnostics_sha256):
                raise MonteCarloContractError(
                    f"Partition {partition_id} diagnostics hashes disagree across ledgers."
                )
            diagnostic_sha256 = _verify_file(
                diagnostics_path,
                index_row.run_diagnostics_sha256,
                label=f"partition {partition_id} run diagnostics",
            )
            manifest_path = diagnostics_path.parent / "run_manifest.csv"
            manifest = _manifest_for_partition(
                manifest_path,
                ledger=complete,
                expected_weather_member_id=weather_member_id,
                expected_scenario_id=scenario_id,
                design=design,
            )
            expected_rows = len(manifest)
            if (
                expected_rows != int(index_row.run_count)
                or int(diagnostic_metadata.get("row_count", -1)) != expected_rows
            ):
                raise MonteCarloContractError(
                    f"Partition {partition_id} row counts disagree across ledgers."
                )
            expected_by_run_id = manifest.set_index("run_id", drop=True)
            expected_by_run_id.index = expected_by_run_id.index.astype(str)
            seen_run_ids: set[str] = set()
            try:
                chunks = pd.read_csv(
                    diagnostics_path,
                    chunksize=chunk_rows,
                    keep_default_na=False,
                    float_precision="round_trip",
                    dtype={column: str for column in _DIAGNOSTIC_STRING_COLUMNS},
                )
                for chunk in chunks:
                    required_columns = {
                        "run_id",
                        *_SPOOL_COLUMNS,
                        *_MANIFEST_IDENTITY_COLUMNS,
                        *_GLOBAL_PROVENANCE_MAP,
                        *_WEATHER_PROVENANCE_MAP,
                        *_CELL_METADATA_COLUMNS,
                    }
                    _require_columns(
                        chunk,
                        required_columns,
                        label=f"partition {partition_id} diagnostics",
                    )
                    run_ids = chunk["run_id"].astype(str)
                    if run_ids.duplicated().any() or seen_run_ids.intersection(run_ids):
                        raise MonteCarloContractError(
                            f"Partition {partition_id} diagnostics contain duplicate run IDs."
                        )
                    _validate_chunk_identity(
                        chunk,
                        expected_by_run_id,
                        partition_id=partition_id,
                    )
                    _validate_chunk_provenance(
                        chunk,
                        design=design,
                        weather_record=weather_record,
                        scenario_record=scenario_record,
                        cell_hashes=cell_hashes,
                        partition_id=partition_id,
                    )
                    for column in (
                        "model_scenario_sha256",
                        "effective_thermal_assumptions_sha256",
                    ):
                        values = chunk[column].astype(str).unique()
                        if len(values) != 1:
                            raise MonteCarloContractError(
                                f"Partition {partition_id} mixes {column!r} provenance."
                            )
                        provenance_key = (scenario_id, column)
                        previous = scenario_provenance.setdefault(
                            provenance_key, str(values[0])
                        )
                        if previous != str(values[0]):
                            raise MonteCarloContractError(
                                f"Model scenario {scenario_id!r} changes {column!r} "
                                "across weather partitions."
                            )
                    for metric in DEFAULT_DISTRIBUTION_METRICS:
                        values = pd.to_numeric(chunk[metric], errors="coerce").to_numpy(
                            dtype=float
                        )
                        if not np.isfinite(values).all() or (values < 0.0).any():
                            raise MonteCarloContractError(
                                f"Partition {partition_id} metric {metric!r} is invalid."
                            )
                        chunk[metric] = values
                    _update_cell_metadata(
                        chunk,
                        cell_metadata,
                        partition_id=partition_id,
                    )
                    _append_group_spools(chunk, spool_dir, analysis_paths)
                    _append_weighted_spools(
                        chunk,
                        spool_dir,
                        weighted_paths,
                        weights=expanded_weights,
                        weather_counts=weather_counts,
                        seed_count=len(seeds),
                    )
                    seen_run_ids.update(run_ids)
                    total_processed_rows += len(chunk)
            except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                raise MonteCarloContractError(
                    f"Cannot stream partition diagnostics {diagnostics_path}."
                ) from exc
            if seen_run_ids != set(expected_by_run_id.index):
                raise MonteCarloContractError(
                    f"Partition {partition_id} diagnostics do not exactly match its manifest."
                )
            input_ledger_records.append(
                {
                    "partition_id": partition_id,
                    "run_count": expected_rows,
                    "partition_complete_sha256": str(index_row.partition_complete_sha256),
                    "run_manifest_sha256": str(
                        complete["artifacts"]["run_manifest.csv"]["sha256"]
                    ),
                    "run_diagnostics_sha256": diagnostic_sha256,
                }
            )

        if total_processed_rows != int(design["expected_run_count"]):
            raise MonteCarloContractError(
                "Streamed post-processing row count does not match the design."
            )
        _validate_weight_metadata(weights, cell_metadata)

        distribution_frames: list[pd.DataFrame] = []
        variance_frames: list[pd.DataFrame] = []
        max_analysis_group_rows = 0
        for key, path in sorted(analysis_paths.items()):
            frame = pd.read_csv(path, float_precision="round_trip")
            if _analysis_group_key(frame) != key:
                raise MonteCarloContractError("An analysis spool filename/key is inconsistent.")
            max_analysis_group_rows = max(max_analysis_group_rows, len(frame))
            distribution_frames.append(distribution_summary(frame))
            variance_frames.append(variance_contributions(frame))
        distributions = pd.concat(distribution_frames, ignore_index=True).sort_values(
            [*_GROUP_COLUMNS, "metric"], kind="stable"
        )
        variance = pd.concat(variance_frames, ignore_index=True).sort_values(
            [*_GROUP_COLUMNS, "metric", "component"], kind="stable"
        )

        weighted_frames: list[pd.DataFrame] = []
        max_weighted_group_rows = 0
        for key, path in sorted(weighted_paths.items()):
            frame = pd.read_csv(path, float_precision="round_trip")
            max_weighted_group_rows = max(max_weighted_group_rows, len(frame))
            expected_weight = float(expected_weight_by_region[key[2]])
            weighted_frames.append(
                _weighted_distribution_summary(
                    frame,
                    expected_total_weight=expected_weight,
                )
            )
        weighted = pd.concat(weighted_frames, ignore_index=True).sort_values(
            [*_WEIGHTED_GROUP_COLUMNS, "metric"], kind="stable"
        )

        with tempfile.TemporaryDirectory(
            prefix=".postprocess_staging_", dir=root
        ) as staging_raw:
            staging = Path(staging_raw)
            _write_csv(distributions, staging / UNWEIGHTED_OUTPUT_FILENAME)
            _write_csv(variance, staging / VARIANCE_OUTPUT_FILENAME)
            _write_csv(weighted, staging / WEIGHTED_OUTPUT_FILENAME)

            renovation_path = staging / RENOVATION_OUTPUT_FILENAME
            renovation_groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
            for key, path in analysis_paths.items():
                renovation_groups[(key[0], key[2], key[3])].append(path)
            renovation_written = False
            max_paired_group_rows = 0
            for _, paths in sorted(renovation_groups.items()):
                group = pd.concat(
                    [pd.read_csv(path, float_precision="round_trip") for path in sorted(paths)],
                    ignore_index=True,
                )
                max_paired_group_rows = max(max_paired_group_rows, len(group))
                deltas = paired_renovation_deltas(group)
                if not deltas.empty:
                    _append_csv(deltas, renovation_path)
                    renovation_written = True
            if not renovation_written:
                _write_csv(_empty_renovation_frame(), renovation_path)

            model_scenario_path = staging / MODEL_SCENARIO_OUTPUT_FILENAME
            multiple_scenarios = len(scenario_by_id) > 1
            max_model_scenario_group_rows = 0
            if multiple_scenarios:
                scenario_groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
                for key, path in analysis_paths.items():
                    scenario_groups[(key[0], key[1], key[2])].append(path)
                model_written = False
                for _, paths in sorted(scenario_groups.items()):
                    group = pd.concat(
                        [
                            pd.read_csv(path, float_precision="round_trip")
                            for path in sorted(paths)
                        ],
                        ignore_index=True,
                    )
                    max_model_scenario_group_rows = max(
                        max_model_scenario_group_rows, len(group)
                    )
                    deltas = paired_model_scenario_deltas(group)
                    if not deltas.empty:
                        _append_csv(deltas, model_scenario_path)
                        model_written = True
                if not model_written:
                    _write_csv(_empty_model_scenario_frame(), model_scenario_path)

            output_names = [
                UNWEIGHTED_OUTPUT_FILENAME,
                VARIANCE_OUTPUT_FILENAME,
                RENOVATION_OUTPUT_FILENAME,
                WEIGHTED_OUTPUT_FILENAME,
            ]
            if multiple_scenarios:
                output_names.append(MODEL_SCENARIO_OUTPUT_FILENAME)
            output_artifacts = {
                name: _output_ledger(staging / name) for name in output_names
            }
            postprocess_summary = {
                "status": "PASS",
                "postprocess_contract_version": POSTPROCESS_CONTRACT_VERSION,
                "scope": (
                    "authenticated bounded-memory run-level analysis of completed "
                    "partition diagnostics"
                ),
                "source_execution_status": str(source_summary.get("status")),
                "source_design_sha256": str(design["design_sha256"]),
                "source_design_file_sha256": source_design_sha256,
                "source_monte_carlo_summary_sha256": source_summary_sha256,
                "source_partition_index_sha256": source_partition_index_sha256,
                "source_partition_input_ledger_sha256": canonical_sha256(
                    input_ledger_records
                ),
                "source_partition_count": len(partition_index),
                "processed_run_count": total_processed_rows,
                "expected_run_count": int(design["expected_run_count"]),
                "stock_weights_sha256": str(weights["stock_weights_sha256"].iloc[0]),
                "stock_weights_source_sha256": str(
                    weights["stock_weights_source_sha256"].iloc[0]
                ),
                "analysis_group_count": len(analysis_paths),
                "weighted_analysis_group_count": len(weighted_paths),
                "maximum_in_memory_analysis_group_rows": max_analysis_group_rows,
                "maximum_in_memory_weighted_group_rows": max_weighted_group_rows,
                "maximum_in_memory_paired_renovation_group_rows": (
                    max_paired_group_rows
                ),
                "maximum_in_memory_paired_model_scenario_group_rows": (
                    max_model_scenario_group_rows
                ),
                "diagnostics_chunk_rows": chunk_rows,
                "dwelling_hour_files_read": 0,
                "paired_model_scenario_output": (
                    MODEL_SCENARIO_OUTPUT_FILENAME if multiple_scenarios else None
                ),
                "output_artifacts": output_artifacts,
                "interval_interpretation": (
                    "empirical intervals over only the included weather, occupant, stock, "
                    "and declared model-scenario axes; not complete prediction intervals"
                ),
            }
            for name in output_names:
                (staging / name).replace(root / name)
            _atomic_json(postprocess_summary, root / POSTPROCESS_SUMMARY_FILENAME)
            return postprocess_summary


def postprocessing_status(production_dir: str | Path) -> dict[str, Any]:
    """Return a read-only status for the committed post-processing artifact set."""

    root = Path(production_dir).resolve()
    summary_path = root / POSTPROCESS_SUMMARY_FILENAME
    if not summary_path.is_file():
        return {
            "status": "NOT_RUN",
            "production_dir": str(root),
            "summary_path": str(summary_path),
        }
    try:
        summary = _read_json(summary_path, label="post-processing summary")
        if summary.get("status") != "PASS" or summary.get(
            "postprocess_contract_version"
        ) != POSTPROCESS_CONTRACT_VERSION:
            raise MonteCarloContractError("Post-processing summary status/contract is invalid.")
        _verify_file(
            root / "streaming_design_contract.json",
            summary.get("source_design_file_sha256"),
            label="post-processing source design",
        )
        _verify_file(
            root / "monte_carlo_summary.json",
            summary.get("source_monte_carlo_summary_sha256"),
            label="post-processing source Monte Carlo summary",
        )
        _verify_file(
            root / "partition_index.csv",
            summary.get("source_partition_index_sha256"),
            label="post-processing source partition index",
        )
        artifacts = summary.get("output_artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise MonteCarloContractError("Post-processing output ledger is missing.")
        for filename, metadata in artifacts.items():
            if not isinstance(metadata, dict):
                raise MonteCarloContractError("Post-processing artifact metadata is invalid.")
            _verify_file(
                root / str(filename),
                metadata.get("sha256"),
                label=f"post-processing output {filename}",
            )
        return {
            "status": "PASS",
            "production_dir": str(root),
            "source_design_sha256": summary.get("source_design_sha256"),
            "processed_run_count": summary.get("processed_run_count"),
            "source_partition_count": summary.get("source_partition_count"),
            "output_artifacts": artifacts,
            "validation_scope": (
                "committed outputs and source root commit hashes; a full rerun also "
                "reauthenticates every partition ledger, manifest, and diagnostics file"
            ),
        }
    except MonteCarloContractError as exc:
        return {
            "status": "INVALID",
            "production_dir": str(root),
            "reason": str(exc),
        }


__all__ = [
    "POSTPROCESS_CONTRACT_VERSION",
    "postprocess_production_results",
    "postprocessing_status",
]
