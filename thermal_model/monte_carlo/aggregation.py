"""2050 stock weighting and diversified coincident-load aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import MonteCarloContractError, MonteCarloResult, diagnostics_to_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STOCK_WEIGHTS_PATH = (
    PROJECT_ROOT
    / "BE_building_stock/data/scenarios/renovation/"
    "archetype_matrix_2050_renovation_scenarios.csv"
)
MODELLED_BELGIAN_STOCK_DWELLINGS = 5_537_385.0
EXCLUDED_R5_R6_DWELLINGS = 290_438.0
EXCLUDED_R5_R6_SHARE = 0.049836
AUTHORITATIVE_STOCK_CONTENT_SHA256 = (
    "6aaedbdacd83f058dcb0aaf3026cf1ffae1412140fd5acdb609bd0f027495659"
)
AUTHORITATIVE_STOCK_SOURCE_SHA256 = (
    "710a8d6d9d250bb487349638f7e14dff681a9856a083d6f66ac1f239f9475594"
)
STOCK_REQUIRED_COLUMNS = {
    "scenario",
    "target_year",
    "region",
    "archetype_id",
    "dwelling_type",
    "construction_period",
    "state_id",
    "renovation_state",
    "state_dwellings",
    "state_dwellings_2050",
    "state_share_within_region_2050",
    "regional_number_of_dwellings",
    "regional_modelled_stock_dwellings",
}
STOCK_KEY_COLUMNS = ("scenario", "region", "archetype_id", "state_id")
STOCK_PROVENANCE_COLUMNS = (
    "model_contract_version",
    "central_thermal_assumptions_sha256",
    "effective_thermal_assumptions_sha256",
    "behaviour_assumptions_sha256",
    "occupant_distribution_sha256",
    "model_scenario_sha256",
    "member_sha256",
    "metadata_sha256",
    "climate_manifest_sha256",
    "morph_contract_sha256",
    "facade_source_sha256_json",
    "weather_contract_sha256",
    "weather_forcing_sha256",
)
STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION = (
    "gate5_stock_partition_provenance_v1"
)
STOCK_PARTITION_IDENTITY_FIELDS = (
    "climate_scenario_id",
    "weather_member_id",
    "weather_pair_id",
    "observed_pvgis_year",
    "climate_target",
    "model_scenario_id",
    "model_scenario_axis",
)
STOCK_PARTITION_SEED_PROVENANCE_FIELDS = (
    "occupant_seed_count",
    "occupant_seeds_json",
    "occupant_seed_bank_sha256",
)
STOCK_PARTITION_STOCK_PROVENANCE_FIELDS = (
    "stock_scenario_id",
    "target_year",
    "stock_weights_sha256",
    "stock_weights_source_sha256",
)
STOCK_PARTITION_PHYSICS_PROVENANCE_FIELDS = (
    "archetype_state_provenance_json",
    "archetype_state_provenance_sha256",
)
STOCK_PARTITION_VERBOSE_PROVENANCE_FIELDS = (
    *STOCK_PARTITION_IDENTITY_FIELDS,
    *STOCK_PROVENANCE_COLUMNS,
    *STOCK_PARTITION_PHYSICS_PROVENANCE_FIELDS,
    *STOCK_PARTITION_SEED_PROVENANCE_FIELDS,
    *STOCK_PARTITION_STOCK_PROVENANCE_FIELDS,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> str:
    digest = str(value)
    if len(digest) != 64:
        raise MonteCarloContractError(f"{name} must be a 64-character SHA-256 digest.")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise MonteCarloContractError(f"{name} must be hexadecimal.") from exc
    return digest


def _stock_content_sha256(frame: pd.DataFrame) -> str:
    """Hash normalized stock-contract values, independent of CSV byte formatting."""

    ordered_columns = sorted(STOCK_REQUIRED_COLUMNS)
    canonical = frame.loc[:, ordered_columns].sort_values(
        list(STOCK_KEY_COLUMNS), kind="stable"
    )
    records: list[dict[str, object]] = []
    numeric = {
        "target_year",
        "state_dwellings",
        "state_dwellings_2050",
        "state_share_within_region_2050",
        "regional_number_of_dwellings",
        "regional_modelled_stock_dwellings",
    }
    for row in canonical.itertuples(index=False, name=None):
        record: dict[str, object] = {}
        for column, value in zip(ordered_columns, row):
            if column == "target_year":
                record[column] = int(value)
            elif column in numeric:
                record[column] = float(value)
            else:
                record[column] = str(value)
        records.append(record)
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_stock_weights(
    frame: pd.DataFrame,
    *,
    source_sha256: str | None = None,
    require_authoritative_shape: bool = True,
) -> pd.DataFrame:
    """Validate the exact regional/archetype/renovation weight contract."""

    missing = sorted(STOCK_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise MonteCarloContractError(f"Stock weights are missing columns: {missing}.")
    existing_content_hashes: set[str] = set()
    if "stock_weights_sha256" in frame.columns:
        existing_content_hashes = set(frame["stock_weights_sha256"].dropna().astype(str))
        if len(existing_content_hashes) != 1:
            raise MonteCarloContractError(
                "Stock weights must carry one consistent content checksum."
            )
    existing_source_hashes: set[str] = set()
    if "stock_weights_source_sha256" in frame.columns:
        existing_source_hashes = {
            value
            for value in frame["stock_weights_source_sha256"].dropna().astype(str)
            if value.strip()
        }
        if len(existing_source_hashes) > 1:
            raise MonteCarloContractError(
                "Stock weights must carry one consistent source-file checksum."
            )

    result = frame.loc[:, list(STOCK_REQUIRED_COLUMNS)].copy(deep=True)
    key = list(STOCK_KEY_COLUMNS)
    if result.duplicated(key).any():
        raise MonteCarloContractError("Stock-weight keys must be unique.")
    if set(result["scenario"].astype(str)) != {"central"}:
        raise MonteCarloContractError("Only the declared central stock scenario is available.")
    target_year = pd.to_numeric(result["target_year"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(target_year).all() or not np.equal(target_year, 2050.0).all():
        raise MonteCarloContractError("Stock weights must target 2050.")
    result["target_year"] = target_year.astype(int)
    if not result["renovation_state"].astype(str).equals(result["state_id"].astype(str)):
        raise MonteCarloContractError("renovation_state must remain an exact state_id alias.")
    numeric_columns = [
        "state_dwellings",
        "state_dwellings_2050",
        "state_share_within_region_2050",
        "regional_number_of_dwellings",
        "regional_modelled_stock_dwellings",
    ]
    result[numeric_columns] = result[numeric_columns].apply(pd.to_numeric, errors="raise")
    values = result[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise MonteCarloContractError("Stock counts/shares must be finite and non-negative.")
    if (result["regional_modelled_stock_dwellings"] <= 0.0).any():
        raise MonteCarloContractError(
            "Every stock-weight row must reference a positive regional stock total."
        )
    if not np.allclose(
        result["state_dwellings"],
        result["state_dwellings_2050"],
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise MonteCarloContractError("state_dwellings alias differs from state_dwellings_2050.")
    expected_shares = (
        result["state_dwellings_2050"]
        / result["regional_modelled_stock_dwellings"]
    )
    if not np.allclose(
        result["state_share_within_region_2050"],
        expected_shares,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise MonteCarloContractError(
            "Regional state shares do not reconcile with dwelling counts."
        )

    reconstructed_archetypes = result.groupby(
        ["region", "archetype_id"], sort=False
    )["state_dwellings_2050"].sum()
    reported_archetypes = result.groupby(
        ["region", "archetype_id"], sort=False
    )["regional_number_of_dwellings"].first()
    if not np.allclose(
        reconstructed_archetypes,
        reported_archetypes,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise MonteCarloContractError("State weights do not reconstruct regional archetypes.")
    regional = reconstructed_archetypes.groupby("region").sum()
    reported_regional = result.groupby("region")[
        "regional_modelled_stock_dwellings"
    ].first()
    if not np.allclose(regional, reported_regional, rtol=0.0, atol=2.0e-6):
        raise MonteCarloContractError("Archetype weights do not reconstruct regional totals.")
    if require_authoritative_shape:
        if len(result) != 225:
            raise MonteCarloContractError("Authoritative stock contract must contain 225 rows.")
        if result[["archetype_id", "state_id"]].drop_duplicates().shape[0] != 75:
            raise MonteCarloContractError("Authoritative stock contract must contain 75 physics cells.")
        if len(set(result["region"])) != 3:
            raise MonteCarloContractError("Authoritative stock contract must contain three regions.")
        if not np.isclose(
            result["state_dwellings_2050"].sum(),
            MODELLED_BELGIAN_STOCK_DWELLINGS,
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise MonteCarloContractError("Modelled Belgian stock total is not 5,537,385.")
    content_sha256 = _stock_content_sha256(result)
    if existing_content_hashes:
        existing = _validate_sha256(
            next(iter(existing_content_hashes)), "stock_weights_sha256"
        )
        if existing != content_sha256:
            raise MonteCarloContractError(
                "Stock-weight content differs from its stored checksum."
            )
    if source_sha256 is not None:
        source_digest = _validate_sha256(source_sha256, "source_sha256")
        if existing_source_hashes and next(iter(existing_source_hashes)) != source_digest:
            raise MonteCarloContractError(
                "Stock-weight source checksum differs from the supplied source checksum."
            )
    elif existing_source_hashes:
        source_digest = _validate_sha256(
            next(iter(existing_source_hashes)), "stock_weights_source_sha256"
        )
    else:
        source_digest = ""
    if require_authoritative_shape and (
        content_sha256 != AUTHORITATIVE_STOCK_CONTENT_SHA256
        or source_digest != AUTHORITATIVE_STOCK_SOURCE_SHA256
    ):
        raise MonteCarloContractError(
            "Authoritative full-stock execution requires the pinned 2050 stock "
            "content and source-file checksums."
        )
    result["stock_scenario_id"] = result["scenario"].astype(str)
    result["stock_weights_sha256"] = content_sha256
    result["stock_weights_source_sha256"] = source_digest
    return result


def load_stock_weights(
    path: str | Path = DEFAULT_STOCK_WEIGHTS_PATH,
) -> pd.DataFrame:
    """Load the authoritative deterministic 2050 stock weights."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Stock-weight matrix does not exist: {resolved}")
    return validate_stock_weights(
        pd.read_csv(resolved),
        source_sha256=_sha256_file(resolved),
        require_authoritative_shape=True,
    )


def _validated_stock_input(
    stock_weights: pd.DataFrame | None,
    *,
    require_full_stock: bool,
) -> pd.DataFrame:
    """Always revalidate caller-owned weights and verify any stored checksum."""

    supplied = load_stock_weights() if stock_weights is None else stock_weights
    return validate_stock_weights(
        supplied,
        require_authoritative_shape=require_full_stock,
    )


def _group_provenance(group: pd.DataFrame) -> dict[str, object]:
    """Return constant run provenance or reject a physically mixed stock group."""

    missing = sorted(set(STOCK_PROVENANCE_COLUMNS).difference(group.columns))
    if missing:
        raise MonteCarloContractError(
            f"Stock results are missing provenance columns: {missing}."
        )
    result: dict[str, object] = {}
    for column in STOCK_PROVENANCE_COLUMNS:
        values = group[column].astype(str).unique()
        if len(values) != 1:
            raise MonteCarloContractError(
                f"Stock group mixes incompatible {column} values."
            )
        result[column] = values[0]
    return result


def _cell_physics_provenance(group: pd.DataFrame) -> dict[str, str]:
    if "archetype_state_sha256" not in group.columns:
        raise MonteCarloContractError(
            "Stock results are missing archetype_state_sha256 provenance."
        )
    mixed = group.groupby(["archetype_id", "state_id"], sort=False)[
        "archetype_state_sha256"
    ].nunique()
    if (mixed != 1).any():
        cells = [tuple(value) for value in mixed.index[mixed != 1].tolist()]
        raise MonteCarloContractError(
            f"Stock cells mix incompatible archetype physics checksums: {cells}."
        )
    mapping = (
        group[["archetype_id", "state_id", "archetype_state_sha256"]]
        .drop_duplicates()
        .sort_values(["archetype_id", "state_id"], kind="stable")
        .astype(str)
        .to_dict("records")
    )
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return {
        "archetype_state_provenance_json": encoded,
        "archetype_state_provenance_sha256": hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest(),
    }


def _validated_ordered_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    raw = tuple(seeds)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise MonteCarloContractError(
            "Stock occupant seeds must be integers, not coerced values."
        )
    ordered = tuple(int(value) for value in raw)
    if (
        not ordered
        or len(set(ordered)) != len(ordered)
        or any(value < 0 or value > 2**32 - 1 for value in ordered)
    ):
        raise MonteCarloContractError(
            "Stock aggregation needs a non-empty ordered set of unique uint32 seeds."
        )
    return ordered


def _seed_provenance(seeds: Sequence[int]) -> dict[str, object]:
    ordered = _validated_ordered_seeds(seeds)
    encoded = json.dumps(list(ordered), separators=(",", ":"))
    return {
        "occupant_seed_count": len(ordered),
        "occupant_seeds_json": encoded,
        "occupant_seed_bank_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def _stock_partition_provenance_record(
    *parts: Mapping[str, object],
) -> dict[str, str]:
    """Return the versioned digest that links compact and verbose stock rows.

    Only the declared verbose fields enter the digest.  This makes the compact
    hourly reference reproducible from an annual/provenance row while keeping
    annual metrics and other derived values outside the provenance identity.
    """

    merged: dict[str, object] = {}
    for part in parts:
        for key, value in part.items():
            name = str(key)
            if name in merged and str(merged[name]) != str(value):
                raise MonteCarloContractError(
                    f"Conflicting {name} values in stock-partition provenance."
                )
            merged[name] = value
    missing = sorted(set(STOCK_PARTITION_VERBOSE_PROVENANCE_FIELDS).difference(merged))
    if missing:
        raise MonteCarloContractError(
            f"Stock-partition provenance is missing fields: {missing}."
        )
    normalized: dict[str, str] = {}
    for field in STOCK_PARTITION_VERBOSE_PROVENANCE_FIELDS:
        value = merged[field]
        if pd.isna(value):
            raise MonteCarloContractError(
                f"Stock-partition provenance field {field} must not be null."
            )
        normalized[field] = str(value)
    payload = {
        "contract_version": STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION,
        "provenance": normalized,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "stock_partition_provenance_contract_version": (
            STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION
        ),
        "stock_partition_provenance_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _stock_partition_provenance_sha256(
    *parts: Mapping[str, object],
) -> str:
    """Return the digest from the versioned compact-provenance contract."""

    return _stock_partition_provenance_record(*parts)[
        "stock_partition_provenance_sha256"
    ]


def _validate_stock_partition_provenance_record(
    record: Mapping[str, object],
) -> None:
    """Recompute a compact reference from the verbose row and reject drift."""

    version = str(record.get("stock_partition_provenance_contract_version", ""))
    if version != STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION:
        raise MonteCarloContractError(
            "Unsupported stock-partition provenance contract version "
            f"{version!r}; expected {STOCK_PARTITION_PROVENANCE_CONTRACT_VERSION!r}."
        )
    declared = _validate_sha256(
        str(record.get("stock_partition_provenance_sha256", "")),
        "stock_partition_provenance_sha256",
    )
    expected = _stock_partition_provenance_record(record)[
        "stock_partition_provenance_sha256"
    ]
    if declared != expected:
        raise MonteCarloContractError(
            "Stock-partition provenance checksum does not match its verbose record."
        )


def stock_distribution_summary(stock_aggregation: pd.DataFrame) -> pd.DataFrame:
    """Summarize empirical stock outcomes across weather members, keeping RCPs apart."""

    group_columns = [
        "climate_scenario_id",
        "model_scenario_id",
        "model_scenario_axis",
        "stock_scenario_id",
        "target_year",
        "region",
    ]
    metrics = (
        "annual_heating_GWh",
        "annual_potential_sensible_cooling_GWh",
        "coincident_peak_heating_MW",
        "coincident_peak_potential_cooling_MW",
    )
    required = {
        *group_columns,
        *metrics,
        "weather_member_id",
        "occupant_seed_count",
        "occupant_seeds_json",
        "occupant_seed_bank_sha256",
        "stock_weights_sha256",
        "stock_weights_source_sha256",
        "stock_partition_provenance_contract_version",
        "stock_partition_provenance_sha256",
        "archetype_state_provenance_json",
        "archetype_state_provenance_sha256",
        *STOCK_PROVENANCE_COLUMNS,
    }
    missing = sorted(required.difference(stock_aggregation.columns))
    if missing:
        raise MonteCarloContractError(
            f"Stock-distribution input is missing columns: {missing}."
        )
    for row_index, row in stock_aggregation.iterrows():
        try:
            _validate_stock_partition_provenance_record(row.to_dict())
        except MonteCarloContractError as exc:
            raise MonteCarloContractError(
                f"Invalid stock-partition provenance at row {row_index}: {exc}"
            ) from exc
    records: list[dict[str, object]] = []
    member_columns = (
        "weather_member_id",
        "weather_pair_id",
        "observed_pvgis_year",
        "climate_target",
        "member_sha256",
        "metadata_sha256",
        "facade_source_sha256_json",
        "weather_contract_sha256",
        "weather_forcing_sha256",
        "stock_partition_provenance_sha256",
    )
    constant_provenance = tuple(
        column
        for column in STOCK_PROVENANCE_COLUMNS
        if column not in member_columns
    ) + (
        "occupant_seed_count",
        "occupant_seeds_json",
        "occupant_seed_bank_sha256",
        "stock_weights_sha256",
        "stock_weights_source_sha256",
        "stock_partition_provenance_contract_version",
        "archetype_state_provenance_json",
        "archetype_state_provenance_sha256",
    )
    for key, group in stock_aggregation.groupby(group_columns, sort=True, dropna=False):
        identity = dict(zip(group_columns, key))
        if group["weather_member_id"].duplicated().any():
            raise MonteCarloContractError(
                f"Stock-distribution group {identity} has duplicate weather members."
            )
        constants: dict[str, object] = {}
        for column in constant_provenance:
            values = group[column].astype(str).unique()
            if len(values) != 1:
                raise MonteCarloContractError(
                    f"Stock-distribution group {identity} mixes {column}."
                )
            constants[column] = group[column].iloc[0]
        member_records = (
            group.loc[:, list(member_columns)]
            .astype(str)
            .sort_values("weather_member_id", kind="stable")
            .to_dict("records")
        )
        member_json = json.dumps(
            member_records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise MonteCarloContractError(
                    f"Stock-distribution metric {metric} contains non-finite values."
                )
            records.append(
                {
                    **identity,
                    **constants,
                    "weather_member_count": len(values),
                    "weather_member_provenance_json": member_json,
                    "weather_member_provenance_sha256": hashlib.sha256(
                        member_json.encode("utf-8")
                    ).hexdigest(),
                    "metric": metric,
                    "minimum": float(np.min(values)),
                    "p05": float(np.quantile(values, 0.05)),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "maximum": float(np.max(values)),
                    "standard_deviation": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    ),
                    "interval_interpretation": (
                        "descriptive empirical interval over included paired weather members; "
                        "not a complete prediction interval"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def stock_contribution_summary(
    results: Sequence[MonteCarloResult],
    *,
    stock_weights: pd.DataFrame | None = None,
    require_full_stock: bool = True,
    occupant_seed_order: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Return weighted annual contributions by region, type, period and state."""

    if not results:
        raise MonteCarloContractError("Stock contributions require simulation results.")
    diagnostics = pd.DataFrame.from_records(
        [diagnostics_to_record(result.diagnostics) for result in results]
    )
    if diagnostics["run_id"].duplicated().any():
        raise MonteCarloContractError("Stock contributions received duplicate run IDs.")
    weights = _validated_stock_input(
        stock_weights, require_full_stock=require_full_stock
    )
    physics_keys = ["archetype_id", "state_id"]
    expected_cells = set(map(tuple, weights[physics_keys].drop_duplicates().to_numpy()))
    result_cells = set(map(tuple, diagnostics[physics_keys].drop_duplicates().to_numpy()))
    if require_full_stock and result_cells != expected_cells:
        raise MonteCarloContractError(
            "Stock contribution reporting requires all 75 physical cells."
        )
    if not result_cells.issubset(expected_cells):
        raise MonteCarloContractError("Results contain cells absent from stock weights.")
    active_weights = weights.merge(
        diagnostics[physics_keys].drop_duplicates(), on=physics_keys, how="inner"
    )
    axes = list(STOCK_PARTITION_IDENTITY_FIELDS)
    records: list[dict] = []
    for top_key, group in diagnostics.groupby(axes, sort=True):
        identity = dict(zip(axes, top_key))
        provenance = _group_provenance(group)
        cell_provenance = _cell_physics_provenance(group)
        if group.duplicated([*physics_keys, "occupant_seed"]).any():
            raise MonteCarloContractError(
                f"Contribution group {identity} has duplicate physical-cell/seed runs."
            )
        group_cells = set(map(tuple, group[physics_keys].drop_duplicates().to_numpy()))
        if require_full_stock and group_cells != expected_cells:
            raise MonteCarloContractError(
                f"Contribution group {identity} does not contain all 75 physical cells."
            )
        seed_sets = group.groupby(physics_keys)["occupant_seed"].agg(
            lambda values: tuple(sorted(int(value) for value in values))
        )
        distinct_seed_sets = set(seed_sets.tolist())
        if len(distinct_seed_sets) != 1:
            raise MonteCarloContractError(
                "All contribution cells must use the identical occupant-seed set."
            )
        observed_seeds = next(iter(distinct_seed_sets))
        ordered_seeds = (
            _validated_ordered_seeds(occupant_seed_order)
            if occupant_seed_order is not None
            else observed_seeds
        )
        if set(ordered_seeds) != set(observed_seeds) or len(ordered_seeds) != len(
            observed_seeds
        ):
            raise MonteCarloContractError(
                "Declared occupant-seed order differs from the stock-result seed set."
            )
        seed_provenance = _seed_provenance(ordered_seeds)
        cell_means = group.groupby(physics_keys, as_index=False).agg(
            mean_annual_heating_kWh=("annual_heating_kWh", "mean"),
            mean_annual_cooling_kWh=("annual_cooling_kWh", "mean"),
            floor_area_m2=("floor_area_m2", "first"),
            occupant_seed_count=("occupant_seed", "nunique"),
        )
        seed_counts = set(cell_means["occupant_seed_count"].astype(int))
        if len(seed_counts) != 1:
            raise MonteCarloContractError(
                "Stock contribution cells must use one common seed count."
            )
        joined = active_weights.merge(cell_means, on=physics_keys, how="left")
        positive_missing = joined["mean_annual_heating_kWh"].isna() & (
            joined["state_dwellings_2050"] > 0.0
        )
        if positive_missing.any():
            missing = joined.loc[positive_missing, physics_keys].drop_duplicates()
            raise MonteCarloContractError(
                f"Missing positive-weight contribution cells: {missing.to_dict('records')}."
            )
        joined = joined.dropna(subset=["mean_annual_heating_kWh"]).copy()
        joined["weighted_heating_kWh"] = (
            joined["state_dwellings_2050"] * joined["mean_annual_heating_kWh"]
        )
        joined["weighted_cooling_kWh"] = (
            joined["state_dwellings_2050"] * joined["mean_annual_cooling_kWh"]
        )
        joined["weighted_floor_area_m2"] = (
            joined["state_dwellings_2050"] * joined["floor_area_m2"]
        )
        total_heating = float(joined["weighted_heating_kWh"].sum())
        total_cooling = float(joined["weighted_cooling_kWh"].sum())
        source_hashes = set(joined["stock_weights_sha256"].astype(str))
        source_file_hashes = set(joined["stock_weights_source_sha256"].astype(str))
        if len(source_hashes) != 1 or len(source_file_hashes) != 1:
            raise MonteCarloContractError("Contribution weights carry inconsistent checksums.")
        stock_provenance = {
            "stock_scenario_id": "central",
            "target_year": 2050,
            "stock_weights_sha256": next(iter(source_hashes)),
            "stock_weights_source_sha256": next(iter(source_file_hashes)),
        }
        partition_provenance = _stock_partition_provenance_record(
            identity,
            provenance,
            cell_provenance,
            seed_provenance,
            stock_provenance,
        )
        for dimension in ("region", "dwelling_type", "construction_period", "state_id"):
            for value, contribution in joined.groupby(dimension, sort=True, dropna=False):
                heating = float(contribution["weighted_heating_kWh"].sum())
                cooling = float(contribution["weighted_cooling_kWh"].sum())
                floor_area = float(contribution["weighted_floor_area_m2"].sum())
                records.append(
                    {
                        **identity,
                        **provenance,
                        **cell_provenance,
                        **seed_provenance,
                        **stock_provenance,
                        **partition_provenance,
                        "contribution_dimension": dimension,
                        "contribution_value": str(value),
                        "modelled_dwellings": float(
                            contribution["state_dwellings_2050"].sum()
                        ),
                        "stock_floor_area_m2": floor_area,
                        "annual_heating_GWh": heating / 1.0e6,
                        "annual_potential_sensible_cooling_GWh": cooling / 1.0e6,
                        "heating_intensity_kWh_m2": (
                            heating / floor_area if floor_area > 0.0 else 0.0
                        ),
                        "potential_cooling_intensity_kWh_m2": (
                            cooling / floor_area if floor_area > 0.0 else 0.0
                        ),
                        "share_of_stock_heating": (
                            heating / total_heating if total_heating > 0.0 else 0.0
                        ),
                        "share_of_stock_potential_cooling": (
                            cooling / total_cooling if total_cooling > 0.0 else 0.0
                        ),
                        "cooling_interpretation": (
                            "potential useful sensible load under universal ideal 26C control"
                        ),
                    }
                )
    result = pd.DataFrame.from_records(records)
    for _, group in result.groupby([*axes, "contribution_dimension"]):
        if group["annual_heating_GWh"].sum() > 0.0 and not np.isclose(
            group["share_of_stock_heating"].sum(), 1.0, rtol=0.0, atol=1.0e-10
        ):
            raise MonteCarloContractError("Stock heating contribution shares do not sum to one.")
        if group["annual_potential_sensible_cooling_GWh"].sum() > 0.0 and not np.isclose(
            group["share_of_stock_potential_cooling"].sum(),
            1.0,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise MonteCarloContractError("Stock cooling contribution shares do not sum to one.")
    return result


def aggregate_stock_results(
    results: Sequence[MonteCarloResult],
    *,
    stock_weights: pd.DataFrame | None = None,
    require_full_stock: bool = True,
    occupant_seed_order: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate diversified seed-mean profiles to regional and national loads.

    Behaviour is averaged within each physical stock cell before multiplying by
    dwelling counts.  This avoids imposing one household realization on every
    dwelling.  Weather remains common within a member, as required for a
    physically meaningful coincident peak.
    """

    if not results:
        raise MonteCarloContractError("Stock aggregation requires simulation results.")
    diagnostics = pd.DataFrame.from_records(
        [diagnostics_to_record(result.diagnostics) for result in results]
    )
    if diagnostics["run_id"].duplicated().any():
        raise MonteCarloContractError("Stock aggregation received duplicate run IDs.")
    hourly_by_run = {result.diagnostics.run_id: result.hourly for result in results}
    weights = _validated_stock_input(
        stock_weights, require_full_stock=require_full_stock
    )
    physics_keys = ["archetype_id", "state_id"]
    expected_cells = set(map(tuple, weights[physics_keys].drop_duplicates().to_numpy()))
    result_cells = set(map(tuple, diagnostics[physics_keys].drop_duplicates().to_numpy()))
    if require_full_stock and result_cells != expected_cells:
        missing = sorted(expected_cells - result_cells)
        extra = sorted(result_cells - expected_cells)
        raise MonteCarloContractError(
            f"Stock aggregation needs all weighted physics cells; missing={missing}, extra={extra}."
        )
    if not result_cells.issubset(expected_cells):
        raise MonteCarloContractError("Results contain cells absent from the stock matrix.")
    active_weights = weights.merge(
        diagnostics[physics_keys].drop_duplicates(), on=physics_keys, how="inner"
    )

    top_axes = list(STOCK_PARTITION_IDENTITY_FIELDS)
    summary_records: list[dict] = []
    hourly_records: list[pd.DataFrame] = []
    for top_key, top_group in diagnostics.groupby(top_axes, sort=True):
        top_identity = dict(zip(top_axes, top_key))
        provenance = _group_provenance(top_group)
        cell_provenance = _cell_physics_provenance(top_group)
        cell_profiles: dict[tuple[str, str], dict] = {}
        common_seed_set: tuple[int, ...] | None = None
        reference_timestamps: pd.DatetimeIndex | None = None
        for cell_key, cell_group in top_group.groupby(physics_keys, sort=True):
            if cell_group["floor_area_m2"].nunique() != 1:
                raise MonteCarloContractError(f"Floor area differs within cell {cell_key}.")
            if cell_group["occupant_seed"].duplicated().any():
                raise MonteCarloContractError(
                    f"Stock cell {cell_key} contains duplicate occupant seeds."
                )
            seeds = tuple(sorted(int(value) for value in cell_group["occupant_seed"]))
            if common_seed_set is None:
                common_seed_set = seeds
            elif seeds != common_seed_set:
                raise MonteCarloContractError(
                    "All stock cells must use the same occupant-seed set."
                )
            heating_arrays: list[np.ndarray] = []
            cooling_arrays: list[np.ndarray] = []
            for row in cell_group.sort_values("occupant_seed").itertuples(index=False):
                hourly = hourly_by_run[row.run_id]
                timestamps = pd.DatetimeIndex(hourly["timestamp_utc"])
                if reference_timestamps is None:
                    reference_timestamps = timestamps
                elif not timestamps.equals(reference_timestamps):
                    raise MonteCarloContractError(
                        "Hourly stock results are not timestamp-aligned within a weather member."
                    )
                heating_arrays.append(hourly["heating_demand_W"].to_numpy(dtype=float))
                cooling_arrays.append(hourly["cooling_demand_W"].to_numpy(dtype=float))
                heat = heating_arrays[-1]
                cool = cooling_arrays[-1]
                if (
                    not np.isfinite(heat).all()
                    or not np.isfinite(cool).all()
                    or (heat < 0.0).any()
                    or (cool < 0.0).any()
                ):
                    raise MonteCarloContractError(
                        f"Stock run {row.run_id} contains invalid heating/cooling power."
                    )
                expected_metrics = {
                    "annual_heating_kWh": float(heat.sum()) / 1000.0,
                    "annual_cooling_kWh": float(cool.sum()) / 1000.0,
                    "peak_heating_W": float(heat.max()),
                    "peak_cooling_W": float(cool.max()),
                }
                for metric, expected in expected_metrics.items():
                    if not np.isclose(
                        float(getattr(row, metric)),
                        expected,
                        rtol=1.0e-9,
                        atol=1.0e-6,
                    ):
                        raise MonteCarloContractError(
                            f"Stock run {row.run_id} {metric} does not reconcile with hourly data."
                        )
            cell_profiles[cell_key] = {
                "heating_W": np.mean(np.vstack(heating_arrays), axis=0),
                "cooling_W": np.mean(np.vstack(cooling_arrays), axis=0),
                "mean_heating_kWh": float(cell_group["annual_heating_kWh"].mean()),
                "mean_cooling_kWh": float(cell_group["annual_cooling_kWh"].mean()),
                "mean_peak_heating_W": float(cell_group["peak_heating_W"].mean()),
                "mean_peak_cooling_W": float(cell_group["peak_cooling_W"].mean()),
                "floor_area_m2": float(cell_group["floor_area_m2"].iloc[0]),
            }
        if reference_timestamps is None or common_seed_set is None:
            raise MonteCarloContractError("Stock aggregation group contains no hourly runs.")
        ordered_seeds = (
            _validated_ordered_seeds(occupant_seed_order)
            if occupant_seed_order is not None
            else common_seed_set
        )
        if set(ordered_seeds) != set(common_seed_set) or len(ordered_seeds) != len(
            common_seed_set
        ):
            raise MonteCarloContractError(
                "Declared occupant-seed order differs from the stock-result seed set."
            )
        seed_provenance = _seed_provenance(ordered_seeds)
        if require_full_stock and set(cell_profiles) != expected_cells:
            raise MonteCarloContractError(
                f"Stock group {top_identity} does not contain all 75 physical cells."
            )
        stock_content_hashes = set(active_weights["stock_weights_sha256"].astype(str))
        stock_source_hashes = set(
            active_weights["stock_weights_source_sha256"].astype(str)
        )
        if len(stock_content_hashes) != 1 or len(stock_source_hashes) != 1:
            raise MonteCarloContractError(
                "Stock weights carry inconsistent content/source checksums."
            )
        stock_provenance = {
            "stock_scenario_id": "central",
            "target_year": 2050,
            "stock_weights_sha256": next(iter(stock_content_hashes)),
            "stock_weights_source_sha256": next(iter(stock_source_hashes)),
        }
        partition_provenance = _stock_partition_provenance_record(
            top_identity,
            provenance,
            cell_provenance,
            seed_provenance,
            stock_provenance,
        )

        region_names: list[str] = sorted(str(value) for value in active_weights["region"].unique())
        regional_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for region in [*region_names, "Belgium_modelled_stock"]:
            selected_weights = (
                active_weights.loc[active_weights["region"] == region]
                if region != "Belgium_modelled_stock"
                else active_weights
            )
            heating_stock_W = np.zeros(len(reference_timestamps), dtype=float)
            cooling_stock_W = np.zeros(len(reference_timestamps), dtype=float)
            weighted_heating_kWh = 0.0
            weighted_cooling_kWh = 0.0
            sum_peak_heating_W = 0.0
            sum_peak_cooling_W = 0.0
            stock_floor_area_m2 = 0.0
            total_dwellings = 0.0
            positive_cells = 0
            for weight_row in selected_weights.itertuples(index=False):
                cell_key = (weight_row.archetype_id, weight_row.state_id)
                profile = cell_profiles.get(cell_key)
                if profile is None:
                    if float(weight_row.state_dwellings_2050) > 0.0:
                        raise MonteCarloContractError(
                            f"Missing positive-weight stock cell {cell_key}."
                        )
                    continue
                count = float(weight_row.state_dwellings_2050)
                positive_cells += int(count > 0.0)
                total_dwellings += count
                stock_floor_area_m2 += count * profile["floor_area_m2"]
                heating_stock_W += count * profile["heating_W"]
                cooling_stock_W += count * profile["cooling_W"]
                weighted_heating_kWh += count * profile["mean_heating_kWh"]
                weighted_cooling_kWh += count * profile["mean_cooling_kWh"]
                sum_peak_heating_W += count * profile["mean_peak_heating_W"]
                sum_peak_cooling_W += count * profile["mean_peak_cooling_W"]
            annual_heating_GWh = float(heating_stock_W.sum()) / 1.0e9
            annual_cooling_GWh = float(cooling_stock_W.sum()) / 1.0e9
            if not np.isclose(
                annual_heating_GWh,
                weighted_heating_kWh / 1.0e6,
                rtol=1.0e-10,
                atol=1.0e-8,
            ) or not np.isclose(
                annual_cooling_GWh,
                weighted_cooling_kWh / 1.0e6,
                rtol=1.0e-10,
                atol=1.0e-8,
            ):
                raise MonteCarloContractError(
                    "Weighted annual diagnostics do not reconcile with hourly stock energy."
                )
            peak_heat_index = int(np.argmax(heating_stock_W))
            peak_cool_index = int(np.argmax(cooling_stock_W))
            coincident_heat_W = float(heating_stock_W[peak_heat_index])
            coincident_cool_W = float(cooling_stock_W[peak_cool_index])
            if coincident_heat_W > sum_peak_heating_W + 1.0e-5 or (
                coincident_cool_W > sum_peak_cooling_W + 1.0e-5
            ):
                raise MonteCarloContractError(
                    "Coincident stock peak exceeds the sum of individual peaks."
                )
            summary_records.append(
                {
                    **top_identity,
                    **provenance,
                    **cell_provenance,
                    **seed_provenance,
                    **stock_provenance,
                    **partition_provenance,
                    "region": region,
                    "modelled_dwellings": total_dwellings,
                    "stock_floor_area_m2": stock_floor_area_m2,
                    "physics_cell_count": len(cell_profiles),
                    "positive_weight_row_count": positive_cells,
                    "annual_heating_GWh": annual_heating_GWh,
                    "annual_potential_sensible_cooling_GWh": annual_cooling_GWh,
                    "heating_intensity_kWh_m2": (
                        annual_heating_GWh * 1.0e6 / stock_floor_area_m2
                        if stock_floor_area_m2 > 0.0
                        else 0.0
                    ),
                    "potential_cooling_intensity_kWh_m2": (
                        annual_cooling_GWh * 1.0e6 / stock_floor_area_m2
                        if stock_floor_area_m2 > 0.0
                        else 0.0
                    ),
                    "coincident_peak_heating_MW": coincident_heat_W / 1.0e6,
                    "coincident_peak_potential_cooling_MW": coincident_cool_W / 1.0e6,
                    "sum_individual_peak_heating_MW": sum_peak_heating_W / 1.0e6,
                    "sum_individual_peak_potential_cooling_MW": sum_peak_cooling_W / 1.0e6,
                    "heating_diversity_factor": (
                        coincident_heat_W / sum_peak_heating_W
                        if sum_peak_heating_W > 0.0
                        else 0.0
                    ),
                    "cooling_diversity_factor": (
                        coincident_cool_W / sum_peak_cooling_W
                        if sum_peak_cooling_W > 0.0
                        else 0.0
                    ),
                    "heating_full_load_equivalent_hours": (
                        annual_heating_GWh * 1000.0 / (coincident_heat_W / 1.0e6)
                        if coincident_heat_W > 0.0
                        else 0.0
                    ),
                    "potential_cooling_full_load_equivalent_hours": (
                        annual_cooling_GWh * 1000.0 / (coincident_cool_W / 1.0e6)
                        if coincident_cool_W > 0.0
                        else 0.0
                    ),
                    "peak_heating_timestamp_utc": reference_timestamps[peak_heat_index],
                    "peak_potential_cooling_timestamp_utc": reference_timestamps[peak_cool_index],
                    "cooling_interpretation": (
                        "potential useful sensible load under universal ideal 26C control"
                    ),
                    "stock_coverage": (
                        "R1-R4 modelled stock; R5-R6 residual excluded"
                        if require_full_stock
                        else "caller-supplied partial stock subset"
                    ),
                }
            )
            hourly_records.append(
                pd.DataFrame(
                    {
                        "timestamp_utc": reference_timestamps,
                        **{key: value for key, value in top_identity.items()},
                        "stock_scenario_id": stock_provenance[
                            "stock_scenario_id"
                        ],
                        "target_year": stock_provenance["target_year"],
                        "region": region,
                        **partition_provenance,
                        "heating_demand_MW": heating_stock_W / 1.0e6,
                        "potential_sensible_cooling_demand_MW": cooling_stock_W / 1.0e6,
                    }
                )
            )
            regional_arrays[region] = (heating_stock_W, cooling_stock_W)

        if region_names:
            summed_heat = sum(regional_arrays[region][0] for region in region_names)
            summed_cool = sum(regional_arrays[region][1] for region in region_names)
            national_heat, national_cool = regional_arrays["Belgium_modelled_stock"]
            if not np.allclose(summed_heat, national_heat, rtol=0.0, atol=1.0e-5) or (
                not np.allclose(summed_cool, national_cool, rtol=0.0, atol=1.0e-5)
            ):
                raise MonteCarloContractError(
                    "Regional hourly stock series do not sum to Belgium."
                )
    return (
        pd.DataFrame.from_records(summary_records),
        pd.concat(hourly_records, ignore_index=True),
    )
