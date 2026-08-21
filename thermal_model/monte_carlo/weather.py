"""Read-only bridge from the climate ensemble to Gate-5 simulations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from climate.src.load_cordex import (
    load_config,
    resolve_config_path,
    sha256_file,
)
from climate.src.load_observed import load_facade_templates
from climate.src.morph import load_delta_contract
from climate.src.transpose_facades import add_facade_irradiance
from climate.src.validate import validate_hourly_frame as validate_climate_hourly_frame

from .contracts import (
    CLIMATE_SCENARIOS,
    MonteCarloContractError,
    WeatherMember,
    canonical_sha256,
    complete_weather_forcing_sha256,
    validate_weather_member,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIMATE_CONFIG_PATH = PROJECT_ROOT / "climate/config.yaml"
MANIFEST_REQUIRED_COLUMNS = {
    "member_id",
    "scenario",
    "observed_pvgis_year",
    "climate_target",
    "is_leap_year",
    "row_count",
    "timestamp_start_utc",
    "timestamp_end_utc",
    "member_path",
    "member_sha256",
    "metadata_path",
    "metadata_sha256",
}
def _parse_bool_series(series: pd.Series, label: str) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise MonteCarloContractError(f"{label} must contain only true/false values.")
    return normalized.eq("true")


def _assert_under_base(path: Path, base: Path, label: str) -> None:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise MonteCarloContractError(
            f"{label} resolves outside the climate project directory: {path}."
        ) from exc


def _facade_hashes_from_config(config: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    facades = config["observed_weather"]["facades"]
    expected = {"north", "east", "south", "west"}
    if set(facades) != expected:
        raise MonteCarloContractError(
            f"Climate façade config must contain {sorted(expected)}; got {sorted(facades)}."
        )
    return tuple(
        (orientation, str(facades[orientation]["csv_sha256"]))
        for orientation in ("north", "east", "south", "west")
    )


def forcing_sha256(frame: pd.DataFrame) -> str:
    """Hash every timestamp and weather value consumed by Gate 4 or Gate 5."""

    return complete_weather_forcing_sha256(frame)


def _weather_contract_sha256(
    row: Mapping[str, Any],
    *,
    config_sha256: str,
    manifest_sha256: str,
    morph_contract_sha256: str,
    facade_source_sha256: tuple[tuple[str, str], ...],
) -> str:
    return canonical_sha256(
        {
            "contract": "gate5_weather_contract_v1",
            "member_id": str(row["member_id"]),
            "member_sha256": str(row["member_sha256"]),
            "metadata_sha256": str(row["metadata_sha256"]),
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "morph_contract_sha256": morph_contract_sha256,
            "facade_method": "pvgis_facade_shapes_monthly_alpha_v1",
            "facade_source_sha256": dict(facade_source_sha256),
        }
    )


def load_weather_catalog(
    config_path: str | Path = DEFAULT_CLIMATE_CONFIG_PATH,
) -> pd.DataFrame:
    """Load and validate the authoritative 54-row climate-member catalog."""

    resolved_config = Path(config_path).resolve()
    config = load_config(resolved_config)
    ensemble = config["observed_weather"]["ensemble"]
    ensemble_dir = resolve_config_path(config, ensemble["directory"])
    manifest_path = ensemble_dir / ensemble["manifest_csv"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Climate ensemble manifest is missing: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    frame = pd.read_csv(manifest_path)
    missing = sorted(MANIFEST_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise MonteCarloContractError(f"Climate manifest is missing columns: {missing}.")
    if len(frame) != 54 or frame["member_id"].duplicated().any():
        raise MonteCarloContractError(
            "Climate manifest must contain exactly 54 unique member identifiers."
        )
    result = frame.copy()
    result["observed_pvgis_year"] = pd.to_numeric(
        result["observed_pvgis_year"], errors="raise"
    ).astype(int)
    result["row_count"] = pd.to_numeric(result["row_count"], errors="raise").astype(int)
    result["is_leap_year"] = _parse_bool_series(result["is_leap_year"], "is_leap_year")
    if set(result["scenario"]) != set(CLIMATE_SCENARIOS):
        raise MonteCarloContractError("Climate manifest does not cover the three declared RCPs.")
    counts = result.groupby("scenario").size()
    if not counts.eq(18).all():
        raise MonteCarloContractError("Each RCP must contain 18 paired weather years.")
    expected_years = list(range(2006, 2024))
    for scenario, group in result.groupby("scenario"):
        if sorted(group["observed_pvgis_year"].tolist()) != expected_years:
            raise MonteCarloContractError(
                f"Climate scenario {scenario} does not contain PVGIS years 2006-2023."
            )
    expected_rows = np.where(result["is_leap_year"], 8784, 8760)
    if not np.array_equal(result["row_count"].to_numpy(dtype=int), expected_rows):
        raise MonteCarloContractError("Manifest leap flags and row counts are inconsistent.")
    if set(result["climate_target"].astype(str)) != {str(ensemble["climate_target"])}:
        raise MonteCarloContractError("Manifest climate target differs from config.yaml.")

    delta = load_delta_contract(config)
    facade_hashes = _facade_hashes_from_config(config)
    config_sha = sha256_file(resolved_config)
    result["weather_pair_id"] = result["observed_pvgis_year"].map(
        lambda year: f"pvgis_{int(year)}"
    )
    result["manifest_sha256"] = manifest_sha
    result["morph_contract_sha256"] = delta.csv_sha256
    result["facade_source_sha256_json"] = json.dumps(
        dict(facade_hashes), sort_keys=True, separators=(",", ":")
    )
    result["weather_contract_sha256"] = [
        _weather_contract_sha256(
            row._asdict(),
            config_sha256=config_sha,
            manifest_sha256=manifest_sha,
            morph_contract_sha256=delta.csv_sha256,
            facade_source_sha256=facade_hashes,
        )
        for row in result.itertuples(index=False)
    ]
    return result.sort_values(
        ["scenario", "observed_pvgis_year"], kind="stable"
    ).reset_index(drop=True)


def load_weather_member(
    member_id: str,
    config_path: str | Path = DEFAULT_CLIMATE_CONFIG_PATH,
) -> WeatherMember:
    """Load, hash-check, transpose and validate one climate ensemble member."""

    identifier = str(member_id).strip()
    if not identifier:
        raise MonteCarloContractError("member_id must be non-empty.")
    resolved_config = Path(config_path).resolve()
    config = load_config(resolved_config)
    catalog = load_weather_catalog(resolved_config)
    selected = catalog.loc[catalog["member_id"] == identifier]
    if len(selected) != 1:
        raise MonteCarloContractError(
            f"Unknown climate member {identifier!r}; expected one catalog match."
        )
    row = selected.iloc[0]
    member_path = resolve_config_path(config, str(row["member_path"]))
    metadata_path = resolve_config_path(config, str(row["metadata_path"]))
    _assert_under_base(member_path, Path(config["_base_dir"]), "member_path")
    _assert_under_base(metadata_path, Path(config["_base_dir"]), "metadata_path")
    for path, expected, label in (
        (member_path, str(row["member_sha256"]), "weather member"),
        (metadata_path, str(row["metadata_sha256"]), "weather metadata"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Configured {label} is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise MonteCarloContractError(
                f"{label} checksum mismatch for {identifier}: expected {expected}, got {actual}."
            )

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    expected_identity = {
        "member_id": identifier,
        "scenario": str(row["scenario"]),
        "observed_pvgis_year": int(row["observed_pvgis_year"]),
        "climate_target": str(row["climate_target"]),
        "row_count": int(row["row_count"]),
    }
    for field, expected in expected_identity.items():
        if metadata.get(field) != expected:
            raise MonteCarloContractError(
                f"Weather metadata {field}={metadata.get(field)!r}; expected {expected!r}."
            )

    horizontal = pd.read_csv(member_path)
    horizontal = validate_climate_hourly_frame(
        horizontal,
        int(row["observed_pvgis_year"]),
        config,
        identifier,
    )
    delta = load_delta_contract(config)
    facade_templates = load_facade_templates(config)
    complete = add_facade_irradiance(
        horizontal,
        str(row["scenario"]),
        delta.frame,
        facade_templates,
    )
    facade_hashes = tuple(
        (orientation, facade_templates[orientation].source_sha256)
        for orientation in ("north", "east", "south", "west")
    )
    observed = config["observed_weather"]
    chain = config["model_chain"]
    member = WeatherMember(
        member_id=identifier,
        climate_scenario_id=str(row["scenario"]),
        climate_target=str(row["climate_target"]),
        weather_pair_id=str(row["weather_pair_id"]),
        observed_pvgis_year=int(row["observed_pvgis_year"]),
        is_leap_year=bool(row["is_leap_year"]),
        row_count=int(row["row_count"]),
        frame=complete,
        site_id=(
            f"pvgis_brussels_{float(observed['latitude']):.3f}_"
            f"{float(observed['longitude']):.3f}"
        ),
        latitude=float(observed["latitude"]),
        longitude=float(observed["longitude"]),
        elevation_m=float(observed["elevation_m"]),
        timezone=str(observed["timestamp"]["timezone"]),
        gcm_model=str(chain["gcm_model"]),
        rcm_model=str(chain["rcm_model"]),
        cordex_ensemble_member=str(chain["ensemble_member"]),
        member_sha256=str(row["member_sha256"]),
        metadata_sha256=str(row["metadata_sha256"]),
        manifest_sha256=str(row["manifest_sha256"]),
        morph_contract_sha256=delta.csv_sha256,
        facade_source_sha256=facade_hashes,
        weather_contract_sha256=str(row["weather_contract_sha256"]),
        forcing_sha256=forcing_sha256(complete),
    )
    return validate_weather_member(member)


def load_weather_members(
    member_ids: Iterable[str],
    config_path: str | Path = DEFAULT_CLIMATE_CONFIG_PATH,
) -> list[WeatherMember]:
    """Load a deterministic ordered collection of unique members."""

    identifiers = [str(member_id).strip() for member_id in member_ids]
    if not identifiers or any(not value for value in identifiers):
        raise MonteCarloContractError("At least one non-empty weather member is required.")
    if len(set(identifiers)) != len(identifiers):
        raise MonteCarloContractError("Weather member identifiers must be unique.")
    return [load_weather_member(identifier, config_path) for identifier in identifiers]
