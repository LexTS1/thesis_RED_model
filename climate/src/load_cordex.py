"""Load and strictly validate the canonical daily CORDEX input files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "timestamp",
    "T_out_C",
    "I_solar_W_m2",
    "scenario",
    "window",
    "gcm_model",
    "rcm_model",
    "ensemble_member",
    "source_files",
}


@dataclass(frozen=True)
class CordexSource:
    """One validated CORDEX daily series and its immutable provenance."""

    key: str
    frame: pd.DataFrame
    metadata: dict[str, Any]
    csv_path: Path
    metadata_path: Path
    csv_sha256: str
    metadata_sha256: str

    @property
    def scenario(self) -> str:
        return str(self.frame["scenario"].iat[0])

    @property
    def window(self) -> str:
        return str(self.frame["window"].iat[0])


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the climate configuration and retain its base directory."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Climate configuration is empty or invalid: {config_path}")

    required = {
        "dataset",
        "model_chain",
        "spatial_extraction",
        "physical_ranges",
        "solar_alpha_safety",
        "baseline_source",
        "sources",
        "outputs",
        "validation",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Climate configuration is missing sections: {sorted(missing)}")

    config["_config_path"] = config_path
    config["_base_dir"] = config_path.parent
    return config


def resolve_config_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a config path relative to the directory containing config.yaml."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_base_dir"]) / path


def _assert_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Configured {label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {label} {path}: expected {expected}, got {actual}"
        )
    return actual


def _assert_constant(frame: pd.DataFrame, column: str, expected: str, key: str) -> None:
    values = frame[column].drop_duplicates().tolist()
    if values != [expected]:
        raise ValueError(
            f"Source {key!r} has unexpected {column} values {values}; expected {expected!r}."
        )


def _validate_metadata(
    metadata: Mapping[str, Any], config: Mapping[str, Any], spec: Mapping[str, Any], key: str
) -> None:
    expected_dataset = config["dataset"]["name"]
    if metadata.get("dataset") != expected_dataset:
        raise ValueError(
            f"Source {key!r} metadata dataset is {metadata.get('dataset')!r}; "
            f"expected {expected_dataset!r}."
        )
    if metadata.get("scenario") != spec["scenario"] or metadata.get("window") != spec["window"]:
        raise ValueError(f"Source {key!r} metadata scenario/window does not match config.yaml.")
    if metadata.get("model_chain") != config["model_chain"]:
        raise ValueError(f"Source {key!r} metadata model chain does not match config.yaml.")
    if int(metadata.get("row_count", -1)) != int(spec["expected_rows"]):
        raise ValueError(f"Source {key!r} metadata row count does not match config.yaml.")

    spatial = metadata.get("spatial_extraction", {})
    expected_spatial = config["spatial_extraction"]
    for field in ("method", "selected_indices"):
        if spatial.get(field) != expected_spatial[field]:
            raise ValueError(f"Source {key!r} metadata spatial field {field!r} is inconsistent.")
    for field in ("target_lat", "target_lon", "selected_lat", "selected_lon"):
        if not np.isclose(float(spatial.get(field)), float(expected_spatial[field]), atol=1e-10):
            raise ValueError(f"Source {key!r} metadata spatial field {field!r} is inconsistent.")

    temperature = metadata.get("temperature", {})
    radiation = metadata.get("radiation", {})
    if temperature.get("variable") != config["dataset"]["variables"]["temperature"]:
        raise ValueError(f"Source {key!r} does not contain the configured temperature variable.")
    if temperature.get("output_units") != "degC":
        raise ValueError(f"Source {key!r} temperature is not expressed in degC.")
    if radiation.get("variable") != config["dataset"]["variables"]["solar_radiation"]:
        raise ValueError(f"Source {key!r} does not contain the configured solar variable.")
    if radiation.get("output_units") != "W m-2":
        raise ValueError(f"Source {key!r} radiation is not expressed in W m-2.")
    if metadata.get("time_calendar") != "gregorian":
        raise ValueError(f"Source {key!r} does not use the expected Gregorian calendar.")


def _validate_frame(
    frame: pd.DataFrame, config: Mapping[str, Any], spec: Mapping[str, Any], key: str
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Source {key!r} is missing required columns: {sorted(missing)}")
    if len(frame) != int(spec["expected_rows"]):
        raise ValueError(
            f"Source {key!r} contains {len(frame)} rows; expected {spec['expected_rows']}."
        )

    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="raise")
    if prepared["timestamp"].isna().any():
        raise ValueError(f"Source {key!r} contains missing timestamps.")
    if prepared["timestamp"].duplicated().any():
        raise ValueError(f"Source {key!r} contains duplicate timestamps.")
    if not prepared["timestamp"].is_monotonic_increasing:
        raise ValueError(f"Source {key!r} timestamps are not strictly ordered.")

    start = pd.Timestamp(str(spec["period_start"])) + pd.Timedelta(hours=12)
    end = pd.Timestamp(str(spec["period_end"])) + pd.Timedelta(hours=12)
    expected_index = pd.date_range(start=start, end=end, freq="D")
    actual_index = pd.DatetimeIndex(prepared["timestamp"])
    if not actual_index.equals(expected_index):
        missing_dates = expected_index.difference(actual_index)
        extra_dates = actual_index.difference(expected_index)
        raise ValueError(
            f"Source {key!r} is not a continuous daily series for the configured period; "
            f"missing={len(missing_dates)}, extra={len(extra_dates)}."
        )

    numeric = prepared[["T_out_C", "I_solar_W_m2"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Source {key!r} contains non-finite temperature or solar values.")
    prepared[["T_out_C", "I_solar_W_m2"]] = numeric.astype(float)

    t_min, t_max = map(float, config["physical_ranges"]["temperature_C"])
    i_min, i_max = map(float, config["physical_ranges"]["solar_W_m2"])
    if not prepared["T_out_C"].between(t_min, t_max).all():
        raise ValueError(f"Source {key!r} contains temperatures outside {t_min} to {t_max} degC.")
    if not prepared["I_solar_W_m2"].between(i_min, i_max).all():
        raise ValueError(f"Source {key!r} contains solar values outside {i_min} to {i_max} W/m2.")

    _assert_constant(prepared, "scenario", str(spec["scenario"]), key)
    _assert_constant(prepared, "window", str(spec["window"]), key)
    for field in ("gcm_model", "rcm_model", "ensemble_member"):
        _assert_constant(prepared, field, str(config["model_chain"][field]), key)

    prepared["year"] = prepared["timestamp"].dt.year.astype(int)
    prepared["month"] = prepared["timestamp"].dt.month.astype(int)
    prepared["day"] = prepared["timestamp"].dt.day.astype(int)
    if prepared["year"].nunique() != int(spec["expected_years"]):
        raise ValueError(f"Source {key!r} does not contain the configured number of years.")
    return prepared


def load_cordex_source(config: Mapping[str, Any], key: str) -> CordexSource:
    """Load one configured source after validating bytes, metadata, and daily data."""

    if key not in config["sources"]:
        raise KeyError(f"Unknown CORDEX source: {key}")
    spec = config["sources"][key]
    csv_path = resolve_config_path(config, spec["csv"])
    metadata_path = resolve_config_path(config, spec["metadata"])
    csv_hash = _assert_hash(csv_path, str(spec["csv_sha256"]), f"CSV for {key}")
    metadata_hash = _assert_hash(
        metadata_path, str(spec["metadata_sha256"]), f"metadata for {key}"
    )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata sidecar is not a JSON object: {metadata_path}")
    _validate_metadata(metadata, config, spec, key)

    frame = pd.read_csv(csv_path)
    prepared = _validate_frame(frame, config, spec, key)
    return CordexSource(
        key=key,
        frame=prepared,
        metadata=metadata,
        csv_path=csv_path,
        metadata_path=metadata_path,
        csv_sha256=csv_hash,
        metadata_sha256=metadata_hash,
    )


def load_all_sources(config: Mapping[str, Any]) -> dict[str, CordexSource]:
    """Load every configured source in deterministic configuration order."""

    return {key: load_cordex_source(config, key) for key in config["sources"]}
