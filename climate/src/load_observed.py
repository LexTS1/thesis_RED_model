"""Load, clean, validate, and persist PVGIS-SARAH3 hourly observations."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .load_cordex import load_config, resolve_config_path, sha256_file


LOGGER = logging.getLogger("climate.load_observed")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
PVGIS_TIME_PATTERN = re.compile(r"^\d{8}:\d{4}$")
IRRADIANCE_COLUMNS = ("I_beam_W_m2", "I_diffuse_W_m2", "I_reflected_W_m2")


@dataclass(frozen=True)
class PvgisPlane:
    """A validated PVGIS plane-of-array export with normalized UTC timestamps."""

    label: str
    frame: pd.DataFrame
    source_path: Path
    source_sha256: str
    header_metadata: dict[str, Any]


def _find_header_row(path: Path) -> tuple[int, list[str]]:
    metadata_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line.lstrip().startswith("time,"):
                return line_number, metadata_lines
            metadata_lines.append(line.rstrip("\n"))
    raise ValueError(f"No PVGIS header row beginning with 'time,' was found in {path}.")


def _metadata_value(lines: Sequence[str], prefix: str) -> str:
    for line in lines:
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"PVGIS metadata does not contain {prefix!r}.")


def _first_number(value: str, field: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not match:
        raise ValueError(f"PVGIS metadata field {field!r} contains no number: {value!r}")
    return float(match.group(0))


def parse_pvgis_header(lines: Sequence[str]) -> dict[str, Any]:
    """Extract the stable location, database, slope, and azimuth metadata."""

    return {
        "latitude": _first_number(
            _metadata_value(lines, "Latitude (decimal degrees)"), "latitude"
        ),
        "longitude": _first_number(
            _metadata_value(lines, "Longitude (decimal degrees)"), "longitude"
        ),
        "elevation_m": _first_number(_metadata_value(lines, "Elevation (m)"), "elevation"),
        "radiation_database": _metadata_value(lines, "Radiation database"),
        "slope_deg": _first_number(_metadata_value(lines, "Slope"), "slope"),
        "azimuth_pvgis_deg": _first_number(_metadata_value(lines, "Azimuth"), "azimuth"),
    }


def _validate_header(
    metadata: Mapping[str, Any], observed: Mapping[str, Any], spec: Mapping[str, Any], label: str
) -> None:
    expected = {
        "latitude": float(observed["latitude"]),
        "longitude": float(observed["longitude"]),
        "elevation_m": float(observed["elevation_m"]),
        "slope_deg": float(spec["slope_deg"]),
        "azimuth_pvgis_deg": float(spec["azimuth_pvgis_deg"]),
    }
    for field, value in expected.items():
        if not np.isclose(float(metadata[field]), value, atol=1e-9):
            raise ValueError(
                f"PVGIS {label} metadata {field}={metadata[field]!r}; expected {value!r}."
            )
    if metadata["radiation_database"] != observed["radiation_database"]:
        raise ValueError(
            f"PVGIS {label} uses {metadata['radiation_database']!r}; expected "
            f"{observed['radiation_database']!r}."
        )


def clamp_negative_irradiance(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
    logger: logging.Logger = LOGGER,
) -> pd.DataFrame:
    """Clamp negative irradiance to zero and warn once per affected column."""

    result = frame.copy()
    for column in columns:
        count = int((result[column] < 0.0).sum())
        if count:
            minimum = float(result[column].min())
            logger.warning(
                "Clamping negative PVGIS irradiance: source=%s column=%s count=%d minimum=%.10f",
                label,
                column,
                count,
                minimum,
            )
            result[column] = result[column].clip(lower=0.0)
    return result


def _normalize_timestamps(series: pd.Series, observed: Mapping[str, Any]) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), format="%Y%m%d:%H%M", errors="raise")
    source_minute = int(observed["timestamp"]["source_minute"])
    minutes = sorted(int(value) for value in parsed.dt.minute.unique())
    if minutes != [source_minute]:
        raise ValueError(f"PVGIS timestamps have minute values {minutes}; expected [{source_minute}].")
    if observed["timestamp"]["normalization"] != "floor_to_hour":
        raise ValueError("Only floor_to_hour PVGIS timestamp normalization is supported.")
    if observed["timestamp"]["timezone"] != "UTC":
        raise ValueError("The canonical PVGIS timestamp timezone must be UTC.")
    return parsed.dt.floor("h").dt.tz_localize("UTC")


def _validate_hourly_coverage(
    timestamps: pd.Series, observed: Mapping[str, Any], label: str
) -> None:
    if timestamps.duplicated().any():
        raise ValueError(f"PVGIS {label} contains duplicate normalized timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"PVGIS {label} timestamps are not strictly ordered.")
    start = pd.Timestamp(str(observed["period_start"]), tz="UTC")
    end = pd.Timestamp(str(observed["period_end"]), tz="UTC") + pd.Timedelta(hours=23)
    expected = pd.date_range(start=start, end=end, freq="h")
    actual = pd.DatetimeIndex(timestamps)
    if not actual.equals(expected):
        raise ValueError(
            f"PVGIS {label} is not the configured continuous hourly series; "
            f"missing={len(expected.difference(actual))}, extra={len(actual.difference(expected))}."
        )
    if len(actual) != int(observed["expected_rows"]):
        raise ValueError(
            f"PVGIS {label} contains {len(actual)} rows; expected {observed['expected_rows']}."
        )


def load_pvgis_plane(
    config: Mapping[str, Any], label: str, spec: Mapping[str, Any]
) -> PvgisPlane:
    """Load one configured PVGIS plane and preserve all irradiance components."""

    observed = config["observed_weather"]
    path = resolve_config_path(config, spec["csv"])
    if not path.is_file():
        raise FileNotFoundError(f"Configured PVGIS {label} file does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != str(spec["csv_sha256"]):
        raise ValueError(
            f"SHA-256 mismatch for PVGIS {label}: expected {spec['csv_sha256']}, got {actual_hash}."
        )

    header_row, metadata_lines = _find_header_row(path)
    metadata = parse_pvgis_header(metadata_lines)
    _validate_header(metadata, observed, spec, label)
    raw = pd.read_csv(path, skiprows=header_row, low_memory=False)
    if "time" not in raw.columns:
        raise ValueError(f"PVGIS {label} data header has no time column.")
    data = raw.loc[
        raw["time"].astype(str).map(lambda value: bool(PVGIS_TIME_PATTERN.fullmatch(value)))
    ].copy()
    required = {"time", "Gb(i)", "Gd(i)", "Gr(i)", "H_sun", "T2m", "WS10m", "Int"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"PVGIS {label} is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError(f"PVGIS {label} contains no hourly data rows.")

    timestamps = _normalize_timestamps(data["time"], observed)
    _validate_hourly_coverage(timestamps, observed, label)
    result = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "I_beam_W_m2": pd.to_numeric(data["Gb(i)"], errors="coerce"),
            "I_diffuse_W_m2": pd.to_numeric(data["Gd(i)"], errors="coerce"),
            "I_reflected_W_m2": pd.to_numeric(data["Gr(i)"], errors="coerce"),
            "sun_height_deg": pd.to_numeric(data["H_sun"], errors="coerce"),
            "T_out_C": pd.to_numeric(data["T2m"], errors="coerce"),
            "wind_speed_10m_m_s": pd.to_numeric(data["WS10m"], errors="coerce"),
            "pvgis_reconstructed": pd.to_numeric(data["Int"], errors="coerce"),
        }
    )
    numeric_columns = [column for column in result.columns if column != "timestamp_utc"]
    if result[numeric_columns].isna().any().any():
        raise ValueError(f"PVGIS {label} contains non-numeric or missing required values.")
    if not np.isfinite(result[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"PVGIS {label} contains non-finite required values.")

    result = clamp_negative_irradiance(result, IRRADIANCE_COLUMNS, label)
    result["pvgis_reconstructed"] = result["pvgis_reconstructed"].astype(int).astype(bool)
    if result["pvgis_reconstructed"].any():
        raise ValueError(f"PVGIS {label} unexpectedly contains reconstructed radiation values.")
    if (result["wind_speed_10m_m_s"] < 0.0).any():
        raise ValueError(f"PVGIS {label} contains negative wind speed.")
    return PvgisPlane(
        label=label,
        frame=result,
        source_path=path,
        source_sha256=actual_hash,
        header_metadata=metadata,
    )


def load_observed_weather(config: Mapping[str, Any]) -> tuple[pd.DataFrame, PvgisPlane]:
    """Return the canonical cleaned horizontal hourly weather frame."""

    observed = config["observed_weather"]
    plane = load_pvgis_plane(config, "horizontal", observed["horizontal"])
    if not np.allclose(plane.frame["I_reflected_W_m2"], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("Horizontal PVGIS reflected irradiance must be zero at slope 0 degrees.")
    frame = plane.frame.rename(
        columns={
            "I_beam_W_m2": "I_beam_horizontal_W_m2",
            "I_diffuse_W_m2": "I_diffuse_horizontal_W_m2",
        }
    ).drop(columns="I_reflected_W_m2")
    frame["I_solar_W_m2"] = (
        frame["I_beam_horizontal_W_m2"] + frame["I_diffuse_horizontal_W_m2"]
    )
    ordered = [
        "timestamp_utc",
        "T_out_C",
        "I_beam_horizontal_W_m2",
        "I_diffuse_horizontal_W_m2",
        "I_solar_W_m2",
        "wind_speed_10m_m_s",
        "sun_height_deg",
        "pvgis_reconstructed",
    ]
    frame = frame[ordered]

    t_min, t_max = map(float, config["physical_ranges"]["temperature_C"])
    i_min, i_max = map(float, config["physical_ranges"]["solar_W_m2"])
    if not frame["T_out_C"].between(t_min, t_max).all():
        raise ValueError("Observed PVGIS temperature lies outside configured physical bounds.")
    for column in (
        "I_beam_horizontal_W_m2",
        "I_diffuse_horizontal_W_m2",
        "I_solar_W_m2",
    ):
        if not frame[column].between(i_min, i_max).all():
            raise ValueError(f"Observed PVGIS {column} lies outside configured physical bounds.")
    if not np.allclose(
        frame["I_solar_W_m2"],
        frame["I_beam_horizontal_W_m2"] + frame["I_diffuse_horizontal_W_m2"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Observed PVGIS GHI is not beam plus diffuse irradiance.")
    night = frame["sun_height_deg"] < 0.0
    if (frame.loc[night, "I_solar_W_m2"] != 0.0).any():
        raise ValueError("Observed PVGIS irradiance is non-zero while sun height is below zero.")
    return frame, plane


def split_complete_years(
    frame: pd.DataFrame, timestamp_column: str = "timestamp_utc"
) -> dict[int, pd.DataFrame]:
    """Split a continuous UTC frame into complete 8,760/8,784-hour years."""

    timestamps = pd.DatetimeIndex(frame[timestamp_column])
    if timestamps.tz is None or str(timestamps.tz) != "UTC":
        raise ValueError("Complete-year splitting requires timezone-aware UTC timestamps.")
    years: dict[int, pd.DataFrame] = {}
    for year, group in frame.groupby(frame[timestamp_column].dt.year, sort=True):
        expected = pd.date_range(
            start=pd.Timestamp(year=int(year), month=1, day=1, tz="UTC"),
            end=pd.Timestamp(year=int(year), month=12, day=31, hour=23, tz="UTC"),
            freq="h",
        )
        actual = pd.DatetimeIndex(group[timestamp_column])
        if not actual.equals(expected):
            raise ValueError(f"PVGIS year {year} is not a complete hourly UTC calendar year.")
        years[int(year)] = group.reset_index(drop=True).copy()
    return years


def load_facade_templates(config: Mapping[str, Any]) -> dict[str, PvgisPlane]:
    """Load four aligned PVGIS 90-degree façade templates."""

    observed = config["observed_weather"]
    planes = {
        orientation: load_pvgis_plane(config, orientation, spec)
        for orientation, spec in observed["facades"].items()
    }
    reference = pd.DatetimeIndex(next(iter(planes.values())).frame["timestamp_utc"])
    for orientation, plane in planes.items():
        if not pd.DatetimeIndex(plane.frame["timestamp_utc"]).equals(reference):
            raise ValueError(f"PVGIS façade {orientation} timestamps do not align.")
    return planes


def _format_utc(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    serializable = frame.copy()
    serializable["timestamp_utc"] = _format_utc(serializable["timestamp_utc"])
    serializable.to_csv(temporary, index=False, float_format="%.6f", lineterminator="\n")
    temporary.replace(path)


def build_clean_observed(config: Mapping[str, Any]) -> dict[str, Path]:
    """Build the deterministic cleaned horizontal PVGIS dataset and metadata."""

    frame, plane = load_observed_weather(config)
    years = split_complete_years(frame)
    processed = config["observed_weather"]["processed"]
    output_dir = resolve_config_path(config, processed["directory"])
    csv_path = output_dir / processed["clean_weather"]
    metadata_path = output_dir / processed["clean_metadata"]
    _atomic_write_csv(frame, csv_path)

    metadata = {
        "schema_version": 1,
        "source": {
            "path": str(plane.source_path.relative_to(Path(config["_base_dir"]))),
            "sha256": plane.source_sha256,
            "header": plane.header_metadata,
        },
        "timestamp": {
            "source_format": "YYYYMMDD:HHMM",
            "source_minute": int(config["observed_weather"]["timestamp"]["source_minute"]),
            "normalization": "floor_to_hour",
            "timezone": "UTC",
            "first": frame["timestamp_utc"].iat[0].isoformat(),
            "last": frame["timestamp_utc"].iat[-1].isoformat(),
        },
        "row_count": int(len(frame)),
        "year_count": int(len(years)),
        "year_rows": {str(year): int(len(year_frame)) for year, year_frame in years.items()},
        "columns": list(frame.columns),
        "units": {
            "T_out_C": "degC",
            "I_beam_horizontal_W_m2": "W/m2",
            "I_diffuse_horizontal_W_m2": "W/m2",
            "I_solar_W_m2": "W/m2",
            "wind_speed_10m_m_s": "m/s",
            "sun_height_deg": "degree",
            "pvgis_reconstructed": "boolean",
        },
        "ranges": {
            column: {"minimum": float(frame[column].min()), "maximum": float(frame[column].max())}
            for column in (
                "T_out_C",
                "I_beam_horizontal_W_m2",
                "I_diffuse_horizontal_W_m2",
                "I_solar_W_m2",
                "wind_speed_10m_m_s",
                "sun_height_deg",
            )
        },
        "reconstructed_row_count": int(frame["pvgis_reconstructed"].sum()),
        "output": {
            "path": str(csv_path.relative_to(Path(config["_base_dir"]))),
            "sha256": sha256_file(csv_path),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_name(f".{metadata_path.name}.writing")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    return {"clean_weather": csv_path, "metadata": metadata_path}


def load_clean_observed(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the persisted clean file after verifying its recorded output hash."""

    processed = config["observed_weather"]["processed"]
    output_dir = resolve_config_path(config, processed["directory"])
    csv_path = output_dir / processed["clean_weather"]
    metadata_path = output_dir / processed["clean_metadata"]
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Clean PVGIS artifacts are missing; run python3 -m climate.src.load_observed first."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_hash = str(metadata["output"]["sha256"])
    actual_hash = sha256_file(csv_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Clean PVGIS hash mismatch: expected {expected_hash}, got {actual_hash}."
        )

    frame = pd.read_csv(csv_path)
    expected_columns = list(metadata["columns"])
    if list(frame.columns) != expected_columns:
        raise ValueError("Clean PVGIS columns do not match the recorded metadata schema.")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if frame["pvgis_reconstructed"].dtype != bool:
        normalized = frame["pvgis_reconstructed"].astype(str).str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError("Clean PVGIS reconstruction flags are not boolean.")
        frame["pvgis_reconstructed"] = normalized.eq("true")
    years = split_complete_years(frame)
    if len(frame) != int(metadata["row_count"]) or len(years) != int(metadata["year_count"]):
        raise ValueError("Clean PVGIS row/year counts do not match recorded metadata.")
    return frame, metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cleaned PVGIS-SARAH3 weather file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s - %(message)s",
    )
    try:
        paths = build_clean_observed(load_config(args.config))
    except Exception as exc:
        LOGGER.error("Observed PVGIS build failed: %s", exc)
        if args.verbose:
            LOGGER.exception("Detailed failure")
        return 1
    for name, path in paths.items():
        LOGGER.info("Wrote %s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
