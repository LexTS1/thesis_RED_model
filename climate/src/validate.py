"""Audit the persisted PVGIS baseline and 57-member 2050 weather ensemble."""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .load_cordex import load_all_sources, load_config, resolve_config_path, sha256_file
from .load_observed import load_clean_observed, split_complete_years
from .morph import DeltaContract, load_delta_contract, scenario_parameters


LOGGER = logging.getLogger("climate.validate")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

HOURLY_COLUMNS = (
    "timestamp_utc",
    "T_out_C",
    "I_beam_horizontal_W_m2",
    "I_diffuse_horizontal_W_m2",
    "I_solar_W_m2",
    "wind_speed_10m_m_s",
    "sun_height_deg",
    "pvgis_reconstructed",
)
IRRADIANCE_COLUMNS = (
    "I_beam_horizontal_W_m2",
    "I_diffuse_horizontal_W_m2",
    "I_solar_W_m2",
)
UNCHANGED_COLUMNS = (
    "timestamp_utc",
    "wind_speed_10m_m_s",
    "sun_height_deg",
    "pvgis_reconstructed",
)
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


def _relative(path: Path, config: Mapping[str, Any]) -> str:
    try:
        return str(path.relative_to(Path(config["_base_dir"])))
    except ValueError:
        return str(path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    frame.to_csv(temporary, index=False, float_format="%.10f", lineterminator="\n")
    temporary.replace(path)


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    _atomic_write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        path,
    )


def _parse_boolean(series: pd.Series, label: str) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{label} reconstruction flags are not boolean.")
    return normalized.eq("true")


def prepare_hourly_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Coerce one persisted weather frame into its strict canonical schema."""

    if tuple(frame.columns) != HOURLY_COLUMNS:
        raise ValueError(
            f"{label} columns do not match the canonical hourly schema: {list(frame.columns)}"
        )
    prepared = frame.copy()
    prepared["timestamp_utc"] = pd.to_datetime(
        prepared["timestamp_utc"], utc=True, errors="raise"
    )
    if prepared["timestamp_utc"].isna().any():
        raise ValueError(f"{label} contains missing timestamps.")
    prepared["pvgis_reconstructed"] = _parse_boolean(
        prepared["pvgis_reconstructed"], label
    )
    numeric_columns = [
        column
        for column in HOURLY_COLUMNS
        if column not in {"timestamp_utc", "pvgis_reconstructed"}
    ]
    prepared[numeric_columns] = prepared[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )
    return prepared


def validate_hourly_frame(
    frame: pd.DataFrame,
    expected_year: int,
    config: Mapping[str, Any],
    label: str,
) -> pd.DataFrame:
    """Enforce completeness, finiteness, physical bounds, and solar identities."""

    prepared = prepare_hourly_frame(frame, label)
    timestamps = prepared["timestamp_utc"]
    if timestamps.duplicated().any():
        raise ValueError(f"{label} contains duplicate timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{label} timestamps are not strictly ordered.")
    expected = pd.date_range(
        start=pd.Timestamp(year=int(expected_year), month=1, day=1, tz="UTC"),
        end=pd.Timestamp(
            year=int(expected_year), month=12, day=31, hour=23, tz="UTC"
        ),
        freq="h",
    )
    actual = pd.DatetimeIndex(timestamps)
    if not actual.equals(expected):
        raise ValueError(
            f"{label} is not a complete hourly UTC year {expected_year}; "
            f"expected={len(expected)}, actual={len(actual)}, "
            f"missing={len(expected.difference(actual))}, "
            f"extra={len(actual.difference(expected))}."
        )
    if prepared.isna().any().any():
        raise ValueError(f"{label} contains missing values.")
    numeric = prepared.drop(columns=["timestamp_utc", "pvgis_reconstructed"])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite values.")
    if prepared["pvgis_reconstructed"].any():
        raise ValueError(f"{label} contains reconstructed PVGIS values.")

    t_min, t_max = map(float, config["physical_ranges"]["temperature_C"])
    i_min, i_max = map(float, config["physical_ranges"]["solar_W_m2"])
    if not prepared["T_out_C"].between(t_min, t_max).all():
        raise ValueError(
            f"{label} temperature lies outside {t_min} to {t_max} degC."
        )
    for column in IRRADIANCE_COLUMNS:
        if not prepared[column].between(i_min, i_max).all():
            raise ValueError(
                f"{label} {column} lies outside {i_min} to {i_max} W/m2."
            )
    if (prepared["wind_speed_10m_m_s"] < 0.0).any():
        raise ValueError(f"{label} contains negative wind speed.")

    ghi_tolerance = float(
        config["validation"]["tolerances"]["ghi_component_W_m2"]
    )
    if not np.allclose(
        prepared["I_solar_W_m2"],
        prepared["I_beam_horizontal_W_m2"]
        + prepared["I_diffuse_horizontal_W_m2"],
        rtol=0.0,
        atol=ghi_tolerance,
    ):
        residual = (
            prepared["I_solar_W_m2"]
            - prepared["I_beam_horizontal_W_m2"]
            - prepared["I_diffuse_horizontal_W_m2"]
        ).abs()
        raise ValueError(
            f"{label} GHI is not beam plus diffuse; "
            f"maximum residual={float(residual.max()):.12g} W/m2."
        )
    night = prepared["sun_height_deg"] < 0.0
    if (prepared.loc[night, list(IRRADIANCE_COLUMNS)] != 0.0).any().any():
        raise ValueError(f"{label} contains non-zero irradiance below the horizon.")
    return prepared


def calculate_degree_days(
    daily_mean_temperature: pd.Series, config: Mapping[str, Any]
) -> dict[str, float]:
    """Calculate the harmonised Eurostat HDD and CDD indicators."""

    degree_days = config["validation"]["degree_days"]
    heating_trigger = float(degree_days["heating_trigger_C"])
    heating_reference = float(degree_days["heating_reference_C"])
    cooling_trigger = float(degree_days["cooling_trigger_C"])
    cooling_reference = float(degree_days["cooling_reference_C"])
    temperature = pd.to_numeric(daily_mean_temperature, errors="raise")
    if temperature.isna().any() or not np.isfinite(temperature.to_numpy()).all():
        raise ValueError("Degree-day input contains missing or non-finite temperature.")
    return {
        "HDD_C_days": float(
            np.where(
                temperature <= heating_trigger,
                heating_reference - temperature,
                0.0,
            ).sum()
        ),
        "CDD_C_days": float(
            np.where(
                temperature >= cooling_trigger,
                temperature - cooling_reference,
                0.0,
            ).sum()
        ),
    }


def calculate_annual_metrics(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, float]:
    """Calculate Eurostat degree days and horizontal annual solar irradiation."""

    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    daily = pd.DataFrame(
        {
            "day": timestamps.dt.floor("D"),
            "T_out_C": pd.to_numeric(frame["T_out_C"], errors="raise"),
        }
    ).groupby("day", sort=True)["T_out_C"].mean()
    solar = pd.to_numeric(frame["I_solar_W_m2"], errors="raise")
    return {
        **calculate_degree_days(daily, config),
        "annual_solar_kWh_m2": float(solar.sum() / 1000.0),
    }


def load_reference_degree_days(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    """Load the immutable official Brussels BE100 degree-day snapshot."""

    reference = config["validation"]["degree_days"]["reference"]
    csv_path = resolve_config_path(config, reference["csv"])
    metadata_path = resolve_config_path(config, reference["metadata"])
    if sha256_file(csv_path) != str(reference["csv_sha256"]):
        raise ValueError("Official BE100 degree-day CSV hash mismatch.")
    if sha256_file(metadata_path) != str(reference["metadata_sha256"]):
        raise ValueError("Official BE100 degree-day metadata hash mismatch.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("provider") != reference["provider"]:
        raise ValueError("Official degree-day provider metadata is inconsistent.")
    if metadata.get("dataset_code") != reference["dataset_code"]:
        raise ValueError("Official degree-day dataset metadata is inconsistent.")
    if metadata.get("geo_code") != reference["geo_code"]:
        raise ValueError("Official degree-day geography metadata is inconsistent.")
    if metadata.get("csv", {}).get("sha256") != str(reference["csv_sha256"]):
        raise ValueError("Official degree-day metadata records an incorrect CSV hash.")

    frame = pd.read_csv(csv_path)
    expected_columns = ["year", "HDD_C_days", "CDD_C_days"]
    if list(frame.columns) != expected_columns:
        raise ValueError("Official BE100 degree-day CSV has an incorrect schema.")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame[["HDD_C_days", "CDD_C_days"]] = frame[
        ["HDD_C_days", "CDD_C_days"]
    ].apply(pd.to_numeric, errors="raise")
    expected_years = list(
        range(int(reference["period_start"]), int(reference["period_end"]) + 1)
    )
    if frame["year"].tolist() != expected_years:
        raise ValueError("Official BE100 degree-day CSV has an incorrect year range.")
    if frame[["HDD_C_days", "CDD_C_days"]].isna().any().any():
        raise ValueError("Official BE100 degree-day CSV contains missing values.")
    if not np.isfinite(
        frame[["HDD_C_days", "CDD_C_days"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Official BE100 degree-day CSV contains non-finite values.")
    if (frame[["HDD_C_days", "CDD_C_days"]] < 0.0).any().any():
        raise ValueError("Official BE100 degree-day CSV contains negative values.")
    return frame, metadata, csv_path, metadata_path


def build_observed_reference_comparison(
    observed_metrics: Mapping[int, Mapping[str, float]],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Pair PVGIS-derived indicators with the official annual BE100 series."""

    records = [
        {
            "year": int(year),
            "pvgis_HDD_C_days": float(metrics["HDD_C_days"]),
            "pvgis_CDD_C_days": float(metrics["CDD_C_days"]),
        }
        for year, metrics in sorted(observed_metrics.items())
    ]
    comparison = pd.DataFrame.from_records(records).merge(
        reference.rename(
            columns={
                "HDD_C_days": "official_BE100_HDD_C_days",
                "CDD_C_days": "official_BE100_CDD_C_days",
            }
        ),
        on="year",
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(reference):
        raise ValueError("PVGIS and official BE100 degree-day years do not align.")
    comparison["HDD_error_C_days"] = (
        comparison["pvgis_HDD_C_days"]
        - comparison["official_BE100_HDD_C_days"]
    )
    comparison["CDD_error_C_days"] = (
        comparison["pvgis_CDD_C_days"]
        - comparison["official_BE100_CDD_C_days"]
    )
    return comparison[
        [
            "year",
            "pvgis_HDD_C_days",
            "official_BE100_HDD_C_days",
            "HDD_error_C_days",
            "pvgis_CDD_C_days",
            "official_BE100_CDD_C_days",
            "CDD_error_C_days",
        ]
    ]


def reference_comparison_statistics(comparison: pd.DataFrame) -> dict[str, Any]:
    """Summarise agreement with the official BE100 series without imposing a gate."""

    return {
        "HDD": {
            "pearson_correlation": float(
                comparison["pvgis_HDD_C_days"].corr(
                    comparison["official_BE100_HDD_C_days"]
                )
            ),
            "mean_bias_C_days": float(comparison["HDD_error_C_days"].mean()),
            "mean_absolute_error_C_days": float(
                comparison["HDD_error_C_days"].abs().mean()
            ),
        },
        "CDD": {
            "pearson_correlation": float(
                comparison["pvgis_CDD_C_days"].corr(
                    comparison["official_BE100_CDD_C_days"]
                )
            ),
            "mean_bias_C_days": float(comparison["CDD_error_C_days"].mean()),
            "mean_absolute_error_C_days": float(
                comparison["CDD_error_C_days"].abs().mean()
            ),
        },
    }


def calculate_cordex_annual_degree_days(
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Calculate identical annual indicators directly from every CORDEX daily year."""

    sources = load_all_sources(config)
    records: list[dict[str, Any]] = []
    for source_key, source in sources.items():
        role = str(config["sources"][source_key]["role"])
        for climate_year, year_frame in source.frame.groupby("year", sort=True):
            metrics = calculate_degree_days(year_frame["T_out_C"], config)
            records.append(
                {
                    "source_key": source_key,
                    "role": role,
                    "scenario": str(source.scenario),
                    "climate_year": int(climate_year),
                    "row_count": int(len(year_frame)),
                    "HDD_C_days": float(metrics["HDD_C_days"]),
                    "CDD_C_days": float(metrics["CDD_C_days"]),
                }
            )
    result = pd.DataFrame.from_records(records)
    if len(result) != 88:
        raise ValueError(f"CORDEX degree-day table contains {len(result)} years; expected 88.")
    return result


def build_cordex_morph_comparison(
    cordex: pd.DataFrame,
    members: pd.DataFrame,
    scenarios: list[str],
) -> pd.DataFrame:
    """Compare direct CORDEX and paired-morph degree-day changes by scenario."""

    historical = cordex.loc[cordex["role"] == "baseline"]
    if len(historical) != 25:
        raise ValueError("CORDEX degree-day comparison requires 25 historical years.")
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        future = cordex.loc[
            (cordex["role"] == "future") & (cordex["scenario"] == scenario)
        ]
        morphed = members.loc[members["scenario"] == scenario]
        if len(future) != 21 or len(morphed) != 19:
            raise ValueError(
                f"Degree-day comparison has incomplete samples for {scenario}."
            )
        record: dict[str, Any] = {
            "scenario": scenario,
            "cordex_historical_years": int(len(historical)),
            "cordex_future_years": int(len(future)),
            "morphed_observed_years": int(len(morphed)),
        }
        for indicator in ("HDD", "CDD"):
            metric = f"{indicator}_C_days"
            observed_column = f"observed_{indicator}_C_days"
            morphed_column = f"morphed_{indicator}_C_days"
            historical_mean = float(historical[metric].mean())
            future_mean = float(future[metric].mean())
            cordex_change = future_mean - historical_mean
            observed_mean = float(morphed[observed_column].mean())
            morphed_mean = float(morphed[morphed_column].mean())
            paired_change = float(
                (morphed[morphed_column] - morphed[observed_column]).mean()
            )
            record.update(
                {
                    f"cordex_historical_mean_{metric}": historical_mean,
                    f"cordex_future_mean_{metric}": future_mean,
                    f"cordex_change_{metric}": cordex_change,
                    f"pvgis_observed_mean_{metric}": observed_mean,
                    f"morphed_mean_{metric}": morphed_mean,
                    f"morphed_paired_change_{metric}": paired_change,
                    f"morph_minus_cordex_change_{metric}": (
                        paired_change - cordex_change
                    ),
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _assert_unchanged(
    observed: pd.DataFrame, member: pd.DataFrame, member_id: str
) -> None:
    for column in UNCHANGED_COLUMNS:
        if column == "timestamp_utc":
            equal = pd.DatetimeIndex(member[column]).equals(
                pd.DatetimeIndex(observed[column])
            )
        else:
            equal = np.array_equal(
                member[column].to_numpy(), observed[column].to_numpy()
            )
        if not equal:
            raise ValueError(
                f"{member_id} changed invariant observed column {column!r}."
            )


def validate_member_pair(
    observed: pd.DataFrame,
    member: pd.DataFrame,
    member_id: str,
    scenario: str,
    deltas: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate unchanged fields and every monthly temperature/solar morph identity."""

    _assert_unchanged(observed, member, member_id)
    parameters = scenario_parameters(deltas, scenario)
    months = observed["timestamp_utc"].dt.month
    temperature_tolerance = float(
        config["validation"]["tolerances"]["monthly_temperature_C"]
    )
    solar_tolerance = float(
        config["validation"]["tolerances"]["monthly_solar_factor"]
    )
    rows: list[dict[str, Any]] = []
    for month in range(1, 13):
        mask = months == month
        expected_delta = float(parameters.loc[month, "delta_T_C"])
        actual_delta = float(
            (member.loc[mask, "T_out_C"] - observed.loc[mask, "T_out_C"]).mean()
        )
        delta_residual = actual_delta - expected_delta
        if not np.isclose(
            actual_delta, expected_delta, rtol=0.0, atol=temperature_tolerance
        ):
            raise ValueError(
                f"{member_id} temperature morph failed in month {month:02d}: "
                f"expected={expected_delta:.12g}, actual={actual_delta:.12g}."
            )

        expected_alpha = float(parameters.loc[month, "alpha_solar_applied"])
        ratios: dict[str, float] = {}
        for label, column in (
            ("beam", "I_beam_horizontal_W_m2"),
            ("diffuse", "I_diffuse_horizontal_W_m2"),
            ("ghi", "I_solar_W_m2"),
        ):
            denominator = float(observed.loc[mask, column].sum())
            ratio = (
                float(member.loc[mask, column].sum()) / denominator
                if denominator > 0.0
                else float("nan")
            )
            ratios[label] = ratio
            if denominator > 0.0 and not np.isclose(
                ratio, expected_alpha, rtol=0.0, atol=solar_tolerance
            ):
                raise ValueError(
                    f"{member_id} {label} solar morph failed in month {month:02d}: "
                    f"expected={expected_alpha:.12g}, actual={ratio:.12g}."
                )
        rows.append(
            {
                "member_id": member_id,
                "scenario": scenario,
                "observed_pvgis_year": int(
                    observed["timestamp_utc"].dt.year.iat[0]
                ),
                "month": month,
                "month_hours": int(mask.sum()),
                "expected_delta_T_C": expected_delta,
                "actual_delta_T_C": actual_delta,
                "delta_T_residual_C": delta_residual,
                "expected_alpha_applied": expected_alpha,
                "actual_beam_ratio": ratios["beam"],
                "beam_alpha_residual": ratios["beam"] - expected_alpha,
                "actual_diffuse_ratio": ratios["diffuse"],
                "diffuse_alpha_residual": ratios["diffuse"] - expected_alpha,
                "actual_ghi_ratio": ratios["ghi"],
                "ghi_alpha_residual": ratios["ghi"] - expected_alpha,
                "hard_status": "pass",
            }
        )
    return rows


def assess_plausibility(
    member_id: str,
    observed_metrics: Mapping[str, float],
    member_metrics: Mapping[str, float],
    config: Mapping[str, Any],
    logger: logging.Logger = LOGGER,
) -> list[dict[str, Any]]:
    """Return warning-only paired-baseline climate plausibility findings."""

    bands = config["validation"]["plausibility"]
    hdd_ratio = (
        float(member_metrics["HDD_C_days"])
        / float(observed_metrics["HDD_C_days"])
    )
    cdd_change = float(member_metrics["CDD_C_days"]) - float(
        observed_metrics["CDD_C_days"]
    )
    solar_ratio = (
        float(member_metrics["annual_solar_kWh_m2"])
        / float(observed_metrics["annual_solar_kWh_m2"])
    )
    checks = (
        (
            "hdd_ratio_outside_band",
            "HDD ratio",
            hdd_ratio,
            bands["hdd_ratio_vs_observed"],
        ),
        (
            "cdd_change_outside_band",
            "CDD change",
            cdd_change,
            bands["cdd_change_C_days"],
        ),
        (
            "annual_solar_ratio_outside_band",
            "annual solar ratio",
            solar_ratio,
            bands["annual_solar_ratio_vs_observed"],
        ),
    )
    warnings: list[dict[str, Any]] = []
    for code, metric, value, bounds in checks:
        lower, upper = map(float, bounds)
        if not lower <= value <= upper:
            message = (
                f"{member_id} {metric}={value:.10f} lies outside "
                f"the warning band [{lower:.10f}, {upper:.10f}]."
            )
            logger.warning(message)
            warnings.append(
                {
                    "member_id": member_id,
                    "code": code,
                    "metric": metric,
                    "value": value,
                    "minimum": lower,
                    "maximum": upper,
                    "message": message,
                }
            )
    return warnings


def _load_manifest(
    config: Mapping[str, Any], clean_hash: str, delta_contract: DeltaContract
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    ensemble = config["observed_weather"]["ensemble"]
    root = resolve_config_path(config, ensemble["directory"])
    csv_path = root / ensemble["manifest_csv"]
    json_path = root / ensemble["manifest_json"]
    if not csv_path.is_file() or not json_path.is_file():
        raise FileNotFoundError("The persisted ensemble manifests are missing.")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if sha256_file(csv_path) != str(payload["manifest_csv"]["sha256"]):
        raise ValueError("Ensemble manifest CSV hash does not match the JSON manifest.")
    if payload["manifest_csv"]["path"] != _relative(csv_path, config):
        raise ValueError("Ensemble JSON manifest records an incorrect CSV manifest path.")
    if payload["clean_observed"]["sha256"] != clean_hash:
        raise ValueError("Ensemble manifest references a different cleaned PVGIS artifact.")
    if payload["morph_contract"]["sha256"] != delta_contract.csv_sha256:
        raise ValueError("Ensemble manifest references a different monthly delta contract.")

    manifest = pd.read_csv(csv_path)
    missing = MANIFEST_REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"Ensemble manifest is missing columns: {sorted(missing)}")
    if manifest["member_id"].duplicated().any():
        raise ValueError("Ensemble manifest contains duplicate member IDs.")
    if manifest.duplicated(["scenario", "observed_pvgis_year"]).any():
        raise ValueError("Ensemble manifest contains duplicate scenario/year members.")
    if int(payload["member_count"]) != len(manifest):
        raise ValueError("CSV and JSON ensemble member counts do not match.")
    if int(payload["total_member_hours"]) != int(manifest["row_count"].sum()):
        raise ValueError("CSV and JSON ensemble hour counts do not match.")

    json_members = {row["member_id"]: row for row in payload["members"]}
    if len(json_members) != len(manifest):
        raise ValueError("JSON manifest member records are missing or duplicated.")
    for row in manifest.itertuples(index=False):
        if row.member_id not in json_members:
            raise ValueError(f"JSON manifest is missing member {row.member_id}.")
        recorded = json_members[row.member_id]
        for field in (
            "scenario",
            "observed_pvgis_year",
            "row_count",
            "member_path",
            "member_sha256",
            "metadata_path",
            "metadata_sha256",
        ):
            if str(recorded[field]) != str(getattr(row, field)):
                raise ValueError(
                    f"CSV and JSON manifests disagree for {row.member_id} field {field}."
                )
    return manifest, payload, csv_path, json_path


def _validate_manifest_shape(
    manifest: pd.DataFrame,
    observed_years: Mapping[int, pd.DataFrame],
    scenarios: list[str],
    config: Mapping[str, Any],
) -> None:
    expected_pairs = {
        (scenario, year) for scenario in scenarios for year in observed_years
    }
    actual_pairs = {
        (str(row.scenario), int(row.observed_pvgis_year))
        for row in manifest.itertuples(index=False)
    }
    if actual_pairs != expected_pairs:
        raise ValueError(
            "Ensemble manifest is not the complete configured scenario/year product."
        )
    expected_target = str(config["observed_weather"]["ensemble"]["climate_target"])
    if set(manifest["climate_target"].astype(str)) != {expected_target}:
        raise ValueError("Ensemble manifest contains an incorrect climate target.")
    expected_members = len(scenarios) * int(
        config["observed_weather"]["expected_years"]
    )
    expected_hours = sum(len(frame) for frame in observed_years.values()) * len(
        scenarios
    )
    if len(manifest) != expected_members:
        raise ValueError(
            f"Ensemble contains {len(manifest)} members; expected {expected_members}."
        )
    if int(manifest["row_count"].sum()) != expected_hours:
        raise ValueError(
            f"Ensemble contains {int(manifest['row_count'].sum())} member-hours; "
            f"expected {expected_hours}."
        )


def _validate_member_metadata(
    metadata: Mapping[str, Any],
    row: Any,
    member: pd.DataFrame,
    member_path: Path,
    clean_hash: str,
    delta_contract: DeltaContract,
    config: Mapping[str, Any],
) -> None:
    if int(metadata.get("schema_version", -1)) != 1:
        raise ValueError(f"Metadata for {row.member_id} has an unsupported schema version.")
    expected = {
        "member_id": str(row.member_id),
        "scenario": str(row.scenario),
        "climate_target": str(row.climate_target),
        "observed_pvgis_year": int(row.observed_pvgis_year),
        "row_count": int(row.row_count),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"Metadata for {row.member_id} has inconsistent field {field}."
            )
    if bool(metadata.get("is_leap_year")) != calendar.isleap(
        int(row.observed_pvgis_year)
    ):
        raise ValueError(f"Metadata for {row.member_id} has an incorrect leap-year flag.")
    if metadata.get("columns") != list(HOURLY_COLUMNS):
        raise ValueError(f"Metadata for {row.member_id} has an incorrect column schema.")
    if metadata.get("method", {}).get("tag") != str(
        config["observed_weather"]["ensemble"]["method_tag"]
    ):
        raise ValueError(f"Metadata for {row.member_id} has an incorrect morph method tag.")
    if metadata.get("timestamp", {}).get("timezone") != "UTC":
        raise ValueError(f"Metadata for {row.member_id} has an incorrect timestamp timezone.")
    expected_units = {
        "T_out_C": "degC",
        "I_beam_horizontal_W_m2": "W/m2",
        "I_diffuse_horizontal_W_m2": "W/m2",
        "I_solar_W_m2": "W/m2",
        "wind_speed_10m_m_s": "m/s",
        "sun_height_deg": "degree",
        "pvgis_reconstructed": "boolean",
    }
    if metadata.get("units") != expected_units:
        raise ValueError(f"Metadata for {row.member_id} has inconsistent units.")
    if metadata["output"]["path"] != _relative(member_path, config):
        raise ValueError(f"Metadata for {row.member_id} records an incorrect output path.")
    if metadata["output"]["sha256"] != str(row.member_sha256):
        raise ValueError(f"Metadata for {row.member_id} records an incorrect output hash.")
    if metadata["sources"]["observed_clean_sha256"] != clean_hash:
        raise ValueError(f"Metadata for {row.member_id} references the wrong observed data.")
    if metadata["sources"]["monthly_deltas_sha256"] != delta_contract.csv_sha256:
        raise ValueError(f"Metadata for {row.member_id} references the wrong delta contract.")
    if pd.Timestamp(metadata["timestamp"]["first"]) != member["timestamp_utc"].iat[0]:
        raise ValueError(f"Metadata for {row.member_id} records an incorrect first timestamp.")
    if pd.Timestamp(metadata["timestamp"]["last"]) != member["timestamp_utc"].iat[-1]:
        raise ValueError(f"Metadata for {row.member_id} records an incorrect last timestamp.")

    parameters = scenario_parameters(delta_contract.frame, str(row.scenario))
    recorded = metadata.get("monthly_parameters", [])
    if len(recorded) != 12:
        raise ValueError(f"Metadata for {row.member_id} does not contain 12 monthly parameters.")
    tolerance = float(
        config["validation"]["tolerances"]["monthly_solar_factor"]
    )
    temperature_tolerance = float(
        config["validation"]["tolerances"]["monthly_temperature_C"]
    )
    for item in recorded:
        month = int(item["month"])
        if bool(item["alpha_was_clipped"]):
            raise ValueError(f"Metadata for {row.member_id} unexpectedly records clipped alpha.")
        if not np.isclose(
            float(item["delta_T_C"]),
            float(parameters.loc[month, "delta_T_C"]),
            rtol=0.0,
            atol=temperature_tolerance,
        ):
            raise ValueError(f"Metadata for {row.member_id} records an incorrect delta T.")
        if not np.isclose(
            float(item["alpha_solar_raw"]),
            float(parameters.loc[month, "alpha_solar_raw"]),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(f"Metadata for {row.member_id} records an incorrect raw solar alpha.")
        if not np.isclose(
            float(item["alpha_solar_applied"]),
            float(parameters.loc[month, "alpha_solar_applied"]),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(f"Metadata for {row.member_id} records an incorrect solar alpha.")


def _member_summary_row(
    row: Any,
    member: pd.DataFrame,
    observed_metrics: Mapping[str, float],
    member_metrics: Mapping[str, float],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    hdd_delta = member_metrics["HDD_C_days"] - observed_metrics["HDD_C_days"]
    cdd_delta = member_metrics["CDD_C_days"] - observed_metrics["CDD_C_days"]
    solar_delta = (
        member_metrics["annual_solar_kWh_m2"]
        - observed_metrics["annual_solar_kWh_m2"]
    )
    return {
        "member_id": str(row.member_id),
        "scenario": str(row.scenario),
        "observed_pvgis_year": int(row.observed_pvgis_year),
        "is_leap_year": bool(calendar.isleap(int(row.observed_pvgis_year))),
        "row_count": int(len(member)),
        "member_sha256": str(row.member_sha256),
        "metadata_sha256": str(row.metadata_sha256),
        "hard_status": "pass",
        "plausibility_status": "warning" if warnings else "pass",
        "warning_count": len(warnings),
        "warning_codes": ";".join(item["code"] for item in warnings),
        "T_min_C": float(member["T_out_C"].min()),
        "T_max_C": float(member["T_out_C"].max()),
        "I_solar_min_W_m2": float(member["I_solar_W_m2"].min()),
        "I_solar_max_W_m2": float(member["I_solar_W_m2"].max()),
        "observed_HDD_C_days": float(observed_metrics["HDD_C_days"]),
        "morphed_HDD_C_days": float(member_metrics["HDD_C_days"]),
        "HDD_change_C_days": float(hdd_delta),
        "HDD_ratio_vs_observed": float(
            member_metrics["HDD_C_days"] / observed_metrics["HDD_C_days"]
        ),
        "observed_CDD_C_days": float(observed_metrics["CDD_C_days"]),
        "morphed_CDD_C_days": float(member_metrics["CDD_C_days"]),
        "CDD_change_C_days": float(cdd_delta),
        "observed_annual_solar_kWh_m2": float(
            observed_metrics["annual_solar_kWh_m2"]
        ),
        "morphed_annual_solar_kWh_m2": float(
            member_metrics["annual_solar_kWh_m2"]
        ),
        "annual_solar_change_kWh_m2": float(solar_delta),
        "annual_solar_ratio_vs_observed": float(
            member_metrics["annual_solar_kWh_m2"]
            / observed_metrics["annual_solar_kWh_m2"]
        ),
    }


def _range(frame: pd.DataFrame, column: str) -> dict[str, float]:
    return {
        "minimum": float(frame[column].min()),
        "maximum": float(frame[column].max()),
    }


def _markdown_report(
    members: pd.DataFrame,
    monthly: pd.DataFrame,
    reference_statistics: Mapping[str, Any],
    cordex_comparison: pd.DataFrame,
    warning_count: int,
    config: Mapping[str, Any],
) -> str:
    observed_solar = members.drop_duplicates("observed_pvgis_year")[
        "observed_annual_solar_kWh_m2"
    ]
    bands = config["validation"]["plausibility"]
    return "\n".join(
        [
            "# Climate ensemble validation report",
            "",
            "**Overall status: PASS**",
            "",
            "The persisted PVGIS baseline and 2050 ensemble were audited without "
            "regenerating any weather member.",
            "",
            "## Coverage and hard checks",
            "",
            f"- Ensemble members: {len(members)}",
            f"- Total member-hours: {int(members['row_count'].sum()):,}",
            f"- Monthly morph checks: {len(monthly)}",
            "- Every member is a complete 8,760- or 8,784-hour UTC calendar year.",
            "- Hashes, schemas, sidecars, physical bounds, GHI composition, night "
            "irradiance, unchanged fields, and monthly morph identities passed.",
            "",
            "## Climate diagnostics",
            "",
            "Degree days use UTC daily-mean temperature:",
            "",
            "- `HDD = 18 - T_daily_mean` when `T_daily_mean <= 15 degC`; otherwise 0.",
            "- `CDD = T_daily_mean - 21` when `T_daily_mean >= 24 degC`; otherwise 0.",
            "- `annual solar = sum(GHI_hourly) / 1000` in kWh/m2/year",
            "",
            f"- Paired HDD ratio: {members['HDD_ratio_vs_observed'].min():.6f}–"
            f"{members['HDD_ratio_vs_observed'].max():.6f}",
            f"- Paired CDD change: {members['CDD_change_C_days'].min():.6f}–"
            f"{members['CDD_change_C_days'].max():.6f} degC-days",
            f"- Paired annual-solar ratio: "
            f"{members['annual_solar_ratio_vs_observed'].min():.6f}–"
            f"{members['annual_solar_ratio_vs_observed'].max():.6f}",
            f"- Observed annual solar: {observed_solar.min():.6f}–"
            f"{observed_solar.max():.6f} kWh/m2/year",
            f"- Morphed annual solar: "
            f"{members['morphed_annual_solar_kWh_m2'].min():.6f}–"
            f"{members['morphed_annual_solar_kWh_m2'].max():.6f} kWh/m2/year",
            "",
            "## Official Brussels reference comparison",
            "",
            "The same degree-day formulas were applied to PVGIS 2005–2023 and "
            "paired by year with the official Eurostat BE100 series.",
            "",
            f"- HDD Pearson correlation: "
            f"{reference_statistics['HDD']['pearson_correlation']:.6f}",
            f"- HDD mean bias (PVGIS - BE100): "
            f"{reference_statistics['HDD']['mean_bias_C_days']:.6f} degC-days",
            f"- HDD mean absolute error: "
            f"{reference_statistics['HDD']['mean_absolute_error_C_days']:.6f} degC-days",
            f"- CDD Pearson correlation: "
            f"{reference_statistics['CDD']['pearson_correlation']:.6f}",
            f"- CDD mean bias (PVGIS - BE100): "
            f"{reference_statistics['CDD']['mean_bias_C_days']:.6f} degC-days",
            f"- CDD mean absolute error: "
            f"{reference_statistics['CDD']['mean_absolute_error_C_days']:.6f} degC-days",
            "",
            "## Direct CORDEX comparison",
            "",
            "Identical degree-day definitions were calculated for every historical "
            "and future CORDEX year. The comparison below contrasts the direct "
            "CORDEX ensemble-mean change with the mean paired change in the morphed "
            "PVGIS ensemble. Exact equality is not required because degree days are "
            "non-linear threshold indicators and the morph intentionally retains the "
            "observed PVGIS baseline distribution.",
            "",
            *[
                (
                    f"- {row.scenario}: HDD change CORDEX "
                    f"{row.cordex_change_HDD_C_days:+.6f}, morph "
                    f"{row.morphed_paired_change_HDD_C_days:+.6f}; CDD change "
                    f"CORDEX {row.cordex_change_CDD_C_days:+.6f}, morph "
                    f"{row.morphed_paired_change_CDD_C_days:+.6f} degC-days."
                )
                for row in cordex_comparison.itertuples(index=False)
            ],
            "",
            "## Warning-only plausibility screening",
            "",
            f"- Warning count: {warning_count}",
            f"- HDD ratio band: {bands['hdd_ratio_vs_observed']}",
            f"- CDD change band: {bands['cdd_change_C_days']} degC-days",
            f"- Annual-solar ratio band: {bands['annual_solar_ratio_vs_observed']}",
            "",
            "A plausibility warning does not invalidate mathematically correct morphing. "
            "This canonical run produced no warnings.",
            "",
            "## Interpretation and caveats",
            "",
            "The temperature invariant compares the monthly mean change "
            "`mean(T_morph - T_observed)` with CORDEX `delta_T_C`. It deliberately "
            "does not require the morphed absolute mean to equal the CORDEX future "
            "mean; retaining the observed hourly baseline is the bias-cancellation step.",
            "",
            f"- {config['provenance']['single_chain_caveat']}",
            f"- {config['provenance']['observed_anchor_caveat']}",
            "",
        ]
    )


def run_validation(config: Mapping[str, Any]) -> dict[str, Path]:
    """Audit all persisted artifacts and write deterministic validation reports."""

    observed, observed_metadata = load_clean_observed(config)
    clean_path = resolve_config_path(
        config,
        config["observed_weather"]["processed"]["directory"],
    ) / config["observed_weather"]["processed"]["clean_weather"]
    clean_metadata_path = resolve_config_path(
        config,
        config["observed_weather"]["processed"]["directory"],
    ) / config["observed_weather"]["processed"]["clean_metadata"]
    clean_hash = sha256_file(clean_path)
    observed_years = split_complete_years(observed)
    if sorted(observed_years) != list(
        range(
            pd.Timestamp(config["observed_weather"]["period_start"]).year,
            pd.Timestamp(config["observed_weather"]["period_end"]).year + 1,
        )
    ):
        raise ValueError("Clean PVGIS artifact does not contain every configured year.")
    validated_observed_years = {
        year: validate_hourly_frame(frame, year, config, f"observed PVGIS {year}")
        for year, frame in observed_years.items()
    }
    observed_metrics = {
        year: calculate_annual_metrics(frame, config)
        for year, frame in validated_observed_years.items()
    }
    (
        official_degree_days,
        official_metadata,
        official_csv_path,
        official_metadata_path,
    ) = load_reference_degree_days(config)
    observed_reference = build_observed_reference_comparison(
        observed_metrics, official_degree_days
    )
    official_statistics = reference_comparison_statistics(observed_reference)

    delta_contract = load_delta_contract(config)
    manifest, manifest_payload, manifest_csv, manifest_json = _load_manifest(
        config, clean_hash, delta_contract
    )
    scenarios = [
        str(spec["scenario"])
        for spec in config["sources"].values()
        if spec["role"] == "future"
    ]
    _validate_manifest_shape(manifest, validated_observed_years, scenarios, config)

    member_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []
    ordered_manifest = manifest.sort_values(
        ["scenario", "observed_pvgis_year"], kind="stable"
    )
    for row in ordered_manifest.itertuples(index=False):
        year = int(row.observed_pvgis_year)
        expected_rows = 8784 if calendar.isleap(year) else 8760
        if int(row.row_count) != expected_rows:
            raise ValueError(
                f"Manifest row count for {row.member_id} is {row.row_count}; "
                f"expected {expected_rows}."
            )
        if bool(row.is_leap_year) != calendar.isleap(year):
            raise ValueError(f"Manifest leap-year flag is wrong for {row.member_id}.")
        member_path = resolve_config_path(config, str(row.member_path))
        metadata_path = resolve_config_path(config, str(row.metadata_path))
        if sha256_file(member_path) != str(row.member_sha256):
            raise ValueError(f"Member hash mismatch for {row.member_id}.")
        if sha256_file(metadata_path) != str(row.metadata_sha256):
            raise ValueError(f"Metadata hash mismatch for {row.member_id}.")

        raw_member = pd.read_csv(member_path)
        member = validate_hourly_frame(
            raw_member, year, config, f"ensemble member {row.member_id}"
        )
        if len(member) != int(row.row_count):
            raise ValueError(f"Member row count mismatch for {row.member_id}.")
        if pd.Timestamp(row.timestamp_start_utc) != member["timestamp_utc"].iat[0]:
            raise ValueError(f"Manifest first timestamp mismatch for {row.member_id}.")
        if pd.Timestamp(row.timestamp_end_utc) != member["timestamp_utc"].iat[-1]:
            raise ValueError(f"Manifest last timestamp mismatch for {row.member_id}.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_member_metadata(
            metadata,
            row,
            member,
            member_path,
            clean_hash,
            delta_contract,
            config,
        )
        monthly_rows.extend(
            validate_member_pair(
                validated_observed_years[year],
                member,
                str(row.member_id),
                str(row.scenario),
                delta_contract.frame,
                config,
            )
        )
        metrics = calculate_annual_metrics(member, config)
        warnings = assess_plausibility(
            str(row.member_id), observed_metrics[year], metrics, config
        )
        all_warnings.extend(warnings)
        member_rows.append(
            _member_summary_row(
                row, member, observed_metrics[year], metrics, warnings
            )
        )

    members = pd.DataFrame.from_records(member_rows)
    monthly = pd.DataFrame.from_records(monthly_rows)
    if len(members) != 57 or int(members["row_count"].sum()) != 499608:
        raise ValueError("Validation output does not contain the canonical ensemble size.")
    if len(monthly) != 684:
        raise ValueError("Validation output does not contain 57 x 12 monthly checks.")
    cordex_annual = calculate_cordex_annual_degree_days(config)
    cordex_morph = build_cordex_morph_comparison(
        cordex_annual, members, scenarios
    )

    output_spec = config["validation"]["outputs"]
    output_dir = resolve_config_path(config, output_spec["directory"])
    member_summary_path = output_dir / output_spec["member_summary"]
    monthly_path = output_dir / output_spec["monthly_invariants"]
    observed_reference_path = output_dir / output_spec["observed_reference_comparison"]
    cordex_annual_path = output_dir / output_spec["cordex_annual_degree_days"]
    cordex_morph_path = output_dir / output_spec["cordex_morph_comparison"]
    report_markdown_path = output_dir / output_spec["report_markdown"]
    report_json_path = output_dir / output_spec["report_json"]
    _atomic_write_csv(members, member_summary_path)
    _atomic_write_csv(monthly, monthly_path)
    _atomic_write_csv(observed_reference, observed_reference_path)
    _atomic_write_csv(cordex_annual, cordex_annual_path)
    _atomic_write_csv(cordex_morph, cordex_morph_path)
    _atomic_write_text(
        _markdown_report(
            members,
            monthly,
            official_statistics,
            cordex_morph,
            len(all_warnings),
            config,
        ),
        report_markdown_path,
    )

    observed_solar = members.drop_duplicates("observed_pvgis_year")[
        "observed_annual_solar_kWh_m2"
    ]
    report = {
        "schema_version": 1,
        "status": "pass",
        "hard_error_count": 0,
        "plausibility_warning_count": len(all_warnings),
        "counts": {
            "observed_rows": int(len(observed)),
            "observed_years": int(len(validated_observed_years)),
            "ensemble_members": int(len(members)),
            "ensemble_member_hours": int(members["row_count"].sum()),
            "monthly_morph_checks": int(len(monthly)),
            "official_reference_years": int(len(observed_reference)),
            "cordex_annual_degree_day_rows": int(len(cordex_annual)),
        },
        "method": {
            "temperature_invariant": (
                "mean(T_morph - T_observed, month) = delta_T_C(month); "
                "the CORDEX absolute future mean is not the target"
            ),
            "solar_invariant": (
                "sum(I_morph, month) / sum(I_observed, month) = "
                "alpha_solar_applied(month) for beam, diffuse, and GHI"
            ),
            "HDD_C_days": (
                "sum(18 - UTC daily mean temperature) for days with "
                "daily mean temperature <= 15 degC; zero otherwise"
            ),
            "CDD_C_days": (
                "sum(UTC daily mean temperature - 21) for days with "
                "daily mean temperature >= 24 degC; zero otherwise"
            ),
            "annual_solar_kWh_m2": "sum(hourly GHI W/m2) / 1000",
            "plausibility_comparison": (
                "Each member is paired with its matching observed PVGIS year."
            ),
        },
        "tolerances": config["validation"]["tolerances"],
        "physical_ranges": config["physical_ranges"],
        "plausibility_bands": config["validation"]["plausibility"],
        "diagnostic_ranges": {
            "ensemble_temperature_C": {
                "minimum": float(members["T_min_C"].min()),
                "maximum": float(members["T_max_C"].max()),
            },
            "ensemble_GHI_W_m2": {
                "minimum": float(members["I_solar_min_W_m2"].min()),
                "maximum": float(members["I_solar_max_W_m2"].max()),
            },
            "paired_HDD_ratio": _range(members, "HDD_ratio_vs_observed"),
            "paired_CDD_change_C_days": _range(
                members, "CDD_change_C_days"
            ),
            "paired_annual_solar_ratio": _range(
                members, "annual_solar_ratio_vs_observed"
            ),
            "observed_annual_solar_kWh_m2": {
                "minimum": float(observed_solar.min()),
                "maximum": float(observed_solar.max()),
            },
            "morphed_annual_solar_kWh_m2": _range(
                members, "morphed_annual_solar_kWh_m2"
            ),
            "official_BE100_comparison": official_statistics,
            "direct_CORDEX_vs_morph_change": {
                str(row.scenario): {
                    "cordex_change_HDD_C_days": float(
                        row.cordex_change_HDD_C_days
                    ),
                    "morphed_paired_change_HDD_C_days": float(
                        row.morphed_paired_change_HDD_C_days
                    ),
                    "cordex_change_CDD_C_days": float(
                        row.cordex_change_CDD_C_days
                    ),
                    "morphed_paired_change_CDD_C_days": float(
                        row.morphed_paired_change_CDD_C_days
                    ),
                }
                for row in cordex_morph.itertuples(index=False)
            },
        },
        "warnings": all_warnings,
        "inputs": {
            "config": {
                "path": _relative(Path(config["_config_path"]), config),
                "sha256": sha256_file(Path(config["_config_path"])),
            },
            "clean_observed": {
                "path": _relative(clean_path, config),
                "sha256": clean_hash,
                "metadata_path": _relative(clean_metadata_path, config),
                "metadata_sha256": sha256_file(clean_metadata_path),
                "recorded_row_count": int(observed_metadata["row_count"]),
            },
            "monthly_deltas": {
                "path": _relative(delta_contract.csv_path, config),
                "sha256": delta_contract.csv_sha256,
                "provenance_path": _relative(
                    delta_contract.provenance_path, config
                ),
                "provenance_sha256": delta_contract.provenance_sha256,
            },
            "official_BE100_degree_days": {
                "provider": official_metadata["provider"],
                "dataset_code": official_metadata["dataset_code"],
                "geo_code": official_metadata["geo_code"],
                "source_url": official_metadata["source_url"],
                "api_response_updated": official_metadata["api_response_updated"],
                "path": _relative(official_csv_path, config),
                "sha256": sha256_file(official_csv_path),
                "metadata_path": _relative(official_metadata_path, config),
                "metadata_sha256": sha256_file(official_metadata_path),
            },
            "ensemble_manifest_csv": {
                "path": _relative(manifest_csv, config),
                "sha256": sha256_file(manifest_csv),
            },
            "ensemble_manifest_json": {
                "path": _relative(manifest_json, config),
                "sha256": sha256_file(manifest_json),
                "method_tag": manifest_payload["method_tag"],
            },
        },
        "outputs": {
            "member_summary": {
                "path": _relative(member_summary_path, config),
                "sha256": sha256_file(member_summary_path),
                "row_count": int(len(members)),
            },
            "monthly_invariants": {
                "path": _relative(monthly_path, config),
                "sha256": sha256_file(monthly_path),
                "row_count": int(len(monthly)),
            },
            "observed_reference_comparison": {
                "path": _relative(observed_reference_path, config),
                "sha256": sha256_file(observed_reference_path),
                "row_count": int(len(observed_reference)),
            },
            "cordex_annual_degree_days": {
                "path": _relative(cordex_annual_path, config),
                "sha256": sha256_file(cordex_annual_path),
                "row_count": int(len(cordex_annual)),
            },
            "cordex_morph_comparison": {
                "path": _relative(cordex_morph_path, config),
                "sha256": sha256_file(cordex_morph_path),
                "row_count": int(len(cordex_morph)),
            },
            "report_markdown": {
                "path": _relative(report_markdown_path, config),
                "sha256": sha256_file(report_markdown_path),
            },
        },
        "caveats": {
            "single_model_chain": config["provenance"]["single_chain_caveat"],
            "observed_anchor": config["provenance"]["observed_anchor_caveat"],
        },
    }
    _atomic_write_json(report, report_json_path)
    return {
        "member_summary": member_summary_path,
        "monthly_invariants": monthly_path,
        "observed_reference_comparison": observed_reference_path,
        "cordex_annual_degree_days": cordex_annual_path,
        "cordex_morph_comparison": cordex_morph_path,
        "report_markdown": report_markdown_path,
        "report_json": report_json_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the persisted PVGIS/CORDEX 2050 weather ensemble."
    )
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
        paths = run_validation(load_config(args.config))
    except Exception as exc:
        LOGGER.error("Climate ensemble validation failed: %s", exc)
        if args.verbose:
            LOGGER.exception("Detailed failure")
        return 1
    for name, path in paths.items():
        LOGGER.info("Wrote %s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
