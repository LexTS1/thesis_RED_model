"""Compute monthly CORDEX climatologies, climate deltas, and yearly variability."""

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

from .load_cordex import (
    CordexSource,
    load_all_sources,
    load_config,
    resolve_config_path,
    sha256_file,
)


LOGGER = logging.getLogger("climate.deltas")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def compute_monthly_climatologies(
    sources: Mapping[str, CordexSource], config: Mapping[str, Any]
) -> pd.DataFrame:
    """Return day-weighted calendar-month means for every configured source."""

    records: list[dict[str, Any]] = []
    for key, source in sources.items():
        spec = config["sources"][key]
        grouped = source.frame.groupby("month", sort=True).agg(
            n_days=("T_out_C", "size"),
            T_mean_C=("T_out_C", "mean"),
            I_solar_mean_W_m2=("I_solar_W_m2", "mean"),
        )
        if grouped.index.tolist() != list(range(1, 13)):
            raise ValueError(f"Source {key!r} does not contain all twelve calendar months.")
        for month, values in grouped.iterrows():
            records.append(
                {
                    "source_key": key,
                    "scenario": source.scenario,
                    "window": source.window,
                    "period_start": str(spec["period_start"]),
                    "period_end": str(spec["period_end"]),
                    "n_years": int(source.frame["year"].nunique()),
                    "month": int(month),
                    "month_name": calendar.month_abbr[int(month)],
                    "n_days": int(values["n_days"]),
                    "T_mean_C": float(values["T_mean_C"]),
                    "I_solar_mean_W_m2": float(values["I_solar_mean_W_m2"]),
                }
            )
    return pd.DataFrame.from_records(records)


def apply_solar_alpha_safety(
    alpha_raw: float,
    scenario: str,
    month: int,
    minimum: float,
    maximum: float,
    logger: logging.Logger = LOGGER,
) -> tuple[float, bool]:
    """Apply the loose solar-factor safety net and warn if it activates."""

    alpha_applied = float(np.clip(float(alpha_raw), float(minimum), float(maximum)))
    was_clipped = bool(alpha_raw < minimum or alpha_raw > maximum)
    if was_clipped:
        logger.warning(
            "Solar alpha safety clamp activated: scenario=%s month=%02d "
            "alpha_raw=%.10f alpha_applied=%.10f bounds=[%.3f, %.3f]",
            scenario,
            month,
            alpha_raw,
            alpha_applied,
            minimum,
            maximum,
        )
    return alpha_applied, was_clipped


def compute_monthly_deltas(
    climatologies: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Compute one monthly temperature shift and solar factor per future RCP."""

    baseline_key = str(config["baseline_source"])
    historical = climatologies.loc[
        climatologies["source_key"] == baseline_key
    ].set_index("month")
    if len(historical) != 12:
        raise ValueError("The configured historical baseline does not contain twelve months.")

    minimum = float(config["solar_alpha_safety"]["minimum"])
    maximum = float(config["solar_alpha_safety"]["maximum"])
    records: list[dict[str, Any]] = []
    for key, spec in config["sources"].items():
        if spec["role"] != "future":
            continue
        future = climatologies.loc[climatologies["source_key"] == key].set_index("month")
        if len(future) != 12:
            raise ValueError(f"Future source {key!r} does not contain twelve months.")
        for month in range(1, 13):
            hist_row = historical.loc[month]
            future_row = future.loc[month]
            hist_solar = float(hist_row["I_solar_mean_W_m2"])
            if hist_solar <= 0.0:
                raise ValueError(f"Historical solar climatology is non-positive in month {month}.")
            alpha_raw = float(future_row["I_solar_mean_W_m2"]) / hist_solar
            alpha_applied, was_clipped = apply_solar_alpha_safety(
                alpha_raw,
                scenario=str(spec["scenario"]),
                month=month,
                minimum=minimum,
                maximum=maximum,
            )
            records.append(
                {
                    "scenario": str(spec["scenario"]),
                    "month": month,
                    "month_name": calendar.month_abbr[month],
                    "n_hist_days": int(hist_row["n_days"]),
                    "n_future_days": int(future_row["n_days"]),
                    "T_hist_C": float(hist_row["T_mean_C"]),
                    "T_future_C": float(future_row["T_mean_C"]),
                    "delta_T_C": float(future_row["T_mean_C"] - hist_row["T_mean_C"]),
                    "I_hist_W_m2": hist_solar,
                    "I_future_W_m2": float(future_row["I_solar_mean_W_m2"]),
                    "alpha_solar_raw": alpha_raw,
                    "alpha_solar_applied": alpha_applied,
                    "alpha_was_clipped": was_clipped,
                }
            )
    return pd.DataFrame.from_records(records)


def compute_year_month_variability(
    sources: Mapping[str, CordexSource],
    climatologies: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Retain future year-month anomalies alongside the scenario climatologies."""

    baseline_key = str(config["baseline_source"])
    historical = climatologies.loc[
        climatologies["source_key"] == baseline_key
    ].set_index("month")
    records: list[dict[str, Any]] = []

    for key, source in sources.items():
        spec = config["sources"][key]
        if spec["role"] != "future":
            continue
        future_climatology = climatologies.loc[
            climatologies["source_key"] == key
        ].set_index("month")
        grouped = source.frame.groupby(["year", "month"], sort=True).agg(
            n_days=("T_out_C", "size"),
            T_future_year_month_C=("T_out_C", "mean"),
            I_future_year_month_W_m2=("I_solar_W_m2", "mean"),
        )
        expected_rows = int(spec["expected_years"]) * 12
        if len(grouped) != expected_rows:
            raise ValueError(
                f"Future source {key!r} has {len(grouped)} year-month groups; "
                f"expected {expected_rows}."
            )

        for (year, month), values in grouped.iterrows():
            future_row = future_climatology.loc[int(month)]
            hist_row = historical.loc[int(month)]
            future_solar = float(future_row["I_solar_mean_W_m2"])
            hist_solar = float(hist_row["I_solar_mean_W_m2"])
            year_solar = float(values["I_future_year_month_W_m2"])
            records.append(
                {
                    "scenario": str(spec["scenario"]),
                    "climate_year": int(year),
                    "month": int(month),
                    "month_name": calendar.month_abbr[int(month)],
                    "n_days": int(values["n_days"]),
                    "T_future_year_month_C": float(values["T_future_year_month_C"]),
                    "T_future_climatology_C": float(future_row["T_mean_C"]),
                    "T_anomaly_from_future_climatology_C": float(
                        values["T_future_year_month_C"] - future_row["T_mean_C"]
                    ),
                    "delta_T_vs_historical_C": float(
                        values["T_future_year_month_C"] - hist_row["T_mean_C"]
                    ),
                    "I_future_year_month_W_m2": year_solar,
                    "I_future_climatology_W_m2": future_solar,
                    "solar_factor_from_future_climatology": year_solar / future_solar,
                    "alpha_solar_vs_historical": year_solar / hist_solar,
                }
            )
    return pd.DataFrame.from_records(records)


def _relative_to_config(path: Path, config: Mapping[str, Any]) -> str:
    try:
        return str(path.relative_to(Path(config["_base_dir"])))
    except ValueError:
        return str(path)


def _annual_diagnostics(
    sources: Mapping[str, CordexSource], config: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    baseline = sources[str(config["baseline_source"])].frame
    output: dict[str, dict[str, float]] = {}
    for key, source in sources.items():
        if config["sources"][key]["role"] != "future":
            continue
        output[source.scenario] = {
            "historical_temperature_mean_C": float(baseline["T_out_C"].mean()),
            "future_temperature_mean_C": float(source.frame["T_out_C"].mean()),
            "temperature_delta_C": float(
                source.frame["T_out_C"].mean() - baseline["T_out_C"].mean()
            ),
            "historical_solar_mean_W_m2": float(baseline["I_solar_W_m2"].mean()),
            "future_solar_mean_W_m2": float(source.frame["I_solar_W_m2"].mean()),
            "solar_ratio": float(
                source.frame["I_solar_W_m2"].mean()
                / baseline["I_solar_W_m2"].mean()
            ),
        }
    return output


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    frame.to_csv(temporary, index=False, float_format="%.10f", lineterminator="\n")
    temporary.replace(path)


def write_outputs(
    climatologies: pd.DataFrame,
    deltas: pd.DataFrame,
    variability: pd.DataFrame,
    sources: Mapping[str, CordexSource],
    config: Mapping[str, Any],
) -> dict[str, Path]:
    """Write deterministic tables and a provenance manifest with output hashes."""

    output_dir = resolve_config_path(config, config["outputs"]["directory"])
    output_paths = {
        "monthly_climatologies": output_dir / config["outputs"]["monthly_climatologies"],
        "monthly_deltas": output_dir / config["outputs"]["monthly_deltas"],
        "year_month_variability": output_dir / config["outputs"]["year_month_variability"],
    }
    _atomic_write_csv(climatologies, output_paths["monthly_climatologies"])
    _atomic_write_csv(deltas, output_paths["monthly_deltas"])
    _atomic_write_csv(variability, output_paths["year_month_variability"])

    provenance_path = output_dir / config["outputs"]["provenance"]
    source_manifest: dict[str, Any] = {}
    for key, source in sources.items():
        source_manifest[key] = {
            "scenario": source.scenario,
            "window": source.window,
            "period_start": str(config["sources"][key]["period_start"]),
            "period_end": str(config["sources"][key]["period_end"]),
            "row_count": int(len(source.frame)),
            "year_count": int(source.frame["year"].nunique()),
            "csv": _relative_to_config(source.csv_path, config),
            "csv_sha256": source.csv_sha256,
            "metadata": _relative_to_config(source.metadata_path, config),
            "metadata_sha256": source.metadata_sha256,
            "raw_cordex_archives": source.metadata.get("source_files", []),
        }

    alpha_min_index = deltas["alpha_solar_raw"].idxmin()
    alpha_max_index = deltas["alpha_solar_raw"].idxmax()
    alpha_min_row = deltas.loc[alpha_min_index]
    alpha_max_row = deltas.loc[alpha_max_index]
    manifest = {
        "schema_version": 1,
        "dataset": config["dataset"],
        "model_chain": config["model_chain"],
        "spatial_extraction": config["spatial_extraction"],
        "climate_target": {
            "interpretation": "2050 climate represented by complete calendar years 2050-2070",
            "future_year_count": 21,
            "historical_period": "1981-2005",
            "future_period": "2050-2070",
        },
        "method": {
            "monthly_climatology": "day-weighted arithmetic mean grouped by calendar month",
            "temperature_delta": "T_future_month_mean - T_historical_month_mean",
            "solar_alpha_raw": "I_future_month_mean / I_historical_month_mean",
            "solar_alpha_applied": "clip(alpha_solar_raw, 0.7, 1.3)",
            "temperature_variance_stretch": "not calculated",
        },
        "solar_alpha_safety": {
            "minimum": float(config["solar_alpha_safety"]["minimum"]),
            "maximum": float(config["solar_alpha_safety"]["maximum"]),
            "clipped_month_count": int(deltas["alpha_was_clipped"].sum()),
            "observed_minimum": {
                "scenario": str(alpha_min_row["scenario"]),
                "month": int(alpha_min_row["month"]),
                "value": float(alpha_min_row["alpha_solar_raw"]),
            },
            "observed_maximum": {
                "scenario": str(alpha_max_row["scenario"]),
                "month": int(alpha_max_row["month"]),
                "value": float(alpha_max_row["alpha_solar_raw"]),
            },
        },
        "morphing_contract": {
            "temperature_shift": "monthly_deltas_2050.csv column delta_T_C",
            "solar_factor": "monthly_deltas_2050.csv column alpha_solar_applied",
            "solar_factor_observed_range": [
                float(alpha_min_row["alpha_solar_raw"]),
                float(alpha_max_row["alpha_solar_raw"]),
            ],
            "year_month_variability_role": (
                "Inter-annual diagnostic and future sampling summary; its year-month "
                "solar ratios are not the morph alpha and are not safety-clipped."
            ),
        },
        "sources": source_manifest,
        "outputs": {
            name: {
                "path": _relative_to_config(path, config),
                "row_count": int(
                    {
                        "monthly_climatologies": len(climatologies),
                        "monthly_deltas": len(deltas),
                        "year_month_variability": len(variability),
                    }[name]
                ),
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
        },
        "annual_day_weighted_diagnostics": _annual_diagnostics(sources, config),
        "caveats": {
            "single_model_chain": config["provenance"]["single_chain_caveat"],
            "observed_anchor": config["provenance"]["observed_anchor_caveat"],
            "hourly_downscaling": (
                "Implemented downstream as monthly PVGIS morphing in "
                "climate.src.build_ensemble; demand-model interface integration remains deferred."
            ),
        },
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = provenance_path.with_name(f".{provenance_path.name}.writing")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(provenance_path)
    output_paths["provenance"] = provenance_path
    return output_paths


def build(config: Mapping[str, Any]) -> dict[str, Path]:
    """Run the complete climatology and delta build."""

    sources = load_all_sources(config)
    climatologies = compute_monthly_climatologies(sources, config)
    deltas = compute_monthly_deltas(climatologies, config)
    variability = compute_year_month_variability(sources, climatologies, config)

    if len(climatologies) != 48 or len(deltas) != 36 or len(variability) != 756:
        raise ValueError(
            "Unexpected output dimensions: "
            f"climatologies={len(climatologies)}, deltas={len(deltas)}, "
            f"variability={len(variability)}"
        )
    return write_outputs(climatologies, deltas, variability, sources, config)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build monthly CORDEX climatologies and 2050 climate deltas."
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
        config = load_config(args.config)
        paths = build(config)
    except Exception as exc:
        LOGGER.error("Climate delta build failed: %s", exc)
        if args.verbose:
            LOGGER.exception("Detailed failure")
        return 1

    LOGGER.info("Climate delta build completed without input or continuity errors.")
    for name, path in paths.items():
        LOGGER.info("Wrote %s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
