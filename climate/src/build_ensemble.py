"""Build the deterministic 57-member hourly 2050 weather ensemble."""

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

from .load_cordex import load_config, resolve_config_path, sha256_file
from .load_observed import (
    build_clean_observed,
    load_clean_observed,
    split_complete_years,
)
from .morph import DeltaContract, load_delta_contract, morph_observed_year, scenario_parameters


LOGGER = logging.getLogger("climate.build_ensemble")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _relative(path: Path, config: Mapping[str, Any]) -> str:
    try:
        return str(path.relative_to(Path(config["_base_dir"])))
    except ValueError:
        return str(path)


def _format_utc(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_csv(frame: pd.DataFrame, path: Path, timestamp: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    serializable = frame.copy()
    if timestamp:
        serializable["timestamp_utc"] = _format_utc(serializable["timestamp_utc"])
    serializable.to_csv(temporary, index=False, float_format="%.10f", lineterminator="\n")
    temporary.replace(path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _monthly_parameters(parameters: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "month": int(month),
            "delta_T_C": float(row["delta_T_C"]),
            "alpha_solar_raw": float(row["alpha_solar_raw"]),
            "alpha_solar_applied": float(row["alpha_solar_applied"]),
            "alpha_was_clipped": bool(row["alpha_was_clipped"]),
        }
        for month, row in parameters.iterrows()
    ]


def validate_morphed_member(
    observed: pd.DataFrame,
    morphed: pd.DataFrame,
    scenario: str,
    deltas: pd.DataFrame,
) -> None:
    """Validate all morph invariants for one observed-year/scenario pair."""

    if not pd.DatetimeIndex(morphed["timestamp_utc"]).equals(
        pd.DatetimeIndex(observed["timestamp_utc"])
    ):
        raise ValueError("Morphed member timestamps drifted from the observed year.")
    parameters = scenario_parameters(deltas, scenario)
    months = observed["timestamp_utc"].dt.month
    for month in range(1, 13):
        mask = months == month
        expected_delta = float(parameters.loc[month, "delta_T_C"])
        actual_delta = float((morphed.loc[mask, "T_out_C"] - observed.loc[mask, "T_out_C"]).mean())
        if not np.isclose(actual_delta, expected_delta, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Temperature morph invariant failed for {scenario} month {month}: "
                f"expected {expected_delta}, got {actual_delta}."
            )
        expected_alpha = float(parameters.loc[month, "alpha_solar_applied"])
        for column in (
            "I_beam_horizontal_W_m2",
            "I_diffuse_horizontal_W_m2",
            "I_solar_W_m2",
        ):
            denominator = float(observed.loc[mask, column].sum())
            if denominator <= 0.0:
                continue
            actual_alpha = float(morphed.loc[mask, column].sum()) / denominator
            if not np.isclose(actual_alpha, expected_alpha, rtol=0.0, atol=2e-12):
                raise ValueError(
                    f"Solar morph invariant failed for {scenario} month {month} "
                    f"column {column}: expected {expected_alpha}, got {actual_alpha}."
                )


def _member_metadata(
    member_id: str,
    scenario: str,
    observed_year: int,
    member: pd.DataFrame,
    parameters: pd.DataFrame,
    member_path: Path,
    clean_metadata: Mapping[str, Any],
    delta_contract: DeltaContract,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ensemble = config["observed_weather"]["ensemble"]
    return {
        "schema_version": 1,
        "member_id": member_id,
        "scenario": scenario,
        "climate_target": str(ensemble["climate_target"]),
        "observed_pvgis_year": int(observed_year),
        "is_leap_year": bool(calendar.isleap(observed_year)),
        "row_count": int(len(member)),
        "timestamp": {
            "timezone": "UTC",
            "normalization": "PVGIS :10 source stamps floored to top of hour",
            "first": member["timestamp_utc"].iat[0].isoformat(),
            "last": member["timestamp_utc"].iat[-1].isoformat(),
        },
        "method": {
            "tag": str(ensemble["method_tag"]),
            "temperature": "T_morph = T_observed + delta_T_C(month)",
            "beam": "beam_morph = beam_observed * alpha_solar_applied(month)",
            "diffuse": "diffuse_morph = diffuse_observed * alpha_solar_applied(month)",
            "ghi": "I_solar_W_m2 = morphed beam + morphed diffuse",
            "unchanged": [
                "timestamp_utc",
                "wind_speed_10m_m_s",
                "sun_height_deg",
                "pvgis_reconstructed",
            ],
        },
        "monthly_parameters": _monthly_parameters(parameters),
        "sources": {
            "observed_raw_path": clean_metadata["source"]["path"],
            "observed_raw_sha256": clean_metadata["source"]["sha256"],
            "observed_clean_path": clean_metadata["output"]["path"],
            "observed_clean_sha256": clean_metadata["output"]["sha256"],
            "monthly_deltas_path": _relative(delta_contract.csv_path, config),
            "monthly_deltas_sha256": delta_contract.csv_sha256,
            "delta_provenance_path": _relative(delta_contract.provenance_path, config),
            "delta_provenance_sha256": delta_contract.provenance_sha256,
        },
        "columns": list(member.columns),
        "units": clean_metadata["units"],
        "statistics": {
            "T_out_C": {
                "minimum": float(member["T_out_C"].min()),
                "maximum": float(member["T_out_C"].max()),
                "mean": float(member["T_out_C"].mean()),
            },
            "I_solar_W_m2": {
                "minimum": float(member["I_solar_W_m2"].min()),
                "maximum": float(member["I_solar_W_m2"].max()),
                "mean": float(member["I_solar_W_m2"].mean()),
            },
        },
        "output": {
            "path": _relative(member_path, config),
            "sha256": sha256_file(member_path),
        },
        "facade_forcing": (
            "Not materialized in this member; use climate.src.transpose_facades "
            "with the aligned PVGIS facade templates."
        ),
        "caveats": {
            "single_model_chain": config["provenance"]["single_chain_caveat"],
            "observed_anchor": config["provenance"]["observed_anchor_caveat"],
            "weather_axis": "One member for each complete observed PVGIS year 2005-2023.",
        },
    }


def build_ensemble(config: Mapping[str, Any]) -> dict[str, Path]:
    """Clean PVGIS and build all 3 RCP x 19 observed-year ensemble members."""

    clean_paths = build_clean_observed(config)
    observed, clean_metadata = load_clean_observed(config)
    observed_years = split_complete_years(observed)
    delta_contract = load_delta_contract(config)
    scenarios = [
        str(spec["scenario"])
        for spec in config["sources"].values()
        if spec["role"] == "future"
    ]
    ensemble = config["observed_weather"]["ensemble"]
    output_root = resolve_config_path(config, ensemble["directory"])
    manifest_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        parameters = scenario_parameters(delta_contract.frame, scenario)
        for observed_year, observed_frame in observed_years.items():
            member = morph_observed_year(
                observed_frame, scenario, delta_contract.frame, config=config
            )
            validate_morphed_member(observed_frame, member, scenario, delta_contract.frame)
            member_id = f"weather_2050_{scenario}_pvgis_{observed_year}"
            member_path = output_root / scenario / f"{member_id}.csv"
            metadata_path = member_path.with_suffix(".metadata.json")
            _atomic_write_csv(member, member_path)
            metadata = _member_metadata(
                member_id,
                scenario,
                observed_year,
                member,
                parameters,
                member_path,
                clean_metadata,
                delta_contract,
                config,
            )
            _atomic_write_json(metadata, metadata_path)
            manifest_rows.append(
                {
                    "member_id": member_id,
                    "scenario": scenario,
                    "observed_pvgis_year": int(observed_year),
                    "climate_target": str(ensemble["climate_target"]),
                    "is_leap_year": bool(calendar.isleap(observed_year)),
                    "row_count": int(len(member)),
                    "timestamp_start_utc": member["timestamp_utc"].iat[0].isoformat(),
                    "timestamp_end_utc": member["timestamp_utc"].iat[-1].isoformat(),
                    "member_path": _relative(member_path, config),
                    "member_sha256": sha256_file(member_path),
                    "metadata_path": _relative(metadata_path, config),
                    "metadata_sha256": sha256_file(metadata_path),
                    "T_min_C": float(member["T_out_C"].min()),
                    "T_max_C": float(member["T_out_C"].max()),
                    "T_mean_C": float(member["T_out_C"].mean()),
                    "I_solar_min_W_m2": float(member["I_solar_W_m2"].min()),
                    "I_solar_max_W_m2": float(member["I_solar_W_m2"].max()),
                    "I_solar_mean_W_m2": float(member["I_solar_W_m2"].mean()),
                }
            )

    manifest = pd.DataFrame.from_records(manifest_rows)
    expected_members = len(scenarios) * int(config["observed_weather"]["expected_years"])
    if len(manifest) != expected_members:
        raise ValueError(f"Built {len(manifest)} members; expected {expected_members}.")
    expected_hours = len(observed) * len(scenarios)
    if int(manifest["row_count"].sum()) != expected_hours:
        raise ValueError(
            f"Built {int(manifest['row_count'].sum())} member-hours; expected {expected_hours}."
        )

    manifest_csv = output_root / ensemble["manifest_csv"]
    _atomic_write_csv(manifest, manifest_csv, timestamp=False)
    manifest_json = output_root / ensemble["manifest_json"]
    payload = {
        "schema_version": 1,
        "method_tag": str(ensemble["method_tag"]),
        "climate_target": str(ensemble["climate_target"]),
        "member_count": int(len(manifest)),
        "total_member_hours": int(manifest["row_count"].sum()),
        "scenarios": scenarios,
        "observed_pvgis_years": sorted(int(year) for year in observed_years),
        "leap_years": sorted(int(year) for year in observed_years if calendar.isleap(year)),
        "morph_contract": {
            "path": _relative(delta_contract.csv_path, config),
            "sha256": delta_contract.csv_sha256,
            "alpha_minimum": float(delta_contract.frame["alpha_solar_applied"].min()),
            "alpha_maximum": float(delta_contract.frame["alpha_solar_applied"].max()),
            "clipped_month_count": int(delta_contract.frame["alpha_was_clipped"].sum()),
        },
        "observed_source": clean_metadata["source"],
        "clean_observed": clean_metadata["output"],
        "manifest_csv": {
            "path": _relative(manifest_csv, config),
            "sha256": sha256_file(manifest_csv),
        },
        "ensemble_ranges": {
            "T_out_C": {
                "minimum": float(manifest["T_min_C"].min()),
                "maximum": float(manifest["T_max_C"].max()),
            },
            "I_solar_W_m2": {
                "minimum": float(manifest["I_solar_min_W_m2"].min()),
                "maximum": float(manifest["I_solar_max_W_m2"].max()),
            },
        },
        "facade_forcing": {
            "delivery": "on_demand",
            "module": "climate.src.transpose_facades",
            "orientations": ["south", "east", "west", "north"],
            "materialized_in_members": False,
            "sources": {
                orientation: {
                    "path": spec["csv"],
                    "sha256": spec["csv_sha256"],
                    "slope_deg": int(spec["slope_deg"]),
                    "azimuth_pvgis_deg": int(spec["azimuth_pvgis_deg"]),
                }
                for orientation, spec in config["observed_weather"]["facades"].items()
            },
        },
        "members": manifest_rows,
    }
    _atomic_write_json(payload, manifest_json)
    return {
        **clean_paths,
        "manifest_csv": manifest_csv,
        "manifest_json": manifest_json,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 57-member hourly PVGIS/CORDEX 2050 ensemble."
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
        paths = build_ensemble(load_config(args.config))
    except Exception as exc:
        LOGGER.error("Ensemble build failed: %s", exc)
        if args.verbose:
            LOGGER.exception("Detailed failure")
        return 1
    for name, path in paths.items():
        LOGGER.info("Wrote %s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
