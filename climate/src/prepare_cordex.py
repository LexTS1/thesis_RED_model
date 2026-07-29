"""Validate raw CDS CORDEX NetCDF files and extract canonical daily point series."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import netCDF4
import numpy as np
import pandas as pd

from .load_cordex import load_config, resolve_config_path, sha256_file


LOGGER = logging.getLogger("climate.prepare_cordex")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _relative(path: Path, config: Mapping[str, Any]) -> str:
    try:
        return str(path.relative_to(Path(config["_base_dir"])))
    except ValueError:
        return str(path)


def _verify_file(
    config: Mapping[str, Any], path_value: str, expected_hash: str, label: str
) -> tuple[Path, str]:
    path = resolve_config_path(config, path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected_hash):
        raise ValueError(
            f"SHA-256 mismatch for {label} {path}: expected {expected_hash}, got {actual}"
        )
    return path, actual


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    frame.to_csv(temporary, index=False, float_format="%.10f", lineterminator="\n")
    temporary.replace(path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _scenario_id(scenario: str) -> str:
    mapping = {"rcp_2_6": "rcp26", "rcp_4_5": "rcp45", "rcp_8_5": "rcp85"}
    try:
        return mapping[scenario]
    except KeyError as exc:
        raise ValueError(f"Unsupported CORDEX scenario {scenario!r}") from exc


def _provenance_summary(payload: Mapping[str, Any], output_name: str) -> dict[str, Any]:
    activities = payload.get("activity", {})
    orchestrations = [
        value for value in activities.values() if value.get("prov:label") == "orchestrate"
    ]
    subsets = [
        value for value in activities.values() if str(value.get("prov:label", "")).startswith("subset_")
    ]
    entities = payload.get("entity", {})
    output_entities = [
        key for key, value in entities.items() if value.get("prov:label") == output_name
    ]
    if len(orchestrations) != 1 or len(subsets) != 1 or len(output_entities) != 1:
        raise ValueError(f"CDS provenance does not uniquely describe {output_name}")

    derived = payload.get("wasDerivedFrom", {})
    source_entities = [
        relation.get("prov:usedEntity")
        for relation in derived.values()
        if relation.get("prov:generatedEntity") == output_entities[0]
    ]
    if len(source_entities) != 1:
        raise ValueError(f"CDS provenance has no unique upstream catalogue entity for {output_name}")

    agents = payload.get("agent", {})
    software = sorted(
        value.get("prov:label")
        for value in agents.values()
        if value.get("prov:type", {}).get("$") == "prov:SoftwareAgent"
    )
    orchestration = orchestrations[0]
    subset = subsets[0]
    return {
        "catalogue_entity": str(source_entities[0]),
        "workflow_start": str(orchestration.get("prov:startTime")),
        "workflow_end": str(orchestration.get("prov:endTime")),
        "subset_time": str(subset.get("roocs:time")),
        "subset_time_components": str(subset.get("roocs:time_components")),
        "subset_area": str(subset.get("roocs:area")),
        "apply_fixes": bool(subset.get("roocs:apply_fixes")),
        "software_agents": software,
    }


def _read_variable(
    config: Mapping[str, Any], spec: Mapping[str, Any], source_key: str, raw_key: str
) -> dict[str, Any]:
    raw_spec = spec["raw"][raw_key]
    variable_name = config["dataset"]["variables"][raw_key]
    netcdf_path, netcdf_hash = _verify_file(
        config,
        raw_spec["netcdf"],
        raw_spec["netcdf_sha256"],
        f"{source_key} {variable_name} NetCDF",
    )
    provenance_path, provenance_hash = _verify_file(
        config,
        raw_spec["provenance_json"],
        raw_spec["provenance_json_sha256"],
        f"{source_key} {variable_name} CDS provenance JSON",
    )
    image_path, image_hash = _verify_file(
        config,
        raw_spec["provenance_image"],
        raw_spec["provenance_image_sha256"],
        f"{source_key} {variable_name} CDS provenance image",
    )
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance_payload = json.load(handle)
    provenance = _provenance_summary(provenance_payload, netcdf_path.name)

    with netCDF4.Dataset(netcdf_path) as dataset:
        expected_globals = {
            "project_id": "CORDEX",
            "CORDEX_domain": str(config["dataset"]["domain_id"]),
            "driving_model_id": "CNRM-CERFACS-CNRM-CM5",
            "model_id": "CNRM-ALADIN63",
            "driving_model_ensemble_member": str(config["model_chain"]["ensemble_member"]),
            "experiment_id": _scenario_id(str(spec["scenario"])),
            "frequency": "day",
            "rcm_version_id": str(config["dataset"]["cordex_file_version"]),
        }
        for attribute, expected in expected_globals.items():
            actual = getattr(dataset, attribute, None)
            if actual != expected:
                raise ValueError(
                    f"{source_key} {variable_name} has {attribute}={actual!r}; expected {expected!r}"
                )
        if variable_name not in dataset.variables:
            raise ValueError(f"{netcdf_path} does not contain {variable_name}")

        variable = dataset.variables[variable_name]
        expected_units = "K" if variable_name == "tas" else "W m-2"
        if getattr(variable, "units", None) != expected_units:
            raise ValueError(
                f"{source_key} {variable_name} units are {getattr(variable, 'units', None)!r}; "
                f"expected {expected_units!r}"
            )
        if getattr(variable, "cell_methods", None) != "time: mean":
            raise ValueError(f"{source_key} {variable_name} is not a daily time mean")

        time = dataset.variables["time"]
        calendar_name = str(getattr(time, "calendar", ""))
        if calendar_name != "gregorian":
            raise ValueError(f"{source_key} {variable_name} calendar is not Gregorian")
        raw_dates = netCDF4.num2date(
            time[:],
            time.units,
            calendar=calendar_name,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        dates = pd.DatetimeIndex(pd.to_datetime([date.isoformat() for date in raw_dates]))
        expected_dates = pd.date_range(
            pd.Timestamp(str(spec["period_start"])) + pd.Timedelta(hours=12),
            pd.Timestamp(str(spec["period_end"])) + pd.Timedelta(hours=12),
            freq="D",
        )
        if not dates.equals(expected_dates):
            raise ValueError(f"{source_key} {variable_name} does not cover its configured period")

        latitudes = np.asarray(dataset.variables["lat"][:], dtype=float)
        longitudes = np.asarray(dataset.variables["lon"][:], dtype=float)
        distance = (latitudes - float(config["spatial_extraction"]["target_lat"])) ** 2 + (
            (longitudes - float(config["spatial_extraction"]["target_lon"]))
            * np.cos(np.deg2rad(float(config["spatial_extraction"]["target_lat"])))
        ) ** 2
        selected_y, selected_x = np.unravel_index(np.argmin(distance), distance.shape)
        expected_indices = config["spatial_extraction"]["selected_indices"]
        if (int(selected_x), int(selected_y)) != (
            int(expected_indices["x"]),
            int(expected_indices["y"]),
        ):
            raise ValueError(f"{source_key} {variable_name} selects an unexpected grid cell")
        selected_lat = float(latitudes[selected_y, selected_x])
        selected_lon = float(longitudes[selected_y, selected_x])
        if not np.isclose(
            selected_lat, float(config["spatial_extraction"]["selected_lat"]), atol=1e-10
        ) or not np.isclose(
            selected_lon, float(config["spatial_extraction"]["selected_lon"]), atol=1e-10
        ):
            raise ValueError(f"{source_key} {variable_name} selected coordinates changed")

        values = np.ma.asarray(variable[:, selected_y, selected_x])
        if np.ma.is_masked(values) and np.any(np.ma.getmaskarray(values)):
            raise ValueError(f"{source_key} {variable_name} contains missing values")
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{source_key} {variable_name} contains non-finite values")
        global_attributes = {
            name: getattr(dataset, name)
            for name in (
                "tracking_id",
                "creation_date",
                "driving_experiment",
                "experiment",
                "contact",
                "institution",
            )
            if hasattr(dataset, name)
        }

    return {
        "variable": variable_name,
        "values": values,
        "dates": dates,
        "input_units": expected_units,
        "netcdf": {
            "path": _relative(netcdf_path, config),
            "sha256": netcdf_hash,
            "size_bytes": int(netcdf_path.stat().st_size),
        },
        "provenance_json": {
            "path": _relative(provenance_path, config),
            "sha256": provenance_hash,
        },
        "provenance_image": {
            "path": _relative(image_path, config),
            "sha256": image_hash,
        },
        "cds_provenance": provenance,
        "netcdf_global_attributes": global_attributes,
    }


def prepare_source(
    config: Mapping[str, Any], source_key: str, spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one tas/rsds pair and write its canonical daily CSV and metadata."""

    temperature = _read_variable(config, spec, source_key, "temperature")
    radiation = _read_variable(config, spec, source_key, "solar_radiation")
    if not temperature["dates"].equals(radiation["dates"]):
        raise ValueError(f"{source_key} tas and rsds timestamps do not align")

    temperatures_c = temperature["values"] - 273.15
    solar = radiation["values"]
    t_min, t_max = map(float, config["physical_ranges"]["temperature_C"])
    i_min, i_max = map(float, config["physical_ranges"]["solar_W_m2"])
    if not np.logical_and(temperatures_c >= t_min, temperatures_c <= t_max).all():
        raise ValueError(f"{source_key} extracted temperature violates configured bounds")
    if not np.logical_and(solar >= i_min, solar <= i_max).all():
        raise ValueError(f"{source_key} extracted solar radiation violates configured bounds")

    source_paths = [temperature["netcdf"]["path"], radiation["netcdf"]["path"]]
    frame = pd.DataFrame(
        {
            "timestamp": temperature["dates"],
            "T_out_C": temperatures_c,
            "I_solar_W_m2": solar,
            "scenario": str(spec["scenario"]),
            "role": str(spec["role"]),
            "window": str(spec["window"]),
            "gcm_model": str(config["model_chain"]["gcm_model"]),
            "rcm_model": str(config["model_chain"]["rcm_model"]),
            "ensemble_member": str(config["model_chain"]["ensemble_member"]),
            "source_files": json.dumps(source_paths, separators=(",", ":")),
        }
    )
    if len(frame) != int(spec["expected_rows"]):
        raise ValueError(f"{source_key} produced {len(frame)} rows, not {spec['expected_rows']}")

    csv_path = resolve_config_path(config, spec["csv"])
    metadata_path = resolve_config_path(config, spec["metadata"])
    _atomic_write_csv(frame, csv_path)
    csv_hash = sha256_file(csv_path)
    metadata = {
        "schema_version": 2,
        "source_key": source_key,
        "dataset": config["dataset"]["name"],
        "dataset_identity": config["dataset"],
        "scenario": str(spec["scenario"]),
        "role": str(spec["role"]),
        "window": str(spec["window"]),
        "period_start": str(spec["period_start"]),
        "period_end": str(spec["period_end"]),
        "row_count": int(len(frame)),
        "year_count": int(frame["timestamp"].dt.year.nunique()),
        "model_chain": config["model_chain"],
        "spatial_extraction": config["spatial_extraction"],
        "time_calendar": "gregorian",
        "timestamp_convention": config["cordex_processing"]["timestamp_convention"],
        "temperature": {
            "variable": temperature["variable"],
            "input_units": temperature["input_units"],
            "output_units": "degC",
            "conversion": config["cordex_processing"]["temperature_conversion"],
            "raw": {key: value for key, value in temperature.items() if key not in {"values", "dates", "variable", "input_units"}},
        },
        "radiation": {
            "variable": radiation["variable"],
            "input_units": radiation["input_units"],
            "output_units": "W m-2",
            "conversion": config["cordex_processing"]["solar_conversion"],
            "raw": {key: value for key, value in radiation.items() if key not in {"values", "dates", "variable", "input_units"}},
        },
        "source_files": source_paths,
        "processing": {
            "script": config["cordex_processing"]["script"],
            "python": platform.python_version(),
            "netCDF4": netCDF4.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "output": {
            "csv": _relative(csv_path, config),
            "csv_sha256": csv_hash,
        },
    }
    _atomic_write_json(metadata, metadata_path)
    return {
        "source_key": source_key,
        "scenario": str(spec["scenario"]),
        "role": str(spec["role"]),
        "window": str(spec["window"]),
        "period_start": str(spec["period_start"]),
        "period_end": str(spec["period_end"]),
        "row_count": int(len(frame)),
        "year_count": int(frame["timestamp"].dt.year.nunique()),
        "csv": _relative(csv_path, config),
        "csv_sha256": csv_hash,
        "metadata": _relative(metadata_path, config),
        "metadata_sha256": sha256_file(metadata_path),
        "raw_netcdf": [temperature["netcdf"], radiation["netcdf"]],
        "raw_provenance_json": [
            temperature["provenance_json"],
            radiation["provenance_json"],
        ],
        "raw_provenance_images": [
            temperature["provenance_image"],
            radiation["provenance_image"],
        ],
    }


def prepare_all(config: Mapping[str, Any]) -> dict[str, Path]:
    """Build every configured daily source and its hash manifest from raw NetCDF."""

    outputs: dict[str, dict[str, Any]] = {}
    for source_key, spec in config["sources"].items():
        LOGGER.info("Preparing CORDEX daily source %s", source_key)
        outputs[source_key] = prepare_source(config, source_key, spec)

    manifest_path = resolve_config_path(config, config["cordex_processing"]["manifest"])
    manifest = {
        "schema_version": 1,
        "dataset": config["dataset"],
        "model_chain": config["model_chain"],
        "spatial_extraction": config["spatial_extraction"],
        "climate_target": {
            "label": "2050-centred IPCC mid-term period",
            "period": "2041-2060",
            "year_count": 20,
        },
        "baseline": {
            "period": "2006-2023",
            "year_count": 18,
            "scenario_matching": config["scenario_baselines"],
        },
        "processing": {
            "script": config["cordex_processing"]["script"],
            "timestamp_convention": config["cordex_processing"]["timestamp_convention"],
            "temperature_conversion": config["cordex_processing"]["temperature_conversion"],
            "solar_conversion": config["cordex_processing"]["solar_conversion"],
            "python": platform.python_version(),
            "netCDF4": netCDF4.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "sources": outputs,
        "caveats": {
            "single_model_chain": config["provenance"]["single_chain_caveat"],
            "observed_anchor": config["provenance"]["observed_anchor_caveat"],
        },
    }
    _atomic_write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        **{
            f"{key}_csv": resolve_config_path(config, value["csv"])
            for key, value in outputs.items()
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw CDS CORDEX NetCDF files and extract Brussels daily series."
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
        paths = prepare_all(load_config(args.config))
    except Exception as exc:
        LOGGER.error("CORDEX preparation failed: %s", exc)
        if args.verbose:
            LOGGER.exception("Detailed failure")
        return 1
    for name, path in paths.items():
        LOGGER.info("Wrote %s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
