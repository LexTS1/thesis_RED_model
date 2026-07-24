"""Apply the monthly CORDEX delta contract to cleaned hourly PVGIS years."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .load_cordex import resolve_config_path, sha256_file
from .load_observed import split_complete_years


MORPH_REQUIRED_COLUMNS = {
    "timestamp_utc",
    "T_out_C",
    "I_beam_horizontal_W_m2",
    "I_diffuse_horizontal_W_m2",
    "I_solar_W_m2",
    "wind_speed_10m_m_s",
    "sun_height_deg",
    "pvgis_reconstructed",
}


@dataclass(frozen=True)
class DeltaContract:
    """Validated monthly climate deltas and their source provenance."""

    frame: pd.DataFrame
    csv_path: Path
    csv_sha256: str
    provenance_path: Path
    provenance_sha256: str


def load_delta_contract(config: Mapping[str, Any]) -> DeltaContract:
    """Load only the canonical monthly delta table used by the hourly morph."""

    output_dir = resolve_config_path(config, config["outputs"]["directory"])
    csv_path = output_dir / config["outputs"]["monthly_deltas"]
    provenance_path = output_dir / config["outputs"]["provenance"]
    if not csv_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Monthly CORDEX delta artifacts are missing.")
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    csv_hash = sha256_file(csv_path)
    expected_hash = provenance["outputs"]["monthly_deltas"]["sha256"]
    if csv_hash != expected_hash:
        raise ValueError(
            f"Monthly delta hash mismatch: expected {expected_hash}, got {csv_hash}."
        )

    frame = pd.read_csv(csv_path)
    required = {
        "scenario",
        "month",
        "delta_T_C",
        "alpha_solar_raw",
        "alpha_solar_applied",
        "alpha_was_clipped",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Monthly delta contract is missing columns: {sorted(missing)}")
    if len(frame) != 36 or frame.duplicated(["scenario", "month"]).any():
        raise ValueError("Monthly delta contract must contain 36 unique scenario-month rows.")
    if not frame.groupby("scenario")["month"].apply(
        lambda values: sorted(values.astype(int).tolist()) == list(range(1, 13))
    ).all():
        raise ValueError("Every delta scenario must contain calendar months 1 through 12.")
    if frame["alpha_was_clipped"].astype(bool).any():
        raise ValueError("Canonical real-data morph contract unexpectedly contains clipped solar alpha.")
    minimum = float(config["solar_alpha_safety"]["minimum"])
    maximum = float(config["solar_alpha_safety"]["maximum"])
    if not frame["alpha_solar_applied"].between(minimum, maximum).all():
        raise ValueError("Applied solar alpha lies outside the configured safety bounds.")
    np.testing.assert_allclose(
        frame["alpha_solar_applied"], frame["alpha_solar_raw"], rtol=0.0, atol=1e-12
    )
    return DeltaContract(
        frame=frame,
        csv_path=csv_path,
        csv_sha256=csv_hash,
        provenance_path=provenance_path,
        provenance_sha256=sha256_file(provenance_path),
    )


def scenario_parameters(deltas: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Return a month-indexed, validated view of one scenario's morph parameters."""

    selected = deltas.loc[deltas["scenario"] == scenario].copy().sort_values("month")
    if selected["month"].astype(int).tolist() != list(range(1, 13)):
        available = sorted(str(value) for value in deltas["scenario"].unique())
        raise ValueError(
            f"Scenario {scenario!r} has no complete monthly delta set; available={available}."
        )
    return selected.set_index("month")


def morph_observed_year(
    frame: pd.DataFrame,
    scenario: str,
    deltas: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Morph one complete observed UTC year using monthly additive/multiplicative deltas."""

    missing = MORPH_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Observed year is missing morph columns: {sorted(missing)}")
    years = split_complete_years(frame)
    if len(years) != 1:
        raise ValueError("morph_observed_year requires exactly one complete observed year.")
    observed_year = next(iter(years))
    source = years[observed_year]
    parameters = scenario_parameters(deltas, scenario)
    months = source["timestamp_utc"].dt.month
    delta_t = months.map(parameters["delta_T_C"]).to_numpy(dtype=float)
    alpha = months.map(parameters["alpha_solar_applied"]).to_numpy(dtype=float)
    if not np.isfinite(delta_t).all() or not np.isfinite(alpha).all():
        raise ValueError("Morph parameter lookup produced missing or non-finite values.")

    result = source.copy()
    result["T_out_C"] = source["T_out_C"].to_numpy(dtype=float) + delta_t
    result["I_beam_horizontal_W_m2"] = (
        source["I_beam_horizontal_W_m2"].to_numpy(dtype=float) * alpha
    )
    result["I_diffuse_horizontal_W_m2"] = (
        source["I_diffuse_horizontal_W_m2"].to_numpy(dtype=float) * alpha
    )
    result["I_solar_W_m2"] = (
        result["I_beam_horizontal_W_m2"] + result["I_diffuse_horizontal_W_m2"]
    )

    for unchanged in (
        "timestamp_utc",
        "wind_speed_10m_m_s",
        "sun_height_deg",
        "pvgis_reconstructed",
    ):
        if not result[unchanged].equals(source[unchanged]):
            raise ValueError(f"Morph unexpectedly changed invariant column {unchanged!r}.")
    if not np.isfinite(
        result[[column for column in MORPH_REQUIRED_COLUMNS if column != "timestamp_utc"]]
        .select_dtypes(include=[np.number, "bool"])
        .to_numpy(dtype=float)
    ).all():
        raise ValueError("Morphed weather contains non-finite values.")
    if (result[["I_beam_horizontal_W_m2", "I_diffuse_horizontal_W_m2", "I_solar_W_m2"]] < 0.0).any().any():
        raise ValueError("Morphed weather contains negative irradiance.")
    if not np.allclose(
        result["I_solar_W_m2"],
        result["I_beam_horizontal_W_m2"] + result["I_diffuse_horizontal_W_m2"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Morphed GHI is not beam plus diffuse irradiance.")
    night = result["sun_height_deg"] < 0.0
    if (result.loc[night, "I_solar_W_m2"] != 0.0).any():
        raise ValueError("Morphed irradiance is non-zero while sun height is below zero.")

    if config is not None:
        t_min, t_max = map(float, config["physical_ranges"]["temperature_C"])
        i_min, i_max = map(float, config["physical_ranges"]["solar_W_m2"])
        if not result["T_out_C"].between(t_min, t_max).all():
            raise ValueError("Morphed temperature lies outside configured physical bounds.")
        if not result["I_solar_W_m2"].between(i_min, i_max).all():
            raise ValueError("Morphed GHI lies outside configured physical bounds.")

    result.attrs.update(
        {
            "scenario": scenario,
            "observed_year": observed_year,
            "method": "monthly_delta_morph_v1",
        }
    )
    return result
