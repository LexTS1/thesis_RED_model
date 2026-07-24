"""Add four on-demand PVGIS façade irradiance series to a morphed member."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .load_observed import PvgisPlane
from .morph import scenario_parameters


ORIENTATIONS = ("south", "east", "west", "north")


def build_facade_irradiance(
    member_frame: pd.DataFrame,
    scenario: str,
    deltas: pd.DataFrame,
    facade_templates: Mapping[str, PvgisPlane],
) -> pd.DataFrame:
    """Return aligned morphed façade totals without altering the canonical member."""

    missing = sorted(set(ORIENTATIONS).difference(facade_templates))
    if missing:
        raise ValueError(f"Missing PVGIS façade templates: {missing}")
    timestamps = pd.DatetimeIndex(member_frame["timestamp_utc"])
    years = sorted(set(int(year) for year in timestamps.year))
    if len(years) != 1:
        raise ValueError("Façade construction requires one observed calendar year.")
    observed_year = years[0]
    parameters = scenario_parameters(deltas, scenario)
    alpha = member_frame["timestamp_utc"].dt.month.map(
        parameters["alpha_solar_applied"]
    ).to_numpy(dtype=float)
    result = pd.DataFrame({"timestamp_utc": member_frame["timestamp_utc"].copy()})

    for orientation in ORIENTATIONS:
        template = facade_templates[orientation].frame
        selected = template.loc[template["timestamp_utc"].dt.year == observed_year].reset_index(drop=True)
        selected_timestamps = pd.DatetimeIndex(selected["timestamp_utc"])
        if not selected_timestamps.equals(timestamps):
            raise ValueError(
                f"PVGIS façade {orientation} does not align with member year {observed_year}."
            )
        observed_total = selected[
            ["I_beam_W_m2", "I_diffuse_W_m2", "I_reflected_W_m2"]
        ].sum(axis=1).to_numpy(dtype=float)
        result[f"I_{orientation}_W_m2"] = observed_total * alpha

    values = result[[f"I_{orientation}_W_m2" for orientation in ORIENTATIONS]]
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0.0).any().any():
        raise ValueError("Morphed façade forcing contains invalid irradiance.")
    result.attrs.update(
        {
            "scenario": scenario,
            "observed_year": observed_year,
            "method": "pvgis_facade_shapes_monthly_alpha_v1",
            "source_sha256": {
                orientation: facade_templates[orientation].source_sha256
                for orientation in ORIENTATIONS
            },
        }
    )
    return result


def add_facade_irradiance(
    member_frame: pd.DataFrame,
    scenario: str,
    deltas: pd.DataFrame,
    facade_templates: Mapping[str, PvgisPlane],
) -> pd.DataFrame:
    """Return a copy of a canonical member with four façade-total columns appended."""

    facades = build_facade_irradiance(member_frame, scenario, deltas, facade_templates)
    result = member_frame.copy()
    for column in facades.columns:
        if column != "timestamp_utc":
            result[column] = facades[column].to_numpy(dtype=float)
    return result
