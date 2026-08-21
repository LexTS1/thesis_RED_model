"""Create deterministic thesis figures from validated climate artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from .load_cordex import load_config, resolve_config_path, sha256_file
from .load_observed import load_clean_observed, split_complete_years
from .morph import load_delta_contract


LOGGER = logging.getLogger("climate.plot_validation")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

SCENARIO_LABELS = {
    "rcp_2_6": "RCP2.6",
    "rcp_4_5": "RCP4.5",
    "rcp_8_5": "RCP8.5",
}
SCENARIO_COLORS = {
    "rcp_2_6": "#0072B2",
    "rcp_4_5": "#E69F00",
    "rcp_8_5": "#D55E00",
}
OBSERVED_COLOR = "#333333"
REFERENCE_COLOR = "#009E73"
DIRECT_CORDEX_COLOR = "#0072B2"
MORPHED_COLOR = "#D55E00"
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _relative(path: Path, config: Mapping[str, Any]) -> str:
    try:
        return str(path.relative_to(Path(config["_base_dir"])))
    except ValueError:
        return str(path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def _save_figure(
    figure: plt.Figure, output_dir: Path, basename: str
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    png_temp = png_path.with_name(f".{png_path.name}.writing")
    pdf_temp = pdf_path.with_name(f".{pdf_path.name}.writing")
    figure.savefig(
        png_temp,
        format="png",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "climate.src.plot_validation"},
    )
    figure.savefig(
        pdf_temp,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "climate.src.plot_validation",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    png_temp.replace(png_path)
    pdf_temp.replace(pdf_path)
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path}


def _load_validation_tables(
    config: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    Path,
]:
    outputs = config["validation"]["outputs"]
    root = resolve_config_path(config, outputs["directory"])
    report_path = root / outputs["report_json"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass" or int(report.get("hard_error_count", -1)) != 0:
        raise ValueError("Validation figures require a passing validation report.")
    member_path = root / outputs["member_summary"]
    monthly_path = root / outputs["monthly_invariants"]
    observed_reference_path = root / outputs["observed_reference_comparison"]
    cordex_morph_path = root / outputs["cordex_morph_comparison"]
    if sha256_file(member_path) != report["outputs"]["member_summary"]["sha256"]:
        raise ValueError("Validated member summary hash mismatch.")
    if sha256_file(monthly_path) != report["outputs"]["monthly_invariants"]["sha256"]:
        raise ValueError("Validated monthly-invariant table hash mismatch.")
    if (
        sha256_file(observed_reference_path)
        != report["outputs"]["observed_reference_comparison"]["sha256"]
    ):
        raise ValueError("Validated PVGIS--BE100 comparison hash mismatch.")
    if (
        sha256_file(cordex_morph_path)
        != report["outputs"]["cordex_morph_comparison"]["sha256"]
    ):
        raise ValueError("Validated CORDEX--morph comparison hash mismatch.")
    return (
        pd.read_csv(member_path),
        pd.read_csv(monthly_path),
        pd.read_csv(observed_reference_path),
        pd.read_csv(cordex_morph_path),
        report,
        report_path,
    )


def plot_monthly_parameters(
    deltas: pd.DataFrame,
    scenarios: list[str],
) -> plt.Figure:
    """Plot the two monthly CORDEX change-factor contracts."""

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    months = np.arange(1, 13)
    for scenario in scenarios:
        selected = deltas.loc[deltas["scenario"] == scenario].sort_values("month")
        axes[0].plot(
            months,
            selected["delta_T_C"],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )
        axes[1].plot(
            months,
            selected["alpha_solar_applied"],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )
    for axis in axes:
        axis.set_xticks(months, MONTH_LABELS)
        axis.set_xlim(1, 12)
        axis.legend(frameon=False, ncol=3, loc="upper left")
    axes[0].set_title("(a) Monthly temperature change")
    axes[0].set_ylabel("ΔT (°C)")
    axes[0].set_xlabel("Calendar month")
    axes[1].set_title("(b) Monthly solar scaling factor")
    axes[1].set_ylabel("α applied (–)")
    axes[1].set_xlabel("Calendar month")
    axes[1].axhline(1.0, color="#666666", linewidth=0.8)
    return figure


def plot_observed_reference_degree_days(
    comparison: pd.DataFrame,
) -> plt.Figure:
    """Compare annual PVGIS degree days with the official BE100 series."""

    ordered = comparison.sort_values("year", kind="stable")
    years = ordered["year"].to_numpy(dtype=int)
    panels = (
        (
            "HDD",
            "pvgis_HDD_C_days",
            "official_BE100_HDD_C_days",
            "(a) Heating degree days",
        ),
        (
            "CDD",
            "pvgis_CDD_C_days",
            "official_BE100_CDD_C_days",
            "(b) Cooling degree days",
        ),
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.6), sharex=True, constrained_layout=True
    )
    for axis, (metric, pvgis_column, reference_column, title) in zip(
        axes, panels
    ):
        pvgis = ordered[pvgis_column].to_numpy(dtype=float)
        reference = ordered[reference_column].to_numpy(dtype=float)
        pvgis_mean = float(np.mean(pvgis))
        reference_mean = float(np.mean(reference))
        correlation = float(np.corrcoef(pvgis, reference)[0, 1])

        axis.plot(
            years,
            pvgis,
            color=OBSERVED_COLOR,
            marker="o",
            markersize=4.0,
            linewidth=1.6,
            label="PVGIS",
        )
        axis.plot(
            years,
            reference,
            color=REFERENCE_COLOR,
            marker="s",
            markersize=3.8,
            linewidth=1.6,
            label="Eurostat BE100",
        )
        axis.axhline(
            pvgis_mean,
            color=OBSERVED_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )
        axis.axhline(
            reference_mean,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )
        axis.set_title(title)
        axis.set_xlabel("Calendar year")
        axis.set_ylabel(f"{metric} (°C·days)")
        axis.set_xticks([2006, 2009, 2012, 2015, 2018, 2021, 2023])
        axis.set_xlim(2005.5, 2023.5)
        if metric == "HDD":
            axis.legend(frameon=False, loc="upper right")
        annotation_y = 0.03 if metric == "HDD" else 0.97
        annotation_va = "bottom" if metric == "HDD" else "top"
        axis.text(
            0.02,
            annotation_y,
            (
                f"Annual mean: PVGIS {pvgis_mean:.2f}; "
                f"BE100 {reference_mean:.2f}\n"
                f"r = {correlation:.3f}"
            ),
            transform=axis.transAxes,
            ha="left",
            va=annotation_va,
            fontsize=7.4,
            bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.90},
        )
    return figure


def plot_cordex_morph_degree_day_changes(
    comparison: pd.DataFrame,
    scenarios: list[str],
) -> plt.Figure:
    """Compare direct CORDEX and morphed-PVGIS changes with grouped bars."""

    indexed = comparison.set_index("scenario")
    x_positions = np.arange(len(scenarios), dtype=float)
    bar_width = 0.34
    panels = (
        (
            "cordex_change_HDD_C_days",
            "morphed_paired_change_HDD_C_days",
            "HDD change (°C·days)",
        ),
        (
            "cordex_change_CDD_C_days",
            "morphed_paired_change_CDD_C_days",
            "CDD change (°C·days)",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, (cordex_column, morph_column, y_label) in zip(axes, panels):
        direct = indexed.loc[scenarios, cordex_column].to_numpy(dtype=float)
        morphed = indexed.loc[scenarios, morph_column].to_numpy(dtype=float)
        direct_bars = axis.bar(
            x_positions - bar_width / 2.0,
            direct,
            width=bar_width,
            color=DIRECT_CORDEX_COLOR,
            label="Direct CORDEX",
        )
        morphed_bars = axis.bar(
            x_positions + bar_width / 2.0,
            morphed,
            width=bar_width,
            color=MORPHED_COLOR,
            label="Morphed PVGIS",
        )
        axis.axhline(0.0, color="#666666", linewidth=0.8)
        axis.set_ylabel(y_label)
        axis.set_xlabel("Scenario")
        axis.set_xticks(
            x_positions,
            [SCENARIO_LABELS[scenario] for scenario in scenarios],
        )
        axis.bar_label(direct_bars, fmt="%+.1f", padding=3, fontsize=7.4)
        axis.bar_label(morphed_bars, fmt="%+.1f", padding=3, fontsize=7.4)
        axis.margins(y=0.18)
    axes[0].legend(frameon=False, loc="lower left")
    return figure


def select_weather_morphing_worked_example_day(
    observed: pd.DataFrame,
    month: int = 7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a complete, high-signal PVGIS day using an explicit rule."""

    required = {"timestamp_utc", "T_out_C", "I_solar_W_m2"}
    missing = required.difference(observed.columns)
    if missing:
        raise ValueError(f"Observed weather lacks worked-example fields: {sorted(missing)}")

    candidate = observed.copy()
    candidate["timestamp_utc"] = pd.to_datetime(
        candidate["timestamp_utc"], utc=True, errors="raise"
    )
    candidate = candidate.loc[candidate["timestamp_utc"].dt.month == month].copy()
    candidate["date_utc"] = candidate["timestamp_utc"].dt.floor("D")
    daily = candidate.groupby("date_utc", sort=True).agg(
        hour_count=("timestamp_utc", "size"),
        temperature_min_C=("T_out_C", "min"),
        temperature_max_C=("T_out_C", "max"),
        ghi_max_W_m2=("I_solar_W_m2", "max"),
        ghi_sum_Wh_m2=("I_solar_W_m2", "sum"),
    )
    daily = daily.loc[daily["hour_count"] == 24].copy()
    if daily.empty:
        raise ValueError(f"No complete observed days are available for month {month}.")
    daily["temperature_range_C"] = (
        daily["temperature_max_C"] - daily["temperature_min_C"]
    )
    score_columns = ("temperature_range_C", "ghi_max_W_m2", "ghi_sum_Wh_m2")
    daily["selection_score"] = daily.loc[:, score_columns].rank(
        method="average", pct=True
    ).mean(axis=1)
    selected_date = daily.sort_values(
        ["selection_score", "date_utc"],
        ascending=[False, True],
        kind="stable",
    ).index[0]
    selected = candidate.loc[candidate["date_utc"] == selected_date].copy()
    selected = selected.sort_values("timestamp_utc", kind="stable").drop(
        columns="date_utc"
    )
    selected_hours = selected["timestamp_utc"].dt.hour.to_numpy(dtype=int)
    if not np.array_equal(selected_hours, np.arange(24)):
        raise ValueError("Selected worked-example day is not a complete UTC day.")

    record = daily.loc[selected_date]
    metadata = {
        "date_utc": selected_date.strftime("%Y-%m-%d"),
        "candidate_day_count": int(len(daily)),
        "selection_rule": (
            "Maximum mean percentile rank of daily temperature range, "
            "maximum hourly GHI, and daily GHI sum among complete July days."
        ),
        "temperature_min_C": float(record["temperature_min_C"]),
        "temperature_max_C": float(record["temperature_max_C"]),
        "temperature_range_C": float(record["temperature_range_C"]),
        "ghi_max_W_m2": float(record["ghi_max_W_m2"]),
        "ghi_sum_Wh_m2": float(record["ghi_sum_Wh_m2"]),
        "selection_score": float(record["selection_score"]),
    }
    return selected, metadata


def plot_weather_morphing_worked_example(
    observed_day: pd.DataFrame,
    delta_temperature_C: float,
    solar_factor: float,
) -> plt.Figure:
    """Illustrate morphing with one selected, complete observed PVGIS day."""

    day = observed_day.sort_values("timestamp_utc", kind="stable").copy()
    timestamps = pd.to_datetime(day["timestamp_utc"], utc=True, errors="raise")
    hours = timestamps.dt.hour.to_numpy(dtype=float)
    if len(day) != 24 or not np.array_equal(hours, np.arange(24, dtype=float)):
        raise ValueError("Worked-example plot requires one complete UTC day.")
    example_hour = 12
    observed_temperature = day["T_out_C"].to_numpy(dtype=float)
    observed_ghi = day["I_solar_W_m2"].to_numpy(dtype=float)
    morphed_temperature = observed_temperature + delta_temperature_C
    morphed_ghi = observed_ghi * solar_factor
    date_label = timestamps.iloc[0].strftime("%d %B %Y").lstrip("0")

    figure, axes = plt.subplots(2, 1, figsize=(10.2, 6.5), constrained_layout=True)
    temperature_axis, solar_axis = axes
    temperature_axis.fill_between(
        hours,
        observed_temperature,
        morphed_temperature,
        color=MORPHED_COLOR,
        alpha=0.10,
        linewidth=0.0,
    )
    temperature_axis.plot(
        hours,
        observed_temperature,
        color=DIRECT_CORDEX_COLOR,
        linewidth=2.0,
        label=f"Observed PVGIS, {date_label}",
    )
    temperature_axis.plot(
        hours,
        morphed_temperature,
        color=MORPHED_COLOR,
        linewidth=2.0,
        label="Morphed RCP4.5 profile",
    )
    temperature_axis.axvline(
        example_hour, color="#888888", linestyle=":", linewidth=0.8
    )

    temperature_axis.set_ylabel("Outdoor temperature (°C)")
    temperature_axis.scatter(
        [example_hour, example_hour],
        [observed_temperature[example_hour], morphed_temperature[example_hour]],
        color=[DIRECT_CORDEX_COLOR, MORPHED_COLOR],
        s=30,
        zorder=3,
    )
    temperature_axis.annotate(
        "",
        xy=(example_hour, morphed_temperature[example_hour]),
        xytext=(example_hour, observed_temperature[example_hour]),
        arrowprops={"arrowstyle": "<->", "color": "#555555", "linewidth": 1.0},
    )
    temperature_axis.text(
        example_hour + 0.35,
        float(
            (observed_temperature[example_hour] + morphed_temperature[example_hour])
            / 2.0
        ),
        f"+{delta_temperature_C:.3f} °C",
        ha="left",
        va="center",
        fontsize=8,
    )
    temperature_axis.legend(frameon=False, loc="upper left", ncol=2)
    temperature_axis.set_xlabel("Hour of day (UTC)")
    temperature_axis.set_xticks(
        [0, 6, 12, 18, 23], ["00:00", "06:00", "12:00", "18:00", "23:00"]
    )
    temperature_axis.set_xlim(0.0, 23.0)

    solar_mask = (hours >= 9.0) & (hours <= 15.0)
    solar_hours = hours[solar_mask]
    solar_observed = observed_ghi[solar_mask]
    solar_morphed = morphed_ghi[solar_mask]
    solar_axis.fill_between(
        solar_hours,
        solar_observed,
        solar_morphed,
        color=MORPHED_COLOR,
        alpha=0.12,
        linewidth=0.0,
    )
    solar_axis.plot(
        solar_hours,
        solar_observed,
        color=DIRECT_CORDEX_COLOR,
        linewidth=2.0,
    )
    solar_axis.plot(
        solar_hours,
        solar_morphed,
        color=MORPHED_COLOR,
        linewidth=2.0,
    )
    solar_axis.axvline(
        example_hour, color="#888888", linestyle=":", linewidth=0.8
    )
    solar_axis.set_ylabel("GHI (W/m²)")
    solar_axis.set_xlabel("Hour of day (UTC)")
    solar_axis.set_xticks(
        [9, 10, 11, 12, 13, 14, 15],
        ["09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00"],
    )
    solar_axis.scatter(
        [example_hour, example_hour],
        [observed_ghi[example_hour], morphed_ghi[example_hour]],
        color=[DIRECT_CORDEX_COLOR, MORPHED_COLOR],
        s=30,
        zorder=3,
    )
    solar_axis.text(
        0.02,
        0.92,
        f"12:00 UTC: {observed_ghi[example_hour]:.2f} → "
        f"{morphed_ghi[example_hour]:.2f} W/m²\n"
        f"(×{solar_factor:.5f})",
        transform=solar_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    solar_axis.set_xlim(9.0, 15.0)
    solar_values = np.concatenate((solar_observed, solar_morphed))
    solar_padding = max(float(np.ptp(solar_values)) * 0.10, 8.0)
    solar_axis.set_ylim(
        float(solar_values.min()) - solar_padding,
        float(solar_values.max()) + solar_padding,
    )
    return figure


def plot_morph_residuals(
    monthly: pd.DataFrame,
    scenarios: list[str],
) -> plt.Figure:
    """Plot all member-month temperature and GHI morph residuals."""

    ordered = monthly.copy()
    ordered["scenario"] = pd.Categorical(
        ordered["scenario"], categories=scenarios, ordered=True
    )
    ordered = ordered.sort_values(
        ["scenario", "observed_pvgis_year", "month"], kind="stable"
    )
    row_index = (
        ordered[["scenario", "observed_pvgis_year"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    row_index["row"] = np.arange(len(row_index))
    ordered = ordered.merge(
        row_index, on=["scenario", "observed_pvgis_year"], validate="many_to_one"
    )

    temperature = ordered.pivot(
        index="row", columns="month", values="delta_T_residual_C"
    ).to_numpy()
    solar = ordered.pivot(
        index="row", columns="month", values="ghi_alpha_residual"
    ).to_numpy()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 8.2),
        sharey=True,
        constrained_layout=True,
    )
    for axis, values, title, colorbar_label in (
        (
            axes[0],
            temperature,
            "(a) Recovered ΔT − expected ΔT",
            "Temperature residual (°C)",
        ),
        (
            axes[1],
            solar,
            "(b) Recovered GHI ratio − α",
            "Solar-factor residual (–)",
        ),
    ):
        # Keep an all-zero, passing heatmap interpretable by scaling it to the
        # configured 1e-8 monthly identity tolerance rather than machine epsilon.
        limit = max(float(np.nanmax(np.abs(values))), 1.0e-8)
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        axis.set_title(title)
        axis.set_xticks(np.arange(12), MONTH_LABELS)
        axis.set_xlabel("Calendar month")
        axis.grid(False)
        axis.text(
            0.02,
            0.015,
            f"max |residual| = {float(np.nanmax(np.abs(values))):.2e}\nall cells pass",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.88},
        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
        colorbar.set_label(colorbar_label)
        colorbar.formatter.set_powerlimits((-2, 2))
        colorbar.update_ticks()
        for boundary in (18.5, 37.5):
            axis.axhline(boundary, color="#333333", linewidth=0.8)

    labels = [
        f"{SCENARIO_LABELS[str(row.scenario)]} · {int(row.observed_pvgis_year)}"
        for row in row_index.itertuples(index=False)
    ]
    labelled_rows = [
        index
        for index, row in row_index.iterrows()
        if int(row["observed_pvgis_year"]) in {2006, 2009, 2012, 2015, 2018, 2021, 2023}
    ]
    axes[0].set_yticks(
        labelled_rows,
        [labels[index] for index in labelled_rows],
    )
    axes[0].tick_params(axis="y", labelsize=6.3)
    axes[0].set_ylabel("Morphed member (scenario · PVGIS year)")
    return figure


def _duration_statistics(
    year_frames: Mapping[int, pd.DataFrame], quantiles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = np.vstack(
        [
            np.quantile(
                frame["T_out_C"].to_numpy(dtype=float),
                quantiles,
                method="linear",
            )
            for _, frame in sorted(year_frames.items())
        ]
    )
    return (
        np.median(curves, axis=0),
        np.quantile(curves, 0.05, axis=0),
        np.quantile(curves, 0.95, axis=0),
    )


def _load_member_temperature_years(
    config: Mapping[str, Any], scenarios: list[str]
) -> dict[str, dict[int, pd.DataFrame]]:
    ensemble = config["observed_weather"]["ensemble"]
    root = resolve_config_path(config, ensemble["directory"])
    manifest = pd.read_csv(root / ensemble["manifest_csv"])
    result: dict[str, dict[int, pd.DataFrame]] = {scenario: {} for scenario in scenarios}
    for row in manifest.itertuples(index=False):
        scenario = str(row.scenario)
        if scenario not in result:
            continue
        path = resolve_config_path(config, str(row.member_path))
        if sha256_file(path) != str(row.member_sha256):
            raise ValueError(f"Member hash mismatch while plotting {row.member_id}.")
        result[scenario][int(row.observed_pvgis_year)] = pd.read_csv(
            path, usecols=["T_out_C"]
        )
    expected_years = int(config["observed_weather"]["expected_years"])
    if any(len(years) != expected_years for years in result.values()):
        raise ValueError(
            f"Temperature-duration curves require {expected_years} years per scenario."
        )
    return result


def plot_temperature_duration(
    observed_years: Mapping[int, pd.DataFrame],
    member_years: Mapping[str, Mapping[int, pd.DataFrame]],
    scenarios: list[str],
) -> plt.Figure:
    """Plot median annual temperature-duration curves and inter-annual envelopes."""

    exceedance = np.linspace(0.0, 100.0, 1001)
    quantiles = 1.0 - exceedance / 100.0
    statistics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "observed": _duration_statistics(observed_years, quantiles)
    }
    statistics.update(
        {
            scenario: _duration_statistics(member_years[scenario], quantiles)
            for scenario in scenarios
        }
    )

    figure = plt.figure(figsize=(11.2, 7.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[2.2, 1.0])
    main = figure.add_subplot(grid[0, :])
    hot = figure.add_subplot(grid[1, 0])
    cold = figure.add_subplot(grid[1, 1])
    axes = (main, hot, cold)
    for key in ("observed", *scenarios):
        median, lower, upper = statistics[key]
        color = OBSERVED_COLOR if key == "observed" else SCENARIO_COLORS[key]
        label = "Observed PVGIS" if key == "observed" else SCENARIO_LABELS[key]
        for axis in axes:
            axis.fill_between(
                exceedance,
                lower,
                upper,
                color=color,
                alpha=0.10,
                linewidth=0.0,
            )
            axis.plot(
                exceedance,
                median,
                color=color,
                linewidth=1.7,
                label=label,
            )
    main.set_title("(a) Full hourly temperature-duration curves")
    main.set_xlim(0, 100)
    main.set_ylabel("Outdoor temperature (°C)")
    main.set_xlabel("Hours exceeded (%)")
    main.legend(frameon=False, ncol=4, loc="upper right")
    hot.set_title("(b) Hottest 2% of hours")
    hot.set_xlim(0, 2)
    hot.set_ylabel("Outdoor temperature (°C)")
    hot.set_xlabel("Hours exceeded (%)")
    cold.set_title("(c) Coldest 2% of hours")
    cold.set_xlim(98, 100)
    cold.set_ylabel("Outdoor temperature (°C)")
    cold.set_xlabel("Hours exceeded (%)")
    for axis, mask in (
        (hot, exceedance <= 2.0),
        (cold, exceedance >= 98.0),
    ):
        lower_bound = min(
            float(statistics[key][1][mask].min())
            for key in ("observed", *scenarios)
        )
        upper_bound = max(
            float(statistics[key][2][mask].max())
            for key in ("observed", *scenarios)
        )
        padding = max((upper_bound - lower_bound) * 0.08, 0.5)
        axis.set_ylim(lower_bound - padding, upper_bound + padding)
    return figure


def build_validation_figures(config: Mapping[str, Any]) -> dict[str, Path]:
    """Generate all requested figures and a hash-based provenance manifest."""

    _style()
    (
        members,
        monthly,
        observed_reference,
        cordex_morph,
        validation_report,
        report_path,
    ) = _load_validation_tables(config)
    delta_contract = load_delta_contract(config)
    scenarios = [
        str(spec["scenario"])
        for spec in config["sources"].values()
        if spec["role"] == "future"
    ]
    observed, observed_metadata = load_clean_observed(config)
    observed_years = split_complete_years(observed)
    member_years = _load_member_temperature_years(config, scenarios)
    worked_example = delta_contract.frame.loc[
        (delta_contract.frame["scenario"] == "rcp_4_5")
        & (delta_contract.frame["month"] == 7)
    ]
    if len(worked_example) != 1:
        raise ValueError("Expected one RCP4.5 July row for the worked example.")
    worked_example_row = worked_example.iloc[0]
    worked_example_day, worked_example_selection = (
        select_weather_morphing_worked_example_day(observed, month=7)
    )

    figure_spec = config["validation"]["figures"]
    output_dir = resolve_config_path(config, figure_spec["directory"])
    outputs: dict[str, Path] = {}
    figure_builds = (
        (
            "monthly_parameters",
            plot_monthly_parameters(delta_contract.frame, scenarios),
        ),
        (
            "morph_residuals",
            plot_morph_residuals(monthly, scenarios),
        ),
        (
            "temperature_duration",
            plot_temperature_duration(observed_years, member_years, scenarios),
        ),
        (
            "observed_reference_degree_days",
            plot_observed_reference_degree_days(observed_reference),
        ),
        (
            "cordex_morph_degree_day_changes",
            plot_cordex_morph_degree_day_changes(cordex_morph, scenarios),
        ),
        (
            "weather_morphing_worked_example",
            plot_weather_morphing_worked_example(
                worked_example_day,
                float(worked_example_row["delta_T_C"]),
                float(worked_example_row["alpha_solar_applied"]),
            ),
        ),
    )
    output_records: dict[str, Any] = {}
    for key, figure in figure_builds:
        paths = _save_figure(figure, output_dir, str(figure_spec[key]))
        for extension, path in paths.items():
            outputs[f"{key}_{extension}"] = path
        output_records[key] = {
            extension: {
                "path": _relative(path, config),
                "sha256": sha256_file(path),
            }
            for extension, path in paths.items()
        }

    provenance_path = output_dir / figure_spec["provenance"]
    provenance = {
        "schema_version": 1,
        "method": {
            "monthly_parameters": (
                "Twelve monthly delta_T_C and alpha_solar_applied values per RCP."
            ),
            "morph_residuals": (
                "All 54 x 12 recovered-minus-expected temperature and GHI ratios."
            ),
            "temperature_duration": (
                "Median annual duration curve with 5th-95th percentile envelope "
                "across 18 observed weather years; x-axis is hours exceeded."
            ),
            "observed_reference_degree_days": (
                "Annual PVGIS and official Eurostat BE100 HDD/CDD series for "
                "2006-2023, with full-period means and Pearson correlation."
            ),
            "cordex_morph_degree_day_changes": (
                "Grouped-bar comparison of direct CORDEX and morphed-PVGIS "
                "future-minus-reference HDD/CDD changes for each RCP."
            ),
            "weather_morphing_worked_example": (
                "Selected observed PVGIS day transformed using the validated "
                "RCP4.5 July delta_T_C and alpha_solar_applied values; the "
                "temperature panel covers the full UTC day, the GHI panel "
                "covers 09:00-15:00 UTC, and 12:00 UTC supplies the numerical "
                "annotation."
            ),
        },
        "worked_example_selection": worked_example_selection,
        "inputs": {
            "validation_report": {
                "path": _relative(report_path, config),
                "sha256": sha256_file(report_path),
                "status": validation_report["status"],
            },
            "member_summary_sha256": validation_report["outputs"]["member_summary"][
                "sha256"
            ],
            "monthly_invariants_sha256": validation_report["outputs"][
                "monthly_invariants"
            ]["sha256"],
            "monthly_deltas_sha256": delta_contract.csv_sha256,
            "clean_observed_sha256": observed_metadata["output"]["sha256"],
            "observed_reference_comparison_sha256": validation_report["outputs"][
                "observed_reference_comparison"
            ]["sha256"],
            "cordex_morph_comparison_sha256": validation_report["outputs"][
                "cordex_morph_comparison"
            ]["sha256"],
        },
        "outputs": output_records,
    }
    _atomic_write_json(provenance, provenance_path)
    outputs["provenance"] = provenance_path
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis figures from the passing climate validation."
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
        paths = build_validation_figures(load_config(args.config))
    except Exception as exc:
        LOGGER.error("Climate validation figure build failed: %s", exc)
        if args.verbose:
            LOGGER.exception("Detailed failure")
        return 1
    for name, path in paths.items():
        LOGGER.info("Wrote %s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
