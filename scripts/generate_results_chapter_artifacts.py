"""Generate figures and the cell-level appendix table for the Results chapter.

The script reads only verified model outputs already present in the
repository.  It does not rerun the stock, climate, behavioural, or thermal
models.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures" / "results_chapter"
TABLE_PATH = ROOT / "tables" / "dwelling_level_results.tex"

RCP_ORDER = ["rcp_2_6", "rcp_4_5", "rcp_8_5"]
RCP_LABELS = {
    "rcp_2_6": "RCP2.6",
    "rcp_4_5": "RCP4.5",
    "rcp_8_5": "RCP8.5",
}
STATE_ORDER = [
    "TABULA_existing",
    "TABULA_standard_B_proxy",
    "TABULA_advanced_A_proxy",
]
STATE_LABELS = {
    "TABULA_existing": "Existing",
    "TABULA_standard_B_proxy": "Standard",
    "TABULA_advanced_A_proxy": "Advanced",
}
STATE_COLOURS = {
    "TABULA_existing": "#c95f45",
    "TABULA_standard_B_proxy": "#ebb247",
    "TABULA_advanced_A_proxy": "#449477",
}
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def inverse_weighted_quantiles(values: pd.Series, weights: pd.Series) -> np.ndarray:
    """Return p05, median, and p95 using an inverse weighted empirical CDF."""
    value_array = values.to_numpy(dtype=float)
    weight_array = weights.to_numpy(dtype=float)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    value_array = value_array[valid]
    weight_array = weight_array[valid]
    order = np.argsort(value_array, kind="mergesort")
    value_array = value_array[order]
    weight_array = weight_array[order]
    cumulative = np.cumsum(weight_array)
    targets = np.array([0.05, 0.50, 0.95]) * cumulative[-1]
    return value_array[np.searchsorted(cumulative, targets, side="left")]


def load_diagnostics() -> dict[str, pd.DataFrame]:
    partition_root = ROOT / "thermal_model" / "data" / "monte_carlo" / "current_vs_2050" / "partitions"
    diagnostics: dict[str, pd.DataFrame] = {}
    for partition in sorted(partition_root.iterdir()):
        completion = json.loads((partition / "partition_complete.json").read_text())
        member = completion["weather_member_id"]
        if "rcp_2_6" in member:
            rcp = "rcp_2_6"
        elif "rcp_4_5" in member:
            rcp = "rcp_4_5"
        elif "rcp_8_5" in member:
            rcp = "rcp_8_5"
        else:
            continue
        diagnostics[rcp] = pd.read_csv(partition / "run_diagnostics.csv")
    if set(diagnostics) != set(RCP_ORDER):
        raise RuntimeError(f"Expected three future-weather partitions, found {sorted(diagnostics)}")
    return diagnostics


def load_national_2050_weights() -> pd.DataFrame:
    path = (
        ROOT
        / "BE_building_stock"
        / "data"
        / "scenarios"
        / "renovation"
        / "archetype_matrix_2050_renovation_scenarios.csv"
    )
    frame = pd.read_csv(path)
    frame = frame.loc[frame["scenario"].eq("central")]
    return (
        frame.groupby(["archetype_id", "state_id"], as_index=False)["state_dwellings_2050"]
        .sum()
        .rename(columns={"state_dwellings_2050": "weight_dwellings"})
    )


def plot_hourly_temperature() -> None:
    observed_path = (
        ROOT
        / "climate"
        / "data"
        / "processed"
        / "observed"
        / "pvgis_sarah3_horizontal_hourly_2006_2023.csv"
    )
    observed = pd.read_csv(observed_path, parse_dates=["timestamp_utc"])
    observed = observed.loc[observed["timestamp_utc"].dt.year.eq(2015)].copy()

    day = np.arange(len(observed)) / 24.0
    historical = observed["T_out_C"].to_numpy(dtype=float)
    colours = {"rcp_2_6": "#2b78b8", "rcp_4_5": "#e28b2d", "rcp_8_5": "#a74745"}
    future_series: dict[str, np.ndarray] = {}
    for rcp in RCP_ORDER:
        path = (
            ROOT
            / "climate"
            / "data"
            / "processed"
            / "ensemble_2050"
            / rcp
            / f"weather_2050_{rcp}_pvgis_2015.csv"
        )
        future = pd.read_csv(path)
        future_series[rcp] = future["T_out_C"].to_numpy(dtype=float)

    month_starts = pd.date_range("2015-01-01", "2016-01-01", freq="MS", inclusive="left")
    month_ticks = (month_starts.dayofyear - 1).to_numpy()[::2]
    month_labels = MONTH_LABELS[::2]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.8), sharex=True, sharey=True)

    historical_axis = axes[0, 0]
    historical_axis.plot(day, historical, color="#555555", linewidth=0.58)
    historical_axis.set_title(
        f"(a) Historical 2015 | mean {historical.mean():.2f} $^\circ$C",
        loc="left",
    )

    for ax, rcp, panel in zip(axes.flat[1:], RCP_ORDER, ["b", "c", "d"]):
        values = future_series[rcp]
        ax.plot(day, historical, color="#a9a9a9", linewidth=0.48, alpha=0.86)
        ax.fill_between(day, historical, values, color=colours[rcp], alpha=0.20, linewidth=0)
        ax.plot(day, values, color=colours[rcp], linewidth=0.62)
        ax.set_title(
            f"({panel}) {RCP_LABELS[rcp]} | mean {values.mean():.2f} $^\circ$C",
            loc="left",
        )

    for ax in axes.flat:
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)
        ax.set_xlim(0, 364)
        ax.set_ylim(-6, 36)
        ax.grid(axis="x", visible=False)
    fig.supylabel("Outdoor temperature ($^\circ$C)")
    fig.supxlabel("Month in the retained 2015 chronology")
    fig.legend(
        handles=[
            Line2D([0], [0], color="#a9a9a9", linewidth=1.1, label="Historical reference"),
            Line2D([0], [0], color=colours["rcp_2_6"], linewidth=1.3, label="RCP2.6"),
            Line2D([0], [0], color=colours["rcp_4_5"], linewidth=1.3, label="RCP4.5"),
            Line2D([0], [0], color=colours["rcp_8_5"], linewidth=1.3, label="RCP8.5"),
        ],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
    )
    fig.subplots_adjust(hspace=0.30, wspace=0.10, top=0.88, bottom=0.12, left=0.10)
    save_figure(fig, "fig_results_hourly_temperature_2050")


def plot_hourly_solar_irradiance() -> None:
    observed_path = (
        ROOT
        / "climate"
        / "data"
        / "processed"
        / "observed"
        / "pvgis_sarah3_horizontal_hourly_2006_2023.csv"
    )
    observed = pd.read_csv(observed_path, parse_dates=["timestamp_utc"])
    observed = observed.loc[observed["timestamp_utc"].dt.year.eq(2015)].copy()

    day = np.arange(len(observed)) / 24.0
    historical = observed["I_solar_W_m2"].to_numpy(dtype=float)
    colours = {"rcp_2_6": "#2b78b8", "rcp_4_5": "#e28b2d", "rcp_8_5": "#a74745"}
    future_series: dict[str, np.ndarray] = {}
    for rcp in RCP_ORDER:
        path = (
            ROOT
            / "climate"
            / "data"
            / "processed"
            / "ensemble_2050"
            / rcp
            / f"weather_2050_{rcp}_pvgis_2015.csv"
        )
        future = pd.read_csv(path)
        future_series[rcp] = future["I_solar_W_m2"].to_numpy(dtype=float)

    month_starts = pd.date_range("2015-01-01", "2016-01-01", freq="MS", inclusive="left")
    month_ticks = (month_starts.dayofyear - 1).to_numpy()[::2]
    month_labels = MONTH_LABELS[::2]
    upper_limit = np.ceil(
        max(historical.max(), *(values.max() for values in future_series.values())) / 100.0
    ) * 100.0
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.8), sharex=True, sharey=True)

    historical_axis = axes[0, 0]
    historical_axis.plot(day, historical, color="#555555", linewidth=0.48)
    historical_axis.set_title(
        f"(a) Historical 2015 | mean {historical.mean():.1f} W/m$^2$",
        loc="left",
    )

    for ax, rcp, panel in zip(axes.flat[1:], RCP_ORDER, ["b", "c", "d"]):
        values = future_series[rcp]
        ax.plot(day, historical, color="#a9a9a9", linewidth=0.40, alpha=0.86)
        ax.fill_between(day, historical, values, color=colours[rcp], alpha=0.20, linewidth=0)
        ax.plot(day, values, color=colours[rcp], linewidth=0.52)
        ax.set_title(
            f"({panel}) {RCP_LABELS[rcp]} | mean {values.mean():.1f} W/m$^2$",
            loc="left",
        )

    for ax in axes.flat:
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)
        ax.set_xlim(0, 364)
        ax.set_ylim(0, upper_limit)
        ax.grid(axis="x", visible=False)
    fig.supylabel("Global horizontal irradiance (W/m$^2$)")
    fig.supxlabel("Month in the retained 2015 chronology")
    fig.legend(
        handles=[
            Line2D([0], [0], color="#a9a9a9", linewidth=1.1, label="Historical reference"),
            Line2D([0], [0], color=colours["rcp_2_6"], linewidth=1.3, label="RCP2.6"),
            Line2D([0], [0], color=colours["rcp_4_5"], linewidth=1.3, label="RCP4.5"),
            Line2D([0], [0], color=colours["rcp_8_5"], linewidth=1.3, label="RCP8.5"),
        ],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
    )
    fig.subplots_adjust(hspace=0.30, wspace=0.10, top=0.88, bottom=0.12, left=0.10)
    save_figure(fig, "fig_results_hourly_solar_irradiance_2050")


def plot_dwelling_state_distributions(diagnostics: dict[str, pd.DataFrame], weights: pd.DataFrame) -> None:
    frame = diagnostics["rcp_4_5"].merge(weights, on=["archetype_id", "state_id"], how="left", validate="many_to_one")
    frame = frame.loc[frame["weight_dwellings"].gt(0)].copy()
    metrics = [
        ("heating_intensity_kWh_m2", "Heating intensity", "kWh/(m$^2$ yr)", 1.0),
        ("cooling_intensity_kWh_m2", "Potential cooling intensity", "kWh/(m$^2$ yr)", 1.0),
        ("peak_heating_W", "Individual heating peak", "kW per dwelling", 1 / 1000),
        ("peak_cooling_W", "Individual cooling peak", "kW per dwelling", 1 / 1000),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0))
    for ax, (metric, title, ylabel, scale) in zip(axes.flat, metrics):
        for position, state in enumerate(STATE_ORDER):
            state_frame = frame.loc[frame["state_id"].eq(state)]
            p05, median, p95 = inverse_weighted_quantiles(
                state_frame[metric] * scale,
                state_frame["weight_dwellings"],
            )
            ax.errorbar(
                position,
                median,
                yerr=np.array([[median - p05], [p95 - median]]),
                fmt="o",
                markersize=5.5,
                capsize=3.5,
                color=STATE_COLOURS[state],
                ecolor=STATE_COLOURS[state],
                linewidth=1.5,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(STATE_ORDER)))
        ax.set_xticklabels([STATE_LABELS[state] for state in STATE_ORDER])
        ax.grid(axis="x", visible=False)
    fig.subplots_adjust(hspace=0.42, wspace=0.32)
    save_figure(fig, "fig_results_dwelling_state_distributions")


def plot_hourly_stock_demand() -> None:
    path = (
        ROOT
        / "thermal_model"
        / "data"
        / "monte_carlo"
        / "current_vs_2050"
        / "factorial_stock_hourly_all_weather_and_weights.csv"
    )
    frame = pd.read_csv(path, parse_dates=["timestamp_utc"])
    frame = frame.loc[
        frame["stock_year"].eq(2050)
        & frame["region"].eq("Belgium_modelled_stock")
        & frame["weather_member_id"].str.contains("rcp_")
    ].copy()
    fig, (heating_axis, cooling_axis) = plt.subplots(
        2,
        1,
        figsize=(7.25, 4.6),
        sharex=True,
    )
    member = "weather_2050_rcp_4_5_pvgis_2015"
    values = frame.loc[frame["weather_member_id"].eq(member)].sort_values("timestamp_utc")
    day = np.arange(len(values)) / 24.0
    heating_axis.plot(
        day,
        values["heating_demand_MW"] / 1000,
        color="#c65f42",
        linewidth=0.72,
    )
    cooling_axis.plot(
        day,
        values["potential_sensible_cooling_demand_MW"] / 1000,
        color="#2879b9",
        linewidth=0.72,
    )

    month_starts = pd.date_range("2015-01-01", "2016-01-01", freq="MS", inclusive="left")
    month_ticks = (month_starts.dayofyear - 1).to_numpy()[::2]
    for ax in (heating_axis, cooling_axis):
        ax.set_xlim(0, 364)
        ax.set_ylim(bottom=0)
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(MONTH_LABELS[::2])
        ax.grid(axis="x", visible=False)
    heating_axis.set_title("(a) Useful heating | RCP4.5", loc="left")
    cooling_axis.set_title("(b) Potential cooling | RCP4.5", loc="left")
    heating_axis.set_ylabel("National heating power (GW)", color="#a64f37")
    cooling_axis.set_ylabel("National cooling power (GW)", color="#216797")
    heating_axis.tick_params(axis="y", colors="#a64f37")
    cooling_axis.tick_params(axis="y", colors="#216797")
    fig.supxlabel("Month in the retained 2015 chronology")
    fig.subplots_adjust(hspace=0.32, top=0.95, bottom=0.13, left=0.11, right=0.98)
    save_figure(fig, "fig_results_hourly_stock_demand")


def plot_stock_summary() -> None:
    aggregation_path = (
        ROOT
        / "thermal_model"
        / "data"
        / "monte_carlo"
        / "supervisor_results_preliminary"
        / "stock_aggregation.csv"
    )
    aggregation = pd.read_csv(aggregation_path)
    aggregation = aggregation.loc[
        aggregation["region"].eq("Belgium_modelled_stock")
    ].set_index("climate_scenario_id").loc[RCP_ORDER]

    case_path = (
        ROOT
        / "thermal_model"
        / "data"
        / "monte_carlo"
        / "current_vs_2050"
        / "factorial_annual_case_summary.csv"
    )
    cases = pd.read_csv(case_path)
    cases = cases.loc[cases["case_id"].eq("Q11")]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.8))
    x = np.arange(len(RCP_ORDER))
    width = 0.34
    peak_specs = [
        (
            axes[0, 0],
            "coincident_peak_heating_MW",
            "sum_individual_peak_heating_MW",
            "(a) Heating peaks",
            "#b9553d",
            "#e1a08e",
        ),
        (
            axes[0, 1],
            "coincident_peak_potential_cooling_MW",
            "sum_individual_peak_potential_cooling_MW",
            "(b) Cooling peaks",
            "#2778b4",
            "#91bddc",
        ),
    ]
    for ax, coincident_column, individual_column, title, dark_colour, light_colour in peak_specs:
        coincident_bars = ax.bar(
            x - width / 2,
            aggregation[coincident_column].to_numpy() / 1000,
            width,
            color=dark_colour,
            edgecolor=dark_colour,
        )
        individual_bars = ax.bar(
            x + width / 2,
            aggregation[individual_column].to_numpy() / 1000,
            width,
            color=light_colour,
            edgecolor=dark_colour,
            hatch="//",
            linewidth=0.7,
        )
        ax.bar_label(coincident_bars, fmt="%.1f", padding=2, fontsize=6.8)
        ax.bar_label(individual_bars, fmt="%.1f", padding=2, fontsize=6.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["2.6", "4.5", "8.5"])
        ax.set_xlabel("RCP")
        ax.set_ylim(0, max(aggregation[individual_column]) / 1000 * 1.18)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("National thermal peak (GW)")

    fig.legend(
        handles=[
            Patch(facecolor="#666666", edgecolor="#666666", label="Coincident national peak"),
            Patch(
                facecolor="#d0d0d0",
                edgecolor="#666666",
                hatch="//",
                label="Sum of individual dwelling peaks",
            ),
        ],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=8.2,
    )

    interval_specs = [
        (axes[1, 0], "annual_heating_TWh", "(c) Annual heating demand", "#b9553d", (68, 79)),
        (axes[1, 1], "annual_potential_sensible_cooling_TWh", "(d) Annual potential-cooling demand", "#2778b4", (2.3, 3.45)),
    ]
    for ax, metric, title, colour, limits in interval_specs:
        subset = cases.loc[cases["metric"].eq(metric)].set_index("climate_scenario_id").loc[RCP_ORDER]
        y = np.arange(len(RCP_ORDER))[::-1]
        medians = subset["median"].to_numpy()
        errors = np.vstack((medians - subset["p05"].to_numpy(), subset["p95"].to_numpy() - medians))
        ax.errorbar(
            medians,
            y,
            xerr=errors,
            fmt="o",
            markersize=5.5,
            color=colour,
            ecolor=colour,
            capsize=4,
            linewidth=1.6,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([RCP_LABELS[rcp] for rcp in RCP_ORDER])
        ax.set_xlim(*limits)
        ax.set_xlabel("TWh/year")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.32, top=0.88, bottom=0.12, left=0.11, right=0.98)
    save_figure(fig, "fig_results_stock_peaks_and_seed_intervals")


def latex_period(value: str) -> str:
    return value.replace("pre-", "pre-").replace("post-", "post-").replace("-", "--")


def generate_appendix_table(diagnostics: dict[str, pd.DataFrame], weights: pd.DataFrame) -> None:
    frame = diagnostics["rcp_4_5"].merge(weights, on=["archetype_id", "state_id"], how="left", validate="many_to_one")
    group_columns = [
        "archetype_id",
        "dwelling_type",
        "construction_period",
        "state_id",
        "floor_area_m2",
        "weight_dwellings",
    ]
    medians = (
        frame.groupby(group_columns, as_index=False)[
            [
                "annual_heating_kWh",
                "heating_intensity_kWh_m2",
                "peak_heating_W",
                "annual_cooling_kWh",
                "cooling_intensity_kWh_m2",
                "peak_cooling_W",
            ]
        ].median()
    )
    medians["cell_number"] = medians["archetype_id"].str.extract(r"(\d+)$").astype(int)
    medians["state_order"] = medians["state_id"].map({state: number for number, state in enumerate(STATE_ORDER)})
    medians = medians.sort_values(["cell_number", "state_order"])

    type_labels = {
        "Detached house": "Detached",
        "Semi-detached house": "Semi-detached",
        "Terraced house": "Terraced",
        "Apartment, enclosed": "Apt., enclosed",
        "Apartment, exposed": "Apt., exposed",
    }
    header = (
        r"No. & Type & Period & State & \shortstack{2050 stock\\($10^3$ dwell.)} & "
        r"\shortstack{$E_H$\\(MWh/yr)} & \shortstack{$e_H$\\(kWh/m$^2$yr)} & "
        r"\shortstack{$P_H^{\mathrm{ind}}$\\(kW)} & \shortstack{$E_C$\\(MWh/yr)} & "
        r"\shortstack{$e_C$\\(kWh/m$^2$yr)} & \shortstack{$P_C^{\mathrm{ind}}$\\(kW)} \\"
    )
    lines = [
        "% Auto-generated by scripts/generate_results_chapter_artifacts.py -- do not edit by hand.",
        "% Requires: \\usepackage{longtable, booktabs, pdflscape, amsmath}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.4pt}",
        "\\begin{longtable}{rlllr*{6}{r}}",
        "\\caption{Dwelling-level thermal-demand results for all 75 physical archetype--state cells under the representative RCP4.5 2050 weather member. Entries are medians across 160 occupant seeds. The stock column gives the projected 2050 national dwelling weight; zero-weight cells were simulated but do not contribute to $Q_{11}$.}",
        "\\label{tab:appendix_dwelling_level_results}\\\\",
        "\\toprule",
        header,
        "\\midrule",
        "\\endfirsthead",
        r"\multicolumn{11}{c}{\tablename\ \thetable\ -- continued} \\",
        "\\toprule",
        header,
        "\\midrule",
        "\\endhead",
        r"\midrule \multicolumn{11}{r}{\textit{continued on next page}} \\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for row in medians.itertuples(index=False):
        lines.append(
            f"{row.cell_number} & {type_labels[row.dwelling_type]} & {latex_period(row.construction_period)} & "
            f"{STATE_LABELS[row.state_id]} & {row.weight_dwellings / 1000:.1f} & "
            f"{row.annual_heating_kWh / 1000:.2f} & {row.heating_intensity_kWh_m2:.1f} & "
            f"{row.peak_heating_W / 1000:.2f} & {row.annual_cooling_kWh / 1000:.3f} & "
            f"{row.cooling_intensity_kWh_m2:.2f} & {row.peak_cooling_W / 1000:.2f} \\\\"
        )
    lines.extend(
        [
            "\\end{longtable}",
            "\\noindent\\scriptsize $E_H$ and $E_C$ are annual useful heating and potential sensible-cooling energy per dwelling; $e_H$ and $e_C$ are the corresponding floor-area intensities; $P_H^{\\mathrm{ind}}$ and $P_C^{\\mathrm{ind}}$ are individual dwelling peaks. Cooling assumes universal ideal control at 26~$^\\circ$C.",
            "",
        ]
    )
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    diagnostics = load_diagnostics()
    weights = load_national_2050_weights()
    plot_hourly_temperature()
    plot_hourly_solar_irradiance()
    plot_dwelling_state_distributions(diagnostics, weights)
    plot_hourly_stock_demand()
    plot_stock_summary()
    generate_appendix_table(diagnostics, weights)


if __name__ == "__main__":
    main()
