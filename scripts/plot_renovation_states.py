"""Generate figures for the single national 2050 renovation projection."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_FILE = (
    ROOT
    / "BE_building_stock"
    / "data"
    / "scenarios"
    / "renovation"
    / "archetype_matrix_2050_renovation_scenarios.csv"
)
TRAJECTORY_FILE = (
    ROOT
    / "BE_building_stock"
    / "data"
    / "scenarios"
    / "renovation"
    / "renovation_state_trajectory_2025_2050.csv"
)
OUT = ROOT / "BE_building_stock" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    }
)

data = pd.read_csv(SCENARIO_FILE)
trajectory = pd.read_csv(TRAJECTORY_FILE)
projections = data["scenario"].drop_duplicates().tolist()
if projections != ["central"]:
    raise ValueError(f"Expected the canonical central projection; found {projections}")
if trajectory["projection"].drop_duplicates().tolist() != ["central"]:
    raise ValueError("Annual trajectory does not contain the canonical projection")

STATE_ORDER = [
    "TABULA_existing",
    "TABULA_standard_B_proxy",
    "TABULA_advanced_A_proxy",
]
STATE_LABEL = {
    "TABULA_existing": "Existing / as-is",
    "TABULA_standard_B_proxy": "Standard refurbishment",
    "TABULA_advanced_A_proxy": "Low-energy refurbishment",
}
STATE_COLOR = {
    "TABULA_existing": "#c95f45",
    "TABULA_standard_B_proxy": "#e8b04a",
    "TABULA_advanced_A_proxy": "#3f8f70",
}
REGION_ORDER = ["Flemish Region", "Walloon Region", "Brussels-Capital Region"]
REGION_LABEL = {
    "Flemish Region": "Flanders",
    "Walloon Region": "Wallonia",
    "Brussels-Capital Region": "Brussels",
}
REGION_COLOR = {
    "Flemish Region": "#d8a800",
    "Walloon Region": "#c44e52",
    "Brussels-Capital Region": "#3572b0",
}
PERIOD_ORDER = ["pre-1946", "1946-1970", "1971-1990", "1991-2005", "post-2005"]


def state_composition(frame: pd.DataFrame, count_column: str) -> list[float]:
    total = float(frame[count_column].sum())
    return [
        100 * float(frame.loc[frame["state_id"] == state, count_column].sum()) / total
        for state in STATE_ORDER
    ]


# Calibrated 2025 composition and the resulting 2050 composition
scopes: list[tuple[str, pd.DataFrame]] = [("Belgium", data)] + [
    (REGION_LABEL[region], data[data["region"] == region]) for region in REGION_ORDER
]
fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharey=True)
for axis, (title, frame) in zip(axes.flat, scopes):
    values_by_year = np.asarray(
        [
            state_composition(frame, "initial_state_dwellings_2025"),
            state_composition(frame, "state_dwellings_2050"),
        ]
    )
    bottom = np.zeros(2)
    for state_index, state in enumerate(STATE_ORDER):
        values = values_by_year[:, state_index]
        bars = axis.bar(
            np.arange(2),
            values,
            bottom=bottom,
            color=STATE_COLOR[state],
            label=STATE_LABEL[state],
        )
        for bar, value, base in zip(bars, values, bottom):
            if value >= 5:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        bottom += values
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["2025 calibration", "2050 projection"])
    axis.set_title(title)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
for axis in axes[:, 0]:
    axis.set_ylabel("Share of modelled R1-R4 stock")
    axis.yaxis.set_major_formatter(PercentFormatter())
handles, labels = axes.flat[-1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    frameon=False,
)
fig.suptitle(
    "Renovation-state projection with 2.8% annual activity and a 40/50/10 depth distribution",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
for extension in ("png", "pdf"):
    fig.savefig(
        OUT / f"fig_renovation_state_projection_2050.{extension}",
        bbox_inches="tight",
    )
plt.close(fig)


# Thesis figure: national renovation-state composition.
fig, state_axis = plt.subplots(figsize=(5.4, 4.6))
national_by_year = np.asarray(
    [
        state_composition(data, "initial_state_dwellings_2025"),
        state_composition(data, "state_dwellings_2050"),
    ]
)
bottom = np.zeros(2)
for state_index, state in enumerate(STATE_ORDER):
    values = national_by_year[:, state_index]
    bars = state_axis.bar(
        np.arange(2),
        values,
        bottom=bottom,
        width=0.62,
        color=STATE_COLOR[state],
        label=STATE_LABEL[state],
    )
    for bar, value, base in zip(bars, values, bottom):
        if value >= 5:
            state_axis.text(
                bar.get_x() + bar.get_width() / 2,
                base + value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )
    bottom += values
state_axis.set_xticks([0, 1])
state_axis.set_xticklabels(["2025", "2050"])
state_axis.set_ylim(0, 100)
state_axis.yaxis.set_major_formatter(PercentFormatter())
state_axis.set_ylabel("Share of Belgian modelled R1-R4 stock")
state_axis.grid(axis="y", alpha=0.25)
state_axis.legend(
    frameon=False,
    fontsize=8,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=1,
)

fig.tight_layout()
for extension in ("png", "pdf"):
    fig.savefig(
        OUT / f"fig_renovation_state_composition_be_2025_2050.{extension}",
        bbox_inches="tight",
    )
plt.close(fig)


# Thesis figure: annual regional share represented by Standard or Low-energy
# refurbishment.
fig, trajectory_axis = plt.subplots(figsize=(6.8, 4.5))
for region in REGION_ORDER:
    region_trajectory = trajectory[trajectory["region"] == region].sort_values("year")
    trajectory_axis.plot(
        region_trajectory["year"],
        100 * region_trajectory["improved_envelope_share"],
        color=REGION_COLOR[region],
        label=REGION_LABEL[region],
        linewidth=2.4,
        marker="o",
        markevery=5,
        markersize=3.5,
    )
trajectory_axis.set_xlim(2025, 2050)
trajectory_axis.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
trajectory_axis.set_ylim(0, 80)
trajectory_axis.yaxis.set_major_formatter(PercentFormatter())
trajectory_axis.set_ylabel("Share in Standard or Low-energy state")
trajectory_axis.grid(alpha=0.25)
trajectory_axis.legend(frameon=False, fontsize=8, loc="upper left")

fig.tight_layout()
for extension in ("png", "pdf"):
    fig.savefig(
        OUT / f"fig_renovation_improved_share_by_region_2025_2050.{extension}",
        bbox_inches="tight",
    )
plt.close(fig)


# Cumulative transition events relative to initially transition-eligible stock.
# Medium and advanced events may concern the same dwelling in different years.
eligible = data[data["state_id"].isin(STATE_ORDER[:2])].copy()
period_summary = (
    eligible.groupby("construction_period", as_index=False)
    .agg(
        initial_eligible=("initial_state_dwellings_2025", "sum"),
        medium_outflow=("state_outflow_to_standard_2025_2050", "sum"),
        advanced_outflow=("state_outflow_to_advanced_2025_2050", "sum"),
    )
    .set_index("construction_period")
    .loc[PERIOD_ORDER]
)
period_summary["medium_percent"] = (
    100 * period_summary["medium_outflow"] / period_summary["initial_eligible"]
)
period_summary["advanced_percent"] = (
    100 * period_summary["advanced_outflow"] / period_summary["initial_eligible"]
)

fig, axis = plt.subplots(figsize=(8.8, 4.8))
x = np.arange(len(PERIOD_ORDER))
axis.bar(
    x,
    period_summary["medium_percent"],
    0.62,
    label="Medium: Existing to Standard",
    color="#4292c6",
)
axis.bar(
    x,
    period_summary["advanced_percent"],
    0.62,
    bottom=period_summary["medium_percent"],
    label="Advanced: Existing/Standard to Low energy",
    color="#08519c",
    alpha=0.62,
    hatch="///",
)
axis.set_xticks(x)
axis.set_xticklabels(PERIOD_ORDER)
maximum = float(
    (period_summary["medium_percent"] + period_summary["advanced_percent"]).max()
)
axis.set_ylim(0, max(10, 1.12 * maximum))
axis.yaxis.set_major_formatter(PercentFormatter())
axis.set_ylabel("Transition events / initially eligible stock")
axis.set_xlabel("Construction period")
axis.set_title("Depth-resolved transition events under heat-loss priority")
axis.legend(frameon=False, fontsize=8)
axis.grid(axis="y", alpha=0.25)
fig.text(
    0.5,
    0.025,
    "A dwelling can contribute a medium event and an advanced event in different years. "
    "Post-2005 Existing remains eligible for Advanced renovation.",
    ha="center",
    fontsize=7.5,
)
fig.tight_layout(rect=(0, 0.065, 1, 1))
for extension in ("png", "pdf"):
    fig.savefig(
        OUT / f"fig_renovation_priority_by_period.{extension}",
        bbox_inches="tight",
    )
plt.close(fig)

print("Saved national projection, annual trajectory and transition-priority figures.")
