"""Generate figures for TABULA envelope parameters and infiltration conversion."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "BE_building_stock" / "data"
PHYSICAL = DATA / "matrices" / "national" / "base_physical_archetype_matrix.csv"
PACKAGES = (
    DATA
    / "assumptions"
    / "renovation"
    / "renovation_physical_state_packages_TABULA.csv"
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

PERIOD_ORDER = ["pre-1946", "1946-1970", "1971-1990", "1991-2005", "post-2005"]
physical = pd.read_csv(PHYSICAL)
packages = pd.read_csv(PACKAGES).set_index("state_id")

u_values = (
    physical.drop_duplicates("construction_period")
    .set_index("construction_period")
    .loc[
        PERIOD_ORDER,
        [
            "U_facade_W_m2K",
            "U_roof_W_m2K",
            "U_floor_W_m2K",
            "U_window_W_m2K",
            "U_door_W_m2K",
        ],
    ]
)
normal_airflow = (
    physical.groupby("construction_period")["infiltration_airflow_normal_m3_h"]
    .mean()
    .loc[PERIOD_ORDER]
)

components = {
    "U_facade_W_m2K": "Facade",
    "U_roof_W_m2K": "Roof",
    "U_floor_W_m2K": "Floor",
    "U_window_W_m2K": "Window",
    "U_door_W_m2K": "Door",
}
colors = ["#2c6e8f", "#5aa9c9", "#a8d5e2", "#e8a33d", "#7a6f9b"]
fig, (axis_u, axis_inf) = plt.subplots(
    1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2, 1]}
)
x = np.arange(len(PERIOD_ORDER))
width = 0.18
for index, (column, label) in enumerate(components.items()):
    axis_u.bar(
        x + (index - 2) * width,
        u_values[column],
        width,
        label=label,
        color=colors[index],
    )
axis_u.set_xticks(x)
axis_u.set_xticklabels(PERIOD_ORDER, rotation=20, ha="right")
axis_u.set_ylabel("U-value (W/m²K)")
axis_u.set_title("Existing-state envelope parameters")
axis_u.legend(frameon=False, fontsize=9, ncol=2)
axis_u.grid(axis="y", alpha=0.25)

axis_inf.plot(x, normal_airflow, "o-", color="#c95f45", linewidth=2)
for x_value, value in zip(x, normal_airflow):
    axis_inf.annotate(
        f"{value:.2f}",
        (x_value, value),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
axis_inf.set_xticks(x)
axis_inf.set_xticklabels(PERIOD_ORDER, rotation=20, ha="right")
axis_inf.set_ylabel("Mean infiltration airflow proxy (m³/h)")
axis_inf.set_title("Normal-pressure infiltration airflow: q₅₀ / 20")
axis_inf.grid(axis="y", alpha=0.25)
fig.tight_layout()
for extension in ("png", "pdf"):
    fig.savefig(OUT / f"fig_uvalues_by_period.{extension}", bbox_inches="tight")
plt.close(fig)


# Compare the three physical state definitions.
existing = physical[
    physical["construction_period"] == "pre-1946"
].iloc[0]
standard = packages.loc["TABULA_standard_B_proxy"]
advanced = packages.loc["TABULA_advanced_A_proxy"]
labels = ["Facade", "Roof", "Floor", "Window", "Door"]
fields = [
    "U_facade_W_m2K",
    "U_roof_W_m2K",
    "U_floor_W_m2K",
    "U_window_W_m2K",
    "U_door_W_m2K",
]
series = {
    "Existing pre-1946 example": [float(existing[field]) for field in fields],
    "TABULA standard / B proxy": [float(standard[field]) for field in fields],
    "TABULA advanced / A proxy": [float(advanced[field]) for field in fields],
}
series_colors = ["#c95f45", "#e8b04a", "#3f8f70"]
fig, axis = plt.subplots(figsize=(8.8, 4.8))
x = np.arange(len(labels))
width = 0.26
for index, ((label, values), color) in enumerate(zip(series.items(), series_colors)):
    bars = axis.bar(
        x + (index - 1) * width,
        values,
        width,
        label=label,
        color=color,
    )
    axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
axis.set_xticks(x)
axis.set_xticklabels(labels)
axis.set_ylabel("U-value (W/m²K)")
axis.set_title("Envelope parameters of the three TABULA model states")
axis.legend(frameon=False, fontsize=9)
axis.grid(axis="y", alpha=0.25)
fig.tight_layout()
for extension in ("png", "pdf"):
    fig.savefig(
        OUT / f"fig_renovation_thermal_effect.{extension}",
        bbox_inches="tight",
    )
plt.close(fig)

print("Saved U-value, infiltration-conversion, and state-package figures.")
