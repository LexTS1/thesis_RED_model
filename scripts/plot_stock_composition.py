"""Generate building-stock composition figures from the derived Statbel CSVs.

Outputs (300 dpi PNG + PDF) to BE_building_stock/figures/:
  fig_dwelling_type_composition.{png,pdf}
  fig_construction_period.{png,pdf}
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "BE_building_stock" / "data" / "derived"
OUT = ROOT / "BE_building_stock" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

REGIONS = ["Flanders", "Wallonia", "Brussels", "Belgium"]

# ---------------------------------------------------------------------------
# Figure 1: dwelling-type composition per region (share of all R1-R6 dwellings)
# ---------------------------------------------------------------------------
split = pd.read_csv(DATA / "regional_stock" / "regional_dwelling_type_split.csv")
split["region"] = split["region"].map({
    "Flemish Region": "Flanders",
    "Walloon Region": "Wallonia",
    "Brussels-Capital Region": "Brussels",
})
cols = {
    "terraced_dwellings_R1": "R1 terraced",
    "semi_detached_dwellings_R2": "R2 semi-detached",
    "detached_dwellings_R3": "R3 detached",
    "apartment_dwellings_R4": "R4 apartment",
    "commercial_other_dwellings_R5_R6": "R5+R6 excluded",
}
# add Belgium as sum
be = split[list(cols)].sum()
be["region"] = "Belgium"
split = pd.concat([split, pd.DataFrame([be])], ignore_index=True)
split = split.set_index("region").loc[REGIONS]

# convert to % of total dwellings in region
pct = split[list(cols)].div(split[list(cols)].sum(axis=1), axis=0) * 100
pct.columns = list(cols.values())

colors = ["#2c6e8f", "#5aa9c9", "#a8d5e2", "#e8a33d", "#c0c0c0"]
fig, ax = plt.subplots(figsize=(8, 4.2))
left = pd.Series(0.0, index=pct.index)
for c, col in zip(pct.columns, colors):
    ax.barh(pct.index, pct[c], left=left, color=col, label=c, edgecolor="white", linewidth=0.6)
    for reg in pct.index:
        v = pct.loc[reg, c]
        if v > 3.5:
            ax.text(left[reg] + v / 2, reg, f"{v:.0f}%", va="center", ha="center",
                    color="white" if c != "R5+R6 excluded" else "#555", fontsize=9)
    left += pct[c]
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(PercentFormatter())
ax.set_xlabel("Share of regional dwelling stock")
ax.set_title("Belgian residential dwelling-stock composition by type (Statbel 2025)", fontsize=12, pad=12)
ax.invert_yaxis()
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_dwelling_type_composition.{ext}", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: construction-period distribution per region
# ---------------------------------------------------------------------------
cp = pd.read_csv(DATA / "construction_periods" / "dwellings_by_construction_period.csv")
periods = cp["Construction-period class"].tolist()
region_share = {
    "Flanders": cp["Flanders share (%)"],
    "Wallonia": cp["Wallonia share (%)"],
    "Brussels": cp["Brussels share (%)"],
    "Belgium": cp["Belgium share (%)"],
}
import numpy as np
x = np.arange(len(periods))
w = 0.2
fig, ax = plt.subplots(figsize=(9, 4.6))
rcolors = {"Flanders": "#f2c14e", "Wallonia": "#e07a5f", "Brussels": "#3d6480", "Belgium": "#555555"}
for i, (reg, vals) in enumerate(region_share.items()):
    ax.bar(x + (i - 1.5) * w, vals, w, label=reg, color=rcolors[reg])
ax.set_xticks(x)
ax.set_xticklabels(periods, rotation=25, ha="right")
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_ylabel("Share of dwellings (known period)")
ax.set_title(
    "Descriptive dwelling stock by construction period and region (Census 2021)",
    fontsize=12,
    pad=12,
)
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_construction_period.{ext}", bbox_inches="tight")
plt.close(fig)

print("Saved:", *[p.name for p in sorted(OUT.glob("fig_dwelling_type_composition.*"))],
      *[p.name for p in sorted(OUT.glob("fig_construction_period.*"))])
