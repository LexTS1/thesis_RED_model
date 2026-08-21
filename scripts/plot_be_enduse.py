"""
Belgian residential final energy consumption by end-use.

Generates the thesis figure for the "Residential Energy Demand" section from
Eurostat household energy consumption by end-use (dataset nrg_d_hhq), using the
complete Belgian end-use split for reference year 2020.

Outputs: PDF (vector, for \\includegraphics) + PNG (preview) in ../figures/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# --- Data -------------------------------------------------------------------
# Eurostat nrg_d_hhq, Belgium 2020 (most recent complete end-use split).
eurostat = {
    "Space heating": 72.7,
    "Water heating": 11.7,
    "Lighting & appliances": 13.2,
    "Cooking": 1.7,
    "Space cooling": 0.1,
    "Other": 0.6,
}
# Muted, print-friendly palette (space heating emphasised).
COLORS = {
    "Space heating": "#b5482f",
    "Water heating": "#e08a3c",
    "Lighting & appliances": "#4a7ba6",
    "Cooking": "#7aa66b",
    "Space cooling": "#8c8c8c",
    "Other": "#cccccc",
}
labels = list(eurostat.keys())


def horizontal_bar(data, title, fname):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    vals = [data[k] for k in labels]
    ypos = np.arange(len(labels))[::-1]
    bars = ax.barh(ypos, vals, color=[COLORS[k] for k in labels],
                   edgecolor="white", height=0.72)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Share of household final energy consumption (%)")
    ax.set_xlim(0, max(vals) * 1.14)
    for b, v in zip(bars, vals):
        if v > 0:
            ax.text(b.get_width() + max(vals) * 0.015,
                    b.get_y() + b.get_height() / 2,
                    f"{v:.1f}%", va="center", ha="left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


horizontal_bar(
    eurostat,
    "Belgian residential final energy consumption by end-use",
    "fig_be_enduse_eurostat",
)

print("Wrote figures to", OUT)
for f in sorted(OUT.glob("*")):
    print(" -", f.name)
