"""
Belgian residential final energy consumption by end-use.

Generates three candidate figures for the thesis "Residential Energy Demand"
section, drawing on the two authoritative sources for Belgium:

  1. Eurostat, household energy consumption by end-use (dataset nrg_d_hhq).
     Full end-use split shown for the most recent complete year (2020);
     the 2024 edition reports a space-heating share of 71.2 % for Belgium
     (3rd highest in the EU, after Luxembourg and Estonia).
  2. ODYSSEE-MURE, sectoral profile - households, Belgium (2022).

The two sources differ slightly because of methodological differences
(ODYSSEE normalises to climate and reallocates some end-uses), so both are
provided. The grouped comparison (figure 3) is the recommended primary.

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
# For reference, the published space-heating headline has since edged down:
# 70.8 % in 2023 and 71.2 % in the 2024 edition.
eurostat = {
    "Space heating": 72.7,
    "Water heating": 11.7,
    "Lighting & appliances": 13.2,
    "Cooking": 1.7,
    "Space cooling": 0.1,
    "Other": 0.6,
}
# ODYSSEE-MURE, households, Belgium 2022.
odyssee = {
    "Space heating": 74.0,
    "Water heating": 14.0,
    "Lighting & appliances": 11.0,
    "Cooking": 1.7,
    "Space cooling": 0.0,
    "Other": 0.0,
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


def horizontal_bar(data, title, subtitle, fname):
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
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.02, 0.015, subtitle, fontsize=7.5, color="#666666",
             ha="left", va="bottom")
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def grouped_compare(fname):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(labels))
    w = 0.38
    e_vals = [eurostat[k] for k in labels]
    o_vals = [odyssee[k] for k in labels]
    b1 = ax.bar(x - w / 2, e_vals, w, label="Eurostat (nrg_d_hhq), 2020",
                color="#b5482f", edgecolor="white")
    b2 = ax.bar(x + w / 2, o_vals, w, label="ODYSSEE-MURE, 2022",
                color="#4a7ba6", edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.8,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Share of household\nfinal energy (%)")
    ax.set_ylim(0, 82)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


horizontal_bar(
    eurostat,
    "Belgian residential final energy consumption by end-use",
    "Source: Eurostat, household energy consumption (nrg_d_hhq), 2020. "
    "2024 edition: space heating = 71.2 %.",
    "fig_be_enduse_eurostat",
)
horizontal_bar(
    odyssee,
    "Belgian residential final energy consumption by end-use",
    "Source: ODYSSEE-MURE, sectoral profile - households, Belgium, 2022.",
    "fig_be_enduse_odyssee",
)
grouped_compare("fig_be_enduse_compare")

print("Wrote figures to", OUT)
for f in sorted(OUT.glob("*")):
    print(" -", f.name)
