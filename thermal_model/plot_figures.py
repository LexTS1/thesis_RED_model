"""Create deterministic, publication-ready figures for the thermal model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parent
VALIDATION_ROOT = MODULE_ROOT / "data/validation"
DEFAULT_OUTPUT_DIR = MODULE_ROOT / "figures"
VALIDATION_RESULTS = VALIDATION_ROOT / "deterministic_archetype_validation.csv"
VALIDATION_SUMMARY = VALIDATION_ROOT / "validation_summary.json"
SENSITIVITY_RESULTS = VALIDATION_ROOT / "sensitivity_results.csv"
SENSITIVITY_SUMMARY = VALIDATION_ROOT / "sensitivity_summary.json"

STATE_ORDER = (
    "TABULA_existing",
    "TABULA_standard_B_proxy",
    "TABULA_advanced_A_proxy",
)
STATE_LABELS = {
    "TABULA_existing": "Existing",
    "TABULA_standard_B_proxy": "Standard (EPB2010 proxy)",
    "TABULA_advanced_A_proxy": "Advanced (LE proxy)",
}
STATE_COLORS = {
    "TABULA_existing": "#0072B2",
    "TABULA_standard_B_proxy": "#E69F00",
    "TABULA_advanced_A_proxy": "#009E73",
}
DWELLING_ORDER = (
    "Detached house",
    "Semi-detached house",
    "Terraced house",
    "Apartment, enclosed",
    "Apartment, exposed",
)
DWELLING_LABELS = {
    "Detached house": "Detached",
    "Semi-detached house": "Semi-detached",
    "Terraced house": "Terraced",
    "Apartment, enclosed": "Enclosed apt.",
    "Apartment, exposed": "Exposed apt.",
}
DWELLING_MARKERS = {
    "Detached house": "o",
    "Semi-detached house": "s",
    "Terraced house": "^",
    "Apartment, enclosed": "D",
    "Apartment, exposed": "P",
}
PERIOD_ORDER = ("pre-1946", "1946-1970", "1971-1990", "1991-2005", "post-2005")
PERIOD_LABELS = ("<1946", "1946–70", "1971–90", "1991–2005", ">2005")

SENSITIVITY_LABELS = {
    "mass_light": "Light thermal mass",
    "mass_heavy": "Heavy thermal mass",
    "shading_unshaded": "No fixed shading ($F_{sh}=1.0$)",
    "frame_fraction_0_2": "Window-frame fraction 0.20",
    "infiltration_half": "Infiltration ×0.5",
    "infiltration_one_and_half": "Infiltration ×1.5",
    "ventilation_ach_0_3": "Ventilation 0.3 h⁻¹",
    "ventilation_ach_0_6": "Ventilation 0.6 h⁻¹",
    "heating_setpoint_18": "Heating setpoint 18 °C",
    "heating_setpoint_22": "Heating setpoint 22 °C",
    "cooling_setpoint_24": "Cooling setpoint 24 °C",
    "cooling_setpoint_28": "Cooling setpoint 28 °C",
    "internal_gains_1_5": "Internal gains 1.5 W/m²",
    "internal_gains_4_5": "Internal gains 4.5 W/m²",
    "all_opaque_boundaries_exterior": "All opaque boundaries exterior",
    "solar_disabled": "Façade solar gains disabled",
    "advanced_hrv_disabled": "HRV disabled (advanced state)",
}


class FigureDataError(ValueError):
    """Raised when validated figure inputs are incomplete or inconsistent."""


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(MODULE_ROOT))
    except ValueError:
        return str(path)


def _load_inputs() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    results = pd.read_csv(VALIDATION_RESULTS)
    summary = json.loads(VALIDATION_SUMMARY.read_text(encoding="utf-8"))
    sensitivity = pd.read_csv(SENSITIVITY_RESULTS)
    sensitivity_summary = json.loads(
        SENSITIVITY_SUMMARY.read_text(encoding="utf-8")
    )

    if summary.get("verification_status") != "PASS":
        raise FigureDataError("Figures require a passing verification result.")
    if summary.get("validation_status") != "PASS":
        raise FigureDataError("Figures require a passing validation result.")
    if len(results) != 75 or results.duplicated(["archetype_id", "state_id"]).any():
        raise FigureDataError("Expected 75 unique archetype/state validation cells.")
    if set(results["state_id"]) != set(STATE_ORDER):
        raise FigureDataError("Validation results do not contain the three frozen states.")
    if set(results["dwelling_type"]) != set(DWELLING_ORDER):
        raise FigureDataError("Validation results contain an unexpected dwelling type.")
    if len(sensitivity) != int(sensitivity_summary.get("case_count", -1)):
        raise FigureDataError("Sensitivity table does not reconcile with its summary.")
    if not all(sensitivity_summary.get("directional_checks", {}).values()):
        raise FigureDataError("Sensitivity direction checks must pass before plotting.")
    if results.select_dtypes(include="number").isna().any().any():
        raise FigureDataError("Validation results contain missing numeric values.")
    if sensitivity.select_dtypes(include="number").isna().any().any():
        raise FigureDataError("Sensitivity results contain missing numeric values.")
    return results, summary, sensitivity, sensitivity_summary


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
        metadata={"Software": "thermal_model.plot_figures"},
    )
    figure.savefig(
        pdf_temp,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "thermal_model.plot_figures",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    png_temp.replace(png_path)
    pdf_temp.replace(pdf_path)
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path}


def _acceptance_limits(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tolerance = np.maximum(15.0, 0.30 * target)
    return np.maximum(0.0, target - tolerance), target + tolerance


def _agreement_figure(
    results: pd.DataFrame, summary: Mapping[str, Any]
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.7, 5.7), constrained_layout=True)
    comparison = summary["tabula_comparison"]
    minimum, maximum = 5.0, 400.0
    reference_x = np.geomspace(minimum, maximum, 500)
    lower, upper = _acceptance_limits(reference_x)
    axis.fill_between(
        reference_x,
        np.maximum(lower, minimum / 2.0),
        upper,
        color="#B8B8B8",
        alpha=0.28,
        linewidth=0.0,
        zorder=0,
    )
    axis.plot(reference_x, reference_x, color="#333333", linewidth=1.1, zorder=1)

    for state_id in STATE_ORDER:
        state = results.loc[results["state_id"] == state_id]
        for dwelling_type in DWELLING_ORDER:
            group = state.loc[state["dwelling_type"] == dwelling_type]
            inside = group["within_predeclared_tabula_band"].astype(bool)
            axis.scatter(
                group["tabula_heating_target_kWh_m2"],
                group["model_heating_kWh_m2"],
                s=34,
                marker=DWELLING_MARKERS[dwelling_type],
                color=STATE_COLORS[state_id],
                edgecolor=np.where(inside, "white", "#B2182B"),
                linewidth=np.where(inside, 0.55, 1.5),
                zorder=3,
            )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(minimum, maximum)
    axis.set_ylim(minimum, maximum)
    axis.set_xlabel("TABULA reference (kWh/m²·yr; logarithmic scale)")
    axis.set_ylabel("5R1C model (kWh/m²·yr; logarithmic scale)")
    axis.set_title(
        "Deterministic annual heating demand against TABULA references\n"
        f"{comparison['within_band_cells']}/{summary['cell_count']} cells "
        f"({100.0 * comparison['pass_rate']:.1f}%) within the predeclared band",
        fontsize=11,
    )
    state_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=STATE_COLORS[item],
            markeredgecolor="white",
            markersize=6,
            label=STATE_LABELS[item],
        )
        for item in STATE_ORDER
    ]
    dwelling_handles = [
        Line2D(
            [0],
            [0],
            marker=DWELLING_MARKERS[item],
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="#333333",
            markersize=6,
            label=DWELLING_LABELS[item],
        )
        for item in DWELLING_ORDER
    ]
    state_legend = axis.legend(
        handles=state_handles,
        title="Physical state",
        loc="upper left",
        frameon=False,
    )
    axis.add_artist(state_legend)
    axis.legend(
        handles=dwelling_handles,
        title="Dwelling type",
        loc="lower right",
        frameon=False,
    )
    outside_ids = ", ".join(
        results.loc[
            ~results["within_predeclared_tabula_band"].astype(bool), "archetype_id"
        ]
        .drop_duplicates()
        .str.replace("BE_TABULA_", "BE ", regex=False)
    )
    axis.text(
        0.03,
        0.025,
        f"Red outline: outside band ({outside_ids})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color="#8B1A1A",
    )
    return figure


def _archetype_heatmap(results: pd.DataFrame, state_id: str) -> plt.Figure:
    minimum = float(results["model_heating_kWh_m2"].min())
    maximum = float(results["model_heating_kWh_m2"].max())
    norm = LogNorm(vmin=max(1.0, minimum), vmax=maximum)
    figure, axis = plt.subplots(figsize=(6.3, 4.5), constrained_layout=True)
    state = results.loc[results["state_id"] == state_id]
    matrix = (
        state.pivot(
            index="dwelling_type",
            columns="construction_period",
            values="model_heating_kWh_m2",
        )
        .reindex(index=DWELLING_ORDER, columns=PERIOD_ORDER)
        .to_numpy(dtype=float)
    )
    image = axis.imshow(matrix, cmap="viridis", norm=norm, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm(value) < 0.58 else "#111111",
            )
    axis.set_title(f"Annual 5R1C heating intensity — {STATE_LABELS[state_id]}", fontsize=11)
    axis.set_xticks(range(len(PERIOD_LABELS)), PERIOD_LABELS, rotation=30, ha="right")
    axis.set_yticks(
        range(len(DWELLING_ORDER)),
        [DWELLING_LABELS[item] for item in DWELLING_ORDER],
    )
    axis.set_xlabel("Construction period")
    axis.set_ylabel("Dwelling type")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, pad=0.03, shrink=0.90)
    colorbar.set_label("Annual heating demand (kWh/m²·yr; logarithmic colour scale)")
    return figure


def _sensitivity_rows(sensitivity: pd.DataFrame, value_column: str) -> pd.DataFrame:
    plotted = sensitivity.loc[
        ~sensitivity["case_id"].isin({"central", "advanced_hrv_central"})
    ].copy()
    missing_labels = sorted(set(plotted["case_id"]).difference(SENSITIVITY_LABELS))
    if missing_labels:
        raise FigureDataError(f"Missing sensitivity labels for {missing_labels}.")
    plotted["magnitude"] = plotted[value_column].abs()
    return plotted.sort_values("magnitude", ascending=True).reset_index(drop=True)


def _heating_sensitivity_figure(sensitivity: pd.DataFrame) -> plt.Figure:
    plotted = _sensitivity_rows(sensitivity, "delta_heating_kWh_m2")
    y = np.arange(len(plotted))
    heat_delta = plotted["delta_heating_kWh_m2"].to_numpy(dtype=float)
    heat_colors = np.where(heat_delta >= 0.0, "#D55E00", "#0072B2")
    figure, axis = plt.subplots(figsize=(7.5, 6.7), constrained_layout=True)
    axis.barh(y, heat_delta, height=0.66, color=heat_colors, alpha=0.90)
    axis.axvline(0.0, color="#333333", linewidth=0.8)
    for position, value in zip(y, heat_delta):
        axis.annotate(
            f"{value:+.1f}",
            (value, position),
            xytext=(4 if value >= 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=7,
        )
    heat_limit = 1.18 * float(np.abs(heat_delta).max())
    axis.set_xlim(-heat_limit, heat_limit)
    axis.set_yticks(
        y,
        [SENSITIVITY_LABELS[item] for item in plotted["case_id"]],
    )
    axis.set_xlabel("Δ annual heating demand (kWh/m²·yr)")
    axis.set_title("One-at-a-time sensitivity of annual heating demand", fontsize=11)
    return figure


def _cooling_sensitivity_figure(sensitivity: pd.DataFrame) -> plt.Figure:
    plotted = _sensitivity_rows(sensitivity, "delta_cooling_kWh_m2")
    y = np.arange(len(plotted))
    cool_delta = plotted["delta_cooling_kWh_m2"].to_numpy(dtype=float)
    cool_colors = np.where(cool_delta >= 0.0, "#D55E00", "#0072B2")
    figure, axis = plt.subplots(figsize=(7.5, 6.7), constrained_layout=True)
    axis.hlines(y, 0.0, cool_delta, color="#A0A0A0", linewidth=1.0)
    axis.scatter(cool_delta, y, s=32, color=cool_colors, zorder=3)
    axis.axvline(0.0, color="#333333", linewidth=0.8)
    cool_limit = max(0.25, 1.15 * float(np.abs(cool_delta).max()))
    axis.set_xlim(-cool_limit, cool_limit)
    axis.set_yticks(
        y,
        [SENSITIVITY_LABELS[item] for item in plotted["case_id"]],
    )
    axis.set_xlabel("Δ annual sensible-cooling demand (kWh/m²·yr)")
    axis.set_title("One-at-a-time sensitivity of annual cooling demand", fontsize=11)
    return figure


def create_figures(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Generate all Gate 3 thermal-model figures and provenance."""

    _style()
    results, summary, sensitivity, sensitivity_summary = _load_inputs()
    figure_builders = {
        "fig_tabula_model_agreement": lambda: _agreement_figure(results, summary),
        "fig_heating_intensity_existing": lambda: _archetype_heatmap(
            results, "TABULA_existing"
        ),
        "fig_heating_intensity_standard": lambda: _archetype_heatmap(
            results, "TABULA_standard_B_proxy"
        ),
        "fig_heating_intensity_advanced": lambda: _archetype_heatmap(
            results, "TABULA_advanced_A_proxy"
        ),
        "fig_heating_sensitivity": lambda: _heating_sensitivity_figure(sensitivity),
        "fig_cooling_sensitivity": lambda: _cooling_sensitivity_figure(sensitivity),
    }
    written: dict[str, dict[str, str]] = {}
    for basename, builder in figure_builders.items():
        paths = _save_figure(builder(), output_dir, basename)
        written[basename] = {
            kind: _display_path(path) for kind, path in paths.items()
        }
        written[basename].update(
            {f"{kind}_sha256": _sha256(path) for kind, path in paths.items()}
        )

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "generator": "thermal_model.plot_figures",
        "generator_sha256": _sha256(Path(__file__)),
        "source_files": {
            str(path.relative_to(MODULE_ROOT)): _sha256(path)
            for path in (
                VALIDATION_RESULTS,
                VALIDATION_SUMMARY,
                SENSITIVITY_RESULTS,
                SENSITIVITY_SUMMARY,
            )
        },
        "validation_status": summary["validation_status"],
        "verification_status": summary["verification_status"],
        "figures": written,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "figure_provenance.json"
    temporary = provenance_path.with_name(f".{provenance_path.name}.writing")
    temporary.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(provenance_path)
    return provenance


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create publication-ready thermal-model validation figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for PNG, PDF and provenance outputs.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    provenance = create_figures(args.output_dir)
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
