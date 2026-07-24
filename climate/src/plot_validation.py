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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path]:
    outputs = config["validation"]["outputs"]
    root = resolve_config_path(config, outputs["directory"])
    report_path = root / outputs["report_json"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass" or int(report.get("hard_error_count", -1)) != 0:
        raise ValueError("Validation figures require a passing validation report.")
    member_path = root / outputs["member_summary"]
    monthly_path = root / outputs["monthly_invariants"]
    if sha256_file(member_path) != report["outputs"]["member_summary"]["sha256"]:
        raise ValueError("Validated member summary hash mismatch.")
    if sha256_file(monthly_path) != report["outputs"]["monthly_invariants"]["sha256"]:
        raise ValueError("Validated monthly-invariant table hash mismatch.")
    return (
        pd.read_csv(member_path),
        pd.read_csv(monthly_path),
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
        if int(row["observed_pvgis_year"]) in {2005, 2008, 2011, 2014, 2017, 2020, 2023}
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
    if any(len(years) != 19 for years in result.values()):
        raise ValueError("Temperature-duration curves require 19 years per scenario.")
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
    members, monthly, validation_report, report_path = _load_validation_tables(config)
    delta_contract = load_delta_contract(config)
    scenarios = [
        str(spec["scenario"])
        for spec in config["sources"].values()
        if spec["role"] == "future"
    ]
    observed, _ = load_clean_observed(config)
    observed_years = split_complete_years(observed)
    member_years = _load_member_temperature_years(config, scenarios)

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
                "All 57 x 12 recovered-minus-expected temperature and GHI ratios."
            ),
            "temperature_duration": (
                "Median annual duration curve with 5th-95th percentile envelope "
                "across 19 observed weather years; x-axis is hours exceeded."
            ),
        },
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
