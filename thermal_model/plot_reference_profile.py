"""Create the illustrative deterministic reference-year demand profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .contracts import (
    DEFAULT_ASSUMPTIONS_PATH,
    SimulationInput,
    load_assumption_contract,
)
from .core import preprocess_archetype, simulate
from .validation import (
    BASE_MATRIX_PATH,
    CLIMATE_CONFIG_PATH,
    HDD_COMPARISON_PATH,
    STATE_MATRIX_PATH,
    build_reference_schedules,
    load_reference_weather,
    load_unique_archetype_states,
)


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODULE_ROOT / "figures"
BASENAME = "fig_reference_year_daily_demand"
STOCK_MATRIX_PATH = (
    MODULE_ROOT.parent
    / "BE_building_stock/data/matrices/national/stock_weighted_archetype_matrix.csv"
)
STATE_ID = "TABULA_existing"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(figure: plt.Figure, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{BASENAME}.png"
    pdf_path = output_dir / f"{BASENAME}.pdf"
    png_temp = png_path.with_name(f".{png_path.name}.writing")
    pdf_temp = pdf_path.with_name(f".{pdf_path.name}.writing")
    figure.savefig(
        png_temp,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "thermal_model.plot_reference_profile"},
    )
    figure.savefig(
        pdf_temp,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "thermal_model.plot_reference_profile",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    png_temp.replace(png_path)
    pdf_temp.replace(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def create_reference_profile(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Plot the selected dwelling category under the Gate-3 conditions."""

    assumptions = load_assumption_contract(DEFAULT_ASSUMPTIONS_PATH)
    weather, weather_metadata = load_reference_weather()
    if int(weather_metadata["selected_year"]) != 2015:
        raise ValueError("The frozen reference-year rule must select 2015.")

    states_by_key = {
        (state.archetype_id, state.state_id): state
        for state in load_unique_archetype_states()
    }
    stock = pd.read_csv(STOCK_MATRIX_PATH)
    required_columns = {"archetype_id", "dwelling_type", "number_of_dwellings"}
    missing = required_columns.difference(stock.columns)
    if missing:
        raise ValueError(f"Stock matrix is missing required columns: {sorted(missing)}")
    if stock["number_of_dwellings"].isna().any() or (
        stock["number_of_dwellings"] <= 0.0
    ).any():
        raise ValueError("All national archetype weights must be positive.")

    stock["dwelling_category"] = stock["dwelling_type"].where(
        ~stock["dwelling_type"].str.startswith("Apartment"), "Apartment"
    )
    dwelling_category = "Semi-detached house"
    selected_stock = stock.loc[
        stock["dwelling_category"] == dwelling_category
    ].copy()
    selected_stock["weight"] = (
        selected_stock["number_of_dwellings"]
        / selected_stock["number_of_dwellings"].sum()
    )

    weighted_daily: pd.DataFrame | None = None
    case_diagnostics: list[dict[str, Any]] = []
    for row in selected_stock.itertuples(index=False):
        state = states_by_key[(row.archetype_id, STATE_ID)]
        prepared = preprocess_archetype(state, assumptions)
        schedules = build_reference_schedules(
            weather["timestamp_utc"], prepared.floor_area_m2, assumptions
        )
        result = simulate(
            SimulationInput(
                archetype=prepared,
                weather=weather,
                schedules=schedules,
                weather_member_id=str(weather_metadata["weather_member_id"]),
                occupant_seed=0,
                model_scenario="methodology_semi_detached_reference_profile",
            ),
            assumptions,
        )
        case_daily = (
            result.hourly.set_index("timestamp_utc")
            [["heating_demand_W", "cooling_demand_W"]]
            .resample("D")
            .sum()
            / 1000.0
        )
        case_daily.columns = ["heating_kWh", "cooling_kWh"]
        contribution = case_daily * float(row.weight)
        weighted_daily = (
            contribution
            if weighted_daily is None
            else weighted_daily.add(contribution, fill_value=0.0)
        )
        case_diagnostics.append(
            {
                "archetype_id": prepared.archetype_id,
                "dwelling_type": prepared.dwelling_type,
                "construction_period": prepared.construction_period,
                "floor_area_m2": prepared.floor_area_m2,
                "number_of_dwellings": float(row.number_of_dwellings),
                "category_weight": float(row.weight),
                "annual_heating_kWh": result.diagnostics.annual_heating_kWh,
                "annual_cooling_kWh": result.diagnostics.annual_cooling_kWh,
                "warmup_cycles": result.diagnostics.warmup_cycles,
                "max_abs_energy_balance_residual_W": (
                    result.diagnostics.max_abs_energy_balance_residual_W
                ),
            }
        )

    if weighted_daily is None:
        raise ValueError(
            "No semi-detached-house archetypes were selected from the stock matrix."
        )
    daily = weighted_daily
    if len(daily) != 365:
        raise ValueError("The 2015 daily profile must contain 365 complete days.")
    expected_heating = sum(
        case["category_weight"] * case["annual_heating_kWh"]
        for case in case_diagnostics
    )
    expected_cooling = sum(
        case["category_weight"] * case["annual_cooling_kWh"]
        for case in case_diagnostics
    )
    if abs(daily["heating_kWh"].sum() - expected_heating) > 1e-9:
        raise ValueError("Weighted daily heating does not conserve annual energy.")
    if abs(daily["cooling_kWh"].sum() - expected_cooling) > 1e-9:
        raise ValueError("Weighted daily cooling does not conserve annual energy.")

    modelled_dwellings = float(stock["number_of_dwellings"].sum())
    excluded_dwellings = float(stock["excluded_residual_R5_R6_dwellings"].iloc[0])
    category_dwellings = float(selected_stock["number_of_dwellings"].sum())
    all_dwellings = modelled_dwellings + excluded_dwellings

    dates = pd.DatetimeIndex(daily.index).tz_convert("UTC").tz_localize(None)
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    heating_color = "#D55E00"
    cooling_color = "#0072B2"
    axis.fill_between(
        dates,
        0.0,
        daily["heating_kWh"].to_numpy(),
        color=heating_color,
        alpha=0.24,
        linewidth=0.0,
    )
    axis.plot(
        dates,
        daily["heating_kWh"],
        color=heating_color,
        linewidth=0.85,
        label="Useful heating",
    )
    axis.fill_between(
        dates,
        0.0,
        daily["cooling_kWh"].to_numpy(),
        color=cooling_color,
        alpha=0.35,
        linewidth=0.0,
    )
    axis.plot(
        dates,
        daily["cooling_kWh"],
        color=cooling_color,
        linewidth=1.0,
        label="Sensible cooling",
    )
    axis.set_xlim(dates[0], dates[-1])
    axis.set_ylim(bottom=0.0)
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axis.set_ylabel("Daily useful demand (kWh/day)")
    axis.set_xlabel("2015")
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    axis.tick_params(labelsize=8)
    figure.tight_layout()

    png_path, pdf_path = _save(figure, output_dir)
    provenance = {
        "schema_version": 1,
        "generator": "thermal_model.plot_reference_profile",
        "generator_sha256": _sha256(Path(__file__)),
        "case": {
            "dwelling_category": dwelling_category,
            "state_id": STATE_ID,
            "archetype_ids": selected_stock["archetype_id"].tolist(),
            "category_dwellings": category_dwellings,
            "national_share_all_dwellings": category_dwellings / all_dwellings,
            "share_within_modelled_R1_R4": category_dwellings / modelled_dwellings,
            "weighting": "2025 national number_of_dwellings within category",
            "weather_member_id": weather_metadata["weather_member_id"],
            "reference_year": weather_metadata["selected_year"],
            "heating_setpoint_C": assumptions.number("control.heating_reference"),
            "cooling_setpoint_C": assumptions.number("control.cooling_reference"),
            "internal_gains_W_m2": assumptions.number("validation.internal_gains"),
        },
        "diagnostics": {
            "stock_weighted_annual_heating_kWh_per_dwelling": expected_heating,
            "stock_weighted_annual_cooling_kWh_per_dwelling": expected_cooling,
            "peak_daily_heating_kWh_per_dwelling": float(
                daily["heating_kWh"].max()
            ),
            "peak_daily_cooling_kWh_per_dwelling": float(
                daily["cooling_kWh"].max()
            ),
            "maximum_warmup_cycles": max(
                case["warmup_cycles"] for case in case_diagnostics
            ),
            "max_abs_energy_balance_residual_W": max(
                case["max_abs_energy_balance_residual_W"]
                for case in case_diagnostics
            ),
            "archetype_results": case_diagnostics,
        },
        "source_files": {
            str(DEFAULT_ASSUMPTIONS_PATH.relative_to(MODULE_ROOT)): _sha256(
                DEFAULT_ASSUMPTIONS_PATH
            ),
            str(BASE_MATRIX_PATH.relative_to(MODULE_ROOT.parent)): _sha256(
                BASE_MATRIX_PATH
            ),
            str(STOCK_MATRIX_PATH.relative_to(MODULE_ROOT.parent)): _sha256(
                STOCK_MATRIX_PATH
            ),
            str(STATE_MATRIX_PATH.relative_to(MODULE_ROOT.parent)): _sha256(
                STATE_MATRIX_PATH
            ),
            str(HDD_COMPARISON_PATH.relative_to(MODULE_ROOT.parent)): _sha256(
                HDD_COMPARISON_PATH
            ),
            str(CLIMATE_CONFIG_PATH.relative_to(MODULE_ROOT.parent)): _sha256(
                CLIMATE_CONFIG_PATH
            ),
        },
        "outputs": {
            "png": png_path.name,
            "png_sha256": _sha256(png_path),
            "pdf": pdf_path.name,
            "pdf_sha256": _sha256(pdf_path),
        },
        "interpretation": (
            "Illustrative 2025-stock-weighted semi-detached-house mean of "
            "verified hourly outputs; not a validation target or acceptance "
            "criterion. The plot intentionally has no internal title because "
            "the thesis caption identifies the figure."
        ),
    }
    provenance_path = output_dir / f"{BASENAME}.provenance.json"
    temporary = provenance_path.with_name(f".{provenance_path.name}.writing")
    temporary.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(provenance_path)
    return provenance


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the deterministic 2015 reference demand profile."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for the independent PNG, PDF and provenance files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(
        json.dumps(
            create_reference_profile(args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
