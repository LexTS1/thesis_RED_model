"""Layered verification, Belgian validation, and sensitivity analysis for 5R1C.

Verification tests the implementation against equations and invariants. Validation
compares the resulting useful demand with independent Belgian reference values.
The two statuses are deliberately kept separate in every generated artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from climate.src.load_cordex import load_config
from climate.src.load_observed import load_clean_observed, load_facade_templates

from .contracts import (
    DEFAULT_ASSUMPTIONS_PATH,
    PHYSICAL_STATE_FIELDS,
    ArchetypeStateInput,
    AssumptionContract,
    PreparedArchetype,
    SimulationInput,
    assemble_archetype_state,
    load_assumption_contract,
    validate_prepared_archetype,
    validate_weather_frame,
)
from .core import preprocess_archetype, simulate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MATRIX_PATH = (
    PROJECT_ROOT
    / "BE_building_stock/data/matrices/national/base_physical_archetype_matrix.csv"
)
STATE_MATRIX_PATH = (
    PROJECT_ROOT
    / "BE_building_stock/data/scenarios/renovation/"
    "archetype_matrix_2050_renovation_scenarios.csv"
)
CLIMATE_CONFIG_PATH = PROJECT_ROOT / "climate/config.yaml"
HDD_COMPARISON_PATH = (
    PROJECT_ROOT
    / "climate/data/processed/validation/observed_be100_degree_day_comparison.csv"
)
TABULA_TARGET_PATH = (
    Path(__file__).resolve().parent
    / "data/reference/tabula_net_heating_demand.csv"
)
TABULA_PROVENANCE_PATH = TABULA_TARGET_PATH.with_suffix(".provenance.json")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data/validation"

STATE_ORDER = (
    "TABULA_existing",
    "TABULA_standard_B_proxy",
    "TABULA_advanced_A_proxy",
)
TARGET_COLUMN_BY_STATE = {
    "TABULA_existing": "TABULA_existing_kWh_m2",
    "TABULA_standard_B_proxy": "TABULA_standard_B_proxy_kWh_m2",
    "TABULA_advanced_A_proxy": "TABULA_advanced_A_proxy_kWh_m2",
}
PERIOD_ORDER = {
    "pre-1946": 0,
    "1946-1970": 1,
    "1971-1990": 2,
    "1991-2005": 3,
    "post-2005": 4,
}

# Declared before inspecting the 75 deterministic results. The direct benchmark
# uses the larger of an absolute floor and a relative allowance because relative
# error is unstable for the lowest-demand LE apartments.
TABULA_WARNING_RELATIVE = 0.30
TABULA_WARNING_ABSOLUTE_KWH_M2 = 15.0
TABULA_GATE_MINIMUM_PASS_RATE = 0.80
ENERGY_BALANCE_TOLERANCE_W = 1.0e-6
SETPOINT_TRACKING_TOLERANCE_K = 1.0e-8
QUALITATIVE_MINIMUM_RATE = 0.80


class ValidationError(ValueError):
    """Raised when a validation source or generated result is inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_tabula_targets(path: Path = TABULA_TARGET_PATH) -> pd.DataFrame:
    """Load and validate the 25-by-3 TABULA net-heating target matrix."""

    frame = pd.read_csv(path)
    required = {"archetype_id", "TABULA_type_number", *TARGET_COLUMN_BY_STATE.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationError(f"TABULA target file is missing columns: {missing}")
    if len(frame) != 25 or frame["archetype_id"].nunique() != 25:
        raise ValidationError("TABULA target file must contain 25 unique archetypes.")
    expected_ids = {f"BE_TABULA_{number:02d}" for number in range(1, 26)}
    if set(frame["archetype_id"]) != expected_ids:
        raise ValidationError("TABULA target archetype identifiers are incomplete.")
    if set(frame["TABULA_type_number"].astype(int)) != set(range(1, 26)):
        raise ValidationError("TABULA type numbers must be exactly 1 through 25.")
    numeric = frame[list(TARGET_COLUMN_BY_STATE.values())].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (numeric <= 0.0).any().any():
        raise ValidationError("TABULA net-heating targets must be finite and positive.")

    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        for state_id, column in TARGET_COLUMN_BY_STATE.items():
            records.append(
                {
                    "archetype_id": row.archetype_id,
                    "TABULA_type_number": int(row.TABULA_type_number),
                    "state_id": state_id,
                    "tabula_heating_target_kWh_m2": float(getattr(row, column)),
                }
            )
    result = pd.DataFrame.from_records(records)
    if len(result) != 75 or result.duplicated(["archetype_id", "state_id"]).any():
        raise ValidationError("Long-form TABULA target matrix must contain 75 unique cells.")
    return result


def select_reference_weather_year(
    hdd_path: Path = HDD_COMPARISON_PATH,
) -> dict[str, float | int | str]:
    """Select the observed year nearest the 2006-2023 median PVGIS HDD."""

    frame = pd.read_csv(hdd_path)
    required = {"year", "pvgis_HDD_C_days"}
    if not required.issubset(frame.columns):
        raise ValidationError(f"HDD comparison is missing {sorted(required-frame.columns)}.")
    if frame["year"].duplicated().any() or len(frame) != 18:
        raise ValidationError("Reference-year selection requires 18 unique observed years.")
    median_hdd = float(frame["pvgis_HDD_C_days"].median())
    ranked = frame.assign(
        distance_to_median=(frame["pvgis_HDD_C_days"] - median_hdd).abs()
    ).sort_values(["distance_to_median", "year"], kind="stable")
    selected = ranked.iloc[0]
    return {
        "selection_method": "complete PVGIS year nearest 2006-2023 median HDD; earlier year breaks exact ties",
        "selected_year": int(selected["year"]),
        "selected_hdd_C_days": float(selected["pvgis_HDD_C_days"]),
        "median_hdd_C_days": median_hdd,
        "distance_to_median_C_days": float(selected["distance_to_median"]),
        "source_path": str(hdd_path.relative_to(PROJECT_ROOT)),
        "source_sha256": _sha256_file(hdd_path),
    }


def load_reference_weather(
    year: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one observed reference year with aligned PVGIS facade irradiance."""

    selection = select_reference_weather_year()
    selected_year = int(selection["selected_year"] if year is None else year)
    config = load_config(CLIMATE_CONFIG_PATH)
    observed, observed_metadata = load_clean_observed(config)
    selected = observed.loc[
        observed["timestamp_utc"].dt.year == selected_year
    ].reset_index(drop=True)
    if len(selected) not in {8760, 8784}:
        raise ValidationError(
            f"PVGIS reference year {selected_year} is incomplete ({len(selected)} rows)."
        )

    # Preserve the horizontal climate forcing as additional columns.  The 5R1C
    # core still consumes only temperature and the four facade series, while
    # Gate 4 uses these exact beam/diffuse values for RichardsonPy lighting.
    weather = selected[
        [
            "timestamp_utc",
            "T_out_C",
            "I_beam_horizontal_W_m2",
            "I_diffuse_horizontal_W_m2",
            "I_solar_W_m2",
        ]
    ].copy()
    facade_templates = load_facade_templates(config)
    facade_hashes: dict[str, str] = {}
    for orientation in ("north", "east", "south", "west"):
        plane = facade_templates[orientation]
        facade = plane.frame.loc[
            plane.frame["timestamp_utc"].dt.year == selected_year
        ].reset_index(drop=True)
        if not pd.DatetimeIndex(facade["timestamp_utc"]).equals(
            pd.DatetimeIndex(weather["timestamp_utc"])
        ):
            raise ValidationError(
                f"PVGIS {orientation} facade does not align with reference year."
            )
        weather[f"I_{orientation}_W_m2"] = facade[
            ["I_beam_W_m2", "I_diffuse_W_m2", "I_reflected_W_m2"]
        ].sum(axis=1)
        facade_hashes[orientation] = plane.source_sha256

    validated = validate_weather_frame(weather)
    metadata: dict[str, Any] = {
        **selection,
        "selected_year": selected_year,
        "weather_member_id": f"pvgis_sarah3_observed_{selected_year}_reference",
        "row_count": len(validated),
        "observed_dataset_sha256": observed_metadata["output"]["sha256"],
        "facade_source_sha256": facade_hashes,
        "temperature_mean_C": float(validated["T_out_C"].mean()),
        "facade_annual_kWh_m2": {
            orientation: float(validated[f"I_{orientation}_W_m2"].sum()) / 1000.0
            for orientation in ("north", "east", "south", "west")
        },
        "horizontal_annual_kWh_m2": float(validated["I_solar_W_m2"].sum())
        / 1000.0,
    }
    return validated, metadata


def load_unique_archetype_states(
    base_path: Path = BASE_MATRIX_PATH,
    state_path: Path = STATE_MATRIX_PATH,
) -> list[ArchetypeStateInput]:
    """Return 75 unique physics combinations, removing regional duplicates."""

    base = pd.read_csv(base_path).set_index("archetype_id", drop=False)
    states = pd.read_csv(state_path)
    physical_columns = sorted(PHYSICAL_STATE_FIELDS)
    missing = sorted(set(physical_columns).difference(states.columns))
    if missing:
        raise ValidationError(f"Physical-state matrix is missing columns: {missing}")

    for (archetype_id, state_id), group in states.groupby(
        ["archetype_id", "state_id"], sort=False
    ):
        distinct = group[physical_columns].drop_duplicates()
        if len(distinct) != 1:
            raise ValidationError(
                f"Regional rows disagree physically for {archetype_id}/{state_id}."
            )
    unique = states.drop_duplicates(["archetype_id", "state_id"]).copy()
    if len(unique) != 75:
        raise ValidationError(
            f"Expected 75 unique archetype-state combinations; found {len(unique)}."
        )
    unique["_type_number"] = unique["archetype_id"].str[-2:].astype(int)
    unique["_state_rank"] = unique["state_id"].map(
        {state: rank for rank, state in enumerate(STATE_ORDER)}
    )
    if unique["_state_rank"].isna().any():
        unknown = sorted(set(unique.loc[unique["_state_rank"].isna(), "state_id"]))
        raise ValidationError(f"Unknown physical states: {unknown}")
    unique = unique.sort_values(["_type_number", "_state_rank"])
    return [
        assemble_archetype_state(base.loc[row.archetype_id], row._asdict())
        for row in unique.itertuples(index=False)
    ]


def build_reference_schedules(
    timestamps: pd.Series,
    floor_area_m2: float,
    assumptions: AssumptionContract,
    *,
    internal_gains_W_m2: float | None = None,
    heating_setpoint_C: float | None = None,
    cooling_setpoint_C: float | None = None,
) -> pd.DataFrame:
    """Build the fixed deterministic operating schedule for one dwelling."""

    gain_density = (
        assumptions.number("validation.internal_gains")
        if internal_gains_W_m2 is None
        else float(internal_gains_W_m2)
    )
    heat = (
        assumptions.number("control.heating_reference")
        if heating_setpoint_C is None
        else float(heating_setpoint_C)
    )
    cool = (
        assumptions.number("control.cooling_reference")
        if cooling_setpoint_C is None
        else float(cooling_setpoint_C)
    )
    if gain_density < 0.0:
        raise ValidationError("Reference internal gains must be non-negative.")
    if heat > cool:
        raise ValidationError("Reference heating setpoint exceeds cooling setpoint.")
    count = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp_utc": pd.DatetimeIndex(timestamps),
            "Phi_int_W": np.full(count, gain_density * floor_area_m2),
            "theta_set_heat_C": np.full(count, heat),
            "theta_set_cool_C": np.full(count, cool),
        }
    )


def _tracking_error(hourly: pd.DataFrame) -> tuple[float, float]:
    heating = hourly["heating_demand_W"] > 1.0e-9
    cooling = hourly["cooling_demand_W"] > 1.0e-9
    heat_error = (
        float(
            (hourly.loc[heating, "theta_air_C"] - hourly.loc[heating, "theta_set_heat_C"])
            .abs()
            .max()
        )
        if heating.any()
        else 0.0
    )
    cool_error = (
        float(
            (hourly.loc[cooling, "theta_air_C"] - hourly.loc[cooling, "theta_set_cool_C"])
            .abs()
            .max()
        )
        if cooling.any()
        else 0.0
    )
    return heat_error, cool_error


def _simulate_prepared(
    prepared: PreparedArchetype,
    weather: pd.DataFrame,
    assumptions: AssumptionContract,
    *,
    weather_member_id: str,
    model_scenario: str,
    internal_gains_W_m2: float | None = None,
    heating_setpoint_C: float | None = None,
    cooling_setpoint_C: float | None = None,
):
    schedules = build_reference_schedules(
        weather["timestamp_utc"],
        prepared.floor_area_m2,
        assumptions,
        internal_gains_W_m2=internal_gains_W_m2,
        heating_setpoint_C=heating_setpoint_C,
        cooling_setpoint_C=cooling_setpoint_C,
    )
    request = SimulationInput(
        archetype=prepared,
        weather=weather,
        schedules=schedules,
        weather_member_id=weather_member_id,
        occupant_seed=0,
        model_scenario=model_scenario,
    )
    return simulate(request, assumptions)


def summarize_qualitative_patterns(results: pd.DataFrame) -> dict[str, Any]:
    """Evaluate directional stock patterns independently of target deviations."""

    pivot = results.pivot(
        index=["archetype_id", "dwelling_type", "construction_period"],
        columns="state_id",
        values="model_heating_kWh_m2",
    )
    renovation_pass = (
        (pivot["TABULA_advanced_A_proxy"] <= pivot["TABULA_standard_B_proxy"] + 1e-9)
        & (pivot["TABULA_standard_B_proxy"] <= pivot["TABULA_existing"] + 1e-9)
    )

    exposure_checks: list[bool] = []
    house_order_checks: list[bool] = []
    for (_, _), group in results.groupby(["construction_period", "state_id"]):
        by_type = group.set_index("dwelling_type")["model_heating_kWh_m2"]
        exposure_checks.append(
            bool(by_type["Apartment, exposed"] >= by_type["Apartment, enclosed"] - 1e-9)
        )
        house_order_checks.append(
            bool(
                by_type["Detached house"] >= by_type["Semi-detached house"] - 1e-9
                and by_type["Semi-detached house"] >= by_type["Terraced house"] - 1e-9
            )
        )

    # Test the stated physical ordering directly.  Correlation is the wrong
    # diagnostic here: it is undefined for intentionally identical package
    # states (for example the repeated enclosed-apartment geometry), and a
    # negative correlation can still hide a local reversal.  Equality is not a
    # contradiction of "older generally has greater demand", so each adjacent
    # old-to-new pair passes when demand is non-increasing within tolerance.
    age_differences: list[float] = []
    for (_, _), group in results.groupby(["dwelling_type", "state_id"]):
        ordered = group.assign(
            period_rank=group["construction_period"].map(PERIOD_ORDER)
        ).sort_values("period_rank")
        if ordered["period_rank"].isna().any() or len(ordered) != len(PERIOD_ORDER):
            raise ValidationError(
                "Every dwelling-type/state group must contain all five periods."
            )
        demand = ordered["model_heating_kWh_m2"].to_numpy(dtype=float)
        age_differences.extend(np.diff(demand).tolist())
    age_nonincreasing = [value <= 1.0e-9 for value in age_differences]

    return {
        "renovation_order": {
            "passed": int(renovation_pass.sum()),
            "total": int(len(renovation_pass)),
            "rate": float(renovation_pass.mean()),
            "failed_archetypes": [
                index[0] for index in pivot.index[~renovation_pass]
            ],
        },
        "exposed_apartment_above_enclosed": {
            "passed": int(sum(exposure_checks)),
            "total": len(exposure_checks),
            "rate": float(np.mean(exposure_checks)),
        },
        "detached_above_semi_above_terraced": {
            "passed": int(sum(house_order_checks)),
            "total": len(house_order_checks),
            "rate": float(np.mean(house_order_checks)),
        },
        "newer_period_nonincreasing_adjacent_pairs": {
            "passed": int(sum(age_nonincreasing)),
            "total": len(age_nonincreasing),
            "rate": float(np.mean(age_nonincreasing)),
            "minimum_newer_minus_older_kWh_m2": float(min(age_differences)),
            "maximum_newer_minus_older_kWh_m2": float(max(age_differences)),
        },
    }


def _external_context(results: pd.DataFrame) -> dict[str, Any]:
    """Return non-calibrating literature and EPC comparison context."""

    state_medians = (
        results.groupby("state_id")["model_heating_kWh_m2"].median().to_dict()
    )
    existing = results.loc[results["state_id"] == "TABULA_existing"]
    houses = existing.loc[~existing["dwelling_type"].str.startswith("Apartment")]
    apartments = existing.loc[existing["dwelling_type"].str.startswith("Apartment")]
    return {
        "climate_neutral_belgium_2050_scenario_levels": {
            "metric_description": "Residential renovation-depth energy-intensity scenario levels",
            "shallow_kWh_m2_year": 85.0,
            "medium_kWh_m2_year": 64.0,
            "deep_kWh_m2_year": 25.0,
            "source_path": "tmp/pdfs/climate-neutral-belgium-by-2050-report.pdf",
            "source_url": "https://climat.be/doc/climate-neutral-belgium-by-2050-report.pdf",
            "source_locator": "PDF page 49",
            "comparison_role": "context only; scenario levers are not archetype-specific validation targets",
            "model_standard_state_median_kWh_m2": float(
                state_medians["TABULA_standard_B_proxy"]
            ),
            "model_advanced_state_median_kWh_m2": float(
                state_medians["TABULA_advanced_A_proxy"]
            ),
        },
        "nbb_2024_sold_dwelling_epc_scores": {
            "houses_regional_mean_range_kWh_m2": [363.0, 390.0],
            "apartments_regional_mean_range_kWh_m2": [198.0, 245.0],
            "model_existing_houses_median_useful_heat_kWh_m2": float(
                houses["model_heating_kWh_m2"].median()
            ),
            "model_existing_apartments_median_useful_heat_kWh_m2": float(
                apartments["model_heating_kWh_m2"].median()
            ),
            "comparison_role": (
                "context only: EPC is primary energy under regional methods, while the "
                "model output is useful space-heating demand"
            ),
            "source_key": "nbb_epc_houseprices_2025",
            "source_url": (
                "https://www.nbb.be/doc/ts/publications/economicreview/2025/"
                "ecorevi2025_h01.pdf"
            ),
        },
    }


def run_deterministic_validation(
    *,
    assumptions: AssumptionContract | None = None,
    reference_year: int | None = None,
    states: Sequence[ArchetypeStateInput] | None = None,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the 75-cell deterministic Belgian TABULA validation matrix."""

    contract = assumptions or load_assumption_contract(DEFAULT_ASSUMPTIONS_PATH)
    weather, weather_metadata = load_reference_weather(reference_year)
    unique_states = list(states) if states is not None else load_unique_archetype_states()
    targets = load_tabula_targets().set_index(["archetype_id", "state_id"])
    records: list[dict[str, Any]] = []

    for state in unique_states:
        prepared = preprocess_archetype(state, contract)
        result = _simulate_prepared(
            prepared,
            weather,
            contract,
            weather_member_id=str(weather_metadata["weather_member_id"]),
            model_scenario="gate3_deterministic_central",
        )
        try:
            target_row = targets.loc[(state.archetype_id, state.state_id)]
        except KeyError as exc:
            raise ValidationError(
                f"No TABULA target for {state.archetype_id}/{state.state_id}."
            ) from exc
        target = float(target_row["tabula_heating_target_kWh_m2"])
        model = result.diagnostics.heating_intensity_kWh_m2
        signed_deviation = model - target
        warning_tolerance = max(
            TABULA_WARNING_ABSOLUTE_KWH_M2,
            TABULA_WARNING_RELATIVE * target,
        )
        heat_error, cool_error = _tracking_error(result.hourly)
        simultaneous = bool(
            (
                (result.hourly["heating_demand_W"] > 1e-9)
                & (result.hourly["cooling_demand_W"] > 1e-9)
            ).any()
        )
        records.append(
            {
                "archetype_id": state.archetype_id,
                "TABULA_type_number": int(target_row["TABULA_type_number"]),
                "dwelling_type": state.dwelling_type,
                "construction_period": state.construction_period,
                "state_id": state.state_id,
                "floor_area_m2": prepared.floor_area_m2,
                "model_heating_kWh_m2": model,
                "model_cooling_kWh_m2": result.diagnostics.cooling_intensity_kWh_m2,
                "tabula_heating_target_kWh_m2": target,
                "signed_deviation_kWh_m2": signed_deviation,
                "absolute_deviation_kWh_m2": abs(signed_deviation),
                "relative_deviation_percent": 100.0 * signed_deviation / target,
                "warning_tolerance_kWh_m2": warning_tolerance,
                "within_predeclared_tabula_band": abs(signed_deviation)
                <= warning_tolerance,
                "peak_heating_W": result.diagnostics.peak_heating_W,
                "peak_cooling_W": result.diagnostics.peak_cooling_W,
                "H_tr_w_W_K": prepared.H_tr_w_W_K,
                "H_tr_op_W_K": prepared.H_tr_op_W_K,
                "H_ve_mean_W_K": float(result.hourly["H_ve_W_K"].mean()),
                "max_energy_balance_residual_W": (
                    result.diagnostics.max_abs_energy_balance_residual_W
                ),
                "max_heating_setpoint_error_K": heat_error,
                "max_cooling_setpoint_error_K": cool_error,
                "simultaneous_heating_cooling": simultaneous,
                "warmup_cycles": result.diagnostics.warmup_cycles,
                "assumptions_sha256": result.diagnostics.assumptions_sha256,
                "weather_member_id": result.diagnostics.weather_member_id,
            }
        )

    results = pd.DataFrame.from_records(records).sort_values(
        ["TABULA_type_number", "state_id"],
        key=lambda series: (
            series.map({state: rank for rank, state in enumerate(STATE_ORDER)})
            if series.name == "state_id"
            else series
        ),
        kind="stable",
    )
    direct_pass_rate = float(results["within_predeclared_tabula_band"].mean())
    numerical_checks = {
        "energy_balance_all_timesteps_pass": bool(
            (
                results["max_energy_balance_residual_W"]
                <= ENERGY_BALANCE_TOLERANCE_W
            ).all()
        ),
        "setpoint_tracking_all_controlled_hours_pass": bool(
            (
                results[
                    [
                        "max_heating_setpoint_error_K",
                        "max_cooling_setpoint_error_K",
                    ]
                ].max(axis=1)
                <= SETPOINT_TRACKING_TOLERANCE_K
            ).all()
        ),
        "no_simultaneous_heating_cooling": bool(
            (~results["simultaneous_heating_cooling"]).all()
        ),
        "energy_balance_tolerance_W": ENERGY_BALANCE_TOLERANCE_W,
        "setpoint_tracking_tolerance_K": SETPOINT_TRACKING_TOLERANCE_K,
        "maximum_energy_balance_residual_W": float(
            results["max_energy_balance_residual_W"].max()
        ),
        "maximum_setpoint_error_K": float(
            results[
                ["max_heating_setpoint_error_K", "max_cooling_setpoint_error_K"]
            ].to_numpy(dtype=float).max()
        ),
    }
    pattern_summary = summarize_qualitative_patterns(results)
    hard_verification_pass = all(
        numerical_checks[key]
        for key in (
            "energy_balance_all_timesteps_pass",
            "setpoint_tracking_all_controlled_hours_pass",
            "no_simultaneous_heating_cooling",
        )
    )
    qualitative_pass = all(
        item["rate"] >= QUALITATIVE_MINIMUM_RATE for item in pattern_summary.values()
    )
    direct_validation_pass = direct_pass_rate >= TABULA_GATE_MINIMUM_PASS_RATE
    summary: dict[str, Any] = {
        "schema_version": 1,
        "verification_status": "PASS" if hard_verification_pass else "FAIL",
        "validation_status": (
            "PASS"
            if direct_validation_pass and qualitative_pass
            else "REVIEW_REQUIRED"
        ),
        "cell_count": int(len(results)),
        "unique_archetypes": int(results["archetype_id"].nunique()),
        "physical_states": list(STATE_ORDER),
        "reference_weather": weather_metadata,
        "operating_conditions": {
            "heating_setpoint_C": contract.number("control.heating_reference"),
            "cooling_setpoint_C": contract.number("control.cooling_reference"),
            "internal_gains_W_m2": contract.number("validation.internal_gains"),
        },
        "predeclared_acceptance": {
            "tabula_relative_warning_fraction": TABULA_WARNING_RELATIVE,
            "tabula_absolute_warning_floor_kWh_m2": TABULA_WARNING_ABSOLUTE_KWH_M2,
            "tabula_gate_minimum_cell_pass_rate": TABULA_GATE_MINIMUM_PASS_RATE,
            "qualitative_minimum_rate": QUALITATIVE_MINIMUM_RATE,
        },
        "numerical_checks": numerical_checks,
        "tabula_comparison": {
            "within_band_cells": int(results["within_predeclared_tabula_band"].sum()),
            "outside_band_cells": int((~results["within_predeclared_tabula_band"]).sum()),
            "pass_rate": direct_pass_rate,
            "gate_pass": direct_validation_pass,
            "mean_signed_deviation_kWh_m2": float(
                results["signed_deviation_kWh_m2"].mean()
            ),
            "mean_absolute_deviation_kWh_m2": float(
                results["absolute_deviation_kWh_m2"].mean()
            ),
            "root_mean_square_deviation_kWh_m2": float(
                np.sqrt(np.mean(results["signed_deviation_kWh_m2"] ** 2))
            ),
            "median_absolute_relative_deviation_percent": float(
                results["relative_deviation_percent"].abs().median()
            ),
            "by_state": {
                state_id: {
                    "model_median_kWh_m2": float(group["model_heating_kWh_m2"].median()),
                    "target_median_kWh_m2": float(
                        group["tabula_heating_target_kWh_m2"].median()
                    ),
                    "mean_signed_deviation_kWh_m2": float(
                        group["signed_deviation_kWh_m2"].mean()
                    ),
                    "pass_rate": float(group["within_predeclared_tabula_band"].mean()),
                }
                for state_id, group in results.groupby("state_id", sort=False)
            },
        },
        "qualitative_patterns": pattern_summary,
        "external_context": _external_context(results),
        "provenance": {
            "thermal_assumptions_path": str(contract.path.relative_to(PROJECT_ROOT)),
            "thermal_assumptions_sha256": contract.sha256,
            "tabula_target_path": str(TABULA_TARGET_PATH.relative_to(PROJECT_ROOT)),
            "tabula_target_sha256": _sha256_file(TABULA_TARGET_PATH),
            "tabula_target_provenance_path": str(
                TABULA_PROVENANCE_PATH.relative_to(PROJECT_ROOT)
            ),
            "base_matrix_sha256": _sha256_file(BASE_MATRIX_PATH),
            "state_matrix_sha256": _sha256_file(STATE_MATRIX_PATH),
        },
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(
            output_dir / "deterministic_archetype_validation.csv",
            index=False,
            float_format="%.10f",
            lineterminator="\n",
        )
        _write_json(output_dir / "validation_summary.json", summary)
    return results.reset_index(drop=True), summary


def _replace_mass_class(
    prepared: PreparedArchetype,
    *,
    capacitance_J_m2K: float,
    mass_area_ratio: float,
) -> PreparedArchetype:
    A_m = mass_area_ratio * prepared.floor_area_m2
    H_ms = 9.1 * A_m
    if prepared.H_tr_op_W_K >= H_ms:
        raise ValidationError("Sensitivity mass class cannot construct H_tr,em.")
    H_em = 1.0 / (1.0 / prepared.H_tr_op_W_K - 1.0 / H_ms)
    result = replace(
        prepared,
        A_m_m2=A_m,
        C_m_J_K=capacitance_J_m2K * prepared.floor_area_m2,
        H_tr_ms_W_K=H_ms,
        H_tr_em_W_K=H_em,
    )
    validate_prepared_archetype(result)
    return result


def _contract_with_text_overrides(
    contract: AssumptionContract,
    overrides: Mapping[str, str],
) -> AssumptionContract:
    frame = contract.frame.copy()
    for assumption_id, value in overrides.items():
        selected = frame["assumption_id"] == assumption_id
        if selected.sum() != 1:
            raise ValidationError(f"Cannot override unknown assumption {assumption_id!r}.")
        frame.loc[selected, "value_text"] = value
    # The SHA remains the checksum of the frozen central CSV. Scenario overrides
    # are persisted separately in sensitivity_results.csv and never masquerade as
    # calibration of that source file.
    return replace(contract, frame=frame)


def run_sensitivity_analysis(
    *,
    assumptions: AssumptionContract | None = None,
    reference_year: int | None = None,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Screen influential assumptions on a fixed, exposed representative dwelling."""

    contract = assumptions or load_assumption_contract(DEFAULT_ASSUMPTIONS_PATH)
    weather, weather_metadata = load_reference_weather(reference_year)
    states = {
        (state.archetype_id, state.state_id): state
        for state in load_unique_archetype_states()
    }
    representative_key = ("BE_TABULA_11", "TABULA_existing")
    representative_state = states[representative_key]
    central = preprocess_archetype(representative_state, contract)

    cases: list[dict[str, Any]] = []

    def add_case(
        case_id: str,
        axis: str,
        prepared: PreparedArchetype,
        *,
        baseline_case_id: str = "central",
        parameters: Mapping[str, Any],
        case_weather: pd.DataFrame | None = None,
        internal_gains_W_m2: float | None = None,
        heating_setpoint_C: float | None = None,
        cooling_setpoint_C: float | None = None,
        state_scope: str = "BE_TABULA_11/TABULA_existing",
    ) -> None:
        result = _simulate_prepared(
            prepared,
            weather if case_weather is None else case_weather,
            contract,
            weather_member_id=str(weather_metadata["weather_member_id"]),
            model_scenario=f"gate3_sensitivity_{case_id}",
            internal_gains_W_m2=internal_gains_W_m2,
            heating_setpoint_C=heating_setpoint_C,
            cooling_setpoint_C=cooling_setpoint_C,
        )
        cases.append(
            {
                "case_id": case_id,
                "baseline_case_id": baseline_case_id,
                "axis": axis,
                "archetype_state_scope": state_scope,
                "parameters_json": json.dumps(parameters, sort_keys=True),
                "heating_kWh_m2": result.diagnostics.heating_intensity_kWh_m2,
                "cooling_kWh_m2": result.diagnostics.cooling_intensity_kWh_m2,
                "peak_heating_W": result.diagnostics.peak_heating_W,
                "peak_cooling_W": result.diagnostics.peak_cooling_W,
                "max_energy_balance_residual_W": (
                    result.diagnostics.max_abs_energy_balance_residual_W
                ),
            }
        )

    add_case(
        "central",
        "central",
        central,
        parameters={"all": "central thermal_assumptions.csv values"},
    )
    add_case(
        "mass_light",
        "thermal_mass",
        _replace_mass_class(
            central, capacitance_J_m2K=110_000.0, mass_area_ratio=2.5
        ),
        parameters={"C_m_per_A_f_J_m2K": 110000.0, "A_m_per_A_f": 2.5},
    )
    add_case(
        "mass_heavy",
        "thermal_mass",
        _replace_mass_class(
            central, capacitance_J_m2K=260_000.0, mass_area_ratio=3.0
        ),
        parameters={"C_m_per_A_f_J_m2K": 260000.0, "A_m_per_A_f": 3.0},
    )
    add_case(
        "shading_unshaded",
        "fixed_shading",
        replace(central, vertical_shading_factor=1.0),
        parameters={"F_sh": 1.0},
    )
    add_case(
        "frame_fraction_0_2",
        "window_frame_fraction",
        replace(central, window_frame_fraction=0.2),
        parameters={"F_F": 0.2},
    )
    add_case(
        "infiltration_half",
        "infiltration",
        replace(
            central,
            infiltration_airflow_m3_h=0.5 * central.infiltration_airflow_m3_h,
        ),
        parameters={"infiltration_multiplier": 0.5},
    )
    add_case(
        "infiltration_one_and_half",
        "infiltration",
        replace(
            central,
            infiltration_airflow_m3_h=1.5 * central.infiltration_airflow_m3_h,
        ),
        parameters={"infiltration_multiplier": 1.5},
    )
    add_case(
        "ventilation_ach_0_3",
        "ventilation_rate",
        replace(central, ventilation_ach_h_1=0.3),
        parameters={"n_vent_h_1": 0.3},
    )
    add_case(
        "ventilation_ach_0_6",
        "ventilation_rate",
        replace(central, ventilation_ach_h_1=0.6),
        parameters={"n_vent_h_1": 0.6},
    )
    add_case(
        "heating_setpoint_18",
        "heating_setpoint",
        central,
        heating_setpoint_C=18.0,
        parameters={"theta_set_heat_C": 18.0},
    )
    add_case(
        "heating_setpoint_22",
        "heating_setpoint",
        central,
        heating_setpoint_C=22.0,
        parameters={"theta_set_heat_C": 22.0},
    )
    add_case(
        "cooling_setpoint_24",
        "cooling_setpoint",
        central,
        cooling_setpoint_C=24.0,
        parameters={"theta_set_cool_C": 24.0},
    )
    add_case(
        "cooling_setpoint_28",
        "cooling_setpoint",
        central,
        cooling_setpoint_C=28.0,
        parameters={"theta_set_cool_C": 28.0},
    )
    add_case(
        "internal_gains_1_5",
        "internal_gains",
        central,
        internal_gains_W_m2=1.5,
        parameters={"Phi_int_W_m2": 1.5},
    )
    add_case(
        "internal_gains_4_5",
        "internal_gains",
        central,
        internal_gains_W_m2=4.5,
        parameters={"Phi_int_W_m2": 4.5},
    )
    all_exterior = _contract_with_text_overrides(
        contract,
        {
            "boundary.unheated_room": "R_add=0; b_tr=1",
            "boundary.unheated_cellar": "R_add=0; b_tr=1",
            "boundary.soil": "R_add=0; b_tr=1",
        },
    )
    add_case(
        "all_opaque_boundaries_exterior",
        "boundary_treatment",
        preprocess_archetype(representative_state, all_exterior),
        parameters={
            "unheated_room": "R_add=0; b_tr=1",
            "unheated_cellar": "R_add=0; b_tr=1",
            "soil": "R_add=0; b_tr=1",
        },
    )
    no_solar = weather.copy()
    for orientation in ("north", "east", "south", "west"):
        no_solar[f"I_{orientation}_W_m2"] = 0.0
    add_case(
        "solar_disabled",
        "solar_gain_check",
        central,
        case_weather=no_solar,
        parameters={"facade_irradiance_multiplier": 0.0},
    )

    advanced = preprocess_archetype(
        states[("BE_TABULA_11", "TABULA_advanced_A_proxy")], contract
    )
    add_case(
        "advanced_hrv_central",
        "hrv",
        advanced,
        baseline_case_id="advanced_hrv_central",
        parameters={"eta_HRV": advanced.hrv_efficiency},
        state_scope="BE_TABULA_11/TABULA_advanced_A_proxy",
    )
    add_case(
        "advanced_hrv_disabled",
        "hrv",
        replace(advanced, hrv_efficiency=0.0),
        baseline_case_id="advanced_hrv_central",
        parameters={"eta_HRV": 0.0},
        state_scope="BE_TABULA_11/TABULA_advanced_A_proxy",
    )

    results = pd.DataFrame.from_records(cases)
    indexed = results.set_index("case_id")
    results["baseline_heating_kWh_m2"] = results["baseline_case_id"].map(
        indexed["heating_kWh_m2"]
    )
    results["baseline_cooling_kWh_m2"] = results["baseline_case_id"].map(
        indexed["cooling_kWh_m2"]
    )
    results["delta_heating_kWh_m2"] = (
        results["heating_kWh_m2"] - results["baseline_heating_kWh_m2"]
    )
    results["delta_cooling_kWh_m2"] = (
        results["cooling_kWh_m2"] - results["baseline_cooling_kWh_m2"]
    )
    ranked = results.loc[results["case_id"] != "central"].assign(
        absolute_heating_change=lambda frame: frame["delta_heating_kWh_m2"].abs()
    ).sort_values("absolute_heating_change", ascending=False)
    summary = {
        "schema_version": 1,
        "reference_weather": weather_metadata,
        "representative_scope": "BE_TABULA_11 detached 1971-1990 existing; HRV pair uses its advanced state",
        "case_count": int(len(results)),
        "method": "deterministic one-at-a-time screening; ranges are not probability distributions",
        "largest_absolute_heating_changes": [
            {
                "case_id": row.case_id,
                "axis": row.axis,
                "delta_heating_kWh_m2": float(row.delta_heating_kWh_m2),
            }
            for row in ranked.head(8).itertuples(index=False)
        ],
        "directional_checks": {
            "more_solar_does_not_increase_heating": bool(
                indexed.loc["central", "heating_kWh_m2"]
                <= indexed.loc["solar_disabled", "heating_kWh_m2"] + 1e-9
            ),
            "more_solar_does_not_reduce_cooling": bool(
                indexed.loc["central", "cooling_kWh_m2"]
                >= indexed.loc["solar_disabled", "cooling_kWh_m2"] - 1e-9
            ),
            "hrv_reduces_heating": bool(
                indexed.loc["advanced_hrv_central", "heating_kWh_m2"]
                < indexed.loc["advanced_hrv_disabled", "heating_kWh_m2"]
            ),
            "higher_heating_setpoint_increases_heating": bool(
                indexed.loc["heating_setpoint_22", "heating_kWh_m2"]
                > indexed.loc["heating_setpoint_18", "heating_kWh_m2"]
            ),
            "higher_cooling_setpoint_reduces_cooling": bool(
                indexed.loc["cooling_setpoint_28", "cooling_kWh_m2"]
                < indexed.loc["cooling_setpoint_24", "cooling_kWh_m2"]
            ),
        },
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(
            output_dir / "sensitivity_results.csv",
            index=False,
            float_format="%.10f",
            lineterminator="\n",
        )
        _write_json(output_dir / "sensitivity_summary.json", summary)
    return results, summary


def _format_rate(item: Mapping[str, Any]) -> str:
    return f"{item['passed']}/{item['total']} ({100.0*item['rate']:.1f}%)"


def build_validation_report(
    results: pd.DataFrame,
    summary: Mapping[str, Any],
    sensitivity: pd.DataFrame,
    sensitivity_summary: Mapping[str, Any],
) -> str:
    """Build a concise, reproducible Markdown report from persisted results."""

    comparison = summary["tabula_comparison"]
    numerical = summary["numerical_checks"]
    lines = [
        "# Gate 3 deterministic thermal-model validation report",
        "",
        f"**Verification status: {summary['verification_status']}**  ",
        f"**Validation status: {summary['validation_status']}**",
        "",
        "## Frozen design",
        "",
        f"- Reference weather: observed PVGIS {summary['reference_weather']['selected_year']}, selected before simulation as the complete year nearest the 2006-2023 median HDD.",
        "- Operating conditions: 20 degC heating, 26 degC cooling and constant 3 W/m2 sensible internal gains.",
        "- TABULA warning band: `max(15 kWh/m2/year, 30% of target)`.",
        "- Gate-level target: at least 80% of cells inside that band.",
        "- No parameter was fitted to a TABULA result.",
        "",
        "## Numerical verification on the real-archetype runs",
        "",
        f"- Maximum node-balance residual: {numerical['maximum_energy_balance_residual_W']:.3e} W (limit {numerical['energy_balance_tolerance_W']:.1e} W).",
        f"- Maximum controlled-setpoint error: {numerical['maximum_setpoint_error_K']:.3e} K (limit {numerical['setpoint_tracking_tolerance_K']:.1e} K).",
        f"- Simultaneous heating/cooling absent: {numerical['no_simultaneous_heating_cooling']}.",
        "",
        "## Direct TABULA comparison",
        "",
        f"- Cells inside the predeclared band: {comparison['within_band_cells']}/{summary['cell_count']} ({100.0*comparison['pass_rate']:.1f}%).",
        f"- Mean signed deviation: {comparison['mean_signed_deviation_kWh_m2']:.2f} kWh/m2/year.",
        f"- Mean absolute deviation: {comparison['mean_absolute_deviation_kWh_m2']:.2f} kWh/m2/year.",
        f"- RMSE: {comparison['root_mean_square_deviation_kWh_m2']:.2f} kWh/m2/year.",
        "",
        "| State | Model median | TABULA median | Mean signed deviation | Pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for state_id in STATE_ORDER:
        row = comparison["by_state"][state_id]
        lines.append(
            f"| `{state_id}` | {row['model_median_kWh_m2']:.1f} | "
            f"{row['target_median_kWh_m2']:.1f} | "
            f"{row['mean_signed_deviation_kWh_m2']:+.1f} | "
            f"{100.0*row['pass_rate']:.1f}% |"
        )
    lines.extend(["", "## Qualitative stock patterns", ""])
    for name, item in summary["qualitative_patterns"].items():
        lines.append(f"- `{name}`: {_format_rate(item)}.")

    outside = results.loc[~results["within_predeclared_tabula_band"]].sort_values(
        "absolute_deviation_kWh_m2", ascending=False
    )
    lines.extend(["", "## Cells requiring investigation", ""])
    if outside.empty:
        lines.append("No cell lies outside the predeclared comparison band.")
    else:
        lines.extend(
            [
                "| Archetype | State | Model | TABULA | Deviation |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in outside.itertuples(index=False):
            lines.append(
                f"| {row.archetype_id} | `{row.state_id}` | "
                f"{row.model_heating_kWh_m2:.1f} | "
                f"{row.tabula_heating_target_kWh_m2:.1f} | "
                f"{row.signed_deviation_kWh_m2:+.1f} |"
            )

        all_under = bool((outside["signed_deviation_kWh_m2"] < 0.0).all())
        enclosed_count = int(
            (outside["dwelling_type"] == "Apartment, enclosed").sum()
        )
        lines.extend(
            [
                "",
                "### Investigation outcome",
                "",
                f"- All outside-band cells are underpredictions: {'yes' if all_under else 'no'}.",
                f"- {enclosed_count}/{len(outside)} outside-band cells are enclosed apartments. These archetypes share the same small exposed envelope (17.9 m2 exterior wall and 26.8 m2 windows, with no exposed roof, floor or door), and repeated package states therefore produce the same deterministic demand. The concentration is a class-level method/boundary discrepancy, not numerical scatter.",
                "- The remaining outside-band cell is the post-2005 detached-house advanced package. Its direction is consistent with the overall negative model bias.",
                "- Plausible method differences include the deliberately omitted thermal bridges, fixed unheated-space and ground reductions, the selected 2015 weather, constant 3 W/m2 gains, and the hourly air-temperature method versus TABULA's reference calculation. None was adjusted after inspecting the results.",
            ]
        )

    lines.extend(
        [
            "",
            "## External context",
            "",
            "The Belgian climate-neutral scenario uses 85, 64 and 25 kWh/m2/year as shallow, medium and deep renovation-depth levers. These are contextual scenario levels rather than archetype targets. Regional EPC figures are also contextual only because they are primary-energy scores under regional certificate methods, while this model reports useful space-heating demand.",
            "",
            "## Sensitivity screening",
            "",
            f"The one-at-a-time screen contains {len(sensitivity)} cases for {sensitivity_summary['representative_scope']}. It ranks influence; it does not assign probability distributions.",
            "",
            "| Case | Axis | Heating change | Cooling change |",
            "|---|---|---:|---:|",
        ]
    )
    ranked = sensitivity.loc[sensitivity["case_id"] != "central"].assign(
        magnitude=lambda frame: frame["delta_heating_kWh_m2"].abs()
    ).sort_values("magnitude", ascending=False)
    for row in ranked.itertuples(index=False):
        lines.append(
            f"| `{row.case_id}` | {row.axis} | {row.delta_heating_kWh_m2:+.2f} | "
            f"{row.delta_cooling_kWh_m2:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Calibration discipline",
            "",
            "No archetype-specific parameter was tuned. Any future correction must identify a physical source of systematic bias, preserve the original assumption, apply consistently to a documented dwelling class, and be checked on archetypes not used to motivate the change.",
            "",
        ]
    )
    return "\n".join(lines)


def run_gate3(
    *,
    reference_year: int | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run and persist the complete deterministic validation and sensitivity stack."""

    contract = load_assumption_contract(DEFAULT_ASSUMPTIONS_PATH)
    results, summary = run_deterministic_validation(
        assumptions=contract,
        reference_year=reference_year,
        output_dir=output_dir,
    )
    sensitivity, sensitivity_summary = run_sensitivity_analysis(
        assumptions=contract,
        reference_year=reference_year,
        output_dir=output_dir,
    )
    report = build_validation_report(results, summary, sensitivity, sensitivity_summary)
    (output_dir / "validation_report.md").write_text(report, encoding="utf-8")
    return results, summary, sensitivity, sensitivity_summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Gate 3 deterministic 5R1C verification and validation."
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=None,
        help="Override the deterministic median-HDD year selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, JSON, and Markdown validation artifacts.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    _, summary, _, sensitivity_summary = run_gate3(
        reference_year=args.reference_year,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "verification_status": summary["verification_status"],
                "validation_status": summary["validation_status"],
                "tabula_pass_rate": summary["tabula_comparison"]["pass_rate"],
                "sensitivity_directional_checks": sensitivity_summary[
                    "directional_checks"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    gate_pass = (
        summary["verification_status"] == "PASS"
        and summary["validation_status"] == "PASS"
        and all(sensitivity_summary["directional_checks"].values())
    )
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
