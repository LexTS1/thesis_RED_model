"""Authenticated paired current-versus-2050 stock/climate counterfactual.

The experiment is a two-by-two factorial design.  The building-stock weights
(2025 or 2050) and weather forcing (observed reference or 2050 morph) are
changed independently while every physical archetype, model assumption, and
occupant-seed identity is held fixed.  One simulated dwelling-year is streamed
to both stock-weight sets, so the comparison never confounds a stock effect
with a different occupant draw and never approximates coincident peaks by
summing individual peaks.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from climate.src.load_cordex import load_config
from thermal_model.behaviour import (
    BehaviourRequest,
    dwelling_class,
    generate_behaviour,
    load_behaviour_assumptions,
)
from thermal_model.contracts import ArchetypeStateInput, load_assumption_contract
from thermal_model.validation import (
    CLIMATE_CONFIG_PATH,
    HDD_COMPARISON_PATH,
    load_reference_weather,
    load_unique_archetype_states,
)

from .contracts import (
    CLIMATE_SCENARIOS,
    MODEL_CONTRACT_VERSION,
    OBSERVED_REFERENCE_SCENARIO,
    MonteCarloContractError,
    MonteCarloResult,
    WeatherMember,
    archetype_state_sha256,
    canonical_sha256,
    complete_weather_forcing_sha256,
    diagnostics_to_record,
    validate_weather_member,
)
from .design import build_balanced_manifest, ordered_seed_bank_sha256
from .interface import _simulate_with_behaviour
from .runner import (
    _acquire_streaming_execution_lock,
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _read_json,
    _release_streaming_execution_lock,
    _sha256_file,
    _verify_file,
)
from .scenarios import resolve_model_scenario
from .weather import load_weather_member


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR_RESULTS_DIR = (
    PROJECT_ROOT / "thermal_model/data/monte_carlo/supervisor_results_preliminary"
)
STOCK_SOURCE_PATH = (
    PROJECT_ROOT
    / "BE_building_stock/data/scenarios/renovation/"
    "archetype_matrix_2050_renovation_scenarios.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "thermal_model/data/monte_carlo/current_vs_2050"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "thermal_model/figures/current_vs_2050"
CONTRACT_VERSION = "gate5_current_vs_2050_factorial_v1"
OBSERVED_MEMBER_ID = "weather_observed_reference_pvgis_2015"
BELGIUM_REGION_ID = "Belgium_modelled_stock"
TARGET_YEARS = (2025, 2050)
COUNT_COLUMN = {
    2025: "initial_state_dwellings_2025",
    2050: "state_dwellings_2050",
}
CASE_ORDER = ("Q00", "Q10", "Q01", "Q11")
CASE_LABELS = {
    "Q00": "2025 stock / observed climate",
    "Q10": "2050 stock / observed climate",
    "Q01": "2025 stock / 2050 climate",
    "Q11": "2050 stock / 2050 climate",
}
EFFECT_ORDER = ("renovation", "climate", "interaction", "combined")
EFFECT_FORMULAE = {
    "renovation": "Q10 - Q00",
    "climate": "Q01 - Q00",
    "interaction": "Q11 - Q10 - Q01 + Q00",
    "combined": "Q11 - Q00",
}


def _observed_weather_member() -> WeatherMember:
    """Construct the checksum-bound, unmorphed PVGIS-2015 reference member."""

    frame, metadata = load_reference_weather(year=2015)
    config = load_config(CLIMATE_CONFIG_PATH)
    observed = config["observed_weather"]
    forcing_hash = complete_weather_forcing_sha256(frame)
    facade_hashes = tuple(
        (orientation, str(metadata["facade_source_sha256"][orientation]))
        for orientation in ("north", "east", "south", "west")
    )
    metadata_hash = canonical_sha256(
        {
            "contract": "observed_reference_weather_metadata_v1",
            "metadata": metadata,
        }
    )
    no_morph_hash = canonical_sha256(
        {
            "contract": "observed_reference_no_morph_v1",
            "reference_year": 2015,
            "temperature_delta_C": 0.0,
            "irradiance_multiplier": 1.0,
        }
    )
    weather_contract_hash = canonical_sha256(
        {
            "contract": "gate5_observed_reference_weather_contract_v1",
            "member_id": OBSERVED_MEMBER_ID,
            "observed_dataset_sha256": metadata["observed_dataset_sha256"],
            "metadata_sha256": metadata_hash,
            "selection_evidence_sha256": _sha256_file(HDD_COMPARISON_PATH),
            "no_morph_contract_sha256": no_morph_hash,
            "facade_source_sha256": dict(facade_hashes),
            "forcing_sha256": forcing_hash,
        }
    )
    return validate_weather_member(
        WeatherMember(
            member_id=OBSERVED_MEMBER_ID,
            climate_scenario_id=OBSERVED_REFERENCE_SCENARIO,
            climate_target="observed_reference_year",
            weather_pair_id="pvgis_2015",
            observed_pvgis_year=2015,
            is_leap_year=False,
            row_count=len(frame),
            frame=frame,
            site_id=(
                f"pvgis_brussels_{float(observed['latitude']):.3f}_"
                f"{float(observed['longitude']):.3f}"
            ),
            latitude=float(observed["latitude"]),
            longitude=float(observed["longitude"]),
            elevation_m=float(observed["elevation_m"]),
            timezone=str(observed["timestamp"]["timezone"]),
            gcm_model="not_applicable_observed",
            rcm_model="not_applicable_observed",
            cordex_ensemble_member="not_applicable_observed",
            member_sha256=str(metadata["observed_dataset_sha256"]),
            metadata_sha256=metadata_hash,
            manifest_sha256=_sha256_file(HDD_COMPARISON_PATH),
            morph_contract_sha256=no_morph_hash,
            facade_source_sha256=facade_hashes,
            weather_contract_sha256=weather_contract_hash,
            forcing_sha256=forcing_hash,
        )
    )


def load_comparison_weather_member(member_id: str) -> WeatherMember:
    """Load either the declared observed reference or an ensemble member."""

    if str(member_id) == OBSERVED_MEMBER_ID:
        return _observed_weather_member()
    return load_weather_member(str(member_id))


def load_dual_year_stock_weights(path: str | Path = STOCK_SOURCE_PATH) -> pd.DataFrame:
    """Validate the common 2025/2050 regional stock-weight contract."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Stock source does not exist: {resolved}")
    source = pd.read_csv(resolved)
    required = {
        "scenario",
        "region",
        "archetype_id",
        "dwelling_type",
        "construction_period",
        "state_id",
        "renovation_state",
        "regional_number_of_dwellings",
        "regional_modelled_stock_dwellings",
        *COUNT_COLUMN.values(),
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise MonteCarloContractError(f"Dual-year stock source misses {missing}.")
    frame = source.loc[:, sorted(required)].copy(deep=True)
    if len(frame) != 225 or frame.duplicated(["region", "archetype_id", "state_id"]).any():
        raise MonteCarloContractError(
            "Dual-year stock contract must contain 225 unique regional/state rows."
        )
    if frame[["archetype_id", "state_id"]].drop_duplicates().shape[0] != 75:
        raise MonteCarloContractError("Dual-year stock contract must cover 75 physics cells.")
    if set(frame["scenario"].astype(str)) != {"central"}:
        raise MonteCarloContractError("The factorial comparison accepts central stock only.")
    if not frame["renovation_state"].astype(str).equals(frame["state_id"].astype(str)):
        raise MonteCarloContractError("renovation_state differs from state_id.")
    numeric = [
        "regional_number_of_dwellings",
        "regional_modelled_stock_dwellings",
        *COUNT_COLUMN.values(),
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all() or (
        frame[numeric].to_numpy(dtype=float) < 0.0
    ).any():
        raise MonteCarloContractError("Dual-year stock counts must be finite/non-negative.")
    archetype_reported = frame.groupby(["region", "archetype_id"], sort=False)[
        "regional_number_of_dwellings"
    ].first()
    for year, column in COUNT_COLUMN.items():
        reconstructed = frame.groupby(["region", "archetype_id"], sort=False)[column].sum()
        if not np.allclose(reconstructed, archetype_reported, rtol=0.0, atol=1.0e-6):
            raise MonteCarloContractError(
                f"{year} state counts do not reconstruct regional archetype totals."
            )
    regional_reported = frame.groupby("region", sort=False)[
        "regional_modelled_stock_dwellings"
    ].first()
    for year, column in COUNT_COLUMN.items():
        regional = frame.groupby("region", sort=False)[column].sum()
        if not np.allclose(regional, regional_reported, rtol=0.0, atol=2.0e-6):
            raise MonteCarloContractError(
                f"{year} state counts do not reconstruct regional stock totals."
            )
    if not np.isclose(frame[COUNT_COLUMN[2025]].sum(), frame[COUNT_COLUMN[2050]].sum(), atol=1e-5):
        raise MonteCarloContractError("2025 and 2050 modelled dwelling totals differ.")
    frame = frame.sort_values(["region", "archetype_id", "state_id"], kind="stable")
    frame["stock_source_sha256"] = _sha256_file(resolved)
    content_columns = [
        "region",
        "archetype_id",
        "state_id",
        COUNT_COLUMN[2025],
        COUNT_COLUMN[2050],
    ]
    frame["dual_year_stock_weights_sha256"] = canonical_sha256(
        {"contract": "dual_year_stock_weights_v1", "records": frame[content_columns].to_dict("records")}
    )
    return frame.reset_index(drop=True)


class _DualYearAccumulator:
    """Retain seed-level annual totals and true coincident stock-hour profiles."""

    def __init__(self, weights: pd.DataFrame, seeds: Sequence[int]) -> None:
        self.weights = load_dual_year_stock_weights(STOCK_SOURCE_PATH)
        expected_hash = str(weights["dual_year_stock_weights_sha256"].iloc[0])
        if str(self.weights["dual_year_stock_weights_sha256"].iloc[0]) != expected_hash:
            raise MonteCarloContractError("Worker stock weights differ from the design.")
        self.seeds = tuple(int(value) for value in seeds)
        self.seed_rank = {value: index + 1 for index, value in enumerate(self.seeds)}
        self.seed_set = set(self.seeds)
        self.cells = set(
            map(tuple, self.weights[["archetype_id", "state_id"]].drop_duplicates().to_numpy())
        )
        self.regions = tuple(sorted(self.weights["region"].astype(str).unique()))
        self.region_order = (*self.regions, BELGIUM_REGION_ID)
        self.weights_by_cell = {
            tuple(key): group.copy(deep=True)
            for key, group in self.weights.groupby(["archetype_id", "state_id"], sort=False)
        }
        self.timestamps: pd.DatetimeIndex | None = None
        self.heating_W: np.ndarray | None = None
        self.cooling_W: np.ndarray | None = None
        self.seen: set[tuple[str, str, int]] = set()
        self.run_ids: set[str] = set()
        self.member_id: str | None = None
        self.cell_floor_area: dict[tuple[str, str], float] = {}
        self.cell_physics_sha: dict[tuple[str, str], str] = {}
        self.annual_heat = defaultdict(float)
        self.annual_cool = defaultdict(float)
        self.sum_peak_heat = defaultdict(float)
        self.sum_peak_cool = defaultdict(float)

    @property
    def completed_run_count(self) -> int:
        return len(self.seen)

    def _register(self, record: Mapping[str, Any]) -> None:
        required = {
            "run_id", "archetype_id", "state_id", "occupant_seed", "floor_area_m2",
            "annual_heating_kWh", "annual_cooling_kWh", "peak_heating_W",
            "peak_cooling_W", "weather_member_id", "archetype_state_sha256",
        }
        missing = sorted(required.difference(record))
        if missing:
            raise MonteCarloContractError(f"Counterfactual diagnostic misses {missing}.")
        member_id = str(record["weather_member_id"])
        if self.member_id is None:
            self.member_id = member_id
        elif member_id != self.member_id:
            raise MonteCarloContractError("Accumulator received mixed weather members.")
        run_id = str(record["run_id"])
        cell = (str(record["archetype_id"]), str(record["state_id"]))
        seed = int(record["occupant_seed"])
        key = (*cell, seed)
        if run_id in self.run_ids or key in self.seen:
            raise MonteCarloContractError(f"Duplicate counterfactual run {key}.")
        if cell not in self.cells or seed not in self.seed_set:
            raise MonteCarloContractError(f"Unexpected counterfactual run {key}.")
        values = np.asarray(
            [record["floor_area_m2"], record["annual_heating_kWh"],
             record["annual_cooling_kWh"], record["peak_heating_W"],
             record["peak_cooling_W"]], dtype=float,
        )
        if not np.isfinite(values).all() or values[0] <= 0.0 or (values[1:] < 0.0).any():
            raise MonteCarloContractError(f"Invalid counterfactual diagnostic {key}.")
        area, annual_heat, annual_cool, peak_heat, peak_cool = values
        prior_area = self.cell_floor_area.setdefault(cell, float(area))
        if not np.isclose(prior_area, area, rtol=0.0, atol=1e-10):
            raise MonteCarloContractError(f"Floor area varies inside physics cell {cell}.")
        physics_hash = str(record["archetype_state_sha256"])
        prior_hash = self.cell_physics_sha.setdefault(cell, physics_hash)
        if physics_hash != prior_hash:
            raise MonteCarloContractError(f"Physics checksum varies inside cell {cell}.")
        for weight in self.weights_by_cell[cell].itertuples(index=False):
            region = str(weight.region)
            for year, column in COUNT_COLUMN.items():
                count = float(getattr(weight, column))
                for target_region in (region, BELGIUM_REGION_ID):
                    annual_key = (year, target_region, seed)
                    self.annual_heat[annual_key] += count * annual_heat
                    self.annual_cool[annual_key] += count * annual_cool
                    peak_key = (year, target_region)
                    self.sum_peak_heat[peak_key] += count * peak_heat / len(self.seeds)
                    self.sum_peak_cool[peak_key] += count * peak_cool / len(self.seeds)
        self.run_ids.add(run_id)
        self.seen.add(key)

    def add(self, result: MonteCarloResult) -> None:
        record = diagnostics_to_record(result.diagnostics)
        hourly = result.hourly
        timestamps = pd.DatetimeIndex(hourly["timestamp_utc"])
        heat = pd.to_numeric(hourly["heating_demand_W"], errors="raise").to_numpy(float)
        cool = pd.to_numeric(hourly["cooling_demand_W"], errors="raise").to_numpy(float)
        if not np.isclose(heat.sum() / 1000.0, float(record["annual_heating_kWh"]), rtol=1e-9, atol=1e-6):
            raise MonteCarloContractError("Hourly heating does not reconcile with diagnostics.")
        if not np.isclose(cool.sum() / 1000.0, float(record["annual_cooling_kWh"]), rtol=1e-9, atol=1e-6):
            raise MonteCarloContractError("Hourly cooling does not reconcile with diagnostics.")
        if self.timestamps is None:
            self.timestamps = timestamps.copy()
            shape = (len(TARGET_YEARS), len(self.region_order), len(timestamps))
            self.heating_W = np.zeros(shape, dtype=float)
            self.cooling_W = np.zeros(shape, dtype=float)
        elif not timestamps.equals(self.timestamps):
            raise MonteCarloContractError("Counterfactual hourly timestamps are misaligned.")
        self._register(record)
        assert self.heating_W is not None and self.cooling_W is not None
        cell = (str(record["archetype_id"]), str(record["state_id"]))
        national = self.region_order.index(BELGIUM_REGION_ID)
        for weight in self.weights_by_cell[cell].itertuples(index=False):
            region_index = self.region_order.index(str(weight.region))
            for year_index, year in enumerate(TARGET_YEARS):
                scale = float(getattr(weight, COUNT_COLUMN[year])) / len(self.seeds)
                self.heating_W[year_index, region_index] += scale * heat
                self.cooling_W[year_index, region_index] += scale * cool
                self.heating_W[year_index, national] += scale * heat
                self.cooling_W[year_index, national] += scale * cool

    def restore(
        self,
        diagnostics: pd.DataFrame,
        *,
        timestamp_ns: np.ndarray,
        heating_W: np.ndarray,
        cooling_W: np.ndarray,
        region_order: Sequence[str],
    ) -> None:
        if self.seen or self.timestamps is not None:
            raise MonteCarloContractError("Restore requires an empty accumulator.")
        if tuple(str(value) for value in region_order) != self.region_order:
            raise MonteCarloContractError("Checkpoint region order changed.")
        timestamps = pd.DatetimeIndex(pd.to_datetime(timestamp_ns, utc=True))
        expected = (len(TARGET_YEARS), len(self.region_order), len(timestamps))
        heat = np.asarray(heating_W, dtype=float)
        cool = np.asarray(cooling_W, dtype=float)
        if heat.shape != expected or cool.shape != expected:
            raise MonteCarloContractError("Checkpoint array shape changed.")
        if not np.isfinite(heat).all() or not np.isfinite(cool).all():
            raise MonteCarloContractError("Checkpoint arrays contain non-finite values.")
        for record in diagnostics.to_dict("records"):
            self._register(record)
        self.timestamps = timestamps
        self.heating_W = heat.copy()
        self.cooling_W = cool.copy()
        self._reconcile(partial=True)

    def snapshot(self) -> dict[str, np.ndarray]:
        if self.timestamps is None or self.heating_W is None or self.cooling_W is None:
            raise MonteCarloContractError("Cannot checkpoint an empty accumulator.")
        return {
            "timestamp_ns": self.timestamps.asi8.copy(),
            "heating_W": self.heating_W.copy(),
            "cooling_W": self.cooling_W.copy(),
        }

    def _stock_geometry(self) -> dict[tuple[int, str], tuple[float, float]]:
        geometry: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
        for weight in self.weights.itertuples(index=False):
            cell = (str(weight.archetype_id), str(weight.state_id))
            if cell not in self.cell_floor_area:
                continue
            area = self.cell_floor_area[cell]
            for year, column in COUNT_COLUMN.items():
                count = float(getattr(weight, column))
                for region in (str(weight.region), BELGIUM_REGION_ID):
                    geometry[(year, region)][0] += count
                    geometry[(year, region)][1] += count * area
        return {key: (value[0], value[1]) for key, value in geometry.items()}

    def _reconcile(self, *, partial: bool) -> None:
        assert self.heating_W is not None and self.cooling_W is not None
        completed_seeds = sorted({item[2] for item in self.seen})
        if not completed_seeds:
            raise MonteCarloContractError("Accumulator has no complete seed.")
        for year_index, year in enumerate(TARGET_YEARS):
            for region_index, region in enumerate(self.region_order):
                mean_heat = np.mean([self.annual_heat[(year, region, seed)] for seed in completed_seeds])
                mean_cool = np.mean([self.annual_cool[(year, region, seed)] for seed in completed_seeds])
                if partial:
                    fraction = len(completed_seeds) / len(self.seeds)
                    array_heat = self.heating_W[year_index, region_index].sum() / 1000.0 / fraction
                    array_cool = self.cooling_W[year_index, region_index].sum() / 1000.0 / fraction
                else:
                    array_heat = self.heating_W[year_index, region_index].sum() / 1000.0
                    array_cool = self.cooling_W[year_index, region_index].sum() / 1000.0
                if not np.isclose(mean_heat, array_heat, rtol=1e-10, atol=1e-4):
                    raise MonteCarloContractError(f"Heating array does not reconcile for {year}/{region}.")
                if not np.isclose(mean_cool, array_cool, rtol=1e-10, atol=1e-4):
                    raise MonteCarloContractError(f"Cooling array does not reconcile for {year}/{region}.")

    def finalize(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        expected = {(*cell, seed) for cell in self.cells for seed in self.seeds}
        if self.seen != expected or len(self.run_ids) != len(expected):
            raise MonteCarloContractError("Counterfactual partition is incomplete.")
        if self.timestamps is None or self.heating_W is None or self.cooling_W is None:
            raise MonteCarloContractError("Counterfactual partition has no hourly arrays.")
        self._reconcile(partial=False)
        geometry = self._stock_geometry()
        annual_records: list[dict[str, Any]] = []
        for year in TARGET_YEARS:
            for region in self.region_order:
                dwellings, floor_area = geometry[(year, region)]
                for seed in self.seeds:
                    heat = self.annual_heat[(year, region, seed)]
                    cool = self.annual_cool[(year, region, seed)]
                    annual_records.append(
                        {
                            "weather_member_id": self.member_id,
                            "stock_year": year,
                            "region": region,
                            "occupant_seed": seed,
                            "occupant_seed_rank": self.seed_rank[seed],
                            "modelled_dwellings": dwellings,
                            "stock_floor_area_m2": floor_area,
                            "annual_heating_GWh": heat / 1e6,
                            "annual_potential_sensible_cooling_GWh": cool / 1e6,
                            "stock_heating_intensity_kWh_m2": heat / floor_area,
                            "stock_cooling_intensity_kWh_m2": cool / floor_area,
                        }
                    )
        hourly_records: list[pd.DataFrame] = []
        summary_records: list[dict[str, Any]] = []
        for year_index, year in enumerate(TARGET_YEARS):
            for region_index, region in enumerate(self.region_order):
                heat = self.heating_W[year_index, region_index]
                cool = self.cooling_W[year_index, region_index]
                dwellings, floor_area = geometry[(year, region)]
                heat_peak = int(np.argmax(heat))
                cool_peak = int(np.argmax(cool))
                hourly_records.append(
                    pd.DataFrame(
                        {
                            "timestamp_utc": self.timestamps,
                            "weather_member_id": self.member_id,
                            "stock_year": year,
                            "region": region,
                            "heating_demand_MW": heat / 1e6,
                            "potential_sensible_cooling_demand_MW": cool / 1e6,
                        }
                    )
                )
                summary_records.append(
                    {
                        "weather_member_id": self.member_id,
                        "stock_year": year,
                        "region": region,
                        "modelled_dwellings": dwellings,
                        "stock_floor_area_m2": floor_area,
                        "annual_heating_GWh": heat.sum() / 1e9,
                        "annual_potential_sensible_cooling_GWh": cool.sum() / 1e9,
                        "heating_intensity_kWh_m2": heat.sum() / 1000.0 / floor_area,
                        "potential_cooling_intensity_kWh_m2": cool.sum() / 1000.0 / floor_area,
                        "coincident_peak_heating_MW": heat[heat_peak] / 1e6,
                        "coincident_peak_potential_cooling_MW": cool[cool_peak] / 1e6,
                        "sum_individual_peak_heating_MW": self.sum_peak_heat[(year, region)] / 1e6,
                        "sum_individual_peak_potential_cooling_MW": self.sum_peak_cool[(year, region)] / 1e6,
                        "peak_heating_timestamp_utc": self.timestamps[heat_peak],
                        "peak_potential_cooling_timestamp_utc": self.timestamps[cool_peak],
                    }
                )
        return (
            pd.DataFrame.from_records(annual_records),
            pd.concat(hourly_records, ignore_index=True),
            pd.DataFrame.from_records(summary_records),
        )


def _design_inputs() -> tuple[
    tuple[ArchetypeStateInput, ...], tuple[WeatherMember, ...], tuple[int, ...], pd.DataFrame, dict[str, Any]
]:
    supervisor_contract_path = SUPERVISOR_RESULTS_DIR / "streaming_design_contract.json"
    supervisor = _read_json(supervisor_contract_path)
    unsigned = {key: value for key, value in supervisor.items() if key != "design_sha256"}
    if canonical_sha256(unsigned) != str(supervisor.get("design_sha256")):
        raise MonteCarloContractError("Supervisor streaming design checksum is invalid.")
    seeds = tuple(int(value) for value in supervisor["occupant_seeds"])
    if len(seeds) != 160 or ordered_seed_bank_sha256(seeds) != supervisor["occupant_seed_bank_sha256"]:
        raise MonteCarloContractError("Comparison requires the exact authenticated n=160 seed bank.")
    future_ids = tuple(item["weather_member_id"] for item in supervisor["weather_members"])
    expected_future_ids = tuple(f"weather_2050_{scenario}_pvgis_2015" for scenario in CLIMATE_SCENARIOS)
    if future_ids != expected_future_ids:
        raise MonteCarloContractError("Supervisor weather selection changed.")
    states = tuple(load_unique_archetype_states())
    if len(states) != 75:
        raise MonteCarloContractError("Comparison requires all 75 physics cells.")
    weather = (_observed_weather_member(), *(load_weather_member(value) for value in future_ids))
    weights = load_dual_year_stock_weights()
    weighted_cells = set(map(tuple, weights[["archetype_id", "state_id"]].drop_duplicates().to_numpy()))
    state_cells = {(item.archetype_id, item.state_id) for item in states}
    if state_cells != weighted_cells:
        raise MonteCarloContractError("Archetype states do not match dual-year stock cells.")
    scenario = resolve_model_scenario("central")
    thermal = load_assumption_contract()
    behaviour = load_behaviour_assumptions()
    weather_contracts = [
        {
            "weather_member_id": item.member_id,
            "climate_scenario_id": item.climate_scenario_id,
            "weather_pair_id": item.weather_pair_id,
            "weather_contract_sha256": item.weather_contract_sha256,
            "weather_forcing_sha256": item.forcing_sha256,
        }
        for item in weather
    ]
    partition_specs = [
        {
            "partition_id": "factorial_" + canonical_sha256(
                {
                    "contract": CONTRACT_VERSION,
                    "weather_member_id": item.member_id,
                    "weather_contract_sha256": item.weather_contract_sha256,
                    "seed_bank_sha256": ordered_seed_bank_sha256(seeds),
                    "stock_weights_sha256": str(weights["dual_year_stock_weights_sha256"].iloc[0]),
                    "model_scenario": scenario.definition(),
                    "thermal_assumptions_sha256": thermal.sha256,
                    "behaviour_assumptions_sha256": behaviour.sha256,
                }
            )[:24],
            "weather_member_id": item.member_id,
        }
        for item in weather
    ]
    payload = {
        "contract_version": CONTRACT_VERSION,
        "status": "PREPARED_PRELIMINARY_REPRESENTATIVE_WEATHER_FACTORIAL",
        "scope": (
            "2x2 stock-year/weather factorial using the same 75 physics cells and exact "
            "authenticated 160-seed prefix; one unmorphed PVGIS-2015 chronology and one "
            "PVGIS-2015-based 2050 morph per RCP; within-RCP weather variability excluded"
        ),
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "original_convergence_status": "NOT_CONVERGED_AT_N160",
        "fixed_budget_qualification": "administrative educational/illustrative estimate",
        "occupant_seed_count": len(seeds),
        "occupant_seeds": list(seeds),
        "occupant_seed_bank_sha256": ordered_seed_bank_sha256(seeds),
        "archetype_state_count": len(states),
        "archetype_state_sha256": canonical_sha256(
            {"cells": [
                {"archetype_id": item.archetype_id, "state_id": item.state_id,
                 "sha256": archetype_state_sha256(item)}
                for item in states
            ]}
        ),
        "weather_members": weather_contracts,
        "stock_years": list(TARGET_YEARS),
        "stock_count_columns": {str(key): value for key, value in COUNT_COLUMN.items()},
        "stock_source_path": str(STOCK_SOURCE_PATH.relative_to(PROJECT_ROOT)),
        "stock_source_sha256": str(weights["stock_source_sha256"].iloc[0]),
        "dual_year_stock_weights_sha256": str(weights["dual_year_stock_weights_sha256"].iloc[0]),
        "supervisor_design_path": str(supervisor_contract_path.relative_to(PROJECT_ROOT)),
        "supervisor_design_sha256": _sha256_file(supervisor_contract_path),
        "model_scenario_id": "central",
        "thermal_assumptions_sha256": thermal.sha256,
        "behaviour_assumptions_sha256": behaviour.sha256,
        "partition_specs": partition_specs,
        "dwelling_year_run_count": len(states) * len(weather) * len(seeds),
        "weighted_stock_accumulations_per_run": len(TARGET_YEARS),
        "case_definitions": CASE_LABELS,
        "effect_formulae": EFFECT_FORMULAE,
    }
    payload["design_sha256"] = canonical_sha256(payload)
    return states, weather, seeds, weights, payload


def prepare(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Persist or authenticate the immutable 2x2 execution design."""

    destination = Path(output_dir).resolve()
    _, _, _, _, payload = _design_inputs()
    path = destination / "factorial_design_contract.json"
    if path.exists():
        existing = _read_json(path)
        if existing != payload:
            raise MonteCarloContractError("Output directory belongs to another factorial design.")
    else:
        _atomic_json(payload, path)
    if _read_json(path) != payload:
        raise MonteCarloContractError("Persisted factorial design failed authentication.")
    return payload


def _validate_partition(
    partition_dir: Path,
    *,
    partition_id: str,
    member_id: str,
    design_sha256: str,
    expected_run_count: int,
) -> dict[str, Any]:
    complete = _read_json(partition_dir / "partition_complete.json")
    if (
        complete.get("status") != "PRELIMINARY_REPRESENTATIVE_WEATHER_PARTITION_COMPLETE"
        or complete.get("contract_version") != CONTRACT_VERSION
        or complete.get("design_sha256") != design_sha256
        or complete.get("partition_id") != partition_id
        or complete.get("weather_member_id") != member_id
        or int(complete.get("run_count", -1)) != expected_run_count
    ):
        raise MonteCarloContractError(f"Completed factorial partition {partition_id} changed.")
    artifacts = complete.get("artifacts")
    required = {"run_manifest.csv", "run_diagnostics.csv", "annual_by_seed.csv", "stock_hourly.csv", "stock_summary.csv"}
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise MonteCarloContractError(f"Partition {partition_id} artifact ledger is incomplete.")
    for name, metadata in artifacts.items():
        path = partition_dir / name
        _verify_file(path, str(metadata["sha256"]), label=f"factorial {partition_id}/{name}")
        if sum(1 for _ in path.open("rb")) - 1 != int(metadata["row_count"]):
            raise MonteCarloContractError(f"Factorial {partition_id}/{name} row count changed.")
    return complete


def _run_partition(
    partition_id: str,
    member_id: str,
    states: tuple[ArchetypeStateInput, ...],
    seeds: tuple[int, ...],
    weights: pd.DataFrame,
    output_dir: str,
    design_sha256: str,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    partition_dir = destination / "partitions" / partition_id
    lock = _acquire_streaming_execution_lock(
        partition_dir, purpose=f"current-versus-2050 partition {partition_id}"
    )
    try:
        design = _read_json(destination / "factorial_design_contract.json")
        if design.get("design_sha256") != design_sha256:
            raise MonteCarloContractError("Factorial design changed before execution.")
        member = load_comparison_weather_member(member_id)
        spec = next(
            (item for item in design["partition_specs"] if item["weather_member_id"] == member_id),
            None,
        )
        if spec is None or spec["partition_id"] != partition_id:
            raise MonteCarloContractError("Factorial partition identity changed.")
        scenario = resolve_model_scenario("central")
        manifest = build_balanced_manifest(states, [member], seeds, [scenario])
        expected_run_count = len(states) * len(seeds)
        complete_path = partition_dir / "partition_complete.json"
        if complete_path.exists():
            complete = _validate_partition(
                partition_dir,
                partition_id=partition_id,
                member_id=member_id,
                design_sha256=design_sha256,
                expected_run_count=expected_run_count,
            )
            return {"partition_id": partition_id, "run_count": complete["run_count"], "reused": True}
        accumulator = _DualYearAccumulator(weights, seeds)
        diagnostics_records: list[dict[str, Any]] = []
        completed_seed_count = 0
        active_slot: int | None = None
        progress_path = partition_dir / "progress.json"
        expected_run_ids = set(manifest["run_id"].astype(str))
        expected_run_id_sha256 = canonical_sha256({"run_ids": sorted(expected_run_ids)})
        if progress_path.exists():
            progress = _read_json(progress_path)
            if (
                progress.get("contract_version") != CONTRACT_VERSION
                or progress.get("design_sha256") != design_sha256
                or progress.get("partition_id") != partition_id
                or progress.get("expected_run_id_sha256") != expected_run_id_sha256
            ):
                raise MonteCarloContractError("Factorial checkpoint belongs to another design.")
            completed_seed_count = int(progress["completed_seed_count"])
            if not 1 <= completed_seed_count <= len(seeds):
                raise MonteCarloContractError("Factorial checkpoint seed count is invalid.")
            if progress["completed_occupant_seeds"] != list(seeds[:completed_seed_count]):
                raise MonteCarloContractError("Factorial checkpoint seed prefix changed.")
            active_slot = int(progress["active_slot"])
            arrays_path = partition_dir / f"progress_slot_{active_slot}_arrays.npz"
            diagnostics_path = partition_dir / f"progress_slot_{active_slot}_diagnostics.csv"
            _verify_file(arrays_path, progress["arrays_sha256"], label="factorial checkpoint arrays")
            _verify_file(diagnostics_path, progress["diagnostics_sha256"], label="factorial checkpoint diagnostics")
            restored = pd.read_csv(diagnostics_path, float_precision="round_trip")
            expected_prefix = set(
                manifest.loc[manifest["occupant_seed_rank"] <= completed_seed_count, "run_id"].astype(str)
            )
            if len(restored) != len(states) * completed_seed_count or set(restored["run_id"].astype(str)) != expected_prefix:
                raise MonteCarloContractError("Factorial checkpoint is incomplete.")
            with np.load(arrays_path, allow_pickle=False) as stored:
                if set(stored.files) != {"timestamp_ns", "heating_W", "cooling_W"}:
                    raise MonteCarloContractError("Factorial checkpoint array schema changed.")
                accumulator.restore(
                    restored,
                    timestamp_ns=stored["timestamp_ns"],
                    heating_W=stored["heating_W"],
                    cooling_W=stored["cooling_W"],
                    region_order=progress["region_order"],
                )
            diagnostics_records = restored.to_dict("records")
        expected_by_factor = manifest.set_index(["archetype_id", "state_id", "occupant_seed"])["run_id"].to_dict()
        ordered_states = sorted(states, key=lambda item: (item.archetype_id, item.state_id))
        central_thermal = load_assumption_contract()
        behaviour_contract = load_behaviour_assumptions()
        representative_type = {"SFH": "Detached house", "MFH": "Apartment, enclosed"}
        for seed_rank in range(completed_seed_count + 1, len(seeds) + 1):
            seed = seeds[seed_rank - 1]
            behaviour_by_class: dict[str, Any] = {}
            for state in ordered_states:
                household_class = dwelling_class(state.dwelling_type)
                if household_class not in behaviour_by_class:
                    behaviour_by_class[household_class] = generate_behaviour(
                        BehaviourRequest(
                            dwelling_type=representative_type[household_class],
                            weather=member.frame.copy(deep=True),
                            weather_member_id=member.member_id,
                            seed=seed,
                        ),
                        behaviour_contract,
                    )
                result = _simulate_with_behaviour(
                    state, member, seed, scenario, behaviour_by_class[household_class], central_thermal
                )
                expected_run_id = expected_by_factor[(state.archetype_id, state.state_id, seed)]
                if result.diagnostics.run_id != expected_run_id:
                    raise MonteCarloContractError("Executed factorial run differs from manifest.")
                accumulator.add(result)
                diagnostics_records.append(diagnostics_to_record(result.diagnostics))
            completed = pd.DataFrame.from_records(diagnostics_records)
            next_slot = 0 if active_slot is None else 1 - active_slot
            arrays_path = partition_dir / f"progress_slot_{next_slot}_arrays.npz"
            diagnostics_path = partition_dir / f"progress_slot_{next_slot}_diagnostics.csv"
            _atomic_npz(accumulator.snapshot(), arrays_path)
            _atomic_csv(completed, diagnostics_path)
            progress = {
                "contract_version": CONTRACT_VERSION,
                "design_sha256": design_sha256,
                "partition_id": partition_id,
                "expected_run_id_sha256": expected_run_id_sha256,
                "active_slot": next_slot,
                "completed_seed_count": seed_rank,
                "completed_occupant_seeds": list(seeds[:seed_rank]),
                "completed_run_count": len(completed),
                "region_order": list(accumulator.region_order),
                "arrays_sha256": _sha256_file(arrays_path),
                "diagnostics_sha256": _sha256_file(diagnostics_path),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(progress, progress_path)
            active_slot = next_slot
            if seed_rank == 1 or seed_rank % 10 == 0 or seed_rank == len(seeds):
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] {member_id}: "
                    f"{seed_rank}/{len(seeds)} seeds complete",
                    flush=True,
                )
        diagnostics = pd.DataFrame.from_records(diagnostics_records)
        if len(diagnostics) != expected_run_count or set(diagnostics["run_id"].astype(str)) != expected_run_ids:
            raise MonteCarloContractError("Final factorial diagnostics are incomplete.")
        annual, hourly, summary = accumulator.finalize()
        outputs = {
            "run_manifest.csv": manifest,
            "run_diagnostics.csv": diagnostics,
            "annual_by_seed.csv": annual,
            "stock_hourly.csv": hourly,
            "stock_summary.csv": summary,
        }
        for name, frame in outputs.items():
            _atomic_csv(frame, partition_dir / name)
        artifacts = {
            name: {"sha256": _sha256_file(partition_dir / name), "row_count": len(frame)}
            for name, frame in outputs.items()
        }
        complete = {
            "status": "PRELIMINARY_REPRESENTATIVE_WEATHER_PARTITION_COMPLETE",
            "contract_version": CONTRACT_VERSION,
            "design_sha256": design_sha256,
            "partition_id": partition_id,
            "weather_member_id": member_id,
            "run_count": expected_run_count,
            "occupant_seed_count": len(seeds),
            "occupant_seed_bank_sha256": ordered_seed_bank_sha256(seeds),
            "artifacts": artifacts,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(complete, complete_path)
        _validate_partition(
            partition_dir,
            partition_id=partition_id,
            member_id=member_id,
            design_sha256=design_sha256,
            expected_run_count=expected_run_count,
        )
        return {"partition_id": partition_id, "run_count": expected_run_count, "reused": False}
    finally:
        _release_streaming_execution_lock(lock)


def execute(output_dir: str | Path = DEFAULT_OUTPUT_DIR, *, max_workers: int = 4) -> dict[str, Any]:
    """Run or resume the four weather partitions with one root writer lock."""

    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or not 1 <= max_workers <= 4:
        raise MonteCarloContractError("Factorial max_workers must be an integer from 1 to 4.")
    destination = Path(output_dir).resolve()
    lock = _acquire_streaming_execution_lock(destination, purpose="current-versus-2050 coordinator")
    try:
        states, _, seeds, weights, payload = _design_inputs()
        persisted = prepare(destination)
        if persisted != payload:
            raise MonteCarloContractError("Prepared factorial design changed before execution.")
        arguments = [
            (
                spec["partition_id"], spec["weather_member_id"], states, seeds,
                weights, str(destination), payload["design_sha256"],
            )
            for spec in payload["partition_specs"]
        ]
        results: list[dict[str, Any]] = []
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] advancing 4 factorial weather "
            f"partitions with {max_workers} workers ({payload['dwelling_year_run_count']} runs)",
            flush=True,
        )
        if max_workers == 1:
            results = [_run_partition(*args) for args in arguments]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_partition, *args): args[1] for args in arguments}
                for future in as_completed(futures):
                    member_id = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        for pending in futures:
                            pending.cancel()
                        raise MonteCarloContractError(f"Factorial partition {member_id} failed.") from exc
                    print(
                        f"[{datetime.now(timezone.utc).isoformat()}] completed {len(results)}/4 "
                        f"factorial partitions",
                        flush=True,
                    )
        summary = {
            "status": "PRELIMINARY_REPRESENTATIVE_WEATHER_FACTORIAL_SIMULATION_COMPLETE",
            "contract_version": CONTRACT_VERSION,
            "design_sha256": payload["design_sha256"],
            "partition_count": len(results),
            "dwelling_year_run_count": sum(int(item["run_count"]) for item in results),
            "occupant_seed_count": len(seeds),
            "original_convergence_status": "NOT_CONVERGED_AT_N160",
            "within_rcp_weather_variability_included": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(summary, destination / "factorial_simulation_summary.json")
        return summary
    finally:
        _release_streaming_execution_lock(lock)


def _case_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    design = _read_json(output_dir / "factorial_design_contract.json")
    annual_frames: dict[str, pd.DataFrame] = {}
    hourly_frames: dict[str, pd.DataFrame] = {}
    summary_frames: dict[str, pd.DataFrame] = {}
    for spec in design["partition_specs"]:
        partition = output_dir / "partitions" / spec["partition_id"]
        _validate_partition(
            partition,
            partition_id=spec["partition_id"],
            member_id=spec["weather_member_id"],
            design_sha256=design["design_sha256"],
            expected_run_count=75 * 160,
        )
        annual_frames[spec["weather_member_id"]] = pd.read_csv(partition / "annual_by_seed.csv")
        hourly_frames[spec["weather_member_id"]] = pd.read_csv(partition / "stock_hourly.csv")
        summary_frames[spec["weather_member_id"]] = pd.read_csv(partition / "stock_summary.csv")
    records: list[dict[str, Any]] = []
    peak_records: list[dict[str, Any]] = []
    for scenario in CLIMATE_SCENARIOS:
        future_id = f"weather_2050_{scenario}_pvgis_2015"
        for case_id, member_id, stock_year in (
            ("Q00", OBSERVED_MEMBER_ID, 2025),
            ("Q10", OBSERVED_MEMBER_ID, 2050),
            ("Q01", future_id, 2025),
            ("Q11", future_id, 2050),
        ):
            selected = annual_frames[member_id].loc[
                (annual_frames[member_id]["stock_year"] == stock_year)
                & (annual_frames[member_id]["region"] == BELGIUM_REGION_ID)
            ].copy()
            if len(selected) != 160 or selected["occupant_seed"].duplicated().any():
                raise MonteCarloContractError(f"Annual case {scenario}/{case_id} is incomplete.")
            for row in selected.itertuples(index=False):
                records.append(
                    {
                        "climate_scenario_id": scenario,
                        "case_id": case_id,
                        "case_label": CASE_LABELS[case_id],
                        "weather_member_id": member_id,
                        "stock_year": stock_year,
                        "occupant_seed": int(row.occupant_seed),
                        "occupant_seed_rank": int(row.occupant_seed_rank),
                        "modelled_dwellings": float(row.modelled_dwellings),
                        "stock_floor_area_m2": float(row.stock_floor_area_m2),
                        "annual_heating_TWh": float(row.annual_heating_GWh) / 1000.0,
                        "annual_potential_sensible_cooling_TWh": float(row.annual_potential_sensible_cooling_GWh) / 1000.0,
                        "stock_heating_intensity_kWh_m2": float(row.stock_heating_intensity_kWh_m2),
                        "stock_cooling_intensity_kWh_m2": float(row.stock_cooling_intensity_kWh_m2),
                    }
                )
            selected_summary = summary_frames[member_id].loc[
                (summary_frames[member_id]["stock_year"] == stock_year)
                & (summary_frames[member_id]["region"] == BELGIUM_REGION_ID)
            ]
            if len(selected_summary) != 1:
                raise MonteCarloContractError(f"Peak case {scenario}/{case_id} is incomplete.")
            row = selected_summary.iloc[0]
            peak_records.append(
                {
                    "climate_scenario_id": scenario,
                    "case_id": case_id,
                    "case_label": CASE_LABELS[case_id],
                    "weather_member_id": member_id,
                    "stock_year": stock_year,
                    "coincident_peak_heating_GW": float(row["coincident_peak_heating_MW"]) / 1000.0,
                    "coincident_peak_potential_cooling_GW": float(row["coincident_peak_potential_cooling_MW"]) / 1000.0,
                    "peak_heating_timestamp_utc": row["peak_heating_timestamp_utc"],
                    "peak_potential_cooling_timestamp_utc": row["peak_potential_cooling_timestamp_utc"],
                }
            )
    cases = pd.DataFrame.from_records(records)
    peaks = pd.DataFrame.from_records(peak_records)
    hourly = pd.concat(hourly_frames.values(), ignore_index=True)
    return cases, peaks, hourly


def _effect_table(cases: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "annual_heating_TWh",
        "annual_potential_sensible_cooling_TWh",
        "stock_heating_intensity_kWh_m2",
        "stock_cooling_intensity_kWh_m2",
    )
    records: list[dict[str, Any]] = []
    for (scenario, seed), group in cases.groupby(["climate_scenario_id", "occupant_seed"], sort=False):
        indexed = group.set_index("case_id")
        if set(indexed.index) != set(CASE_ORDER):
            raise MonteCarloContractError("Paired annual case group is incomplete.")
        for metric in metrics:
            q = {case: float(indexed.loc[case, metric]) for case in CASE_ORDER}
            effects = {
                "renovation": q["Q10"] - q["Q00"],
                "climate": q["Q01"] - q["Q00"],
                "interaction": q["Q11"] - q["Q10"] - q["Q01"] + q["Q00"],
                "combined": q["Q11"] - q["Q00"],
            }
            if not np.isclose(
                effects["combined"],
                effects["renovation"] + effects["climate"] + effects["interaction"],
                rtol=0.0,
                atol=1e-10,
            ):
                raise MonteCarloContractError("2x2 effect identity does not close.")
            for effect_id, value in effects.items():
                records.append(
                    {
                        "climate_scenario_id": scenario,
                        "occupant_seed": int(seed),
                        "occupant_seed_rank": int(indexed["occupant_seed_rank"].iloc[0]),
                        "metric": metric,
                        "effect_id": effect_id,
                        "formula": EFFECT_FORMULAE[effect_id],
                        "effect_value": value,
                        "effect_percent_of_Q00": 100.0 * value / q["Q00"] if q["Q00"] else np.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def _peak_effect_table(peaks: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    metrics = ("coincident_peak_heating_GW", "coincident_peak_potential_cooling_GW")
    for scenario, group in peaks.groupby("climate_scenario_id", sort=False):
        indexed = group.set_index("case_id")
        for metric in metrics:
            q = {case: float(indexed.loc[case, metric]) for case in CASE_ORDER}
            effects = {
                "renovation": q["Q10"] - q["Q00"],
                "climate": q["Q01"] - q["Q00"],
                "interaction": q["Q11"] - q["Q10"] - q["Q01"] + q["Q00"],
                "combined": q["Q11"] - q["Q00"],
            }
            for effect_id, value in effects.items():
                records.append(
                    {
                        "climate_scenario_id": scenario,
                        "metric": metric,
                        "effect_id": effect_id,
                        "formula": EFFECT_FORMULAE[effect_id],
                        "effect_value": value,
                        "effect_percent_of_Q00": 100.0 * value / q["Q00"] if q["Q00"] else np.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def _distribution_summary(frame: pd.DataFrame, value_columns: Sequence[str], group_columns: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_columns), sort=False, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_columns, keys))
        for metric in value_columns:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(float)
            records.append(
                {
                    **base,
                    ("summary_metric" if "metric" in base else "metric"): metric,
                    "occupant_seed_count": len(values),
                    "p05": float(np.quantile(values, 0.05)),
                    "median": float(np.quantile(values, 0.50)),
                    "mean": float(np.mean(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                }
            )
    return pd.DataFrame.from_records(records)


def _crosscheck_existing(cases: pd.DataFrame, peaks: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    existing = pd.read_csv(SUPERVISOR_RESULTS_DIR / "supervisor_stock_annual_by_seed.csv")
    existing = existing.loc[existing["region"] == BELGIUM_REGION_ID].copy()
    q11 = cases.loc[cases["case_id"] == "Q11"].copy()
    joined = q11.merge(
        existing,
        on=["climate_scenario_id", "occupant_seed"],
        how="inner",
        validate="one_to_one",
        suffixes=("_rerun", "_existing"),
    )
    if len(joined) != 3 * 160:
        raise MonteCarloContractError("Existing Q11 annual cross-check is incomplete.")
    annual_heat_error_GWh = np.max(
        np.abs(joined["annual_heating_TWh"] * 1000.0 - joined["annual_heating_GWh"])
    )
    annual_cool_error_GWh = np.max(
        np.abs(
            joined["annual_potential_sensible_cooling_TWh"] * 1000.0
            - joined["annual_potential_sensible_cooling_GWh"]
        )
    )
    index = pd.read_csv(SUPERVISOR_RESULTS_DIR / "partition_index.csv")
    peak_errors: list[float] = []
    hourly_errors: list[float] = []
    for row in index.itertuples(index=False):
        existing_hourly = pd.read_csv(SUPERVISOR_RESULTS_DIR / row.stock_hourly_path)
        existing_hourly = existing_hourly.loc[existing_hourly["region"] == BELGIUM_REGION_ID]
        design = _read_json(output_dir / "factorial_design_contract.json")
        spec = next(item for item in design["partition_specs"] if item["weather_member_id"] == row.weather_member_id)
        rerun_hourly = pd.read_csv(output_dir / "partitions" / spec["partition_id"] / "stock_hourly.csv")
        rerun_hourly = rerun_hourly.loc[
            (rerun_hourly["region"] == BELGIUM_REGION_ID) & (rerun_hourly["stock_year"] == 2050)
        ]
        if len(existing_hourly) != len(rerun_hourly):
            raise MonteCarloContractError("Existing Q11 hourly cross-check length changed.")
        for old_col, new_col in (
            ("heating_demand_MW", "heating_demand_MW"),
            ("potential_sensible_cooling_demand_MW", "potential_sensible_cooling_demand_MW"),
        ):
            hourly_errors.append(
                float(np.max(np.abs(existing_hourly[old_col].to_numpy(float) - rerun_hourly[new_col].to_numpy(float))))
            )
        existing_peak = peaks.loc[
            (peaks["climate_scenario_id"] == row.climate_scenario_id) & (peaks["case_id"] == "Q11")
        ].iloc[0]
        peak_errors.extend(
            [
                abs(existing_hourly["heating_demand_MW"].max() / 1000.0 - existing_peak["coincident_peak_heating_GW"]),
                abs(existing_hourly["potential_sensible_cooling_demand_MW"].max() / 1000.0 - existing_peak["coincident_peak_potential_cooling_GW"]),
            ]
        )
    checks = {
        "q11_annual_max_abs_heating_error_GWh": float(annual_heat_error_GWh),
        "q11_annual_max_abs_cooling_error_GWh": float(annual_cool_error_GWh),
        "q11_hourly_max_abs_error_MW": float(max(hourly_errors)),
        "q11_peak_max_abs_error_GW": float(max(peak_errors)),
    }
    if checks["q11_annual_max_abs_heating_error_GWh"] > 1e-5 or checks["q11_annual_max_abs_cooling_error_GWh"] > 1e-5:
        raise MonteCarloContractError("Q11 annual rerun does not reproduce supervisor results.")
    if checks["q11_hourly_max_abs_error_MW"] > 1e-6 or checks["q11_peak_max_abs_error_GW"] > 1e-9:
        raise MonteCarloContractError("Q11 hourly rerun does not reproduce supervisor results.")
    return checks


def postprocess(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Authenticate all partitions and calculate cases, effects, and checks."""

    destination = Path(output_dir).resolve()
    simulation = _read_json(destination / "factorial_simulation_summary.json")
    if simulation.get("status") != "PRELIMINARY_REPRESENTATIVE_WEATHER_FACTORIAL_SIMULATION_COMPLETE":
        raise MonteCarloContractError("Factorial simulations are not complete.")
    cases, peaks, hourly = _case_tables(destination)
    effects = _effect_table(cases)
    peak_effects = _peak_effect_table(peaks)
    case_summary = _distribution_summary(
        cases,
        (
            "annual_heating_TWh",
            "annual_potential_sensible_cooling_TWh",
            "stock_heating_intensity_kWh_m2",
            "stock_cooling_intensity_kWh_m2",
        ),
        ("climate_scenario_id", "case_id", "case_label"),
    )
    effect_summary = _distribution_summary(
        effects,
        ("effect_value", "effect_percent_of_Q00"),
        ("climate_scenario_id", "metric", "effect_id", "formula"),
    )
    q00 = cases.loc[cases["case_id"] == "Q00"]
    for column in (
        "annual_heating_TWh",
        "annual_potential_sensible_cooling_TWh",
        "stock_heating_intensity_kWh_m2",
        "stock_cooling_intensity_kWh_m2",
    ):
        pivot = q00.pivot(index="occupant_seed", columns="climate_scenario_id", values=column)
        if not np.allclose(pivot.to_numpy(float), pivot.iloc[:, [0]].to_numpy(float), rtol=0.0, atol=1e-12):
            raise MonteCarloContractError("Q00 copies differ across RCP labels.")
    checks = _crosscheck_existing(cases, peaks, destination)
    outputs = {
        "factorial_annual_cases_by_seed.csv": cases,
        "factorial_annual_case_summary.csv": case_summary,
        "factorial_annual_effects_by_seed.csv": effects,
        "factorial_annual_effect_summary.csv": effect_summary,
        "factorial_coincident_peak_cases.csv": peaks,
        "factorial_coincident_peak_effects.csv": peak_effects,
        "factorial_stock_hourly_all_weather_and_weights.csv": hourly,
    }
    for name, frame in outputs.items():
        _atomic_csv(frame, destination / name)
    ledger = {
        name: {"sha256": _sha256_file(destination / name), "row_count": len(frame)}
        for name, frame in outputs.items()
    }
    summary = {
        "status": "PRELIMINARY_REPRESENTATIVE_WEATHER_FACTORIAL_POSTPROCESS_COMPLETE",
        "contract_version": CONTRACT_VERSION,
        "design_sha256": simulation["design_sha256"],
        "original_convergence_status": "NOT_CONVERGED_AT_N160",
        "within_rcp_weather_variability_included": False,
        "paired_seed_effect_identity_verified": True,
        "q00_rcp_copy_identity_verified": True,
        "existing_q11_reproduction_checks": checks,
        "artifacts": ledger,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["postprocess_sha256"] = canonical_sha256(summary)
    _atomic_json(summary, destination / "factorial_postprocess_summary.json")
    return summary


def _configure_plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _save_figure(figure: Any, basename: str, figure_dir: Path) -> list[Path]:
    paths = [figure_dir / f"{basename}.png", figure_dir / f"{basename}.pdf"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(paths[0], bbox_inches="tight", facecolor="white")
    figure.savefig(paths[1], bbox_inches="tight", facecolor="white", metadata={"Creator": "thermal_model.monte_carlo.current_vs_2050"})
    return paths


def _plot_case_metric(summary: pd.DataFrame, metric: str, ylabel: str, title: str, basename: str, figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    colors = {"Q00": "#4C78A8", "Q10": "#59A14F", "Q01": "#F28E2B", "Q11": "#E15759"}
    scenarios = list(CLIMATE_SCENARIOS)
    x = np.arange(len(scenarios), dtype=float)
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for index, case_id in enumerate(CASE_ORDER):
        selected = summary.loc[(summary["metric"] == metric) & (summary["case_id"] == case_id)].set_index("climate_scenario_id").loc[scenarios]
        median = selected["median"].to_numpy(float)
        lower = median - selected["p05"].to_numpy(float)
        upper = selected["p95"].to_numpy(float) - median
        ax.bar(x + (index - 1.5) * width, median, width, color=colors[case_id], label=f"{case_id}: {CASE_LABELS[case_id]}")
        ax.errorbar(x + (index - 1.5) * width, median, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#333333", capsize=2, linewidth=0.8)
    ax.set_xticks(x, [value.replace("rcp_", "RCP").replace("_", ".") for value in scenarios])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(title, y=0.98)
    fig.legend(
        handles,
        labels,
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
    )
    fig.subplots_adjust(top=0.77, bottom=0.20, left=0.12, right=0.98)
    ax.text(0.0, -0.19, "Bars: median across 160 matched occupant seeds; whiskers: p05-p95. One PVGIS-2015 chronology per RCP.", transform=ax.transAxes, fontsize=8)
    paths = _save_figure(fig, basename, figure_dir)
    plt.close(fig)
    return paths


def _plot_effect_metric(summary: pd.DataFrame, metric: str, ylabel: str, basename: str, figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    colors = {
        "renovation": "#449477",
        "climate": "#E28B2D",
        "interaction": "#9B7AA0",
        "combined": "#2879B9",
    }
    scenarios = list(CLIMATE_SCENARIOS)
    x = np.arange(len(scenarios), dtype=float)
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for index, effect_id in enumerate(EFFECT_ORDER):
        selected = summary.loc[
            (summary["metric"] == metric)
            & (summary["effect_id"] == effect_id)
            & (summary["summary_metric"] == "effect_value")
        ]
        selected = selected.set_index("climate_scenario_id").loc[scenarios]
        median = selected["median"].to_numpy(float)
        lower = median - selected["p05"].to_numpy(float)
        upper = selected["p95"].to_numpy(float) - median
        ax.bar(x + (index - 1.5) * width, median, width, color=colors[effect_id], label=f"{effect_id}: {EFFECT_FORMULAE[effect_id]}")
        ax.errorbar(x + (index - 1.5) * width, median, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#333333", capsize=2, linewidth=0.8)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, [value.replace("rcp_", "RCP").replace("_", ".") for value in scenarios])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=8.5,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.12, right=0.98)
    paths = _save_figure(fig, basename, figure_dir)
    plt.close(fig)
    return paths


def _plot_peak_metric(peaks: pd.DataFrame, metric: str, ylabel: str, title: str, basename: str, figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    colors = {"Q00": "#4C78A8", "Q10": "#59A14F", "Q01": "#F28E2B", "Q11": "#E15759"}
    scenarios = list(CLIMATE_SCENARIOS)
    x = np.arange(len(scenarios), dtype=float)
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for index, case_id in enumerate(CASE_ORDER):
        selected = peaks.loc[peaks["case_id"] == case_id].set_index("climate_scenario_id").loc[scenarios]
        ax.bar(x + (index - 1.5) * width, selected[metric].to_numpy(float), width, color=colors[case_id], label=f"{case_id}: {CASE_LABELS[case_id]}")
    ax.set_xticks(x, [value.replace("rcp_", "RCP").replace("_", ".") for value in scenarios])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(title, y=0.98)
    fig.legend(
        handles,
        labels,
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
    )
    fig.subplots_adjust(top=0.77, bottom=0.20, left=0.12, right=0.98)
    ax.text(0.0, -0.19, "Coincident peak of the stock-level hourly profile averaged over 160 matched seeds; not the sum of dwelling peaks.", transform=ax.transAxes, fontsize=8)
    paths = _save_figure(fig, basename, figure_dir)
    plt.close(fig)
    return paths


def report(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> dict[str, Any]:
    """Generate separate publication figures and a concise interpretation."""

    destination = Path(output_dir).resolve()
    figures = Path(figure_dir).resolve()
    post = _read_json(destination / "factorial_postprocess_summary.json")
    unsigned = {key: value for key, value in post.items() if key != "postprocess_sha256"}
    if canonical_sha256(unsigned) != post.get("postprocess_sha256"):
        raise MonteCarloContractError("Factorial postprocess summary checksum is invalid.")
    cases_summary = pd.read_csv(destination / "factorial_annual_case_summary.csv")
    effects_summary = pd.read_csv(destination / "factorial_annual_effect_summary.csv")
    if not {"metric", "summary_metric"}.issubset(effects_summary.columns):
        raise MonteCarloContractError("Effect summary metric columns are incomplete.")
    peaks = pd.read_csv(destination / "factorial_coincident_peak_cases.csv")
    peak_effects = pd.read_csv(destination / "factorial_coincident_peak_effects.csv")
    _configure_plots()
    created: list[Path] = []
    created += _plot_case_metric(cases_summary, "annual_heating_TWh", "Useful heating [TWh/year]", "Paired stock-climate cases: useful heating", "factorial_heating_energy", figures)
    created += _plot_case_metric(cases_summary, "annual_potential_sensible_cooling_TWh", "Potential sensible cooling [TWh/year]", "Paired stock-climate cases: potential cooling", "factorial_cooling_energy", figures)
    created += _plot_effect_metric(effects_summary, "annual_heating_TWh", "Change from Q00 [TWh/year]", "factorial_heating_effects", figures)
    created += _plot_effect_metric(effects_summary, "annual_potential_sensible_cooling_TWh", "Change from Q00 [TWh/year]", "factorial_cooling_effects", figures)
    created += _plot_peak_metric(peaks, "coincident_peak_heating_GW", "Useful heating peak [GWth]", "Coincident stock heating peak", "factorial_heating_peak", figures)
    created += _plot_peak_metric(peaks, "coincident_peak_potential_cooling_GW", "Potential sensible cooling peak [GWth]", "Coincident stock cooling peak", "factorial_cooling_peak", figures)
    case_summary = cases_summary.set_index(["climate_scenario_id", "case_id", "metric"])
    effect_summary = effects_summary.set_index(["climate_scenario_id", "metric", "effect_id", "summary_metric"])
    lines = [
        "# Current-versus-2050 2×2 counterfactual results",
        "",
        "**Status:** PRELIMINARY_REPRESENTATIVE_WEATHER_FACTORIAL_REPORT_COMPLETE.",
        "",
        "This paired factorial separates the change in Belgian residential useful thermal demand into a 2025-to-2050 renovation-stock effect, a 2015-reference-to-2050 climate effect, their interaction, and their combined effect. Every case uses the same 75 physical archetype states, central assumptions, and exact 160 occupant seeds. Q00 and Q10 use the unmorphed PVGIS-2015 reference chronology; Q01 and Q11 use its 2041–2060 morph under the stated RCP.",
        "",
        "The n=160 choice remains an administrative fixed computational budget. The original convergence status remains **NOT_CONVERGED_AT_N160**. Results condition on one representative chronology per RCP and therefore exclude within-RCP weather-year variability.",
        "",
        "## Four cases",
        "",
        "| RCP | Case | Heating median [p05, p95] TWh | Cooling median [p05, p95] TWh | Heating peak GWth | Cooling peak GWth |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario in CLIMATE_SCENARIOS:
        for case_id in CASE_ORDER:
            heat = case_summary.loc[(scenario, case_id, "annual_heating_TWh")]
            cool = case_summary.loc[(scenario, case_id, "annual_potential_sensible_cooling_TWh")]
            peak = peaks.loc[(peaks["climate_scenario_id"] == scenario) & (peaks["case_id"] == case_id)].iloc[0]
            lines.append(
                f"| {scenario.replace('rcp_', 'RCP').replace('_', '.')} | {case_id} | "
                f"{heat['median']:.2f} [{heat['p05']:.2f}, {heat['p95']:.2f}] | "
                f"{cool['median']:.2f} [{cool['p05']:.2f}, {cool['p95']:.2f}] | "
                f"{peak['coincident_peak_heating_GW']:.2f} | "
                f"{peak['coincident_peak_potential_cooling_GW']:.2f} |"
            )
    lines += [
        "",
        "## Paired effect decomposition",
        "",
        "For annual energy, renovation = Q10−Q00, climate = Q01−Q00, interaction = Q11−Q10−Q01+Q00, and combined = Q11−Q00. The identity combined = renovation + climate + interaction is verified for every seed before aggregation.",
        "",
        "| RCP | End use | Renovation | Climate | Interaction | Combined | Combined vs Q00 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in CLIMATE_SCENARIOS:
        for metric, label in (
            ("annual_heating_TWh", "Heating [TWh/year]"),
            ("annual_potential_sensible_cooling_TWh", "Cooling [TWh/year]"),
        ):
            values = {
                effect: effect_summary.loc[(scenario, metric, effect, "effect_value")]["median"]
                for effect in EFFECT_ORDER
            }
            percent = effect_summary.loc[(scenario, metric, "combined", "effect_percent_of_Q00")]["median"]
            lines.append(
                f"| {scenario.replace('rcp_', 'RCP').replace('_', '.')} | {label} | "
                f"{values['renovation']:+.2f} | {values['climate']:+.2f} | "
                f"{values['interaction']:+.2f} | {values['combined']:+.2f} | {percent:+.1f}% |"
            )
    lines += [
        "",
        "## Coincident peak effect decomposition",
        "",
        "Peak effects use the same four-case algebra, applied to the coincident peak of each stock-level mean hourly profile. They are not the sum of individual dwelling peaks.",
        "",
        "| RCP | End use | Renovation | Climate | Interaction | Combined | Combined vs Q00 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in CLIMATE_SCENARIOS:
        for metric, label in (
            ("coincident_peak_heating_GW", "Heating [GWth]"),
            ("coincident_peak_potential_cooling_GW", "Cooling [GWth]"),
        ):
            selected = peak_effects.loc[
                (peak_effects["climate_scenario_id"] == scenario)
                & (peak_effects["metric"] == metric)
            ].set_index("effect_id")
            values = {effect: float(selected.loc[effect, "effect_value"]) for effect in EFFECT_ORDER}
            percent = float(selected.loc["combined", "effect_percent_of_Q00"])
            lines.append(
                f"| {scenario.replace('rcp_', 'RCP').replace('_', '.')} | {label} | "
                f"{values['renovation']:+.2f} | {values['climate']:+.2f} | "
                f"{values['interaction']:+.2f} | {values['combined']:+.2f} | {percent:+.1f}% |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    for scenario in CLIMATE_SCENARIOS:
        heating_combined = effect_summary.loc[(scenario, "annual_heating_TWh", "combined", "effect_percent_of_Q00")]["median"]
        heating_renovation = effect_summary.loc[(scenario, "annual_heating_TWh", "renovation", "effect_percent_of_Q00")]["median"]
        heating_climate = effect_summary.loc[(scenario, "annual_heating_TWh", "climate", "effect_percent_of_Q00")]["median"]
        cooling_combined = effect_summary.loc[(scenario, "annual_potential_sensible_cooling_TWh", "combined", "effect_percent_of_Q00")]["median"]
        lines.append(
            f"- {scenario.replace('rcp_', 'RCP').replace('_', '.')}: the 2050 stock projection changes median heating by {heating_renovation:+.1f}% under observed-reference weather; climate alone changes it by {heating_climate:+.1f}%; together they change heating by {heating_combined:+.1f}%. Potential sensible cooling changes by {cooling_combined:+.1f}% in the combined case."
        )
    lines += [
        "",
        "The renovation and climate effects are not assumed additive. The positive heating interaction means that part of the two savings mechanisms overlaps: once renovation has removed envelope losses, less heating demand remains for warmer weather to remove. Adding the renovation and climate main effects without the interaction would therefore overstate the combined heating reduction.",
        "",
        "Annual potential cooling increases in every combined case, while the combined coincident cooling peak is lower than Q00. This is physically possible: the more insulated 2050 stock retains gains over more hours, but its improved envelope also limits heat transfer during the single most severe outdoor condition. The result concerns a stock-wide coincident peak, not every dwelling's individual design load.",
        "",
        "The RCP cooling results are not monotonic in the pathway label for this one selected chronology. Cooling responds to the monthly temperature deltas, irradiance scaling and their seasonal timing; an RCP name alone does not impose a monotonic value on every monthly forcing in one downscaled model chain. The planned all-54-member screen is needed before interpreting differences between RCPs as robust pathway ordering.",
        "",
        "Q00 combines 2025 stock weights with the observed PVGIS-2015 reference chronology. It is a controlled current-stock reference counterfactual, not a reconstruction of actual meteorological conditions in 2025.",
        "",
        "The cooling values are potential useful sensible loads under universal ideal 26°C control, not a forecast of actual Belgian air-conditioner electricity. Heating and cooling are useful thermal demand, not delivered fuel or heat-pump electricity. Capacity planning would additionally require technology efficiencies, adoption, sizing margins, networks, and supply-side coincidence.",
        "",
        "## Verification",
        "",
        f"The rerun reproduces the existing Q11 annual heating totals within {post['existing_q11_reproduction_checks']['q11_annual_max_abs_heating_error_GWh']:.3g} GWh and cooling within {post['existing_q11_reproduction_checks']['q11_annual_max_abs_cooling_error_GWh']:.3g} GWh. The maximum hourly discrepancy is {post['existing_q11_reproduction_checks']['q11_hourly_max_abs_error_MW']:.3g} MW. Q00 is identical across the three RCP labels, and the four-term algebra closes for every seed.",
        "",
        "## Figure files",
        "",
    ]
    for path in created:
        if path.suffix == ".png":
            lines.append(f"- `{path.relative_to(PROJECT_ROOT)}`")
    report_path = destination / "CURRENT_VS_2050_RESULTS.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {str(path.relative_to(PROJECT_ROOT)): _sha256_file(path) for path in created}
    artifacts[str(report_path.relative_to(PROJECT_ROOT))] = _sha256_file(report_path)
    summary = {
        "status": "PRELIMINARY_REPRESENTATIVE_WEATHER_FACTORIAL_REPORT_COMPLETE",
        "contract_version": CONTRACT_VERSION,
        "design_sha256": post["design_sha256"],
        "original_convergence_status": "NOT_CONVERGED_AT_N160",
        "within_rcp_weather_variability_included": False,
        "figure_count": len(created) // 2,
        "artifacts": artifacts,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["report_sha256"] = canonical_sha256(summary)
    _atomic_json(summary, destination / "factorial_reporting_summary.json")
    return summary


def audit(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Read-only authentication of the full factorial result chain."""

    destination = Path(output_dir).resolve()
    try:
        design = _read_json(destination / "factorial_design_contract.json")
        unsigned_design = {key: value for key, value in design.items() if key != "design_sha256"}
        if canonical_sha256(unsigned_design) != design.get("design_sha256"):
            raise MonteCarloContractError("Factorial design checksum is invalid.")
        simulation = _read_json(destination / "factorial_simulation_summary.json")
        post = _read_json(destination / "factorial_postprocess_summary.json")
        report_summary = _read_json(destination / "factorial_reporting_summary.json")
        for payload, hash_name in ((post, "postprocess_sha256"), (report_summary, "report_sha256")):
            unsigned = {key: value for key, value in payload.items() if key != hash_name}
            if canonical_sha256(unsigned) != payload.get(hash_name):
                raise MonteCarloContractError(f"Factorial {hash_name} is invalid.")
        if simulation.get("design_sha256") != design["design_sha256"] or post.get("design_sha256") != design["design_sha256"] or report_summary.get("design_sha256") != design["design_sha256"]:
            raise MonteCarloContractError("Factorial artifact chain mixes design identities.")
        for name, metadata in post["artifacts"].items():
            _verify_file(destination / name, metadata["sha256"], label=f"factorial postprocess {name}")
        for relative, digest in report_summary["artifacts"].items():
            _verify_file(PROJECT_ROOT / relative, digest, label=f"factorial report {relative}")
        return {
            "status": "PRELIMINARY_REPRESENTATIVE_WEATHER_FACTORIAL_AUDIT_COMPLETE",
            "design_sha256": design["design_sha256"],
            "dwelling_year_run_count": simulation["dwelling_year_run_count"],
            "figure_count": report_summary["figure_count"],
            "original_convergence_status": "NOT_CONVERGED_AT_N160",
            "within_rcp_weather_variability_included": False,
            "q11_reproduction_checks": post["existing_q11_reproduction_checks"],
        }
    except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        if isinstance(exc, MonteCarloContractError):
            raise
        raise MonteCarloContractError("Factorial audit failed on malformed artifacts.") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "postprocess", "report", "audit", "all"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output_dir)
    elif args.command == "run":
        result = execute(args.output_dir, max_workers=args.max_workers)
    elif args.command == "postprocess":
        result = postprocess(args.output_dir)
    elif args.command == "report":
        result = report(args.output_dir, args.figure_dir)
    elif args.command == "audit":
        result = audit(args.output_dir)
    else:
        prepare(args.output_dir)
        execute(args.output_dir, max_workers=args.max_workers)
        postprocess(args.output_dir)
        report(args.output_dir, args.figure_dir)
        result = audit(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
