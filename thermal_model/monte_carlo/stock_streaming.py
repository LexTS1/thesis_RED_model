"""Bounded-memory stock accumulation for production Monte Carlo execution."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .aggregation import (
    STOCK_PARTITION_IDENTITY_FIELDS,
    STOCK_PROVENANCE_COLUMNS,
    _seed_provenance,
    _stock_partition_provenance_record,
    _validated_stock_input,
)
from .contracts import (
    MonteCarloContractError,
    MonteCarloResult,
    diagnostics_to_record,
)


STOCK_GROUP_AXES = STOCK_PARTITION_IDENTITY_FIELDS
BELGIUM_REGION_ID = "Belgium_modelled_stock"
CONTRIBUTION_DIMENSIONS = (
    "region",
    "dwelling_type",
    "construction_period",
    "state_id",
)


class StreamingStockAccumulator:
    """Consume one dwelling-year at a time and retain only weighted stock sums.

    One instance represents exactly one weather member and one model scenario.
    Hourly-array memory is proportional to ``regions x hours``, independent of
    the dwelling-year count.  A compact run-ID/completeness ledger grows with
    the number of runs, but no run-level hourly table is retained.
    """

    def __init__(
        self,
        stock_weights: pd.DataFrame | None,
        occupant_seeds: Sequence[int],
        *,
        require_full_stock: bool = True,
    ) -> None:
        self.weights = _validated_stock_input(
            stock_weights, require_full_stock=require_full_stock
        )
        self.require_full_stock = bool(require_full_stock)
        raw_seeds = tuple(occupant_seeds)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in raw_seeds
        ):
            raise MonteCarloContractError("Occupant seeds must be integers, not coerced values.")
        self.occupant_seeds = tuple(int(value) for value in raw_seeds)
        if any(value < 0 or value > 2**32 - 1 for value in self.occupant_seeds):
            raise MonteCarloContractError("Occupant seeds must be uint32-compatible.")
        self.seed_provenance = _seed_provenance(self.occupant_seeds)
        self._seed_set = set(self.occupant_seeds)
        self.physics_keys = ("archetype_id", "state_id")
        self.expected_cells = set(
            map(
                tuple,
                self.weights[list(self.physics_keys)].drop_duplicates().to_numpy(),
            )
        )
        self.region_names = tuple(sorted(self.weights["region"].astype(str).unique()))
        self.region_order = (*self.region_names, BELGIUM_REGION_ID)
        self._weights_by_cell = {
            tuple(key): group.copy(deep=True)
            for key, group in self.weights.groupby(
                list(self.physics_keys), sort=False, dropna=False
            )
        }
        self._identity: dict[str, object] | None = None
        self._provenance: dict[str, object] | None = None
        self._timestamps: pd.DatetimeIndex | None = None
        self._heating_W: np.ndarray | None = None
        self._cooling_W: np.ndarray | None = None
        self._seen_run_ids: set[str] = set()
        self._seen_cell_seeds: set[tuple[str, str, int]] = set()
        self._cell_floor_area: dict[tuple[str, str], float] = {}
        self._cell_physics_sha256: dict[tuple[str, str], str] = {}
        self._annual_heating_kWh = defaultdict(float)
        self._annual_cooling_kWh = defaultdict(float)
        self._sum_peak_heating_W = defaultdict(float)
        self._sum_peak_cooling_W = defaultdict(float)
        self._stock_floor_area_m2 = defaultdict(float)
        self._modelled_dwellings = defaultdict(float)
        self._contributions: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {
                "modelled_dwellings": 0.0,
                "stock_floor_area_m2": 0.0,
                "annual_heating_kWh": 0.0,
                "annual_cooling_kWh": 0.0,
            }
        )

    @property
    def completed_run_count(self) -> int:
        return len(self._seen_run_ids)

    def _register_identity(self, record: Mapping[str, object]) -> None:
        identity = {column: record[column] for column in STOCK_GROUP_AXES}
        provenance = {column: str(record[column]) for column in STOCK_PROVENANCE_COLUMNS}
        if self._identity is None:
            self._identity = identity
            self._provenance = provenance
            return
        if identity != self._identity:
            raise MonteCarloContractError(
                "Streaming stock accumulator received mixed weather/scenario identities."
            )
        assert self._provenance is not None
        for column, value in provenance.items():
            if value != self._provenance[column]:
                raise MonteCarloContractError(
                    f"Streaming stock accumulator mixes incompatible {column} values."
                )

    def _register_record(self, record: Mapping[str, object]) -> None:
        missing = sorted(
            {
                "run_id",
                "archetype_id",
                "state_id",
                "occupant_seed",
                "floor_area_m2",
                "annual_heating_kWh",
                "annual_cooling_kWh",
                "peak_heating_W",
                "peak_cooling_W",
                "archetype_state_sha256",
                *STOCK_GROUP_AXES,
                *STOCK_PROVENANCE_COLUMNS,
            }.difference(record)
        )
        if missing:
            raise MonteCarloContractError(
                f"Streaming stock diagnostic is missing columns: {missing}."
            )
        self._register_identity(record)
        run_id = str(record["run_id"])
        if run_id in self._seen_run_ids:
            raise MonteCarloContractError(f"Duplicate streamed run ID {run_id}.")
        cell = (str(record["archetype_id"]), str(record["state_id"]))
        if cell not in self.expected_cells:
            raise MonteCarloContractError(
                f"Streamed physical cell {cell} is absent from the stock weights."
            )
        seed = int(record["occupant_seed"])
        if seed not in self._seed_set:
            raise MonteCarloContractError(
                f"Streamed occupant seed {seed} is absent from the declared seed bank."
            )
        cell_seed = (*cell, seed)
        if cell_seed in self._seen_cell_seeds:
            raise MonteCarloContractError(
                f"Duplicate streamed physical-cell/seed run {cell_seed}."
            )
        metrics = np.asarray(
            [
                record["floor_area_m2"],
                record["annual_heating_kWh"],
                record["annual_cooling_kWh"],
                record["peak_heating_W"],
                record["peak_cooling_W"],
            ],
            dtype=float,
        )
        if not np.isfinite(metrics).all() or metrics[0] <= 0.0 or (metrics[1:] < 0.0).any():
            raise MonteCarloContractError(
                f"Streamed run {run_id} contains invalid stock metrics."
            )
        floor_area, annual_heat, annual_cool, peak_heat, peak_cool = metrics
        prior_area = self._cell_floor_area.setdefault(cell, float(floor_area))
        if not np.isclose(prior_area, floor_area, rtol=0.0, atol=1.0e-10):
            raise MonteCarloContractError(f"Floor area differs within stock cell {cell}.")
        physics_sha = str(record["archetype_state_sha256"])
        prior_physics_sha = self._cell_physics_sha256.setdefault(cell, physics_sha)
        if physics_sha != prior_physics_sha:
            raise MonteCarloContractError(
                f"Stock cell {cell} mixes incompatible archetype physics checksums."
            )

        seed_scale = 1.0 / len(self.occupant_seeds)
        cell_weights = self._weights_by_cell[cell]
        for weight in cell_weights.itertuples(index=False):
            region = str(weight.region)
            count = float(weight.state_dwellings_2050)
            scale = count * seed_scale
            for target_region in (region, BELGIUM_REGION_ID):
                self._annual_heating_kWh[target_region] += scale * annual_heat
                self._annual_cooling_kWh[target_region] += scale * annual_cool
                self._sum_peak_heating_W[target_region] += scale * peak_heat
                self._sum_peak_cooling_W[target_region] += scale * peak_cool
                self._stock_floor_area_m2[target_region] += scale * floor_area
                self._modelled_dwellings[target_region] += scale
            for dimension in CONTRIBUTION_DIMENSIONS:
                key = (dimension, str(getattr(weight, dimension)))
                contribution = self._contributions[key]
                contribution["modelled_dwellings"] += scale
                contribution["stock_floor_area_m2"] += scale * floor_area
                contribution["annual_heating_kWh"] += scale * annual_heat
                contribution["annual_cooling_kWh"] += scale * annual_cool
        self._seen_run_ids.add(run_id)
        self._seen_cell_seeds.add(cell_seed)

    def add(self, result: MonteCarloResult) -> None:
        """Consume one complete result and immediately discard dwelling hourly data."""

        record = diagnostics_to_record(result.diagnostics)
        hourly = result.hourly
        required = {"timestamp_utc", "heating_demand_W", "cooling_demand_W"}
        missing = sorted(required.difference(hourly.columns))
        if missing:
            raise MonteCarloContractError(
                f"Streamed hourly result is missing columns: {missing}."
            )
        timestamps = pd.DatetimeIndex(hourly["timestamp_utc"])
        heat = pd.to_numeric(hourly["heating_demand_W"], errors="raise").to_numpy(
            dtype=float
        )
        cool = pd.to_numeric(hourly["cooling_demand_W"], errors="raise").to_numpy(
            dtype=float
        )
        if (
            len(timestamps) == 0
            or len(heat) != len(timestamps)
            or len(cool) != len(timestamps)
            or not np.isfinite(heat).all()
            or not np.isfinite(cool).all()
            or (heat < 0.0).any()
            or (cool < 0.0).any()
        ):
            raise MonteCarloContractError(
                f"Streamed run {record['run_id']} contains invalid hourly loads."
            )
        expected = {
            "annual_heating_kWh": float(heat.sum()) / 1000.0,
            "annual_cooling_kWh": float(cool.sum()) / 1000.0,
            "peak_heating_W": float(heat.max()),
            "peak_cooling_W": float(cool.max()),
        }
        for metric, value in expected.items():
            if not np.isclose(
                float(record[metric]), value, rtol=1.0e-9, atol=1.0e-6
            ):
                raise MonteCarloContractError(
                    f"Streamed run {record['run_id']} {metric} does not reconcile hourly."
                )
        if self._timestamps is None:
            self._timestamps = timestamps.copy()
            shape = (len(self.region_order), len(timestamps))
            self._heating_W = np.zeros(shape, dtype=float)
            self._cooling_W = np.zeros(shape, dtype=float)
        elif not timestamps.equals(self._timestamps):
            raise MonteCarloContractError(
                "Streamed stock results are not timestamp-aligned."
            )
        self._register_record(record)
        assert self._heating_W is not None and self._cooling_W is not None
        seed_scale = 1.0 / len(self.occupant_seeds)
        cell = (str(record["archetype_id"]), str(record["state_id"]))
        national_index = self.region_order.index(BELGIUM_REGION_ID)
        for weight in self._weights_by_cell[cell].itertuples(index=False):
            scale = float(weight.state_dwellings_2050) * seed_scale
            region_index = self.region_order.index(str(weight.region))
            self._heating_W[region_index] += scale * heat
            self._cooling_W[region_index] += scale * cool
            self._heating_W[national_index] += scale * heat
            self._cooling_W[national_index] += scale * cool

    def snapshot_arrays(self) -> dict[str, np.ndarray]:
        """Return a checkpoint-safe copy of cumulative stock-hour arrays."""

        if self._timestamps is None or self._heating_W is None or self._cooling_W is None:
            raise MonteCarloContractError("Cannot checkpoint an empty stock accumulator.")
        return {
            "timestamp_ns": self._timestamps.asi8.copy(),
            "heating_W": self._heating_W.copy(),
            "cooling_W": self._cooling_W.copy(),
        }

    def restore(
        self,
        diagnostics: pd.DataFrame,
        *,
        timestamp_ns: np.ndarray,
        heating_W: np.ndarray,
        cooling_W: np.ndarray,
        region_order: Sequence[str],
    ) -> None:
        """Restore a verified cumulative checkpoint before continuing execution."""

        if self.completed_run_count or self._timestamps is not None:
            raise MonteCarloContractError("Stock accumulator restore requires an empty instance.")
        if tuple(str(value) for value in region_order) != self.region_order:
            raise MonteCarloContractError("Checkpoint stock-region order is incompatible.")
        timestamps = pd.to_datetime(np.asarray(timestamp_ns, dtype=np.int64), utc=True)
        expected_shape = (len(self.region_order), len(timestamps))
        heat = np.asarray(heating_W, dtype=float)
        cool = np.asarray(cooling_W, dtype=float)
        if (
            heat.shape != expected_shape
            or cool.shape != expected_shape
            or not np.isfinite(heat).all()
            or not np.isfinite(cool).all()
            or (heat < 0.0).any()
            or (cool < 0.0).any()
        ):
            raise MonteCarloContractError("Checkpoint stock-hour arrays are invalid.")
        for record in diagnostics.to_dict("records"):
            self._register_record(record)
        self._timestamps = pd.DatetimeIndex(timestamps)
        self._heating_W = heat.copy()
        self._cooling_W = cool.copy()
        self._reconcile_annual_arrays(partial=True)

    def _reconcile_annual_arrays(self, *, partial: bool) -> None:
        if self._heating_W is None or self._cooling_W is None:
            raise MonteCarloContractError("Stock-hour arrays are unavailable.")
        for index, region in enumerate(self.region_order):
            hourly_heat = float(self._heating_W[index].sum()) / 1000.0
            hourly_cool = float(self._cooling_W[index].sum()) / 1000.0
            if not np.isclose(
                hourly_heat,
                self._annual_heating_kWh[region],
                rtol=1.0e-10,
                atol=1.0e-5,
            ) or not np.isclose(
                hourly_cool,
                self._annual_cooling_kWh[region],
                rtol=1.0e-10,
                atol=1.0e-5,
            ):
                qualifier = "partial " if partial else ""
                raise MonteCarloContractError(
                    f"{qualifier}stock-hour energy does not reconcile for {region}."
                )

    def _validate_complete(self) -> None:
        expected = {
            (*cell, seed) for cell in self.expected_cells for seed in self.occupant_seeds
        }
        if self._seen_cell_seeds != expected:
            missing = len(expected - self._seen_cell_seeds)
            extra = len(self._seen_cell_seeds - expected)
            raise MonteCarloContractError(
                f"Streamed stock design is incomplete or duplicated; missing={missing}, extra={extra}."
            )
        if len(self._seen_run_ids) != len(expected):
            raise MonteCarloContractError("Streamed run-ID ledger is incomplete.")
        self._reconcile_annual_arrays(partial=False)

    def finalize(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return annual/peak, hourly, and contribution tables after full coverage."""

        self._validate_complete()
        assert self._identity is not None and self._provenance is not None
        assert self._timestamps is not None
        assert self._heating_W is not None and self._cooling_W is not None
        cell_provenance_json = json.dumps(
            [
                {
                    "archetype_id": cell[0],
                    "state_id": cell[1],
                    "archetype_state_sha256": self._cell_physics_sha256[cell],
                }
                for cell in sorted(self._cell_physics_sha256)
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        base = {
            **self._identity,
            **self._provenance,
            **self.seed_provenance,
            "stock_scenario_id": "central",
            "target_year": 2050,
            "stock_weights_sha256": str(self.weights["stock_weights_sha256"].iloc[0]),
            "stock_weights_source_sha256": str(
                self.weights["stock_weights_source_sha256"].iloc[0]
            ),
            "archetype_state_provenance_json": cell_provenance_json,
            "archetype_state_provenance_sha256": hashlib.sha256(
                cell_provenance_json.encode("utf-8")
            ).hexdigest(),
        }
        base.update(_stock_partition_provenance_record(base))
        hourly_base = {
            key: base[key]
            for key in (
                *STOCK_GROUP_AXES,
                "stock_scenario_id",
                "target_year",
                "stock_partition_provenance_contract_version",
                "stock_partition_provenance_sha256",
            )
        }
        summary_records: list[dict[str, object]] = []
        hourly_records: list[pd.DataFrame] = []
        for index, region in enumerate(self.region_order):
            heat = self._heating_W[index]
            cool = self._cooling_W[index]
            heat_peak_index = int(np.argmax(heat))
            cool_peak_index = int(np.argmax(cool))
            coincident_heat_W = float(heat[heat_peak_index])
            coincident_cool_W = float(cool[cool_peak_index])
            summed_heat_W = self._sum_peak_heating_W[region]
            summed_cool_W = self._sum_peak_cooling_W[region]
            if coincident_heat_W > summed_heat_W + 1.0e-5 or (
                coincident_cool_W > summed_cool_W + 1.0e-5
            ):
                raise MonteCarloContractError(
                    "Coincident stock peak exceeds the sum of individual peaks."
                )
            selected = (
                self.weights.loc[self.weights["region"].astype(str) == region]
                if region != BELGIUM_REGION_ID
                else self.weights
            )
            annual_heat_GWh = self._annual_heating_kWh[region] / 1.0e6
            annual_cool_GWh = self._annual_cooling_kWh[region] / 1.0e6
            floor_area = self._stock_floor_area_m2[region]
            summary_records.append(
                {
                    **base,
                    "region": region,
                    "modelled_dwellings": self._modelled_dwellings[region],
                    "stock_floor_area_m2": floor_area,
                    "physics_cell_count": len(self.expected_cells),
                    "positive_weight_row_count": int(
                        (selected["state_dwellings_2050"] > 0.0).sum()
                    ),
                    "annual_heating_GWh": annual_heat_GWh,
                    "annual_potential_sensible_cooling_GWh": annual_cool_GWh,
                    "heating_intensity_kWh_m2": (
                        annual_heat_GWh * 1.0e6 / floor_area if floor_area > 0.0 else 0.0
                    ),
                    "potential_cooling_intensity_kWh_m2": (
                        annual_cool_GWh * 1.0e6 / floor_area if floor_area > 0.0 else 0.0
                    ),
                    "coincident_peak_heating_MW": coincident_heat_W / 1.0e6,
                    "coincident_peak_potential_cooling_MW": coincident_cool_W / 1.0e6,
                    "sum_individual_peak_heating_MW": summed_heat_W / 1.0e6,
                    "sum_individual_peak_potential_cooling_MW": summed_cool_W / 1.0e6,
                    "heating_diversity_factor": (
                        coincident_heat_W / summed_heat_W if summed_heat_W > 0.0 else 0.0
                    ),
                    "cooling_diversity_factor": (
                        coincident_cool_W / summed_cool_W if summed_cool_W > 0.0 else 0.0
                    ),
                    "heating_full_load_equivalent_hours": (
                        self._annual_heating_kWh[region] * 1000.0 / coincident_heat_W
                        if coincident_heat_W > 0.0
                        else 0.0
                    ),
                    "potential_cooling_full_load_equivalent_hours": (
                        self._annual_cooling_kWh[region] * 1000.0 / coincident_cool_W
                        if coincident_cool_W > 0.0
                        else 0.0
                    ),
                    "peak_heating_timestamp_utc": self._timestamps[heat_peak_index],
                    "peak_potential_cooling_timestamp_utc": self._timestamps[
                        cool_peak_index
                    ],
                    "cooling_interpretation": (
                        "potential useful sensible load under universal ideal 26C control"
                    ),
                    "stock_coverage": (
                        "R1-R4 modelled stock; R5-R6 residual excluded"
                        if self.require_full_stock
                        else "caller-supplied partial stock subset"
                    ),
                }
            )
            hourly_records.append(
                pd.DataFrame(
                    {
                        "timestamp_utc": self._timestamps,
                        **hourly_base,
                        "region": region,
                        "heating_demand_MW": heat / 1.0e6,
                        "potential_sensible_cooling_demand_MW": cool / 1.0e6,
                    }
                )
            )

        national_heat = self._annual_heating_kWh[BELGIUM_REGION_ID]
        national_cool = self._annual_cooling_kWh[BELGIUM_REGION_ID]
        contribution_records: list[dict[str, object]] = []
        for (dimension, value), contribution in sorted(self._contributions.items()):
            heat = contribution["annual_heating_kWh"]
            cool = contribution["annual_cooling_kWh"]
            floor_area = contribution["stock_floor_area_m2"]
            contribution_records.append(
                {
                    **base,
                    "contribution_dimension": dimension,
                    "contribution_value": value,
                    "modelled_dwellings": contribution["modelled_dwellings"],
                    "stock_floor_area_m2": floor_area,
                    "annual_heating_GWh": heat / 1.0e6,
                    "annual_potential_sensible_cooling_GWh": cool / 1.0e6,
                    "heating_intensity_kWh_m2": heat / floor_area if floor_area > 0.0 else 0.0,
                    "potential_cooling_intensity_kWh_m2": (
                        cool / floor_area if floor_area > 0.0 else 0.0
                    ),
                    "share_of_stock_heating": heat / national_heat if national_heat > 0.0 else 0.0,
                    "share_of_stock_potential_cooling": (
                        cool / national_cool if national_cool > 0.0 else 0.0
                    ),
                    "cooling_interpretation": (
                        "potential useful sensible load under universal ideal 26C control"
                    ),
                }
            )
        contributions = pd.DataFrame.from_records(contribution_records)
        for _, group in contributions.groupby("contribution_dimension", sort=False):
            if group["annual_heating_GWh"].sum() > 0.0 and not np.isclose(
                group["share_of_stock_heating"].sum(), 1.0, rtol=0.0, atol=1.0e-10
            ):
                raise MonteCarloContractError(
                    "Streamed stock heating contribution shares do not sum to one."
                )
            if group["annual_potential_sensible_cooling_GWh"].sum() > 0.0 and not np.isclose(
                group["share_of_stock_potential_cooling"].sum(),
                1.0,
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise MonteCarloContractError(
                    "Streamed stock cooling contribution shares do not sum to one."
                )
        return (
            pd.DataFrame.from_records(summary_records),
            pd.concat(hourly_records, ignore_index=True),
            contributions,
        )
