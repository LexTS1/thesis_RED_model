"""Typed contracts for the Gate-5 Monte Carlo orchestration layer.

The deterministic 5R1C solver remains in :mod:`thermal_model.core`.  These
contracts identify the uncertainty axes and carry enough provenance to make a
single stochastic dwelling run reproducible without giving the core knowledge
of climate manifests, RichardsonPy, stock weights, or sensitivity scenarios.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from thermal_model.behaviour.contracts import validate_behaviour_weather
from thermal_model.contracts import ArchetypeStateInput, validate_weather_frame


MODEL_CONTRACT_VERSION = "residential_energy_demand_gate5_v1"
CLIMATE_SCENARIOS = ("rcp_2_6", "rcp_4_5", "rcp_8_5")
OBSERVED_REFERENCE_SCENARIO = "observed_reference"
WEATHER_SCENARIOS = (*CLIMATE_SCENARIOS, OBSERVED_REFERENCE_SCENARIO)
COMPLETE_WEATHER_FORCING_COLUMNS = (
    "T_out_C",
    "I_beam_horizontal_W_m2",
    "I_diffuse_horizontal_W_m2",
    "I_solar_W_m2",
    "I_north_W_m2",
    "I_east_W_m2",
    "I_south_W_m2",
    "I_west_W_m2",
)


class MonteCarloContractError(ValueError):
    """Raised when a Gate-5 input or result violates its declared contract."""


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping with stable ordering and separators."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def complete_weather_forcing_sha256(frame: pd.DataFrame) -> str:
    """Hash every timestamp and weather value consumed by Gates 4 and 5.

    This primitive lives in the contracts module so both the climate loader and
    :func:`validate_weather_member` can use the exact same implementation
    without introducing a contracts-to-weather import cycle.
    """

    missing = sorted(
        {"timestamp_utc", *COMPLETE_WEATHER_FORCING_COLUMNS}.difference(frame.columns)
    )
    if missing:
        raise MonteCarloContractError(
            f"Weather forcing hash is missing columns: {missing}."
        )
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    )
    if timestamps.has_duplicates:
        raise MonteCarloContractError(
            "Cannot hash weather forcing with duplicate timestamps."
        )
    digest = hashlib.sha256()
    digest.update(b"gate5_complete_weather_forcing_v1\0")
    digest.update(np.asarray(timestamps.asi8, dtype="<i8").tobytes(order="C"))
    for column in COMPLETE_WEATHER_FORCING_COLUMNS:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise MonteCarloContractError(
                f"Cannot hash non-finite forcing column {column}."
            )
        digest.update(column.encode("ascii") + b"\0")
        digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(str(value)) != 64:
        raise MonteCarloContractError(f"{name} must contain a 64-character SHA-256 digest.")
    try:
        int(str(value), 16)
    except ValueError as exc:
        raise MonteCarloContractError(f"{name} is not hexadecimal.") from exc


@dataclass(frozen=True)
class WeatherMember:
    """One validated climate member with on-demand façade forcing attached."""

    member_id: str
    climate_scenario_id: str
    climate_target: str
    weather_pair_id: str
    observed_pvgis_year: int
    is_leap_year: bool
    row_count: int
    frame: pd.DataFrame
    site_id: str
    latitude: float
    longitude: float
    elevation_m: float
    timezone: str
    gcm_model: str
    rcm_model: str
    cordex_ensemble_member: str
    member_sha256: str
    metadata_sha256: str
    manifest_sha256: str
    morph_contract_sha256: str
    facade_source_sha256: tuple[tuple[str, str], ...]
    weather_contract_sha256: str
    forcing_sha256: str


@dataclass(frozen=True)
class ModelScenario:
    """One allow-listed epistemic sensitivity, independent of random seeds."""

    scenario_id: str
    axis: str
    description: str
    mass_capacitance_J_m2K: float | None = None
    mass_area_ratio_m2_m2: float | None = None
    vertical_shading_factor: float | None = None
    infiltration_multiplier: float = 1.0

    def definition(self) -> dict[str, Any]:
        """Return the canonical, serializable scenario definition."""

        return asdict(self)


@dataclass(frozen=True)
class MonteCarloDiagnostics:
    """Annual metrics and complete identifiers for one dwelling realization."""

    run_id: str
    archetype_id: str
    dwelling_type: str
    dwelling_class: str
    construction_period: str
    state_id: str
    archetype_state_sha256: str
    floor_area_m2: float
    climate_scenario_id: str
    weather_member_id: str
    weather_pair_id: str
    observed_pvgis_year: int
    climate_target: str
    occupant_seed: int
    occupant_count: int
    richardson_seed: int
    model_scenario_id: str
    model_scenario_axis: str
    annual_heating_kWh: float
    annual_cooling_kWh: float
    heating_intensity_kWh_m2: float
    cooling_intensity_kWh_m2: float
    peak_heating_W: float
    peak_cooling_W: float
    heating_full_load_equivalent_hours: float
    cooling_full_load_equivalent_hours: float
    heating_controlled_hours: int
    cooling_controlled_hours: int
    heating_setpoint_hours: tuple[tuple[float, int], ...]
    cooling_setpoint_hours: tuple[tuple[float, int], ...]
    iso_no_load_trial_above_cooling_setpoint_hours: int
    iso_no_load_trial_below_heating_setpoint_hours: int
    hrv_bypass_hours: int
    annual_internal_gains_kWh: float
    annual_household_electricity_kWh: float
    max_abs_energy_balance_residual_W: float
    warmup_cycles: int
    model_contract_version: str
    central_thermal_assumptions_sha256: str
    effective_thermal_assumptions_sha256: str
    behaviour_assumptions_sha256: str
    occupant_distribution_sha256: str
    model_scenario_sha256: str
    member_sha256: str
    metadata_sha256: str
    climate_manifest_sha256: str
    morph_contract_sha256: str
    facade_source_sha256: tuple[tuple[str, str], ...]
    weather_contract_sha256: str
    weather_forcing_sha256: str


@dataclass(frozen=True)
class MonteCarloResult:
    """Side-effect-free Gate-5 response: hourly series plus annual diagnostics."""

    hourly: pd.DataFrame
    diagnostics: MonteCarloDiagnostics


@dataclass(frozen=True)
class RunSpec:
    """One row of a balanced, restartable experiment manifest."""

    run_id: str
    archetype_id: str
    dwelling_type: str
    construction_period: str
    state_id: str
    archetype_state_sha256: str
    climate_scenario_id: str
    weather_member_id: str
    weather_pair_id: str
    observed_pvgis_year: int
    occupant_seed: int
    occupant_seed_rank: int
    model_scenario_id: str
    model_scenario_axis: str
    weather_contract_sha256: str
    model_scenario_sha256: str


def validate_weather_member(member: WeatherMember) -> WeatherMember:
    """Validate identities, provenance and both thermal/behavioural forcing."""

    if not isinstance(member, WeatherMember):
        raise MonteCarloContractError("weather_member must be a WeatherMember instance.")
    for name in (
        "member_id",
        "climate_target",
        "weather_pair_id",
        "site_id",
        "gcm_model",
        "rcm_model",
        "cordex_ensemble_member",
    ):
        if not str(getattr(member, name)).strip():
            raise MonteCarloContractError(f"WeatherMember.{name} must be non-empty.")
    if member.climate_scenario_id not in WEATHER_SCENARIOS:
        raise MonteCarloContractError(
            f"Unknown climate scenario {member.climate_scenario_id!r}."
        )
    if member.timezone != "UTC":
        raise MonteCarloContractError("WeatherMember.timezone must be UTC.")
    if member.weather_pair_id != f"pvgis_{member.observed_pvgis_year}":
        raise MonteCarloContractError("Weather pair identifier does not match the PVGIS year.")
    expected_rows = 8784 if member.is_leap_year else 8760
    if member.row_count != expected_rows or len(member.frame) != expected_rows:
        raise MonteCarloContractError(
            f"WeatherMember row count must be {expected_rows}; got "
            f"metadata={member.row_count}, frame={len(member.frame)}."
        )
    if not np.isfinite(
        np.asarray([member.latitude, member.longitude, member.elevation_m], dtype=float)
    ).all():
        raise MonteCarloContractError("WeatherMember site coordinates must be finite.")
    for name in (
        "member_sha256",
        "metadata_sha256",
        "manifest_sha256",
        "morph_contract_sha256",
        "weather_contract_sha256",
        "forcing_sha256",
    ):
        _validate_sha256(getattr(member, name), f"WeatherMember.{name}")
    facade = dict(member.facade_source_sha256)
    if set(facade) != {"north", "east", "south", "west"}:
        raise MonteCarloContractError("Façade provenance must cover north/east/south/west.")
    for orientation, checksum in facade.items():
        _validate_sha256(checksum, f"facade_source_sha256[{orientation}]")

    thermal = validate_weather_frame(member.frame)
    behavioural = validate_behaviour_weather(member.frame)
    if not pd.DatetimeIndex(thermal["timestamp_utc"]).equals(
        pd.DatetimeIndex(behavioural["timestamp_utc"])
    ):
        raise MonteCarloContractError("Thermal and behavioural weather timestamps differ.")
    actual_forcing_sha256 = complete_weather_forcing_sha256(member.frame)
    if actual_forcing_sha256 != member.forcing_sha256:
        raise MonteCarloContractError(
            "WeatherMember forcing checksum does not match its actual frame: "
            f"declared {member.forcing_sha256}, computed {actual_forcing_sha256}."
        )
    # Return a defensive copy: validation must not expose caller-owned storage to
    # downstream code that may add temporary columns.
    return replace(member, frame=member.frame.copy(deep=True))


def validate_model_scenario(scenario: ModelScenario) -> None:
    """Reject incomplete or physically impossible scenario declarations."""

    if not isinstance(scenario, ModelScenario):
        raise MonteCarloContractError("model_scenario must resolve to ModelScenario.")
    if not scenario.scenario_id.strip() or not scenario.axis.strip():
        raise MonteCarloContractError("Model scenario identifiers must be non-empty.")
    if not np.isfinite(float(scenario.infiltration_multiplier)) or (
        scenario.infiltration_multiplier <= 0.0
    ):
        raise MonteCarloContractError("Infiltration multiplier must be finite and positive.")
    mass_values = (
        scenario.mass_capacitance_J_m2K,
        scenario.mass_area_ratio_m2_m2,
    )
    if (mass_values[0] is None) != (mass_values[1] is None):
        raise MonteCarloContractError(
            "Mass capacitance and effective-area overrides must be declared together."
        )
    if mass_values[0] is not None and (
        float(mass_values[0]) <= 0.0 or float(mass_values[1]) <= 0.0
    ):
        raise MonteCarloContractError("Mass overrides must be positive.")
    if scenario.vertical_shading_factor is not None and not (
        0.0 <= float(scenario.vertical_shading_factor) <= 1.0
    ):
        raise MonteCarloContractError("Vertical shading factor must be within [0, 1].")


def diagnostics_to_record(diagnostics: MonteCarloDiagnostics) -> dict[str, Any]:
    """Flatten one diagnostic object into a stable CSV-ready record."""

    record = asdict(diagnostics)
    for name in (
        "heating_setpoint_hours",
        "cooling_setpoint_hours",
        "facade_source_sha256",
    ):
        record[f"{name}_json"] = json.dumps(record.pop(name), separators=(",", ":"))
    return record


def archetype_identity(state: ArchetypeStateInput) -> dict[str, str]:
    """Return the four physical stock identifiers used by every run ID."""

    return {
        "archetype_id": state.archetype_id,
        "dwelling_type": state.dwelling_type,
        "construction_period": state.construction_period,
        "state_id": state.state_id,
    }


def archetype_state_sha256(state: ArchetypeStateInput) -> str:
    """Fingerprint the complete validated archetype-state physics contract."""

    if not isinstance(state, ArchetypeStateInput):
        raise MonteCarloContractError(
            "archetype_state_sha256 requires a validated ArchetypeStateInput."
        )
    return canonical_sha256(
        {
            "contract": "gate5_archetype_state_v1",
            "fields": asdict(state),
        }
    )
