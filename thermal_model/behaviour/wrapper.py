"""Reproducible RichardsonPy adapter for hourly 5R1C boundary conditions."""

from __future__ import annotations

import calendar
import importlib.metadata
import random
import threading
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pandas as pd

from .contracts import (
    DEFAULT_OCCUPANT_DISTRIBUTION_PATH,
    BehaviourAssumptionContract,
    BehaviourContractError,
    BehaviourDiagnostics,
    BehaviourRequest,
    BehaviourResult,
    load_behaviour_assumptions,
    load_occupant_distribution,
    validate_behaviour_request,
    validate_behaviour_result,
    weather_forcing_sha256,
)


_RICHARDSON_LOCK = threading.RLock()

_SFH_TYPES = {
    "Detached house",
    "Semi-detached house",
    "Terraced house",
}
_MFH_TYPES = {
    "Apartment, enclosed",
    "Apartment, exposed",
}


def dwelling_class(dwelling_type: str) -> str:
    """Map exact TABULA dwelling labels to RichardsonPy's SFH/MFH switch."""

    if dwelling_type in _SFH_TYPES:
        return "SFH"
    if dwelling_type in _MFH_TYPES:
        return "MFH"
    raise BehaviourContractError(f"Unsupported dwelling type {dwelling_type!r}.")


def _seed_streams(seed: int) -> tuple[np.random.Generator, int]:
    occupant_sequence, richardson_sequence = np.random.SeedSequence(seed).spawn(2)
    occupant_rng = np.random.default_rng(occupant_sequence)
    richardson_rng = np.random.default_rng(richardson_sequence)
    richardson_seed = int(
        richardson_rng.integers(0, 2**32, dtype=np.uint32)
    )
    return occupant_rng, richardson_seed


def sample_occupant_count(
    dwelling_type: str,
    seed: int,
    *,
    distribution_path=None,
) -> int:
    """Sample 1–5 occupants from the Census distribution conditional on type."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise BehaviourContractError("seed must be an integer.")
    if not 0 <= int(seed) <= 2**32 - 1:
        raise BehaviourContractError("seed must be between 0 and 2**32-1.")
    distribution, _ = load_occupant_distribution(
        distribution_path
        if distribution_path is not None
        else DEFAULT_OCCUPANT_DISTRIBUTION_PATH
    )
    household_class = dwelling_class(dwelling_type)
    selected = distribution.loc[distribution["dwelling_class"] == household_class]
    rng, _ = _seed_streams(int(seed))
    return int(
        rng.choice(
            selected["occupant_count"].to_numpy(dtype=int),
            p=selected["probability"].to_numpy(dtype=float),
        )
    )


@contextmanager
def _isolated_richardson_randomness(seed: int, *, leap_year: bool) -> Iterator[None]:
    """Seed RichardsonPy without leaking or racing global RNG/class changes."""

    import richardsonpy.classes.stochastic_el_load_wrapper as stochastic_wrapper

    with _RICHARDSON_LOCK:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        original_get_month = stochastic_wrapper.ElectricityProfile._get_month

        use_leap_calendar = leap_year

        def corrected_get_month(instance, day, leap_year=False):
            # ElectricLoad passes a zero-based day and omits the leap flag while
            # the helper compares one-based cumulative day counts.  Correct both
            # locally and restore the dependency class immediately afterwards.
            return original_get_month(instance, day + 1, leap_year=use_leap_calendar)
        try:
            random.seed(seed)
            np.random.seed(seed)
            stochastic_wrapper.ElectricityProfile._get_month = corrected_get_month
            yield
        finally:
            stochastic_wrapper.ElectricityProfile._get_month = original_get_month
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def _fixed_cet_to_local(values_utc: np.ndarray) -> np.ndarray:
    """Map UTC forcing to a periodic fixed-CET (+01:00) model year."""

    return np.roll(values_utc, 1)


def _fixed_cet_to_utc(values_local: np.ndarray) -> np.ndarray:
    """Map a periodic fixed-CET (+01:00) profile back to UTC timestamps."""

    return np.roll(values_local, -1)


def _annual_kwh(power_W: np.ndarray | pd.Series) -> float:
    return float(np.asarray(power_W, dtype=float).sum()) / 1000.0


def generate_behaviour(
    request: BehaviourRequest,
    assumptions: BehaviourAssumptionContract | None = None,
) -> BehaviourResult:
    """Generate one validated household profile and narrow thermal schedule.

    RichardsonPy runs at its native one-minute resolution.  Hourly appliance
    and lighting powers are arithmetic means, so annual electrical energy is
    conserved exactly.  Ten-minute active occupancy is aggregated as both a
    mean active-person count (metabolic heat) and any-active boolean (setpoint).
    """

    contract = assumptions or load_behaviour_assumptions()
    validated = validate_behaviour_request(request)
    expected_version = contract.text("software.richardsonpy_version")
    try:
        installed_version = importlib.metadata.version("richardsonpy")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BehaviourContractError(
            "RichardsonPy is required only for the behaviour layer; install its pinned requirement."
        ) from exc
    if installed_version != expected_version:
        raise BehaviourContractError(
            f"Expected richardsonpy {expected_version}, found {installed_version}."
        )

    import richardsonpy.classes.electric_load as electric_load
    import richardsonpy.classes.occupancy as occupancy

    household_class = dwelling_class(validated.dwelling_type)
    distribution, distribution_sha256 = load_occupant_distribution()
    occupant_rng, richardson_seed = _seed_streams(validated.seed)
    if validated.occupant_count is None:
        selected_distribution = distribution.loc[
            distribution["dwelling_class"] == household_class
        ]
        occupant_count = int(
            occupant_rng.choice(
                selected_distribution["occupant_count"].to_numpy(dtype=int),
                p=selected_distribution["probability"].to_numpy(dtype=float),
            )
        )
    else:
        occupant_count = validated.occupant_count

    timestamps = pd.DatetimeIndex(validated.weather["timestamp_utc"])
    year = int(timestamps[0].year)
    days = len(timestamps) // 24
    initial_day = pd.Timestamp(year=year, month=1, day=1).isoweekday()
    beam_local_hourly = _fixed_cet_to_local(
        validated.weather["I_beam_horizontal_W_m2"].to_numpy(dtype=float)
    )
    diffuse_local_hourly = _fixed_cet_to_local(
        validated.weather["I_diffuse_horizontal_W_m2"].to_numpy(dtype=float)
    )
    beam_local_minute_kW_m2 = np.repeat(beam_local_hourly / 1000.0, 60)
    diffuse_local_minute_kW_m2 = np.repeat(diffuse_local_hourly / 1000.0, 60)

    with _isolated_richardson_randomness(
        richardson_seed, leap_year=calendar.isleap(year)
    ):
        occupancy_object = occupancy.Occupancy(
            number_occupants=occupant_count,
            initial_day=initial_day,
            nb_days=days,
        )
        electricity_object = electric_load.ElectricLoad(
            occ_profile=occupancy_object.occupancy,
            total_nb_occ=occupant_count,
            q_direct=beam_local_minute_kW_m2,
            q_diffuse=diffuse_local_minute_kW_m2,
            annual_demand=contract.number("electricity.annual_reference_kWh"),
            is_sfh=household_class == "SFH",
            randomize_appliances=contract.boolean("electricity.randomize_appliances"),
            prev_heat_dev=contract.boolean("electricity.prev_heat_dev"),
            light_config=int(contract.number("electricity.light_config")),
            timestep=int(contract.number("solver.native_timestep_seconds")),
            initial_day=initial_day,
            season_light_mod=contract.boolean("electricity.season_light_mod"),
            do_normalization=True,
            save_app_light=True,
        )

    minute_count = len(timestamps) * 60
    ten_minute_count = len(timestamps) * 6
    if len(electricity_object.app_load) != minute_count:
        raise BehaviourContractError("RichardsonPy appliance profile has unexpected length.")
    if len(electricity_object.light_load) != minute_count:
        raise BehaviourContractError("RichardsonPy lighting profile has unexpected length.")
    if len(occupancy_object.occupancy) != ten_minute_count:
        raise BehaviourContractError("RichardsonPy occupancy profile has unexpected length.")

    app_local = np.asarray(electricity_object.app_load, dtype=float).reshape(-1, 60).mean(axis=1)
    light_local = np.asarray(electricity_object.light_load, dtype=float).reshape(-1, 60).mean(axis=1)
    occupancy_ten_minute = np.asarray(occupancy_object.occupancy, dtype=float).reshape(-1, 6)
    active_mean_local = occupancy_ten_minute.mean(axis=1)
    active_state_local = occupancy_ten_minute.max(axis=1) > 0.0

    appliance_W = _fixed_cet_to_utc(app_local)
    lighting_W = _fixed_cet_to_utc(light_local)
    active_mean = _fixed_cet_to_utc(active_mean_local)
    active_state = _fixed_cet_to_utc(active_state_local)
    total_electricity_W = appliance_W + lighting_W

    occupant_gain_W = (
        active_mean * contract.number("occupancy.metabolic_sensible_W_person")
    )
    appliance_gain_W = (
        appliance_W * contract.number("electricity.appliance_sensible_fraction")
    )
    lighting_gain_W = (
        lighting_W * contract.number("electricity.lighting_sensible_fraction")
    )
    total_internal_gain_W = occupant_gain_W + appliance_gain_W + lighting_gain_W
    heating_setpoint_C = np.where(
        active_state,
        contract.number("control.heating_active_C"),
        contract.number("control.heating_inactive_C"),
    )
    cooling_setpoint_C = np.full(
        len(timestamps), contract.number("control.cooling_C"), dtype=float
    )

    hourly = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "active_occupants_mean": active_mean,
            "active_occupancy": active_state.astype(bool),
            "appliance_electricity_W": appliance_W,
            "lighting_electricity_W": lighting_W,
            "total_electricity_W": total_electricity_W,
            "occupant_sensible_gain_W": occupant_gain_W,
            "appliance_sensible_gain_W": appliance_gain_W,
            "lighting_sensible_gain_W": lighting_gain_W,
            "Phi_int_W": total_internal_gain_W,
            "theta_set_heat_C": heating_setpoint_C,
            "theta_set_cool_C": cooling_setpoint_C,
        }
    )

    local_standard_timestamps = timestamps + pd.Timedelta(hours=1)
    day_mask = (local_standard_timestamps.hour >= 7) & (
        local_standard_timestamps.hour < 23
    )
    weekday_mask = local_standard_timestamps.weekday < 5
    diagnostics = BehaviourDiagnostics(
        dwelling_type=validated.dwelling_type,
        dwelling_class=household_class,
        occupant_count=occupant_count,
        occupant_count_bin="5_or_more" if occupant_count == 5 else str(occupant_count),
        seed=validated.seed,
        richardson_seed=richardson_seed,
        weather_member_id=validated.weather_member_id,
        weather_forcing_sha256=weather_forcing_sha256(validated.weather),
        richardsonpy_version=installed_version,
        prev_heat_dev=contract.boolean("electricity.prev_heat_dev"),
        annual_electricity_target_kWh=contract.number("electricity.annual_reference_kWh"),
        annual_appliance_electricity_kWh=_annual_kwh(appliance_W),
        annual_lighting_electricity_kWh=_annual_kwh(lighting_W),
        annual_total_electricity_kWh=_annual_kwh(total_electricity_W),
        active_occupant_hours=float(active_mean.sum()),
        active_occupancy_hours=int(active_state.sum()),
        mean_internal_gain_W=float(total_internal_gain_W.mean()),
        peak_internal_gain_W=float(total_internal_gain_W.max()),
        mean_electricity_W=float(total_electricity_W.mean()),
        peak_electricity_W=float(total_electricity_W.max()),
        daytime_mean_electricity_W=float(total_electricity_W[day_mask].mean()),
        nighttime_mean_electricity_W=float(total_electricity_W[~day_mask].mean()),
        weekday_mean_electricity_W=float(total_electricity_W[weekday_mask].mean()),
        weekend_mean_electricity_W=float(total_electricity_W[~weekday_mask].mean()),
        behaviour_assumptions_sha256=contract.sha256,
        occupant_distribution_sha256=distribution_sha256,
    )
    result = BehaviourResult(hourly=hourly, diagnostics=diagnostics)
    validate_behaviour_result(result, validated, contract)
    return result
