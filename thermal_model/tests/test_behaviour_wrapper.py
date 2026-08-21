from __future__ import annotations

import calendar
import random
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from thermal_model.behaviour import BehaviourRequest, generate_behaviour
from thermal_model.behaviour.coupling import flatten_lighting_gain
from thermal_model.behaviour.wrapper import _isolated_richardson_randomness
from thermal_model.validation import load_reference_weather


@pytest.fixture(scope="module")
def reference_weather() -> tuple[pd.DataFrame, dict]:
    return load_reference_weather()


@pytest.fixture(scope="module")
def generated_behaviour(reference_weather):
    weather, metadata = reference_weather
    request = BehaviourRequest(
        dwelling_type="Detached house",
        weather=weather,
        weather_member_id=metadata["weather_member_id"],
        seed=20250805,
    )
    return request, generate_behaviour(request)


def test_generated_profile_is_aligned_normalized_and_physical(
    generated_behaviour,
) -> None:
    request, result = generated_behaviour
    hourly = result.hourly

    assert len(hourly) == 8760
    assert pd.DatetimeIndex(hourly["timestamp_utc"]).equals(
        pd.DatetimeIndex(request.weather["timestamp_utc"])
    )
    assert result.diagnostics.prev_heat_dev is True
    assert result.diagnostics.annual_total_electricity_kWh == pytest.approx(
        3500.0, abs=1.0e-6
    )
    assert (hourly["total_electricity_W"] >= 0.0).all()
    assert (hourly["Phi_int_W"] >= 0.0).all()
    assert hourly["total_electricity_W"].nunique() > 1
    assert set(hourly["theta_set_heat_C"].unique()) == {18.0, 20.0}
    assert set(hourly["theta_set_cool_C"].unique()) == {26.0}
    assert result.schedules.columns.tolist() == [
        "timestamp_utc",
        "Phi_int_W",
        "theta_set_heat_C",
        "theta_set_cool_C",
    ]


def test_fixed_seed_reproduces_profile_exactly(generated_behaviour) -> None:
    request, first = generated_behaviour
    second = generate_behaviour(request)

    pd.testing.assert_frame_equal(first.hourly, second.hourly, check_exact=True)
    assert first.diagnostics == second.diagnostics


def test_a_different_seed_changes_the_profile(generated_behaviour) -> None:
    request, first = generated_behaviour
    different = generate_behaviour(replace(request, seed=request.seed + 1))

    assert not first.hourly["total_electricity_W"].equals(
        different.hourly["total_electricity_W"]
    )


def test_richardson_rng_and_month_patch_are_isolated() -> None:
    import richardsonpy.classes.stochastic_el_load_wrapper as stochastic_wrapper

    original_method = stochastic_wrapper.ElectricityProfile._get_month
    random.seed(98765)
    np.random.seed(98765)
    expected_python = random.random()
    expected_numpy = np.random.random()
    random.seed(98765)
    np.random.seed(98765)

    with _isolated_richardson_randomness(123, leap_year=True):
        profile = object.__new__(stochastic_wrapper.ElectricityProfile)
        assert profile._get_month(0) == 1
        assert profile._get_month(31) == 2
        assert profile._get_month(59) == 2
        assert profile._get_month(60) == 3

    assert stochastic_wrapper.ElectricityProfile._get_month is original_method
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert calendar.isleap(2016)


def test_flattened_lighting_preserves_annual_energy(generated_behaviour) -> None:
    _, result = generated_behaviour
    flattened = flatten_lighting_gain(result)
    non_lighting = (
        result.hourly["occupant_sensible_gain_W"]
        + result.hourly["appliance_sensible_gain_W"]
    )
    original_lighting = result.hourly["lighting_sensible_gain_W"]
    inferred_flat_lighting = flattened["Phi_int_W"] - non_lighting

    assert np.ptp(inferred_flat_lighting.to_numpy(dtype=float)) <= 1.0e-12
    assert inferred_flat_lighting.sum() == pytest.approx(
        original_lighting.sum(), abs=1.0e-8
    )
