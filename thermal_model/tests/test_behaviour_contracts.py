from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.behaviour import (
    BehaviourContractError,
    load_behaviour_assumptions,
    load_occupant_distribution,
    sample_occupant_count,
    validate_behaviour_weather,
)
from thermal_model.validation import load_reference_weather


def test_behaviour_assumption_contract_is_complete_and_typed() -> None:
    contract = load_behaviour_assumptions()

    assert len(contract.frame) == 18
    assert contract.text("software.richardsonpy_version") == "0.2.2"
    assert contract.boolean("electricity.prev_heat_dev") is True
    assert contract.number("electricity.annual_reference_kWh") == 3500.0
    assert contract.number("control.heating_inactive_C") == 18.0
    assert contract.number("control.heating_active_C") == 20.0
    assert contract.number("control.cooling_C") == 26.0


def test_occupant_distribution_is_conditional_and_reconciles() -> None:
    distribution, checksum = load_occupant_distribution()

    assert len(distribution) == 10
    assert len(checksum) == 64
    sums = distribution.groupby("dwelling_class")["probability"].sum()
    assert sums.to_dict() == pytest.approx({"MFH": 1.0, "SFH": 1.0})
    one_person = distribution.loc[
        distribution["occupant_count"] == 1
    ].set_index("dwelling_class")["probability"]
    assert one_person["MFH"] > one_person["SFH"]


def test_sampling_is_reproducible_and_depends_on_dwelling_class() -> None:
    sfh = [sample_occupant_count("Detached house", seed) for seed in range(2000)]
    mfh = [sample_occupant_count("Apartment, enclosed", seed) for seed in range(2000)]
    distribution, _ = load_occupant_distribution()

    assert sfh == [sample_occupant_count("Detached house", seed) for seed in range(2000)]
    assert np.mean(sfh) > np.mean(mfh)
    assert set(sfh).issubset({1, 2, 3, 4, 5})
    assert set(mfh).issubset({1, 2, 3, 4, 5})
    for household_class, draws in (("SFH", sfh), ("MFH", mfh)):
        target = distribution.loc[
            distribution["dwelling_class"] == household_class
        ].set_index("occupant_count")["probability"]
        observed = pd.Series(draws).value_counts(normalize=True)
        for count in range(1, 6):
            assert observed[count] == pytest.approx(target[count], abs=0.03)


def test_reference_weather_contains_one_aligned_lighting_forcing() -> None:
    weather, _ = load_reference_weather()
    validated = validate_behaviour_weather(weather)

    expected = (
        validated["I_beam_horizontal_W_m2"]
        + validated["I_diffuse_horizontal_W_m2"]
    )
    assert validated["I_solar_W_m2"].to_numpy() == pytest.approx(
        expected.to_numpy(), abs=1.0e-6
    )


def test_behaviour_weather_rejects_a_mismatched_horizontal_total() -> None:
    weather, _ = load_reference_weather()
    broken = weather.copy()
    broken.loc[100, "I_solar_W_m2"] += 1.0

    with pytest.raises(BehaviourContractError, match="must equal"):
        validate_behaviour_weather(broken)


def test_occupant_distribution_rejects_nonunit_probability(tmp_path: Path) -> None:
    distribution, _ = load_occupant_distribution()
    distribution.loc[
        (distribution["dwelling_class"] == "SFH")
        & (distribution["occupant_count"] == 1),
        "probability",
    ] += 0.01
    path = tmp_path / "broken.csv"
    distribution.to_csv(path, index=False)

    with pytest.raises(BehaviourContractError, match="do not sum to one"):
        load_occupant_distribution(path)
