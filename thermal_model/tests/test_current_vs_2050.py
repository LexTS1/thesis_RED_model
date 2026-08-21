from __future__ import annotations

import numpy as np
import pandas as pd

from thermal_model.monte_carlo.contracts import OBSERVED_REFERENCE_SCENARIO
from thermal_model.monte_carlo.current_vs_2050 import (
    CASE_ORDER,
    OBSERVED_MEMBER_ID,
    _effect_table,
    _observed_weather_member,
    load_dual_year_stock_weights,
)


def test_observed_reference_member_is_unmorphed_and_complete() -> None:
    member = _observed_weather_member()
    assert member.member_id == OBSERVED_MEMBER_ID
    assert member.climate_scenario_id == OBSERVED_REFERENCE_SCENARIO
    assert member.observed_pvgis_year == 2015
    assert member.weather_pair_id == "pvgis_2015"
    assert member.row_count == 8760
    assert len(member.frame) == 8760


def test_dual_year_weights_preserve_stock_and_archetype_totals() -> None:
    weights = load_dual_year_stock_weights()
    assert len(weights) == 225
    assert weights[["archetype_id", "state_id"]].drop_duplicates().shape[0] == 75
    assert np.isclose(
        weights["initial_state_dwellings_2025"].sum(),
        weights["state_dwellings_2050"].sum(),
        atol=1e-5,
    )
    grouped = weights.groupby(["region", "archetype_id"])
    assert np.allclose(
        grouped["initial_state_dwellings_2025"].sum(),
        grouped["regional_number_of_dwellings"].first(),
        rtol=0.0,
        atol=1e-6,
    )


def test_paired_factorial_effect_identity() -> None:
    values = {"Q00": 100.0, "Q10": 70.0, "Q01": 90.0, "Q11": 65.0}
    rows = []
    for case_id in CASE_ORDER:
        rows.append(
            {
                "climate_scenario_id": "rcp_4_5",
                "case_id": case_id,
                "occupant_seed": 42,
                "occupant_seed_rank": 1,
                "annual_heating_TWh": values[case_id],
                "annual_potential_sensible_cooling_TWh": values[case_id] / 10.0,
                "stock_heating_intensity_kWh_m2": values[case_id] / 2.0,
                "stock_cooling_intensity_kWh_m2": values[case_id] / 20.0,
            }
        )
    effects = _effect_table(pd.DataFrame(rows))
    heat = effects.loc[effects["metric"] == "annual_heating_TWh"].set_index("effect_id")
    assert heat.loc["renovation", "effect_value"] == -30.0
    assert heat.loc["climate", "effect_value"] == -10.0
    assert heat.loc["interaction", "effect_value"] == 5.0
    assert heat.loc["combined", "effect_value"] == -35.0
