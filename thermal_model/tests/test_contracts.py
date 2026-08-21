from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.contracts import (
    ARCHETYPE_STATE_FIELD_SPECS,
    ASSUMPTION_BINDINGS,
    RESULT_FIELD_SPECS,
    SCHEDULE_FIELD_SPECS,
    WEATHER_FIELD_SPECS,
    ContractError,
    PreparedArchetype,
    SimulationDiagnostics,
    SimulationInput,
    SimulationResult,
    assemble_archetype_state,
    load_assumption_contract,
    validate_assumption_contract,
    validate_prepared_archetype,
    validate_schedule_frame,
    validate_simulation_input,
    validate_simulation_result,
    validate_weather_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_MATRIX = (
    PROJECT_ROOT
    / "BE_building_stock/data/matrices/national/base_physical_archetype_matrix.csv"
)
STATE_MATRIX = (
    PROJECT_ROOT
    / "BE_building_stock/data/scenarios/renovation/archetype_matrix_2050_renovation_scenarios.csv"
)


@pytest.fixture(scope="module")
def contract():
    return load_assumption_contract()


@pytest.fixture(scope="module")
def prepared(contract) -> PreparedArchetype:
    return PreparedArchetype(
        archetype_id="BE_TABULA_TEST",
        dwelling_type="Detached house",
        construction_period="pre-1946",
        state_id="TABULA_existing",
        floor_area_m2=100.0,
        zone_volume_m3=275.0,
        window_area_north_m2=5.0,
        window_area_east_m2=5.0,
        window_area_south_m2=10.0,
        window_area_west_m2=5.0,
        glazing_g_value=0.85,
        window_frame_fraction=0.3,
        non_normal_irradiance_factor=0.9,
        vertical_shading_factor=0.6,
        A_t_m2=450.0,
        A_m_m2=250.0,
        C_m_J_K=16_500_000.0,
        H_tr_w_W_K=50.0,
        H_tr_op_W_K=100.0,
        H_tr_is_W_K=1552.5,
        H_tr_ms_W_K=2275.0,
        H_tr_em_W_K=104.59770114942529,
        infiltration_airflow_m3_h=100.0,
        ventilation_ach_h_1=0.4,
        ventilation_system="existing_unspecified",
        hrv_efficiency=0.0,
        summer_bypass=False,
        air_density_kg_m3=1.2,
        air_specific_heat_J_kgK=1005.0,
        assumptions_sha256=contract.sha256,
    )


def hourly_inputs(year: int = 2021) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range(
        f"{year}-01-01T00:00:00Z",
        f"{year}-12-31T23:00:00Z",
        freq="h",
    )
    count = len(timestamps)
    weather = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "T_out_C": np.full(count, 5.0),
            "I_north_W_m2": np.zeros(count),
            "I_east_W_m2": np.zeros(count),
            "I_south_W_m2": np.zeros(count),
            "I_west_W_m2": np.zeros(count),
        }
    )
    schedules = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "Phi_int_W": np.full(count, 300.0),
            "theta_set_heat_C": np.full(count, 20.0),
            "theta_set_cool_C": np.full(count, 26.0),
        }
    )
    return weather, schedules


def test_assumption_contract_is_complete_typed_and_fully_bound(contract) -> None:
    assert len(contract.frame) == 69
    assert len(contract.sha256) == 64
    assert set(contract.frame["assumption_id"]) == set(ASSUMPTION_BINDINGS)
    assert contract.number("network.mass_capacitance_ratio") == 165_000.0
    assert contract.text("network.mass_class") == "medium"

    broken = contract.frame.copy()
    broken.loc[
        broken["assumption_id"] == "network.mass_capacitance_ratio", "unit"
    ] = ""
    with pytest.raises(ContractError, match="Numeric assumptions require units"):
        validate_assumption_contract(broken)


def test_field_schemas_expose_units_for_every_required_value() -> None:
    for schema in (
        ARCHETYPE_STATE_FIELD_SPECS,
        WEATHER_FIELD_SPECS,
        SCHEDULE_FIELD_SPECS,
        RESULT_FIELD_SPECS,
    ):
        assert schema
        assert all(spec.unit for spec in schema.values())


def test_all_real_2050_physical_states_join_to_base_geometry() -> None:
    base = pd.read_csv(BASE_MATRIX).set_index("archetype_id", drop=False)
    states = pd.read_csv(STATE_MATRIX)
    assert len(states) == 225
    assembled = [
        assemble_archetype_state(base.loc[row.archetype_id], row._asdict())
        for row in states.itertuples(index=False)
    ]
    assert len(assembled) == 225
    assert {item.state_id for item in assembled} == {
        "TABULA_existing",
        "TABULA_standard_B_proxy",
        "TABULA_advanced_A_proxy",
    }
    existing = next(item for item in assembled if item.state_id == "TABULA_existing")
    assert existing.hrv_eta == 0.0
    assert existing.summer_bypass is False


def test_archetype_join_rejects_wrong_identity_and_infiltration() -> None:
    base = pd.read_csv(BASE_MATRIX).iloc[0]
    states = pd.read_csv(STATE_MATRIX)
    state = states.loc[states["archetype_id"] == base["archetype_id"]].iloc[0].copy()

    wrong_id = state.copy()
    wrong_id["archetype_id"] = "BE_TABULA_OTHER"
    with pytest.raises(ContractError, match="different archetype_id"):
        assemble_archetype_state(base, wrong_id)

    broken_flow = state.copy()
    broken_flow["infiltration_airflow_normal_m3_h"] += 1.0
    with pytest.raises(ContractError, match="inconsistent with q50/n-factor"):
        assemble_archetype_state(base, broken_flow)


def test_prepared_archetype_schema_rejects_invalid_network(prepared) -> None:
    validate_prepared_archetype(prepared)
    with pytest.raises(ContractError, match="H_tr_op_W_K < H_tr_ms_W_K"):
        validate_prepared_archetype(
            replace(prepared, H_tr_op_W_K=prepared.H_tr_ms_W_K)
        )


@pytest.mark.parametrize("year, expected", [(2021, 8760), (2020, 8784)])
def test_hourly_inputs_accept_complete_utc_calendar_years(year: int, expected: int) -> None:
    weather, schedules = hourly_inputs(year)
    assert len(validate_weather_frame(weather)) == expected
    assert len(validate_schedule_frame(schedules)) == expected


def test_hourly_validators_reject_missing_values_bounds_and_naive_time() -> None:
    weather, schedules = hourly_inputs()
    with pytest.raises(ContractError, match="missing required columns"):
        validate_weather_frame(weather.drop(columns="I_north_W_m2"))

    negative_solar = weather.copy()
    negative_solar.loc[0, "I_south_W_m2"] = -1.0
    with pytest.raises(ContractError, match="below 0.0 W/m2"):
        validate_weather_frame(negative_solar)

    naive = weather.copy()
    naive["timestamp_utc"] = naive["timestamp_utc"].dt.tz_localize(None)
    with pytest.raises(ContractError, match="timezone-naive"):
        validate_weather_frame(naive)

    invalid_deadband = schedules.copy()
    invalid_deadband.loc[0, "theta_set_heat_C"] = 28.0
    with pytest.raises(ContractError, match="Heating setpoint exceeds cooling setpoint"):
        validate_schedule_frame(invalid_deadband)


def test_simulation_input_requires_exact_timestamp_alignment(prepared) -> None:
    weather, schedules = hourly_inputs()
    request = SimulationInput(prepared, weather, schedules, "weather_test", 42)
    validated = validate_simulation_input(request)
    assert str(validated.weather["timestamp_utc"].dt.tz) == "UTC"

    _, shifted = hourly_inputs(2020)
    with pytest.raises(ContractError, match="not exactly aligned"):
        validate_simulation_input(
            SimulationInput(prepared, weather, shifted, "weather_test", 42)
        )


def test_simulation_result_schema_reconciles_hourly_and_annual_values(prepared) -> None:
    weather, schedules = hourly_inputs()
    request = validate_simulation_input(
        SimulationInput(prepared, weather, schedules, "weather_test", 42)
    )
    count = len(weather)
    hourly = pd.DataFrame(
        {
            "timestamp_utc": request.weather["timestamp_utc"],
            "T_out_C": request.weather["T_out_C"],
            "theta_air_C": np.full(count, 20.0),
            "theta_surface_C": np.full(count, 20.0),
            "theta_mass_C": np.full(count, 20.0),
            "theta_operative_C": np.full(count, 20.0),
            "theta_air_free_running_C": np.full(count, 20.0),
            "Phi_internal_W": request.schedules["Phi_int_W"],
            "Phi_solar_W": np.zeros(count),
            "heating_demand_W": np.full(count, 100.0),
            "cooling_demand_W": np.zeros(count),
            "theta_set_heat_C": request.schedules["theta_set_heat_C"],
            "theta_set_cool_C": request.schedules["theta_set_cool_C"],
            "H_ve_W_K": np.full(count, 50.0),
            "hrv_bypass_active": np.zeros(count, dtype=bool),
        }
    )
    annual_heating = count * 100.0 / 1000.0
    diagnostics = SimulationDiagnostics(
        archetype_id=prepared.archetype_id,
        state_id=prepared.state_id,
        weather_member_id="weather_test",
        occupant_seed=42,
        model_scenario="central",
        assumptions_sha256=prepared.assumptions_sha256,
        annual_heating_kWh=annual_heating,
        annual_cooling_kWh=0.0,
        heating_intensity_kWh_m2=annual_heating / prepared.floor_area_m2,
        cooling_intensity_kWh_m2=0.0,
        peak_heating_W=100.0,
        peak_cooling_W=0.0,
        max_abs_energy_balance_residual_W=0.0,
        warmup_cycles=2,
    )
    validated = validate_simulation_result(SimulationResult(hourly, diagnostics), request)
    assert validated.diagnostics.annual_heating_kWh == pytest.approx(876.0)

    simultaneous = hourly.copy()
    simultaneous.loc[0, "cooling_demand_W"] = 10.0
    with pytest.raises(ContractError, match="simultaneous"):
        validate_simulation_result(SimulationResult(simultaneous, diagnostics), request)
