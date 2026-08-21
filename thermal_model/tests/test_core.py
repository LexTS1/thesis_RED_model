from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.contracts import (
    ContractError,
    PreparedArchetype,
    SimulationInput,
    assemble_archetype_state,
    load_assumption_contract,
)
from thermal_model.core import (
    ThermalCoreError,
    evaluate_5r1c_hour,
    preprocess_archetype,
    simulate,
    solar_gains,
    solve_ideal_hour,
    split_gains,
    ventilation_conductance,
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
def synthetic(contract) -> PreparedArchetype:
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


def zero_facades() -> dict[str, float]:
    return {
        "I_north_W_m2": 0.0,
        "I_east_W_m2": 0.0,
        "I_south_W_m2": 0.0,
        "I_west_W_m2": 0.0,
    }


def constant_year(
    archetype: PreparedArchetype,
    *,
    theta_out_C: float = 5.0,
    internal_gain_W: float = 0.0,
) -> SimulationInput:
    timestamps = pd.date_range(
        "2021-01-01T00:00:00Z", "2021-12-31T23:00:00Z", freq="h"
    )
    count = len(timestamps)
    weather = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "T_out_C": np.full(count, theta_out_C),
            "I_north_W_m2": np.zeros(count),
            "I_east_W_m2": np.zeros(count),
            "I_south_W_m2": np.zeros(count),
            "I_west_W_m2": np.zeros(count),
        }
    )
    schedules = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "Phi_int_W": np.full(count, internal_gain_W),
            "theta_set_heat_C": np.full(count, 20.0),
            "theta_set_cool_C": np.full(count, 26.0),
        }
    )
    return SimulationInput(archetype, weather, schedules, "constant_2021", 0)


def test_preprocessor_matches_independent_be_tabula_01_oracle(contract) -> None:
    base = pd.read_csv(BASE_MATRIX)
    states = pd.read_csv(STATE_MATRIX)
    base_row = base.loc[base["archetype_id"] == "BE_TABULA_01"].iloc[0]
    state_row = states.loc[
        (states["archetype_id"] == "BE_TABULA_01")
        & (states["state_id"] == "TABULA_existing")
    ].iloc[0]
    prepared = preprocess_archetype(
        assemble_archetype_state(base_row, state_row), contract
    )

    assert prepared.H_tr_w_W_K == pytest.approx(206.0, abs=1e-12)
    assert prepared.H_tr_op_W_K == pytest.approx(899.529437070993, abs=1e-10)
    assert prepared.H_tr_is_W_K == pytest.approx(4331.475, abs=1e-12)
    assert prepared.H_tr_ms_W_K == pytest.approx(6347.25, abs=1e-12)
    assert prepared.H_tr_em_W_K == pytest.approx(1048.060037863, abs=1e-9)
    assert prepared.glazing_g_value == 0.85


def test_preprocessor_accepts_all_stock_states_and_maps_glazing(contract) -> None:
    base = pd.read_csv(BASE_MATRIX).set_index("archetype_id", drop=False)
    states = pd.read_csv(STATE_MATRIX)
    prepared = [
        preprocess_archetype(
            assemble_archetype_state(base.loc[row.archetype_id], row._asdict()),
            contract,
        )
        for row in states.itertuples(index=False)
    ]
    assert len(prepared) == 225
    assert {item.glazing_g_value for item in prepared} == {0.85, 0.75, 0.67}
    assert all(item.H_tr_op_W_K < item.H_tr_ms_W_K for item in prepared)
    assert all(
        item.H_tr_em_W_K
        == pytest.approx(
            1.0 / (1.0 / item.H_tr_op_W_K - 1.0 / item.H_tr_ms_W_K),
            rel=1e-13,
        )
        for item in prepared
    )


def test_preprocessor_rejects_unsupported_glazing_and_invalid_equivalent_branch(
    contract,
) -> None:
    base = pd.read_csv(BASE_MATRIX).iloc[0]
    states = pd.read_csv(STATE_MATRIX)
    state = states.loc[states["archetype_id"] == base["archetype_id"]].iloc[0].copy()
    state["U_window_W_m2K"] = 2.7
    with pytest.raises(ThermalCoreError, match="no unique documented glazing mapping"):
        preprocess_archetype(assemble_archetype_state(base, state), contract)

    broken = contract.frame.copy()
    broken.loc[
        broken["assumption_id"] == "network.mass_surface_coefficient",
        "value_numeric",
    ] = 0.01
    altered = replace(contract, frame=broken)
    state["U_window_W_m2K"] = 5.0
    with pytest.raises(ThermalCoreError, match="require 0 < H_tr,op < H_tr,ms"):
        preprocess_archetype(assemble_archetype_state(base, state), altered)

    nonzero_bridges = contract.frame.copy()
    nonzero_bridges.loc[
        nonzero_bridges["assumption_id"] == "boundary.thermal_bridges",
        "value_numeric",
    ] = 0.1
    with pytest.raises(ThermalCoreError, match="zero-thermal-bridge scope"):
        preprocess_archetype(
            assemble_archetype_state(base, state),
            replace(contract, frame=nonzero_bridges),
        )


def test_transmission_uses_gross_window_area_but_solar_uses_glazed_area(
    synthetic,
) -> None:
    assert synthetic.H_tr_w_W_K == 50.0
    irradiance = {
        "I_north_W_m2": 100.0,
        "I_east_W_m2": 200.0,
        "I_south_W_m2": 300.0,
        "I_west_W_m2": 400.0,
    }
    expected = (
        (100.0 * 5.0 + 200.0 * 5.0 + 300.0 * 10.0 + 400.0 * 5.0)
        * (1.0 - 0.3)
        * 0.85
        * 0.9
        * 0.6
    )
    assert solar_gains(synthetic, irradiance) == pytest.approx(expected, abs=1e-12)
    no_windows = replace(
        synthetic,
        window_area_north_m2=0.0,
        window_area_east_m2=0.0,
        window_area_south_m2=0.0,
        window_area_west_m2=0.0,
        H_tr_w_W_K=0.0,
    )
    assert solar_gains(no_windows, irradiance) == 0.0


def test_ventilation_recovery_never_applies_to_infiltration(synthetic) -> None:
    balanced = replace(
        synthetic,
        ventilation_system="balanced_mechanical_HRV",
        hrv_efficiency=0.8,
        summer_bypass=True,
    )
    nominal = ventilation_conductance(balanced)
    bypass = ventilation_conductance(balanced, bypass_active=True)
    infiltration_H = 1.2 * 1005.0 * 100.0 / 3600.0
    full_ventilation_H = 1.2 * 1005.0 * (0.4 * 275.0) / 3600.0
    assert nominal.H_ve_W_K == pytest.approx(
        infiltration_H + (1.0 - 0.8) * full_ventilation_H
    )
    assert bypass.H_ve_W_K == pytest.approx(infiltration_H + full_ventilation_H)
    assert nominal.infiltration_airflow_m3_h == bypass.infiltration_airflow_m3_h


def test_zero_airflow_remains_a_well_defined_solver_edge_case(synthetic) -> None:
    sealed = replace(
        synthetic,
        window_area_north_m2=0.0,
        window_area_east_m2=0.0,
        window_area_south_m2=0.0,
        window_area_west_m2=0.0,
        H_tr_w_W_K=0.0,
        infiltration_airflow_m3_h=0.0,
        ventilation_ach_h_1=0.0,
    )
    ventilation = ventilation_conductance(sealed)
    solution = evaluate_5r1c_hour(
        sealed,
        theta_mass_previous_C=20.0,
        theta_out_C=5.0,
        gains=split_gains(sealed, 0.0, 0.0),
        ventilation=ventilation,
    )

    assert ventilation.H_ve_W_K == 0.0
    assert np.isfinite(solution.theta_air_C)
    assert np.isfinite(solution.theta_mass_end_C)
    assert solution.max_abs_energy_balance_residual_W < 1e-8


def test_gain_allocation_matches_iso_window_correction(synthetic) -> None:
    allocation = split_gains(synthetic, Phi_int_W=300.0, Phi_solar_W=500.0)
    base = 0.5 * 300.0 + 500.0
    mass_fraction = 250.0 / 450.0
    surface_fraction = 1.0 - mass_fraction - 50.0 / (9.1 * 450.0)
    assert allocation.Phi_ia_W == 150.0
    assert allocation.Phi_m_W == pytest.approx(mass_fraction * base)
    assert allocation.Phi_st_W == pytest.approx(surface_fraction * base)
    correction = 50.0 / (9.1 * 450.0) * base
    assert (
        300.0 + 500.0
        - allocation.Phi_ia_W
        - allocation.Phi_m_W
        - allocation.Phi_st_W
    ) == pytest.approx(correction)


def test_annex_c_numeric_oracle_and_node_energy_balances(synthetic) -> None:
    ventilation = ventilation_conductance(synthetic)
    gains = split_gains(synthetic, 300.0, 0.0)
    free = evaluate_5r1c_hour(
        synthetic,
        theta_mass_previous_C=20.0,
        theta_out_C=5.0,
        gains=gains,
        ventilation=ventilation,
    )
    test = evaluate_5r1c_hour(
        synthetic,
        theta_mass_previous_C=20.0,
        theta_out_C=5.0,
        gains=gains,
        ventilation=ventilation,
        signed_hvac_load_W=1000.0,
    )
    assert ventilation.H_ve_W_K == pytest.approx(70.35, abs=1e-12)
    assert free.theta_air_C == pytest.approx(18.53482140317, abs=1e-11)
    assert test.theta_air_C == pytest.approx(19.62178089399, abs=1e-11)
    assert free.max_abs_energy_balance_residual_W < 1e-8
    assert test.max_abs_energy_balance_residual_W < 1e-8

    ideal = solve_ideal_hour(
        synthetic,
        theta_mass_previous_C=20.0,
        theta_out_C=5.0,
        irradiance_W_m2=zero_facades(),
        Phi_int_W=300.0,
        theta_set_heat_C=20.0,
        theta_set_cool_C=26.0,
    )
    assert ideal.signed_hvac_load_W == pytest.approx(1347.96062705309, abs=1e-9)
    assert ideal.theta_air_C == pytest.approx(20.0, abs=1e-12)
    assert ideal.theta_mass_end_C == pytest.approx(19.63029241044, abs=1e-11)
    assert ideal.theta_mass_C == pytest.approx(19.81514620522, abs=1e-11)
    assert ideal.theta_surface_C == pytest.approx(19.71484017581, abs=1e-11)


def test_free_decay_matches_closed_form_crank_nicolson_ratio(synthetic) -> None:
    ventilation = ventilation_conductance(synthetic)
    gains = split_gains(synthetic, 0.0, 0.0)
    solution = evaluate_5r1c_hour(
        synthetic,
        theta_mass_previous_C=20.0,
        theta_out_C=5.0,
        gains=gains,
        ventilation=ventilation,
    )
    H_1 = ventilation.H_ve_W_K * synthetic.H_tr_is_W_K / (
        ventilation.H_ve_W_K + synthetic.H_tr_is_W_K
    )
    H_2 = H_1 + synthetic.H_tr_w_W_K
    H_3 = H_2 * synthetic.H_tr_ms_W_K / (H_2 + synthetic.H_tr_ms_W_K)
    dynamic = H_3 + synthetic.H_tr_em_W_K
    storage = synthetic.C_m_J_K / 3600.0
    ratio = (storage - 0.5 * dynamic) / (storage + 0.5 * dynamic)
    expected_end = 5.0 + ratio * (20.0 - 5.0)
    assert solution.theta_mass_end_C == pytest.approx(expected_end, abs=1e-12)
    assert 5.0 < solution.theta_mass_end_C < 20.0


def test_ideal_cooling_and_bypass_decision_are_fixed_for_the_hour(synthetic) -> None:
    balanced = replace(
        synthetic,
        ventilation_system="balanced_mechanical_HRV",
        hrv_efficiency=0.8,
        summer_bypass=True,
    )
    bypassed = solve_ideal_hour(
        balanced,
        theta_mass_previous_C=30.0,
        theta_out_C=20.0,
        irradiance_W_m2=zero_facades(),
        Phi_int_W=2000.0,
        theta_set_heat_C=20.0,
        theta_set_cool_C=26.0,
    )
    assert bypassed.ventilation.bypass_active is True
    assert bypassed.ventilation.effective_hrv_efficiency == 0.0
    assert bypassed.cooling_demand_W > 0.0
    assert bypassed.heating_demand_W == 0.0
    assert bypassed.theta_air_C == pytest.approx(26.0, abs=1e-10)

    hot_outside = solve_ideal_hour(
        balanced,
        theta_mass_previous_C=30.0,
        theta_out_C=35.0,
        irradiance_W_m2=zero_facades(),
        Phi_int_W=2000.0,
        theta_set_heat_C=20.0,
        theta_set_cool_C=26.0,
    )
    assert hot_outside.ventilation.bypass_active is False
    assert hot_outside.ventilation.effective_hrv_efficiency == 0.8
    assert hot_outside.cooling_demand_W > 0.0


def test_periodic_year_converges_to_correct_steady_heating_load(
    synthetic, contract
) -> None:
    result = simulate(constant_year(synthetic), contract)
    ventilation = ventilation_conductance(synthetic)
    envelope_to_air = 1.0 / (
        1.0 / synthetic.H_tr_is_W_K
        + 1.0 / (synthetic.H_tr_op_W_K + synthetic.H_tr_w_W_K)
    )
    expected_load = (ventilation.H_ve_W_K + envelope_to_air) * (20.0 - 5.0)
    np.testing.assert_allclose(
        result.hourly["heating_demand_W"], expected_load, rtol=0.0, atol=1e-7
    )
    assert (result.hourly["cooling_demand_W"] == 0.0).all()
    assert result.diagnostics.warmup_cycles == 2
    assert result.diagnostics.max_abs_energy_balance_residual_W < 1e-7
    assert result.diagnostics.annual_heating_kWh == pytest.approx(
        expected_load * 8760.0 / 1000.0, rel=1e-11
    )


def test_core_rejects_negative_forcing_and_checksum_mismatch(synthetic, contract) -> None:
    bad_solar = zero_facades()
    bad_solar["I_south_W_m2"] = -0.1
    with pytest.raises(ThermalCoreError, match="must be non-negative"):
        solar_gains(synthetic, bad_solar)

    wrong_checksum = replace(synthetic, assumptions_sha256="0" * 64)
    with pytest.raises(ThermalCoreError, match="checksum"):
        simulate(constant_year(wrong_checksum), contract)

    with pytest.raises(ContractError, match="Heating setpoint exceeds"):
        solve_ideal_hour(
            synthetic,
            theta_mass_previous_C=20.0,
            theta_out_C=5.0,
            irradiance_W_m2=zero_facades(),
            Phi_int_W=0.0,
            theta_set_heat_C=27.0,
            theta_set_cool_C=26.0,
        )


def test_periodic_warmup_fails_explicitly_when_cycle_limit_is_too_small(
    synthetic, contract
) -> None:
    one_cycle = contract.frame.copy()
    one_cycle.loc[
        one_cycle["assumption_id"] == "solver.warmup_max_cycles", "value_numeric"
    ] = 1
    with pytest.raises(ThermalCoreError, match="convergence failed after 1"):
        simulate(constant_year(synthetic), replace(contract, frame=one_cycle))
