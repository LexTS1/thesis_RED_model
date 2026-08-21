from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from thermal_model.behaviour import (
    BehaviourRequest,
    generate_behaviour,
    load_behaviour_assumptions,
)
from thermal_model.contracts import load_assumption_contract, validate_archetype_state
from thermal_model.core import preprocess_archetype
from thermal_model.monte_carlo import (
    ConvergenceRule,
    MonteCarloContractError,
    MonteCarloResult,
    PROSPECTIVE_N160_CONVERGENCE_RULE,
    PROSPECTIVE_N320_N640_CONVERGENCE_RULE,
    StreamingStockAccumulator,
    aggregate_stock_results,
    archetype_state_sha256,
    build_balanced_manifest,
    distribution_summary,
    evaluate_seed_convergence,
    execute_streaming_stock_design,
    forcing_sha256,
    load_stock_weights,
    load_weather_catalog,
    load_weather_member,
    paired_renovation_deltas,
    stock_contribution_summary,
    stock_distribution_summary,
    validate_stock_weights,
    variance_contributions,
)
from thermal_model.monte_carlo.aggregation import MODELLED_BELGIAN_STOCK_DWELLINGS
from thermal_model.monte_carlo.interface import _simulate_with_behaviour
from thermal_model.monte_carlo import interface as monte_carlo_interface
from thermal_model.monte_carlo import runner as monte_carlo_runner
from thermal_model.monte_carlo.contracts import validate_weather_member
from thermal_model.monte_carlo.scenarios import (
    apply_archetype_scenario,
    effective_assumption_contract,
    resolve_model_scenario,
)
from thermal_model.validation import load_unique_archetype_states


@pytest.fixture(scope="module")
def states():
    selected = [
        item
        for item in load_unique_archetype_states()
        if item.archetype_id == "BE_TABULA_11"
        and item.state_id in {"TABULA_existing", "TABULA_advanced_A_proxy"}
    ]
    assert len(selected) == 2
    return selected


@pytest.fixture(scope="module")
def weather_2015():
    return load_weather_member("weather_2050_rcp_4_5_pvgis_2015")


@pytest.fixture(scope="module")
def weather_2010():
    return load_weather_member("weather_2050_rcp_4_5_pvgis_2010")


@pytest.fixture(scope="module")
def coupled_result(states, weather_2015):
    state = next(item for item in states if item.state_id == "TABULA_existing")
    seed = 424242
    behaviour = generate_behaviour(
        BehaviourRequest(
            dwelling_type=state.dwelling_type,
            weather=weather_2015.frame,
            weather_member_id=weather_2015.member_id,
            seed=seed,
        ),
        load_behaviour_assumptions(),
    )
    result = _simulate_with_behaviour(
        state,
        weather_2015,
        seed,
        resolve_model_scenario("central"),
        behaviour,
        load_assumption_contract(),
    )
    return state, behaviour, result


def test_weather_catalog_is_paired_and_member_forcing_is_complete(weather_2015) -> None:
    catalog = load_weather_catalog()
    assert len(catalog) == 54
    assert catalog.groupby("scenario").size().to_dict() == {
        "rcp_2_6": 18,
        "rcp_4_5": 18,
        "rcp_8_5": 18,
    }
    paired_year_sets = catalog.groupby("scenario")["observed_pvgis_year"].agg(
        lambda values: tuple(sorted(values))
    )
    assert paired_year_sets.nunique() == 1
    assert len(weather_2015.frame) == 8760
    assert {
        "I_beam_horizontal_W_m2",
        "I_diffuse_horizontal_W_m2",
        "I_north_W_m2",
        "I_east_W_m2",
        "I_south_W_m2",
        "I_west_W_m2",
    }.issubset(weather_2015.frame.columns)
    assert forcing_sha256(weather_2015.frame) == weather_2015.forcing_sha256
    altered = weather_2015.frame.copy()
    altered.loc[100, "I_south_W_m2"] += 0.001
    assert forcing_sha256(altered) != weather_2015.forcing_sha256


def test_weather_member_rejects_frame_mutation_without_matching_provenance(
    weather_2015,
) -> None:
    altered = weather_2015.frame.copy(deep=True)
    altered.loc[100, "I_south_W_m2"] += 0.001
    tampered = replace(weather_2015, frame=altered)
    with pytest.raises(MonteCarloContractError, match="forcing checksum"):
        validate_weather_member(tampered)


def test_run_id_fingerprints_complete_archetype_physics(states, weather_2015) -> None:
    state = next(item for item in states if item.state_id == "TABULA_existing")
    changed_u_record = dict(state.__dict__)
    changed_u_record["U_window_W_m2K"] += 0.01
    changed_u = validate_archetype_state(changed_u_record)

    changed_geometry_record = dict(state.__dict__)
    changed_geometry_record["roof_area_m2"] += 0.5
    changed_geometry_record["total_building_envelope_area_m2"] += 0.5
    changed_geometry = validate_archetype_state(changed_geometry_record)

    identities = {
        (item.archetype_id, item.dwelling_type, item.construction_period, item.state_id)
        for item in (state, changed_u, changed_geometry)
    }
    assert len(identities) == 1
    state_hashes = {
        archetype_state_sha256(item)
        for item in (state, changed_u, changed_geometry)
    }
    assert len(state_hashes) == 3

    run_ids = {
        build_balanced_manifest([item], [weather_2015], [1234]).loc[0, "run_id"]
        for item in (state, changed_u, changed_geometry)
    }
    assert len(run_ids) == 3


def test_structural_scenarios_rebuild_dependencies_and_keep_axes_separate(states) -> None:
    state = next(item for item in states if item.state_id == "TABULA_existing")
    central_contract = load_assumption_contract()
    central = preprocess_archetype(state, central_contract)
    heavy_contract = effective_assumption_contract(central_contract, "mass_heavy")
    heavy = preprocess_archetype(state, heavy_contract)
    assert heavy_contract.sha256 != central_contract.sha256
    assert heavy.C_m_J_K == pytest.approx(260_000.0 * heavy.floor_area_m2)
    assert heavy.A_m_m2 == pytest.approx(3.0 * heavy.floor_area_m2)
    assert heavy.H_tr_ms_W_K != central.H_tr_ms_W_K
    assert heavy.H_tr_em_W_K == pytest.approx(
        1.0 / (1.0 / heavy.H_tr_op_W_K - 1.0 / heavy.H_tr_ms_W_K)
    )
    scaled = apply_archetype_scenario(state, "infiltration_half")
    assert scaled.q50_m3_h == pytest.approx(0.5 * state.q50_m3_h)
    assert scaled.n50_h_1 == pytest.approx(0.5 * state.n50_h_1)
    assert scaled.infiltration_airflow_normal_m3_h == pytest.approx(
        0.5 * state.infiltration_airflow_normal_m3_h
    )
    assert scaled.infiltration_ach_normal_h_1 == pytest.approx(
        0.5 * state.infiltration_ach_normal_h_1
    )
    with pytest.raises(ValueError, match="Unknown model scenario"):
        resolve_model_scenario("hidden_random_override")


def test_balanced_manifest_preserves_common_seed_order(
    states, weather_2015, weather_2010
) -> None:
    seeds = (4_000_000_000, 7, 123456)
    manifest = build_balanced_manifest(
        states,
        [weather_2015, weather_2010],
        seeds,
        ["central", "mass_heavy"],
    )
    assert len(manifest) == 2 * 2 * 3 * 2
    assert manifest["run_id"].is_unique
    assert set(manifest["occupant_seed_rank"]) == {1, 2, 3}
    observed = (
        manifest.sort_values("occupant_seed_rank")
        .groupby(
            [
                "archetype_id",
                "state_id",
                "weather_member_id",
                "model_scenario_id",
            ]
        )["occupant_seed"]
        .agg(tuple)
    )
    assert all(value == seeds for value in observed)


def test_integrated_result_diagnostics_are_reconciled_and_reproducible(
    coupled_result, weather_2015
) -> None:
    state, behaviour, first = coupled_result
    second = _simulate_with_behaviour(
        state,
        weather_2015,
        first.diagnostics.occupant_seed,
        resolve_model_scenario("central"),
        behaviour,
        load_assumption_contract(),
    )
    pd.testing.assert_frame_equal(first.hourly, second.hourly, check_exact=True)
    assert first.diagnostics == second.diagnostics
    hourly = first.hourly
    diagnostics = first.diagnostics
    assert "theta_air_free_running_C" in hourly
    assert diagnostics.heating_full_load_equivalent_hours == pytest.approx(
        diagnostics.annual_heating_kWh * 1000.0 / diagnostics.peak_heating_W
    )
    assert sum(count for _, count in diagnostics.heating_setpoint_hours) == len(hourly)
    assert sum(count for _, count in diagnostics.cooling_setpoint_hours) == len(hourly)
    assert diagnostics.iso_no_load_trial_above_cooling_setpoint_hours == int(
        (
            hourly["theta_air_free_running_C"]
            > hourly["theta_set_cool_C"] + 1.0e-9
        ).sum()
    )
    assert diagnostics.iso_no_load_trial_below_heating_setpoint_hours == int(
        (
            hourly["theta_air_free_running_C"]
            < hourly["theta_set_heat_C"] - 1.0e-9
        ).sum()
    )
    assert diagnostics.heating_controlled_hours == int(
        (hourly["heating_demand_W"] > 1.0e-9).sum()
    )
    assert diagnostics.weather_forcing_sha256 == weather_2015.forcing_sha256


def test_public_interface_matches_manual_composition_without_mutating_inputs(
    coupled_result, weather_2015, monkeypatch
) -> None:
    state, behaviour, expected = coupled_result
    weather_before = weather_2015.frame.copy(deep=True)
    monkeypatch.setattr(
        monte_carlo_interface,
        "generate_behaviour",
        lambda request, contract: behaviour,
    )
    actual = monte_carlo_interface.simulate(
        state,
        weather_2015,
        expected.diagnostics.occupant_seed,
        "central",
    )
    pd.testing.assert_frame_equal(actual.hourly, expected.hourly, check_exact=True)
    assert actual.diagnostics == expected.diagnostics
    pd.testing.assert_frame_equal(weather_2015.frame, weather_before, check_exact=True)


def _synthetic_diagnostics() -> pd.DataFrame:
    central_contract = load_assumption_contract()
    behaviour_contract = load_behaviour_assumptions()
    _, occupant_distribution_sha256 = (
        monte_carlo_runner.load_occupant_distribution()
    )
    scenario_sha256 = monte_carlo_runner.model_scenario_sha256(
        resolve_model_scenario("central"), central_contract.sha256
    )
    records = []
    for weather_index, weather_id in enumerate(("w1", "w2", "w3")):
        for seed in (11, 22, 33):
            seed_effect = {11: -1.0, 22: 0.0, 33: 1.0}[seed]
            records.append(
                {
                    "archetype_id": "a1",
                    "state_id": "TABULA_existing",
                    "climate_scenario_id": "rcp_4_5",
                    "model_scenario_id": "central",
                    "weather_member_id": weather_id,
                    "archetype_state_sha256": "a" * 64,
                    "model_scenario_sha256": scenario_sha256,
                    "model_contract_version": monte_carlo_runner.MODEL_CONTRACT_VERSION,
                    "central_thermal_assumptions_sha256": central_contract.sha256,
                    "behaviour_assumptions_sha256": behaviour_contract.sha256,
                    "occupant_distribution_sha256": occupant_distribution_sha256,
                    "weather_contract_sha256": f"{weather_index + 1:064x}",
                    "weather_forcing_sha256": f"{weather_index + 11:064x}",
                    "occupant_seed": seed,
                    "heating_intensity_kWh_m2": 100.0 + 10.0 * weather_index + seed_effect,
                    "cooling_intensity_kWh_m2": 5.0 + weather_index + seed_effect,
                    "peak_heating_W": 5000.0 + 100.0 * weather_index + 10.0 * seed_effect,
                    "peak_cooling_W": 1000.0 + 50.0 * weather_index + 5.0 * seed_effect,
                    "annual_heating_kWh": 10_000.0,
                    "annual_cooling_kWh": 500.0,
                    "heating_full_load_equivalent_hours": 2000.0,
                    "cooling_full_load_equivalent_hours": 500.0,
                }
            )
    return pd.DataFrame.from_records(records)


def test_distribution_and_anova_summaries_do_not_pool_rcps() -> None:
    diagnostics = _synthetic_diagnostics()
    summary = distribution_summary(diagnostics)
    assert set(summary["climate_scenario_id"]) == {"rcp_4_5"}
    assert set(summary["sample_count"]) == {9}
    decomposition = variance_contributions(diagnostics)
    shares = decomposition.groupby("metric")["sum_of_squares_share"].sum()
    assert np.allclose(shares, 1.0, rtol=0.0, atol=1.0e-12)
    heating = decomposition.loc[
        decomposition["metric"] == "heating_intensity_kWh_m2"
    ].set_index("component")
    assert heating.loc["weather_year", "sum_of_squares_share"] > heating.loc[
        "occupant_seed", "sum_of_squares_share"
    ]


def test_nested_seed_convergence_requires_two_successive_expansions() -> None:
    diagnostics = _synthetic_diagnostics()
    rule = ConvergenceRule(
        checkpoints=(1, 2, 3),
        relative_tolerance=0.15,
        required_consecutive_expansions=2,
    )
    convergence = evaluate_seed_convergence(
        diagnostics,
        seed_order=(11, 22, 33),
        rule=rule,
    )
    assert not convergence.loc[
        convergence["seed_count"] == 1, "criterion_pass"
    ].any()
    assert convergence.loc[
        convergence["seed_count"] == 3, "converged_at_checkpoint"
    ].all()
    assert convergence.loc[
        convergence["seed_count"] == 3, "panel_converged_at_checkpoint"
    ].all()
    assert convergence["occupant_seed_bank_count"].eq(3).all()
    assert convergence["occupant_seed_bank_sha256"].nunique() == 1
    assert convergence.groupby("seed_count")[
        "occupant_seed_prefix_sha256"
    ].nunique().eq(1).all()
    assert convergence.loc[
        convergence["seed_count"] == 3, "occupant_seed_prefix_sha256"
    ].eq(convergence["occupant_seed_bank_sha256"].iloc[0]).all()


def test_runner_reconstructs_convergence_decision_and_panel_contract(tmp_path) -> None:
    diagnostics = _synthetic_diagnostics()
    second_cell = diagnostics.copy(deep=True)
    second_cell["archetype_id"] = "a2"
    second_cell["state_id"] = "TABULA_advanced_A_proxy"
    second_cell["archetype_state_sha256"] = "b" * 64
    third_cell = diagnostics.copy(deep=True)
    third_cell["archetype_id"] = "a3"
    third_cell["state_id"] = "TABULA_standard"
    third_cell["archetype_state_sha256"] = "c" * 64
    diagnostics = pd.concat(
        [diagnostics, second_cell, third_cell], ignore_index=True
    )
    seeds = (11, 22, 33)
    rule = ConvergenceRule(
        checkpoints=(1, 2, 3),
        relative_tolerance=0.15,
        required_consecutive_expansions=2,
    )
    evidence = evaluate_seed_convergence(
        diagnostics,
        seed_order=seeds,
        rule=rule,
    )
    contract_provenance = {
        column: str(evidence[column].iloc[0])
        for column in monte_carlo_runner._CONVERGENCE_CONTRACT_COLUMNS
    }
    assert {
        "archetype_state_sha256",
        "model_scenario_sha256",
        "weather_panel_sha256",
        *monte_carlo_runner._CONVERGENCE_CONTRACT_COLUMNS,
    }.issubset(evidence.columns)

    expected_state_hashes = {
        (str(row.archetype_id), str(row.state_id)): str(row.archetype_state_sha256)
        for row in evidence[
            ["archetype_id", "state_id", "archetype_state_sha256"]
        ].drop_duplicates().itertuples(index=False)
    }
    expected_scenario_hashes = {
        str(row.model_scenario_id): str(row.model_scenario_sha256)
        for row in evidence[
            ["model_scenario_id", "model_scenario_sha256"]
        ].drop_duplicates().itertuples(index=False)
    }
    expected_weather_hashes = {
        str(row.climate_scenario_id): str(row.weather_panel_sha256)
        for row in evidence[
            ["climate_scenario_id", "weather_panel_sha256"]
        ].drop_duplicates().itertuples(index=False)
    }

    def validate(
        frame: pd.DataFrame,
        *,
        declared_rule=rule,
        match_execution: bool = False,
    ):
        path = tmp_path / "convergence_evidence.csv"
        frame.to_csv(path, index=False)
        return monte_carlo_runner._validate_convergence_evidence(
            seeds,
            convergence_results_path=path,
            convergence_results_sha256=monte_carlo_runner._sha256_file(path),
            require_convergence_evidence=True,
            convergence_rule=declared_rule,
            convergence_rule_source="explicit",
            expected_climate_scenario_ids=("rcp_4_5",),
            expected_contract_provenance=contract_provenance,
            expected_archetype_state_sha256=expected_state_hashes,
            expected_model_scenario_sha256=expected_scenario_hashes,
            expected_weather_panel_sha256=expected_weather_hashes,
            require_panel_matches_execution=match_execution,
        )

    metadata, _ = validate(evidence)
    assert metadata["status"] == "VERIFIED"
    assert metadata["first_panel_converged_checkpoint"] == 3
    assert metadata["representative_physical_cell_count"] == 3
    assert metadata["representative_panel_climate_scenario_ids"] == ["rcp_4_5"]
    assert len(metadata["convergence_rule_sha256"]) == 64
    assert len(metadata["representative_panel_group_sha256"]) == 64
    matched_metadata, _ = validate(evidence, match_execution=True)
    assert matched_metadata["panel_execution_match_status"] == "VERIFIED"

    fabricated = evidence.iloc[[0]].copy()
    fabricated["criterion_pass"] = True
    fabricated["panel_converged_at_checkpoint"] = True
    with pytest.raises(MonteCarloContractError, match="checkpoints do not exactly match"):
        validate(fabricated)

    missing_statistic = evidence.drop(evidence.index[0])
    with pytest.raises(MonteCarloContractError, match="incomplete"):
        validate(missing_statistic)

    false_panel_flag = evidence.copy(deep=True)
    false_panel_flag.loc[
        false_panel_flag["seed_count"] == 3,
        "panel_converged_at_checkpoint",
    ] = False
    with pytest.raises(MonteCarloContractError, match="reconstructed panel decision"):
        validate(false_panel_flag)

    wrong_tolerance = evidence.copy(deep=True)
    wrong_tolerance["relative_tolerance"] = 0.01
    with pytest.raises(MonteCarloContractError, match="tolerance differs"):
        validate(wrong_tolerance)

    stale_contract = evidence.copy(deep=True)
    stale_contract["central_thermal_assumptions_sha256"] = "0" * 64
    with pytest.raises(MonteCarloContractError, match="current model contract"):
        validate(stale_contract)

    stale_weather_panel = evidence.copy(deep=True)
    stale_weather_panel["weather_panel_sha256"] = "d" * 64
    with pytest.raises(MonteCarloContractError, match="differs from the selected"):
        validate(stale_weather_panel, match_execution=True)

    one_cell = evidence.loc[evidence["archetype_id"] == "a1"]
    with pytest.raises(MonteCarloContractError, match="representative multi-cell"):
        validate(one_cell)

    with pytest.raises(MonteCarloContractError, match="climate scenario"):
        path = tmp_path / "convergence_evidence.csv"
        evidence.to_csv(path, index=False)
        monte_carlo_runner._validate_convergence_evidence(
            seeds,
            convergence_results_path=path,
            convergence_results_sha256=monte_carlo_runner._sha256_file(path),
            require_convergence_evidence=True,
            convergence_rule=rule,
            convergence_rule_source="explicit",
            expected_climate_scenario_ids=("rcp_2_6", "rcp_4_5"),
            expected_contract_provenance=contract_provenance,
            expected_archetype_state_sha256={},
            expected_model_scenario_sha256={},
            expected_weather_panel_sha256={},
            require_panel_matches_execution=False,
        )

    with pytest.raises(MonteCarloContractError, match="declared active rule"):
        validate(evidence, declared_rule=ConvergenceRule())


def test_paired_renovation_deltas_use_identical_weather_and_seed_cells() -> None:
    existing = _synthetic_diagnostics()
    advanced = existing.copy()
    advanced["state_id"] = "TABULA_advanced_A_proxy"
    advanced["heating_intensity_kWh_m2"] -= 40.0
    paired = paired_renovation_deltas(pd.concat([existing, advanced], ignore_index=True))
    selected = paired.loc[paired["metric"] == "heating_intensity_kWh_m2"]
    assert len(selected) == 9
    assert np.allclose(selected["delta"], -40.0)


def test_authoritative_stock_weights_reconcile() -> None:
    weights = load_stock_weights()
    assert len(weights) == 225
    assert weights[["archetype_id", "state_id"]].drop_duplicates().shape[0] == 75
    assert weights["state_dwellings_2050"].sum() == pytest.approx(
        MODELLED_BELGIAN_STOCK_DWELLINGS, abs=1.0e-5
    )
    assert (weights["state_dwellings_2050"] == 0.0).sum() == 23


def _synthetic_stock_weights() -> pd.DataFrame:
    counts = {
        ("R1", "a1"): 10.0,
        ("R1", "a2"): 20.0,
        ("R2", "a1"): 5.0,
        ("R2", "a2"): 15.0,
    }
    records = []
    regional_totals = {"R1": 30.0, "R2": 20.0}
    for (region, archetype), count in counts.items():
        state = f"state_{archetype}"
        records.append(
            {
                "scenario": "central",
                "target_year": 2050,
                "region": region,
                "archetype_id": archetype,
                "dwelling_type": "Detached house",
                "construction_period": "test",
                "state_id": state,
                "renovation_state": state,
                "state_dwellings": count,
                "state_dwellings_2050": count,
                "state_share_within_region_2050": count / regional_totals[region],
                "regional_number_of_dwellings": count,
                "regional_modelled_stock_dwellings": regional_totals[region],
            }
        )
    return pd.DataFrame.from_records(records)


def test_stock_aggregation_distinguishes_coincident_and_summed_peaks(coupled_result) -> None:
    _, _, template = coupled_result
    hours = len(template.hourly)
    results = []
    for archetype, state, power, floor_area in (
        ("a1", "state_a1", 100.0, 100.0),
        ("a2", "state_a2", 200.0, 50.0),
    ):
        for seed in (1, 2):
            hourly = template.hourly.copy(deep=True)
            heating = np.zeros(hours)
            heating[0 if archetype == "a1" else 1] = power
            hourly["heating_demand_W"] = heating
            hourly["cooling_demand_W"] = 0.0
            diagnostics = replace(
                template.diagnostics,
                run_id=f"{archetype}_{seed}",
                archetype_id=archetype,
                state_id=state,
                floor_area_m2=floor_area,
                occupant_seed=seed,
                annual_heating_kWh=power / 1000.0,
                annual_cooling_kWh=0.0,
                heating_intensity_kWh_m2=power / 1000.0 / floor_area,
                cooling_intensity_kWh_m2=0.0,
                peak_heating_W=power,
                peak_cooling_W=0.0,
            )
            results.append(MonteCarloResult(hourly=hourly, diagnostics=diagnostics))
    summary, hourly_stock = aggregate_stock_results(
        results,
        stock_weights=_synthetic_stock_weights(),
        require_full_stock=False,
    )
    national = summary.loc[summary["region"] == "Belgium_modelled_stock"].iloc[0]
    assert national["modelled_dwellings"] == 50.0
    assert national["annual_heating_GWh"] == pytest.approx(8.5e-6)
    assert national["coincident_peak_heating_MW"] == pytest.approx(0.007)
    assert national["sum_individual_peak_heating_MW"] == pytest.approx(0.0085)
    assert national["heating_diversity_factor"] < 1.0
    regions = hourly_stock.loc[
        hourly_stock["region"].isin(["R1", "R2"])
    ].groupby("timestamp_utc")["heating_demand_MW"].sum()
    belgium = hourly_stock.loc[
        hourly_stock["region"] == "Belgium_modelled_stock"
    ].set_index("timestamp_utc")["heating_demand_MW"]
    pd.testing.assert_series_equal(regions, belgium, check_names=False)
    contributions = stock_contribution_summary(
        results,
        stock_weights=_synthetic_stock_weights(),
        require_full_stock=False,
    )
    share_sums = contributions.groupby("contribution_dimension")[
        "share_of_stock_heating"
    ].sum()
    assert np.allclose(share_sums, 1.0, rtol=0.0, atol=1.0e-12)
    assert set(contributions["contribution_dimension"]) == {
        "region",
        "dwelling_type",
        "construction_period",
        "state_id",
    }


def test_stock_weights_are_always_revalidated_and_stale_hashes_are_rejected() -> None:
    validated = validate_stock_weights(
        _synthetic_stock_weights(), require_authoritative_shape=False
    )
    assert validated["stock_weights_sha256"].str.len().eq(64).all()
    stale = validated.copy(deep=True)
    stale.loc[0, "dwelling_type"] = "Mutated type"
    with pytest.raises(MonteCarloContractError, match="stored checksum"):
        validate_stock_weights(stale, require_authoritative_shape=False)
    negative = _synthetic_stock_weights()
    negative.loc[0, "state_dwellings"] = -1.0
    with pytest.raises(MonteCarloContractError, match="non-negative"):
        validate_stock_weights(negative, require_authoritative_shape=False)


def test_streaming_accumulator_matches_direct_stock_aggregation(coupled_result) -> None:
    _, _, template = coupled_result
    hours = len(template.hourly)
    results = []
    for archetype, state, power, floor_area in (
        ("a1", "state_a1", 100.0, 100.0),
        ("a2", "state_a2", 200.0, 50.0),
    ):
        for seed in (2, 1):
            hourly = template.hourly.copy(deep=True)
            heating = np.zeros(hours)
            heating[0 if archetype == "a1" else 1] = power
            hourly["heating_demand_W"] = heating
            hourly["cooling_demand_W"] = 0.0
            diagnostics = replace(
                template.diagnostics,
                run_id=f"stream_{archetype}_{seed}",
                archetype_id=archetype,
                state_id=state,
                archetype_state_sha256=("1" if archetype == "a1" else "2") * 64,
                floor_area_m2=floor_area,
                occupant_seed=seed,
                annual_heating_kWh=power / 1000.0,
                annual_cooling_kWh=0.0,
                heating_intensity_kWh_m2=power / 1000.0 / floor_area,
                cooling_intensity_kWh_m2=0.0,
                peak_heating_W=power,
                peak_cooling_W=0.0,
            )
            results.append(MonteCarloResult(hourly=hourly, diagnostics=diagnostics))
    weights = _synthetic_stock_weights()
    expected_summary, expected_hourly = aggregate_stock_results(
        results,
        stock_weights=weights,
        require_full_stock=False,
        occupant_seed_order=(2, 1),
    )
    accumulator = StreamingStockAccumulator(
        weights, (2, 1), require_full_stock=False
    )
    for result in results:
        accumulator.add(result)
    actual_summary, actual_hourly, actual_contributions = accumulator.finalize()
    assert actual_summary["stock_coverage"].eq(
        "caller-supplied partial stock subset"
    ).all()
    compare_columns = [
        "region",
        "annual_heating_GWh",
        "coincident_peak_heating_MW",
        "sum_individual_peak_heating_MW",
        "heating_diversity_factor",
    ]
    pd.testing.assert_frame_equal(
        actual_summary[compare_columns].sort_values("region").reset_index(drop=True),
        expected_summary[compare_columns].sort_values("region").reset_index(drop=True),
        check_exact=False,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    pd.testing.assert_frame_equal(
        actual_hourly[["timestamp_utc", "region", "heating_demand_MW"]]
        .sort_values(["region", "timestamp_utc"])
        .reset_index(drop=True),
        expected_hourly[["timestamp_utc", "region", "heating_demand_MW"]]
        .sort_values(["region", "timestamp_utc"])
        .reset_index(drop=True),
        check_exact=False,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert actual_summary["occupant_seeds_json"].eq("[2,1]").all()
    assert actual_summary["occupant_seed_bank_sha256"].str.len().eq(64).all()
    assert "stock_partition_provenance_sha256" in actual_hourly
    assert actual_hourly["stock_partition_provenance_sha256"].str.len().eq(64).all()
    assert actual_hourly["stock_partition_provenance_contract_version"].eq(
        "gate5_stock_partition_provenance_v1"
    ).all()
    assert actual_summary["stock_partition_provenance_contract_version"].eq(
        "gate5_stock_partition_provenance_v1"
    ).all()
    assert set(actual_hourly["stock_partition_provenance_sha256"]) == set(
        actual_summary["stock_partition_provenance_sha256"]
    )
    assert "occupant_seeds_json" not in actual_hourly
    assert "archetype_state_provenance_json" not in actual_hourly
    assert "facade_source_sha256_json" not in actual_hourly
    stock_distribution_summary(actual_summary)
    tampered_summary = actual_summary.copy(deep=True)
    tampered_summary.loc[0, "central_thermal_assumptions_sha256"] = "f" * 64
    with pytest.raises(MonteCarloContractError, match="checksum does not match"):
        stock_distribution_summary(tampered_summary)
    wrong_version = actual_summary.copy(deep=True)
    wrong_version.loc[0, "stock_partition_provenance_contract_version"] = (
        "gate5_stock_partition_provenance_v999"
    )
    with pytest.raises(MonteCarloContractError, match="Unsupported.*contract version"):
        stock_distribution_summary(wrong_version)
    assert not actual_contributions.empty
    with pytest.raises(MonteCarloContractError, match="integers"):
        StreamingStockAccumulator(weights, (1.5,), require_full_stock=False)
    with pytest.raises(MonteCarloContractError, match="integers"):
        aggregate_stock_results(
            results,
            stock_weights=weights,
            require_full_stock=False,
            occupant_seed_order=(2, 1.5),
        )
    mixed_contract = list(results)
    mixed_contract[-1] = replace(
        mixed_contract[-1],
        diagnostics=replace(
            mixed_contract[-1].diagnostics,
            effective_thermal_assumptions_sha256="f" * 64,
        ),
    )
    with pytest.raises(MonteCarloContractError, match="effective_thermal"):
        aggregate_stock_results(
            mixed_contract,
            stock_weights=weights,
            require_full_stock=False,
        )
    physics_accumulator = StreamingStockAccumulator(
        weights, (2, 1), require_full_stock=False
    )
    physics_accumulator.add(results[0])
    altered_physics = replace(
        results[1],
        diagnostics=replace(
            results[1].diagnostics, archetype_state_sha256="e" * 64
        ),
    )
    with pytest.raises(MonteCarloContractError, match="physics checksums"):
        physics_accumulator.add(altered_physics)


def _weights_for_states(selected_states) -> pd.DataFrame:
    counts = (10.0, 20.0)
    records = []
    total = sum(counts)
    for state, count in zip(sorted(selected_states, key=lambda item: item.state_id), counts):
        records.append(
            {
                "scenario": "central",
                "target_year": 2050,
                "region": "Test region",
                "archetype_id": state.archetype_id,
                "dwelling_type": state.dwelling_type,
                "construction_period": state.construction_period,
                "state_id": state.state_id,
                "renovation_state": state.state_id,
                "state_dwellings": count,
                "state_dwellings_2050": count,
                "state_share_within_region_2050": count / total,
                "regional_number_of_dwellings": total,
                "regional_modelled_stock_dwellings": total,
            }
        )
    return pd.DataFrame.from_records(records)


def test_streaming_runner_requires_convergence_evidence_by_default(
    states, weather_2015, tmp_path
) -> None:
    destination = tmp_path / "must_not_start"
    with pytest.raises(MonteCarloContractError, match="requires convergence evidence"):
        execute_streaming_stock_design(
            states,
            [weather_2015],
            (101, 202),
            output_dir=destination,
            stock_weights=_weights_for_states(states),
            require_full_stock=False,
        )
    assert not destination.exists()
    with pytest.raises(MonteCarloContractError, match="predeclared default"):
        execute_streaming_stock_design(
            states,
            [weather_2015],
            (101, 202),
            output_dir=tmp_path / "custom_full_stock_rule",
            stock_weights=_weights_for_states(states),
            require_full_stock=True,
            convergence_rule=ConvergenceRule(
                checkpoints=(1, 2),
                required_consecutive_expansions=1,
            ),
        )


def test_streaming_prepare_only_authenticates_design_without_starting_workers(
    states, weather_2015, tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "prepared_stock"

    def worker_must_not_start(*args, **kwargs):
        raise AssertionError("prepare-only started a simulation worker")

    monkeypatch.setattr(
        monte_carlo_runner,
        "_advance_streaming_stock_partitions",
        worker_must_not_start,
    )
    prepared = execute_streaming_stock_design(
        states,
        [weather_2015],
        (101, 202),
        output_dir=destination,
        stock_weights=_weights_for_states(states),
        require_full_stock=False,
        require_convergence_evidence=False,
        max_workers=2,
        prepare_only=True,
    )

    assert prepared["status"] == "PREPARED"
    assert prepared["execution_started"] is False
    assert prepared["archetype_state_count"] == len(states)
    assert prepared["weather_member_count"] == 1
    assert prepared["occupant_seed_count"] == 2
    assert prepared["partition_count"] == 1
    assert prepared["runs_per_partition"] == len(states) * 2
    assert prepared["expected_run_count"] == len(states) * 2
    assert prepared["requested_max_workers"] == 2
    assert len(prepared["design_sha256"]) == 64
    assert len(prepared["design_contract_file_sha256"]) == 64
    assert (destination / "streaming_design_contract.json").is_file()
    assert not (destination / "partitions").exists()
    status = monte_carlo_runner.streaming_stock_status(destination)
    assert status["status"] == "PREPARED"
    assert (
        status["design_contract_file_sha256"]
        == prepared["design_contract_file_sha256"]
    )

    repeated = execute_streaming_stock_design(
        states,
        [weather_2015],
        (101, 202),
        output_dir=destination,
        stock_weights=_weights_for_states(states),
        require_full_stock=False,
        require_convergence_evidence=False,
        prepare_only=True,
    )
    assert repeated["design_sha256"] == prepared["design_sha256"]
    assert (
        repeated["design_contract_file_sha256"]
        == prepared["design_contract_file_sha256"]
    )


def test_full_stock_rule_authorization_is_closed_and_provenance_labelled() -> None:
    implicit, source = monte_carlo_runner._resolve_convergence_rule_authorization(
        None,
        require_full_stock=True,
    )
    assert implicit == ConvergenceRule()
    assert source == "production_original_default_n80_implicit"

    explicit_default, source = (
        monte_carlo_runner._resolve_convergence_rule_authorization(
            ConvergenceRule(),
            require_full_stock=True,
        )
    )
    assert explicit_default == ConvergenceRule()
    assert source == "production_original_default_n80_explicit"

    prospective, source = monte_carlo_runner._resolve_convergence_rule_authorization(
        PROSPECTIVE_N160_CONVERGENCE_RULE,
        require_full_stock=True,
    )
    assert prospective is PROSPECTIVE_N160_CONVERGENCE_RULE
    assert source == "production_authorized_prospective_n160_confirmation"

    continuation, source = (
        monte_carlo_runner._resolve_convergence_rule_authorization(
            PROSPECTIVE_N320_N640_CONVERGENCE_RULE,
            require_full_stock=True,
        )
    )
    assert continuation is PROSPECTIVE_N320_N640_CONVERGENCE_RULE
    assert source == "production_authorized_prospective_n320_n640_continuation"

    custom = ConvergenceRule(checkpoints=(1, 2), required_consecutive_expansions=1)
    with pytest.raises(MonteCarloContractError, match="authorized prospective n=160"):
        monte_carlo_runner._resolve_convergence_rule_authorization(
            custom,
            require_full_stock=True,
        )
    partial, source = monte_carlo_runner._resolve_convergence_rule_authorization(
        custom,
        require_full_stock=False,
    )
    assert partial is custom
    assert source == "explicit_custom_partial_workflow"


def test_streaming_runner_resumes_at_last_complete_seed(
    states, weather_2015, coupled_result, tmp_path, monkeypatch
) -> None:
    _, _, template = coupled_result
    seeds = (101, 202)
    manifest = build_balanced_manifest(states, [weather_2015], seeds, ["central"])
    expected_ids = manifest.set_index(
        ["archetype_id", "state_id", "occupant_seed"]
    )["run_id"].to_dict()
    state_rank = {
        key: rank
        for rank, key in enumerate(
            sorted((item.archetype_id, item.state_id) for item in states), start=1
        )
    }

    def fake_result(state, member, seed, scenario, behaviour, central_contract):
        hourly = template.hourly.copy(deep=True)
        power = 100.0 + state_rank[(state.archetype_id, state.state_id)] + seed / 100.0
        heat = np.zeros(len(hourly))
        heat[state_rank[(state.archetype_id, state.state_id)] - 1] = power
        hourly["heating_demand_W"] = heat
        hourly["cooling_demand_W"] = 0.0
        diagnostics = replace(
            template.diagnostics,
            run_id=expected_ids[(state.archetype_id, state.state_id, seed)],
            archetype_id=state.archetype_id,
            dwelling_type=state.dwelling_type,
            construction_period=state.construction_period,
            state_id=state.state_id,
            archetype_state_sha256=archetype_state_sha256(state),
            floor_area_m2=state.floor_surface_area_m2,
            occupant_seed=seed,
            annual_heating_kWh=power / 1000.0,
            annual_cooling_kWh=0.0,
            heating_intensity_kWh_m2=power / 1000.0 / state.floor_surface_area_m2,
            cooling_intensity_kWh_m2=0.0,
            peak_heating_W=power,
            peak_cooling_W=0.0,
        )
        return MonteCarloResult(hourly=hourly, diagnostics=diagnostics)

    monkeypatch.setattr(monte_carlo_runner, "generate_behaviour", lambda *args: object())
    failed = {"done": False}

    def fail_during_second_seed(*args):
        if args[2] == seeds[1] and not failed["done"]:
            failed["done"] = True
            raise RuntimeError("intentional interruption")
        return fake_result(*args)

    monkeypatch.setattr(
        monte_carlo_runner, "_simulate_with_behaviour", fail_during_second_seed
    )
    destination = tmp_path / "streaming"
    with pytest.raises(RuntimeError, match="intentional interruption"):
        execute_streaming_stock_design(
            states,
            [weather_2015],
            seeds,
            output_dir=destination,
            stock_weights=_weights_for_states(states),
            require_full_stock=False,
            require_convergence_evidence=False,
        )
    progress_path = next((destination / "partitions").glob("*/progress.json"))
    progress = pd.read_json(progress_path, typ="series")
    assert int(progress["completed_seed_count"]) == 1
    failure_path = progress_path.parent / "last_failure.json"
    failure = pd.read_json(failure_path, typ="series")
    assert failure["status"] == "FAILED"
    assert int(failure["occupant_seed"]) == seeds[1]

    resumed_calls = []

    def resumed(*args):
        resumed_calls.append(args[2])
        return fake_result(*args)

    monkeypatch.setattr(monte_carlo_runner, "_simulate_with_behaviour", resumed)
    summary = execute_streaming_stock_design(
        states,
        [weather_2015],
        seeds,
        output_dir=destination,
        stock_weights=_weights_for_states(states),
        require_full_stock=False,
        require_convergence_evidence=False,
    )
    assert resumed_calls == [seeds[1]] * len(states)
    assert summary["completed_run_count"] == len(states) * len(seeds)
    assert summary["status"] == "PARTIAL_STOCK_WORKFLOW"
    assert summary["stock_coverage_status"] == "PARTIAL_SUBSET"
    assert (
        summary["convergence_evidence"]["status"]
        == "NOT_VERIFIED_BY_RUNNER"
    )
    recovered = pd.read_json(failure_path, typ="series")
    assert recovered["status"] == "RECOVERED"
    assert (destination / "stock_distribution_summary.csv").is_file()
    distributions = pd.read_csv(destination / "stock_distribution_summary.csv")
    assert set(distributions["metric"]) == {
        "annual_heating_GWh",
        "annual_potential_sensible_cooling_GWh",
        "coincident_peak_heating_MW",
        "coincident_peak_potential_cooling_MW",
    }
    assert distributions["climate_scenario_id"].eq("rcp_4_5").all()
    assert distributions["interval_interpretation"].str.contains(
        "not a complete prediction interval"
    ).all()

    monkeypatch.setattr(
        monte_carlo_runner,
        "_simulate_with_behaviour",
        lambda *args: (_ for _ in ()).throw(AssertionError("completed run re-executed")),
    )
    repeated = execute_streaming_stock_design(
        states,
        [weather_2015],
        seeds,
        output_dir=destination,
        stock_weights=_weights_for_states(states),
        require_full_stock=False,
        require_convergence_evidence=False,
    )
    assert repeated["design_sha256"] == summary["design_sha256"]

    uninterrupted_destination = tmp_path / "streaming_uninterrupted"
    monkeypatch.setattr(monte_carlo_runner, "_simulate_with_behaviour", fake_result)
    uninterrupted = execute_streaming_stock_design(
        states,
        [weather_2015],
        seeds,
        output_dir=uninterrupted_destination,
        stock_weights=_weights_for_states(states),
        require_full_stock=False,
        require_convergence_evidence=False,
    )
    assert uninterrupted["artifact_sha256"] == summary["artifact_sha256"]
    resumed_complete = pd.read_json(
        next((destination / "partitions").glob("*/partition_complete.json")),
        typ="series",
    )
    uninterrupted_complete = pd.read_json(
        next(
            (uninterrupted_destination / "partitions").glob(
                "*/partition_complete.json"
            )
        ),
        typ="series",
    )
    assert resumed_complete["artifacts"] == uninterrupted_complete["artifacts"]

    convergence_input = pd.DataFrame.from_records(
        [
            {
                "archetype_id": state.archetype_id,
                "state_id": state.state_id,
                "climate_scenario_id": weather_2015.climate_scenario_id,
                "model_scenario_id": "central",
                "weather_member_id": weather_2015.member_id,
                "archetype_state_sha256": archetype_state_sha256(state),
                "model_scenario_sha256": template.diagnostics.model_scenario_sha256,
                "model_contract_version": template.diagnostics.model_contract_version,
                "central_thermal_assumptions_sha256": (
                    template.diagnostics.central_thermal_assumptions_sha256
                ),
                "behaviour_assumptions_sha256": (
                    template.diagnostics.behaviour_assumptions_sha256
                ),
                "occupant_distribution_sha256": (
                    template.diagnostics.occupant_distribution_sha256
                ),
                "weather_contract_sha256": weather_2015.weather_contract_sha256,
                "weather_forcing_sha256": weather_2015.forcing_sha256,
                "occupant_seed": seed,
                "heating_intensity_kWh_m2": 50.0,
                "cooling_intensity_kWh_m2": 2.0,
                "peak_heating_W": 4000.0,
                "peak_cooling_W": 500.0,
            }
            for state in states
            for seed in seeds
        ]
    )
    third_panel_cell = convergence_input.loc[
        convergence_input["state_id"] == states[0].state_id
    ].copy(deep=True)
    third_panel_cell["archetype_id"] = "representative_third_cell"
    third_panel_cell["state_id"] = "TABULA_standard"
    third_panel_cell["archetype_state_sha256"] = "e" * 64
    convergence_input = pd.concat(
        [convergence_input, third_panel_cell], ignore_index=True
    )
    convergence_rule = ConvergenceRule(
        checkpoints=(1, 2),
        relative_tolerance=0.02,
        required_consecutive_expansions=1,
    )
    evidence = evaluate_seed_convergence(
        convergence_input,
        seed_order=seeds,
        rule=convergence_rule,
    )
    evidence_path = tmp_path / "selected_convergence_results.csv"
    evidence.to_csv(evidence_path, index=False)
    evidence_sha256 = monte_carlo_runner._sha256_file(evidence_path)

    with pytest.raises(MonteCarloContractError, match="exact ordered seed prefix"):
        execute_streaming_stock_design(
            states,
            [weather_2015],
            tuple(reversed(seeds)),
            output_dir=tmp_path / "wrong_seed_order",
            stock_weights=_weights_for_states(states),
            require_full_stock=False,
            convergence_results_path=evidence_path,
            convergence_results_sha256=evidence_sha256,
            convergence_rule=convergence_rule,
        )

    verified_destination = tmp_path / "streaming_verified"
    verified = execute_streaming_stock_design(
        states,
        [weather_2015],
        seeds,
        output_dir=verified_destination,
        stock_weights=_weights_for_states(states),
        require_full_stock=False,
        convergence_results_path=evidence_path,
        convergence_results_sha256=evidence_sha256,
        convergence_rule=convergence_rule,
    )
    assert verified["status"] == "PARTIAL_STOCK_WORKFLOW"
    assert verified["stock_coverage_status"] == "PARTIAL_SUBSET"
    assert verified["convergence_evidence"]["status"] == "VERIFIED"
    assert verified["convergence_evidence"][
        "first_panel_converged_checkpoint"
    ] == len(seeds)
    persisted_evidence = verified_destination / "convergence_results.csv"
    assert persisted_evidence.read_bytes() == evidence_path.read_bytes()
    assert verified["artifact_sha256"]["convergence_results.csv"] == evidence_sha256


def test_streaming_top_level_pass_requires_full_stock_and_convergence() -> None:
    assert monte_carlo_runner._streaming_design_qualification(
        require_full_stock=True,
        convergence_verified=True,
    )[0] == "PASS"
    assert monte_carlo_runner._streaming_design_qualification(
        require_full_stock=True,
        convergence_verified=False,
    )[0] == "WORKFLOW_CHECK_ONLY"
    assert monte_carlo_runner._streaming_design_qualification(
        require_full_stock=False,
        convergence_verified=True,
    )[0] == "PARTIAL_STOCK_WORKFLOW"
    assert monte_carlo_runner._streaming_design_qualification(
        require_full_stock=False,
        convergence_verified=False,
    )[0] == "PARTIAL_STOCK_WORKFLOW"
