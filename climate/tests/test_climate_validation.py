from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from climate.src.load_cordex import load_config, sha256_file
from climate.src.load_observed import load_clean_observed, split_complete_years
from climate.src.morph import load_delta_contract, morph_observed_year
from climate.src.validate import (
    assess_plausibility,
    calculate_annual_metrics,
    calculate_cordex_annual_degree_days,
    calculate_degree_days,
    load_reference_degree_days,
    reference_comparison_statistics,
    run_validation,
    validate_hourly_frame,
    validate_member_pair,
)
from climate.src.plot_validation import build_validation_figures


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def validation_data():
    config = load_config(CONFIG_PATH)
    observed, _ = load_clean_observed(config)
    years = split_complete_years(observed)
    contract = load_delta_contract(config)
    return config, years, contract


def test_degree_days_use_official_conditional_thresholds() -> None:
    config = load_config(CONFIG_PATH)
    daily = pd.Series([15.0, 15.0001, 23.9999, 24.0])
    metrics = calculate_degree_days(daily, config)
    assert metrics["HDD_C_days"] == pytest.approx(3.0)
    assert metrics["CDD_C_days"] == pytest.approx(3.0)


def test_annual_metrics_use_utc_daily_means_and_hourly_solar_energy() -> None:
    config = load_config(CONFIG_PATH)
    timestamps = pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "T_out_C": [15.0] * 24 + [24.0] * 24,
            "I_solar_W_m2": [100.0] * 48,
        }
    )
    metrics = calculate_annual_metrics(frame, config)
    assert metrics["HDD_C_days"] == pytest.approx(3.0)
    assert metrics["CDD_C_days"] == pytest.approx(3.0)
    assert metrics["annual_solar_kWh_m2"] == pytest.approx(4.8)


def test_complete_leap_year_passes_strict_hourly_validation(validation_data) -> None:
    config, years, _ = validation_data
    validated = validate_hourly_frame(years[2020], 2020, config, "synthetic leap year")
    assert len(validated) == 8784
    assert validated["timestamp_utc"].iat[-1].isoformat() == "2020-12-31T23:00:00+00:00"


def test_hourly_validation_rejects_structural_and_physical_faults(validation_data) -> None:
    config, years, _ = validation_data
    source = years[2005]

    gap = source.drop(index=100).reset_index(drop=True)
    with pytest.raises(ValueError, match="complete hourly UTC year"):
        validate_hourly_frame(gap, 2005, config, "gap")

    duplicate = source.copy()
    duplicate.loc[1, "timestamp_utc"] = duplicate.loc[0, "timestamp_utc"]
    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_hourly_frame(duplicate, 2005, config, "duplicate")

    missing = source.copy()
    missing.loc[10, "T_out_C"] = float("nan")
    with pytest.raises(ValueError, match="missing values"):
        validate_hourly_frame(missing, 2005, config, "missing")

    non_finite = source.copy()
    non_finite.loc[10, "T_out_C"] = float("inf")
    with pytest.raises(ValueError, match="non-finite values"):
        validate_hourly_frame(non_finite, 2005, config, "infinite")

    hot = source.copy()
    hot.loc[10, "T_out_C"] = 50.1
    with pytest.raises(ValueError, match="temperature lies outside"):
        validate_hourly_frame(hot, 2005, config, "hot")

    negative = source.copy()
    negative.loc[10, "I_beam_horizontal_W_m2"] = -0.1
    with pytest.raises(ValueError, match="I_beam_horizontal_W_m2 lies outside"):
        validate_hourly_frame(negative, 2005, config, "negative solar")

    inconsistent = source.copy()
    daylight_index = inconsistent.index[inconsistent["I_solar_W_m2"] > 0.0][0]
    inconsistent.loc[daylight_index, "I_solar_W_m2"] += 1.0
    with pytest.raises(ValueError, match="GHI is not beam plus diffuse"):
        validate_hourly_frame(inconsistent, 2005, config, "inconsistent GHI")

    night = source.copy()
    night_index = night.index[night["sun_height_deg"] < 0.0][0]
    night.loc[night_index, "I_diffuse_horizontal_W_m2"] = 1.0
    night.loc[night_index, "I_solar_W_m2"] = 1.0
    with pytest.raises(ValueError, match="non-zero irradiance below the horizon"):
        validate_hourly_frame(night, 2005, config, "night solar")

    reconstructed = source.copy()
    reconstructed.loc[10, "pvgis_reconstructed"] = True
    with pytest.raises(ValueError, match="reconstructed PVGIS values"):
        validate_hourly_frame(reconstructed, 2005, config, "reconstructed")


def test_member_pair_rejects_changed_fields_and_wrong_monthly_morph(validation_data) -> None:
    config, years, contract = validation_data
    observed = validate_hourly_frame(years[2005], 2005, config, "observed")
    canonical = morph_observed_year(
        observed, "rcp_4_5", contract.frame, config=config
    )
    canonical = validate_hourly_frame(canonical, 2005, config, "canonical member")
    rows = validate_member_pair(
        observed,
        canonical,
        "canonical",
        "rcp_4_5",
        contract.frame,
        config,
    )
    assert len(rows) == 12

    changed_wind = canonical.copy()
    changed_wind.loc[0, "wind_speed_10m_m_s"] += 0.1
    with pytest.raises(ValueError, match="changed invariant observed column"):
        validate_member_pair(
            observed,
            changed_wind,
            "changed-wind",
            "rcp_4_5",
            contract.frame,
            config,
        )

    wrong_temperature = canonical.copy()
    january = wrong_temperature["timestamp_utc"].dt.month == 1
    wrong_temperature.loc[january, "T_out_C"] += 0.1
    with pytest.raises(ValueError, match="temperature morph failed"):
        validate_member_pair(
            observed,
            wrong_temperature,
            "wrong-temperature",
            "rcp_4_5",
            contract.frame,
            config,
        )

    wrong_solar = canonical.copy()
    wrong_solar.loc[january, "I_beam_horizontal_W_m2"] *= 1.01
    wrong_solar["I_solar_W_m2"] = (
        wrong_solar["I_beam_horizontal_W_m2"]
        + wrong_solar["I_diffuse_horizontal_W_m2"]
    )
    with pytest.raises(ValueError, match="beam solar morph failed"):
        validate_member_pair(
            observed,
            wrong_solar,
            "wrong-solar",
            "rcp_4_5",
            contract.frame,
            config,
        )


def test_plausibility_breaches_warn_without_dividing_by_cdd(
    validation_data, caplog
) -> None:
    config, _, _ = validation_data
    observed = {
        "HDD_C_days": 100.0,
        "CDD_C_days": 0.0,
        "annual_solar_kWh_m2": 100.0,
    }
    member = {
        "HDD_C_days": 50.0,
        "CDD_C_days": 151.0,
        "annual_solar_kWh_m2": 120.0,
    }
    logger = logging.getLogger("climate.tests.validation_warnings")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        warnings = assess_plausibility(
            "synthetic-member", observed, member, config, logger=logger
        )
    assert [warning["code"] for warning in warnings] == [
        "hdd_ratio_outside_band",
        "cdd_change_outside_band",
        "annual_solar_ratio_outside_band",
    ]
    assert len(caplog.records) == 3


def test_official_be100_snapshot_and_direct_cordex_degree_days() -> None:
    config = load_config(CONFIG_PATH)
    reference, metadata, csv_path, metadata_path = load_reference_degree_days(config)
    assert len(reference) == 19
    assert reference["year"].tolist() == list(range(2005, 2024))
    assert metadata["dataset_code"] == "nrg_chddr2_a"
    assert metadata["geo_code"] == "BE100"
    assert sha256_file(csv_path) == config["validation"]["degree_days"]["reference"][
        "csv_sha256"
    ]
    assert sha256_file(metadata_path) == config["validation"]["degree_days"][
        "reference"
    ]["metadata_sha256"]

    cordex = calculate_cordex_annual_degree_days(config)
    assert len(cordex) == 88
    assert len(cordex.loc[cordex["role"] == "baseline"]) == 25
    assert (
        cordex.loc[cordex["role"] == "future"]
        .groupby("scenario")
        .size()
        .eq(21)
        .all()
    )


def test_full_validation_reports_and_regeneration_are_deterministic(validation_data) -> None:
    config, _, _ = validation_data
    first = run_validation(config)
    first_hashes = {name: sha256_file(path) for name, path in first.items()}

    members = pd.read_csv(first["member_summary"])
    monthly = pd.read_csv(first["monthly_invariants"])
    report = json.loads(first["report_json"].read_text(encoding="utf-8"))
    observed_reference = pd.read_csv(first["observed_reference_comparison"])
    cordex_annual = pd.read_csv(first["cordex_annual_degree_days"])
    cordex_morph = pd.read_csv(first["cordex_morph_comparison"])
    assert len(members) == 57
    assert members["row_count"].sum() == 499608
    assert members["hard_status"].eq("pass").all()
    assert members["warning_count"].sum() == 0
    assert len(monthly) == 684
    assert monthly["hard_status"].eq("pass").all()
    assert report["status"] == "pass"
    assert report["hard_error_count"] == 0
    assert report["plausibility_warning_count"] == 0
    assert members["HDD_ratio_vs_observed"].min() == pytest.approx(0.7341154614)
    assert members["HDD_ratio_vs_observed"].max() == pytest.approx(0.9048658493)
    assert members["CDD_change_C_days"].min() == pytest.approx(0.0)
    assert members["CDD_change_C_days"].max() == pytest.approx(83.5029332036)
    assert members["annual_solar_ratio_vs_observed"].min() == pytest.approx(1.0611372083)
    assert members["annual_solar_ratio_vs_observed"].max() == pytest.approx(1.0738745860)
    observed_solar = members.drop_duplicates("observed_pvgis_year")[
        "observed_annual_solar_kWh_m2"
    ]
    assert observed_solar.min() == pytest.approx(1040.57899)
    assert observed_solar.max() == pytest.approx(1231.26255)
    assert members["morphed_annual_solar_kWh_m2"].min() == pytest.approx(1105.4469844516)
    assert members["morphed_annual_solar_kWh_m2"].max() == pytest.approx(1317.0392967247)
    assert len(observed_reference) == 19
    statistics = reference_comparison_statistics(observed_reference)
    assert statistics["HDD"]["pearson_correlation"] == pytest.approx(
        0.9869287957
    )
    assert statistics["HDD"]["mean_bias_C_days"] == pytest.approx(
        154.9817763158
    )
    assert statistics["CDD"]["pearson_correlation"] == pytest.approx(
        0.9377052222
    )
    assert statistics["CDD"]["mean_bias_C_days"] == pytest.approx(
        -3.7365131579
    )
    assert len(cordex_annual) == 88
    assert len(cordex_morph) == 3
    expected_changes = {
        "rcp_2_6": (-327.4595043311, -300.3770070175, 17.3987337525, 18.2413800175),
        "rcp_4_5": (-503.0879162358, -452.8474429825, 39.8821223243, 30.0309859825),
        "rcp_8_5": (-624.5217451257, -572.8301938596, 48.1000099401, 38.9224124386),
    }
    for row in cordex_morph.itertuples(index=False):
        expected = expected_changes[row.scenario]
        assert row.cordex_change_HDD_C_days == pytest.approx(expected[0])
        assert row.morphed_paired_change_HDD_C_days == pytest.approx(expected[1])
        assert row.cordex_change_CDD_C_days == pytest.approx(expected[2])
        assert row.morphed_paired_change_CDD_C_days == pytest.approx(expected[3])

    second = run_validation(config)
    second_hashes = {name: sha256_file(path) for name, path in second.items()}
    assert first_hashes == second_hashes


def test_validation_figures_are_complete_and_deterministic(validation_data) -> None:
    config, _, _ = validation_data
    first = build_validation_figures(config)
    first_hashes = {name: sha256_file(path) for name, path in first.items()}
    assert set(first) == {
        "monthly_parameters_png",
        "monthly_parameters_pdf",
        "morph_residuals_png",
        "morph_residuals_pdf",
        "temperature_duration_png",
        "temperature_duration_pdf",
        "provenance",
    }
    assert all(path.stat().st_size > 0 for path in first.values())

    second = build_validation_figures(config)
    second_hashes = {name: sha256_file(path) for name, path in second.items()}
    assert first_hashes == second_hashes
