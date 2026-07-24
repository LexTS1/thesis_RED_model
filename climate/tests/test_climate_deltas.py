from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from climate.src.deltas import (
    apply_solar_alpha_safety,
    build,
    compute_monthly_climatologies,
    compute_monthly_deltas,
    compute_year_month_variability,
)
from climate.src.load_cordex import load_all_sources, load_config, sha256_file


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def climate_data():
    config = load_config(CONFIG_PATH)
    sources = load_all_sources(config)
    climatologies = compute_monthly_climatologies(sources, config)
    deltas = compute_monthly_deltas(climatologies, config)
    variability = compute_year_month_variability(sources, climatologies, config)
    return config, sources, climatologies, deltas, variability


def test_canonical_sources_are_complete_and_hash_verified(climate_data) -> None:
    config, sources, _, _, _ = climate_data
    expected = {
        "historical": (9131, 25, "1981-01-01", "2005-12-31"),
        "rcp_2_6": (7670, 21, "2050-01-01", "2070-12-31"),
        "rcp_4_5": (7670, 21, "2050-01-01", "2070-12-31"),
        "rcp_8_5": (7670, 21, "2050-01-01", "2070-12-31"),
    }
    for key, (rows, years, start, end) in expected.items():
        source = sources[key]
        assert len(source.frame) == rows
        assert source.frame["year"].nunique() == years
        assert source.frame["timestamp"].min().date().isoformat() == start
        assert source.frame["timestamp"].max().date().isoformat() == end
        assert not source.frame["timestamp"].duplicated().any()
        assert not source.frame[["T_out_C", "I_solar_W_m2"]].isna().any().any()
        assert source.csv_sha256 == config["sources"][key]["csv_sha256"]
        assert source.metadata_sha256 == config["sources"][key]["metadata_sha256"]


def test_climatology_and_delta_shapes(climate_data) -> None:
    _, _, climatologies, deltas, variability = climate_data
    assert len(climatologies) == 48
    assert len(deltas) == 36
    assert len(variability) == 756
    assert climatologies.groupby("source_key")["month"].nunique().eq(12).all()
    assert deltas.groupby("scenario")["month"].nunique().eq(12).all()
    assert variability.groupby(["scenario", "climate_year"])["month"].nunique().eq(12).all()
    assert not climatologies.duplicated(["source_key", "month"]).any()
    assert not deltas.duplicated(["scenario", "month"]).any()
    assert not variability.duplicated(["scenario", "climate_year", "month"]).any()


def test_monthly_delta_identities_and_real_alpha_range(climate_data) -> None:
    _, _, _, deltas, _ = climate_data
    np.testing.assert_allclose(
        deltas["delta_T_C"], deltas["T_future_C"] - deltas["T_hist_C"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        deltas["alpha_solar_raw"],
        deltas["I_future_W_m2"] / deltas["I_hist_W_m2"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        deltas["alpha_solar_applied"], deltas["alpha_solar_raw"], rtol=0, atol=0
    )
    assert not deltas["alpha_was_clipped"].any()

    minimum = deltas.loc[deltas["alpha_solar_raw"].idxmin()]
    maximum = deltas.loc[deltas["alpha_solar_raw"].idxmax()]
    assert (minimum["scenario"], int(minimum["month"])) == ("rcp_4_5", 3)
    assert (maximum["scenario"], int(maximum["month"])) == ("rcp_4_5", 6)
    assert float(minimum["alpha_solar_raw"]) == pytest.approx(0.999780419330248)
    assert float(maximum["alpha_solar_raw"]) == pytest.approx(1.1104768062512007)


def test_solar_safety_clamp_warns_only_when_activated(caplog) -> None:
    logger = logging.getLogger("climate.tests.alpha")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        applied, clipped = apply_solar_alpha_safety(
            1.05, "synthetic", 1, 0.7, 1.3, logger=logger
        )
    assert applied == 1.05
    assert clipped is False
    assert not caplog.records

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=logger.name):
        high, high_clipped = apply_solar_alpha_safety(
            1.4, "synthetic_high", 6, 0.7, 1.3, logger=logger
        )
        low, low_clipped = apply_solar_alpha_safety(
            0.6, "synthetic_low", 12, 0.7, 1.3, logger=logger
        )
    assert high == 1.3 and high_clipped
    assert low == 0.7 and low_clipped
    assert len(caplog.records) == 2
    assert "scenario=synthetic_high month=06" in caplog.records[0].message
    assert "alpha_raw=1.4000000000 alpha_applied=1.3000000000" in caplog.records[0].message
    assert "scenario=synthetic_low month=12" in caplog.records[1].message


def test_year_month_variability_decomposition(climate_data) -> None:
    _, _, _, deltas, variability = climate_data
    joined = variability.merge(
        deltas[["scenario", "month", "delta_T_C", "alpha_solar_raw"]],
        on=["scenario", "month"],
        validate="many_to_one",
    )
    np.testing.assert_allclose(
        joined["delta_T_vs_historical_C"],
        joined["delta_T_C"] + joined["T_anomaly_from_future_climatology_C"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        joined["alpha_solar_vs_historical"],
        joined["alpha_solar_raw"] * joined["solar_factor_from_future_climatology"],
        rtol=0,
        atol=1e-12,
    )

    for _, group in variability.groupby(["scenario", "month"]):
        weighted_temperature_anomaly = np.average(
            group["T_anomaly_from_future_climatology_C"], weights=group["n_days"]
        )
        assert weighted_temperature_anomaly == pytest.approx(0.0, abs=2e-14)


def test_day_weighted_annual_acceptance_anchors(climate_data) -> None:
    _, sources, _, _, _ = climate_data
    historical = sources["historical"].frame
    expected = {
        "rcp_2_6": (1.0692174572229511, 1.0617115327501625),
        "rcp_4_5": (1.6336106133914647, 1.0704623389369834),
        "rcp_8_5": (2.063189861533832, 1.0669208700076152),
    }
    for key, (delta_t, alpha) in expected.items():
        future = sources[key].frame
        assert future["T_out_C"].mean() - historical["T_out_C"].mean() == pytest.approx(delta_t)
        assert future["I_solar_W_m2"].mean() / historical["I_solar_W_m2"].mean() == pytest.approx(alpha)


def test_build_outputs_are_byte_deterministic(climate_data) -> None:
    config, _, _, _, _ = climate_data
    first_paths = build(config)
    first_hashes = {name: sha256_file(path) for name, path in first_paths.items()}
    second_paths = build(config)
    second_hashes = {name: sha256_file(path) for name, path in second_paths.items()}
    assert first_hashes == second_hashes

