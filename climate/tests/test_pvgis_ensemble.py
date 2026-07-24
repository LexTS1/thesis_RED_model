from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate.src.build_ensemble import build_ensemble, validate_morphed_member
from climate.src.load_cordex import load_config, sha256_file
from climate.src.load_observed import (
    build_clean_observed,
    clamp_negative_irradiance,
    load_clean_observed,
    load_facade_templates,
    load_observed_weather,
    split_complete_years,
)
from climate.src.morph import load_delta_contract, morph_observed_year, scenario_parameters
from climate.src.transpose_facades import ORIENTATIONS, build_facade_irradiance


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def pvgis_data():
    config = load_config(CONFIG_PATH)
    build_clean_observed(config)
    observed, metadata = load_clean_observed(config)
    years = split_complete_years(observed)
    deltas = load_delta_contract(config)
    return config, observed, metadata, years, deltas


def test_observed_pvgis_is_clean_complete_and_audited(pvgis_data) -> None:
    config, observed, metadata, years, _ = pvgis_data
    assert len(observed) == 166536
    assert sorted(years) == list(range(2005, 2024))
    assert {year: len(frame) for year, frame in years.items() if len(frame) == 8784} == {
        2008: 8784,
        2012: 8784,
        2016: 8784,
        2020: 8784,
    }
    assert all(len(frame) in {8760, 8784} for frame in years.values())
    assert observed["timestamp_utc"].iat[0].isoformat() == "2005-01-01T00:00:00+00:00"
    assert observed["timestamp_utc"].iat[-1].isoformat() == "2023-12-31T23:00:00+00:00"
    assert observed["T_out_C"].min() == pytest.approx(-13.37)
    assert observed["T_out_C"].max() == pytest.approx(37.49)
    assert observed["I_solar_W_m2"].min() == 0.0
    assert observed["I_solar_W_m2"].max() == pytest.approx(944.01)
    assert observed["wind_speed_10m_m_s"].min() == 0.0
    assert observed["wind_speed_10m_m_s"].max() == pytest.approx(14.28)
    assert not observed["pvgis_reconstructed"].any()
    assert not observed.isna().any().any()
    assert not observed["timestamp_utc"].duplicated().any()
    np.testing.assert_allclose(
        observed["I_solar_W_m2"],
        observed["I_beam_horizontal_W_m2"] + observed["I_diffuse_horizontal_W_m2"],
        rtol=0.0,
        atol=2e-14,
    )
    assert (observed.loc[observed["sun_height_deg"] < 0.0, "I_solar_W_m2"] == 0.0).all()
    horizon = observed.loc[observed["sun_height_deg"] == 0.0, "I_solar_W_m2"]
    assert horizon.max() == pytest.approx(10.0)
    assert metadata["source"]["sha256"] == config["observed_weather"]["horizontal"]["csv_sha256"]
    assert metadata["timestamp"]["source_minute"] == 10
    assert metadata["timestamp"]["timezone"] == "UTC"


def test_raw_observed_header_and_hash_are_verified(pvgis_data) -> None:
    config, observed, _, _, _ = pvgis_data
    direct, plane = load_observed_weather(config)
    pd.testing.assert_frame_equal(direct, observed)
    assert plane.header_metadata == {
        "latitude": 50.83,
        "longitude": 4.35,
        "elevation_m": 61.0,
        "radiation_database": "PVGIS-SARAH3",
        "slope_deg": 0.0,
        "azimuth_pvgis_deg": 0.0,
    }
    assert plane.source_sha256 == "a12fa6560bcf2e877d4228ab43636dc4ed92434e67aa40611bbd8a04c1760aa6"


def test_negative_irradiance_clamp_warns_before_clipping(caplog) -> None:
    logger = logging.getLogger("climate.tests.negative_irradiance")
    source = pd.DataFrame(
        {
            "beam": [1.0, -0.25, 2.0],
            "diffuse": [0.0, 0.5, -1.5],
        }
    )
    with caplog.at_level(logging.WARNING, logger=logger.name):
        result = clamp_negative_irradiance(
            source, ["beam", "diffuse"], "synthetic", logger=logger
        )
    assert result["beam"].tolist() == [1.0, 0.0, 2.0]
    assert result["diffuse"].tolist() == [0.0, 0.5, 0.0]
    assert len(caplog.records) == 2
    assert "source=synthetic column=beam count=1 minimum=-0.2500000000" in caplog.records[0].message
    assert "source=synthetic column=diffuse count=1 minimum=-1.5000000000" in caplog.records[1].message


def test_morph_one_complete_leap_year_and_monthly_invariants(pvgis_data) -> None:
    config, _, _, years, contract = pvgis_data
    observed = years[2020]
    morphed = morph_observed_year(observed, "rcp_4_5", contract.frame, config=config)
    assert len(morphed) == 8784
    validate_morphed_member(observed, morphed, "rcp_4_5", contract.frame)
    for column in (
        "timestamp_utc",
        "wind_speed_10m_m_s",
        "sun_height_deg",
        "pvgis_reconstructed",
    ):
        pd.testing.assert_series_equal(morphed[column], observed[column])
    assert not morphed.isna().any().any()
    assert (morphed[["I_beam_horizontal_W_m2", "I_diffuse_horizontal_W_m2", "I_solar_W_m2"]] >= 0).all().all()


def test_on_demand_facades_use_pvgis_components_and_same_alpha(pvgis_data) -> None:
    config, _, _, years, contract = pvgis_data
    templates = load_facade_templates(config)
    member = morph_observed_year(years[2020], "rcp_4_5", contract.frame, config=config)
    facades = build_facade_irradiance(member, "rcp_4_5", contract.frame, templates)
    assert list(facades.columns) == ["timestamp_utc", *[f"I_{o}_W_m2" for o in ORIENTATIONS]]
    assert len(facades) == 8784
    parameters = scenario_parameters(contract.frame, "rcp_4_5")
    alpha = facades["timestamp_utc"].dt.month.map(parameters["alpha_solar_applied"]).to_numpy()
    for orientation in ORIENTATIONS:
        template = templates[orientation].frame
        selected = template.loc[template["timestamp_utc"].dt.year == 2020]
        observed_total = selected[
            ["I_beam_W_m2", "I_diffuse_W_m2", "I_reflected_W_m2"]
        ].sum(axis=1).to_numpy()
        np.testing.assert_allclose(
            facades[f"I_{orientation}_W_m2"], observed_total * alpha, rtol=0.0, atol=1e-12
        )
        assert templates[orientation].header_metadata["slope_deg"] == 90.0
    assert (facades.drop(columns="timestamp_utc") >= 0.0).all().all()
    assert "I_south_W_m2" not in member.columns
    assert facades.attrs["scenario"] == "rcp_4_5"
    assert facades.attrs["observed_year"] == 2020
    assert set(facades.attrs["source_sha256"]) == set(ORIENTATIONS)


def test_materialized_ensemble_contract_and_all_members(pvgis_data) -> None:
    config, observed, _, years, contract = pvgis_data
    output_root = Path(config["_base_dir"]) / config["observed_weather"]["ensemble"]["directory"]
    manifest = pd.read_csv(output_root / "ensemble_2050_manifest.csv")
    payload = json.loads((output_root / "ensemble_2050_manifest.json").read_text())
    assert len(manifest) == 57
    assert manifest["row_count"].sum() == 499608
    assert manifest.groupby("scenario")["observed_pvgis_year"].nunique().eq(19).all()
    assert sorted(manifest.loc[manifest["is_leap_year"], "observed_pvgis_year"].unique()) == [
        2008,
        2012,
        2016,
        2020,
    ]
    assert payload["member_count"] == 57
    assert payload["total_member_hours"] == 499608
    assert payload["facade_forcing"]["materialized_in_members"] is False
    assert payload["morph_contract"]["clipped_month_count"] == 0
    assert payload["morph_contract"]["alpha_minimum"] == pytest.approx(0.9997804193)
    assert payload["morph_contract"]["alpha_maximum"] == pytest.approx(1.1104768063)
    assert payload["ensemble_ranges"]["T_out_C"]["minimum"] == pytest.approx(-12.8788990439)
    assert payload["ensemble_ranges"]["T_out_C"]["maximum"] == pytest.approx(39.8611552783)
    assert payload["ensemble_ranges"]["I_solar_W_m2"]["maximum"] == pytest.approx(1037.196441852263)

    expected_columns = {
        "timestamp_utc",
        "T_out_C",
        "I_beam_horizontal_W_m2",
        "I_diffuse_horizontal_W_m2",
        "I_solar_W_m2",
        "wind_speed_10m_m_s",
        "sun_height_deg",
        "pvgis_reconstructed",
    }
    for row in manifest.itertuples(index=False):
        member_path = Path(config["_base_dir"]) / row.member_path
        metadata_path = Path(config["_base_dir"]) / row.metadata_path
        assert sha256_file(member_path) == row.member_sha256
        assert sha256_file(metadata_path) == row.metadata_sha256
        member = pd.read_csv(member_path)
        assert set(member.columns) == expected_columns
        assert not any(column.startswith("I_south") for column in member.columns)
        assert len(member) == row.row_count
        timestamps = pd.to_datetime(member["timestamp_utc"], utc=True)
        assert timestamps.duplicated().sum() == 0
        assert timestamps.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        assert not member.isna().any().any()
        assert (member[["I_beam_horizontal_W_m2", "I_diffuse_horizontal_W_m2", "I_solar_W_m2"]] >= 0).all().all()
        np.testing.assert_allclose(
            member["I_solar_W_m2"],
            member["I_beam_horizontal_W_m2"] + member["I_diffuse_horizontal_W_m2"],
            rtol=0.0,
            atol=1.1e-10,
        )
        observed_year = years[int(row.observed_pvgis_year)]
        reconstructed = member.copy()
        reconstructed["timestamp_utc"] = timestamps
        validate_morphed_member(observed_year, reconstructed, row.scenario, contract.frame)


def test_ensemble_regeneration_is_byte_deterministic(pvgis_data) -> None:
    config, _, _, _, _ = pvgis_data
    output_root = Path(config["_base_dir"]) / config["observed_weather"]["ensemble"]["directory"]
    tracked_before = {
        str(path.relative_to(output_root)): sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    }
    build_ensemble(config)
    tracked_after = {
        str(path.relative_to(output_root)): sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    }
    assert tracked_before == tracked_after
