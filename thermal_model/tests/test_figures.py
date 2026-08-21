from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from thermal_model.plot_figures import create_figures
from thermal_model.plot_reference_profile import create_reference_profile


EXPECTED_FIGURES = {
    "fig_tabula_model_agreement",
    "fig_heating_intensity_existing",
    "fig_heating_intensity_standard",
    "fig_heating_intensity_advanced",
    "fig_heating_sensitivity",
    "fig_cooling_sensitivity",
}


def test_each_thermal_plot_is_exported_as_an_independent_png_and_pdf(
    tmp_path: Path,
) -> None:
    provenance = create_figures(tmp_path)

    assert set(provenance["figures"]) == EXPECTED_FIGURES
    assert provenance["verification_status"] == "PASS"
    assert provenance["validation_status"] == "PASS"
    for basename in EXPECTED_FIGURES:
        png_path = tmp_path / f"{basename}.png"
        pdf_path = tmp_path / f"{basename}.pdf"
        assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert pdf_path.read_bytes().startswith(b"%PDF-")
        with Image.open(png_path) as image:
            assert image.width >= 1800
            assert image.height >= 1300

    persisted = json.loads(
        (tmp_path / "figure_provenance.json").read_text(encoding="utf-8")
    )
    assert persisted == provenance


def test_reference_profile_is_an_independent_energy_conserving_figure(
    tmp_path: Path,
) -> None:
    provenance = create_reference_profile(tmp_path)

    assert provenance["case"]["dwelling_category"] == "Semi-detached house"
    assert len(provenance["case"]["archetype_ids"]) == 5
    assert abs(provenance["case"]["national_share_all_dwellings"] - 0.1775) < 1e-4
    assert provenance["case"]["state_id"] == "TABULA_existing"
    assert provenance["case"]["reference_year"] == 2015
    assert "not a validation target or acceptance criterion" in provenance[
        "interpretation"
    ]
    for suffix in ("png", "pdf"):
        path = tmp_path / provenance["outputs"][suffix]
        assert path.is_file()
    with Image.open(tmp_path / provenance["outputs"]["png"]) as image:
        assert image.width >= 1800
        assert image.height >= 1100

    persisted = json.loads(
        (
            tmp_path / "fig_reference_year_daily_demand.provenance.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted == provenance
