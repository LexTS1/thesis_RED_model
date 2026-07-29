from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from pandas.errors import MergeError

from stock_joint_distribution import (
    JOINT_DISTRIBUTION_METHOD,
    build_regional_joint_distribution,
    load_cadastral,
    load_hc37,
    regional_stock_totals,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_GEOMETRY_FILE = (
    DATA_DIR
    / "inputs"
    / "physical"
    / "geometry_per_dwelling_type_and_period.csv"
)
DEFAULT_AIRTIGHTNESS_FILE = (
    DATA_DIR / "inputs" / "physical" / "in_and_exfiltration_rates.csv"
)
DEFAULT_U_VALUES_FILE = (
    DATA_DIR / "inputs" / "physical" / "U_values_per_construction_period.csv"
)
DEFAULT_STATE_PACKAGES_FILE = (
    DATA_DIR
    / "assumptions"
    / "renovation"
    / "renovation_physical_state_packages_TABULA.csv"
)
DEFAULT_CADASTRAL_STOCK_FILE = (
    DATA_DIR / "raw" / "statbel" / "building_stock_open_data_2025.xlsx"
)
DEFAULT_JOINT_CENSUS_FILE = (
    DATA_DIR / "raw" / "statbel" / "TF_CENSUS_2021_HC37_1.xlsx"
)
DEFAULT_BASE_OUTPUT = DATA_DIR / "matrices" / "national" / "base_physical_archetype_matrix.csv"
DEFAULT_STOCK_OUTPUT = (
    DATA_DIR / "matrices" / "national" / "stock_weighted_archetype_matrix.csv"
)

EXPECTED_ARCHETYPE_COUNT = 25
WEIGHTING_METHOD = JOINT_DISTRIBUTION_METHOD
INFILTRATION_N_FACTOR = 20.0
AIR_DENSITY_KG_M3 = 1.2
AIR_SPECIFIC_HEAT_J_KG_K = 1005.0

GEOMETRY_PDF_COLUMNS = [
    "floor_surface_area_m2",
    "protected_volume_m3",
    "total_building_envelope_area_m2",
    "roof_area_m2",
    "exterior_wall_area_m2",
    "exterior_wall_bordering_unheated_neighboring_spaces_m2",
    "floor_on_soil_m2",
    "floor_bordering_unheated_neighboring_spaces_m2",
    "doors_area_m2",
    "windows_north_m2",
    "windows_east_m2",
    "windows_south_m2",
    "windows_west_m2",
]

V50_COLUMNS_BY_DWELLING_TYPE = {
    "Detached house": "detached_house_v50_m3_h_m2",
    "Semi-detached house": "semi_detached_house_v50_m3_h_m2",
    "Terraced house": "terraced_house_v50_m3_h_m2",
    "Apartment, enclosed": "apartment_enclosed_v50_m3_h_m2",
    "Apartment, exposed": "apartment_exposed_v50_m3_h_m2",
}

U_VALUE_FIELDS = {
    "facade": ("tabula_facade_age_class", "U_facade_W_m2K"),
    "roof": ("tabula_roof_age_class", "U_roof_W_m2K"),
    "floor": ("tabula_floor_age_class", "U_floor_W_m2K"),
    "windows": ("tabula_window_age_class", "U_window_W_m2K"),
    "doors": ("tabula_door_age_class", "U_door_W_m2K"),
}

MODEL_ASSUMPTIONS = [
    "R1-R3 age profiles use official Statbel 2025 cadastral building counts "
    "jointly classified by building type and construction period, scaled to "
    "the corresponding 2025 T8 dwelling totals.",
    "The R4 age profile uses official Statbel Census 2021 HC37 conventional "
    "dwellings in residential buildings with 2 or 3+ dwellings, scaled to the "
    "regional 2025 T8 R4 dwelling total.",
    "Statbel R4 apartment-building dwellings are split equally between the "
    "enclosed and exposed apartment archetypes because the input has no subtype split.",
    "HC37 supplies exact 2001-2005 and 2006-2010 counts. For cadastral house "
    "profiles only, the 1982-1991 and 2002-2011 bins are allocated uniformly "
    "across years at the TABULA cut-offs.",
    "The residential-archetype scope is R1-R4. R5 commerce houses and R6 other "
    "buildings are reported as excluded residual dwellings rather than assigned "
    "to a TABULA archetype without evidence.",
    "Dwellings with unknown construction period are distributed pro rata by "
    "normalising the known-period counts to 1.",
    "regional_share is empty in the national matrix; regional shares are "
    "reported in the separate regional stock-weighted matrix.",
    "TABULA v50 is converted to an annual-average infiltration airflow proxy "
    "with q50=v50*A_env and Vdot_inf=q50/20. The fixed n-factor 20 is used as "
    "a transparent screening assumption.",
]


def require_columns(data: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = columns - set(data.columns)
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {sorted(missing)}")


def require_unique(data: pd.DataFrame, keys: list[str], table_name: str) -> None:
    duplicate_mask = data.duplicated(keys, keep=False)
    if duplicate_mask.any():
        duplicate_keys = data.loc[duplicate_mask, keys].drop_duplicates().to_dict("records")
        raise ValueError(
            f"{table_name} contains duplicate rows for keys {keys}: {duplicate_keys}"
        )


def require_no_missing(
    data: pd.DataFrame, columns: list[str], table_name: str
) -> None:
    missing_counts = data[columns].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(
            f"{column}={count}" for column, count in missing_counts.items()
        )
        raise ValueError(f"{table_name} has missing required values: {details}")


def parse_pdf_number(value: str) -> float:
    return float(value.strip().replace(" ", "").replace(",", "."))


def find_last_caption_page(page_texts: list[str], caption: str) -> int:
    matches = [index for index, text in enumerate(page_texts) if caption in text]
    if not matches:
        raise ValueError(f"TABULA PDF does not contain the expected caption: {caption}")
    return matches[-1]


def compare_value(
    errors: list[str], label: str, csv_value: float, pdf_value: float
) -> None:
    if not math.isclose(float(csv_value), float(pdf_value), rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"{label}: CSV={csv_value}, PDF={pdf_value}")


def verify_geometry_against_pdf(
    pdf, page_texts: list[str], geometry: pd.DataFrame
) -> list[str]:
    caption_page = find_last_caption_page(
        page_texts, "Table 19: Geometrical data for the national housing typology"
    )
    pdf_rows: list[list[str | None]] = []
    for page_index in (caption_page - 1, caption_page):
        for table in pdf.pages[page_index].extract_tables():
            pdf_rows.extend(table)

    parsed_rows: dict[int, list[float]] = {}
    for row in pdf_rows:
        if not row or not row[0]:
            continue
        match = re.fullmatch(r"Type\s*(\d+)", str(row[0]).strip())
        if not match:
            continue
        type_number = int(match.group(1))
        numeric_values = [parse_pdf_number(str(value)) for value in row[2:] if value is not None]
        if len(numeric_values) != len(GEOMETRY_PDF_COLUMNS):
            raise ValueError(
                f"Could not parse all Table 19 geometry fields for Type{type_number}: "
                f"expected {len(GEOMETRY_PDF_COLUMNS)}, found {len(numeric_values)}"
            )
        parsed_rows[type_number] = numeric_values

    if len(parsed_rows) != EXPECTED_ARCHETYPE_COUNT:
        raise ValueError(
            "Could not parse all Table 19 archetypes from the TABULA PDF: "
            f"expected {EXPECTED_ARCHETYPE_COUNT}, found {len(parsed_rows)}"
        )

    errors: list[str] = []
    geometry_by_number = geometry.set_index("tabula_type_number")
    for type_number, pdf_values in parsed_rows.items():
        if type_number not in geometry_by_number.index:
            errors.append(f"Table 19 Type{type_number} is absent from the geometry CSV")
            continue
        csv_row = geometry_by_number.loc[type_number]
        for field, pdf_value in zip(GEOMETRY_PDF_COLUMNS, pdf_values):
            compare_value(
                errors,
                f"Table 19 Type{type_number} {field}",
                csv_row[field],
                pdf_value,
            )

    window_fields = [
        "windows_north_m2",
        "windows_east_m2",
        "windows_south_m2",
        "windows_west_m2",
    ]
    expected_window_totals = geometry[window_fields].sum(axis=1)
    for row_index, expected_total in expected_window_totals.items():
        compare_value(
            errors,
            f"Derived window total for Type{geometry.loc[row_index, 'tabula_type_number']}",
            geometry.loc[row_index, "windows_total_m2"],
            expected_total,
        )
    return errors


def verify_v50_against_pdf(
    page_texts: list[str], airtightness: pd.DataFrame
) -> list[str]:
    caption_page = find_last_caption_page(
        page_texts, "Table 9: In/exfiltration rates at 50 Pa"
    )
    table_text = "\n".join(page_texts[caption_page - 1 : caption_page + 1])
    table_text = table_text.replace("sce-\nnario", "scenario").replace(
        "upgrade\n2,5", "upgrade scenario\n2,5"
    )
    errors: list[str] = []

    for age_class in airtightness["tabula_v50_age_class"].drop_duplicates():
        pattern = re.compile(
            rf"^{re.escape(str(age_class))}\s+"
            r"(\d+(?:,\d+)?)\s+(\d+(?:,\d+)?)\s+(\d+(?:,\d+)?)\s+"
            r"(\d+(?:,\d+)?)\s+(\d+(?:,\d+)?)\s*$",
            re.MULTILINE,
        )
        match = pattern.search(table_text)
        if not match:
            raise ValueError(f"Could not parse Table 9 row for age class {age_class!r}")
        pdf_values = [parse_pdf_number(value) for value in match.groups()]
        matching_rows = airtightness[airtightness["tabula_v50_age_class"] == age_class]
        for _, csv_row in matching_rows.iterrows():
            for (dwelling_type, field), pdf_value in zip(
                V50_COLUMNS_BY_DWELLING_TYPE.items(), pdf_values
            ):
                compare_value(
                    errors,
                    f"Table 9 {csv_row['construction_period']} {dwelling_type}",
                    csv_row[field],
                    pdf_value,
                )
    return errors


def extract_u_value_from_section(section: str, age_class: str) -> float:
    match = re.search(
        rf"^{re.escape(age_class)}[^\n]*?(\d+(?:,\d+)?)\s*$",
        section,
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Could not parse Table 10 U-value for age class {age_class!r}")
    return parse_pdf_number(match.group(1))


def verify_u_values_against_pdf(
    page_texts: list[str], u_values: pd.DataFrame
) -> list[str]:
    caption_page = find_last_caption_page(
        page_texts, "Table 10: Belgian TABULA sub-typology of construction elements"
    )
    table_text = "\n".join(page_texts[caption_page - 1 : caption_page + 1])
    section_markers = {
        "facade": "Age class/State Facade construction elements",
        "roof": "Age class/State Roof construction elements",
        "floor": "Age class/State Floor construction elements",
        "windows": "Age class/State Windows",
        "doors": "Age class/State Doors",
    }
    ordered_sections = list(section_markers)
    sections: dict[str, str] = {}
    for index, element in enumerate(ordered_sections):
        start_marker = section_markers[element]
        end_marker = (
            section_markers[ordered_sections[index + 1]]
            if index + 1 < len(ordered_sections)
            else "Table 10: Belgian TABULA sub-typology of construction elements"
        )
        start = table_text.find(start_marker)
        end = table_text.find(end_marker, start + len(start_marker))
        if start < 0 or end < 0:
            raise ValueError(f"Could not locate the Table 10 {element} section")
        sections[element] = table_text[start:end]

    errors: list[str] = []
    for element, (age_field, value_field) in U_VALUE_FIELDS.items():
        for _, csv_row in u_values.iterrows():
            pdf_value = extract_u_value_from_section(
                sections[element], str(csv_row[age_field])
            )
            compare_value(
                errors,
                f"Table 10 {csv_row['construction_period']} {value_field}",
                csv_row[value_field],
                pdf_value,
            )
    return errors


def verify_state_packages_against_pdf(
    page_texts: list[str], state_packages: pd.DataFrame
) -> list[str]:
    required = {
        "state_id",
        "physical_parameter_mode",
        "U_facade_W_m2K",
        "U_roof_W_m2K",
        "U_floor_W_m2K",
        "U_window_W_m2K",
        "U_door_W_m2K",
        "v50_m3_h_m2",
        "hrv_eta",
        "summer_bypass",
    }
    require_columns(
        state_packages, required, "renovation_physical_state_packages_TABULA.csv"
    )
    require_unique(
        state_packages,
        ["state_id"],
        "renovation_physical_state_packages_TABULA.csv",
    )
    expected_states = {
        "TABULA_standard_B_proxy": {
            "U_facade_W_m2K": 0.4,
            "U_roof_W_m2K": 0.3,
            "U_floor_W_m2K": 0.4,
            "U_window_W_m2K": 2.0,
            "U_door_W_m2K": 2.9,
            "v50_m3_h_m2": 6.0,
            "hrv_eta": 0.0,
        },
        "TABULA_advanced_A_proxy": {
            "U_facade_W_m2K": 0.25,
            "U_roof_W_m2K": 0.15,
            "U_floor_W_m2K": 0.25,
            "U_window_W_m2K": 1.6,
            "U_door_W_m2K": 1.6,
            "v50_m3_h_m2": 2.5,
            "hrv_eta": 0.8,
        },
    }
    errors: list[str] = []
    indexed = state_packages.set_index("state_id")
    for state_id, expected in expected_states.items():
        if state_id not in indexed.index:
            errors.append(f"Renovation package is missing {state_id}")
            continue
        row = indexed.loc[state_id]
        for field, expected_value in expected.items():
            compare_value(errors, f"{state_id} {field}", row[field], expected_value)

    source_text = "\n".join(page_texts[54:61])
    required_source_fragments = [
        "U facade: 0.4 W/m²K",
        "U floor: 0.4 W/m²K",
        "U roof: 0.3 W/m²K",
        "U window: 2 W/m²K",
        "U door: 2.9 W/m²K",
        "U facade: 0.25 W/m²K",
        "U floor: 0.25 W/m²K",
        "U roof: 0.15 W/m²K",
        "U window: 1.6 W/m²K",
        "U door: 1.6 W/m²K",
        "heat recuperation (η 0,8) with by-pass",
    ]
    for fragment in required_source_fragments:
        if fragment not in source_text:
            errors.append(f"TABULA source fragment could not be located: {fragment}")

    caption_page = find_last_caption_page(
        page_texts, "Table 9: In/exfiltration rates at 50 Pa"
    )
    table_text = "\n".join(page_texts[caption_page - 1 : caption_page + 1])
    table_text = re.sub(
        r"EPB 2010 upgrade sce-\n(6\s+6\s+6\s+6\s+6)\nnario",
        r"EPB 2010 upgrade scenario\n\1",
        table_text,
    ).replace("Low Energy upgrade\n2,5", "Low Energy upgrade scenario\n2,5")
    for source_label, expected_value in (
        ("EPB 2010 upgrade scenario", 6.0),
        ("Low Energy upgrade scenario", 2.5),
    ):
        match = re.search(
            rf"^{re.escape(source_label)}\s+"
            r"(\d+(?:,\d+)?)\s+(\d+(?:,\d+)?)\s+(\d+(?:,\d+)?)\s+"
            r"(\d+(?:,\d+)?)\s+(\d+(?:,\d+)?)\s*$",
            table_text,
            re.MULTILINE,
        )
        if not match:
            errors.append(f"TABULA Table 9 row could not be parsed: {source_label}")
            continue
        for index, value in enumerate(match.groups(), start=1):
            compare_value(
                errors,
                f"Table 9 {source_label} dwelling type {index}",
                parse_pdf_number(value),
                expected_value,
            )
    return errors


def verify_physical_csvs_against_pdf(
    pdf_path: Path,
    geometry: pd.DataFrame,
    airtightness: pd.DataFrame,
    u_values: pd.DataFrame,
    state_packages: pd.DataFrame,
) -> None:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is required to verify the physical CSVs against TABULA"
        ) from exc

    if not pdf_path.is_file():
        raise FileNotFoundError(f"TABULA PDF not found: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        page_texts = [
            page.extract_text(x_tolerance=2, y_tolerance=3) or "" for page in pdf.pages
        ]
        errors = verify_geometry_against_pdf(pdf, page_texts, geometry)
        errors.extend(verify_v50_against_pdf(page_texts, airtightness))
        errors.extend(verify_u_values_against_pdf(page_texts, u_values))
        errors.extend(verify_state_packages_against_pdf(page_texts, state_packages))

    if errors:
        formatted_errors = "\n- ".join(errors)
        raise ValueError(
            "Physical CSV values conflict with the TABULA/VITO PDF; no outputs were "
            f"written:\n- {formatted_errors}"
        )


def add_heat_loss_fields(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["q50_m3_h"] = (
        result["v50_m3_h_m2"] * result["total_building_envelope_area_m2"]
    )
    result["n50_h_1"] = result["q50_m3_h"] / result["protected_volume_m3"]
    result["infiltration_n_factor"] = INFILTRATION_N_FACTOR
    result["infiltration_airflow_normal_m3_h"] = (
        result["q50_m3_h"] / INFILTRATION_N_FACTOR
    )
    result["infiltration_ach_normal_h_1"] = (
        result["n50_h_1"] / INFILTRATION_N_FACTOR
    )
    result["transmission_heat_loss_H_tr_W_K"] = (
        result["U_roof_W_m2K"] * result["roof_area_m2"]
        + result["U_facade_W_m2K"]
        * (
            result["exterior_wall_area_m2"]
            + result[
                "exterior_wall_bordering_unheated_neighboring_spaces_m2"
            ]
        )
        + result["U_floor_W_m2K"]
        * (
            result["floor_on_soil_m2"]
            + result["floor_bordering_unheated_neighboring_spaces_m2"]
        )
        + result["U_window_W_m2K"] * result["windows_total_m2"]
        + result["U_door_W_m2K"] * result["doors_area_m2"]
    )
    result["infiltration_heat_loss_H_inf_W_K"] = (
        AIR_DENSITY_KG_M3
        * AIR_SPECIFIC_HEAT_J_KG_K
        * result["infiltration_airflow_normal_m3_h"]
        / 3600.0
    )
    result["specific_heat_loss_z_W_m2K"] = (
        result["transmission_heat_loss_H_tr_W_K"]
        + result["infiltration_heat_loss_H_inf_W_K"]
    ) / result["floor_surface_area_m2"]
    return result


def reshape_airtightness(airtightness: pd.DataFrame) -> pd.DataFrame:
    required = {
        "construction_period",
        "tabula_v50_age_class",
        *V50_COLUMNS_BY_DWELLING_TYPE.values(),
    }
    require_columns(airtightness, required, "in_and_exfiltration_rates.csv")
    require_unique(
        airtightness, ["construction_period"], "in_and_exfiltration_rates.csv"
    )

    rows = []
    for dwelling_type, source_field in V50_COLUMNS_BY_DWELLING_TYPE.items():
        subset = airtightness[["construction_period", source_field]].copy()
        subset["dwelling_type"] = dwelling_type
        subset = subset.rename(columns={source_field: "v50_m3_h_m2"})
        rows.append(subset)
    result = pd.concat(rows, ignore_index=True)
    result = result[["dwelling_type", "construction_period", "v50_m3_h_m2"]]
    require_unique(
        result,
        ["dwelling_type", "construction_period"],
        "reshaped in_and_exfiltration_rates.csv",
    )
    return result


def create_base_matrix(
    geometry: pd.DataFrame,
    airtightness: pd.DataFrame,
    u_values: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    geometry_keys = ["dwelling_type", "construction_period"]
    require_columns(
        geometry,
        {"tabula_type_number", "windows_total_m2", *geometry_keys, *GEOMETRY_PDF_COLUMNS},
        "geometry_per_dwelling_type_and_period.csv",
    )
    require_no_missing(
        geometry,
        ["tabula_type_number", *geometry_keys],
        "geometry_per_dwelling_type_and_period.csv",
    )
    require_unique(geometry, geometry_keys, "geometry_per_dwelling_type_and_period.csv")
    require_unique(
        geometry, ["tabula_type_number"], "geometry_per_dwelling_type_and_period.csv"
    )
    if len(geometry) != EXPECTED_ARCHETYPE_COUNT:
        raise ValueError(
            "geometry_per_dwelling_type_and_period.csv must contain exactly "
            f"{EXPECTED_ARCHETYPE_COUNT} rows; found {len(geometry)}"
        )

    require_columns(
        u_values,
        {"construction_period"},
        "U_values_per_construction_period.csv",
    )
    require_no_missing(
        u_values,
        ["construction_period"],
        "U_values_per_construction_period.csv",
    )
    expected_types = set(V50_COLUMNS_BY_DWELLING_TYPE)
    expected_periods = set(u_values["construction_period"])
    observed_types = set(geometry["dwelling_type"])
    observed_periods = set(geometry["construction_period"])
    expected_pairs = {
        (dwelling_type, construction_period)
        for dwelling_type in expected_types
        for construction_period in expected_periods
    }
    observed_pairs = set(
        geometry[["dwelling_type", "construction_period"]].itertuples(
            index=False, name=None
        )
    )
    if (
        len(expected_types) != 5
        or len(expected_periods) != 5
        or observed_types != expected_types
        or observed_periods != expected_periods
        or observed_pairs != expected_pairs
    ):
        raise ValueError(
            "Geometry must be the complete 5 dwelling types x 5 construction "
            "periods Cartesian product; "
            f"missing pairs={sorted(expected_pairs - observed_pairs)}, "
            f"unexpected pairs={sorted(observed_pairs - expected_pairs)}"
        )

    geometry_fields = [
        column
        for column in geometry.columns
        if column
        not in {
            "tabula_type_number",
            "dwelling_type",
            "construction_period",
            "source_table",
            "source_url",
        }
    ]
    require_no_missing(
        geometry, geometry_fields, "geometry_per_dwelling_type_and_period.csv"
    )

    u_value_fields = [column for column in u_values.columns if column.startswith("U_")]
    require_columns(
        u_values,
        {"construction_period", *u_value_fields},
        "U_values_per_construction_period.csv",
    )
    if not u_value_fields:
        raise ValueError("U_values_per_construction_period.csv has no U-value fields")
    require_unique(u_values, ["construction_period"], "U_values_per_construction_period.csv")
    require_no_missing(u_values, u_value_fields, "U_values_per_construction_period.csv")

    airtightness_long = reshape_airtightness(airtightness)
    require_no_missing(
        airtightness_long,
        ["v50_m3_h_m2"],
        "reshaped in_and_exfiltration_rates.csv",
    )

    try:
        merged = geometry.merge(
            airtightness_long,
            on=geometry_keys,
            how="left",
            validate="one_to_one",
        )
        merged = merged.merge(
            u_values[["construction_period", *u_value_fields]],
            on="construction_period",
            how="left",
            validate="many_to_one",
        )
    except MergeError as exc:
        raise ValueError(f"Archetype merge created duplicate rows: {exc}") from exc

    required_physical_fields = [*geometry_fields, *u_value_fields, "v50_m3_h_m2"]
    require_no_missing(merged, required_physical_fields, "base archetype merge")
    if len(merged) != EXPECTED_ARCHETYPE_COUNT:
        raise ValueError(
            f"Base archetype merge produced {len(merged)} rows; expected "
            f"{EXPECTED_ARCHETYPE_COUNT}"
        )

    merged = merged.rename(columns={"tabula_type_number": "TABULA_type_number"})
    merged["archetype_id"] = merged["TABULA_type_number"].map(
        lambda number: f"BE_TABULA_{int(number):02d}"
    )
    require_unique(merged, ["archetype_id"], "base archetype matrix")

    output_columns = [
        "archetype_id",
        "dwelling_type",
        "construction_period",
        "TABULA_type_number",
        *geometry_fields,
        *u_value_fields,
        "v50_m3_h_m2",
    ]
    result = (
        merged[output_columns]
        .sort_values("TABULA_type_number")
        .reset_index(drop=True)
    )
    result = add_heat_loss_fields(result)
    return result, geometry_fields, u_value_fields


def create_stock_weighted_matrix(
    base: pd.DataFrame,
    cadastral: pd.DataFrame,
    hc37: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    regional_joint = build_regional_joint_distribution(cadastral, hc37)
    national_joint = (
        regional_joint.groupby(
            ["dwelling_type", "construction_period"], as_index=False, sort=False
        )["regional_number_of_dwellings"]
        .sum()
        .rename(columns={"regional_number_of_dwellings": "number_of_dwellings"})
    )
    result = base.merge(
        national_joint,
        on=["dwelling_type", "construction_period"],
        how="left",
        validate="one_to_one",
    )
    require_no_missing(
        result, ["number_of_dwellings"], "national joint archetype merge"
    )

    modeled_dwellings, all_type_dwellings = regional_stock_totals(cadastral)
    result["national_share"] = result["number_of_dwellings"] / modeled_dwellings
    excluded_dwellings = all_type_dwellings - modeled_dwellings
    result["modelled_stock_dwellings_R1_R4"] = modeled_dwellings
    result["excluded_residual_R5_R6_dwellings"] = excluded_dwellings
    result["excluded_residual_R5_R6_share"] = excluded_dwellings / all_type_dwellings
    result["stock_scope"] = (
        "R1-R4 residential archetype scope; R5 commerce houses and R6 other "
        "buildings excluded"
    )
    result["regional_share"] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    result["weighting_method"] = WEIGHTING_METHOD

    stock_fields = [
        "number_of_dwellings",
        "modelled_stock_dwellings_R1_R4",
        "excluded_residual_R5_R6_dwellings",
        "excluded_residual_R5_R6_share",
        "stock_scope",
        "national_share",
        "regional_share",
        "weighting_method",
    ]
    result = result[[*base.columns, *stock_fields]]

    if len(result) != EXPECTED_ARCHETYPE_COUNT:
        raise ValueError(
            f"Stock-weighted matrix has {len(result)} rows; expected "
            f"{EXPECTED_ARCHETYPE_COUNT}"
        )
    if not math.isclose(result["national_share"].sum(), 1.0, abs_tol=1e-12):
        raise ValueError(
            "National archetype shares do not sum to 1.0: "
            f"{result['national_share'].sum():.15f}"
        )
    if not math.isclose(
        result["number_of_dwellings"].sum(),
        modeled_dwellings,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Archetype dwelling counts do not sum to the Statbel R1-R4 total: "
            f"archetypes={result['number_of_dwellings'].sum()}, "
            f"modeled R1-R4={modeled_dwellings}"
        )
    return result, modeled_dwellings, all_type_dwellings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Belgian base physical and stock-weighted residential "
            "archetype matrices."
        )
    )
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_FILE)
    parser.add_argument("--airtightness", type=Path, default=DEFAULT_AIRTIGHTNESS_FILE)
    parser.add_argument("--u-values", type=Path, default=DEFAULT_U_VALUES_FILE)
    parser.add_argument(
        "--state-packages", type=Path, default=DEFAULT_STATE_PACKAGES_FILE
    )
    parser.add_argument(
        "--cadastral-stock", type=Path, default=DEFAULT_CADASTRAL_STOCK_FILE
    )
    parser.add_argument(
        "--joint-census", type=Path, default=DEFAULT_JOINT_CENSUS_FILE
    )
    parser.add_argument("--base-output", type=Path, default=DEFAULT_BASE_OUTPUT)
    parser.add_argument("--stock-output", type=Path, default=DEFAULT_STOCK_OUTPUT)
    parser.add_argument(
        "--tabula-pdf",
        type=Path,
        required=True,
        help="Path to BE_TABULA_ScientificReport_VITO.pdf for source verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [
        args.geometry,
        args.airtightness,
        args.u_values,
        args.state_packages,
        args.cadastral_stock,
        args.joint_census,
    ]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing input files: {missing_inputs}")

    geometry = pd.read_csv(args.geometry)
    airtightness = pd.read_csv(args.airtightness)
    u_values = pd.read_csv(args.u_values)
    state_packages = pd.read_csv(args.state_packages)
    cadastral = load_cadastral(args.cadastral_stock)
    hc37 = load_hc37(args.joint_census)

    base, _, _ = create_base_matrix(geometry, airtightness, u_values)
    stock_weighted, modeled_dwellings, all_type_dwellings = create_stock_weighted_matrix(
        base, cadastral, hc37
    )

    # Verify all source values before writing either output.
    verify_physical_csvs_against_pdf(
        args.tabula_pdf, geometry, airtightness, u_values, state_packages
    )

    args.base_output.parent.mkdir(parents=True, exist_ok=True)
    args.stock_output.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.base_output, index=False)
    stock_weighted.to_csv(args.stock_output, index=False)

    print("Archetype matrix summary")
    print(f"- base_physical_archetype_matrix.csv rows: {len(base)}")
    print(f"- stock_weighted_archetype_matrix.csv rows: {len(stock_weighted)}")
    print(f"- sum of national_share: {stock_weighted['national_share'].sum():.12f}")
    print(
        "- sum of number_of_dwellings: "
        f"{stock_weighted['number_of_dwellings'].sum():.6f} "
        f"(modeled Statbel R1-R4 total: {modeled_dwellings})"
    )
    print(
        "- excluded R5/R6 residual: "
        f"{all_type_dwellings - modeled_dwellings} "
        f"of {all_type_dwellings} all-type dwellings"
    )
    print(
        "- TABULA verification: Tables 9, 10, and 19 plus the standard and "
        "advanced renovation packages match the physical CSVs"
    )
    print("- assumptions:")
    for assumption in MODEL_ASSUMPTIONS:
        print(f"  - {assumption}")


if __name__ == "__main__":
    main()
