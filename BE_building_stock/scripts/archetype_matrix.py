from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from pandas.errors import MergeError


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
DEFAULT_PERIOD_STOCK_FILE = (
    DATA_DIR
    / "derived"
    / "construction_periods"
    / "dwellings_by_construction_period.csv"
)
DEFAULT_TYPE_STOCK_FILE = (
    DATA_DIR
    / "derived"
    / "stock_composition"
    / "dwellings_by_building_type.csv"
)
DEFAULT_BASE_OUTPUT = DATA_DIR / "matrices" / "national" / "base_physical_archetype_matrix.csv"
DEFAULT_STOCK_OUTPUT = (
    DATA_DIR / "matrices" / "national" / "stock_weighted_archetype_matrix.csv"
)

EXPECTED_ARCHETYPE_COUNT = 25
WEIGHTING_METHOD = "independence_assumption_type_share_times_period_share"

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
    "No dwelling-type x construction-period cross-tabulation is supplied; "
    "national archetype shares use the independence assumption.",
    "Statbel R4 apartment-building dwellings are split equally between the "
    "enclosed and exposed apartment archetypes because the input has no subtype split.",
    "The 2001-2010 construction-period count is split equally between 2001-2005 "
    "and 2006-2010 because the aggregated input crosses the TABULA boundary.",
    "The residential-archetype scope is R1-R4. R5 commerce houses and R6 other "
    "buildings are reported as excluded residual dwellings rather than assigned "
    "to a TABULA archetype without evidence.",
    "Dwellings with unknown construction period are distributed pro rata by "
    "normalising the known-period counts to 1.",
    "regional_share is empty because regional dwelling-type marginals are not "
    "available in the supplied building-type input.",
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


def verify_physical_csvs_against_pdf(
    pdf_path: Path,
    geometry: pd.DataFrame,
    airtightness: pd.DataFrame,
    u_values: pd.DataFrame,
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

    if errors:
        formatted_errors = "\n- ".join(errors)
        raise ValueError(
            "Physical CSV values conflict with the TABULA/VITO PDF; no outputs were "
            f"written:\n- {formatted_errors}"
        )


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
    result = merged[output_columns].sort_values("TABULA_type_number").reset_index(drop=True)
    return result, geometry_fields, u_value_fields


def load_type_shares(
    type_stock: pd.DataFrame,
) -> tuple[dict[str, float], int, int]:
    require_columns(
        type_stock,
        {"Building type containing the dwelling", "Code", "Dwellings"},
        "dwellings_by_building_type.csv",
    )
    require_unique(type_stock, ["Code"], "dwellings_by_building_type.csv")

    total_rows = type_stock[
        type_stock["Building type containing the dwelling"].str.strip().eq("Total")
    ]
    if len(total_rows) != 1:
        raise ValueError(
            "dwellings_by_building_type.csv must contain exactly one Total row"
        )
    all_type_dwellings = int(total_rows.iloc[0]["Dwellings"])

    component_rows = type_stock[~type_stock.index.isin(total_rows.index)]
    component_total = int(component_rows["Dwellings"].sum())
    if component_total != all_type_dwellings:
        raise ValueError(
            "Building-type component counts do not sum to the Statbel total: "
            f"components={component_total}, total={all_type_dwellings}"
        )

    by_code = component_rows.set_index("Code")["Dwellings"]
    required_codes = {"R1", "R2", "R3", "R4", "R5", "R6"}
    missing_codes = required_codes - set(by_code.index)
    if missing_codes:
        raise ValueError(
            f"dwellings_by_building_type.csv is missing codes: {sorted(missing_codes)}"
        )

    modeled_counts = {
        "Detached house": float(by_code["R3"]),
        "Semi-detached house": float(by_code["R2"]),
        "Terraced house": float(by_code["R1"]),
        "Apartment, enclosed": float(by_code["R4"]) / 2.0,
        "Apartment, exposed": float(by_code["R4"]) / 2.0,
    }
    modeled_total = sum(modeled_counts.values())
    return (
        {dwelling_type: count / modeled_total for dwelling_type, count in modeled_counts.items()},
        int(modeled_total),
        all_type_dwellings,
    )


def load_period_shares(period_stock: pd.DataFrame) -> dict[str, float]:
    period_field = "Construction-period class"
    count_field = "Belgium dwellings"
    require_columns(
        period_stock,
        {period_field, count_field},
        "dwellings_by_construction_period.csv",
    )
    require_unique(period_stock, [period_field], "dwellings_by_construction_period.csv")
    by_period = period_stock.set_index(period_field)[count_field]
    required_periods = {
        "Before 1919",
        "1919-1945",
        "1946-1970",
        "1971-1990",
        "1991-2000",
        "2001-2010",
        "2011 onwards",
    }
    missing_periods = required_periods - set(by_period.index)
    if missing_periods:
        raise ValueError(
            "dwellings_by_construction_period.csv is missing classes: "
            f"{sorted(missing_periods)}"
        )

    modeled_counts = {
        "pre-1946": float(by_period["Before 1919"] + by_period["1919-1945"]),
        "1946-1970": float(by_period["1946-1970"]),
        "1971-1990": float(by_period["1971-1990"]),
        "1991-2005": float(by_period["1991-2000"] + 0.5 * by_period["2001-2010"]),
        "post-2005": float(0.5 * by_period["2001-2010"] + by_period["2011 onwards"]),
    }
    modeled_total = sum(modeled_counts.values())
    return {
        construction_period: count / modeled_total
        for construction_period, count in modeled_counts.items()
    }


def create_stock_weighted_matrix(
    base: pd.DataFrame,
    type_stock: pd.DataFrame,
    period_stock: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    type_shares, modeled_dwellings, all_type_dwellings = load_type_shares(type_stock)
    period_shares = load_period_shares(period_stock)

    missing_types = set(base["dwelling_type"]) - set(type_shares)
    missing_periods = set(base["construction_period"]) - set(period_shares)
    if missing_types or missing_periods:
        raise ValueError(
            "Stock marginals do not cover the base archetypes: "
            f"missing dwelling types={sorted(missing_types)}, "
            f"missing periods={sorted(missing_periods)}"
        )

    result = base.copy()
    result["national_share"] = result.apply(
        lambda row: type_shares[row["dwelling_type"]]
        * period_shares[row["construction_period"]],
        axis=1,
    )
    excluded_dwellings = all_type_dwellings - modeled_dwellings
    result["number_of_dwellings"] = modeled_dwellings * result["national_share"]
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
            "Archetype dwelling counts do not sum to the Statbel total: "
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
    parser.add_argument("--period-stock", type=Path, default=DEFAULT_PERIOD_STOCK_FILE)
    parser.add_argument("--type-stock", type=Path, default=DEFAULT_TYPE_STOCK_FILE)
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
        args.period_stock,
        args.type_stock,
    ]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing input files: {missing_inputs}")

    geometry = pd.read_csv(args.geometry)
    airtightness = pd.read_csv(args.airtightness)
    u_values = pd.read_csv(args.u_values)
    period_stock = pd.read_csv(args.period_stock)
    type_stock = pd.read_csv(args.type_stock)

    base, _, _ = create_base_matrix(geometry, airtightness, u_values)
    stock_weighted, modeled_dwellings, all_type_dwellings = create_stock_weighted_matrix(
        base, type_stock, period_stock
    )

    # Verify all source values before writing either output.
    verify_physical_csvs_against_pdf(
        args.tabula_pdf, geometry, airtightness, u_values
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
    print("- TABULA verification: Tables 9, 10, and 19 match the physical CSVs")
    print("- assumptions:")
    for assumption in MODEL_ASSUMPTIONS:
        print(f"  - {assumption}")


if __name__ == "__main__":
    main()
