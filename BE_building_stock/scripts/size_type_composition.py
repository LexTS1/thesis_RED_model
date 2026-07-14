from pathlib import Path
import argparse
import pandas as pd

"""
compact script explanation
"""


YEAR = 2025
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Statbel REFNIS codes used in the building-stock open-data file
BELGIUM_CODE = "01000"
REGIONS = {
    "02000": "Flemish Region",
    "03000": "Walloon Region",
    "04000": "Brussels-Capital Region",
}

# Statbel building-type codes
BUILDING_TYPES = {
    "R1": "Terraced / closed houses",
    "R2": "Semi-detached houses",
    "R3": "Detached/open houses, farms, castles",
    "R4": "Apartment buildings",
    "R5": "Commercial houses",
    "R6": "Other buildings",
}

ALL_BUILDING_CODES = ["R1", "R2", "R3", "R4", "R5", "R6"]
HOUSE_CODES = ["R1", "R2", "R3"]
APARTMENT_CODES = ["R4"]
COMMERCIAL_CODES = ["R5"]
OTHER_CODES = ["R6"]

# Statbel variable codes used in the XLSX
STAT_BUILDINGS = "T1"   # Number of buildings
STAT_DWELLINGS = "T8"   # Number of dwellings / "woongelegenheden"


def clean_code(value, width=None):
    """
    Convert Excel-imported codes to clean strings.

    Example:
    1000.0 -> "01000" if width=5
    " R1 " -> "R1"
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if width is not None:
        text = text.zfill(width)

    return text


def format_percent(value):
    """Return a share as a percentage string with one decimal."""
    return f"{100 * value:.1f}%"


def load_statbel_file(xlsx_path):
    """
    Load and lightly clean the Statbel XLSX file.

    Expected file:
    building_stock_open_data_2025.xlsx
    from Statbel open data:
    Cadastral statistics of the building stock.
    """
    df = pd.read_excel(xlsx_path)

    required_columns = {
        "CD_YEAR",
        "CD_REFNIS",
        "CD_STAT_TYPE",
        "CD_BUILDING_TYPE",
        "MS_VALUE",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "The input file does not contain the expected Statbel columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df["CD_YEAR"] = pd.to_numeric(df["CD_YEAR"], errors="coerce").astype("Int64")
    df["CD_REFNIS"] = df["CD_REFNIS"].apply(lambda x: clean_code(x, width=5))
    df["CD_STAT_TYPE"] = df["CD_STAT_TYPE"].apply(lambda x: clean_code(x).upper())
    df["CD_BUILDING_TYPE"] = df["CD_BUILDING_TYPE"].apply(lambda x: clean_code(x).upper())
    df["MS_VALUE"] = pd.to_numeric(df["MS_VALUE"], errors="coerce").fillna(0)

    return df


def get_value(df, refnis_code, stat_type, building_codes):
    """
    Sum MS_VALUE for:
    - one geographic code
    - one statistic type: T1 buildings or T8 dwellings
    - one or several building-type codes
    """
    subset = df[
        (df["CD_YEAR"] == YEAR)
        & (df["CD_REFNIS"] == refnis_code)
        & (df["CD_STAT_TYPE"] == stat_type)
        & (df["CD_BUILDING_TYPE"].isin(building_codes))
    ]

    if subset.empty:
        raise ValueError(
            "No rows found for "
            f"year={YEAR}, refnis={refnis_code}, "
            f"stat_type={stat_type}, building_codes={building_codes}"
        )

    return int(round(subset["MS_VALUE"].sum()))


def create_regional_distribution(df, output_dir):
    """
    CSV 1:
    Regional distribution of buildings and dwellings.
    """
    total_buildings = get_value(df, BELGIUM_CODE, STAT_BUILDINGS, ALL_BUILDING_CODES)
    total_dwellings = get_value(df, BELGIUM_CODE, STAT_DWELLINGS, ALL_BUILDING_CODES)

    rows = []

    for refnis_code, region_name in REGIONS.items():
        buildings = get_value(df, refnis_code, STAT_BUILDINGS, ALL_BUILDING_CODES)
        dwellings = get_value(df, refnis_code, STAT_DWELLINGS, ALL_BUILDING_CODES)

        rows.append({
            "Region": region_name,
            "Buildings": buildings,
            "Share of Belgian buildings": format_percent(buildings / total_buildings),
            "Dwellings": dwellings,
            "Share of Belgian dwellings": format_percent(dwellings / total_dwellings),
        })

    rows.append({
        "Region": "Belgium",
        "Buildings": total_buildings,
        "Share of Belgian buildings": format_percent(1.0),
        "Dwellings": total_dwellings,
        "Share of Belgian dwellings": format_percent(1.0),
    })

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "regional_distribution_buildings_dwellings.csv", index=False)


def create_houses_vs_apartment_buildings(df, output_dir):
    """
    CSV 2:
    Houses versus apartment buildings, based on number of buildings.
    """
    total_buildings = get_value(df, BELGIUM_CODE, STAT_BUILDINGS, ALL_BUILDING_CODES)

    categories = [
        ("Houses", "R1 + R2 + R3", HOUSE_CODES),
        ("Apartment buildings", "R4", APARTMENT_CODES),
        ("Commercial houses", "R5", COMMERCIAL_CODES),
        ("Other buildings", "R6", OTHER_CODES),
        ("Total", "R1–R6", ALL_BUILDING_CODES),
    ]

    rows = []

    for category, code_label, codes in categories:
        number_of_buildings = get_value(df, BELGIUM_CODE, STAT_BUILDINGS, codes)

        rows.append({
            "Category": category,
            "Building-type codes": code_label,
            "Number of buildings": number_of_buildings,
            "Share of all Belgian buildings": format_percent(number_of_buildings / total_buildings),
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "houses_vs_apartment_buildings.csv", index=False)


def create_dwellings_by_building_type(df, output_dir):
    """
    CSV 3:
    Dwellings by building type.
    """
    total_dwellings = get_value(df, BELGIUM_CODE, STAT_DWELLINGS, ALL_BUILDING_CODES)

    rows = []

    for code, label in BUILDING_TYPES.items():
        dwellings = get_value(df, BELGIUM_CODE, STAT_DWELLINGS, [code])

        rows.append({
            "Building type containing the dwelling": label,
            "Code": code,
            "Dwellings": dwellings,
            "Share of all Belgian dwellings": format_percent(dwellings / total_dwellings),
        })

    rows.append({
        "Building type containing the dwelling": "Total",
        "Code": "R1–R6",
        "Dwellings": total_dwellings,
        "Share of all Belgian dwellings": format_percent(1.0),
    })

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "dwellings_by_building_type.csv", index=False)


def create_regional_houses_apartments_split(df, output_dir):
    """
    CSV 4:
    Regional houses-versus-apartments split, based on dwellings.
    """
    rows = []

    region_items = list(REGIONS.items()) + [(BELGIUM_CODE, "Belgium")]

    for refnis_code, region_name in region_items:
        regional_total_dwellings = get_value(
            df,
            refnis_code,
            STAT_DWELLINGS,
            ALL_BUILDING_CODES,
        )

        dwellings_in_houses = get_value(
            df,
            refnis_code,
            STAT_DWELLINGS,
            HOUSE_CODES,
        )

        dwellings_in_apartment_buildings = get_value(
            df,
            refnis_code,
            STAT_DWELLINGS,
            APARTMENT_CODES,
        )

        other_dwellings = get_value(
            df,
            refnis_code,
            STAT_DWELLINGS,
            COMMERCIAL_CODES + OTHER_CODES,
        )

        rows.append({
            "Region": region_name,
            "Dwellings in houses, R1–R3": dwellings_in_houses,
            "Share of regional dwellings, houses R1–R3": format_percent(
                dwellings_in_houses / regional_total_dwellings
            ),
            "Dwellings in apartment buildings, R4": dwellings_in_apartment_buildings,
            "Share of regional dwellings, apartments R4": format_percent(
                dwellings_in_apartment_buildings / regional_total_dwellings
            ),
            "Other, R5–R6": other_dwellings,
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "regional_houses_apartments_split_dwellings.csv", index=False)


def print_belgian_totals(df):
    """
    Terminal output only:
    Total Belgian building and dwelling stock, 2025.
    """
    total_buildings = get_value(df, BELGIUM_CODE, STAT_BUILDINGS, ALL_BUILDING_CODES)
    total_dwellings = get_value(df, BELGIUM_CODE, STAT_DWELLINGS, ALL_BUILDING_CODES)

    print("\nTotal Belgian building and dwelling stock, 2025")
    print("-" * 55)
    print(f"Buildings: {total_buildings:,}")
    print(f"Dwellings: {total_dwellings:,}")
    print("-" * 55)


def validate_expected_totals(df):
    """
    Optional safety check.

    These are the expected Statbel 2025 headline totals.
    If this fails, the input file, filters, or Statbel coding may differ.
    """
    expected_buildings = 4_668_425
    expected_dwellings = 5_827_823

    total_buildings = get_value(df, BELGIUM_CODE, STAT_BUILDINGS, ALL_BUILDING_CODES)
    total_dwellings = get_value(df, BELGIUM_CODE, STAT_DWELLINGS, ALL_BUILDING_CODES)

    if total_buildings != expected_buildings:
        raise ValueError(
            f"Unexpected Belgian building total: {total_buildings:,}. "
            f"Expected {expected_buildings:,}."
        )

    if total_dwellings != expected_dwellings:
        raise ValueError(
            f"Unexpected Belgian dwelling total: {total_dwellings:,}. "
            f"Expected {expected_dwellings:,}."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate size, type, and composition CSV files from Statbel building stock XLSX."
    )

    parser.add_argument(
        "xlsx_path",
        type=Path,
        nargs="?",
        default=DATA_DIR / "raw" / "statbel" / "building_stock_open_data_2025.xlsx",
        help="Path to building_stock_open_data_2025.xlsx",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "derived" / "stock_composition",
        help="Folder where the generated CSV files will be written.",
    )

    args = parser.parse_args()

    if not args.xlsx_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.xlsx_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_statbel_file(args.xlsx_path)

    validate_expected_totals(df)
    print_belgian_totals(df)

    create_regional_distribution(df, args.output_dir)
    create_houses_vs_apartment_buildings(df, args.output_dir)
    create_dwellings_by_building_type(df, args.output_dir)
    create_regional_houses_apartments_split(df, args.output_dir)

    print("\nCreated CSV files:")
    print(f"- {args.output_dir / 'regional_distribution_buildings_dwellings.csv'}")
    print(f"- {args.output_dir / 'houses_vs_apartment_buildings.csv'}")
    print(f"- {args.output_dir / 'dwellings_by_building_type.csv'}")
    print(f"- {args.output_dir / 'regional_houses_apartments_split_dwellings.csv'}")


if __name__ == "__main__":
    main()
