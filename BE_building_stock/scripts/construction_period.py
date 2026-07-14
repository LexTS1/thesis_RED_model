from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Paths are derived from the repository layout, so the script can be run from
# any working directory.
INPUT_FILE = DATA_DIR / "raw" / "statbel" / "T04_POC_BE_NL.xlsx"
OUTPUT_FILE = (
    DATA_DIR
    / "derived"
    / "construction_periods"
    / "dwellings_by_construction_period.csv"
)
SHEET_NAME = "CENSUS_T04_21_BE_POC_CL_2021"

# Statbel NIS codes for Belgium and its three regions.
AREAS = {
    "01000": "Belgium",
    "04000": "Brussels",
    "02000": "Flanders",
    "03000": "Wallonia",
}

# The output classes combine Statbel's more detailed 2021 period columns.
CONSTRUCTION_PERIODS = {
    "Before 1919": ["Vóór 1919"],
    "1919-1945": ["Van 1919 tot en met 1945"],
    "1946-1970": [
        "Van 1946 tot en met 1960",
        "Van 1961 tot en met 1970",
    ],
    "1971-1990": [
        "Van 1971 tot en met 1980",
        "Van 1981 tot en met 1990",
    ],
    "1991-2000": ["Van 1991 tot en met 2000"],
    "2001-2010": [
        "Van 2001 tot en met 2005",
        "Van 2006 tot en met 2010",
    ],
    "2011 onwards": [
        "Van 2011 tot en met 2015",
        "2016 of later",
    ],
}


def load_statbel_data(path: Path) -> pd.DataFrame:
    """Read the 2021 Statbel table and retain Belgium and regional rows."""
    data = pd.read_excel(path, sheet_name=SHEET_NAME, header=3)
    data = data.rename(
        columns={data.columns[0]: "nis_code", data.columns[1]: "area"}
    )

    # Excel can interpret codes as numbers; normalise them back to five digits.
    data["nis_code"] = data["nis_code"].astype(str).str.replace(".0", "", regex=False)
    data["nis_code"] = data["nis_code"].str.zfill(5)
    data = data[data["nis_code"].isin(AREAS)].set_index("nis_code")

    missing_areas = set(AREAS) - set(data.index)
    if missing_areas:
        raise ValueError(f"Missing Statbel area codes: {sorted(missing_areas)}")

    required_columns = {
        column
        for source_columns in CONSTRUCTION_PERIODS.values()
        for column in source_columns
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing Statbel period columns: {sorted(missing_columns)}")

    return data


def create_construction_period_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return absolute dwelling counts and within-area percentages."""
    rows = []

    for period, source_columns in CONSTRUCTION_PERIODS.items():
        row = {"Construction-period class": period}

        for nis_code, area_name in AREAS.items():
            count = int(data.loc[nis_code, source_columns].sum())
            row[f"{area_name} dwellings"] = count

        rows.append(row)

    result = pd.DataFrame(rows)

    # Unknown construction years are excluded, so the seven percentages for
    # each area describe the known construction-period distribution and sum to 100.
    for area_name in AREAS.values():
        count_column = f"{area_name} dwellings"
        total_with_known_period = result[count_column].sum()
        result[f"{area_name} share (%)"] = (
            100 * result[count_column] / total_with_known_period
        ).round(2)

    # Keep each area's absolute count and percentage next to one another.
    ordered_columns = ["Construction-period class"]
    for area_name in AREAS.values():
        ordered_columns.extend(
            [f"{area_name} dwellings", f"{area_name} share (%)"]
        )

    return result[ordered_columns]


def main() -> None:
    data = load_statbel_data(INPUT_FILE)
    result = create_construction_period_table(data)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Created {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
