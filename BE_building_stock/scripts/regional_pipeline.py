from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

YEAR = 2025
STAT_DWELLINGS = "T8"
R1_R4 = ("R1", "R2", "R3", "R4")
R5_R6 = ("R5", "R6")

REGIONS = {
    "02000": "Flemish Region",
    "03000": "Walloon Region",
    "04000": "Brussels-Capital Region",
}

REGIONAL_PERIOD_FIELDS = {
    "Flemish Region": "Flanders dwellings",
    "Walloon Region": "Wallonia dwellings",
    "Brussels-Capital Region": "Brussels dwellings",
}

TYPE_CODE_TO_ARCHETYPE = {
    "R1": "Terraced house",
    "R2": "Semi-detached house",
    "R3": "Detached house",
}

REGIONALISATION_METHOD = (
    "regional_R1_R4_dwelling_type_share_times_regional_construction_period_share_"
    "independence_assumption"
)

TYPE_SHARE_SOURCE = (
    "Statbel 2025 cadastral dwelling counts; R1-R4 renormalised within the "
    "residential-archetype scope; R5/R6 excluded; R4 split 50/50 between "
    "apartment archetypes"
)

PERIOD_SHARE_SOURCE = (
    "Statbel Census 2021 construction-period counts; shares calculated from "
    "unrounded regional counts; 2001-2010 split 50/50 across the TABULA boundary"
)

APARTMENT_SPLIT_NOTE = (
    "Statbel R4 apartment dwellings split 50/50 between Apartment, enclosed and "
    "Apartment, exposed because apartment position is not identified."
)

DEFAULT_PATHS = {
    "statbel": DATA_DIR / "raw" / "statbel" / "building_stock_open_data_2025.xlsx",
    "periods": (
        DATA_DIR
        / "derived"
        / "construction_periods"
        / "dwellings_by_construction_period.csv"
    ),
    "base_matrix": (
        DATA_DIR / "matrices" / "national" / "base_physical_archetype_matrix.csv"
    ),
    "policy": DATA_DIR / "assumptions" / "renovation" / "regional_policy_targets.csv",
    "priority": (
        DATA_DIR / "assumptions" / "renovation" / "renovation_priority_mapping.csv"
    ),
    "regional_split": (
        DATA_DIR / "derived" / "regional_stock" / "regional_dwelling_type_split.csv"
    ),
    "regional_matrix": (
        DATA_DIR / "matrices" / "regional" / "regional_stock_weighted_archetype_matrix.csv"
    ),
    "state_layer": (
        DATA_DIR / "scenarios" / "renovation" / "renovation_state_layer.csv"
    ),
    "allocation_layer": (
        DATA_DIR
        / "scenarios"
        / "renovation"
        / "renovation_state_layer_with_allocation.csv"
    ),
}


def require_columns(data: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = columns - set(data.columns)
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {sorted(missing)}")


def require_unique(data: pd.DataFrame, keys: list[str], table_name: str) -> None:
    duplicates = data.duplicated(keys, keep=False)
    if duplicates.any():
        values = data.loc[duplicates, keys].drop_duplicates().to_dict("records")
        raise ValueError(f"{table_name} has duplicate keys {keys}: {values}")


def clean_code(value: object, width: int | None = None) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(width) if width else text


def load_statbel(path: Path) -> pd.DataFrame:
    data = pd.read_excel(path)
    require_columns(
        data,
        {"CD_YEAR", "CD_REFNIS", "CD_STAT_TYPE", "CD_BUILDING_TYPE", "MS_VALUE"},
        path.name,
    )
    data = data.copy()
    data["CD_YEAR"] = pd.to_numeric(data["CD_YEAR"], errors="raise").astype(int)
    data["CD_REFNIS"] = data["CD_REFNIS"].map(lambda value: clean_code(value, 5))
    data["CD_STAT_TYPE"] = data["CD_STAT_TYPE"].map(clean_code).str.upper()
    data["CD_BUILDING_TYPE"] = (
        data["CD_BUILDING_TYPE"].map(clean_code).str.upper()
    )
    data["MS_VALUE"] = pd.to_numeric(data["MS_VALUE"], errors="raise")
    return data


def statbel_value(data: pd.DataFrame, region_code: str, building_code: str) -> int:
    selected = data[
        (data["CD_YEAR"] == YEAR)
        & (data["CD_REFNIS"] == region_code)
        & (data["CD_STAT_TYPE"] == STAT_DWELLINGS)
        & (data["CD_BUILDING_TYPE"] == building_code)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one Statbel dwelling row for "
            f"region={region_code}, building_type={building_code}; found {len(selected)}"
        )
    return int(round(float(selected.iloc[0]["MS_VALUE"])))


def build_regional_split(statbel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region_code, region in REGIONS.items():
        counts = {
            code: statbel_value(statbel, region_code, code)
            for code in (*R1_R4, *R5_R6)
        }
        total = sum(counts.values())
        modelled = sum(counts[code] for code in R1_R4)
        residual = sum(counts[code] for code in R5_R6)
        rows.append(
            {
                "region": region,
                "terraced_dwellings_R1": counts["R1"],
                "semi_detached_dwellings_R2": counts["R2"],
                "detached_dwellings_R3": counts["R3"],
                "apartment_dwellings_R4": counts["R4"],
                "commercial_other_dwellings_R5_R6": residual,
                "modelled_residential_dwellings_R1_R4": modelled,
                "total_dwellings_R1_R6": total,
                "share_R1_within_R1_R4": counts["R1"] / modelled,
                "share_R2_within_R1_R4": counts["R2"] / modelled,
                "share_R3_within_R1_R4": counts["R3"] / modelled,
                "share_R4_within_R1_R4": counts["R4"] / modelled,
                "excluded_R5_R6_share_of_R1_R6": residual / total,
                "scope": (
                    "R1-R4 only; R5 commerce houses and R6 other buildings are "
                    "reported but excluded because no defensible TABULA mapping exists"
                ),
            }
        )
    result = pd.DataFrame(rows)
    require_unique(result, ["region"], "regional dwelling-type split")
    return result


def construction_period_shares(
    period_data: pd.DataFrame, region: str
) -> dict[str, float]:
    period_field = "Construction-period class"
    count_field = REGIONAL_PERIOD_FIELDS[region]
    require_columns(period_data, {period_field, count_field}, "construction-period table")
    require_unique(period_data, [period_field], "construction-period table")
    values = period_data.set_index(period_field)[count_field]
    required = {
        "Before 1919",
        "1919-1945",
        "1946-1970",
        "1971-1990",
        "1991-2000",
        "2001-2010",
        "2011 onwards",
    }
    missing = required - set(values.index)
    if missing:
        raise ValueError(f"construction-period table is missing: {sorted(missing)}")
    counts = {
        "pre-1946": float(values["Before 1919"] + values["1919-1945"]),
        "1946-1970": float(values["1946-1970"]),
        "1971-1990": float(values["1971-1990"]),
        "1991-2005": float(values["1991-2000"] + 0.5 * values["2001-2010"]),
        "post-2005": float(0.5 * values["2001-2010"] + values["2011 onwards"]),
    }
    total = sum(counts.values())
    return {period: count / total for period, count in counts.items()}


def dwelling_type_shares(split_row: pd.Series) -> dict[str, float]:
    modelled = float(split_row["modelled_residential_dwellings_R1_R4"])
    shares = {
        TYPE_CODE_TO_ARCHETYPE["R1"]: float(split_row["terraced_dwellings_R1"])
        / modelled,
        TYPE_CODE_TO_ARCHETYPE["R2"]: float(
            split_row["semi_detached_dwellings_R2"]
        )
        / modelled,
        TYPE_CODE_TO_ARCHETYPE["R3"]: float(split_row["detached_dwellings_R3"])
        / modelled,
        "Apartment, enclosed": 0.5
        * float(split_row["apartment_dwellings_R4"])
        / modelled,
        "Apartment, exposed": 0.5
        * float(split_row["apartment_dwellings_R4"])
        / modelled,
    }
    if not math.isclose(sum(shares.values()), 1.0, abs_tol=1e-12):
        raise ValueError("Regional dwelling-type shares do not sum to one")
    return shares


def build_regional_matrix(
    base: pd.DataFrame,
    regional_split: pd.DataFrame,
    period_data: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        base,
        {"archetype_id", "dwelling_type", "construction_period", "TABULA_type_number"},
        "base physical archetype matrix",
    )
    require_unique(base, ["archetype_id"], "base physical archetype matrix")
    if len(base) != 25:
        raise ValueError(f"Base physical matrix must have 25 rows; found {len(base)}")

    national_modelled = float(
        regional_split["modelled_residential_dwellings_R1_R4"].sum()
    )
    rows: list[dict[str, object]] = []
    for _, split_row in regional_split.iterrows():
        region = str(split_row["region"])
        modelled = float(split_row["modelled_residential_dwellings_R1_R4"])
        residual = int(split_row["commercial_other_dwellings_R5_R6"])
        total = int(split_row["total_dwellings_R1_R6"])
        type_shares = dwelling_type_shares(split_row)
        period_shares = construction_period_shares(period_data, region)

        for _, archetype in base.iterrows():
            share = (
                type_shares[str(archetype["dwelling_type"])]
                * period_shares[str(archetype["construction_period"])]
            )
            dwellings = modelled * share
            rows.append(
                {
                    "region": region,
                    "archetype_id": archetype["archetype_id"],
                    "dwelling_type": archetype["dwelling_type"],
                    "construction_period": archetype["construction_period"],
                    "TABULA_type_number": int(archetype["TABULA_type_number"]),
                    "regional_modelled_stock_dwellings": int(modelled),
                    "regional_archetype_share_within_region": share,
                    "regional_share_within_belgium": dwellings / national_modelled,
                    "regional_number_of_dwellings": dwellings,
                    "dwelling_type_share_source": TYPE_SHARE_SOURCE,
                    "construction_period_share_source": PERIOD_SHARE_SOURCE,
                    "regionalisation_method": REGIONALISATION_METHOD,
                    "apartment_position_split_assumption": (
                        APARTMENT_SPLIT_NOTE
                        if str(archetype["dwelling_type"]).startswith("Apartment")
                        else ""
                    ),
                    "excluded_residual_R5_R6_dwellings": residual,
                    "excluded_residual_R5_R6_share": residual / total,
                    **{
                        column: archetype[column]
                        for column in base.columns
                        if column
                        not in {
                            "archetype_id",
                            "dwelling_type",
                            "construction_period",
                            "TABULA_type_number",
                        }
                    },
                }
            )

    result = pd.DataFrame(rows)
    require_unique(result, ["region", "archetype_id"], "regional archetype matrix")
    if len(result) != 75:
        raise ValueError(f"Regional archetype matrix must have 75 rows; found {len(result)}")
    for region, group in result.groupby("region", sort=False):
        share_sum = group["regional_archetype_share_within_region"].sum()
        dwelling_sum = group["regional_number_of_dwellings"].sum()
        modelled = float(group["regional_modelled_stock_dwellings"].iloc[0])
        if not math.isclose(share_sum, 1.0, abs_tol=1e-12):
            raise ValueError(f"{region} archetype shares sum to {share_sum}")
        if not math.isclose(dwelling_sum, modelled, abs_tol=1e-6):
            raise ValueError(f"{region} archetype dwellings do not reconstruct stock")
    return result


def build_renovation_layers(
    regional_matrix: pd.DataFrame,
    policy: pd.DataFrame,
    priority: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_fields = {
        "region",
        "target_2050",
        "target_energy_score_2050_kWh_m2_year",
        "policy_url_primary",
        "policy_target_note",
        "annual_renovation_rate_low",
        "annual_renovation_rate_central",
        "annual_renovation_rate_high",
        "renovation_rate_scenario_meaning",
    }
    priority_fields = {
        "construction_period",
        "renovation_priority",
        "priority_weight",
    }
    require_columns(policy, policy_fields, "regional policy targets")
    require_columns(priority, priority_fields, "renovation priority mapping")
    require_unique(policy, ["region"], "regional policy targets")
    require_unique(priority, ["construction_period"], "renovation priority mapping")

    base_fields = [
        "region",
        "archetype_id",
        "dwelling_type",
        "construction_period",
        "regional_archetype_share_within_region",
        "regional_number_of_dwellings",
        "regional_modelled_stock_dwellings",
        "regionalisation_method",
    ]
    merged = regional_matrix[base_fields].merge(
        priority[list(priority_fields)],
        on="construction_period",
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        policy[list(policy_fields)],
        on="region",
        how="left",
        validate="many_to_one",
    )
    merged["policy_target_note"] = merged["policy_target_note"].fillna("")
    required_merged = [
        column for column in merged.columns if column != "policy_target_note"
    ]
    if merged[required_merged].isna().any().any():
        missing = merged[required_merged].columns[
            merged[required_merged].isna().any()
        ].tolist()
        raise ValueError(f"Renovation-layer merge has missing fields: {missing}")

    merged["baseline_state"] = "TABULA-as-is"
    merged["renovation_share_2025"] = 0.0
    merged["target_label_2050"] = merged["target_2050"]
    merged["policy_source"] = merged["policy_url_primary"]

    state_columns = [
        "region",
        "archetype_id",
        "dwelling_type",
        "construction_period",
        "baseline_state",
        "renovation_priority",
        "renovation_share_2025",
        "annual_renovation_rate_low",
        "annual_renovation_rate_central",
        "annual_renovation_rate_high",
        "renovation_rate_scenario_meaning",
        "target_label_2050",
        "target_energy_score_2050_kWh_m2_year",
        "policy_source",
        "policy_target_note",
    ]
    state_layer = merged[state_columns].copy()

    merged["allocation_raw_weight"] = (
        merged["regional_archetype_share_within_region"]
        * merged["priority_weight"]
    )
    merged["allocation_share_within_region"] = merged.groupby(
        "region", sort=False
    )["allocation_raw_weight"].transform(lambda values: values / values.sum())
    for scenario in ("low", "central", "high"):
        merged[f"annual_renovation_flow_{scenario}"] = (
            merged[f"annual_renovation_rate_{scenario}"]
            * merged["regional_modelled_stock_dwellings"]
            * merged["allocation_share_within_region"]
        )

    allocation_columns = [
        "region",
        "archetype_id",
        "dwelling_type",
        "construction_period",
        "baseline_state",
        "renovation_priority",
        "priority_weight",
        "renovation_share_2025",
        "annual_renovation_rate_low",
        "annual_renovation_rate_central",
        "annual_renovation_rate_high",
        "renovation_rate_scenario_meaning",
        "target_label_2050",
        "target_energy_score_2050_kWh_m2_year",
        "policy_target_note",
        "policy_source",
        "regional_archetype_share_within_region",
        "regional_number_of_dwellings",
        "allocation_raw_weight",
        "allocation_share_within_region",
        "annual_renovation_flow_low",
        "annual_renovation_flow_central",
        "annual_renovation_flow_high",
        "regionalisation_method",
    ]
    allocation_layer = merged[allocation_columns].copy()

    for name, table in (
        ("renovation state layer", state_layer),
        ("renovation allocation layer", allocation_layer),
    ):
        require_unique(table, ["region", "archetype_id"], name)
        if len(table) != 75:
            raise ValueError(f"{name} must have 75 rows; found {len(table)}")
    for region, group in allocation_layer.groupby("region", sort=False):
        total = group["allocation_share_within_region"].sum()
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ValueError(f"{region} renovation allocation shares sum to {total}")
    return state_layer, allocation_layer


def write_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.12g")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all regional stock and renovation-allocation inputs from "
            "the Statbel, Census, TABULA and policy source tables."
        )
    )
    for name in ("statbel", "periods", "base_matrix", "policy", "priority"):
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=Path,
            default=DEFAULT_PATHS[name],
        )
    for name in (
        "regional_split",
        "regional_matrix",
        "state_layer",
        "allocation_layer",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}-output",
            type=Path,
            default=DEFAULT_PATHS[name],
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = {
        name: getattr(args, name)
        for name in ("statbel", "periods", "base_matrix", "policy", "priority")
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing regional-pipeline inputs: {missing}")

    statbel = load_statbel(args.statbel)
    periods = pd.read_csv(args.periods)
    base = pd.read_csv(args.base_matrix)
    policy = pd.read_csv(args.policy)
    priority = pd.read_csv(args.priority)

    regional_split = build_regional_split(statbel)
    regional_matrix = build_regional_matrix(base, regional_split, periods)
    state_layer, allocation_layer = build_renovation_layers(
        regional_matrix, policy, priority
    )

    write_csv(args.regional_split_output, regional_split)
    write_csv(args.regional_matrix_output, regional_matrix)
    write_csv(args.state_layer_output, state_layer)
    write_csv(args.allocation_layer_output, allocation_layer)

    print("Regional pipeline validation passed")
    print(f"- regional dwelling-type rows: {len(regional_split)}")
    print(f"- regional archetype rows: {len(regional_matrix)}")
    print(f"- renovation state rows: {len(state_layer)}")
    print(f"- renovation allocation rows: {len(allocation_layer)}")
    print(
        "- modeled R1-R4 dwellings: "
        f"{int(regional_split['modelled_residential_dwellings_R1_R4'].sum()):,}"
    )
    print(
        "- excluded R5/R6 dwellings: "
        f"{int(regional_split['commercial_other_dwellings_R5_R6'].sum()):,}"
    )


if __name__ == "__main__":
    main()
