from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from archetype_matrix import add_heat_loss_fields
from stock_joint_distribution import (
    JOINT_DISTRIBUTION_METHOD,
    build_regional_joint_distribution,
    load_hc37,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

YEAR = 2025
STAT_DWELLINGS = "T8"
R1_R4 = ("R1", "R2", "R3", "R4")
R5_R6 = ("R5", "R6")
STATE_ORDER = {
    "TABULA_existing": 0,
    "TABULA_standard_B_proxy": 1,
    "TABULA_advanced_A_proxy": 2,
}
ELIGIBLE_STATES = {"TABULA_existing", "TABULA_standard_B_proxy"}
MEDIUM_ELIGIBLE_PERIODS = {
    "pre-1946",
    "1946-1970",
    "1971-1990",
    "1991-2005",
}

REGIONS = {
    "02000": "Flemish Region",
    "03000": "Walloon Region",
    "04000": "Brussels-Capital Region",
}

REGIONALISATION_METHOD = JOINT_DISTRIBUTION_METHOD
TYPE_SHARE_SOURCE = (
    "Statbel 2025 regional T8 dwelling counts by R1-R4 type; R5/R6 retained "
    "as an excluded residual; R4 split equally between apartment archetypes"
)
STOCK_GEOMETRY_MAPPING_ASSUMPTION = (
    "R1-R3 building-age profiles are applied to the matching dwelling totals; "
    "each resulting house-category dwelling retains its whole-house TABULA "
    "geometry; national absolute heat-demand and capacity results are "
    "conditional on this mapping"
)

DEFAULT_PATHS = {
    "statbel": DATA_DIR / "raw" / "statbel" / "building_stock_open_data_2025.xlsx",
    "joint_census": DATA_DIR / "raw" / "statbel" / "TF_CENSUS_2021_HC37_1.xlsx",
    "base_matrix": (
        DATA_DIR / "matrices" / "national" / "base_physical_archetype_matrix.csv"
    ),
    "state_shares": (
        DATA_DIR
        / "assumptions"
        / "renovation"
        / "regional_state_shares_2025.csv"
    ),
    "state_packages": (
        DATA_DIR
        / "assumptions"
        / "renovation"
        / "renovation_physical_state_packages_TABULA.csv"
    ),
    "regional_split": (
        DATA_DIR / "derived" / "regional_stock" / "regional_dwelling_type_split.csv"
    ),
    "joint_distribution": (
        DATA_DIR
        / "derived"
        / "regional_stock"
        / "regional_dwelling_type_period_joint.csv"
    ),
    "regional_matrix": (
        DATA_DIR
        / "matrices"
        / "regional"
        / "regional_stock_weighted_archetype_matrix.csv"
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


def require_no_missing(
    data: pd.DataFrame, columns: list[str], table_name: str
) -> None:
    missing = data[columns].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(
            f"{table_name} has missing values: "
            + ", ".join(f"{column}={count}" for column, count in missing.items())
        )


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
                    "R1-R4 residential archetype scope; R5/R6 retained as "
                    "an excluded residual pending residential-archetype mapping evidence"
                ),
            }
        )
    result = pd.DataFrame(rows)
    require_unique(result, ["region"], "regional dwelling-type split")
    return result


def build_regional_matrix(
    base: pd.DataFrame,
    regional_split: pd.DataFrame,
    joint_distribution: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        base,
        {
            "archetype_id",
            "dwelling_type",
            "construction_period",
            "TABULA_type_number",
            "specific_heat_loss_z_W_m2K",
        },
        "base physical archetype matrix",
    )
    require_unique(base, ["archetype_id"], "base physical archetype matrix")
    if len(base) != 25:
        raise ValueError(f"Base physical matrix must have 25 rows; found {len(base)}")

    require_columns(
        joint_distribution,
        {
            "region",
            "dwelling_type",
            "construction_period",
            "regional_number_of_dwellings",
            "source_profile_measure",
            "source_profile_value",
            "source_profile_share_within_type",
            "joint_distribution_source",
            "joint_distribution_method_detail",
            "regionalisation_method",
            "apartment_position_split_assumption",
        },
        "regional joint distribution",
    )
    require_unique(
        joint_distribution,
        ["region", "dwelling_type", "construction_period"],
        "regional joint distribution",
    )

    try:
        result = joint_distribution.merge(
            base,
            on=["dwelling_type", "construction_period"],
            how="left",
            validate="many_to_one",
        )
        result = result.merge(
            regional_split[
                [
                    "region",
                    "modelled_residential_dwellings_R1_R4",
                    "commercial_other_dwellings_R5_R6",
                    "total_dwellings_R1_R6",
                ]
            ],
            on="region",
            how="left",
            validate="many_to_one",
        )
    except pd.errors.MergeError as exc:
        raise ValueError(f"Regional joint merge created duplicate rows: {exc}") from exc

    require_no_missing(
        result,
        [
            "archetype_id",
            "TABULA_type_number",
            "modelled_residential_dwellings_R1_R4",
            "commercial_other_dwellings_R5_R6",
            "total_dwellings_R1_R6",
        ],
        "regional joint merge",
    )

    national_modelled = float(
        regional_split["modelled_residential_dwellings_R1_R4"].sum()
    )
    result["regional_modelled_stock_dwellings"] = result[
        "modelled_residential_dwellings_R1_R4"
    ].astype(int)
    result["regional_archetype_share_within_region"] = (
        result["regional_number_of_dwellings"]
        / result["regional_modelled_stock_dwellings"]
    )
    result["regional_share_within_belgium"] = (
        result["regional_number_of_dwellings"] / national_modelled
    )
    result["dwelling_type_share_source"] = TYPE_SHARE_SOURCE
    result["construction_period_share_source"] = result[
        "joint_distribution_source"
    ]
    result["stock_geometry_mapping_assumption"] = (
        STOCK_GEOMETRY_MAPPING_ASSUMPTION
    )
    result["excluded_residual_R5_R6_dwellings"] = result[
        "commercial_other_dwellings_R5_R6"
    ].astype(int)
    result["excluded_residual_R5_R6_share"] = (
        result["commercial_other_dwellings_R5_R6"]
        / result["total_dwellings_R1_R6"]
    )

    metadata_columns = [
        "region",
        "archetype_id",
        "dwelling_type",
        "construction_period",
        "TABULA_type_number",
        "regional_modelled_stock_dwellings",
        "regional_archetype_share_within_region",
        "regional_share_within_belgium",
        "regional_number_of_dwellings",
        "dwelling_type_share_source",
        "construction_period_share_source",
        "stock_geometry_mapping_assumption",
        "source_profile_measure",
        "source_profile_value",
        "source_profile_share_within_type",
        "joint_distribution_method_detail",
        "regionalisation_method",
        "apartment_position_split_assumption",
        "excluded_residual_R5_R6_dwellings",
        "excluded_residual_R5_R6_share",
    ]
    physical_columns = [
        column
        for column in base.columns
        if column
        not in {
            "archetype_id",
            "dwelling_type",
            "construction_period",
            "TABULA_type_number",
        }
    ]
    result = result[[*metadata_columns, *physical_columns]]
    region_order = {region: index for index, region in enumerate(REGIONS.values())}
    result["_region_order"] = result["region"].map(region_order)
    result = (
        result.sort_values(["_region_order", "TABULA_type_number"])
        .drop(columns="_region_order")
        .reset_index(drop=True)
    )
    require_unique(result, ["region", "archetype_id"], "regional archetype matrix")
    if len(result) != 75:
        raise ValueError(f"Regional archetype matrix must have 75 rows; found {len(result)}")
    for region, group in result.groupby("region", sort=False):
        if not math.isclose(
            group["regional_archetype_share_within_region"].sum(),
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{region} archetype shares do not sum to one")
        if not math.isclose(
            group["regional_number_of_dwellings"].sum(),
            float(group["regional_modelled_stock_dwellings"].iloc[0]),
            abs_tol=1e-6,
        ):
            raise ValueError(f"{region} archetype dwellings do not reconstruct stock")
    return result


def validate_state_inputs(
    state_shares: pd.DataFrame, state_packages: pd.DataFrame
) -> None:
    require_columns(
        state_shares,
        {
            "region",
            "state_id",
            "regional_state_share",
            "regional_state_percent",
            "regional_state_share_lower",
            "regional_state_share_upper",
            "regional_state_percent_lower",
            "regional_state_percent_upper",
            "rounding_half_width_pp",
            "share_value_role",
            "interval_method",
            "regional_evidence_anchor",
            "reference_date",
            "evidence_population",
            "source_url",
            "source_locator",
            "accessed_on",
            "evidence_limitation",
        },
        "regional_state_shares_2025.csv",
    )
    require_columns(
        state_packages,
        {
            "state_id",
            "tabula_stage",
            "physical_parameter_mode",
            "U_facade_W_m2K",
            "U_roof_W_m2K",
            "U_floor_W_m2K",
            "U_window_W_m2K",
            "U_door_W_m2K",
            "v50_m3_h_m2",
            "ventilation_system",
            "hrv_eta",
            "summer_bypass",
            "source_url",
            "source_locator",
            "accessed_on",
            "applicability_note",
        },
        "renovation_physical_state_packages_TABULA.csv",
    )
    require_unique(state_shares, ["region", "state_id"], "regional state shares")
    require_unique(state_packages, ["state_id"], "physical state packages")
    expected_states = set(STATE_ORDER)
    if set(state_shares["state_id"]) != expected_states:
        raise ValueError("Regional state-share states differ from the three model states")
    if set(state_packages["state_id"]) != expected_states:
        raise ValueError("Physical package states differ from the three model states")
    if set(state_shares["region"]) != set(REGIONS.values()):
        raise ValueError("Regional state shares do not cover the three modeled regions")
    for region, group in state_shares.groupby("region", sort=False):
        if len(group) != 3:
            raise ValueError(f"{region} must contain three state-share rows")
        if not math.isclose(
            pd.to_numeric(group["regional_state_share"]).sum(),
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{region} state shares do not sum to one")
        numeric = group.set_index("state_id")
        for state_id, row in numeric.iterrows():
            share = float(row["regional_state_share"])
            lower = float(row["regional_state_share_lower"])
            upper = float(row["regional_state_share_upper"])
            if not 0 <= lower <= share <= upper <= 1:
                raise ValueError(f"{region}/{state_id} has invalid share bounds")
            if not math.isclose(
                float(row["regional_state_percent"]), 100 * share, abs_tol=1e-12
            ):
                raise ValueError(f"{region}/{state_id} share and percent differ")
            if not math.isclose(
                float(row["regional_state_percent_lower"]),
                100 * lower,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(row["regional_state_percent_upper"]),
                100 * upper,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{region}/{state_id} bound units differ")
        existing = numeric.loc["TABULA_existing"]
        standard = numeric.loc["TABULA_standard_B_proxy"]
        advanced = numeric.loc["TABULA_advanced_A_proxy"]
        expected_lower = 1 - float(standard["regional_state_share_upper"]) - float(
            advanced["regional_state_share_upper"]
        )
        expected_upper = 1 - float(standard["regional_state_share_lower"]) - float(
            advanced["regional_state_share_lower"]
        )
        if not math.isclose(
            float(existing["regional_state_share_lower"]),
            expected_lower,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(existing["regional_state_share_upper"]),
            expected_upper,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{region} residual-existing interval is inconsistent")


def build_renovation_layers(
    regional_matrix: pd.DataFrame,
    state_shares: pd.DataFrame,
    state_packages: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_state_inputs(state_shares, state_packages)
    shares = state_shares.copy()
    packages = state_packages.copy()
    for field in (
        "regional_state_share",
        "regional_state_percent",
        "regional_state_share_lower",
        "regional_state_share_upper",
        "regional_state_percent_lower",
        "regional_state_percent_upper",
        "rounding_half_width_pp",
    ):
        shares[field] = pd.to_numeric(shares[field], errors="raise")

    states = regional_matrix.merge(
        shares,
        on="region",
        how="left",
        validate="many_to_many",
    ).merge(
        packages,
        on="state_id",
        how="left",
        validate="many_to_one",
        suffixes=("_evidence", "_package"),
    )
    if len(states) != 225:
        raise ValueError(f"Renovation state layer must have 225 rows; found {len(states)}")

    states["baseline_year"] = YEAR
    states["regional_state_dwellings_2025"] = (
        states["regional_number_of_dwellings"]
        * states["regional_state_share"]
    )
    states["regional_state_dwellings_2025_lower"] = (
        states["regional_number_of_dwellings"]
        * states["regional_state_share_lower"]
    )
    states["regional_state_dwellings_2025_upper"] = (
        states["regional_number_of_dwellings"]
        * states["regional_state_share_upper"]
    )
    states["state_allocation_assumption"] = (
        "independence/null assumption: P(state | region, archetype) = "
        "P(state | region)"
    )
    states["state_allocation_evidence_status"] = (
        "regional marginal calibrated; archetype-state joint distribution unavailable"
    )
    states["eligible_for_standard_transition"] = (
        (states["state_id"] == "TABULA_existing")
        & states["construction_period"].isin(MEDIUM_ELIGIBLE_PERIODS)
    )
    states["eligible_for_advanced_transition"] = states["state_id"].isin(
        ELIGIBLE_STATES
    )
    states["transition_eligibility_basis"] = (
        "VITO EPB2010 standard measures apply through 1991-2005; "
        "post-2005 existing cells retain the complete standard calibration "
        "state and remain eligible for advanced renovation"
    )

    fixed_mask = (
        states["physical_parameter_mode"] != "age_and_type_specific"
    )
    physical_fields = [
        "U_facade_W_m2K",
        "U_roof_W_m2K",
        "U_floor_W_m2K",
        "U_window_W_m2K",
        "U_door_W_m2K",
        "v50_m3_h_m2",
    ]
    for field in physical_fields:
        package_field = f"{field}_package"
        evidence_field = f"{field}_evidence"
        if package_field in states.columns and evidence_field in states.columns:
            states[field] = states[evidence_field]
            states.loc[fixed_mask, field] = states.loc[fixed_mask, package_field]
        elif package_field in states.columns:
            states.loc[fixed_mask, field] = states.loc[fixed_mask, package_field]
    states["state_evidence_source_url"] = states["source_url_evidence"]
    states["state_evidence_source_locator"] = states["source_locator_evidence"]
    states["state_evidence_accessed_on"] = states["accessed_on_evidence"]
    states["state_physical_source_url"] = states["source_url_package"]
    states["state_physical_source_locator"] = states["source_locator_package"]
    states["state_physical_accessed_on"] = states["accessed_on_package"]
    drop_suffix_columns = [
        column
        for column in states.columns
        if column.endswith("_package") or column.endswith("_evidence")
    ]
    states = states.drop(columns=drop_suffix_columns)

    derived_fields = [
        "q50_m3_h",
        "n50_h_1",
        "infiltration_n_factor",
        "infiltration_airflow_normal_m3_h",
        "infiltration_ach_normal_h_1",
        "transmission_heat_loss_H_tr_W_K",
        "infiltration_heat_loss_H_inf_W_K",
        "specific_heat_loss_z_W_m2K",
    ]
    states = states.drop(
        columns=[column for column in derived_fields if column in states.columns]
    )
    states = add_heat_loss_fields(states)

    states["_region_order"] = states["region"].map(
        {region: index for index, region in enumerate(REGIONS.values())}
    )
    states["_state_order"] = states["state_id"].map(STATE_ORDER)
    states = states.sort_values(
        ["_region_order", "TABULA_type_number", "_state_order"]
    ).reset_index(drop=True)

    allocation = states[states["eligible_for_advanced_transition"]].copy()
    allocation = allocation.sort_values(
        [
            "_region_order",
            "specific_heat_loss_z_W_m2K",
            "_state_order",
            "TABULA_type_number",
        ],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    allocation["priority_rank_for_advanced_within_region"] = (
        allocation.groupby("region", sort=False).cumcount() + 1
    )
    allocation["priority_rank_for_medium_within_region"] = pd.NA
    medium_mask = allocation["eligible_for_standard_transition"]
    allocation.loc[medium_mask, "priority_rank_for_medium_within_region"] = (
        allocation.loc[medium_mask]
        .groupby("region", sort=False)
        .cumcount()
        .add(1)
        .astype("Int64")
    )
    allocation["priority_metric"] = "specific_heat_loss_z_W_m2K_descending"
    allocation["priority_tie_break"] = (
        "current-state z then physical-state order then TABULA type number"
    )
    allocation["eligible_transition_destinations"] = "TABULA_advanced_A_proxy"
    allocation.loc[medium_mask, "eligible_transition_destinations"] = (
        "TABULA_standard_B_proxy|TABULA_advanced_A_proxy"
    )

    states = states.drop(columns=["_region_order", "_state_order"])
    allocation = allocation.drop(columns=["_region_order", "_state_order"])

    require_unique(
        states, ["region", "archetype_id", "state_id"], "renovation state layer"
    )
    require_unique(
        allocation,
        ["region", "archetype_id", "state_id"],
        "renovation allocation layer",
    )
    if len(allocation) != 150:
        raise ValueError(
            f"Renovation allocation layer must have 150 rows; found {len(allocation)}"
        )

    for (region, archetype_id), group in states.groupby(
        ["region", "archetype_id"], sort=False
    ):
        stock = float(group["regional_number_of_dwellings"].iloc[0])
        if not math.isclose(
            group["regional_state_dwellings_2025"].sum(),
            stock,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{region}/{archetype_id} state dwellings do not reconstruct stock"
            )
    for region, group in states.groupby("region", sort=False):
        modelled = float(group["regional_modelled_stock_dwellings"].iloc[0])
        for state_id, state_group in group.groupby("state_id", sort=False):
            calibrated_share = float(state_group["regional_state_share"].iloc[0])
            realised_share = (
                state_group["regional_state_dwellings_2025"].sum() / modelled
            )
            if not math.isclose(
                realised_share, calibrated_share, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{region}/{state_id} does not reproduce its calibrated share"
                )
        advanced_ranks = allocation.loc[
            allocation["region"] == region,
            "priority_rank_for_advanced_within_region",
        ].tolist()
        if advanced_ranks != list(range(1, 51)):
            raise ValueError(f"{region} advanced priority ranks are incomplete")
        medium_ranks = (
            allocation.loc[
                (allocation["region"] == region)
                & allocation["eligible_for_standard_transition"],
                "priority_rank_for_medium_within_region",
            ]
            .astype(int)
            .tolist()
        )
        if medium_ranks != list(range(1, 21)):
            raise ValueError(f"{region} medium priority ranks are incomplete")
    if (
        states[
            [
                "q50_m3_h",
                "infiltration_airflow_normal_m3_h",
                "specific_heat_loss_z_W_m2K",
            ]
        ]
        <= 0
    ).any().any():
        raise ValueError("Physical state layer contains non-positive heat-loss fields")
    return states, allocation


def write_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.12g")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the regional Statbel stock matrix, calibrated 2025 TABULA "
            "state layer, and heat-loss priority allocation."
        )
    )
    for name in (
        "statbel",
        "joint_census",
        "base_matrix",
        "state_shares",
        "state_packages",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=Path,
            default=DEFAULT_PATHS[name],
        )
    for name in (
        "regional_split",
        "joint_distribution",
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
    input_names = (
        "statbel",
        "joint_census",
        "base_matrix",
        "state_shares",
        "state_packages",
    )
    missing = [
        str(getattr(args, name))
        for name in input_names
        if not getattr(args, name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing regional-pipeline inputs: {missing}")

    statbel = load_statbel(args.statbel)
    hc37 = load_hc37(args.joint_census)
    base = pd.read_csv(args.base_matrix)
    state_shares = pd.read_csv(args.state_shares)
    state_packages = pd.read_csv(args.state_packages)

    regional_split = build_regional_split(statbel)
    joint_distribution = build_regional_joint_distribution(statbel, hc37)
    regional_matrix = build_regional_matrix(
        base, regional_split, joint_distribution
    )
    state_layer, allocation_layer = build_renovation_layers(
        regional_matrix, state_shares, state_packages
    )

    write_csv(args.regional_split_output, regional_split)
    write_csv(args.joint_distribution_output, joint_distribution)
    write_csv(args.regional_matrix_output, regional_matrix)
    write_csv(args.state_layer_output, state_layer)
    write_csv(args.allocation_layer_output, allocation_layer)

    print("Regional pipeline validation passed")
    print(f"- regional dwelling-type rows: {len(regional_split)}")
    print(f"- regional dwelling-type x period rows: {len(joint_distribution)}")
    print(f"- regional archetype rows: {len(regional_matrix)}")
    print(f"- calibrated 2025 state rows: {len(state_layer)}")
    print(f"- eligible priority rows: {len(allocation_layer)}")
    print(
        "- modeled R1-R4 dwellings: "
        f"{int(regional_split['modelled_residential_dwellings_R1_R4'].sum()):,}"
    )
    print(
        "- excluded R5/R6 dwellings: "
        f"{int(regional_split['commercial_other_dwellings_R5_R6'].sum()):,}"
    )
    print(
        "- infiltration conversion: q50=v50*A_env; "
        "normal-pressure annual-average airflow=q50/20"
    )
    print(
        "- transition priority: descending current-state "
        "specific_heat_loss_z_W_m2K within each region"
    )


if __name__ == "__main__":
    main()
