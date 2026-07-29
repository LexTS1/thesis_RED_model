"""Build the regional dwelling-type x construction-period distribution.

The model combines two official Statbel joint sources:

* Cadastral 2025 T3.* building counts by R1-R3 type and construction period.
  Each type-specific age profile is scaled to its 2025 T8 dwelling total.
* Census 2021 HC37 conventional-dwelling counts by building size and
  construction period. The RES2 + RES3+ profile is scaled to the 2025 T8 R4
  apartment-dwelling total.

The resulting type x age cells retain the official association between type
and construction period. They do not use a product of independent marginals.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


YEAR = 2025
STAT_DWELLINGS = "T8"
R1_R4 = ("R1", "R2", "R3", "R4")
R5_R6 = ("R5", "R6")

REGIONS = {
    "02000": "Flemish Region",
    "03000": "Walloon Region",
    "04000": "Brussels-Capital Region",
}

TYPE_CODE_TO_ARCHETYPE = {
    "R1": "Terraced house",
    "R2": "Semi-detached house",
    "R3": "Detached house",
}

TABULA_PERIODS = (
    "pre-1946",
    "1946-1970",
    "1971-1990",
    "1991-2005",
    "post-2005",
)

APARTMENT_SPLIT_NOTE = (
    "Statbel R4 apartment dwellings are split 50/50 between Apartment, "
    "enclosed and Apartment, exposed because apartment position is not "
    "identified."
)

JOINT_DISTRIBUTION_METHOD = (
    "official_Statbel_type_by_age_profiles_scaled_to_2025_type_totals"
)

HOUSE_PROFILE_SOURCE = (
    "Statbel 2025 cadastral T3.* building counts jointly classified by R1-R3 "
    "building type and construction period"
)

APARTMENT_PROFILE_SOURCE = (
    "Statbel Census 2021 HC37 conventional dwellings in residential buildings "
    "with 2 or 3+ dwellings, jointly classified by building size and "
    "construction period"
)

HOUSE_PROFILE_METHOD = (
    "Known-period T3.* building counts are rebinned to TABULA periods; the "
    "1982-1991 and 2002-2011 source bins are allocated uniformly across years "
    "at the 1990 and 2005 TABULA cut-offs. The resulting age profile is scaled "
    "to the region/type 2025 T8 dwelling total."
)

APARTMENT_PROFILE_METHOD = (
    "Occupied and unoccupied HC37 conventional dwellings in RES2 and RES3+ are "
    "summed, unknown construction periods are excluded, and the exact HC37 "
    "2001-2005/2006-2010 split is retained. The resulting age profile is scaled "
    "to the regional 2025 T8 R4 dwelling total and then split 50/50 by "
    "apartment position."
)

# Statbel's cadastral construction-period bins do not exactly match the TABULA
# boundaries in two places. The coefficients below allocate the affected broad
# bins in proportion to their number of calendar years.
CADASTRAL_PERIOD_WEIGHTS = {
    "pre-1946": {
        "T3.1": 1.0,  # before 1900
        "T3.2": 1.0,  # 1900-1918
        "T3.3": 1.0,  # 1919-1945
    },
    "1946-1970": {
        "T3.4": 1.0,  # 1946-1961
        "T3.5": 1.0,  # 1962-1970
    },
    "1971-1990": {
        "T3.6": 1.0,  # 1971-1981
        "T3.7.1": 0.9,  # 1982-1990: 9 of 10 years in 1982-1991
    },
    "1991-2005": {
        "T3.7.1": 0.1,  # 1991: 1 of 10 years in 1982-1991
        "T3.7.2": 1.0,  # 1992-2001
        "T3.7.3": 0.4,  # 2002-2005: 4 of 10 years in 2002-2011
    },
    "post-2005": {
        "T3.7.3": 0.6,  # 2006-2011: 6 of 10 years in 2002-2011
        "T3.7.4": 1.0,  # after 2011
    },
}

HC37_PERIOD_MAP = {
    "pre-1946": ("-1918", "1919-1945"),
    "1946-1970": ("1946-1960", "1961-1970"),
    "1971-1990": ("1971-1980", "1981-1990"),
    "1991-2005": ("1991-2000", "2001-2005"),
    "post-2005": ("2006-2010", "2011-2015", "2016+"),
}


def require_columns(data: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = columns - set(data.columns)
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {sorted(missing)}")


def clean_code(value: object, width: int | None = None) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(width) if width else text


def load_cadastral(path: Path) -> pd.DataFrame:
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


def load_hc37(path: Path) -> pd.DataFrame:
    data = pd.read_excel(path, sheet_name="TF_CENSUS_2021_HC37_1")
    require_columns(
        data,
        {
            "CD_REFNIS_LVL_1",
            "CD_TOB_LVL_1",
            "CD_TOB_LVL_2",
            "CD_OCS",
            "CD_POC",
            "MS_LOGEMENTS",
        },
        path.name,
    )
    data = data.copy()
    data["CD_REFNIS_LVL_1"] = data["CD_REFNIS_LVL_1"].map(
        lambda value: clean_code(value, 4)
    )
    for column in ("CD_TOB_LVL_1", "CD_TOB_LVL_2", "CD_OCS", "CD_POC"):
        data[column] = data[column].map(clean_code).str.upper()
    data["MS_LOGEMENTS"] = pd.to_numeric(data["MS_LOGEMENTS"], errors="raise")
    return data


def statbel_value(
    data: pd.DataFrame,
    region_code: str,
    building_code: str,
    stat_type: str = STAT_DWELLINGS,
) -> float:
    selected = data[
        (data["CD_YEAR"] == YEAR)
        & (data["CD_REFNIS"] == region_code)
        & (data["CD_STAT_TYPE"] == stat_type)
        & (data["CD_BUILDING_TYPE"] == building_code)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one Statbel row for "
            f"region={region_code}, building_type={building_code}, "
            f"stat_type={stat_type}; found {len(selected)}"
        )
    return float(selected.iloc[0]["MS_VALUE"])


def cadastral_house_period_profile(
    cadastral: pd.DataFrame, region_code: str, building_code: str
) -> tuple[dict[str, float], dict[str, float]]:
    source_codes = {
        source_code
        for weights in CADASTRAL_PERIOD_WEIGHTS.values()
        for source_code in weights
    }
    source_values = {
        source_code: statbel_value(
            cadastral, region_code, building_code, source_code
        )
        for source_code in source_codes
    }
    period_values = {
        period: sum(
            source_values[source_code] * weight
            for source_code, weight in source_weights.items()
        )
        for period, source_weights in CADASTRAL_PERIOD_WEIGHTS.items()
    }
    total = sum(period_values.values())
    if total <= 0:
        raise ValueError(
            f"No known-period cadastral buildings for {region_code} {building_code}"
        )
    shares = {period: period_values[period] / total for period in TABULA_PERIODS}
    return period_values, shares


def hc37_apartment_period_profile(
    hc37: pd.DataFrame, region_code: str
) -> tuple[dict[str, float], dict[str, float]]:
    selected = hc37[
        (hc37["CD_REFNIS_LVL_1"] == region_code[-4:])
        & (hc37["CD_TOB_LVL_1"] == "RES")
        & (hc37["CD_TOB_LVL_2"].isin(("RES2", "RES3+")))
        & (hc37["CD_OCS"].isin(("DW_OC", "DW_NOC")))
        & (hc37["CD_POC"] != "UNK")
    ]
    if selected.empty:
        raise ValueError(f"No HC37 apartment-proxy rows for region {region_code}")
    by_period = selected.groupby("CD_POC", sort=False)["MS_LOGEMENTS"].sum()
    required = {
        source_period
        for source_periods in HC37_PERIOD_MAP.values()
        for source_period in source_periods
    }
    missing = required - set(by_period.index)
    if missing:
        raise ValueError(
            f"HC37 apartment profile for {region_code} is missing: {sorted(missing)}"
        )
    period_values = {
        period: float(by_period.loc[list(source_periods)].sum())
        for period, source_periods in HC37_PERIOD_MAP.items()
    }
    total = sum(period_values.values())
    if total <= 0:
        raise ValueError(f"No known-period HC37 apartment dwellings for {region_code}")
    shares = {period: period_values[period] / total for period in TABULA_PERIODS}
    return period_values, shares


def build_regional_joint_distribution(
    cadastral: pd.DataFrame, hc37: pd.DataFrame
) -> pd.DataFrame:
    """Return the 75 regional archetype cells before physical-data merging."""
    rows: list[dict[str, object]] = []
    for region_code, region in REGIONS.items():
        for building_code in ("R1", "R2", "R3"):
            source_values, period_shares = cadastral_house_period_profile(
                cadastral, region_code, building_code
            )
            dwelling_type = TYPE_CODE_TO_ARCHETYPE[building_code]
            target_dwellings = statbel_value(
                cadastral, region_code, building_code
            )
            for period in TABULA_PERIODS:
                rows.append(
                    {
                        "region": region,
                        "statbel_building_type_code": building_code,
                        "dwelling_type": dwelling_type,
                        "construction_period": period,
                        "source_profile_measure": "2025 cadastral buildings",
                        "source_profile_value": source_values[period],
                        "source_profile_share_within_type": period_shares[period],
                        "target_2025_dwellings_archetype_type": target_dwellings,
                        "regional_number_of_dwellings": (
                            target_dwellings * period_shares[period]
                        ),
                        "joint_distribution_source": HOUSE_PROFILE_SOURCE,
                        "joint_distribution_method_detail": HOUSE_PROFILE_METHOD,
                        "regionalisation_method": JOINT_DISTRIBUTION_METHOD,
                        "apartment_position_split_assumption": "",
                    }
                )

        source_values, period_shares = hc37_apartment_period_profile(
            hc37, region_code
        )
        target_r4_dwellings = statbel_value(cadastral, region_code, "R4")
        for dwelling_type in ("Apartment, enclosed", "Apartment, exposed"):
            for period in TABULA_PERIODS:
                rows.append(
                    {
                        "region": region,
                        "statbel_building_type_code": "R4",
                        "dwelling_type": dwelling_type,
                        "construction_period": period,
                        "source_profile_measure": (
                            "2021 conventional dwellings in RES2+RES3+"
                        ),
                        "source_profile_value": 0.5 * source_values[period],
                        "source_profile_share_within_type": period_shares[period],
                        "target_2025_dwellings_archetype_type": (
                            0.5 * target_r4_dwellings
                        ),
                        "regional_number_of_dwellings": (
                            0.5 * target_r4_dwellings * period_shares[period]
                        ),
                        "joint_distribution_source": APARTMENT_PROFILE_SOURCE,
                        "joint_distribution_method_detail": (
                            APARTMENT_PROFILE_METHOD
                        ),
                        "regionalisation_method": JOINT_DISTRIBUTION_METHOD,
                        "apartment_position_split_assumption": (
                            APARTMENT_SPLIT_NOTE
                        ),
                    }
                )

    result = pd.DataFrame(rows)
    keys = ["region", "dwelling_type", "construction_period"]
    if result.duplicated(keys, keep=False).any():
        raise ValueError("Regional joint distribution contains duplicate cells")
    if len(result) != 75:
        raise ValueError(
            f"Regional joint distribution must have 75 rows; found {len(result)}"
        )

    for (region, dwelling_type), group in result.groupby(
        ["region", "dwelling_type"], sort=False
    ):
        target = float(group["target_2025_dwellings_archetype_type"].iloc[0])
        actual = float(group["regional_number_of_dwellings"].sum())
        if not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"{region} {dwelling_type} does not reconstruct its T8 target: "
                f"{actual} != {target}"
            )
    return result


def regional_stock_totals(cadastral: pd.DataFrame) -> tuple[int, int]:
    modelled = round(
        sum(
            statbel_value(cadastral, region_code, code)
            for region_code in REGIONS
            for code in R1_R4
        )
    )
    all_types = round(
        sum(
            statbel_value(cadastral, region_code, code)
            for region_code in REGIONS
            for code in (*R1_R4, *R5_R6)
        )
    )
    return int(modelled), int(all_types)
