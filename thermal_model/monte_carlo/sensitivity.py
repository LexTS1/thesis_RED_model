"""Prospectively frozen structural-sensitivity screen for Gate 5.

The screen is deliberately separate from the authoritative central-production
run.  It uses the same validated stock runner and post-processor, but its
restricted weather sample and unverified occupant-seed count can only receive
the ``WORKFLOW_CHECK_ONLY`` execution qualification.  A separate authenticated
summary records whether the *paired structural deltas* have stabilised.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from thermal_model.validation import load_unique_archetype_states

from .contracts import MonteCarloContractError, canonical_sha256
from .convergence_runner import PANEL_SPECS
from .design import make_seed_bank, ordered_seed_bank_sha256
from .postprocess import (
    MODEL_SCENARIO_OUTPUT_FILENAME,
    POSTPROCESS_SUMMARY_FILENAME,
    postprocess_production_results,
    postprocessing_status,
)
from .runner import execute_streaming_stock_design, streaming_stock_status
from .scenarios import model_scenario_sha256, resolve_model_scenario
from .weather import load_weather_catalog, load_weather_members


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRUCTURAL_SENSITIVITY_BASE_DIR = (
    PROJECT_ROOT / "thermal_model/data/monte_carlo"
)
STRUCTURAL_SENSITIVITY_CONTRACT_VERSION = (
    "gate5_structural_sensitivity_screen_v1"
)
STRUCTURAL_SENSITIVITY_SUMMARY_VERSION = (
    "gate5_structural_sensitivity_decision_v1"
)
SENSITIVITY_CONTRACT_FILENAME = "structural_sensitivity_contract.json"
SENSITIVITY_RESULTS_FILENAME = "structural_sensitivity_stability.csv"
SENSITIVITY_SUMMARY_FILENAME = "structural_sensitivity_summary.json"

SENSITIVITY_MASTER_SEED: Final[int] = 20250808
SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS: Final[tuple[int, ...]] = (40, 80, 160)
SENSITIVITY_PROSPECTIVE_CHECKPOINTS: Final[tuple[int, ...]] = (10, 20, 40, 80, 160)
SENSITIVITY_RELATIVE_TOLERANCE: Final[float] = 0.05
SENSITIVITY_REQUIRED_CONSECUTIVE_EXPANSIONS: Final[int] = 2
SENSITIVITY_STATISTICS: Final[tuple[str, ...]] = ("mean", "median", "p95")
SENSITIVITY_METRIC_FLOORS: Final[dict[str, float]] = {
    "heating_intensity_kWh_m2": 1.0,
    "cooling_intensity_kWh_m2": 1.0,
    "peak_heating_W": 100.0,
    "peak_cooling_W": 100.0,
}

# The order is part of the contract.  ``central`` is included locally so every
# structural endpoint is an exact common-random-number delta, independent of a
# separately executed production run.
SENSITIVITY_MODEL_SCENARIOS: Final[tuple[str, ...]] = (
    "central",
    "infiltration_half",
    "infiltration_one_and_half",
    "mass_light",
    "mass_heavy",
    "shading_unshaded",
)
SENSITIVITY_COMPARISON_SCENARIOS: Final[tuple[str, ...]] = tuple(
    item for item in SENSITIVITY_MODEL_SCENARIOS if item != "central"
)
SENSITIVITY_CLIMATE_SCENARIOS: Final[tuple[str, ...]] = (
    "rcp_2_6",
    "rcp_4_5",
    "rcp_8_5",
)

# Selection was frozen before structural simulations.  Only RCP4.5 forcing was
# inspected to choose the six observed-year strata; those same years are then
# replicated under every RCP.  This avoids choosing a different weather sample
# for each climate pathway.
SENSITIVITY_WEATHER_STRATA: Final[tuple[dict[str, Any], ...]] = (
    {
        "observed_pvgis_year": 2010,
        "role": "maximum_HDD20",
        "selection_metric": "annual heating-degree days below 20 C",
        "selection_value": 3770.133946111267,
        "selection_unit": "K_day",
    },
    {
        "observed_pvgis_year": 2013,
        "role": "minimum_facade_irradiance",
        "selection_metric": "annual sum of four facade irradiance series",
        "selection_value": 2508110.5789812105,
        "selection_unit": "Wh_m2_orientation_sum",
    },
    {
        "observed_pvgis_year": 2015,
        "role": "standardised_four_metric_medoid",
        "selection_metric": (
            "minimum Euclidean distance to the RCP4.5 centroid after z-scoring "
            "HDD20, CDD26, maximum outdoor temperature and four-facade irradiance"
        ),
        "selection_value": 0.7535086517087153,
        "selection_unit": "dimensionless_standardised_distance",
    },
    {
        "observed_pvgis_year": 2019,
        "role": "maximum_hourly_outdoor_temperature",
        "selection_metric": "maximum hourly outdoor dry-bulb temperature",
        "selection_value": 38.1812933842,
        "selection_unit": "degC",
    },
    {
        "observed_pvgis_year": 2020,
        "role": "maximum_CDD26",
        "selection_metric": "annual cooling-degree days above 26 C",
        "selection_value": 34.04258150989584,
        "selection_unit": "K_day",
    },
    {
        "observed_pvgis_year": 2022,
        "role": "maximum_facade_irradiance",
        "selection_metric": "annual sum of four facade irradiance series",
        "selection_value": 2930206.1567293904,
        "selection_unit": "Wh_m2_orientation_sum",
    },
)
SENSITIVITY_WEATHER_YEARS: Final[tuple[int, ...]] = tuple(
    int(item["observed_pvgis_year"]) for item in SENSITIVITY_WEATHER_STRATA
)

SENSITIVITY_PANEL_CELLS: Final[tuple[dict[str, str], ...]] = tuple(
    {
        "demand_role": str(item["demand_role"]),
        "archetype_id": str(item["archetype_id"]),
        "state_id": str(item["state_id"]),
    }
    for item in PANEL_SPECS
)


def default_structural_sensitivity_output_dir(target_seed_count: int = 40) -> Path:
    """Return the stage-specific default without creating it."""

    stage = _validate_stage(target_seed_count)
    return DEFAULT_STRUCTURAL_SENSITIVITY_BASE_DIR / (
        f"structural_sensitivity_screen_n{stage:03d}"
    )


def _validate_stage(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise MonteCarloContractError(
            "Structural-sensitivity target_seed_count must be an integer."
        )
    stage = int(value)
    if stage not in SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS:
        raise MonteCarloContractError(
            "Structural-sensitivity stages are prospectively restricted to "
            f"{SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS}; got {stage}."
        )
    return stage


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonteCarloContractError(f"Cannot read {label} JSON {path}.") from exc
    if not isinstance(payload, dict):
        raise MonteCarloContractError(f"{label} JSON must contain an object: {path}.")
    return payload


def _require_file_sha256(path: Path, expected: Any, *, label: str) -> str:
    digest = str(expected).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MonteCarloContractError(f"{label} ledger contains an invalid SHA-256.")
    if not path.is_file():
        raise MonteCarloContractError(f"{label} is missing: {path}.")
    actual = _sha256_file(path)
    if actual != digest:
        raise MonteCarloContractError(
            f"{label} checksum mismatch: expected {digest}, got {actual}."
        )
    return actual


def _active_checkpoints(target_seed_count: int) -> tuple[int, ...]:
    stage = _validate_stage(target_seed_count)
    return tuple(value for value in SENSITIVITY_PROSPECTIVE_CHECKPOINTS if value <= stage)


def _selection_payload() -> dict[str, Any]:
    return {
        "selection_basis_climate_scenario_id": "rcp_4_5",
        "selection_metric_definitions": {
            "HDD20_K_day": "sum(max(20 - T_out_C, 0)) / 24 over hourly forcing",
            "CDD26_K_day": "sum(max(T_out_C - 26, 0)) / 24 over hourly forcing",
            "maximum_outdoor_temperature_degC": "max(T_out_C)",
            "facade_irradiance_orientation_sum_Wh_m2": (
                "sum over hours and the south, east, west and north facade "
                "irradiance series"
            ),
            "standardised_medoid_distance": (
                "Euclidean distance to zero after each of the four RCP4.5 "
                "metrics is z-scored across the 18 years using population "
                "standard deviation (ddof=0)"
            ),
        },
        "weather_strata": [dict(item) for item in SENSITIVITY_WEATHER_STRATA],
        "replication_climate_scenario_ids": list(SENSITIVITY_CLIMATE_SCENARIOS),
        "model_scenario_ids": list(SENSITIVITY_MODEL_SCENARIOS),
        "representative_delta_stability_panel": [
            dict(item) for item in SENSITIVITY_PANEL_CELLS
        ],
        "master_seed": SENSITIVITY_MASTER_SEED,
        "prospective_seed_checkpoints": list(SENSITIVITY_PROSPECTIVE_CHECKPOINTS),
        "statistics": list(SENSITIVITY_STATISTICS),
        "metrics_and_absolute_floors": dict(SENSITIVITY_METRIC_FLOORS),
        "relative_tolerance": SENSITIVITY_RELATIVE_TOLERANCE,
        "required_consecutive_passing_expansions": (
            SENSITIVITY_REQUIRED_CONSECUTIVE_EXPANSIONS
        ),
        "median_sign_guard": (
            "fail a median criterion when both adjacent estimates exceed the "
            "metric floor in magnitude and their signs differ"
        ),
    }


SENSITIVITY_SELECTION_SHA256: Final[str] = canonical_sha256(_selection_payload())


def _design_basis_sha256(design: Mapping[str, Any]) -> str:
    """Hash the experiment basis while excluding the staged seed prefix."""

    excluded = {
        "design_sha256",
        "occupant_seeds",
        "occupant_seed_bank_sha256",
        "expected_run_count",
    }
    return canonical_sha256(
        {key: value for key, value in design.items() if key not in excluded}
    )


def _load_contract(root: Path) -> dict[str, Any]:
    path = root / SENSITIVITY_CONTRACT_FILENAME
    contract = _read_json(path, label="structural-sensitivity contract")
    if contract.get("structural_sensitivity_contract_version") != (
        STRUCTURAL_SENSITIVITY_CONTRACT_VERSION
    ):
        raise MonteCarloContractError(
            "Structural-sensitivity contract version is missing or stale."
        )
    declared = str(contract.get("structural_sensitivity_contract_sha256", ""))
    unsigned = {
        key: value
        for key, value in contract.items()
        if key != "structural_sensitivity_contract_sha256"
    }
    if canonical_sha256(unsigned) != declared:
        raise MonteCarloContractError(
            "Structural-sensitivity contract content does not reproduce its checksum."
        )
    stage = _validate_stage(contract.get("target_seed_count"))
    if contract.get("selection_contract_sha256") != SENSITIVITY_SELECTION_SHA256:
        raise MonteCarloContractError(
            "Structural-sensitivity contract differs from the frozen selection/rule."
        )
    expected_seeds = make_seed_bank(stage, master_seed=SENSITIVITY_MASTER_SEED)
    if tuple(contract.get("occupant_seeds", ())) != expected_seeds or contract.get(
        "occupant_seed_bank_sha256"
    ) != ordered_seed_bank_sha256(expected_seeds):
        raise MonteCarloContractError(
            "Structural-sensitivity contract changes the frozen nested seed prefix."
        )
    if tuple(contract.get("active_seed_checkpoints", ())) != _active_checkpoints(stage):
        raise MonteCarloContractError(
            "Structural-sensitivity contract changes the active checkpoints."
        )
    audit = contract.get("weather_selection_audit")
    if not isinstance(audit, dict):
        raise MonteCarloContractError(
            "Structural-sensitivity contract has no weather-selection audit."
        )
    audit_declared = str(audit.get("weather_selection_audit_sha256", ""))
    audit_unsigned = {
        key: value
        for key, value in audit.items()
        if key != "weather_selection_audit_sha256"
    }
    if canonical_sha256(audit_unsigned) != audit_declared:
        raise MonteCarloContractError(
            "Weather-selection audit content does not reproduce its checksum."
        )
    design_path = root / "streaming_design_contract.json"
    _require_file_sha256(
        design_path,
        contract.get("streaming_design_contract_file_sha256"),
        label="bound streaming design contract",
    )
    design = _read_json(design_path, label="streaming design contract")
    unsigned_design = {
        key: value for key, value in design.items() if key != "design_sha256"
    }
    if canonical_sha256(unsigned_design) != design.get("design_sha256"):
        raise MonteCarloContractError(
            "Bound streaming design content does not reproduce its design checksum."
        )
    if design.get("design_sha256") != contract.get("streaming_design_sha256"):
        raise MonteCarloContractError(
            "Structural-sensitivity and streaming contracts have different design IDs."
        )
    if _design_basis_sha256(design) != contract.get("streaming_design_basis_sha256"):
        raise MonteCarloContractError(
            "Structural-sensitivity streaming-design basis checksum is inconsistent."
        )
    return contract


def _load_summary(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    path = root / SENSITIVITY_SUMMARY_FILENAME
    summary = _read_json(path, label="structural-sensitivity summary")
    if summary.get("structural_sensitivity_summary_version") != (
        STRUCTURAL_SENSITIVITY_SUMMARY_VERSION
    ):
        raise MonteCarloContractError(
            "Structural-sensitivity summary version is missing or stale."
        )
    declared = str(summary.get("structural_sensitivity_summary_sha256", ""))
    unsigned = {
        key: value
        for key, value in summary.items()
        if key != "structural_sensitivity_summary_sha256"
    }
    if canonical_sha256(unsigned) != declared:
        raise MonteCarloContractError(
            "Structural-sensitivity summary content does not reproduce its checksum."
        )
    if summary.get("structural_sensitivity_contract_sha256") != contract.get(
        "structural_sensitivity_contract_sha256"
    ):
        raise MonteCarloContractError(
            "Structural-sensitivity summary belongs to another screen contract."
        )
    _require_file_sha256(
        root / POSTPROCESS_SUMMARY_FILENAME,
        summary.get("postprocessing_summary_file_sha256"),
        label="bound post-processing summary",
    )
    if postprocessing_status(root).get("status") != "PASS":
        raise MonteCarloContractError(
            "The post-processing artifact set bound to the sensitivity decision is invalid."
        )
    artifacts = summary.get("output_artifacts")
    if not isinstance(artifacts, dict) or SENSITIVITY_RESULTS_FILENAME not in artifacts:
        raise MonteCarloContractError(
            "Structural-sensitivity summary has no stability-results ledger."
        )
    _require_file_sha256(
        root / SENSITIVITY_RESULTS_FILENAME,
        artifacts[SENSITIVITY_RESULTS_FILENAME].get("sha256"),
        label="structural-sensitivity stability results",
    )
    return summary


def _validate_previous_stage(
    target_seed_count: int,
    previous_stage_dir: str | Path | None,
) -> dict[str, Any] | None:
    if target_seed_count == SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS[0]:
        if previous_stage_dir is not None:
            raise MonteCarloContractError("The n=40 initial stage has no predecessor.")
        return None
    if previous_stage_dir is None:
        raise MonteCarloContractError(
            f"The n={target_seed_count} extension requires its completed predecessor directory."
        )
    previous_root = Path(previous_stage_dir).resolve()
    previous_contract = _load_contract(previous_root)
    previous_summary = _load_summary(previous_root, previous_contract)
    expected_previous = SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS[
        SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS.index(target_seed_count) - 1
    ]
    if int(previous_contract["target_seed_count"]) != expected_previous:
        raise MonteCarloContractError(
            f"The n={target_seed_count} stage requires the n={expected_previous} predecessor."
        )
    if previous_summary.get("status") != (
        f"STRUCTURAL_SENSITIVITY_SCREEN_NOT_STABLE_AT_N{expected_previous}"
    ) or int(previous_summary.get("next_stage_target_seed_count", -1)) != target_seed_count:
        raise MonteCarloContractError(
            "A structural-sensitivity extension is permitted only when its authenticated "
            "predecessor explicitly requested that next stage."
        )
    return {
        "target_seed_count": expected_previous,
        "structural_sensitivity_contract_sha256": previous_contract[
            "structural_sensitivity_contract_sha256"
        ],
        "structural_sensitivity_contract_file_sha256": _sha256_file(
            previous_root / SENSITIVITY_CONTRACT_FILENAME
        ),
        "structural_sensitivity_summary_sha256": previous_summary[
            "structural_sensitivity_summary_sha256"
        ],
        "structural_sensitivity_summary_file_sha256": _sha256_file(
            previous_root / SENSITIVITY_SUMMARY_FILENAME
        ),
        "streaming_design_basis_sha256": previous_contract[
            "streaming_design_basis_sha256"
        ],
        "weather_selection_audit_sha256": previous_contract[
            "weather_selection_audit"
        ]["weather_selection_audit_sha256"],
        "decision": previous_summary["decision"],
    }


def _selected_member_ids() -> tuple[str, ...]:
    catalog = load_weather_catalog()
    selected = catalog.loc[
        catalog["scenario"].isin(SENSITIVITY_CLIMATE_SCENARIOS)
        & catalog["observed_pvgis_year"].isin(SENSITIVITY_WEATHER_YEARS)
    ].sort_values(["scenario", "observed_pvgis_year"], kind="stable")
    expected = len(SENSITIVITY_CLIMATE_SCENARIOS) * len(SENSITIVITY_WEATHER_YEARS)
    if len(selected) != expected:
        raise MonteCarloContractError(
            f"Frozen sensitivity weather selection expected {expected} members, got {len(selected)}."
        )
    observed_rcps = tuple(sorted(selected["scenario"].astype(str).unique()))
    if observed_rcps != tuple(sorted(SENSITIVITY_CLIMATE_SCENARIOS)):
        raise MonteCarloContractError(
            "Frozen sensitivity weather selection no longer covers the three RCPs."
        )
    for scenario, group in selected.groupby("scenario", sort=True):
        years = tuple(sorted(group["observed_pvgis_year"].astype(int)))
        if years != tuple(sorted(SENSITIVITY_WEATHER_YEARS)):
            raise MonteCarloContractError(
                f"Frozen sensitivity weather years are incomplete for {scenario}: {years}."
            )
    return tuple(selected["member_id"].astype(str))


def _audit_weather_selection(selected_members: Sequence[Any]) -> dict[str, Any]:
    """Reconstruct and authenticate the frozen six-year RCP4.5 stratification."""

    catalog = load_weather_catalog()
    rcp_catalog = catalog.loc[catalog["scenario"] == "rcp_4_5"].sort_values(
        "observed_pvgis_year", kind="stable"
    )
    if len(rcp_catalog) != 18 or tuple(
        rcp_catalog["observed_pvgis_year"].astype(int)
    ) != tuple(range(2006, 2024)):
        raise MonteCarloContractError(
            "Weather-selection audit requires all 18 RCP4.5 candidates from 2006-2023."
        )
    by_id = {str(item.member_id): item for item in selected_members}
    missing_ids = tuple(
        identifier
        for identifier in rcp_catalog["member_id"].astype(str)
        if identifier not in by_id
    )
    if missing_ids:
        for item in load_weather_members(missing_ids):
            by_id[str(item.member_id)] = item
    facade_columns = (
        "I_south_W_m2",
        "I_east_W_m2",
        "I_west_W_m2",
        "I_north_W_m2",
    )
    records: list[dict[str, Any]] = []
    for catalog_row in rcp_catalog.itertuples(index=False):
        member = by_id.get(str(catalog_row.member_id))
        if member is None:
            raise MonteCarloContractError(
                f"Weather-selection audit cannot load {catalog_row.member_id}."
            )
        missing_columns = sorted(set(("T_out_C", *facade_columns)).difference(member.frame))
        if missing_columns:
            raise MonteCarloContractError(
                f"Weather-selection candidate {member.member_id} lacks {missing_columns}."
            )
        temperature = member.frame["T_out_C"].to_numpy(dtype=float)
        facade = member.frame[list(facade_columns)].to_numpy(dtype=float)
        if not np.isfinite(temperature).all() or not np.isfinite(facade).all():
            raise MonteCarloContractError(
                f"Weather-selection candidate {member.member_id} contains non-finite forcing."
            )
        records.append(
            {
                "weather_member_id": str(member.member_id),
                "observed_pvgis_year": int(member.observed_pvgis_year),
                "HDD20_K_day": float(np.maximum(20.0 - temperature, 0.0).sum() / 24.0),
                "CDD26_K_day": float(np.maximum(temperature - 26.0, 0.0).sum() / 24.0),
                "maximum_outdoor_temperature_degC": float(np.max(temperature)),
                "facade_irradiance_orientation_sum_Wh_m2": float(np.sum(facade)),
                "weather_contract_sha256": str(member.weather_contract_sha256),
                "weather_forcing_sha256": str(member.forcing_sha256),
                "member_sha256": str(member.member_sha256),
                "metadata_sha256": str(member.metadata_sha256),
            }
        )
    metrics = (
        "HDD20_K_day",
        "CDD26_K_day",
        "maximum_outdoor_temperature_degC",
        "facade_irradiance_orientation_sum_Wh_m2",
    )
    frame = pd.DataFrame.from_records(records).sort_values(
        "observed_pvgis_year", kind="stable"
    )
    values = frame.loc[:, metrics].to_numpy(dtype=float)
    standard_deviation = values.std(axis=0, ddof=0)
    if (standard_deviation <= 0.0).any():
        raise MonteCarloContractError(
            "Weather-selection audit cannot standardize a constant selection metric."
        )
    standardized = (values - values.mean(axis=0)) / standard_deviation
    frame["standardised_four_metric_distance"] = np.sqrt(
        np.square(standardized).sum(axis=1)
    )
    role_years = {
        "maximum_HDD20": int(
            frame.loc[frame["HDD20_K_day"].idxmax(), "observed_pvgis_year"]
        ),
        "minimum_facade_irradiance": int(
            frame.loc[
                frame["facade_irradiance_orientation_sum_Wh_m2"].idxmin(),
                "observed_pvgis_year",
            ]
        ),
        "standardised_four_metric_medoid": int(
            frame.loc[
                frame["standardised_four_metric_distance"].idxmin(),
                "observed_pvgis_year",
            ]
        ),
        "maximum_hourly_outdoor_temperature": int(
            frame.loc[
                frame["maximum_outdoor_temperature_degC"].idxmax(),
                "observed_pvgis_year",
            ]
        ),
        "maximum_CDD26": int(
            frame.loc[frame["CDD26_K_day"].idxmax(), "observed_pvgis_year"]
        ),
        "maximum_facade_irradiance": int(
            frame.loc[
                frame["facade_irradiance_orientation_sum_Wh_m2"].idxmax(),
                "observed_pvgis_year",
            ]
        ),
    }
    expected_role_years = {
        str(item["role"]): int(item["observed_pvgis_year"])
        for item in SENSITIVITY_WEATHER_STRATA
    }
    if role_years != expected_role_years:
        raise MonteCarloContractError(
            "Recomputed RCP4.5 weather strata differ from the prospective selection: "
            f"observed={role_years}, expected={expected_role_years}."
        )
    metric_by_role = {
        "maximum_HDD20": "HDD20_K_day",
        "minimum_facade_irradiance": "facade_irradiance_orientation_sum_Wh_m2",
        "standardised_four_metric_medoid": "standardised_four_metric_distance",
        "maximum_hourly_outdoor_temperature": "maximum_outdoor_temperature_degC",
        "maximum_CDD26": "CDD26_K_day",
        "maximum_facade_irradiance": "facade_irradiance_orientation_sum_Wh_m2",
    }
    for stratum in SENSITIVITY_WEATHER_STRATA:
        year = int(stratum["observed_pvgis_year"])
        row = frame.loc[frame["observed_pvgis_year"] == year]
        if len(row) != 1:
            raise MonteCarloContractError(
                f"Weather-selection audit cannot identify frozen year {year}."
            )
        actual = float(row.iloc[0][metric_by_role[str(stratum["role"])]])
        if not np.isclose(
            actual,
            float(stratum["selection_value"]),
            rtol=1.0e-12,
            atol=1.0e-8,
        ):
            raise MonteCarloContractError(
                f"Frozen selection metric changed for {year}: {actual} versus "
                f"{stratum['selection_value']}."
            )
    candidate_records = frame.to_dict(orient="records")
    audit_payload = {
        "selection_basis_climate_scenario_id": "rcp_4_5",
        "candidate_years": list(range(2006, 2024)),
        "candidate_records": candidate_records,
        "selected_role_to_year": role_years,
    }
    return {
        **audit_payload,
        "weather_selection_audit_sha256": canonical_sha256(audit_payload),
    }


def prepare_structural_sensitivity_screen(
    output_dir: str | Path | None = None,
    *,
    target_seed_count: int = 40,
    previous_stage_dir: str | Path | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Validate and persist one immutable screen stage without simulating."""

    stage = _validate_stage(target_seed_count)
    root = (
        default_structural_sensitivity_output_dir(stage)
        if output_dir is None
        else Path(output_dir).resolve()
    )
    contract_path = root / SENSITIVITY_CONTRACT_FILENAME
    existing_contract = _load_contract(root) if contract_path.exists() else None
    if existing_contract is not None and int(existing_contract["target_seed_count"]) != stage:
        raise MonteCarloContractError(
            f"Output directory is bound to n={existing_contract['target_seed_count']}, not n={stage}."
        )
    if existing_contract is not None and previous_stage_dir is None and stage > 40:
        previous = existing_contract.get("previous_stage")
        if not isinstance(previous, dict):
            raise MonteCarloContractError(
                "Prepared extension contract has no authenticated predecessor lineage."
            )
    else:
        previous = _validate_previous_stage(stage, previous_stage_dir)

    states = tuple(
        sorted(
            load_unique_archetype_states(),
            key=lambda item: (item.archetype_id, item.state_id),
        )
    )
    if len(states) != 75:
        raise MonteCarloContractError(
            f"Structural sensitivity requires all 75 physical stock cells; got {len(states)}."
        )
    member_ids = _selected_member_ids()
    members = tuple(load_weather_members(member_ids))
    weather_selection_audit = _audit_weather_selection(members)
    seeds = make_seed_bank(stage, master_seed=SENSITIVITY_MASTER_SEED)
    scenarios = tuple(resolve_model_scenario(item) for item in SENSITIVITY_MODEL_SCENARIOS)

    prepared = execute_streaming_stock_design(
        states,
        members,
        seeds,
        scenarios,
        output_dir=root,
        require_full_stock=True,
        require_convergence_evidence=False,
        max_workers=max_workers,
        prepare_only=True,
    )
    if prepared.get("status") != "PREPARED":
        raise MonteCarloContractError(
            "The shared stock runner did not return a prepared screen design."
        )
    design_path = root / "streaming_design_contract.json"
    design = _read_json(design_path, label="streaming design contract")
    design_basis_sha256 = _design_basis_sha256(design)
    if previous is not None and previous["streaming_design_basis_sha256"] != design_basis_sha256:
        raise MonteCarloContractError(
            "The extension changes the predecessor's weather, stock, physics, or scenario basis."
        )
    if previous is not None and previous.get("weather_selection_audit_sha256") != (
        weather_selection_audit["weather_selection_audit_sha256"]
    ):
        raise MonteCarloContractError(
            "The extension changes the authenticated RCP4.5 weather-selection candidates."
        )

    # A direct runner execution cannot be retrospectively relabelled as a
    # prospectively frozen screen.
    if not contract_path.exists() and (
        (root / "monte_carlo_summary.json").exists()
        or any((root / "partitions").glob("*/progress.json"))
        or any((root / "partitions").glob("*/partition_complete.json"))
    ):
        raise MonteCarloContractError(
            "Simulation artifacts predate the structural-sensitivity contract; refusing "
            "retrospective adoption. Use a new output directory."
        )

    scenario_by_id = {item.scenario_id: item for item in scenarios}
    role_by_year = {
        int(item["observed_pvgis_year"]): str(item["role"])
        for item in SENSITIVITY_WEATHER_STRATA
    }
    weather_records = []
    for item in design["weather_members"]:
        year = int(item["observed_pvgis_year"])
        weather_records.append({**item, "stratum_role": role_by_year[year]})

    payload = {
        "structural_sensitivity_contract_version": (
            STRUCTURAL_SENSITIVITY_CONTRACT_VERSION
        ),
        "purpose": "STRUCTURAL_SENSITIVITY_SCREEN",
        "production_qualification": "NOT_A_PRODUCTION_PASS",
        "qualification_reason": (
            "all 75 weighted stock cells are covered, but weather is a frozen "
            "six-of-eighteen stratum per RCP and occupant seeds are governed by a "
            "separate paired-delta stability rule"
        ),
        "coverage": {
            "stock_weight_coverage": "AUTHORITATIVE_2050_WEIGHTS",
            "weather_coverage": "STRATIFIED_6_OF_18_PER_RCP",
            "occupant_seed_prefix": f"N{stage}_OF_160",
            "structural_scenario_coverage": "ALL_SIX_DECLARED_SCENARIOS",
        },
        "selection_contract": _selection_payload(),
        "selection_contract_sha256": SENSITIVITY_SELECTION_SHA256,
        "weather_selection_audit": weather_selection_audit,
        "target_seed_count": stage,
        "active_seed_checkpoints": list(_active_checkpoints(stage)),
        "occupant_seeds": list(seeds),
        "occupant_seed_bank_sha256": ordered_seed_bank_sha256(seeds),
        "weather_members": weather_records,
        "model_scenarios": [
            {
                **scenario_by_id[scenario_id].definition(),
                "model_scenario_sha256": model_scenario_sha256(
                    scenario_by_id[scenario_id],
                    str(design["central_thermal_assumptions_sha256"]),
                ),
            }
            for scenario_id in SENSITIVITY_MODEL_SCENARIOS
        ],
        "streaming_design_sha256": str(design["design_sha256"]),
        "streaming_design_contract_file_sha256": _sha256_file(design_path),
        "streaming_design_basis_sha256": design_basis_sha256,
        "expected_archetype_state_count": 75,
        "expected_weather_member_count": 18,
        "expected_partition_count": 108,
        "expected_runs_per_partition": 75 * stage,
        "expected_run_count": 75 * 18 * stage * 6,
        "previous_stage": previous,
        "extension_protocol": {
            "allowed_stage_seed_counts": list(SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS),
            "rule": (
                "advance only when the immediately preceding authenticated stage "
                "is not stable and explicitly requests the next checkpoint"
            ),
            "storage": (
                "each extension uses a new stage-specific output directory and the "
                "same nested prefix; completed predecessor artifacts are immutable"
            ),
            "maximum_seed_count": 160,
        },
        "interpretation": {
            "scenarios_are_probabilities": False,
            "weather_strata_are_probability_weighted": False,
            "permitted_claim": (
                "screened paired response to declared structural assumptions over "
                "the explicitly selected weather and occupant draws"
            ),
            "forbidden_claim": "complete prediction or probability interval",
        },
    }
    contract = {
        **payload,
        "structural_sensitivity_contract_sha256": canonical_sha256(payload),
    }
    if contract_path.exists():
        existing = _load_contract(root)
        if existing != _json_ready(contract):
            raise MonteCarloContractError(
                "Output directory belongs to another structural-sensitivity contract."
            )
    else:
        _atomic_json(contract, contract_path)
        if _load_contract(root) != _json_ready(contract):
            raise MonteCarloContractError(
                "Persisted structural-sensitivity contract failed identity validation."
            )
    return {
        "status": "PREPARED",
        "purpose": "STRUCTURAL_SENSITIVITY_SCREEN",
        "production_qualification": "NOT_A_PRODUCTION_PASS",
        "output_dir": str(root),
        "target_seed_count": stage,
        "active_seed_checkpoints": list(_active_checkpoints(stage)),
        "structural_sensitivity_contract_sha256": contract[
            "structural_sensitivity_contract_sha256"
        ],
        "streaming_design_sha256": design["design_sha256"],
        "archetype_state_count": 75,
        "weather_member_count": 18,
        "model_scenario_count": 6,
        "partition_count": 108,
        "expected_run_count": 75 * 18 * stage * 6,
        "requested_max_workers": max_workers,
    }


def _statistic(values: np.ndarray, statistic: str) -> float:
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    if statistic == "p95":
        return float(np.quantile(values, 0.95))
    raise AssertionError(f"Unsupported sensitivity statistic {statistic!r}.")


def _read_panel_deltas(root: Path, contract: Mapping[str, Any]) -> pd.DataFrame:
    post_path = root / POSTPROCESS_SUMMARY_FILENAME
    post = _read_json(post_path, label="post-processing summary")
    if post.get("status") != "PASS" or post.get("source_execution_status") != (
        "WORKFLOW_CHECK_ONLY"
    ):
        raise MonteCarloContractError(
            "Structural sensitivity requires PASS post-processing of a "
            "WORKFLOW_CHECK_ONLY source execution."
        )
    if post.get("source_design_sha256") != contract.get("streaming_design_sha256"):
        raise MonteCarloContractError(
            "Post-processing and sensitivity contract refer to different designs."
        )
    artifacts = post.get("output_artifacts")
    if not isinstance(artifacts, dict) or MODEL_SCENARIO_OUTPUT_FILENAME not in artifacts:
        raise MonteCarloContractError(
            "Authenticated paired model-scenario deltas are missing."
        )
    paired_path = root / MODEL_SCENARIO_OUTPUT_FILENAME
    _require_file_sha256(
        paired_path,
        artifacts[MODEL_SCENARIO_OUTPUT_FILENAME].get("sha256"),
        label="paired model-scenario deltas",
    )

    required = {
        "archetype_id",
        "state_id",
        "climate_scenario_id",
        "weather_member_id",
        "weather_pair_id",
        "observed_pvgis_year",
        "occupant_seed",
        "baseline_model_scenario_id",
        "comparison_model_scenario_id",
        "comparison_model_scenario_axis",
        "metric",
        "baseline_value",
        "comparison_value",
        "delta",
    }
    panel_keys = {
        (item["archetype_id"], item["state_id"]) for item in SENSITIVITY_PANEL_CELLS
    }
    frames: list[pd.DataFrame] = []
    try:
        chunks = pd.read_csv(
            paired_path,
            chunksize=20_000,
            keep_default_na=False,
            float_precision="round_trip",
        )
        for chunk in chunks:
            missing = sorted(required.difference(chunk.columns))
            if missing:
                raise MonteCarloContractError(
                    f"Paired model-scenario deltas are missing columns: {missing}."
                )
            mask = pd.Series(
                [
                    (str(a), str(s)) in panel_keys
                    for a, s in zip(chunk["archetype_id"], chunk["state_id"])
                ],
                index=chunk.index,
            )
            mask &= chunk["metric"].astype(str).isin(SENSITIVITY_METRIC_FLOORS)
            selected = chunk.loc[mask].copy()
            if not selected.empty:
                frames.append(selected)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MonteCarloContractError(
            f"Cannot stream paired structural deltas {paired_path}."
        ) from exc
    if not frames:
        raise MonteCarloContractError(
            "Paired model-scenario output contains no frozen stability-panel rows."
        )
    result = pd.concat(frames, ignore_index=True)

    string_columns = (
        "archetype_id",
        "state_id",
        "climate_scenario_id",
        "weather_member_id",
        "weather_pair_id",
        "baseline_model_scenario_id",
        "comparison_model_scenario_id",
        "comparison_model_scenario_axis",
        "metric",
    )
    for column in string_columns:
        result[column] = result[column].astype(str)
    result["occupant_seed"] = pd.to_numeric(
        result["occupant_seed"], errors="raise"
    ).astype(np.int64)
    result["observed_pvgis_year"] = pd.to_numeric(
        result["observed_pvgis_year"], errors="raise"
    ).astype(int)
    for column in ("baseline_value", "comparison_value", "delta"):
        values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise MonteCarloContractError(
                f"Paired structural delta column {column!r} contains non-finite values."
            )
        result[column] = values
    if not np.allclose(
        result["comparison_value"] - result["baseline_value"],
        result["delta"],
        rtol=1.0e-11,
        atol=1.0e-9,
    ):
        raise MonteCarloContractError(
            "Paired structural deltas do not equal comparison minus central baseline."
        )
    if set(result["baseline_model_scenario_id"]) != {"central"}:
        raise MonteCarloContractError("Structural deltas change the central baseline.")
    if set(result["comparison_model_scenario_id"]) != set(
        SENSITIVITY_COMPARISON_SCENARIOS
    ):
        raise MonteCarloContractError(
            "Structural deltas do not cover exactly the five comparison scenarios."
        )
    if set(result["climate_scenario_id"]) != set(SENSITIVITY_CLIMATE_SCENARIOS):
        raise MonteCarloContractError(
            "Structural stability panel does not cover exactly the three RCPs."
        )
    if set(result["metric"]) != set(SENSITIVITY_METRIC_FLOORS):
        raise MonteCarloContractError(
            "Structural stability panel does not cover the four frozen metrics."
        )

    scenario_axes = {
        scenario_id: resolve_model_scenario(scenario_id).axis
        for scenario_id in SENSITIVITY_COMPARISON_SCENARIOS
    }
    expected_axes = result["comparison_model_scenario_id"].map(scenario_axes)
    if not result["comparison_model_scenario_axis"].eq(expected_axes).all():
        raise MonteCarloContractError(
            "Structural comparison scenario axes differ from the registry."
        )
    expected_weather = {
        str(item["weather_member_id"]): (
            str(item["climate_scenario_id"]),
            str(item["weather_pair_id"]),
            int(item["observed_pvgis_year"]),
        )
        for item in contract["weather_members"]
    }
    observed_weather = {
        str(row.weather_member_id): (
            str(row.climate_scenario_id),
            str(row.weather_pair_id),
            int(row.observed_pvgis_year),
        )
        for row in result[
            [
                "weather_member_id",
                "climate_scenario_id",
                "weather_pair_id",
                "observed_pvgis_year",
            ]
        ].drop_duplicates().itertuples(index=False)
    }
    if observed_weather != expected_weather:
        raise MonteCarloContractError(
            "Structural stability panel weather identity differs from the frozen 18 members."
        )
    expected_seeds = tuple(int(item) for item in contract["occupant_seeds"])
    if set(result["occupant_seed"].astype(int)) != set(expected_seeds):
        raise MonteCarloContractError(
            "Structural stability panel does not contain the exact staged seed prefix."
        )

    key_columns = [
        "archetype_id",
        "state_id",
        "climate_scenario_id",
        "weather_member_id",
        "occupant_seed",
        "comparison_model_scenario_id",
        "metric",
    ]
    if result.duplicated(key_columns).any():
        raise MonteCarloContractError(
            "Structural stability panel contains duplicate paired cells."
        )
    expected_rows = (
        len(SENSITIVITY_PANEL_CELLS)
        * 18
        * len(expected_seeds)
        * len(SENSITIVITY_COMPARISON_SCENARIOS)
        * len(SENSITIVITY_METRIC_FLOORS)
    )
    if len(result) != expected_rows:
        raise MonteCarloContractError(
            f"Structural stability panel expected {expected_rows} rows, got {len(result)}."
        )
    return result


def evaluate_structural_sensitivity_screen(
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen nested-prefix rule on authenticated paired deltas."""

    root = (
        default_structural_sensitivity_output_dir(40)
        if output_dir is None
        else Path(output_dir).resolve()
    )
    contract = _load_contract(root)
    stage = int(contract["target_seed_count"])
    execution = streaming_stock_status(root)
    if execution.get("status") != "WORKFLOW_CHECK_ONLY":
        raise MonteCarloContractError(
            "Structural sensitivity can be evaluated only after its complete "
            "WORKFLOW_CHECK_ONLY stock execution."
        )
    post_status = postprocessing_status(root)
    if post_status.get("status") != "PASS":
        raise MonteCarloContractError(
            "Structural sensitivity requires authenticated post-processing before evaluation."
        )
    deltas = _read_panel_deltas(root, contract)
    seeds = tuple(int(item) for item in contract["occupant_seeds"])
    seed_rank = {seed: rank for rank, seed in enumerate(seeds, start=1)}
    deltas["occupant_seed_rank"] = deltas["occupant_seed"].map(seed_rank)
    if deltas["occupant_seed_rank"].isna().any():
        raise MonteCarloContractError("A structural delta uses an undeclared seed.")

    role_by_cell = {
        (item["archetype_id"], item["state_id"]): item["demand_role"]
        for item in SENSITIVITY_PANEL_CELLS
    }
    group_columns = [
        "archetype_id",
        "state_id",
        "climate_scenario_id",
        "comparison_model_scenario_id",
        "comparison_model_scenario_axis",
    ]
    records: list[dict[str, Any]] = []
    previous: dict[tuple[Any, ...], float] = {}
    active = tuple(int(item) for item in contract["active_seed_checkpoints"])
    for checkpoint_index, checkpoint in enumerate(active):
        previous_checkpoint = active[checkpoint_index - 1] if checkpoint_index else None
        prefix = deltas.loc[deltas["occupant_seed_rank"] <= checkpoint]
        for group_key, group in prefix.groupby(group_columns, sort=True):
            identity = dict(zip(group_columns, group_key))
            expected_draws = 6 * checkpoint
            draw_counts = group.groupby("metric", sort=False).size()
            if set(draw_counts.index) != set(SENSITIVITY_METRIC_FLOORS) or not draw_counts.eq(
                expected_draws
            ).all():
                raise MonteCarloContractError(
                    f"Unbalanced paired-delta group {identity} at n={checkpoint}."
                )
            for metric, floor in SENSITIVITY_METRIC_FLOORS.items():
                values = group.loc[group["metric"] == metric, "delta"].to_numpy(dtype=float)
                for statistic in SENSITIVITY_STATISTICS:
                    value = _statistic(values, statistic)
                    key = (*group_key, metric, statistic)
                    previous_value = previous.get(key)
                    relative_change = (
                        abs(value - previous_value) / max(abs(value), float(floor))
                        if previous_value is not None
                        else np.nan
                    )
                    sign_guard = bool(
                        statistic == "median"
                        and previous_value is not None
                        and abs(previous_value) > float(floor)
                        and abs(value) > float(floor)
                        and np.sign(previous_value) != np.sign(value)
                    )
                    criterion_pass = bool(
                        previous_value is not None
                        and relative_change <= SENSITIVITY_RELATIVE_TOLERANCE
                        and not sign_guard
                    )
                    records.append(
                        {
                            **identity,
                            "demand_role": role_by_cell[
                                (identity["archetype_id"], identity["state_id"])
                            ],
                            "seed_count": checkpoint,
                            "previous_seed_count": previous_checkpoint,
                            "occupant_seed_prefix_sha256": ordered_seed_bank_sha256(
                                seeds[:checkpoint]
                            ),
                            "weather_member_count": 18,
                            "paired_draw_count": len(values),
                            "metric": metric,
                            "statistic": statistic,
                            "value": value,
                            "previous_value": previous_value,
                            "absolute_floor": float(floor),
                            "relative_change": relative_change,
                            "relative_tolerance": SENSITIVITY_RELATIVE_TOLERANCE,
                            "median_sign_guard_triggered": sign_guard,
                            "criterion_pass": criterion_pass,
                        }
                    )
                    previous[key] = value

    results = pd.DataFrame.from_records(records)
    results["group_all_statistics_pass"] = False
    results["panel_all_groups_pass"] = False
    results["panel_consecutive_passing_expansions"] = 0
    results["panel_stable_at_checkpoint"] = False
    consecutive = 0
    stable_checkpoint: int | None = None
    for checkpoint in active:
        selected = results["seed_count"] == checkpoint
        group_pass = results.loc[selected].groupby(group_columns, sort=False)[
            "criterion_pass"
        ].transform("all")
        results.loc[selected, "group_all_statistics_pass"] = group_pass.to_numpy()
        panel_pass = bool(selected.any() and results.loc[selected, "criterion_pass"].all())
        consecutive = consecutive + 1 if panel_pass else 0
        stable = consecutive >= SENSITIVITY_REQUIRED_CONSECUTIVE_EXPANSIONS
        results.loc[selected, "panel_all_groups_pass"] = panel_pass
        results.loc[selected, "panel_consecutive_passing_expansions"] = consecutive
        results.loc[selected, "panel_stable_at_checkpoint"] = stable
        if stable and stable_checkpoint is None:
            stable_checkpoint = checkpoint

    previous_stage = contract.get("previous_stage")
    if previous_stage is not None and stable_checkpoint is not None and stable_checkpoint <= int(
        previous_stage["target_seed_count"]
    ):
        raise MonteCarloContractError(
            "Extension-stage evaluation contradicts its authenticated predecessor decision."
        )

    if stable_checkpoint is not None:
        status = f"STRUCTURAL_SENSITIVITY_SCREEN_STABLE_AT_N{stable_checkpoint}"
        decision = "STRUCTURAL_SCREEN_COMPLETE_STABLE"
        next_stage = None
    elif stage < SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS[-1]:
        status = f"STRUCTURAL_SENSITIVITY_SCREEN_NOT_STABLE_AT_N{stage}"
        next_stage = SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS[
            SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS.index(stage) + 1
        ]
        decision = f"EXTEND_TO_N{next_stage}"
    else:
        status = f"STRUCTURAL_SENSITIVITY_SCREEN_NOT_STABLE_AT_N{stage}"
        decision = "TERMINAL_NOT_STABLE_AT_DECLARED_MAXIMUM"
        next_stage = None

    order = [
        "demand_role",
        *group_columns,
        "seed_count",
        "previous_seed_count",
        "occupant_seed_prefix_sha256",
        "weather_member_count",
        "paired_draw_count",
        "metric",
        "statistic",
        "value",
        "previous_value",
        "absolute_floor",
        "relative_change",
        "relative_tolerance",
        "median_sign_guard_triggered",
        "criterion_pass",
        "group_all_statistics_pass",
        "panel_all_groups_pass",
        "panel_consecutive_passing_expansions",
        "panel_stable_at_checkpoint",
    ]
    results = results.loc[:, order].sort_values(
        ["seed_count", *group_columns, "metric", "statistic"], kind="stable"
    )
    results_path = root / SENSITIVITY_RESULTS_FILENAME
    _atomic_csv(results, results_path)
    post_path = root / POSTPROCESS_SUMMARY_FILENAME
    post = _read_json(post_path, label="post-processing summary")
    summary_payload = {
        "structural_sensitivity_summary_version": (
            STRUCTURAL_SENSITIVITY_SUMMARY_VERSION
        ),
        "status": status,
        "decision": decision,
        "purpose": "STRUCTURAL_SENSITIVITY_SCREEN",
        "production_qualification": "NOT_A_PRODUCTION_PASS",
        "structural_sensitivity_contract_sha256": contract[
            "structural_sensitivity_contract_sha256"
        ],
        "streaming_design_sha256": contract["streaming_design_sha256"],
        "source_execution_status": "WORKFLOW_CHECK_ONLY",
        "postprocessing_summary_file_sha256": _sha256_file(post_path),
        "postprocessing_contract_version": post["postprocess_contract_version"],
        "target_seed_count": stage,
        "evaluated_checkpoints": list(active),
        "stable_at_seed_count": stable_checkpoint,
        "next_stage_target_seed_count": next_stage,
        "panel_cell_count": len(SENSITIVITY_PANEL_CELLS),
        "climate_scenario_count": len(SENSITIVITY_CLIMATE_SCENARIOS),
        "weather_members_per_climate_scenario": len(SENSITIVITY_WEATHER_YEARS),
        "comparison_model_scenario_count": len(SENSITIVITY_COMPARISON_SCENARIOS),
        "metric_count": len(SENSITIVITY_METRIC_FLOORS),
        "stability_criterion_row_count": len(results),
        "last_checkpoint_all_groups_pass": bool(
            results.loc[
                results["seed_count"] == stage, "panel_all_groups_pass"
            ].all()
        ),
        "last_checkpoint_consecutive_passing_expansions": int(
            results.loc[
                results["seed_count"] == stage,
                "panel_consecutive_passing_expansions",
            ].iloc[0]
        ),
        "output_artifacts": {
            SENSITIVITY_RESULTS_FILENAME: {
                "sha256": _sha256_file(results_path),
                "row_count": len(results),
            }
        },
        "extension_safe_outcome": (
            "no extension required"
            if stable_checkpoint is not None
            else (
                f"prepare a new immutable n={next_stage} stage bound to this summary"
                if next_stage is not None
                else "declared n=160 maximum reached; report non-stabilisation"
            )
        ),
        "interpretation": (
            "Structural endpoints are paired scenario contrasts, not probabilities. "
            "The six weather strata are not probability-weighted and the result is "
            "not a complete prediction interval."
        ),
    }
    summary = {
        **summary_payload,
        "structural_sensitivity_summary_sha256": canonical_sha256(summary_payload),
    }
    _atomic_json(summary, root / SENSITIVITY_SUMMARY_FILENAME)
    return _load_summary(root, contract)


def run_structural_sensitivity_screen(
    output_dir: str | Path | None = None,
    *,
    target_seed_count: int = 40,
    previous_stage_dir: str | Path | None = None,
    max_workers: int = 4,
    postprocess_chunk_rows: int = 2_000,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Prepare, run/resume, post-process and evaluate one declared stage."""

    prepared = prepare_structural_sensitivity_screen(
        output_dir,
        target_seed_count=target_seed_count,
        previous_stage_dir=previous_stage_dir,
        max_workers=max_workers,
    )
    if prepare_only:
        return prepared
    root = Path(prepared["output_dir"])
    contract = _load_contract(root)
    states = tuple(
        sorted(
            load_unique_archetype_states(),
            key=lambda item: (item.archetype_id, item.state_id),
        )
    )
    members = tuple(
        load_weather_members(
            tuple(item["weather_member_id"] for item in contract["weather_members"])
        )
    )
    execution = execute_streaming_stock_design(
        states,
        members,
        tuple(int(item) for item in contract["occupant_seeds"]),
        SENSITIVITY_MODEL_SCENARIOS,
        output_dir=root,
        require_full_stock=True,
        require_convergence_evidence=False,
        max_workers=max_workers,
        prepare_only=False,
    )
    if execution.get("status") != "WORKFLOW_CHECK_ONLY":
        raise MonteCarloContractError(
            "Structural-sensitivity execution received an unexpected qualification; "
            "it must remain WORKFLOW_CHECK_ONLY."
        )
    existing_post = postprocessing_status(root)
    if existing_post.get("status") != "PASS":
        postprocess_production_results(root, chunk_rows=postprocess_chunk_rows)
    return evaluate_structural_sensitivity_screen(root)


def structural_sensitivity_status(
    output_dir: str | Path | None = None,
    *,
    target_seed_count: int = 40,
) -> dict[str, Any]:
    """Return a read-only, authenticated screen-stage snapshot."""

    stage = _validate_stage(target_seed_count)
    root = (
        default_structural_sensitivity_output_dir(stage)
        if output_dir is None
        else Path(output_dir).resolve()
    )
    contract_path = root / SENSITIVITY_CONTRACT_FILENAME
    if not contract_path.is_file():
        base = streaming_stock_status(root)
        return {
            "status": "INITIALIZING" if base.get("status") == "INITIALIZING" else "NOT_PREPARED",
            "purpose": "STRUCTURAL_SENSITIVITY_SCREEN",
            "production_qualification": "NOT_A_PRODUCTION_PASS",
            "output_dir": str(root),
            "target_seed_count": stage,
            "underlying_streaming_status": base.get("status"),
        }
    try:
        contract = _load_contract(root)
        if int(contract["target_seed_count"]) != stage:
            raise MonteCarloContractError(
                f"Requested n={stage} status for an n={contract['target_seed_count']} contract."
            )
        base = streaming_stock_status(root)
        summary_path = root / SENSITIVITY_SUMMARY_FILENAME
        if summary_path.is_file():
            if base.get("status") != "WORKFLOW_CHECK_ONLY":
                raise MonteCarloContractError(
                    "A committed sensitivity decision no longer has its complete "
                    "WORKFLOW_CHECK_ONLY source execution."
                )
            summary = _load_summary(root, contract)
            return {
                "status": summary["status"],
                "decision": summary["decision"],
                "purpose": "STRUCTURAL_SENSITIVITY_SCREEN",
                "production_qualification": "NOT_A_PRODUCTION_PASS",
                "output_dir": str(root),
                "target_seed_count": stage,
                "stable_at_seed_count": summary["stable_at_seed_count"],
                "next_stage_target_seed_count": summary[
                    "next_stage_target_seed_count"
                ],
                "structural_sensitivity_contract_sha256": contract[
                    "structural_sensitivity_contract_sha256"
                ],
                "structural_sensitivity_summary_sha256": summary[
                    "structural_sensitivity_summary_sha256"
                ],
                "underlying_streaming_status": base.get("status"),
            }
        if base.get("status") == "WORKFLOW_CHECK_ONLY":
            post = postprocessing_status(root)
            status = (
                "POSTPROCESSED_AWAITING_EVALUATION"
                if post.get("status") == "PASS"
                else "EXECUTION_COMPLETE_AWAITING_POSTPROCESS"
            )
        else:
            status = str(base.get("status"))
        return {
            "status": status,
            "purpose": "STRUCTURAL_SENSITIVITY_SCREEN",
            "production_qualification": "NOT_A_PRODUCTION_PASS",
            "output_dir": str(root),
            "target_seed_count": stage,
            "structural_sensitivity_contract_sha256": contract[
                "structural_sensitivity_contract_sha256"
            ],
            "underlying_streaming_status": base.get("status"),
            "partition_count": base.get("partition_count"),
            "completed_partition_count": base.get("completed_partition_count"),
            "partition_seed_count_histogram": base.get(
                "partition_seed_count_histogram"
            ),
        }
    except MonteCarloContractError as exc:
        return {
            "status": "INVALID",
            "purpose": "STRUCTURAL_SENSITIVITY_SCREEN",
            "production_qualification": "NOT_A_PRODUCTION_PASS",
            "output_dir": str(root),
            "target_seed_count": stage,
            "reason": str(exc),
        }


__all__ = [
    "DEFAULT_STRUCTURAL_SENSITIVITY_BASE_DIR",
    "SENSITIVITY_ALLOWED_STAGE_SEED_COUNTS",
    "SENSITIVITY_CLIMATE_SCENARIOS",
    "SENSITIVITY_COMPARISON_SCENARIOS",
    "SENSITIVITY_CONTRACT_FILENAME",
    "SENSITIVITY_MASTER_SEED",
    "SENSITIVITY_METRIC_FLOORS",
    "SENSITIVITY_MODEL_SCENARIOS",
    "SENSITIVITY_PANEL_CELLS",
    "SENSITIVITY_PROSPECTIVE_CHECKPOINTS",
    "SENSITIVITY_RELATIVE_TOLERANCE",
    "SENSITIVITY_RESULTS_FILENAME",
    "SENSITIVITY_SELECTION_SHA256",
    "SENSITIVITY_STATISTICS",
    "SENSITIVITY_SUMMARY_FILENAME",
    "SENSITIVITY_WEATHER_STRATA",
    "SENSITIVITY_WEATHER_YEARS",
    "STRUCTURAL_SENSITIVITY_CONTRACT_VERSION",
    "STRUCTURAL_SENSITIVITY_SUMMARY_VERSION",
    "default_structural_sensitivity_output_dir",
    "evaluate_structural_sensitivity_screen",
    "prepare_structural_sensitivity_screen",
    "run_structural_sensitivity_screen",
    "structural_sensitivity_status",
]
