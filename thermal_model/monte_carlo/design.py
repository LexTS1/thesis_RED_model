"""Balanced experiment design, convergence and uncertainty summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from thermal_model.behaviour import (
    load_behaviour_assumptions,
    load_occupant_distribution,
)
from thermal_model.contracts import (
    ArchetypeStateInput,
    load_assumption_contract,
    validate_archetype_state,
)

from .contracts import (
    ModelScenario,
    MonteCarloContractError,
    RunSpec,
    WeatherMember,
    archetype_state_sha256,
    canonical_sha256,
    diagnostics_to_record,
    validate_weather_member,
)
from .interface import _run_id
from .scenarios import (
    effective_assumption_contract,
    model_scenario_sha256,
    resolve_model_scenario,
)


DEFAULT_CONVERGENCE_CHECKPOINTS = (5, 10, 20, 40, 80)
DEFAULT_RELATIVE_CONVERGENCE_TOLERANCE = 0.02
DEFAULT_REQUIRED_CONSECUTIVE_EXPANSIONS = 2
DEFAULT_CONVERGENCE_METRICS: dict[str, float] = {
    "heating_intensity_kWh_m2": 1.0,
    "cooling_intensity_kWh_m2": 1.0,
    "peak_heating_W": 100.0,
    "peak_cooling_W": 100.0,
}
DEFAULT_CONVERGENCE_STATISTICS = ("mean", "median", "p95")
DEFAULT_DISTRIBUTION_METRICS = (
    "annual_heating_kWh",
    "annual_cooling_kWh",
    "heating_intensity_kWh_m2",
    "cooling_intensity_kWh_m2",
    "peak_heating_W",
    "peak_cooling_W",
    "heating_full_load_equivalent_hours",
    "cooling_full_load_equivalent_hours",
)
DEFAULT_ANALYSIS_GROUPS = (
    "archetype_id",
    "state_id",
    "climate_scenario_id",
    "model_scenario_id",
)
CONVERGENCE_GLOBAL_PROVENANCE_COLUMNS = (
    "model_contract_version",
    "central_thermal_assumptions_sha256",
    "behaviour_assumptions_sha256",
    "occupant_distribution_sha256",
)
CONVERGENCE_GROUP_PROVENANCE_COLUMNS = (
    "archetype_state_sha256",
    "model_scenario_sha256",
)
CONVERGENCE_WEATHER_PROVENANCE_COLUMNS = (
    "weather_member_id",
    "weather_contract_sha256",
    "weather_forcing_sha256",
)


@dataclass(frozen=True)
class ConvergenceRule:
    """Predeclared nested-prefix stopping rule for occupant seed counts."""

    checkpoints: tuple[int, ...] = DEFAULT_CONVERGENCE_CHECKPOINTS
    relative_tolerance: float = DEFAULT_RELATIVE_CONVERGENCE_TOLERANCE
    required_consecutive_expansions: int = DEFAULT_REQUIRED_CONSECUTIVE_EXPANSIONS
    metrics_and_absolute_floors: tuple[tuple[str, float], ...] = tuple(
        DEFAULT_CONVERGENCE_METRICS.items()
    )
    statistics: tuple[str, ...] = DEFAULT_CONVERGENCE_STATISTICS


# Prospective extension authorized after the original 5--80 experiment produced
# only its first passing expansion at n=80.  Every numerical decision criterion
# remains identical to the original predeclared rule; only the nested-prefix
# checkpoint sequence is extended by n=160 to obtain the required confirmation.
PROSPECTIVE_N160_CONVERGENCE_RULE: Final[ConvergenceRule] = ConvergenceRule(
    checkpoints=(*DEFAULT_CONVERGENCE_CHECKPOINTS, 160),
    relative_tolerance=DEFAULT_RELATIVE_CONVERGENCE_TOLERANCE,
    required_consecutive_expansions=DEFAULT_REQUIRED_CONSECUTIVE_EXPANSIONS,
    metrics_and_absolute_floors=tuple(DEFAULT_CONVERGENCE_METRICS.items()),
    statistics=DEFAULT_CONVERGENCE_STATISTICS,
)


# Prospectively frozen after the n=160 expansion failed the complete-panel
# criterion.  Both newly declared checkpoints are retained so n=320 can only
# establish the first new pass and n=640 can confirm it.  No numerical
# criterion changes relative to either predecessor protocol.
PROSPECTIVE_N320_N640_CONVERGENCE_RULE: Final[ConvergenceRule] = ConvergenceRule(
    checkpoints=(*DEFAULT_CONVERGENCE_CHECKPOINTS, 160, 320, 640),
    relative_tolerance=DEFAULT_RELATIVE_CONVERGENCE_TOLERANCE,
    required_consecutive_expansions=DEFAULT_REQUIRED_CONSECUTIVE_EXPANSIONS,
    metrics_and_absolute_floors=tuple(DEFAULT_CONVERGENCE_METRICS.items()),
    statistics=DEFAULT_CONVERGENCE_STATISTICS,
)


def make_seed_bank(count: int, *, master_seed: int = 20250808) -> tuple[int, ...]:
    """Create unique uint32 seeds; every prefix is a valid nested experiment."""

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
        raise MonteCarloContractError("Seed-bank count must be a positive integer.")
    if isinstance(master_seed, bool) or not isinstance(master_seed, (int, np.integer)):
        raise MonteCarloContractError("master_seed must be an integer.")
    if not 0 <= int(master_seed) <= 2**32 - 1:
        raise MonteCarloContractError("master_seed must be between zero and 2**32-1.")
    sequences = np.random.SeedSequence(int(master_seed)).spawn(int(count))
    seeds = tuple(int(sequence.generate_state(1, dtype=np.uint32)[0]) for sequence in sequences)
    if len(set(seeds)) != len(seeds):
        raise MonteCarloContractError("Generated seed bank unexpectedly contains duplicates.")
    return seeds


def _validate_seed_sequence(seeds: Iterable[int]) -> tuple[int, ...]:
    raw = tuple(seeds)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise MonteCarloContractError("Occupant seeds must be integers, not coerced values.")
    result = tuple(int(value) for value in raw)
    if not result:
        raise MonteCarloContractError("At least one occupant seed is required.")
    if len(set(result)) != len(result):
        raise MonteCarloContractError("Occupant seeds must be unique.")
    if any(value < 0 or value > 2**32 - 1 for value in result):
        raise MonteCarloContractError("Occupant seeds must be uint32-compatible.")
    return result


def ordered_seed_bank_sha256(seeds: Iterable[int]) -> str:
    """Hash an exact, validated occupant-seed order.

    Order is part of the convergence contract because every checkpoint is a
    nested prefix.  Sorting the values before hashing would therefore destroy
    information required to reproduce the stopping decision.
    """

    validated = _validate_seed_sequence(seeds)
    payload = json.dumps(list(validated), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def convergence_weather_panel_sha256(
    climate_scenario_id: str,
    weather_records: Iterable[Mapping[str, object]],
) -> str:
    """Hash the exact ordered-independent weather panel used for convergence."""

    scenario_id = str(climate_scenario_id).strip()
    if not scenario_id:
        raise MonteCarloContractError(
            "A climate scenario ID is required for a convergence weather panel."
        )
    normalized: list[dict[str, str]] = []
    for raw in weather_records:
        record = {
            column: str(raw[column]).strip()
            for column in CONVERGENCE_WEATHER_PROVENANCE_COLUMNS
        }
        if not record["weather_member_id"]:
            raise MonteCarloContractError(
                "Convergence weather-member identifiers must not be blank."
            )
        for column in ("weather_contract_sha256", "weather_forcing_sha256"):
            digest = record[column].lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise MonteCarloContractError(
                    f"Convergence weather provenance {column!r} is not a SHA-256 digest."
                )
            record[column] = digest
        normalized.append(record)
    if not normalized:
        raise MonteCarloContractError(
            "A convergence weather panel must contain at least one member."
        )
    member_ids = [item["weather_member_id"] for item in normalized]
    if len(set(member_ids)) != len(member_ids):
        raise MonteCarloContractError(
            "A convergence weather panel contains duplicate member identifiers."
        )
    normalized.sort(key=lambda item: item["weather_member_id"])
    return canonical_sha256(
        {
            "contract": "gate5_convergence_weather_panel_v1",
            "climate_scenario_id": scenario_id,
            "members": normalized,
        }
    )


def build_balanced_manifest(
    archetype_states: Sequence[ArchetypeStateInput],
    weather_members: Sequence[WeatherMember],
    occupant_seeds: Iterable[int],
    model_scenarios: Sequence[str | ModelScenario] = ("central",),
) -> pd.DataFrame:
    """Return the sorted full cross-product with common random numbers."""

    validated_states: list[ArchetypeStateInput] = []
    for item in archetype_states:
        if not isinstance(item, ArchetypeStateInput):
            raise MonteCarloContractError(
                "Balanced design archetypes must be ArchetypeStateInput instances."
            )
        validated_states.append(validate_archetype_state(asdict(item)))
    states = tuple(validated_states)
    weather = tuple(validate_weather_member(item) for item in weather_members)
    seeds = _validate_seed_sequence(occupant_seeds)
    scenarios = tuple(resolve_model_scenario(item) for item in model_scenarios)
    if not states or not weather or not scenarios:
        raise MonteCarloContractError(
            "Balanced design needs at least one archetype, weather member and scenario."
        )
    state_keys = [(item.archetype_id, item.state_id) for item in states]
    if len(set(state_keys)) != len(state_keys):
        raise MonteCarloContractError("Archetype/state inputs must be unique.")
    weather_ids = [item.member_id for item in weather]
    if len(set(weather_ids)) != len(weather_ids):
        raise MonteCarloContractError("Weather members must be unique.")
    weather_cells = [
        (item.climate_scenario_id, item.weather_pair_id) for item in weather
    ]
    if len(set(weather_cells)) != len(weather_cells):
        raise MonteCarloContractError(
            "Each selected RCP/weather-pair cell must contain exactly one member."
        )
    pairs_by_rcp = {
        scenario_id: frozenset(
            item.weather_pair_id
            for item in weather
            if item.climate_scenario_id == scenario_id
        )
        for scenario_id in sorted({item.climate_scenario_id for item in weather})
    }
    if len({pairs for pairs in pairs_by_rcp.values()}) > 1:
        raise MonteCarloContractError(
            "Selected RCPs must use identical PVGIS weather-pair sets; "
            f"got {pairs_by_rcp}."
        )
    scenario_ids = [item.scenario_id for item in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise MonteCarloContractError("Model scenarios must be unique.")

    central = load_assumption_contract()
    behaviour_contract = load_behaviour_assumptions()
    _, occupant_distribution_sha = load_occupant_distribution()
    records: list[dict] = []
    for state in states:
        for member in weather:
            for seed_rank, seed in enumerate(seeds, start=1):
                for scenario in scenarios:
                    effective = effective_assumption_contract(central, scenario)
                    spec = RunSpec(
                        run_id=_run_id(
                            state,
                            member,
                            seed,
                            scenario,
                            effective_thermal_sha256=effective.sha256,
                            behaviour_assumptions_sha256=behaviour_contract.sha256,
                            occupant_distribution_sha256=occupant_distribution_sha,
                        ),
                        archetype_id=state.archetype_id,
                        dwelling_type=state.dwelling_type,
                        construction_period=state.construction_period,
                        state_id=state.state_id,
                        archetype_state_sha256=archetype_state_sha256(state),
                        climate_scenario_id=member.climate_scenario_id,
                        weather_member_id=member.member_id,
                        weather_pair_id=member.weather_pair_id,
                        observed_pvgis_year=member.observed_pvgis_year,
                        occupant_seed=seed,
                        occupant_seed_rank=seed_rank,
                        model_scenario_id=scenario.scenario_id,
                        model_scenario_axis=scenario.axis,
                        weather_contract_sha256=member.weather_contract_sha256,
                        model_scenario_sha256=model_scenario_sha256(
                            scenario, central.sha256
                        ),
                    )
                    record = asdict(spec)
                    record["weather_forcing_sha256"] = member.forcing_sha256
                    record["effective_thermal_assumptions_sha256"] = effective.sha256
                    record["behaviour_assumptions_sha256"] = behaviour_contract.sha256
                    record["occupant_distribution_sha256"] = occupant_distribution_sha
                    records.append(record)
    result = pd.DataFrame.from_records(records)
    expected = len(states) * len(weather) * len(seeds) * len(scenarios)
    if len(result) != expected or result["run_id"].duplicated().any():
        raise MonteCarloContractError("Balanced manifest size or run-ID uniqueness failed.")
    factor_columns = [
        "archetype_id",
        "state_id",
        "weather_member_id",
        "model_scenario_id",
    ]
    observed_seed_sets = result.sort_values("occupant_seed_rank").groupby(
        factor_columns
    )["occupant_seed"].agg(lambda values: tuple(values))
    if any(seed_set != seeds for seed_set in observed_seed_sets):
        raise MonteCarloContractError("Manifest did not preserve the common ordered seed set.")
    return result.sort_values(
        [
            "archetype_id",
            "state_id",
            "climate_scenario_id",
            "observed_pvgis_year",
            "occupant_seed_rank",
            "model_scenario_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def results_to_diagnostics_frame(results: Sequence) -> pd.DataFrame:
    """Flatten Gate-5 results and reject duplicate run identifiers."""

    frame = pd.DataFrame.from_records(
        [diagnostics_to_record(result.diagnostics) for result in results]
    )
    if frame.empty:
        raise MonteCarloContractError("No simulation results were supplied.")
    if frame["run_id"].duplicated().any():
        raise MonteCarloContractError("Simulation results contain duplicate run IDs.")
    return frame


def distribution_summary(
    diagnostics: pd.DataFrame,
    *,
    group_columns: Sequence[str] = DEFAULT_ANALYSIS_GROUPS,
    metrics: Sequence[str] = DEFAULT_DISTRIBUTION_METRICS,
) -> pd.DataFrame:
    """Summarize empirical weather/occupant distributions without pooling RCPs."""

    missing = sorted(
        set(group_columns)
        .union(metrics)
        .union({"weather_member_id", "occupant_seed"})
        .difference(diagnostics.columns)
    )
    if missing:
        raise MonteCarloContractError(f"Distribution input is missing columns: {missing}.")
    if "climate_scenario_id" not in group_columns:
        raise MonteCarloContractError("Distribution summaries must keep RCPs separate.")
    records: list[dict] = []
    grouper = list(group_columns)
    for key, group in diagnostics.groupby(grouper, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(grouper, key_tuple))
        if group.duplicated(["weather_member_id", "occupant_seed"]).any():
            raise MonteCarloContractError(
                f"Distribution group {identity} has duplicate weather/seed cells."
            )
        weather_count = group["weather_member_id"].nunique()
        seed_count = group["occupant_seed"].nunique()
        if len(group) != weather_count * seed_count:
            raise MonteCarloContractError(
                f"Distribution group {identity} is not a complete weather-by-seed "
                "rectangle."
            )
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise MonteCarloContractError(f"Metric {metric} contains non-finite values.")
            records.append(
                {
                    **identity,
                    "metric": metric,
                    "sample_count": len(values),
                    "weather_member_count": weather_count,
                    "occupant_seed_count": seed_count,
                    "minimum": float(np.min(values)),
                    "p05": float(np.quantile(values, 0.05)),
                    "p25": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p95": float(np.quantile(values, 0.95)),
                    "maximum": float(np.max(values)),
                    "standard_deviation": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    ),
                    "interval_interpretation": (
                        "descriptive empirical interval over included weather years and seeds"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def variance_contributions(
    diagnostics: pd.DataFrame,
    *,
    group_columns: Sequence[str] = DEFAULT_ANALYSIS_GROUPS,
    metrics: Sequence[str] = (
        "heating_intensity_kWh_m2",
        "cooling_intensity_kWh_m2",
        "peak_heating_W",
        "peak_cooling_W",
    ),
) -> pd.DataFrame:
    """Calculate balanced two-way ANOVA sum-of-squares shares for weather/seeds."""

    required = set(group_columns).union(
        {"weather_member_id", "occupant_seed", *metrics}
    )
    missing = sorted(required.difference(diagnostics.columns))
    if missing:
        raise MonteCarloContractError(f"Variance input is missing columns: {missing}.")
    if "climate_scenario_id" not in group_columns:
        raise MonteCarloContractError("Variance decomposition must keep RCPs separate.")
    records: list[dict] = []
    grouper = list(group_columns)
    for key, group in diagnostics.groupby(grouper, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(grouper, key_tuple))
        if group.duplicated(["weather_member_id", "occupant_seed"]).any():
            raise MonteCarloContractError(
                f"Variance group {identity} has duplicate weather/seed cells."
            )
        weather_ids = sorted(group["weather_member_id"].unique())
        seeds = sorted(int(value) for value in group["occupant_seed"].unique())
        expected_cells = len(weather_ids) * len(seeds)
        if len(group) != expected_cells:
            raise MonteCarloContractError(
                f"Variance group {identity} is not a complete weather-by-seed rectangle."
            )
        for metric in metrics:
            pivot = group.pivot(
                index="weather_member_id", columns="occupant_seed", values=metric
            ).reindex(index=weather_ids, columns=seeds)
            values = pivot.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise MonteCarloContractError(
                    f"Variance metric {metric} contains missing/non-finite cells."
                )
            grand = float(values.mean())
            weather_means = values.mean(axis=1)
            seed_means = values.mean(axis=0)
            ss_weather = float(len(seeds) * np.square(weather_means - grand).sum())
            ss_seed = float(len(weather_ids) * np.square(seed_means - grand).sum())
            residual = values - weather_means[:, None] - seed_means[None, :] + grand
            ss_interaction = float(np.square(residual).sum())
            ss_total = float(np.square(values - grand).sum())
            if not np.isclose(
                ss_weather + ss_seed + ss_interaction,
                ss_total,
                rtol=1.0e-10,
                atol=1.0e-9,
            ):
                raise MonteCarloContractError(
                    f"ANOVA sums of squares do not close for {identity}/{metric}."
                )
            components = {
                "weather_year": ss_weather,
                "occupant_seed": ss_seed,
                "weather_seed_interaction": ss_interaction,
            }
            for component, sum_of_squares in components.items():
                records.append(
                    {
                        **identity,
                        "metric": metric,
                        "component": component,
                        "weather_member_count": len(weather_ids),
                        "occupant_seed_count": len(seeds),
                        "sum_of_squares": sum_of_squares,
                        "total_sum_of_squares": ss_total,
                        "sum_of_squares_share": (
                            sum_of_squares / ss_total if ss_total > 0.0 else 0.0
                        ),
                        "interpretation": (
                            "balanced ANOVA share within included empirical ensemble"
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def paired_renovation_deltas(
    diagnostics: pd.DataFrame,
    *,
    baseline_state_id: str = "TABULA_existing",
    metrics: Sequence[str] = (
        "annual_heating_kWh",
        "annual_cooling_kWh",
        "heating_intensity_kWh_m2",
        "cooling_intensity_kWh_m2",
        "peak_heating_W",
        "peak_cooling_W",
    ),
) -> pd.DataFrame:
    """Calculate paired renovated-minus-existing differences on common draws."""

    pair_columns = [
        "archetype_id",
        "climate_scenario_id",
        "weather_member_id",
        "occupant_seed",
        "model_scenario_id",
    ]
    required = set(pair_columns).union({"state_id", *metrics})
    missing = sorted(required.difference(diagnostics.columns))
    if missing:
        raise MonteCarloContractError(f"Paired-delta input is missing columns: {missing}.")
    states = sorted(set(diagnostics["state_id"]) - {baseline_state_id})
    if not states:
        return pd.DataFrame(
            columns=[*pair_columns, "baseline_state_id", "comparison_state_id", "metric", "delta"]
        )
    if diagnostics.duplicated([*pair_columns, "state_id"]).any():
        raise MonteCarloContractError("Paired-delta input contains duplicate factor cells.")
    records: list[dict] = []
    indexed = diagnostics.set_index([*pair_columns, "state_id"])
    pair_index = diagnostics[pair_columns].drop_duplicates()
    for pair in pair_index.itertuples(index=False, name=None):
        baseline_key = (*pair, baseline_state_id)
        if baseline_key not in indexed.index:
            raise MonteCarloContractError(
                f"Missing paired baseline {baseline_state_id} for factors {pair}."
            )
        baseline = indexed.loc[baseline_key]
        for comparison_state in states:
            comparison_key = (*pair, comparison_state)
            if comparison_key not in indexed.index:
                raise MonteCarloContractError(
                    f"Missing paired state {comparison_state} for factors {pair}."
                )
            comparison = indexed.loc[comparison_key]
            identity = dict(zip(pair_columns, pair))
            for metric in metrics:
                records.append(
                    {
                        **identity,
                        "baseline_state_id": baseline_state_id,
                        "comparison_state_id": comparison_state,
                        "metric": metric,
                        "baseline_value": float(baseline[metric]),
                        "comparison_value": float(comparison[metric]),
                        "delta": float(comparison[metric] - baseline[metric]),
                    }
                )
    return pd.DataFrame.from_records(records)


def paired_model_scenario_deltas(
    diagnostics: pd.DataFrame,
    *,
    baseline_model_scenario_id: str = "central",
    metrics: Sequence[str] = (
        "annual_heating_kWh",
        "annual_cooling_kWh",
        "heating_intensity_kWh_m2",
        "cooling_intensity_kWh_m2",
        "peak_heating_W",
        "peak_cooling_W",
    ),
) -> pd.DataFrame:
    """Return structural-sensitivity minus central results on exact common draws.

    Structural scenarios are epistemic cases, not random samples.  Pairing each
    case with ``central`` on the same archetype, renovation state, weather
    member and occupant seed isolates the declared assumption change.
    """

    pair_columns = [
        "archetype_id",
        "state_id",
        "climate_scenario_id",
        "weather_member_id",
        "weather_pair_id",
        "observed_pvgis_year",
        "occupant_seed",
    ]
    required = set(pair_columns).union(
        {"model_scenario_id", "model_scenario_axis", *metrics}
    )
    missing = sorted(required.difference(diagnostics.columns))
    if missing:
        raise MonteCarloContractError(
            f"Paired model-scenario input is missing columns: {missing}."
        )
    scenario_ids = sorted(
        set(diagnostics["model_scenario_id"]) - {baseline_model_scenario_id}
    )
    columns = [
        *pair_columns,
        "baseline_model_scenario_id",
        "comparison_model_scenario_id",
        "comparison_model_scenario_axis",
        "metric",
        "baseline_value",
        "comparison_value",
        "delta",
    ]
    if not scenario_ids:
        return pd.DataFrame(columns=columns)
    if diagnostics.duplicated([*pair_columns, "model_scenario_id"]).any():
        raise MonteCarloContractError(
            "Paired model-scenario input contains duplicate factor cells."
        )
    indexed = diagnostics.set_index([*pair_columns, "model_scenario_id"])
    pair_index = diagnostics[pair_columns].drop_duplicates()
    records: list[dict] = []
    for pair in pair_index.itertuples(index=False, name=None):
        baseline_key = (*pair, baseline_model_scenario_id)
        if baseline_key not in indexed.index:
            raise MonteCarloContractError(
                f"Missing paired model baseline {baseline_model_scenario_id} "
                f"for factors {pair}."
            )
        baseline = indexed.loc[baseline_key]
        identity = dict(zip(pair_columns, pair))
        for comparison_scenario_id in scenario_ids:
            comparison_key = (*pair, comparison_scenario_id)
            if comparison_key not in indexed.index:
                raise MonteCarloContractError(
                    f"Missing paired model scenario {comparison_scenario_id} "
                    f"for factors {pair}."
                )
            comparison = indexed.loc[comparison_key]
            for metric in metrics:
                records.append(
                    {
                        **identity,
                        "baseline_model_scenario_id": baseline_model_scenario_id,
                        "comparison_model_scenario_id": comparison_scenario_id,
                        "comparison_model_scenario_axis": str(
                            comparison["model_scenario_axis"]
                        ),
                        "metric": metric,
                        "baseline_value": float(baseline[metric]),
                        "comparison_value": float(comparison[metric]),
                        "delta": float(comparison[metric] - baseline[metric]),
                    }
                )
    return pd.DataFrame.from_records(records, columns=columns)


def _statistic(values: np.ndarray, name: str) -> float:
    if name == "mean":
        return float(np.mean(values))
    if name == "median":
        return float(np.median(values))
    if name == "p95":
        return float(np.quantile(values, 0.95))
    raise MonteCarloContractError(f"Unsupported convergence statistic {name!r}.")


def evaluate_seed_convergence(
    diagnostics: pd.DataFrame,
    *,
    seed_order: Sequence[int],
    rule: ConvergenceRule = ConvergenceRule(),
    group_columns: Sequence[str] = DEFAULT_ANALYSIS_GROUPS,
) -> pd.DataFrame:
    """Evaluate a predeclared stopping rule on nested common-seed prefixes."""

    seeds = _validate_seed_sequence(seed_order)
    checkpoints = tuple(int(value) for value in rule.checkpoints)
    if (
        not checkpoints
        or sorted(set(checkpoints)) != list(checkpoints)
        or checkpoints[0] <= 0
    ):
        raise MonteCarloContractError(
            "Convergence checkpoints must be unique, positive and increasing."
        )
    if not 0.0 < rule.relative_tolerance < 1.0:
        raise MonteCarloContractError("Relative convergence tolerance must be within (0, 1).")
    if rule.required_consecutive_expansions <= 0:
        raise MonteCarloContractError("Required consecutive expansions must be positive.")
    metric_floors = dict(rule.metrics_and_absolute_floors)
    if any(float(value) <= 0.0 for value in metric_floors.values()):
        raise MonteCarloContractError("Convergence absolute floors must be positive.")
    required = set(group_columns).union(
        {
            "occupant_seed",
            *metric_floors,
            *CONVERGENCE_GLOBAL_PROVENANCE_COLUMNS,
            *CONVERGENCE_GROUP_PROVENANCE_COLUMNS,
            *CONVERGENCE_WEATHER_PROVENANCE_COLUMNS,
        }
    )
    missing = sorted(required.difference(diagnostics.columns))
    if missing:
        raise MonteCarloContractError(f"Convergence input is missing columns: {missing}.")
    if "climate_scenario_id" not in group_columns:
        raise MonteCarloContractError("Convergence must be assessed separately by RCP.")
    missing_group_identities = sorted(
        set(DEFAULT_ANALYSIS_GROUPS).difference(group_columns)
    )
    if missing_group_identities:
        raise MonteCarloContractError(
            "Runner-ready convergence evidence requires physical/RCP/model group "
            f"identities: {missing_group_identities}."
        )

    global_provenance: dict[str, str] = {}
    for column in CONVERGENCE_GLOBAL_PROVENANCE_COLUMNS:
        values = diagnostics[column].dropna().astype(str).str.strip()
        if len(values) != len(diagnostics) or values.eq("").any() or values.nunique() != 1:
            raise MonteCarloContractError(
                f"Convergence diagnostics have missing or mixed {column!r} provenance."
            )
        global_provenance[column] = str(values.iloc[0])
        if column != "model_contract_version":
            digest = global_provenance[column].lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise MonteCarloContractError(
                    f"Convergence diagnostics {column!r} is not a SHA-256 digest."
                )
            global_provenance[column] = digest

    available = set(int(value) for value in diagnostics["occupant_seed"].unique())
    if not set(seeds).issubset(available):
        missing_seeds = sorted(set(seeds) - available)
        raise MonteCarloContractError(f"Convergence input lacks seed bank values: {missing_seeds}.")
    active_checkpoints = [value for value in checkpoints if value <= len(seeds)]
    seed_bank_sha256 = ordered_seed_bank_sha256(seeds)
    records: list[dict] = []
    grouper = list(group_columns)
    for key, group in diagnostics.groupby(grouper, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(grouper, key_tuple))
        if group.duplicated(["weather_member_id", "occupant_seed"]).any():
            raise MonteCarloContractError(
                f"Convergence group {identity} has duplicate weather/seed cells."
            )
        group_provenance: dict[str, str] = {}
        for column in CONVERGENCE_GROUP_PROVENANCE_COLUMNS:
            values = group[column].dropna().astype(str).str.strip()
            if len(values) != len(group) or values.eq("").any() or values.nunique() != 1:
                raise MonteCarloContractError(
                    f"Convergence group {identity} has missing or mixed {column!r}."
                )
            group_provenance[column] = str(values.iloc[0]).lower()
            digest = group_provenance[column]
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise MonteCarloContractError(
                    f"Convergence group {identity} has an invalid {column!r} digest."
                )
        weather_rows = group[
            list(CONVERGENCE_WEATHER_PROVENANCE_COLUMNS)
        ].drop_duplicates()
        if weather_rows["weather_member_id"].duplicated().any():
            raise MonteCarloContractError(
                f"Convergence group {identity} has conflicting weather provenance."
            )
        weather_panel_sha256 = convergence_weather_panel_sha256(
            str(identity["climate_scenario_id"]),
            weather_rows.to_dict(orient="records"),
        )
        expected_weather = set(group["weather_member_id"].unique())
        previous: dict[tuple[str, str], float] = {}
        consecutive = 0
        for checkpoint in active_checkpoints:
            prefix = set(seeds[:checkpoint])
            selected = group.loc[group["occupant_seed"].isin(prefix)]
            counts = selected.groupby("weather_member_id")["occupant_seed"].nunique()
            if set(counts.index) != expected_weather or not counts.eq(checkpoint).all():
                raise MonteCarloContractError(
                    f"Convergence group {identity} is not balanced at n={checkpoint}."
                )
            checkpoint_records: list[dict] = []
            for metric, absolute_floor in metric_floors.items():
                values = pd.to_numeric(selected[metric], errors="raise").to_numpy(dtype=float)
                for statistic in rule.statistics:
                    value = _statistic(values, statistic)
                    pair = (metric, statistic)
                    previous_value = previous.get(pair)
                    normalized_change = (
                        abs(value - previous_value) / max(abs(value), float(absolute_floor))
                        if previous_value is not None
                        else np.nan
                    )
                    passed = bool(
                        previous_value is not None
                        and normalized_change <= rule.relative_tolerance
                    )
                    checkpoint_records.append(
                        {
                            **identity,
                            **global_provenance,
                            **group_provenance,
                            "weather_panel_sha256": weather_panel_sha256,
                            "occupant_seed_bank_count": len(seeds),
                            "occupant_seed_bank_sha256": seed_bank_sha256,
                            "occupant_seed_prefix_sha256": ordered_seed_bank_sha256(
                                seeds[:checkpoint]
                            ),
                            "seed_count": checkpoint,
                            "previous_seed_count": (
                                active_checkpoints[active_checkpoints.index(checkpoint) - 1]
                                if active_checkpoints.index(checkpoint) > 0
                                else np.nan
                            ),
                            "metric": metric,
                            "statistic": statistic,
                            "value": value,
                            "previous_value": previous_value,
                            "absolute_floor": float(absolute_floor),
                            "relative_change": normalized_change,
                            "relative_tolerance": rule.relative_tolerance,
                            "criterion_pass": passed,
                        }
                    )
                    previous[pair] = value
            all_pass = bool(checkpoint_records) and all(
                item["criterion_pass"] for item in checkpoint_records
            )
            consecutive = consecutive + 1 if all_pass else 0
            converged = consecutive >= rule.required_consecutive_expansions
            for item in checkpoint_records:
                item["all_statistics_pass"] = all_pass
                item["consecutive_passing_expansions"] = consecutive
                item["required_consecutive_expansions"] = (
                    rule.required_consecutive_expansions
                )
                item["converged_at_checkpoint"] = converged
                records.append(item)
    columns = [
        *grouper,
        *CONVERGENCE_GLOBAL_PROVENANCE_COLUMNS,
        *CONVERGENCE_GROUP_PROVENANCE_COLUMNS,
        "weather_panel_sha256",
        "occupant_seed_bank_count",
        "occupant_seed_bank_sha256",
        "occupant_seed_prefix_sha256",
        "seed_count",
        "previous_seed_count",
        "metric",
        "statistic",
        "value",
        "previous_value",
        "absolute_floor",
        "relative_change",
        "relative_tolerance",
        "criterion_pass",
        "all_statistics_pass",
        "consecutive_passing_expansions",
        "required_consecutive_expansions",
        "converged_at_checkpoint",
        "panel_all_groups_statistics_pass",
        "panel_consecutive_passing_expansions",
        "panel_converged_at_checkpoint",
    ]
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return pd.DataFrame(columns=columns)
    weather_hash_counts = result.groupby("climate_scenario_id")[
        "weather_panel_sha256"
    ].nunique()
    if not weather_hash_counts.eq(1).all():
        raise MonteCarloContractError(
            "Every physical/model group within an RCP must use one common weather panel."
        )
    panel_consecutive = 0
    for checkpoint in active_checkpoints:
        selected = result["seed_count"] == checkpoint
        panel_pass = bool(selected.any() and result.loc[selected, "criterion_pass"].all())
        panel_consecutive = panel_consecutive + 1 if panel_pass else 0
        result.loc[selected, "panel_all_groups_statistics_pass"] = panel_pass
        result.loc[selected, "panel_consecutive_passing_expansions"] = panel_consecutive
        result.loc[selected, "panel_converged_at_checkpoint"] = (
            panel_consecutive >= rule.required_consecutive_expansions
        )
    result["panel_all_groups_statistics_pass"] = result[
        "panel_all_groups_statistics_pass"
    ].astype(bool)
    result["panel_consecutive_passing_expansions"] = result[
        "panel_consecutive_passing_expansions"
    ].astype(int)
    result["panel_converged_at_checkpoint"] = result[
        "panel_converged_at_checkpoint"
    ].astype(bool)
    return result[columns]
