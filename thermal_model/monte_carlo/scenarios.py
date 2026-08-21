"""Allow-listed, immutable structural sensitivity scenarios for Gate 5."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Mapping

import numpy as np

from thermal_model.contracts import (
    ArchetypeStateInput,
    AssumptionContract,
    validate_archetype_state,
)

from .contracts import (
    MODEL_CONTRACT_VERSION,
    ModelScenario,
    MonteCarloContractError,
    canonical_sha256,
    validate_model_scenario,
)


MODEL_SCENARIOS: dict[str, ModelScenario] = {
    "central": ModelScenario(
        scenario_id="central",
        axis="central",
        description="Authoritative thermal_assumptions.csv values.",
    ),
    "mass_light": ModelScenario(
        scenario_id="mass_light",
        axis="thermal_mass",
        description="ISO light mass-class sensitivity.",
        mass_capacitance_J_m2K=110_000.0,
        mass_area_ratio_m2_m2=2.5,
    ),
    "mass_heavy": ModelScenario(
        scenario_id="mass_heavy",
        axis="thermal_mass",
        description="ISO heavy mass-class sensitivity.",
        mass_capacitance_J_m2K=260_000.0,
        mass_area_ratio_m2_m2=3.0,
    ),
    "shading_unshaded": ModelScenario(
        scenario_id="shading_unshaded",
        axis="fixed_shading",
        description="Upper fixed-shading bound with F_sh=1.0.",
        vertical_shading_factor=1.0,
    ),
    "infiltration_half": ModelScenario(
        scenario_id="infiltration_half",
        axis="infiltration",
        description="Normal-pressure infiltration airflow multiplied by 0.5.",
        infiltration_multiplier=0.5,
    ),
    "infiltration_one_and_half": ModelScenario(
        scenario_id="infiltration_one_and_half",
        axis="infiltration",
        description="Normal-pressure infiltration airflow multiplied by 1.5.",
        infiltration_multiplier=1.5,
    ),
}

for _scenario in MODEL_SCENARIOS.values():
    validate_model_scenario(_scenario)


def resolve_model_scenario(value: str | ModelScenario) -> ModelScenario:
    """Resolve only a registered scenario; arbitrary hidden overrides are rejected."""

    identifier = value.scenario_id if isinstance(value, ModelScenario) else str(value).strip()
    try:
        registered = MODEL_SCENARIOS[identifier]
    except KeyError as exc:
        raise MonteCarloContractError(
            f"Unknown model scenario {identifier!r}; available={sorted(MODEL_SCENARIOS)}."
        ) from exc
    if isinstance(value, ModelScenario) and value != registered:
        raise MonteCarloContractError(
            f"Scenario object {identifier!r} differs from the registered declaration."
        )
    return registered


def model_scenario_sha256(
    scenario: str | ModelScenario,
    central_thermal_assumptions_sha256: str,
) -> str:
    """Hash the central contract and declared override into an effective checksum."""

    resolved = resolve_model_scenario(scenario)
    return canonical_sha256(
        {
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "central_thermal_assumptions_sha256": central_thermal_assumptions_sha256,
            "model_scenario": resolved.definition(),
        }
    )


def effective_assumption_contract(
    central: AssumptionContract,
    scenario: str | ModelScenario,
) -> AssumptionContract:
    """Return an in-memory scenario contract without editing the source CSV."""

    resolved = resolve_model_scenario(scenario)
    if resolved.scenario_id == "central":
        return central
    frame = central.frame.copy(deep=True)

    def set_numeric(assumption_id: str, value: float) -> None:
        selected = frame["assumption_id"] == assumption_id
        if int(selected.sum()) != 1:
            raise MonteCarloContractError(
                f"Cannot override missing or duplicate thermal assumption {assumption_id!r}."
            )
        frame.loc[selected, "value_numeric"] = float(value)

    if resolved.mass_capacitance_J_m2K is not None:
        set_numeric(
            "network.mass_capacitance_ratio",
            resolved.mass_capacitance_J_m2K,
        )
        set_numeric(
            "network.effective_mass_area_ratio",
            resolved.mass_area_ratio_m2_m2,
        )
        mass_row = frame["assumption_id"] == "network.mass_class"
        if int(mass_row.sum()) != 1:
            raise MonteCarloContractError("Cannot override network.mass_class.")
        frame.loc[mass_row, "value_text"] = (
            "light" if resolved.scenario_id == "mass_light" else "heavy"
        )
    if resolved.vertical_shading_factor is not None:
        set_numeric(
            "solar.external_shading_vertical",
            resolved.vertical_shading_factor,
        )

    effective_sha = model_scenario_sha256(resolved, central.sha256)
    return replace(central, frame=frame, sha256=effective_sha)


def apply_archetype_scenario(
    state: ArchetypeStateInput,
    scenario: str | ModelScenario,
) -> ArchetypeStateInput:
    """Scale all mutually constrained leakage fields together in memory."""

    resolved = resolve_model_scenario(scenario)
    if not isinstance(state, ArchetypeStateInput):
        raise MonteCarloContractError("archetype_state must be ArchetypeStateInput.")
    multiplier = float(resolved.infiltration_multiplier)
    if np.isclose(multiplier, 1.0, rtol=0.0, atol=0.0):
        return state
    values = asdict(state)
    for field in (
        "q50_m3_h",
        "n50_h_1",
        "infiltration_airflow_normal_m3_h",
        "infiltration_ach_normal_h_1",
    ):
        values[field] = float(values[field]) * multiplier
    return validate_archetype_state(values)


def scenario_catalog() -> tuple[ModelScenario, ...]:
    """Return the registry in stable identifier order."""

    return tuple(MODEL_SCENARIOS[key] for key in sorted(MODEL_SCENARIOS))
