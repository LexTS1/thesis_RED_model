"""Isolated occupant-behaviour boundary conditions for the 5R1C core.

Importing :mod:`thermal_model` does not import RichardsonPy.  The optional
dependency is loaded only when :func:`generate_behaviour` is called.
"""

from .contracts import (
    DEFAULT_BEHAVIOUR_ASSUMPTIONS_PATH,
    DEFAULT_OCCUPANT_DISTRIBUTION_PATH,
    BehaviourAssumptionContract,
    BehaviourContractError,
    BehaviourDiagnostics,
    BehaviourRequest,
    BehaviourResult,
    load_behaviour_assumptions,
    load_occupant_distribution,
    validate_behaviour_result,
    validate_behaviour_weather,
)
from .wrapper import dwelling_class, generate_behaviour, sample_occupant_count

__all__ = [
    "DEFAULT_BEHAVIOUR_ASSUMPTIONS_PATH",
    "DEFAULT_OCCUPANT_DISTRIBUTION_PATH",
    "BehaviourAssumptionContract",
    "BehaviourContractError",
    "BehaviourDiagnostics",
    "BehaviourRequest",
    "BehaviourResult",
    "dwelling_class",
    "generate_behaviour",
    "load_behaviour_assumptions",
    "load_occupant_distribution",
    "sample_occupant_count",
    "validate_behaviour_result",
    "validate_behaviour_weather",
]
