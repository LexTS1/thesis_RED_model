"""Executable data contracts for the residential 5R1C demand model.

This module deliberately contains no thermal solver.  It closes the interface
contract first: every physical assumption has an owner, every table/series has
an exact schema, and invalid inputs fail before the 5R1C equations are called.
"""

from __future__ import annotations

import calendar
import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSUMPTIONS_PATH = Path(__file__).with_name("thermal_assumptions.csv")
HOURS_PER_NON_LEAP_YEAR = 8760
HOURS_PER_LEAP_YEAR = 8784
ALLOWED_YEAR_LENGTHS = {HOURS_PER_NON_LEAP_YEAR, HOURS_PER_LEAP_YEAR}
ORIENTATIONS = ("north", "east", "south", "west")
VENTILATION_SYSTEMS = {
    "existing_unspecified",
    "exhaust_air_ventilation",
    "balanced_mechanical_HRV",
}


class ContractError(ValueError):
    """Raised when an input violates the declared thermal-model contract."""


@dataclass(frozen=True)
class FieldSpec:
    """Machine-readable definition of one input or output field."""

    unit: str
    minimum: float | None = None
    maximum: float | None = None
    strictly_positive: bool = False
    kind: str = "number"


ARCHETYPE_STATE_FIELD_SPECS: dict[str, FieldSpec] = {
    "archetype_id": FieldSpec("1", kind="string"),
    "dwelling_type": FieldSpec("1", kind="string"),
    "construction_period": FieldSpec("1", kind="string"),
    "state_id": FieldSpec("1", kind="string"),
    "floor_surface_area_m2": FieldSpec("m2", strictly_positive=True),
    "protected_volume_m3": FieldSpec("m3", strictly_positive=True),
    "total_building_envelope_area_m2": FieldSpec("m2", strictly_positive=True),
    "roof_area_m2": FieldSpec("m2", minimum=0.0),
    "exterior_wall_area_m2": FieldSpec("m2", minimum=0.0),
    "exterior_wall_bordering_unheated_neighboring_spaces_m2": FieldSpec(
        "m2", minimum=0.0
    ),
    "floor_on_soil_m2": FieldSpec("m2", minimum=0.0),
    "floor_bordering_unheated_neighboring_spaces_m2": FieldSpec("m2", minimum=0.0),
    "doors_area_m2": FieldSpec("m2", minimum=0.0),
    "windows_north_m2": FieldSpec("m2", minimum=0.0),
    "windows_east_m2": FieldSpec("m2", minimum=0.0),
    "windows_south_m2": FieldSpec("m2", minimum=0.0),
    "windows_west_m2": FieldSpec("m2", minimum=0.0),
    "windows_total_m2": FieldSpec("m2", minimum=0.0),
    "U_facade_W_m2K": FieldSpec("W/m2K", strictly_positive=True),
    "U_roof_W_m2K": FieldSpec("W/m2K", strictly_positive=True),
    "U_floor_W_m2K": FieldSpec("W/m2K", strictly_positive=True),
    "U_window_W_m2K": FieldSpec("W/m2K", strictly_positive=True),
    "U_door_W_m2K": FieldSpec("W/m2K", strictly_positive=True),
    "q50_m3_h": FieldSpec("m3/h", minimum=0.0),
    "n50_h_1": FieldSpec("1/h", minimum=0.0),
    "infiltration_n_factor": FieldSpec("1", strictly_positive=True),
    "infiltration_airflow_normal_m3_h": FieldSpec("m3/h", minimum=0.0),
    "infiltration_ach_normal_h_1": FieldSpec("1/h", minimum=0.0),
    "ventilation_system": FieldSpec("1", kind="string"),
    "hrv_eta": FieldSpec("1", minimum=0.0, maximum=1.0),
    "summer_bypass": FieldSpec("1", kind="boolean"),
}

BASE_ARCHETYPE_FIELDS = {
    key
    for key in ARCHETYPE_STATE_FIELD_SPECS
    if key
    not in {
        "state_id",
        "ventilation_system",
        "hrv_eta",
        "summer_bypass",
    }
}

PHYSICAL_STATE_FIELDS = {
    "archetype_id",
    "state_id",
    "U_facade_W_m2K",
    "U_roof_W_m2K",
    "U_floor_W_m2K",
    "U_window_W_m2K",
    "U_door_W_m2K",
    "q50_m3_h",
    "n50_h_1",
    "infiltration_n_factor",
    "infiltration_airflow_normal_m3_h",
    "infiltration_ach_normal_h_1",
    "ventilation_system",
    "hrv_eta",
    "summer_bypass",
}

WEATHER_FIELD_SPECS: dict[str, FieldSpec] = {
    "timestamp_utc": FieldSpec("UTC", kind="timestamp"),
    "T_out_C": FieldSpec("degC", minimum=-50.0, maximum=60.0),
    "I_north_W_m2": FieldSpec("W/m2", minimum=0.0, maximum=1500.0),
    "I_east_W_m2": FieldSpec("W/m2", minimum=0.0, maximum=1500.0),
    "I_south_W_m2": FieldSpec("W/m2", minimum=0.0, maximum=1500.0),
    "I_west_W_m2": FieldSpec("W/m2", minimum=0.0, maximum=1500.0),
}

SCHEDULE_FIELD_SPECS: dict[str, FieldSpec] = {
    "timestamp_utc": FieldSpec("UTC", kind="timestamp"),
    "Phi_int_W": FieldSpec("W", minimum=0.0),
    "theta_set_heat_C": FieldSpec("degC", minimum=5.0, maximum=40.0),
    "theta_set_cool_C": FieldSpec("degC", minimum=5.0, maximum=40.0),
}

RESULT_FIELD_SPECS: dict[str, FieldSpec] = {
    "timestamp_utc": FieldSpec("UTC", kind="timestamp"),
    "T_out_C": FieldSpec("degC", minimum=-50.0, maximum=60.0),
    "theta_air_C": FieldSpec("degC", minimum=-100.0, maximum=100.0),
    "theta_surface_C": FieldSpec("degC", minimum=-100.0, maximum=100.0),
    "theta_mass_C": FieldSpec("degC", minimum=-100.0, maximum=100.0),
    "theta_operative_C": FieldSpec("degC", minimum=-100.0, maximum=100.0),
    "theta_air_free_running_C": FieldSpec(
        "degC", minimum=-100.0, maximum=100.0
    ),
    "Phi_internal_W": FieldSpec("W", minimum=0.0),
    "Phi_solar_W": FieldSpec("W", minimum=0.0),
    "heating_demand_W": FieldSpec("W", minimum=0.0),
    "cooling_demand_W": FieldSpec("W", minimum=0.0),
    "theta_set_heat_C": FieldSpec("degC", minimum=5.0, maximum=40.0),
    "theta_set_cool_C": FieldSpec("degC", minimum=5.0, maximum=40.0),
    "H_ve_W_K": FieldSpec("W/K", minimum=0.0),
    "hrv_bypass_active": FieldSpec("1", kind="boolean"),
}


@dataclass(frozen=True)
class ArchetypeStateInput:
    """Exact combined input schema for the archetype preprocessor."""

    archetype_id: str
    dwelling_type: str
    construction_period: str
    state_id: str
    floor_surface_area_m2: float
    protected_volume_m3: float
    total_building_envelope_area_m2: float
    roof_area_m2: float
    exterior_wall_area_m2: float
    exterior_wall_bordering_unheated_neighboring_spaces_m2: float
    floor_on_soil_m2: float
    floor_bordering_unheated_neighboring_spaces_m2: float
    doors_area_m2: float
    windows_north_m2: float
    windows_east_m2: float
    windows_south_m2: float
    windows_west_m2: float
    windows_total_m2: float
    U_facade_W_m2K: float
    U_roof_W_m2K: float
    U_floor_W_m2K: float
    U_window_W_m2K: float
    U_door_W_m2K: float
    q50_m3_h: float
    n50_h_1: float
    infiltration_n_factor: float
    infiltration_airflow_normal_m3_h: float
    infiltration_ach_normal_h_1: float
    ventilation_system: str
    hrv_eta: float
    summer_bypass: bool


@dataclass(frozen=True)
class PreparedArchetype:
    """Exact output schema of the future archetype-to-5R1C preprocessor."""

    archetype_id: str
    dwelling_type: str
    construction_period: str
    state_id: str
    floor_area_m2: float
    zone_volume_m3: float
    window_area_north_m2: float
    window_area_east_m2: float
    window_area_south_m2: float
    window_area_west_m2: float
    glazing_g_value: float
    window_frame_fraction: float
    non_normal_irradiance_factor: float
    vertical_shading_factor: float
    A_t_m2: float
    A_m_m2: float
    C_m_J_K: float
    H_tr_w_W_K: float
    H_tr_op_W_K: float
    H_tr_is_W_K: float
    H_tr_ms_W_K: float
    H_tr_em_W_K: float
    infiltration_airflow_m3_h: float
    ventilation_ach_h_1: float
    ventilation_system: str
    hrv_efficiency: float
    summer_bypass: bool
    air_density_kg_m3: float
    air_specific_heat_J_kgK: float
    assumptions_sha256: str


@dataclass(frozen=True)
class SimulationInput:
    """Exact request schema for ``simulate``."""

    archetype: PreparedArchetype
    weather: pd.DataFrame
    schedules: pd.DataFrame
    weather_member_id: str
    occupant_seed: int
    model_scenario: str = "central"


@dataclass(frozen=True)
class SimulationDiagnostics:
    """Exact annual and provenance fields returned alongside hourly results."""

    archetype_id: str
    state_id: str
    weather_member_id: str
    occupant_seed: int
    model_scenario: str
    assumptions_sha256: str
    annual_heating_kWh: float
    annual_cooling_kWh: float
    heating_intensity_kWh_m2: float
    cooling_intensity_kWh_m2: float
    peak_heating_W: float
    peak_cooling_W: float
    max_abs_energy_balance_residual_W: float
    warmup_cycles: int


@dataclass(frozen=True)
class SimulationResult:
    """Exact response schema for ``simulate``."""

    hourly: pd.DataFrame
    diagnostics: SimulationDiagnostics


@dataclass(frozen=True)
class AssumptionBinding:
    """Ownership of one row in ``thermal_assumptions.csv``."""

    provider_type: str
    provider_keys: tuple[str, ...]
    consumer: str


@dataclass(frozen=True)
class AssumptionContract:
    """Validated assumptions table with typed access and provenance checksum."""

    frame: pd.DataFrame
    path: Path
    sha256: str

    def number(self, assumption_id: str) -> float:
        row = self._row(assumption_id)
        value = row["value_numeric"]
        if pd.isna(value):
            raise ContractError(f"Assumption {assumption_id!r} has no numeric value.")
        return float(value)

    def text(self, assumption_id: str) -> str:
        row = self._row(assumption_id)
        value = row["value_text"]
        if pd.isna(value) or not str(value).strip():
            raise ContractError(f"Assumption {assumption_id!r} has no text value.")
        return str(value)

    def _row(self, assumption_id: str) -> pd.Series:
        selected = self.frame.loc[self.frame["assumption_id"] == assumption_id]
        if len(selected) != 1:
            raise ContractError(
                f"Expected exactly one assumption {assumption_id!r}; found {len(selected)}."
            )
        return selected.iloc[0]


def _b(provider_type: str, keys: str | tuple[str, ...], consumer: str) -> AssumptionBinding:
    if isinstance(keys, str):
        keys = (keys,)
    return AssumptionBinding(provider_type, keys, consumer)


# This is deliberately exhaustive.  ``validate_assumption_bindings`` fails if a
# row is added to or removed from the CSV without updating its implementation
# owner here.
ASSUMPTION_BINDINGS: dict[str, AssumptionBinding] = {
    "scope.model_topology": _b("implementation_policy", "single_zone_5R1C", "model_scope"),
    "scope.end_uses": _b("implementation_policy", "sensible_heating_and_cooling", "model_scope"),
    "scope.standard_basis": _b("implementation_policy", "ISO_13790_Annex_C", "model_scope"),
    "solver.timestep": _b("assumption_contract", "solver.timestep", "solver"),
    "solver.integration": _b("assumption_contract", "solver.integration", "solver"),
    "solver.controller": _b("assumption_contract", "solver.controller", "controller"),
    "solver.test_load": _b("assumption_contract", "solver.test_load", "controller"),
    "solver.load_sign": _b("implementation_policy", "heating_positive_cooling_negative", "solver"),
    "solver.warmup_initial": _b("assumption_contract", "solver.warmup_initial", "solver"),
    "solver.warmup_method": _b("assumption_contract", "solver.warmup_method", "solver"),
    "solver.warmup_tolerance": _b("assumption_contract", "solver.warmup_tolerance", "solver"),
    "solver.warmup_max_cycles": _b("assumption_contract", "solver.warmup_max_cycles", "solver"),
    "solver.leap_year": _b("input_validator", "validate_hourly_calendar", "input_validator"),
    "control.temperature_node": _b("implementation_policy", "theta_air_C", "controller"),
    "control.heating_reference": _b("schedule_column", "theta_set_heat_C", "controller"),
    "control.cooling_reference": _b("schedule_column", "theta_set_cool_C", "controller"),
    "control.setpoint_invariant": _b("input_validator", "validate_setpoint_deadband", "input_validator"),
    "output.operative_weight_air": _b("assumption_contract", "output.operative_weight_air", "output"),
    "output.operative_weight_surface": _b("assumption_contract", "output.operative_weight_surface", "output"),
    "network.total_surface_ratio": _b("assumption_contract", "network.total_surface_ratio", "preprocessor"),
    "network.air_surface_coefficient": _b("assumption_contract", "network.air_surface_coefficient", "preprocessor"),
    "network.mass_class": _b("assumption_contract", "network.mass_class", "preprocessor"),
    "network.mass_capacitance_ratio": _b("assumption_contract", "network.mass_capacitance_ratio", "preprocessor"),
    "network.effective_mass_area_ratio": _b("assumption_contract", "network.effective_mass_area_ratio", "preprocessor"),
    "network.mass_surface_coefficient": _b("assumption_contract", "network.mass_surface_coefficient", "preprocessor"),
    "network.window_conductance": _b("derived_parameter", "H_tr_w_W_K", "preprocessor"),
    "network.opaque_conductance": _b("derived_parameter", "H_tr_op_W_K", "preprocessor"),
    "network.external_mass_conductance": _b("derived_parameter", "H_tr_em_W_K", "preprocessor"),
    "network.gain_air_fraction": _b("solver_equation", ("Phi_int_W", "Phi_ia_W"), "solver"),
    "network.gain_mass_fraction": _b("solver_equation", ("Phi_int_W", "Phi_solar_W", "Phi_m_W"), "solver"),
    "network.gain_surface_fraction": _b("solver_equation", ("Phi_int_W", "Phi_solar_W", "Phi_st_W"), "solver"),
    "boundary.exterior": _b("assumption_contract", "boundary.exterior", "preprocessor"),
    "boundary.unheated_room": _b("assumption_contract", "boundary.unheated_room", "preprocessor"),
    "boundary.unheated_cellar": _b("assumption_contract", "boundary.unheated_cellar", "preprocessor"),
    "boundary.soil": _b("assumption_contract", "boundary.soil", "preprocessor"),
    "boundary.party_elements": _b("assumption_contract", "boundary.party_elements", "preprocessor"),
    "boundary.thermal_bridges": _b("assumption_contract", "boundary.thermal_bridges", "preprocessor"),
    "solar.glazing_single": _b("assumption_contract", "solar.glazing_single", "preprocessor"),
    "solar.glazing_double": _b("assumption_contract", "solar.glazing_double", "preprocessor"),
    "solar.glazing_low_e_2_0": _b("assumption_contract", "solar.glazing_low_e_2_0", "preprocessor"),
    "solar.glazing_low_e_1_6": _b("assumption_contract", "solar.glazing_low_e_1_6", "preprocessor"),
    "solar.frame_fraction": _b("assumption_contract", "solar.frame_fraction", "preprocessor"),
    "solar.non_normal_factor": _b("assumption_contract", "solar.non_normal_factor", "preprocessor"),
    "solar.external_shading_vertical": _b("assumption_contract", "solar.external_shading_vertical", "preprocessor"),
    "solar.external_shading_horizontal": _b("assumption_contract", "solar.external_shading_horizontal", "preprocessor"),
    "solar.gain_equation": _b(
        "solver_equation",
        ("I_north_W_m2", "I_east_W_m2", "I_south_W_m2", "I_west_W_m2", "Phi_solar_W"),
        "solar_gain",
    ),
    "solar.dynamic_shading": _b("implementation_policy", "dynamic_shading_disabled", "solar_gain"),
    "solar.opaque_gains": _b("implementation_policy", "opaque_solar_gains_zero", "solar_gain"),
    "solar.sky_longwave": _b("implementation_policy", "sky_longwave_zero", "solver"),
    "air.rho": _b("assumption_contract", "air.rho", "preprocessor"),
    "air.cp": _b("assumption_contract", "air.cp", "preprocessor"),
    "ventilation.volume_basis": _b("archetype_field", "protected_volume_m3", "preprocessor"),
    "ventilation.volume_crosscheck": _b("input_validator", "validate_volume_to_floor_ratio", "input_validator"),
    "ventilation.infiltration_input": _b("archetype_field", "infiltration_airflow_normal_m3_h", "preprocessor"),
    "ventilation.infiltration_rule_evidence": _b(
        "upstream_project_field",
        ("q50_m3_h", "n50_h_1", "infiltration_n_factor"),
        "input_validator",
    ),
    "ventilation.use_rate_existing": _b("assumption_contract", "ventilation.use_rate_existing", "preprocessor"),
    "ventilation.use_rate_exhaust": _b("assumption_contract", "ventilation.use_rate_exhaust", "preprocessor"),
    "ventilation.use_rate_balanced": _b("assumption_contract", "ventilation.use_rate_balanced", "preprocessor"),
    "ventilation.hrv_existing": _b("archetype_field", "hrv_eta", "preprocessor"),
    "ventilation.hrv_exhaust": _b("archetype_field", "hrv_eta", "preprocessor"),
    "ventilation.hrv_balanced": _b("archetype_field", "hrv_eta", "preprocessor"),
    "ventilation.summer_bypass": _b("archetype_field", "summer_bypass", "solver"),
    "ventilation.heat_transfer": _b("solver_equation", ("H_ve_W_K", "hrv_bypass_active"), "solver"),
    "input.weather_temperature": _b("weather_column", "T_out_C", "solver"),
    "input.facade_irradiance": _b(
        "weather_column",
        ("I_north_W_m2", "I_east_W_m2", "I_south_W_m2", "I_west_W_m2"),
        "solar_gain",
    ),
    "input.internal_gains": _b("schedule_column", "Phi_int_W", "solver"),
    "validation.internal_gains": _b("validation_default", "validation.internal_gains", "validation"),
    "input.schedule_timezone": _b("input_validator", "schedules_normalized_to_timestamp_utc", "input_validator"),
    "input.weather_timezone": _b("input_validator", "weather_timestamp_utc", "input_validator"),
}


ASSUMPTION_REQUIRED_COLUMNS = {
    "assumption_id",
    "category",
    "parameter_name",
    "symbol_or_field",
    "scope_or_key",
    "value_numeric",
    "value_text",
    "unit",
    "sensitivity_low",
    "sensitivity_high",
    "formula_or_rule",
    "evidence_classification",
    "source_title",
    "source_url",
    "source_path",
    "source_locator",
    "rationale",
    "applicability",
    "limitation",
    "uncertainty_treatment",
    "accessed_on",
}

ALLOWED_ASSUMPTION_UNITS = {
    "1",
    "1/h",
    "J/kgK",
    "J/m2K",
    "K",
    "W",
    "W/K",
    "W/m2",
    "W/m2K",
    "degC",
    "kg/m3",
    "m",
    "m2/m2",
    "m2K/W; 1",
    "m3",
    "m3/h",
    "s",
    "year cycles",
}

ALLOWED_BINDING_PROVIDER_TYPES = {
    "archetype_field",
    "assumption_contract",
    "derived_parameter",
    "implementation_policy",
    "input_validator",
    "schedule_column",
    "solver_equation",
    "upstream_project_field",
    "validation_default",
    "weather_column",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assumption_contract(
    frame: pd.DataFrame, *, project_root: Path = PROJECT_ROOT
) -> None:
    """Validate structure, units, ranges, and referenced local sources."""

    missing = sorted(ASSUMPTION_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ContractError(f"Assumptions file is missing required columns: {missing}")
    if frame.empty:
        raise ContractError("Assumptions file contains no rows.")
    if frame["assumption_id"].isna().any() or frame["assumption_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["assumption_id"].duplicated(keep=False), "assumption_id"]
            .dropna()
            .astype(str)
            .unique()
        )
        raise ContractError(f"Assumption identifiers must be non-empty and unique: {duplicates}")

    numeric = pd.to_numeric(frame["value_numeric"], errors="coerce")
    low = pd.to_numeric(frame["sensitivity_low"], errors="coerce")
    high = pd.to_numeric(frame["sensitivity_high"], errors="coerce")
    nonblank_numeric = frame["value_numeric"].notna() & frame["value_numeric"].astype(str).str.strip().ne("")
    if numeric[nonblank_numeric].isna().any():
        bad = frame.loc[nonblank_numeric & numeric.isna(), "assumption_id"].tolist()
        raise ContractError(f"Non-numeric central values in value_numeric: {bad}")
    has_text = frame["value_text"].fillna("").astype(str).str.strip().ne("")
    if (~nonblank_numeric & ~has_text).any():
        bad = frame.loc[~nonblank_numeric & ~has_text, "assumption_id"].tolist()
        raise ContractError(f"Every assumption needs a numeric or text value: {bad}")

    units = frame["unit"].fillna("").astype(str).str.strip()
    if units[nonblank_numeric].eq("").any():
        bad = frame.loc[nonblank_numeric & units.eq(""), "assumption_id"].tolist()
        raise ContractError(f"Numeric assumptions require units: {bad}")
    unknown_units = sorted(set(units[units.ne("")]).difference(ALLOWED_ASSUMPTION_UNITS))
    if unknown_units:
        raise ContractError(f"Unknown assumption units: {unknown_units}")

    one_bound_missing = low.isna() ^ high.isna()
    if one_bound_missing.any():
        bad = frame.loc[one_bound_missing, "assumption_id"].tolist()
        raise ContractError(f"Sensitivity bounds must be supplied as a pair: {bad}")
    inverted = low.notna() & (low > high)
    if inverted.any():
        bad = frame.loc[inverted, "assumption_id"].tolist()
        raise ContractError(f"Sensitivity ranges are inverted: {bad}")
    central_outside = numeric.notna() & low.notna() & ((numeric < low) | (numeric > high))
    if central_outside.any():
        bad = frame.loc[central_outside, "assumption_id"].tolist()
        raise ContractError(f"Central values outside sensitivity ranges: {bad}")

    urls = frame["source_url"].fillna("").astype(str).str.strip()
    bad_urls = frame.loc[urls.ne("") & ~urls.str.startswith("https://"), "assumption_id"].tolist()
    if bad_urls:
        raise ContractError(f"Source URLs must use HTTPS: {bad_urls}")
    for row in frame.loc[frame["source_path"].notna()].itertuples(index=False):
        source_path = str(row.source_path).strip()
        if source_path and not (project_root / source_path).is_file():
            raise ContractError(
                f"Assumption {row.assumption_id!r} references missing local source {source_path!r}."
            )
    parsed_dates = pd.to_datetime(frame["accessed_on"], format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isna().any():
        bad = frame.loc[parsed_dates.isna(), "assumption_id"].tolist()
        raise ContractError(f"Invalid accessed_on dates: {bad}")


def load_assumption_contract(
    path: str | Path = DEFAULT_ASSUMPTIONS_PATH,
) -> AssumptionContract:
    """Load and fully validate the authoritative thermal assumptions CSV."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Thermal assumptions file does not exist: {resolved}")
    frame = pd.read_csv(resolved)
    validate_assumption_contract(frame)
    contract = AssumptionContract(frame=frame, path=resolved, sha256=_sha256_file(resolved))
    validate_assumption_bindings(contract)
    return contract


def validate_assumption_bindings(contract: AssumptionContract) -> None:
    """Require exactly one concrete owner for every assumption row."""

    assumption_ids = set(contract.frame["assumption_id"].astype(str))
    binding_ids = set(ASSUMPTION_BINDINGS)
    missing = sorted(assumption_ids - binding_ids)
    extra = sorted(binding_ids - assumption_ids)
    if missing or extra:
        raise ContractError(
            f"Assumption binding coverage mismatch; missing={missing}, extra={extra}."
        )

    prepared_fields = {field.name for field in fields(PreparedArchetype)}
    for assumption_id, binding in ASSUMPTION_BINDINGS.items():
        if binding.provider_type not in ALLOWED_BINDING_PROVIDER_TYPES:
            raise ContractError(
                f"Assumption {assumption_id!r} has unknown provider type "
                f"{binding.provider_type!r}."
            )
        if not binding.provider_keys or any(not key for key in binding.provider_keys):
            raise ContractError(f"Assumption {assumption_id!r} has no provider key.")
        if binding.provider_type == "assumption_contract" and binding.provider_keys != (
            assumption_id,
        ):
            raise ContractError(
                f"Contract-owned assumption {assumption_id!r} must bind to its own row."
            )
        if binding.provider_type in {"archetype_field", "upstream_project_field"}:
            unknown = set(binding.provider_keys).difference(ARCHETYPE_STATE_FIELD_SPECS)
            if unknown:
                raise ContractError(
                    f"Assumption {assumption_id!r} references unknown archetype fields: "
                    f"{sorted(unknown)}"
                )
        if binding.provider_type == "weather_column":
            unknown = set(binding.provider_keys).difference(WEATHER_FIELD_SPECS)
            if unknown:
                raise ContractError(
                    f"Assumption {assumption_id!r} references unknown weather fields: "
                    f"{sorted(unknown)}"
                )
        if binding.provider_type == "schedule_column":
            unknown = set(binding.provider_keys).difference(SCHEDULE_FIELD_SPECS)
            if unknown:
                raise ContractError(
                    f"Assumption {assumption_id!r} references unknown schedule fields: "
                    f"{sorted(unknown)}"
                )
        if binding.provider_type == "derived_parameter":
            unknown = set(binding.provider_keys).difference(prepared_fields)
            if unknown:
                raise ContractError(
                    f"Assumption {assumption_id!r} references unknown prepared fields: "
                    f"{sorted(unknown)}"
                )


def _as_mapping(record: Mapping[str, Any] | pd.Series, label: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) and not isinstance(record, pd.Series):
        raise ContractError(f"{label} must be a mapping or pandas Series.")
    return record


def _missing_fields(record: Mapping[str, Any], required: set[str]) -> list[str]:
    return sorted(key for key in required if key not in record)


def _coerce_bool(value: Any, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    raise ContractError(f"{field} must be an explicit boolean; got {value!r}.")


def _missing_scalar(value: Any) -> bool:
    return value is None or (not isinstance(value, str) and bool(pd.isna(value))) or (
        isinstance(value, str) and not value.strip()
    )


def assemble_archetype_state(
    base_archetype: Mapping[str, Any] | pd.Series,
    physical_state: Mapping[str, Any] | pd.Series,
) -> ArchetypeStateInput:
    """Join immutable base geometry to one projection-specific physical state.

    The 2050 state matrix intentionally does not repeat geometry, so the join on
    ``archetype_id`` is an explicit part of the contract rather than a hidden
    preprocessing assumption.
    """

    base = _as_mapping(base_archetype, "base_archetype")
    state = _as_mapping(physical_state, "physical_state")
    missing_base = _missing_fields(base, BASE_ARCHETYPE_FIELDS)
    missing_state = _missing_fields(state, PHYSICAL_STATE_FIELDS)
    if missing_base or missing_state:
        raise ContractError(
            f"Cannot assemble archetype state; missing base fields={missing_base}, "
            f"missing physical-state fields={missing_state}."
        )
    if str(base["archetype_id"]) != str(state["archetype_id"]):
        raise ContractError(
            "Base archetype and physical state have different archetype_id values: "
            f"{base['archetype_id']!r} != {state['archetype_id']!r}."
        )

    combined = {key: base[key] for key in BASE_ARCHETYPE_FIELDS}
    combined.update({key: state[key] for key in PHYSICAL_STATE_FIELDS})
    ventilation_system = str(combined["ventilation_system"]).strip()
    if _missing_scalar(combined["hrv_eta"]):
        if ventilation_system in {"existing_unspecified", "exhaust_air_ventilation"}:
            combined["hrv_eta"] = 0.0
        else:
            raise ContractError("balanced_mechanical_HRV requires an explicit hrv_eta.")
    if _missing_scalar(combined["summer_bypass"]):
        if ventilation_system in {"existing_unspecified", "exhaust_air_ventilation"}:
            combined["summer_bypass"] = False
        else:
            raise ContractError("balanced_mechanical_HRV requires an explicit summer_bypass.")
    return validate_archetype_state(combined)


def validate_archetype_state(record: Mapping[str, Any]) -> ArchetypeStateInput:
    """Validate and type the exact archetype-preprocessor input schema."""

    missing = _missing_fields(record, set(ARCHETYPE_STATE_FIELD_SPECS))
    if missing:
        raise ContractError(f"Archetype state is missing required fields: {missing}")
    values: dict[str, Any] = {}
    for name, spec in ARCHETYPE_STATE_FIELD_SPECS.items():
        value = record[name]
        if spec.kind == "string":
            if _missing_scalar(value):
                raise ContractError(f"Archetype field {name} must be a non-empty string.")
            values[name] = str(value).strip()
        elif spec.kind == "boolean":
            values[name] = _coerce_bool(value, name)
        else:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ContractError(f"Archetype field {name} must be numeric; got {value!r}.") from exc
            if not np.isfinite(number):
                raise ContractError(f"Archetype field {name} must be finite; got {value!r}.")
            if spec.strictly_positive and number <= 0.0:
                raise ContractError(f"Archetype field {name} must be greater than zero.")
            if spec.minimum is not None and number < spec.minimum:
                raise ContractError(f"Archetype field {name} must be >= {spec.minimum} {spec.unit}.")
            if spec.maximum is not None and number > spec.maximum:
                raise ContractError(f"Archetype field {name} must be <= {spec.maximum} {spec.unit}.")
            values[name] = number

    if values["ventilation_system"] not in VENTILATION_SYSTEMS:
        raise ContractError(
            f"Unknown ventilation_system {values['ventilation_system']!r}; "
            f"expected one of {sorted(VENTILATION_SYSTEMS)}."
        )
    if values["ventilation_system"] != "balanced_mechanical_HRV":
        if not np.isclose(values["hrv_eta"], 0.0, rtol=0.0, atol=1e-12):
            raise ContractError("Heat recovery is not allowed for existing/exhaust ventilation.")
        if values["summer_bypass"]:
            raise ContractError("summer_bypass is only allowed for balanced_mechanical_HRV.")

    oriented_window_area = sum(values[f"windows_{orientation}_m2"] for orientation in ORIENTATIONS)
    if not np.isclose(
        oriented_window_area, values["windows_total_m2"], rtol=0.0, atol=1e-8
    ):
        raise ContractError(
            "Oriented window areas do not sum to windows_total_m2: "
            f"{oriented_window_area} != {values['windows_total_m2']}."
        )
    component_area = sum(
        values[name]
        for name in (
            "roof_area_m2",
            "exterior_wall_area_m2",
            "exterior_wall_bordering_unheated_neighboring_spaces_m2",
            "floor_on_soil_m2",
            "floor_bordering_unheated_neighboring_spaces_m2",
            "doors_area_m2",
            "windows_total_m2",
        )
    )
    if not np.isclose(
        component_area, values["total_building_envelope_area_m2"], rtol=0.0, atol=0.25
    ):
        raise ContractError(
            "Envelope component areas do not reconcile with total_building_envelope_area_m2 "
            f"within the 0.25 m2 source-rounding tolerance: {component_area} != "
            f"{values['total_building_envelope_area_m2']}."
        )
    volume_to_floor = values["protected_volume_m3"] / values["floor_surface_area_m2"]
    if not 1.5 <= volume_to_floor <= 5.0:
        raise ContractError(
            "protected_volume_m3 / floor_surface_area_m2 is outside the broad "
            f"1.5–5.0 m plausibility range: {volume_to_floor}."
        )

    factor = values["infiltration_n_factor"]
    if not np.isclose(
        values["q50_m3_h"] / factor,
        values["infiltration_airflow_normal_m3_h"],
        rtol=1e-9,
        atol=1e-8,
    ):
        raise ContractError("infiltration_airflow_normal_m3_h is inconsistent with q50/n-factor.")
    if not np.isclose(
        values["n50_h_1"] / factor,
        values["infiltration_ach_normal_h_1"],
        rtol=1e-9,
        atol=1e-10,
    ):
        raise ContractError("infiltration_ach_normal_h_1 is inconsistent with n50/n-factor.")
    return ArchetypeStateInput(**values)


def validate_prepared_archetype(archetype: PreparedArchetype) -> None:
    """Validate the exact preprocessor output before it reaches the solver."""

    if not isinstance(archetype, PreparedArchetype):
        raise ContractError("archetype must be a PreparedArchetype instance.")
    for name in ("archetype_id", "dwelling_type", "construction_period", "state_id"):
        if not str(getattr(archetype, name)).strip():
            raise ContractError(f"Prepared archetype field {name} must be non-empty.")
    nonnegative = (
        "window_area_north_m2",
        "window_area_east_m2",
        "window_area_south_m2",
        "window_area_west_m2",
        "infiltration_airflow_m3_h",
        "ventilation_ach_h_1",
    )
    positive = (
        "floor_area_m2",
        "zone_volume_m3",
        "A_t_m2",
        "A_m_m2",
        "C_m_J_K",
        "H_tr_op_W_K",
        "H_tr_is_W_K",
        "H_tr_ms_W_K",
        "H_tr_em_W_K",
        "air_density_kg_m3",
        "air_specific_heat_J_kgK",
    )
    nonnegative = (*nonnegative, "H_tr_w_W_K")
    bounded_fractions = (
        "glazing_g_value",
        "window_frame_fraction",
        "non_normal_irradiance_factor",
        "vertical_shading_factor",
        "hrv_efficiency",
    )
    for name in (*nonnegative, *positive, *bounded_fractions):
        value = float(getattr(archetype, name))
        if not np.isfinite(value):
            raise ContractError(f"Prepared archetype field {name} must be finite.")
        if name in nonnegative and value < 0.0:
            raise ContractError(f"Prepared archetype field {name} must be non-negative.")
        if name in positive and value <= 0.0:
            raise ContractError(f"Prepared archetype field {name} must be positive.")
        if name in bounded_fractions and not 0.0 <= value <= 1.0:
            raise ContractError(f"Prepared archetype field {name} must be between zero and one.")
    if archetype.ventilation_system not in VENTILATION_SYSTEMS:
        raise ContractError(
            f"Prepared archetype has unknown ventilation system {archetype.ventilation_system!r}."
        )
    if archetype.ventilation_system != "balanced_mechanical_HRV" and (
        not np.isclose(archetype.hrv_efficiency, 0.0, rtol=0.0, atol=1e-12)
        or archetype.summer_bypass
    ):
        raise ContractError("Only balanced mechanical ventilation may have HRV or bypass.")
    if archetype.H_tr_op_W_K >= archetype.H_tr_ms_W_K:
        raise ContractError(
            "Prepared archetype requires H_tr_op_W_K < H_tr_ms_W_K for a positive H_tr_em."
        )
    if archetype.A_m_m2 >= archetype.A_t_m2:
        raise ContractError("Prepared archetype requires A_m_m2 < A_t_m2.")
    expected_H_tr_em = 1.0 / (
        1.0 / archetype.H_tr_op_W_K - 1.0 / archetype.H_tr_ms_W_K
    )
    if not np.isclose(
        archetype.H_tr_em_W_K, expected_H_tr_em, rtol=1e-10, atol=1e-10
    ):
        raise ContractError(
            "H_tr_em_W_K is inconsistent with H_tr_op_W_K and H_tr_ms_W_K."
        )
    surface_gain_fraction = (
        1.0
        - archetype.A_m_m2 / archetype.A_t_m2
        - archetype.H_tr_w_W_K
        / ((archetype.H_tr_ms_W_K / archetype.A_m_m2) * archetype.A_t_m2)
    )
    if surface_gain_fraction < -1e-12:
        raise ContractError(
            "Prepared archetype produces a negative ISO surface-gain allocation fraction."
        )
    if len(archetype.assumptions_sha256) != 64:
        raise ContractError("assumptions_sha256 must contain a 64-character SHA-256 digest.")
    try:
        int(archetype.assumptions_sha256, 16)
    except ValueError as exc:
        raise ContractError("assumptions_sha256 is not hexadecimal.") from exc


def _normalize_timestamps(series: pd.Series, label: str) -> pd.DatetimeIndex:
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label}.timestamp_utc cannot be parsed as timestamps.") from exc
    try:
        timestamps = pd.DatetimeIndex(parsed)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"{label}.timestamp_utc must use one explicit timezone; mixed offsets are not allowed."
        ) from exc
    if timestamps.tz is None:
        raise ContractError(
            f"{label}.timestamp_utc is timezone-naive; convert explicitly to UTC before simulation."
        )
    return timestamps.tz_convert("UTC")


def _validate_hourly_calendar(timestamps: pd.DatetimeIndex, label: str) -> None:
    if len(timestamps) not in ALLOWED_YEAR_LENGTHS:
        raise ContractError(
            f"{label} must contain 8760 or 8784 hourly rows; found {len(timestamps)}."
        )
    if timestamps.has_duplicates:
        raise ContractError(f"{label}.timestamp_utc contains duplicate hours.")
    if not timestamps.is_monotonic_increasing:
        raise ContractError(f"{label}.timestamp_utc must be strictly increasing.")
    intervals = timestamps.to_series(index=range(len(timestamps))).diff().dropna()
    if not intervals.eq(pd.Timedelta(hours=1)).all():
        raise ContractError(f"{label}.timestamp_utc must be continuous at exactly one-hour intervals.")
    years = timestamps.year.unique()
    if len(years) != 1:
        raise ContractError(f"{label} must contain exactly one UTC calendar year.")
    year = int(years[0])
    expected_length = HOURS_PER_LEAP_YEAR if calendar.isleap(year) else HOURS_PER_NON_LEAP_YEAR
    if len(timestamps) != expected_length:
        raise ContractError(
            f"{label} has {len(timestamps)} hours but UTC calendar year {year} requires "
            f"{expected_length}."
        )
    expected_start = pd.Timestamp(year=year, month=1, day=1, hour=0, tz="UTC")
    expected_end = pd.Timestamp(year=year, month=12, day=31, hour=23, tz="UTC")
    if timestamps[0] != expected_start or timestamps[-1] != expected_end:
        raise ContractError(
            f"{label} must span {expected_start.isoformat()} through {expected_end.isoformat()}."
        )


def _validate_hourly_frame(
    frame: pd.DataFrame,
    specs: Mapping[str, FieldSpec],
    label: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ContractError(f"{label} must be a pandas DataFrame.")
    missing = sorted(set(specs).difference(frame.columns))
    if missing:
        raise ContractError(f"{label} is missing required columns: {missing}")
    normalized = frame.copy()
    timestamps = _normalize_timestamps(normalized["timestamp_utc"], label)
    _validate_hourly_calendar(timestamps, label)
    normalized["timestamp_utc"] = timestamps

    for name, spec in specs.items():
        if name == "timestamp_utc":
            continue
        if spec.kind == "boolean":
            try:
                normalized[name] = normalized[name].map(lambda value: _coerce_bool(value, name))
            except ContractError as exc:
                raise ContractError(f"{label}.{name}: {exc}") from exc
            continue
        try:
            values = pd.to_numeric(normalized[name], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{label}.{name} must be numeric.") from exc
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all():
            raise ContractError(f"{label}.{name} contains missing or non-finite values.")
        if spec.strictly_positive and (array <= 0.0).any():
            raise ContractError(f"{label}.{name} must be strictly positive.")
        if spec.minimum is not None and (array < spec.minimum).any():
            minimum = float(array.min())
            raise ContractError(
                f"{label}.{name} contains {minimum}, below {spec.minimum} {spec.unit}."
            )
        if spec.maximum is not None and (array > spec.maximum).any():
            maximum = float(array.max())
            raise ContractError(
                f"{label}.{name} contains {maximum}, above {spec.maximum} {spec.unit}."
            )
        normalized[name] = values
    return normalized


def validate_weather_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize one climate member plus its façade forcing."""

    return _validate_hourly_frame(frame, WEATHER_FIELD_SPECS, "weather")


def validate_schedule_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate UTC-aligned internal gains and temperature bounds."""

    normalized = _validate_hourly_frame(frame, SCHEDULE_FIELD_SPECS, "schedules")
    invalid = normalized["theta_set_heat_C"] > normalized["theta_set_cool_C"]
    if invalid.any():
        first = normalized.loc[invalid, "timestamp_utc"].iloc[0]
        raise ContractError(f"Heating setpoint exceeds cooling setpoint at {first.isoformat()}.")
    return normalized


def validate_simulation_input(request: SimulationInput) -> SimulationInput:
    """Validate and normalize the full ``simulate`` request."""

    if not isinstance(request, SimulationInput):
        raise ContractError("request must be a SimulationInput instance.")
    validate_prepared_archetype(request.archetype)
    if not str(request.weather_member_id).strip():
        raise ContractError("weather_member_id must be non-empty.")
    if not str(request.model_scenario).strip():
        raise ContractError("model_scenario must be non-empty.")
    if isinstance(request.occupant_seed, bool) or not isinstance(
        request.occupant_seed, (int, np.integer)
    ):
        raise ContractError("occupant_seed must be an integer.")
    if not 0 <= int(request.occupant_seed) <= 2**32 - 1:
        raise ContractError("occupant_seed must be between 0 and 2**32-1.")

    weather = validate_weather_frame(request.weather)
    schedules = validate_schedule_frame(request.schedules)
    if not pd.DatetimeIndex(weather["timestamp_utc"]).equals(
        pd.DatetimeIndex(schedules["timestamp_utc"])
    ):
        raise ContractError("Weather and schedule timestamps are not exactly aligned.")
    return SimulationInput(
        archetype=request.archetype,
        weather=weather,
        schedules=schedules,
        weather_member_id=str(request.weather_member_id).strip(),
        occupant_seed=int(request.occupant_seed),
        model_scenario=str(request.model_scenario).strip(),
    )


def validate_simulation_result(
    result: SimulationResult,
    request: SimulationInput,
    *,
    atol: float = 1e-6,
) -> SimulationResult:
    """Validate the exact hourly/annual response contract of ``simulate``."""

    if not isinstance(result, SimulationResult):
        raise ContractError("result must be a SimulationResult instance.")
    validated_request = validate_simulation_input(request)
    hourly = _validate_hourly_frame(result.hourly, RESULT_FIELD_SPECS, "result.hourly")
    expected_timestamps = pd.DatetimeIndex(validated_request.weather["timestamp_utc"])
    if not pd.DatetimeIndex(hourly["timestamp_utc"]).equals(expected_timestamps):
        raise ContractError("Result timestamps do not match the simulation request.")
    if not np.allclose(
        hourly["T_out_C"], validated_request.weather["T_out_C"], rtol=0.0, atol=atol
    ):
        raise ContractError("Result outdoor temperatures do not match the weather input.")
    for result_name, schedule_name in (
        ("Phi_internal_W", "Phi_int_W"),
        ("theta_set_heat_C", "theta_set_heat_C"),
        ("theta_set_cool_C", "theta_set_cool_C"),
    ):
        if not np.allclose(
            hourly[result_name],
            validated_request.schedules[schedule_name],
            rtol=0.0,
            atol=atol,
        ):
            raise ContractError(f"Result {result_name} does not match the corresponding input.")
    simultaneous = (hourly["heating_demand_W"] > atol) & (
        hourly["cooling_demand_W"] > atol
    )
    if simultaneous.any():
        first = hourly.loc[simultaneous, "timestamp_utc"].iloc[0]
        raise ContractError(f"Heating and cooling are simultaneous at {first.isoformat()}.")
    expected_operative = 0.3 * hourly["theta_air_C"] + 0.7 * hourly["theta_surface_C"]
    if not np.allclose(
        hourly["theta_operative_C"], expected_operative, rtol=0.0, atol=atol
    ):
        raise ContractError("theta_operative_C violates the 0.3 air / 0.7 surface definition.")

    diagnostics = result.diagnostics
    if not isinstance(diagnostics, SimulationDiagnostics):
        raise ContractError("result.diagnostics must be a SimulationDiagnostics instance.")
    identity_pairs = {
        "archetype_id": validated_request.archetype.archetype_id,
        "state_id": validated_request.archetype.state_id,
        "weather_member_id": validated_request.weather_member_id,
        "occupant_seed": validated_request.occupant_seed,
        "model_scenario": validated_request.model_scenario,
        "assumptions_sha256": validated_request.archetype.assumptions_sha256,
    }
    for name, expected in identity_pairs.items():
        if getattr(diagnostics, name) != expected:
            raise ContractError(
                f"Diagnostics {name}={getattr(diagnostics, name)!r}; expected {expected!r}."
            )
    nonnegative_diagnostics = (
        "annual_heating_kWh",
        "annual_cooling_kWh",
        "heating_intensity_kWh_m2",
        "cooling_intensity_kWh_m2",
        "peak_heating_W",
        "peak_cooling_W",
        "max_abs_energy_balance_residual_W",
    )
    for name in nonnegative_diagnostics:
        value = float(getattr(diagnostics, name))
        if not np.isfinite(value) or value < 0.0:
            raise ContractError(f"Diagnostics field {name} must be finite and non-negative.")
    if not 1 <= diagnostics.warmup_cycles <= 10:
        raise ContractError("warmup_cycles must be between 1 and 10.")

    annual_heating = float(hourly["heating_demand_W"].sum()) / 1000.0
    annual_cooling = float(hourly["cooling_demand_W"].sum()) / 1000.0
    expected_metrics = {
        "annual_heating_kWh": annual_heating,
        "annual_cooling_kWh": annual_cooling,
        "heating_intensity_kWh_m2": annual_heating / validated_request.archetype.floor_area_m2,
        "cooling_intensity_kWh_m2": annual_cooling / validated_request.archetype.floor_area_m2,
        "peak_heating_W": float(hourly["heating_demand_W"].max()),
        "peak_cooling_W": float(hourly["cooling_demand_W"].max()),
    }
    for name, expected in expected_metrics.items():
        if not np.isclose(float(getattr(diagnostics, name)), expected, rtol=1e-9, atol=atol):
            raise ContractError(
                f"Diagnostics {name}={getattr(diagnostics, name)} does not match hourly result "
                f"value {expected}."
            )
    return SimulationResult(hourly=hourly, diagnostics=diagnostics)
