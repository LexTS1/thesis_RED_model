"""Executable contracts for the isolated RichardsonPy behaviour layer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from thermal_model.contracts import ContractError, validate_schedule_frame, validate_weather_frame


BEHAVIOUR_DIR = Path(__file__).resolve().parent
DEFAULT_BEHAVIOUR_ASSUMPTIONS_PATH = BEHAVIOUR_DIR / "behaviour_assumptions.csv"
DEFAULT_OCCUPANT_DISTRIBUTION_PATH = (
    BEHAVIOUR_DIR / "data/reference/occupant_distribution.csv"
)

HORIZONTAL_IRRADIANCE_COLUMNS = (
    "I_beam_horizontal_W_m2",
    "I_diffuse_horizontal_W_m2",
    "I_solar_W_m2",
)

PROFILE_COLUMNS = (
    "timestamp_utc",
    "active_occupants_mean",
    "active_occupancy",
    "appliance_electricity_W",
    "lighting_electricity_W",
    "total_electricity_W",
    "occupant_sensible_gain_W",
    "appliance_sensible_gain_W",
    "lighting_sensible_gain_W",
    "Phi_int_W",
    "theta_set_heat_C",
    "theta_set_cool_C",
)

REQUIRED_ASSUMPTION_COLUMNS = {
    "assumption_id",
    "value_numeric",
    "value_text",
    "unit",
    "source_title",
    "source_url",
    "source_locator",
    "rationale",
    "limitation",
    "accessed_on",
}

REQUIRED_ASSUMPTION_IDS = {
    "software.richardsonpy_version",
    "solver.native_timestep_seconds",
    "solver.clock_basis",
    "solver.month_index_correction",
    "electricity.annual_reference_kWh",
    "electricity.prev_heat_dev",
    "electricity.randomize_appliances",
    "electricity.light_config",
    "electricity.season_light_mod",
    "electricity.appliance_sensible_fraction",
    "electricity.lighting_sensible_fraction",
    "occupancy.maximum_supported_count",
    "occupancy.hourly_state_rule",
    "occupancy.metabolic_sensible_W_person",
    "control.heating_active_C",
    "control.heating_inactive_C",
    "control.cooling_C",
    "validation.annual_electricity_tolerance_kWh",
}


class BehaviourContractError(ContractError):
    """Raised when a behavioural input or output violates its contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BehaviourAssumptionContract:
    """Validated behaviour assumptions with typed access and provenance."""

    frame: pd.DataFrame
    path: Path
    sha256: str

    def number(self, assumption_id: str) -> float:
        row = self._row(assumption_id)
        value = row["value_numeric"]
        if pd.isna(value):
            raise BehaviourContractError(
                f"Behaviour assumption {assumption_id!r} has no numeric value."
            )
        return float(value)

    def text(self, assumption_id: str) -> str:
        row = self._row(assumption_id)
        value = row["value_text"]
        if pd.isna(value) or not str(value).strip():
            raise BehaviourContractError(
                f"Behaviour assumption {assumption_id!r} has no text value."
            )
        return str(value).strip()

    def boolean(self, assumption_id: str) -> bool:
        value = self.text(assumption_id).lower()
        if value not in {"true", "false"}:
            raise BehaviourContractError(
                f"Behaviour assumption {assumption_id!r} must be true or false."
            )
        return value == "true"

    def _row(self, assumption_id: str) -> pd.Series:
        selected = self.frame.loc[self.frame["assumption_id"] == assumption_id]
        if len(selected) != 1:
            raise BehaviourContractError(
                f"Expected one behaviour assumption {assumption_id!r}; found {len(selected)}."
            )
        return selected.iloc[0]


@dataclass(frozen=True)
class BehaviourRequest:
    """Inputs needed to generate one reproducible household profile."""

    dwelling_type: str
    weather: pd.DataFrame
    weather_member_id: str
    seed: int
    occupant_count: int | None = None


@dataclass(frozen=True)
class BehaviourDiagnostics:
    """Annual sanity checks and complete stochastic provenance."""

    dwelling_type: str
    dwelling_class: str
    occupant_count: int
    occupant_count_bin: str
    seed: int
    richardson_seed: int
    weather_member_id: str
    weather_forcing_sha256: str
    richardsonpy_version: str
    prev_heat_dev: bool
    annual_electricity_target_kWh: float
    annual_appliance_electricity_kWh: float
    annual_lighting_electricity_kWh: float
    annual_total_electricity_kWh: float
    active_occupant_hours: float
    active_occupancy_hours: int
    mean_internal_gain_W: float
    peak_internal_gain_W: float
    mean_electricity_W: float
    peak_electricity_W: float
    daytime_mean_electricity_W: float
    nighttime_mean_electricity_W: float
    weekday_mean_electricity_W: float
    weekend_mean_electricity_W: float
    behaviour_assumptions_sha256: str
    occupant_distribution_sha256: str


@dataclass(frozen=True)
class BehaviourResult:
    """Detailed behavioural profile; only ``schedules`` crosses into 5R1C."""

    hourly: pd.DataFrame
    diagnostics: BehaviourDiagnostics

    @property
    def schedules(self) -> pd.DataFrame:
        """Return the deliberately narrow thermal-core boundary."""

        return self.hourly[
            [
                "timestamp_utc",
                "Phi_int_W",
                "theta_set_heat_C",
                "theta_set_cool_C",
            ]
        ].copy()


def load_behaviour_assumptions(
    path: str | Path = DEFAULT_BEHAVIOUR_ASSUMPTIONS_PATH,
) -> BehaviourAssumptionContract:
    """Load and validate the frozen Gate-4 behavioural assumptions."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Behaviour assumptions file does not exist: {resolved}")
    frame = pd.read_csv(resolved)
    missing_columns = sorted(REQUIRED_ASSUMPTION_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise BehaviourContractError(
            f"Behaviour assumptions are missing columns: {missing_columns}."
        )
    if frame["assumption_id"].isna().any() or frame["assumption_id"].duplicated().any():
        raise BehaviourContractError("Behaviour assumption identifiers must be unique and non-empty.")
    identifiers = set(frame["assumption_id"].astype(str))
    if identifiers != REQUIRED_ASSUMPTION_IDS:
        raise BehaviourContractError(
            "Behaviour assumption coverage mismatch; "
            f"missing={sorted(REQUIRED_ASSUMPTION_IDS-identifiers)}, "
            f"extra={sorted(identifiers-REQUIRED_ASSUMPTION_IDS)}."
        )
    numeric_text = frame["value_numeric"].fillna("").astype(str).str.strip()
    has_numeric = numeric_text.ne("")
    numeric = pd.to_numeric(frame["value_numeric"], errors="coerce")
    if numeric[has_numeric].isna().any():
        bad = frame.loc[has_numeric & numeric.isna(), "assumption_id"].tolist()
        raise BehaviourContractError(f"Non-numeric behaviour values: {bad}.")
    has_text = frame["value_text"].fillna("").astype(str).str.strip().ne("")
    if (~has_numeric & ~has_text).any():
        bad = frame.loc[~has_numeric & ~has_text, "assumption_id"].tolist()
        raise BehaviourContractError(f"Behaviour assumptions without values: {bad}.")
    urls = frame["source_url"].fillna("").astype(str).str.strip()
    if (urls.ne("") & ~urls.str.startswith("https://")).any():
        raise BehaviourContractError("Behaviour source URLs must use HTTPS.")
    dates = pd.to_datetime(frame["accessed_on"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise BehaviourContractError("Behaviour assumptions contain invalid access dates.")
    for column in ("source_title", "source_url", "source_locator", "rationale", "limitation"):
        if frame[column].fillna("").astype(str).str.strip().eq("").any():
            raise BehaviourContractError(
                f"Behaviour assumption column {column!r} must be complete."
            )

    contract = BehaviourAssumptionContract(frame, resolved, _sha256_file(resolved))
    if contract.number("solver.native_timestep_seconds") != 60.0:
        raise BehaviourContractError("RichardsonPy native timestep must remain 60 seconds.")
    if int(contract.number("occupancy.maximum_supported_count")) != 5:
        raise BehaviourContractError("RichardsonPy occupant cap must remain five.")
    if contract.text("solver.clock_basis") != "fixed_CET_UTC_plus_1_periodic_no_DST":
        raise BehaviourContractError("Unsupported behavioural clock basis.")
    if not contract.boolean("solver.month_index_correction"):
        raise BehaviourContractError(
            "The pinned RichardsonPy month-index correction must remain enabled."
        )
    if not contract.boolean("electricity.prev_heat_dev"):
        raise BehaviourContractError("prev_heat_dev must remain true for this model scope.")
    if contract.text("occupancy.hourly_state_rule") != "any_active_10min":
        raise BehaviourContractError("Unsupported hourly occupancy-state rule.")
    for key in (
        "electricity.appliance_sensible_fraction",
        "electricity.lighting_sensible_fraction",
    ):
        if not 0.0 <= contract.number(key) <= 1.0:
            raise BehaviourContractError(f"{key} must be within [0, 1].")
    if contract.number("electricity.annual_reference_kWh") <= 0.0:
        raise BehaviourContractError("Annual electricity reference must be positive.")
    if contract.number("occupancy.metabolic_sensible_W_person") <= 0.0:
        raise BehaviourContractError("Occupant sensible heat must be positive.")
    inactive = contract.number("control.heating_inactive_C")
    active = contract.number("control.heating_active_C")
    cooling = contract.number("control.cooling_C")
    if not inactive <= active <= cooling:
        raise BehaviourContractError("Behavioural heating/cooling setpoints are inconsistent.")
    return contract


def load_occupant_distribution(
    path: str | Path = DEFAULT_OCCUPANT_DISTRIBUTION_PATH,
) -> tuple[pd.DataFrame, str]:
    """Load the frozen Census-derived SFH/MFH occupant-count distribution."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Occupant distribution does not exist: {resolved}")
    frame = pd.read_csv(resolved)
    required = {
        "dwelling_class",
        "occupant_count",
        "source_bin",
        "dwelling_count",
        "probability",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BehaviourContractError(f"Occupant distribution is missing columns: {missing}.")
    if frame.duplicated(["dwelling_class", "occupant_count"]).any():
        raise BehaviourContractError("Occupant distribution keys must be unique.")
    if frame["source_bin"].fillna("").astype(str).str.strip().eq("").any():
        raise BehaviourContractError("Occupant-distribution source bins must be non-empty.")
    if set(frame["dwelling_class"]) != {"SFH", "MFH"}:
        raise BehaviourContractError("Occupant distribution must contain SFH and MFH.")
    numeric = frame[["occupant_count", "dwelling_count", "probability"]].apply(
        pd.to_numeric, errors="raise"
    )
    frame = frame.copy()
    frame[numeric.columns] = numeric
    for dwelling_class, group in frame.groupby("dwelling_class"):
        if set(group["occupant_count"].astype(int)) != {1, 2, 3, 4, 5}:
            raise BehaviourContractError(
                f"{dwelling_class} occupant counts must be exactly 1 through 5."
            )
        if (group[["dwelling_count", "probability"]].to_numpy() <= 0.0).any():
            raise BehaviourContractError(
                f"{dwelling_class} counts and probabilities must be positive."
            )
        if not np.isclose(group["probability"].sum(), 1.0, atol=1.0e-10):
            raise BehaviourContractError(
                f"{dwelling_class} occupant probabilities do not sum to one."
            )
        expected = group["dwelling_count"] / group["dwelling_count"].sum()
        if not np.allclose(group["probability"], expected, atol=5.0e-10, rtol=0.0):
            raise BehaviourContractError(
                f"{dwelling_class} probabilities do not reconcile with dwelling counts."
            )
    return frame.sort_values(["dwelling_class", "occupant_count"]), _sha256_file(resolved)


def validate_behaviour_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the thermal weather plus climate-layer horizontal irradiance."""

    try:
        weather = validate_weather_frame(frame)
    except ContractError as exc:
        raise BehaviourContractError(str(exc)) from exc
    missing = sorted(set(HORIZONTAL_IRRADIANCE_COLUMNS).difference(weather.columns))
    if missing:
        raise BehaviourContractError(
            f"Behaviour weather is missing climate irradiance columns: {missing}."
        )
    for column in HORIZONTAL_IRRADIANCE_COLUMNS:
        try:
            values = pd.to_numeric(weather[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise BehaviourContractError(f"{column} must be numeric.") from exc
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all() or (array < 0.0).any():
            raise BehaviourContractError(f"{column} must be finite and non-negative.")
        weather[column] = values
    composed = weather["I_beam_horizontal_W_m2"] + weather["I_diffuse_horizontal_W_m2"]
    if not np.allclose(
        weather["I_solar_W_m2"], composed, rtol=0.0, atol=1.0e-6
    ):
        raise BehaviourContractError(
            "I_solar_W_m2 must equal horizontal beam plus diffuse irradiance."
        )
    return weather


def validate_behaviour_request(request: BehaviourRequest) -> BehaviourRequest:
    """Validate and normalize one behaviour request."""

    if not isinstance(request, BehaviourRequest):
        raise BehaviourContractError("request must be a BehaviourRequest instance.")
    if not str(request.dwelling_type).strip():
        raise BehaviourContractError("dwelling_type must be non-empty.")
    if not str(request.weather_member_id).strip():
        raise BehaviourContractError("weather_member_id must be non-empty.")
    if isinstance(request.seed, bool) or not isinstance(request.seed, (int, np.integer)):
        raise BehaviourContractError("seed must be an integer.")
    if not 0 <= int(request.seed) <= 2**32 - 1:
        raise BehaviourContractError("seed must be between 0 and 2**32-1.")
    occupant_count = request.occupant_count
    if occupant_count is not None:
        if isinstance(occupant_count, bool) or not isinstance(
            occupant_count, (int, np.integer)
        ):
            raise BehaviourContractError("occupant_count must be an integer or None.")
        if not 1 <= int(occupant_count) <= 5:
            raise BehaviourContractError("occupant_count must be within 1 through 5.")
        occupant_count = int(occupant_count)
    return BehaviourRequest(
        dwelling_type=str(request.dwelling_type).strip(),
        weather=validate_behaviour_weather(request.weather),
        weather_member_id=str(request.weather_member_id).strip(),
        seed=int(request.seed),
        occupant_count=occupant_count,
    )


def validate_behaviour_result(
    result: BehaviourResult,
    request: BehaviourRequest,
    assumptions: BehaviourAssumptionContract,
) -> None:
    """Fail on non-physical, misaligned, or non-normalized generated profiles."""

    if not isinstance(result, BehaviourResult):
        raise BehaviourContractError("result must be a BehaviourResult instance.")
    missing = sorted(set(PROFILE_COLUMNS).difference(result.hourly.columns))
    if missing:
        raise BehaviourContractError(f"Behaviour result is missing columns: {missing}.")
    try:
        schedules = validate_schedule_frame(result.schedules)
    except ContractError as exc:
        raise BehaviourContractError(str(exc)) from exc
    expected_timestamps = pd.DatetimeIndex(request.weather["timestamp_utc"])
    if not pd.DatetimeIndex(schedules["timestamp_utc"]).equals(expected_timestamps):
        raise BehaviourContractError("Behaviour result does not align with request weather.")

    numeric_columns = [column for column in PROFILE_COLUMNS if column not in {"timestamp_utc", "active_occupancy"}]
    numeric = result.hourly[numeric_columns].apply(pd.to_numeric, errors="raise")
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise BehaviourContractError("Behaviour result contains non-finite values.")
    nonnegative = [
        column
        for column in numeric_columns
        if column not in {"theta_set_heat_C", "theta_set_cool_C"}
    ]
    if (numeric[nonnegative].to_numpy(dtype=float) < -1.0e-10).any():
        raise BehaviourContractError("Behaviour electricity and gains must be non-negative.")
    if (numeric["active_occupants_mean"] > result.diagnostics.occupant_count + 1.0e-10).any():
        raise BehaviourContractError("Active occupants exceed sampled household size.")
    occupancy = result.hourly["active_occupancy"]
    if not occupancy.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise BehaviourContractError("active_occupancy must be boolean.")
    recomposed_electricity = (
        numeric["appliance_electricity_W"] + numeric["lighting_electricity_W"]
    )
    if not np.allclose(
        numeric["total_electricity_W"], recomposed_electricity, atol=1.0e-8, rtol=0.0
    ):
        raise BehaviourContractError("Appliance and lighting electricity do not reconcile.")
    recomposed_gains = (
        numeric["occupant_sensible_gain_W"]
        + numeric["appliance_sensible_gain_W"]
        + numeric["lighting_sensible_gain_W"]
    )
    if not np.allclose(numeric["Phi_int_W"], recomposed_gains, atol=1.0e-8, rtol=0.0):
        raise BehaviourContractError("Internal-gain components do not reconcile.")
    expected_occupant_gain = (
        numeric["active_occupants_mean"]
        * assumptions.number("occupancy.metabolic_sensible_W_person")
    )
    expected_appliance_gain = (
        numeric["appliance_electricity_W"]
        * assumptions.number("electricity.appliance_sensible_fraction")
    )
    expected_lighting_gain = (
        numeric["lighting_electricity_W"]
        * assumptions.number("electricity.lighting_sensible_fraction")
    )
    for column, expected in (
        ("occupant_sensible_gain_W", expected_occupant_gain),
        ("appliance_sensible_gain_W", expected_appliance_gain),
        ("lighting_sensible_gain_W", expected_lighting_gain),
    ):
        if not np.allclose(numeric[column], expected, atol=1.0e-8, rtol=0.0):
            raise BehaviourContractError(f"{column} violates the gain-conversion contract.")
    if not np.array_equal(
        occupancy.to_numpy(dtype=bool),
        numeric["active_occupants_mean"].to_numpy(dtype=float) > 0.0,
    ):
        raise BehaviourContractError(
            "Hourly active state and mean active-occupant count disagree."
        )

    annual_electricity = float(numeric["total_electricity_W"].sum()) / 1000.0
    tolerance = assumptions.number("validation.annual_electricity_tolerance_kWh")
    target = assumptions.number("electricity.annual_reference_kWh")
    if abs(annual_electricity - target) > tolerance:
        raise BehaviourContractError(
            f"Annual electricity {annual_electricity:.6f} kWh misses target "
            f"{target:.6f} kWh by more than {tolerance} kWh."
        )
    if not np.isclose(
        annual_electricity,
        result.diagnostics.annual_total_electricity_kWh,
        atol=1.0e-8,
        rtol=0.0,
    ):
        raise BehaviourContractError("Hourly and diagnostic annual electricity disagree.")
    if result.hourly["total_electricity_W"].nunique() <= 1:
        raise BehaviourContractError("Generated electricity profile has no temporal variation.")
    if occupancy.all() or (~occupancy).all():
        raise BehaviourContractError(
            "Generated active-occupancy profile must contain active and inactive hours."
        )
    active_heat = assumptions.number("control.heating_active_C")
    inactive_heat = assumptions.number("control.heating_inactive_C")
    expected_heat = np.where(occupancy.to_numpy(dtype=bool), active_heat, inactive_heat)
    if not np.allclose(numeric["theta_set_heat_C"], expected_heat, atol=0.0, rtol=0.0):
        raise BehaviourContractError("Heating setpoint does not follow active occupancy.")
    expected_cooling = assumptions.number("control.cooling_C")
    if not np.allclose(
        numeric["theta_set_cool_C"], expected_cooling, atol=0.0, rtol=0.0
    ):
        raise BehaviourContractError("Cooling setpoint violates the fixed central contract.")

    diagnostics = result.diagnostics
    if request.occupant_count is not None and diagnostics.occupant_count != request.occupant_count:
        raise BehaviourContractError("Generated occupant count differs from the fixed request.")
    if not 1 <= diagnostics.occupant_count <= int(
        assumptions.number("occupancy.maximum_supported_count")
    ):
        raise BehaviourContractError("Diagnostic occupant count is outside the supported range.")
    expected_diagnostics = {
        "annual_appliance_electricity_kWh": float(
            numeric["appliance_electricity_W"].sum()
        )
        / 1000.0,
        "annual_lighting_electricity_kWh": float(
            numeric["lighting_electricity_W"].sum()
        )
        / 1000.0,
        "active_occupant_hours": float(numeric["active_occupants_mean"].sum()),
        "active_occupancy_hours": int(occupancy.sum()),
        "mean_internal_gain_W": float(numeric["Phi_int_W"].mean()),
        "peak_internal_gain_W": float(numeric["Phi_int_W"].max()),
        "mean_electricity_W": float(numeric["total_electricity_W"].mean()),
        "peak_electricity_W": float(numeric["total_electricity_W"].max()),
    }
    for field, expected in expected_diagnostics.items():
        actual = getattr(diagnostics, field)
        if not np.isclose(actual, expected, atol=1.0e-8, rtol=0.0):
            raise BehaviourContractError(
                f"Diagnostic {field} does not reconcile with the hourly profile."
            )


def weather_forcing_sha256(weather: pd.DataFrame) -> str:
    """Hash only the timestamps and horizontal lighting forcing used here."""

    digest = hashlib.sha256()
    timestamps = pd.DatetimeIndex(weather["timestamp_utc"]).asi8.astype("<i8", copy=False)
    digest.update(timestamps.tobytes())
    for column in HORIZONTAL_IRRADIANCE_COLUMNS:
        values = weather[column].to_numpy(dtype="<f8", copy=True)
        digest.update(values.tobytes())
    return digest.hexdigest()
