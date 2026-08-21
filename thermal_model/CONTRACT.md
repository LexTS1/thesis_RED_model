# Residential thermal model contract

## Status

Gates 1–3 are complete. The assumptions, their implementation owners, the archetype-preprocessor interface, the simulation interface, and the fail-fast input/output checks are executable and tested. Gate 2 implements this frozen interface in `core.py`; the equations are documented in `THERMAL_CORE.md`. Gate 3 adds the layered equation verification, deterministic 75-cell Belgian TABULA validation, calibration rules and sensitivity screen documented in `GATE3_VERIFICATION_VALIDATION.md`.

Authoritative files:

- `thermal_assumptions.csv`: physical, numerical, and scope assumptions with units and sources.
- `contracts.py`: typed schemas, exhaustive assumption bindings, and validators.
- `core.py`: archetype preprocessing, 5R1C equations, ideal control and periodic convergence.
- `validation.py`: reproducible deterministic Belgian validation and sensitivity runner.
- `tests/`: contract, equation, analytical, numerical and validation-provenance tests.
- `data/validation/`: persisted Gate 3 results and machine-readable statuses.

## Assumption ownership

`ASSUMPTION_BINDINGS` assigns every row in `thermal_assumptions.csv` to exactly one provider and consumer. Loading the assumptions file fails if an assumption is unbound or a stale binding remains.

| Provider type | Rows | Meaning |
|---|---:|---|
| Assumptions contract | 35 | Fixed central value or named method read from the CSV |
| Implementation policy | 8 | Explicit scope or exclusion enforced by model structure |
| Archetype field | 6 | Supplied by the combined physical archetype/state |
| Input validator | 5 | Enforced before preprocessing or simulation |
| Solver equation | 5 | Calculated hourly by a specified equation |
| Schedule column | 3 | Supplied by the behavioural wrapper after UTC alignment |
| Derived parameter | 3 | Calculated once by the archetype preprocessor |
| Weather column | 2 | Supplied by the climate layer; the façade binding expands to four orientations |
| Upstream project field | 1 | Audited rather than recomputed silently |
| Validation default | 1 | Used only for the deterministic reference case |

The bindings are consumed by the preprocessor (33 rows), solver (15), validators (6), controller (5), solar-gain calculation (4), scope (3), output definition (2), and deterministic validation (1).

## Archetype preprocessor interface

The physical-state projection matrix does not repeat dwelling geometry. The
join to the base archetype matrix is therefore an explicit contract operation:

```python
state = assemble_archetype_state(base_archetype_row, physical_state_row)
prepared = preprocess_archetype(state, assumptions)
```

`assemble_archetype_state` joins only on an identical `archetype_id`. Extra stock-weight and policy columns are permitted but do not enter the thermal preprocessor.

### Base geometry fields

| Group | Required fields | Units |
|---|---|---|
| Identity | `archetype_id`, `dwelling_type`, `construction_period` | categorical |
| Conditioned geometry | `floor_surface_area_m2`, `protected_volume_m3`, `total_building_envelope_area_m2` | m², m³, m² |
| Opaque areas | `roof_area_m2`, `exterior_wall_area_m2`, `exterior_wall_bordering_unheated_neighboring_spaces_m2`, `floor_on_soil_m2`, `floor_bordering_unheated_neighboring_spaces_m2`, `doors_area_m2` | m² |
| Window areas | `windows_north_m2`, `windows_east_m2`, `windows_south_m2`, `windows_west_m2`, `windows_total_m2` | m² |

The base matrix also contains current-state thermal fields. These are required for traceability, but the selected physical-state row supplies the final thermal values used by preprocessing.

### Physical-state fields

| Group | Required fields | Units |
|---|---|---|
| Identity | `archetype_id`, `state_id` | categorical |
| U-values | `U_facade_W_m2K`, `U_roof_W_m2K`, `U_floor_W_m2K`, `U_window_W_m2K`, `U_door_W_m2K` | W/(m²·K) |
| Leakage | `q50_m3_h`, `n50_h_1`, `infiltration_n_factor`, `infiltration_airflow_normal_m3_h`, `infiltration_ach_normal_h_1` | m³/h, 1/h, 1, m³/h, 1/h |
| Ventilation | `ventilation_system`, `hrv_eta`, `summer_bypass` | categorical, 1, boolean |

Blank HRV efficiency and bypass values are allowed only for `existing_unspecified` and `exhaust_air_ventilation`; they are normalized to zero and false. Balanced HRV requires both values explicitly.

### Preprocessor output

`PreparedArchetype` is the frozen output schema. It contains:

- Identity and conditioned floor area/volume.
- Four oriented window areas.
- Glazing g-value, frame fraction, non-normal incidence factor, and vertical shading factor.
- $A_t$, $A_m$, and $C_m$.
- $H_{tr,w}$, $H_{tr,op}$, $H_{tr,is}$, $H_{tr,ms}$, and $H_{tr,em}$.
- Infiltration airflow and ventilation ACH.
- Ventilation system, HRV efficiency, and summer-bypass flag.
- Air density and specific heat.
- SHA-256 checksum of the assumptions file used to prepare it.

The output validator requires positive opaque and inter-node conductances, permits non-negative $H_{tr,w}$, and enforces $H_{tr,op}<H_{tr,ms}$, which is necessary for a positive equivalent $H_{tr,em}$.

## Simulation interface

The frozen interface is:

```python
simulate(request: SimulationInput) -> SimulationResult
```

`SimulationInput` contains:

- One `PreparedArchetype`.
- One complete hourly weather DataFrame.
- One complete hourly schedules DataFrame.
- `weather_member_id`.
- Unsigned 32-bit `occupant_seed`.
- `model_scenario`, defaulting to `central`.

### Weather schema

| Column | Unit | Rule |
|---|---|---|
| `timestamp_utc` | UTC | Timezone-aware, continuous, unique, strictly increasing |
| `T_out_C` | °C | Finite; contract range −50 to 60°C |
| `I_north_W_m2` | W/m² | 0–1500 |
| `I_east_W_m2` | W/m² | 0–1500 |
| `I_south_W_m2` | W/m² | 0–1500 |
| `I_west_W_m2` | W/m² | 0–1500 |

The climate member and on-demand façade calculation must be joined before this interface is called. Horizontal irradiance may remain as an extra diagnostic column but is not consumed by the thermal solver.

### Schedule schema

| Column | Unit | Rule |
|---|---|---|
| `timestamp_utc` | UTC | Must equal the weather timestamps exactly |
| `Phi_int_W` | W | Finite and non-negative |
| `theta_set_heat_C` | °C | 5–40°C |
| `theta_set_cool_C` | °C | 5–40°C and never below the heating setpoint |

The Gate-4 behavioural wrapper uses a documented periodic fixed-CET (UTC+1)
clock without daylight-saving transitions, then converts its output to the
canonical UTC index. Any alternative wrapper may use local civil time only if
it resolves daylight-saving transitions explicitly. The solver never guesses
the timezone of naive timestamps.

Both weather and schedules must contain one complete UTC calendar year: 8,760 hours for a non-leap year or 8,784 for a leap year. February 29 is never dropped silently.

### Hourly result schema

`SimulationResult.hourly` requires:

| Column | Unit |
|---|---|
| `timestamp_utc` | UTC |
| `T_out_C` | °C |
| `theta_air_C`, `theta_surface_C`, `theta_mass_C`, `theta_operative_C` | °C |
| `Phi_internal_W`, `Phi_solar_W` | W |
| `heating_demand_W`, `cooling_demand_W` | W, separately non-negative |
| `theta_set_heat_C`, `theta_set_cool_C` | °C |
| `H_ve_W_K` | W/K |
| `hrv_bypass_active` | boolean |

The validator checks that forcing and schedules are reproduced in the result, heating and cooling are not simultaneous, and:

$$
\theta_{operative}=0.3\theta_{air}+0.7\theta_{surface}
$$

### Annual diagnostics schema

`SimulationDiagnostics` requires:

- Archetype, state, weather-member, occupant-seed, and model-scenario identifiers.
- Assumptions-file SHA-256.
- Annual heating and cooling energy in kWh.
- Heating and cooling intensity in kWh/m².
- Peak heating and cooling demand in W.
- Maximum absolute energy-balance residual in W.
- Number of warm-up cycles.

Annual energy, intensity, and peak values are recomputed from the hourly result and must reconcile with the diagnostics.

## Automated contract checks

The current tests verify:

- All 69 assumptions have exactly one implementation binding.
- Numeric assumptions have recognized units and valid sensitivity bounds.
- Local source paths in the assumptions file exist.
- All 225 rows of the current 2050 physical-state matrix join to their base
  geometry and satisfy the combined schema.
- Oriented windows reconcile with total window area.
- Envelope areas reconcile within a 0.25 m² source-rounding tolerance.
- Protected volume per floor area remains within a broad 1.5–5.0 m plausibility range.
- Normal infiltration flow equals `q50 / infiltration_n_factor` and ACH equals `n50 / infiltration_n_factor`.
- Ventilation system, HRV, and bypass combinations are valid.
- Weather and schedules are complete, finite, bounded, hourly, timezone-aware, and exactly aligned.
- Heating setpoints never exceed cooling setpoints.
- Prepared 5R1C conductances are physically constructible.
- Hourly results and annual diagnostics reconcile.

Any failed check raises `ContractError`; inputs are not silently clipped, imputed, or reordered.

Gate-level benchmark failures are reported separately as `ValidationError`, `FAIL` or `REVIEW_REQUIRED`; target disagreement is never silently converted into a calibration adjustment.

## Gate-4 behavioural boundary

The optional wrapper in `thermal_model/behaviour/` owns occupancy, appliance
and lighting profiles, the conversion to sensible internal heat, and thermostat
schedules. Only `timestamp_utc`, `Phi_int_W`, `theta_set_heat_C` and
`theta_set_cool_C` cross into `SimulationInput.schedules`. The thermal core has
no RichardsonPy import and does not inspect occupant or electricity fields.

The exact behavioural assumptions, occupant distribution, seed split, clock
convention, aggregation equations, validation checks and deterministic coupling
results are documented in `BEHAVIOURAL_WRAPPER.md`.
