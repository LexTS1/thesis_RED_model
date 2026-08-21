# Gate 3 deterministic thermal-model validation report

**Verification status: PASS**
**Validation status: PASS**

## Frozen design

- Reference weather: observed PVGIS 2015, selected before simulation as the complete year nearest the 2006-2023 median HDD.
- Operating conditions: 20 degC heating, 26 degC cooling and constant 3 W/m2 sensible internal gains.
- TABULA warning band: `max(15 kWh/m2/year, 30% of target)`.
- Gate-level target: at least 80% of cells inside that band.
- No parameter was fitted to a TABULA result.

## Numerical verification on the real-archetype runs

- Maximum node-balance residual: 1.426e-10 W (limit 1.0e-06 W).
- Maximum controlled-setpoint error: 5.329e-14 K (limit 1.0e-08 K).
- Simultaneous heating/cooling absent: True.

## Direct TABULA comparison

- Cells inside the predeclared band: 68/75 (90.7%).
- Mean signed deviation: -18.61 kWh/m2/year.
- Mean absolute deviation: 18.61 kWh/m2/year.
- RMSE: 20.47 kWh/m2/year.

| State | Model median | TABULA median | Mean signed deviation | Pass rate |
|---|---:|---:|---:|---:|
| `TABULA_existing` | 144.3 | 165.0 | -26.6 | 96.0% |
| `TABULA_standard_B_proxy` | 76.6 | 92.0 | -17.8 | 80.0% |
| `TABULA_advanced_A_proxy` | 23.7 | 36.0 | -11.5 | 96.0% |

## Qualitative stock patterns

- `renovation_order`: 25/25 (100.0%).
- `exposed_apartment_above_enclosed`: 15/15 (100.0%).
- `detached_above_semi_above_terraced`: 15/15 (100.0%).
- `newer_period_nonincreasing_adjacent_pairs`: 48/60 (80.0%).

## Cells requiring investigation

| Archetype | State | Model | TABULA | Deviation |
|---|---|---:|---:|---:|
| BE_TABULA_04 | `TABULA_standard_B_proxy` | 36.4 | 62.0 | -25.6 |
| BE_TABULA_09 | `TABULA_standard_B_proxy` | 36.4 | 62.0 | -25.6 |
| BE_TABULA_14 | `TABULA_standard_B_proxy` | 36.4 | 62.0 | -25.6 |
| BE_TABULA_19 | `TABULA_standard_B_proxy` | 36.4 | 62.0 | -25.6 |
| BE_TABULA_24 | `TABULA_existing` | 36.4 | 60.0 | -23.6 |
| BE_TABULA_24 | `TABULA_standard_B_proxy` | 36.4 | 60.0 | -23.6 |
| BE_TABULA_21 | `TABULA_advanced_A_proxy` | 27.5 | 45.0 | -17.5 |

### Investigation outcome

- All outside-band cells are underpredictions: yes.
- 6/7 outside-band cells are enclosed apartments. These archetypes share the same small exposed envelope (17.9 m2 exterior wall and 26.8 m2 windows, with no exposed roof, floor or door), and repeated package states therefore produce the same deterministic demand. The concentration is a class-level method/boundary discrepancy, not numerical scatter.
- The remaining outside-band cell is the post-2005 detached-house advanced package. Its direction is consistent with the overall negative model bias.
- Plausible method differences include the deliberately omitted thermal bridges, fixed unheated-space and ground reductions, the selected 2015 weather, constant 3 W/m2 gains, and the hourly air-temperature method versus TABULA's reference calculation. None was adjusted after inspecting the results.

## External context

The Belgian climate-neutral scenario uses 85, 64 and 25 kWh/m2/year as shallow, medium and deep renovation-depth levers. These are contextual scenario levels rather than archetype targets. Regional EPC figures are also contextual only because they are primary-energy scores under regional certificate methods, while this model reports useful space-heating demand.

## Sensitivity screening

The one-at-a-time screen contains 19 cases for BE_TABULA_11 detached 1971-1990 existing; HRV pair uses its advanced state. It ranks influence; it does not assign probability distributions.

| Case | Axis | Heating change | Cooling change |
|---|---|---:|---:|
| `heating_setpoint_22` | heating_setpoint | +47.77 | +0.09 |
| `heating_setpoint_18` | heating_setpoint | -44.21 | -0.03 |
| `infiltration_one_and_half` | infiltration | +26.64 | +0.04 |
| `infiltration_half` | infiltration | -26.56 | -0.02 |
| `solar_disabled` | solar_gain_check | +21.86 | -0.76 |
| `advanced_hrv_disabled` | hrv | +21.06 | -0.96 |
| `all_opaque_boundaries_exterior` | boundary_treatment | +19.82 | +0.01 |
| `ventilation_ach_0_6` | ventilation_rate | +14.88 | +0.02 |
| `shading_unshaded` | fixed_shading | -13.15 | +1.11 |
| `internal_gains_1_5` | internal_gains | +10.42 | -0.23 |
| `internal_gains_4_5` | internal_gains | -10.25 | +0.29 |
| `ventilation_ach_0_3` | ventilation_rate | -7.43 | -0.01 |
| `frame_fraction_0_2` | window_frame_fraction | -2.93 | +0.20 |
| `mass_light` | thermal_mass | +1.18 | +0.48 |
| `mass_heavy` | thermal_mass | -1.07 | -0.36 |
| `cooling_setpoint_24` | cooling_setpoint | +0.09 | +1.94 |
| `cooling_setpoint_28` | cooling_setpoint | -0.01 | -0.73 |
| `advanced_hrv_central` | hrv | +0.00 | +0.00 |

## Calibration discipline

No archetype-specific parameter was tuned. Any future correction must identify a physical source of systematic bias, preserve the original assumption, apply consistently to a documented dwelling class, and be checked on archetypes not used to motivate the change.
