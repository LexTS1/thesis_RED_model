# Gate 3: verification and Belgian validation of the residential 5R1C model

## 1. Purpose and decision rule

Gate 3 keeps **verification** and **validation** separate:

- Verification asks whether the model implements its declared equations, schemas, physical branches, controller and numerical method correctly.
- Validation asks whether the resulting useful heating and sensible-cooling demand is credible for Belgian residential archetypes.

A verified model can still validate poorly, and agreement with a benchmark cannot prove that the equations were implemented correctly. The generated summary therefore carries independent `verification_status` and `validation_status` fields.

Gate 3 is passed only when:

1. All hard numerical checks pass.
2. At least 80% of the 75 model-to-TABULA cells fall inside the predeclared comparison band.
3. Every qualitative stock-pattern rate is at least 80%.
4. All predeclared sensitivity direction checks pass.

The command-line runner returns a non-zero exit status if any one of these conditions fails.

## 2. Reproduction

Run all thermal-model tests:

```bash
python3 -m pytest thermal_model/tests -q
```

Regenerate the deterministic comparison, summary, sensitivity screen and report:

```bash
python3 -m thermal_model.validation
```

The runner writes only deterministic, versionable artifacts under `thermal_model/data/validation/`. It does not modify the source archetype matrices or `thermal_assumptions.csv`.

## 3. Frozen inputs

### 3.1 Physical scope

The validation matrix is the Cartesian product of:

- 25 Belgian TABULA archetypes: five dwelling types and five construction periods.
- Three physical states: `TABULA_existing`, `TABULA_standard_B_proxy` and `TABULA_advanced_A_proxy`.

The renovation scenario file contains 225 regional rows. `load_unique_archetype_states()` verifies that regional copies of each physical archetype/state are identical and reduces them to exactly 75 unique physics combinations. Any regional disagreement raises `ValidationError`.

### 3.2 Reference weather

The reference-year rule was frozen before inspecting archetype results:

1. Read the complete 2006–2023 PVGIS HDD comparison.
2. Calculate the median PVGIS heating degree-days.
3. Select the complete year with the smallest absolute distance to that median.
4. Break an exact tie in favour of the earlier year.

This selects 2015:

| Quantity | Value |
|---|---:|
| Rows | 8,760 |
| Mean outdoor temperature | 10.978 °C |
| HDD | 2,636.580 °C·day |
| 2006–2023 median HDD | 2,570.980 °C·day |
| Distance to median | 65.600 °C·day |

The climate layer's already-transposed façade series are joined without retransposition:

| Façade | Annual irradiance |
|---|---:|
| North | 303.844 kWh/m² |
| East | 709.420 kWh/m² |
| South | 976.054 kWh/m² |
| West | 689.680 kWh/m² |

The UTC timestamps of temperature and all four façade series must be exactly equal. Missing hours, duplicate timestamps, negative irradiance or any other schema failure stops the run.

### 3.3 Deterministic operation

Every archetype uses:

| Input | Frozen value |
|---|---:|
| Heating setpoint | 20 °C |
| Cooling setpoint | 26 °C |
| Total sensible internal gains | 3 W/m², constant |
| Occupant seed | 0, identifier only; stochastic behaviour is not invoked |
| Thermal, ventilation, solar and boundary assumptions | Central values in `thermal_assumptions.csv` |

These conditions isolate the thermal model. Occupancy and electricity profiles enter only after Gate 3.

### 3.4 TABULA targets and provenance

The direct targets are the **net energy demand for space heating** in Annex 2, Table 20 of the VITO report *Belgian Building Typologies – National Scientific Report*:

- Source: `BE_building_stock/data/inputs/physical/BE_TABULA_ScientificReport_VITO.pdf`.
- Locator: printed pages 87–88, PDF pages 98–99.
- Target mapping: Current → `TABULA_existing`; EPB2010 → `TABULA_standard_B_proxy`; LE → `TABULA_advanced_A_proxy`.
- Source and transcription checksums are stored in `thermal_model/data/reference/tabula_net_heating_demand.provenance.json`.

The 25-by-3 transcription was extracted programmatically and checked against rendered page images. The loader requires all 25 archetype identifiers, type numbers 1–25, all three positive targets and exactly 75 unique long-form cells.

## 4. Input and schema verification

The executable contract in `contracts.py` applies fail-fast validation before preprocessing or simulation.

### 4.1 Archetype and state checks

- Conditioned floor area, protected volume and total envelope area must be finite and positive.
- Individual component areas are non-negative because a physically absent roof, ground floor, door or wall has area zero. All present elements therefore have positive area.
- All U-values are finite and positive.
- Oriented window areas must sum to `windows_total_m2` within $10^{-6}$ m².
- The exposed envelope components must reconcile with the reported total envelope within the documented 0.25 m² TABULA source-rounding tolerance.
- Infiltration flow and ACH must be non-negative and reconcile with the stock layer's $q_{50}$/n-factor conversion.
- Ventilation ACH must be non-negative.
- HRV efficiency must lie in $[0,1]$ and must be consistent with the ventilation-system class.
- $A_m<A_t$ and $0<H_{tr,op}<H_{tr,ms}$ are required so that the 5R1C network is physically constructible.

No invalid field is clipped, imputed, reordered or replaced with a default.

### 4.2 Hourly checks

- Weather and schedule tables contain exactly 8,760 or 8,784 continuous UTC hours covering one calendar year.
- Timestamps are timezone-aware, unique, strictly increasing and exactly aligned across all inputs.
- Irradiance and total sensible internal gains are finite and non-negative.
- Heating and cooling setpoints remain inside the contract bounds.
- The heating setpoint never exceeds the cooling setpoint.

## 5. Equation-level verification

The physical branch tests are implemented in `tests/test_core.py` and `tests/test_verification.py`.

| Requirement | Executable check |
|---|---|
| Zero temperature difference | With all nodes and outdoor air at 20 °C and zero gains, all temperatures remain 20 °C and node residuals close to floating-point tolerance. |
| Zero window area | Solar gain is exactly zero for non-zero façade irradiance. |
| Zero irradiance | Solar gain is exactly zero for non-zero window areas. |
| HRV scope | Changing recovery changes the mechanical-ventilation term but leaves infiltration unchanged. |
| Perfect HRV | $\eta_{HRV}=1$ removes only recoverable ventilation loss; infiltration remains. |
| Summer bypass | Bypass activates only when nominal free-running air temperature is above the cooling setpoint and outdoor air is cooler. Hot outdoor air retains coolth recovery. |
| Opaque branch | Removing the door changes $H_{tr,op}$ by exactly $A_{door}U_{door}$ for its exterior boundary. |
| Gain allocation | The exact ISO allocation identity, including the window correction, is verified. |
| Mode exclusivity | Public heating and cooling series are non-negative and never positive simultaneously. |
| Orientation mapping | An N/E/S/W-only window responds only to its corresponding façade column. |

### 5.1 Important gain-allocation identity

The three ISO node terms should not be required to satisfy a naïve sum because the surface allocation explicitly subtracts a window correction. With:

$$
X=0.5\Phi_{int}+\Phi_{sol}
$$

the implemented identity is:

$$
\Phi_{int}+\Phi_{sol}-\Phi_{ia}-\Phi_m-\Phi_{st}
=
\frac{H_{tr,w}}{9.1A_t}X
$$

Testing the naïve sum would reject the standard's stated 5R1C allocation rather than detect a coding error.

## 6. Analytical thermal tests

### 6.1 Steady-state demand

For constant outdoor temperature, no gains and an air-node setpoint, the simulation converges to the exact 5R1C network-equivalent demand:

$$
\Phi_H=(H_{ve}+H_{tr,eq})(\theta_{set,H}-\theta_e)
$$

where:

$$
H_{tr,eq}
=
\left[
\frac{1}{H_{tr,is}}
+
\frac{1}{H_{tr,op}+H_{tr,w}}
\right]^{-1}
$$

This is the precise meaning of $H_{tr}$ when indoor air is the controlled node. Directly adding $H_{tr,op}+H_{tr,w}$ would incorrectly treat the finite air-to-surface resistance as zero. The test checks all 8,760 hourly loads and annual energy against this oracle.

### 6.2 Free decay

With constant outdoor conditions and no gains or HVAC, the one-state reduced system has the Crank–Nicolson decay ratio:

$$
r
=
\frac{C_m/\Delta t-0.5(H_3+H_{tr,em})}
{C_m/\Delta t+0.5(H_3+H_{tr,em})}
$$

and:

$$
\theta_{m,t}-\theta_e
=
r^t(\theta_{m,0}-\theta_e)
$$

The 240-hour numerical trajectory is monotonic and matches that closed form.

### 6.3 Adiabatic energy conservation

With $H_{tr,w}=H_{tr,em}=H_{ve}=0$, prescribed gains cannot leave the zone. The test verifies every hour that:

$$
\frac{C_m(\theta_{m,t}-\theta_{m,t-1})}{\Delta t}
=
\Phi_{int}
$$

This synthetic edge case bypasses the public prepared-archetype validator deliberately because a fully adiabatic envelope is not a valid Belgian archetype.

### 6.4 Constant internal gains

A separate static two-equation surface/mass oracle is solved with `numpy.linalg.solve`. The hourly solver matches it for 0, 500 and 1,000 W of internal gain, and equal gain increments cause equal reductions in ideal heating while heating remains required.

The reduction need not equal the gain watt-for-watt when the air node is controlled: half of the internal gain enters the surface/mass system and can alter surface temperature and envelope loss. A forced one-to-one assertion would contradict the declared multi-node model.

### 6.5 Solar orientation

Four parameterized tests give the dwelling glazing on only one orientation. Irradiance in that orientation produces the analytically expected gain; irradiance in every other orientation produces exactly zero gain.

## 7. Numerical acceptance tests

The tolerances were fixed before the 75-cell result matrix was examined.

| Check | Acceptance | Evidence |
|---|---:|---:|
| Maximum air/surface/mass balance residual over every solved timestep | $\le10^{-6}$ W | $1.426\times10^{-10}$ W |
| Controlled heating/cooling setpoint error | $\le10^{-8}$ K | $5.329\times10^{-14}$ K |
| Simultaneous heating and cooling | Never | None observed |
| Periodic warm-up endpoint mismatch | $\le0.01$ K within 10 cycles | All 75 converged |
| Reproducibility | Bit-identical hourly frame and diagnostics | Pass |
| Arbitrary initial mass temperature | 5 °C and 35 °C converge to the same reported year within $10^{-8}$ | Pass |
| Insulated/leaky stability | Finite outputs and residual $<10^{-6}$ W for both extremes | Pass |
| Heating/cooling transition | Both modes occur, never overlap, and track their setpoints | Pass |

The stored annual residual is the maximum of all three node residuals over every final-year hour. It is not an annual-average closure that could hide compensating hourly errors.

## 8. Deterministic Belgian validation

### 8.1 Predeclared comparison band

Each cell is flagged as credible when:

$$
|Q_{model}-Q_{TABULA}|
\le
\max(15\ \mathrm{kWh/m^2yr},\ 0.30Q_{TABULA})
$$

The absolute floor avoids unstable percentage judgments for low-energy targets. The gate passes when at least 80% of cells fall inside this band. This is an engineering acceptance rule, not a statistical confidence interval.

### 8.2 Direct results

| Metric | Result |
|---|---:|
| Cells within band | 68/75 (90.7%) |
| Mean signed deviation | −18.61 kWh/m²·yr |
| Mean absolute deviation | 18.61 kWh/m²·yr |
| RMSE | 20.47 kWh/m²·yr |
| Median absolute relative deviation | 21.10% |

| Physical state | Model median | TABULA median | Mean signed deviation | Cell pass rate |
|---|---:|---:|---:|---:|
| Existing | 144.3 | 165.0 | −26.6 | 96% |
| Standard/EPB2010 proxy | 76.6 | 92.0 | −17.8 | 80% |
| Advanced/LE proxy | 23.7 | 36.0 | −11.5 | 96% |

The model has a consistent negative bias relative to TABULA, but it passes the frozen gate without calibration.

### 8.3 Qualitative stock patterns

| Pattern | Result |
|---|---:|
| Advanced ≤ standard ≤ existing for each archetype | 25/25 (100%) |
| Exposed apartment ≥ enclosed apartment | 15/15 (100%) |
| Detached ≥ semi-detached ≥ terraced | 15/15 (100%) |
| Adjacent old-to-new pairs non-increasing | 48/60 (80%) |

The age test compares adjacent periods directly. A correlation was rejected because it is undefined for identical package results and can hide a local reversal. Equality passes because it does not contradict a non-increasing pattern. The twelve small reversals occur in standard or advanced packages where thermal specifications are deliberately period-independent and geometry differences, rather than construction age, can control demand; the largest reversal is 4.30 kWh/m²·yr.

### 8.4 Investigation of the seven outside-band cells

All seven are underpredictions:

- Six are enclosed-apartment cells. These archetypes use the same 100.1 m² floor area and the same small exposed envelope: 17.9 m² exterior wall and 26.8 m² window, without exposed roof, floor or door. Repeated state packages correctly produce identical central results. The concentration therefore indicates a class-level method or boundary discrepancy, not random numerical behavior.
- The remaining cell is the post-2005 detached-house advanced package: 27.5 kWh/m²·yr against a 45 kWh/m²·yr TABULA target.

Plausible systematic differences include deliberately omitted thermal bridges, fixed unheated-space and soil corrections, the selected 2015 weather, constant 3 W/m² gains, fixed ventilation and solar assumptions, and the hourly air-temperature controller versus the TABULA reference method. These factors are disclosed and sensitivity-tested; none was altered after examining the seven cells.

### 8.5 External Belgian context

The Belgian climate-neutral scenario uses 85, 64 and 25 kWh/m²·yr as shallow, medium and deep renovation-depth levers. The model medians of 76.6 and 23.7 kWh/m²·yr for standard and advanced states have a credible scale relative to the medium and deep levers. These scenario values are not archetype-specific validation targets.

The National Bank of Belgium reports average Q1 2024 EPC scores for sold houses of 363–390 kWh/m² and apartments of 198–245 kWh/m² across the regions. They are retained as contextual bounds only: EPC is standardized **primary energy** including technical systems, whereas the 5R1C output is **useful space-heating demand**. Direct numerical acceptance against EPC would compare different metrics and populations.

## 9. Calibration discipline

No archetype-specific parameter was tuned to a TABULA cell. Validation targets remain independent evidence.

If future work identifies a systematic bias:

1. Audit geometry, boundary classification, weather, gains and ventilation first.
2. Change only a parameter with genuine physical uncertainty and independent support.
3. Preserve the original value, source and result.
4. Apply the change consistently to a documented class, never to a single target cell.
5. Revalidate on archetypes that did not motivate the change.
6. Regenerate checksums, deterministic results and the comparison report.

Monte Carlo variability must not be used to conceal a deterministic bias.

## 10. Sensitivity screening

The screen is deterministic and one-at-a-time. It ranks model influence; its endpoints are not probability distributions and do not automatically become Monte Carlo inputs.

The central representative is the exposed `BE_TABULA_11` detached 1971–1990 existing dwelling. The HRV comparison uses its advanced state because the existing state has no HRV.

| Axis | Screened cases | Heating change from relevant baseline |
|---|---|---:|
| Heating setpoint | 18 and 22 °C | −44.21, +47.77 kWh/m²·yr |
| Infiltration | 0.5× and 1.5× central | −26.56, +26.64 |
| Solar gains | all façade irradiance set to zero | +21.86 |
| HRV | advanced state $\eta=0.8$ versus 0 | +21.06 when disabled |
| Boundary treatment | all opaque boundaries treated as exterior | +19.82 |
| Ventilation | 0.3 and 0.6 h⁻¹ | −7.43, +14.88 |
| Fixed shading | $F_{sh}=0.6$ central versus 1.0 | −13.15 when unshaded |
| Internal gains | 1.5 and 4.5 W/m² | +10.42, −10.25 |
| Frame fraction | 0.3 central versus 0.2 | −2.93 |
| Thermal mass | light and heavy classes | +1.18, −1.07 |
| Cooling setpoint | 24 and 28 °C | cooling +1.94, −0.73 |

All directional checks pass:

- More admitted solar does not increase heating and does not reduce cooling.
- HRV reduces heating.
- A higher heating setpoint increases heating.
- A higher cooling setpoint reduces cooling.

For this representative case, heating setpoint, infiltration, solar/HRV treatment and opaque-boundary treatment are the dominant screened assumptions. Thermal mass changes annual energy much less, although it still changes hourly temperatures and peaks.

## 11. Artifacts and audit trail

| Artifact | Role |
|---|---|
| `validation.py` | Deterministic runner, reference-year selection, stock-pattern checks, sensitivity screen and report generator |
| `tests/test_contracts.py` | Input/output contract tests |
| `tests/test_core.py` | Numerical oracles and core solver tests |
| `tests/test_verification.py` | Gate 3 branch, analytical and acceptance tests |
| `tests/test_validation.py` | Benchmark/source provenance, weather, scope and persisted-artifact tests |
| `data/reference/tabula_net_heating_demand.csv` | 25-by-3 TABULA target transcription |
| `data/reference/tabula_net_heating_demand.provenance.json` | Source locator, mapping and checksums |
| `data/validation/deterministic_archetype_validation.csv` | All 75 model/target comparisons and numerical diagnostics |
| `data/validation/validation_summary.json` | Machine-readable thresholds, status, provenance and aggregate metrics |
| `data/validation/sensitivity_results.csv` | All 19 one-at-a-time cases |
| `data/validation/sensitivity_summary.json` | Direction checks and influence ranking |
| `data/validation/validation_report.md` | Reproducibly generated human-readable result report |
| `plot_figures.py` | Reproducible standalone PNG/PDF figure generator |
| `figures/` | Six individually citable Gate 3 figures, captions and checksum provenance |

The generated summary stores SHA-256 checksums for the assumptions file, base archetype matrix, physical-state matrix, TABULA target table and weather sources. A later source change is therefore distinguishable from a solver-code change.

## 12. Gate conclusion and remaining limitations

Gate 3 passes. This establishes that the deterministic solver is internally verified and sufficiently credible, at the declared archetype resolution, to proceed to the behavioural wrapper and later Monte Carlo coupling.

It does not establish measured-building accuracy. The benchmark shares TABULA archetype geometry and package definitions with the model, while measured Belgian hourly useful-demand data are not available here. The result is therefore a method-level archetype validation, strengthened by independent equation tests and external scale checks, not empirical calibration of individual homes.

The seven flagged cells and the overall negative bias remain visible in the artifacts. They should be carried into interpretation and uncertainty discussions rather than removed through hidden calibration.
