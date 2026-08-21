# Deterministic residential 5R1C thermal core

## 1. Purpose and scope

`thermal_model/core.py` implements the deterministic single-zone ISO 13790:2008 simple hourly 5R1C method used by the residential demand model. It converts a validated Belgian TABULA archetype and renovation state into thermal parameters and calculates the ideal hourly sensible heating or cooling power required to maintain prescribed indoor air-temperature bounds.

The core calculates **useful thermal demand**. It does not calculate fuel use, electricity use, heat-pump COP, boiler efficiency, emitter capacity, distribution losses, cycling, humidity, latent cooling, domestic hot water, or HVAC auxiliary electricity.

The implementation uses:

- `thermal_assumptions.csv` for sourced constants and declared model choices.
- `contracts.py` for the input/output schemas and fail-fast validation.
- `core.py` for preprocessing and the deterministic hourly equations.
- `tests/test_core.py` for physical and numerical verification.

The governing standard has been superseded by ISO 52016-1. ISO 13790 is retained deliberately because its 5R1C network is transparent, computationally light, and directly suited to archetype-level Monte Carlo simulation.

## 2. Model topology and sign convention

The 5R1C network contains five thermal resistances and one capacitance:

1. Ventilation/infiltration conductance, $H_{ve}$, between outdoor/supply air and indoor air.
2. Air-to-surface conductance, $H_{tr,is}$.
3. Window conductance, $H_{tr,w}$, between outdoor air and the internal surface node.
4. Surface-to-mass conductance, $H_{tr,ms}$.
5. Equivalent opaque exterior-to-mass conductance, $H_{tr,em}$.
6. Effective thermal capacitance, $C_m$, at the mass node.

Heating is positive and cooling is negative inside the equations:

$$
\Phi_{HC}>0 \quad \text{heating},
\qquad
\Phi_{HC}<0 \quad \text{cooling}
$$

Public results expose separate non-negative series:

$$
\Phi_H=\max(\Phi_{HC},0)
$$

$$
\Phi_C=\max(-\Phi_{HC},0)
$$

Heating and cooling therefore cannot occur simultaneously.

## 3. Variable dictionary

### 3.1 Geometry and envelope inputs

| Symbol | Code field | Unit | Meaning |
|---|---|---:|---|
| $A_f$ | `floor_area_m2` | m² | Conditioned floor area |
| $V_{zone}$ | `zone_volume_m3` | m³ | Protected/conditioned air volume |
| $A_{window,o}$ | `window_area_<orientation>_m2` | m² | Gross window opening area for orientation $o$ |
| $A_{window}$ | sum of oriented window areas | m² | Total gross window area |
| $A_i$ | source archetype area fields | m² | Area of opaque component $i$ |
| $U_i$ | source archetype U-value fields | W/(m²·K) | Thermal transmittance of component $i$ |
| $R_{add,i}$ | boundary assumption | m²·K/W | Additional resistance for an adjacent unheated space |
| $b_{tr,i}$ | boundary assumption | 1 | Temperature adjustment factor for boundary $i$ |
| $A_t$ | `A_t_m2` | m² | Total effective internal surface area |
| $A_m$ | `A_m_m2` | m² | Effective thermal-mass area |
| $C_m$ | `C_m_J_K` | J/K | Effective thermal capacitance |

### 3.2 Conductances

| Symbol | Code field | Unit | Meaning |
|---|---|---:|---|
| $h_{is}$ | assumption `network.air_surface_coefficient` | W/(m²·K) | Heat-transfer coefficient from air to internal surfaces |
| $h_{ms}$ | assumption `network.mass_surface_coefficient` | W/(m²·K) | Heat-transfer coefficient from mass to surface |
| $H_{tr,w}$ | `H_tr_w_W_K` | W/K | Whole-window transmission conductance |
| $H_{tr,op}$ | `H_tr_op_W_K` | W/K | Boundary-corrected opaque transmission conductance |
| $H_{tr,is}$ | `H_tr_is_W_K` | W/K | Air-to-surface conductance |
| $H_{tr,ms}$ | `H_tr_ms_W_K` | W/K | Mass-to-surface conductance |
| $H_{tr,em}$ | `H_tr_em_W_K` | W/K | Equivalent outdoor-to-mass opaque conductance |
| $H_1$ | local solver variable | W/K | Equivalent series conductance of $H_{ve}$ and $H_{tr,is}$ |
| $H_2$ | local solver variable | W/K | $H_1+H_{tr,w}$ |
| $H_3$ | local solver variable | W/K | Equivalent series conductance of $H_2$ and $H_{tr,ms}$ |

### 3.3 Airflow and heat recovery

| Symbol | Code field | Unit | Meaning |
|---|---|---:|---|
| $\rho_{air}$ | `air_density_kg_m3` | kg/m³ | Air density |
| $c_{p,air}$ | `air_specific_heat_J_kgK` | J/(kg·K) | Specific heat of air |
| $\dot V_{inf}$ | `infiltration_airflow_m3_h` | m³/h | Normal-pressure infiltration airflow from the stock layer |
| $n_{vent}$ | `ventilation_ach_h_1` | h⁻¹ | Use-related/mechanical ventilation air-change rate |
| $\dot V_{vent}$ | `ventilation_airflow_m3_h` | m³/h | Ventilation airflow, $n_{vent}V_{zone}$ |
| $\eta_{HRV}$ | `hrv_efficiency` | 1 | Nominal sensible heat-recovery efficiency |
| $\eta_{HRV,eff}$ | `effective_hrv_efficiency` | 1 | Efficiency after the hourly bypass decision |
| $H_{ve}$ | `H_ve_W_K` | W/K | Effective infiltration and ventilation conductance |

### 3.4 Solar and internal gains

| Symbol | Code field | Unit | Meaning |
|---|---|---:|---|
| $I_o$ | `I_<orientation>_W_m2` | W/m² | Façade-transposed irradiance for orientation $o$ |
| $g_n$ | `glazing_g_value` | 1 | Normal-incidence total solar energy transmittance |
| $F_F$ | `window_frame_fraction` | 1 | Fraction of gross window area occupied by the frame |
| $F_W$ | `non_normal_irradiance_factor` | 1 | Reduction from normal-incidence to real-angle transmission |
| $F_{sh}$ | `vertical_shading_factor` | 1 | Fixed aggregate external shading factor |
| $\Phi_{sol,o}$ | orientation contribution | W | Transmitted solar gain from orientation $o$ |
| $\Phi_{sol}$ | `Phi_solar_W` | W | Total transmitted window solar gain |
| $\Phi_{int}$ | `Phi_int_W` | W | Total sensible internal gain supplied by the behavioural wrapper |
| $\Phi_{ia}$ | `Phi_ia_W` | W | Internal gains applied directly to the air node |
| $\Phi_m$ | `Phi_m_W` | W | Gains applied to the mass node |
| $\Phi_{st}$ | `Phi_st_W` | W | Gains applied to the surface node |
| $\Phi_{air}$ | local solver variable | W | $\Phi_{ia}+\Phi_{HC}$, total direct air-node input |

### 3.5 Temperatures, loads, and time

| Symbol | Code field | Unit | Meaning |
|---|---|---:|---|
| $\theta_e$ | `T_out_C` | °C | Outdoor dry-bulb temperature |
| $\theta_{sup}$ | set equal to `T_out_C` | °C | Effective ventilation supply boundary temperature |
| $\theta_{m,t-1}$ | `theta_mass_previous_C` | °C | Mass temperature at the end of the previous hour |
| $\theta_{m,t}$ | `theta_mass_end_C` | °C | Mass temperature at the end of the current hour |
| $\theta_m$ | `theta_mass_C` | °C | Mean mass temperature during the hour |
| $\theta_s$ | `theta_surface_C` | °C | Internal surface temperature |
| $\theta_{air}$ | `theta_air_C` | °C | Indoor air temperature and controlled variable |
| $\theta_{op}$ | `theta_operative_C` | °C | Operative-temperature diagnostic |
| $\theta_{set,H}$ | `theta_set_heat_C` | °C | Hourly heating setpoint |
| $\theta_{set,C}$ | `theta_set_cool_C` | °C | Hourly cooling setpoint |
| $\Phi_{HC}$ | `signed_hvac_load_W` | W | Signed ideal heating/cooling load at the air node |
| $\Phi_{HC,10}$ | test load | W | Signed ISO test load, $\pm10A_f$ |
| $\Delta t$ | `timestep_seconds` | s | Simulation timestep; fixed at 3,600 s |

### 3.6 Locally derived and diagnostic symbols

| Symbol | Unit | Meaning |
|---|---:|---|
| $\Phi_H$, $\Phi_C$ | W | Separate non-negative ideal heating and cooling demands |
| $X$ | W | Combined gain quantity $0.5\Phi_{int}+\Phi_{sol}$ distributed to mass and surface nodes |
| $S$ | W | Combined forcing acting on the internal surface node |
| $\Phi_{m,tot}$ | W | Total forcing used in the dynamic mass-node update |
| $\theta_{air,0}$ | °C | Free-running air temperature with zero HVAC load |
| $\theta_{air,10}$ | °C | Air temperature under the signed 10 W/m² ISO test load |
| $\theta_{set}$ | °C | Applicable heating or cooling setpoint selected by the controller |
| $r_{air}$, $r_s$, $r_m$ | W | Air-, surface-, and mass-node energy-balance residuals |
| $i$ | — | Opaque-envelope component index |
| $o$ | — | Window-orientation index |
| $k$ | — | Complete-year periodic warm-up cycle index |

## 4. Archetype preprocessing

### 4.1 Explicit geometry/state join

The base archetype matrix contains geometry, while the projection-specific
physical-state matrix contains the final U-values, leakage and ventilation
system. They are joined explicitly with:

```python
state = assemble_archetype_state(base_row, physical_state_row)
prepared = preprocess_archetype(state, assumptions)
```

Both rows must have the same `archetype_id`. The join is validated before any conductance is calculated.

The precomputed stock-layer value `transmission_heat_loss_H_tr_W_K` is deliberately **not carried into `PreparedArchetype` and is never consumed by the hourly core**. It is a screening value without the complete ground and unheated-space corrections. Comparisons with it may be performed externally as a diagnostic, but it cannot replace the component calculation below.

### 4.2 Transparent transmission

The whole-window U-value acts on gross window opening area:

$$
H_{tr,w}=U_{window}A_{window}
$$

The frame fraction is not applied here because the archetype U-value is already a whole-window value. $F_F$ reduces solar aperture only.

### 4.3 Corrected opaque transmission

For each opaque element:

$$
H_i=A_i\frac{b_{tr,i}}{1/U_i+R_{add,i}}
$$

and:

$$
H_{tr,op}=\sum_i H_i
$$

The following branches are calculated separately:

| Component | U-value | Boundary class | $R_{add}$ m²K/W | $b_{tr}$ |
|---|---|---|---:|---:|
| Exterior wall | $U_{facade}$ | External air | 0 | 1 |
| Roof | $U_{roof}$ | External air | 0 | 1 |
| Door | $U_{door}$ | External air | 0 | 1 |
| Wall bordering unheated space | $U_{facade}$ | Unheated room | 0.3 | 1 |
| Floor over unheated space/cellar | $U_{floor}$ | Cellar | 0.3 | 0.5 |
| Floor on soil | $U_{floor}$ | Soil | 0 | 0.5 |

Thermal bridges remain zero. Party elements are absent from exposed envelope areas and are treated as adiabatic.

### 4.4 Surface, mass, and equivalent conductances

Central surface and mass parameters are:

$$
A_t=4.5A_f
$$

$$
A_m=2.5A_f
$$

$$
C_m=165000A_f
$$

The node conductances are:

$$
H_{tr,is}=h_{is}A_t
$$

$$
H_{tr,ms}=h_{ms}A_m
$$

with $h_{is}=3.45\ \mathrm{W/(m^2K)}$ and $h_{ms}=9.1\ \mathrm{W/(m^2K)}$.

The opaque conductance is converted to the equivalent exterior-to-mass branch:

$$
H_{tr,em}
=
\left(
\frac{1}{H_{tr,op}}-\frac{1}{H_{tr,ms}}
\right)^{-1}
$$

Preprocessing rejects the archetype unless:

$$
0<H_{tr,op}<H_{tr,ms}
$$

It also checks $A_m<A_t$, verifies the reconstructed $H_{tr,em}$, and rejects any negative surface-gain allocation fraction.

### 4.5 Glazing mapping

The window U-value selects the documented glazing class:

| Whole-window U-value W/(m²K) | Glazing interpretation | $g_n$ |
|---:|---|---:|
| 5.0 | Single glazing | 0.85 |
| 3.5 | Conventional double glazing | 0.75 |
| 2.0 | High-performance low-e double glazing | 0.67 |
| 1.6 | Low Energy package with low-e double glazing | 0.67 |

An unrecognized U-value is rejected rather than silently assigned to the nearest class.

## 5. Ventilation and HRV bypass

Ventilation airflow is:

$$
\dot V_{vent}=n_{vent}V_{zone}
$$

The combined hourly conductance is:

$$
H_{ve}
=
\frac{\rho_{air}c_{p,air}}{3600}
\left[
\dot V_{inf}
+
(1-\eta_{HRV,eff})\dot V_{vent}
\right]
$$

The division by 3,600 converts m³/h to m³/s. Infiltration is outside the heat-recovery term and therefore never receives recovery.

Because heat recovery is already represented by the reduced $H_{ve}$, the solver sets:

$$
\theta_{sup}=\theta_e
$$

It does not also calculate a warmed supply temperature. Doing both would count heat recovery twice.

### 5.1 Bypass decision

Bypass is decided exactly once per hour:

1. Calculate the free-running solution with nominal HRV.
2. Activate bypass only if the nominal free-running air temperature exceeds the cooling setpoint and outdoor air is cooler than that free-running temperature:

   $$
   \theta_{air,0,nom}>\theta_{set,C}
   \quad\text{and}\quad
   \theta_e<\theta_{air,0,nom}
   $$

3. If active, set $\eta_{HRV,eff}=0$ and recompute the free-running solution.
4. Hold that bypass decision and $H_{ve}$ fixed for the free-running, test-load and final-load evaluations.

Freezing the bypass state is necessary because the ideal-load interpolation assumes a linear temperature response to HVAC power. Using the final controlled temperature to reconsider bypass would create a circular and discontinuous controller.

When outdoor air is hotter than indoor air, bypass remains closed so the system retains useful coolth recovery.

## 6. Solar gains

The climate layer supplies façade-transposed irradiance. The thermal core does not transpose it again.

For each vertical orientation $o\in\{N,E,S,W\}$:

$$
\Phi_{sol,o}
=
I_oA_{window,o}(1-F_F)g_nF_WF_{sh}
$$

Total solar gain is:

$$
\Phi_{sol}=\sum_o\Phi_{sol,o}
$$

The central factors are $F_F=0.3$, $F_W=0.9$, and $F_{sh}=0.6$. Each is applied once. Dynamic movable shading is disabled, opaque solar gains are zero, and explicit sky long-wave exchange is zero under the declared scope.

## 7. Gain allocation

The behavioural wrapper supplies total sensible internal gains $\Phi_{int}$. It must not perform its own convective/radiative split.

The ISO allocation is:

$$
\Phi_{ia}=0.5\Phi_{int}
$$

Define:

$$
X=0.5\Phi_{int}+\Phi_{sol}
$$

Then:

$$
\Phi_m=\frac{A_m}{A_t}X
$$

$$
\Phi_{st}
=
\left(
1-\frac{A_m}{A_t}
-\frac{H_{tr,w}}{h_{ms}A_t}
\right)X
$$

where $h_{ms}=9.1\ \mathrm{W/(m^2K)}$.

These terms should not be tested against the naïve identity $\Phi_{ia}+\Phi_m+\Phi_{st}=\Phi_{int}+\Phi_{sol}$. The standard surface expression deliberately includes the window correction:

$$
\Phi_{int}+\Phi_{sol}
-\Phi_{ia}-\Phi_m-\Phi_{st}
=
\frac{H_{tr,w}}{h_{ms}A_t}X
$$

The tests verify this exact identity.

## 8. Hourly 5R1C equations

For a fixed hourly ventilation state and signed HVAC load, the reciprocal notation for positive branches is:

$$
H_1
=
\left(\frac{1}{H_{ve}}+\frac{1}{H_{tr,is}}\right)^{-1}
$$

$$
H_2=H_1+H_{tr,w}
$$

$$
H_3
=
\left(\frac{1}{H_2}+\frac{1}{H_{tr,ms}}\right)^{-1}
$$

The direct air-node input is:

$$
\Phi_{air}=\Phi_{ia}+\Phi_{HC}
$$

The Annex C surface forcing is:

$$
S
=
\Phi_{st}
+H_{tr,w}\theta_e
+H_1\left(
\theta_{sup}+\frac{\Phi_{air}}{H_{ve}}
\right)
$$

and the mass-node forcing is:

$$
\Phi_{m,tot}
=
\Phi_m
+H_{tr,em}\theta_e
+\frac{H_3}{H_2}S
$$

The code evaluates algebraically equivalent continuous forms:

$$
H_1=\frac{H_{ve}H_{tr,is}}{H_{ve}+H_{tr,is}}
$$

$$
H_1\left(\theta_e+\frac{\Phi_{air}}{H_{ve}}\right)
=
H_1\theta_e
+\frac{H_{tr,is}}{H_{ve}+H_{tr,is}}\Phi_{air}
$$

$$
\frac{H_3}{H_2}=\frac{H_{tr,ms}}{H_{tr,ms}+H_2}
$$

These avoid numerical division by $H_{ve}$ or $H_2$. They reproduce the reciprocal expressions for positive branches and keep synthetic zero-airflow and zero-window-conductance edge cases well defined.

### 8.1 Crank–Nicolson mass update

The end-of-hour mass temperature is:

$$
\theta_{m,t}
=
\frac{
\theta_{m,t-1}
\left[
C_m/\Delta t
-0.5(H_3+H_{tr,em})
\right]
+\Phi_{m,tot}
}{
C_m/\Delta t
+0.5(H_3+H_{tr,em})
}
$$

The mean mass temperature reported for the hour is:

$$
\theta_m=\frac{\theta_{m,t-1}+\theta_{m,t}}{2}
$$

Only $\theta_{m,t}$ is propagated to the next hour.

### 8.2 Surface and air temperatures

The surface temperature is:

$$
\theta_s
=
\frac{
H_{tr,ms}\theta_m+S
}{
H_{tr,ms}+H_2
}
$$

The air temperature is:

$$
\theta_{air}
=
\frac{
H_{tr,is}\theta_s
+H_{ve}\theta_e
+\Phi_{ia}
+\Phi_{HC}
}{
H_{tr,is}+H_{ve}
}
$$

The operative-temperature diagnostic is:

$$
\theta_{op}=0.3\theta_{air}+0.7\theta_s
$$

The ideal controller acts on $\theta_{air}$, not $\theta_{op}$.

## 9. Ideal heating and cooling controller

Every hour uses the same previous mass state, gains, outdoor temperature and frozen ventilation/bypass state for all controller evaluations.

### 9.1 Free-running evaluation

Set:

$$
\Phi_{HC}=0
$$

and calculate $\theta_{air,0}$.

- If $\theta_{set,H}\le\theta_{air,0}\le\theta_{set,C}$, the free-running result is accepted.
- If $\theta_{air,0}<\theta_{set,H}$, heating is required.
- If $\theta_{air,0}>\theta_{set,C}$, cooling is required.

### 9.2 Test-load evaluation

The signed test load is:

$$
\Phi_{HC,10}=+10A_f
$$

for heating and:

$$
\Phi_{HC,10}=-10A_f
$$

for cooling.

This produces $\theta_{air,10}$.

### 9.3 Unrestricted ideal load

The required load is found by linear interpolation:

$$
\Phi_{HC}
=
\Phi_{HC,10}
\frac{
\theta_{set}-\theta_{air,0}
}{
\theta_{air,10}-\theta_{air,0}
}
$$

The load is not clipped when its magnitude exceeds the test load. A near-zero test response is a hard numerical error. The final evaluation must reproduce the selected setpoint within $10^{-8}$ K.

Only the mass end state from this final controlled evaluation is committed. The free-running and test-load states are discarded.

## 10. Initial conditions and periodic convergence

The first pass starts with:

$$
\theta_{m,initial}=20^\circ\mathrm{C}
$$

The complete input year is simulated. Its final mass temperature becomes the next cycle's initial mass temperature:

$$
\theta_{m,initial}^{(k+1)}=\theta_{m,end}^{(k)}
$$

Convergence is accepted when:

$$
\left|
\theta_{m,end}^{(k)}-\theta_{m,initial}^{(k)}
\right|
\le0.01\ \mathrm{K}
$$

The maximum is ten complete-year cycles. Failure raises `ThermalCoreError` with the final mismatch and tolerance. Only the final converged cycle appears in the result.

This periodic-year condition removes dependence on the arbitrary first-pass temperature without deleting hours or inventing a synthetic warm-up period.

## 11. Energy-balance diagnostic

Every prescribed-load node evaluation checks three balances.

### Air node

$$
r_{air}
=
\Phi_{ia}+\Phi_{HC}
+H_{ve}(\theta_e-\theta_{air})
+H_{tr,is}(\theta_s-\theta_{air})
$$

### Surface node

$$
r_s
=
\Phi_{st}
+H_{tr,ms}(\theta_m-\theta_s)
+H_{tr,w}(\theta_e-\theta_s)
+H_{tr,is}(\theta_{air}-\theta_s)
$$

### Mass node

$$
r_m
=
\frac{C_m(\theta_{m,t}-\theta_{m,t-1})}{\Delta t}
-\left[
\Phi_m
+H_{tr,em}(\theta_e-\theta_m)
+H_{tr,ms}(\theta_s-\theta_m)
\right]
$$

The annual diagnostic stores:

$$
\max_t\left(
|r_{air}|,
|r_s|,
|r_m|
\right)
$$

These are numerical closure checks, not empirical validation metrics.

## 12. Public simulation result

`simulate(request, assumptions)` returns:

### Hourly series

- UTC timestamp and outdoor temperature.
- Air, surface, mean mass, and operative temperatures.
- Internal and solar gains.
- Non-negative heating and cooling demand.
- Heating and cooling setpoints.
- Effective $H_{ve}$.
- HRV bypass state.

### Annual diagnostics

- Annual heating and cooling energy in kWh.
- Heating and cooling intensity in kWh/m².
- Peak heating and cooling power in W.
- Maximum absolute node-balance residual in W.
- Warm-up cycles.
- Archetype, state, weather member, occupant seed and model-scenario identifiers.
- SHA-256 checksum of the assumptions contract.

The result validator independently reconstructs annual energy, intensity and peaks from the hourly series.

## 13. Verification stack

The deterministic core currently passes the following checks.

### 13.1 Archetype preprocessing

- All 225 rows in the current 2050 physical-state matrix preprocess successfully.
- Every equivalent conductance is positive and reconstructs its defining equation.
- All four documented whole-window U-value classes map to the intended three g-values.
- Unsupported glazing classes and impossible $H_{tr,em}$ networks fail explicitly.

For `BE_TABULA_01` in the existing state, the independent regression oracle is:

| Parameter | Expected W/K |
|---|---:|
| $H_{tr,w}$ | 206.000000000 |
| $H_{tr,op}$ | 899.529437071 |
| $H_{tr,is}$ | 4331.475000000 |
| $H_{tr,ms}$ | 6347.250000000 |
| $H_{tr,em}$ | 1048.060037863 |

### 13.2 Annex C numerical oracle

For the fixed synthetic dwelling in `test_core.py`, with $\theta_e=5^\circ C$, $\theta_{m,t-1}=20^\circ C$, $\Phi_{int}=300\ W$, and no solar gains:

| Quantity | Expected |
|---|---:|
| $H_{ve}$ | 70.35000000000 W/K |
| Free-running $\theta_{air,0}$ | 18.53482140317 °C |
| $+1000\ W$ test $\theta_{air,10}$ | 19.62178089399 °C |
| Required ideal heating load | 1347.96062705309 W |
| Final $\theta_{air}$ | 20.00000000000 °C |
| Final $\theta_{m,t}$ | 19.63029241044 °C |
| Mean $\theta_m$ | 19.81514620522 °C |
| Final $\theta_s$ | 19.71484017581 °C |

The values are hard-coded in the tests rather than calculated with the production functions.

### 13.3 Analytical and invariant tests

- Free decay matches the closed-form Crank–Nicolson decay ratio for the reduced one-state system.
- Air, surface and mass balances close to floating-point tolerance.
- The ideal controller reaches heating and cooling setpoints.
- HRV changes controlled ventilation only; infiltration is invariant.
- Zero infiltration, zero ventilation and zero window conductance remain finite without a singularity.
- Bypass activates only under the documented free-cooling condition.
- Gross window area is used for transmission and glazed area for solar gains.
- The exact ISO gain-allocation identity, including the window correction, is satisfied.
- A constant periodic year converges to the correct network-equivalent steady heating load:

  $$
  \Phi_H
  =
  \left[
  H_{ve}
  +
  \left(
  \frac{1}{H_{tr,is}}
  +
  \frac{1}{H_{tr,op}+H_{tr,w}}
  \right)^{-1}
  \right]
  (\theta_{set,H}-\theta_e)
  $$

This network-equivalent expression is the correct air-setpoint steady-state oracle; simply adding all conductances would ignore the finite air-to-surface resistance.

## 14. Usage

```python
from thermal_model import (
    SimulationInput,
    assemble_archetype_state,
    load_assumption_contract,
    preprocess_archetype,
    simulate,
)

assumptions = load_assumption_contract()
state = assemble_archetype_state(base_row, physical_state_row)
archetype = preprocess_archetype(state, assumptions)

request = SimulationInput(
    archetype=archetype,
    weather=weather_with_facades,
    schedules=hourly_internal_gains_and_setpoints,
    weather_member_id="weather_2050_rcp_4_5_pvgis_2020",
    occupant_seed=42,
    model_scenario="central",
)

result = simulate(request, assumptions)
```

Both input DataFrames must first satisfy the schemas in `CONTRACT.md`.

## 15. Declared limitations

- One thermal zone represents the entire dwelling.
- Thermal bridges are zero.
- Ground and unheated-space effects use fixed TABULA corrections rather than dynamic adjacent temperatures.
- Infiltration uses the stock layer's rule-of-20 screening conversion.
- Ventilation is a fixed reference ACH, not an occupant window-opening model.
- The fixed shading factor does not resolve urban geometry or blinds.
- Dynamic movable shading, opaque solar gains and sky long-wave exchange are excluded.
- HRV frost protection, fans, modulation and bypass hysteresis are excluded.
- Cooling is sensible only.
- The controller has unlimited capacity and no emitter or plant dynamics.
- The periodic warm-up assumes the selected weather and behaviour year repeats.

These limitations must be revisited through deterministic validation and targeted sensitivity tests; they must not be hidden by Monte Carlo variability.

The completed follow-on evidence, including the 75-cell TABULA comparison, predeclared thresholds, outlier investigation and one-at-a-time sensitivity results, is documented in `GATE3_VERIFICATION_VALIDATION.md` and persisted under `data/validation/`.
