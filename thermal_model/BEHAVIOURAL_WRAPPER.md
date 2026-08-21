# Gate 4: reproducible occupant-behaviour coupling

## 1. Purpose and boundary

Gate 4 adds stochastic occupant behaviour without adding behavioural logic to
the deterministic ISO 13790 5R1C solver. The implementation is an optional
subpackage:

```text
thermal_model/
├── core.py                         deterministic 5R1C physics
├── behaviour/
│   ├── contracts.py                behavioural schemas and fail-fast checks
│   ├── wrapper.py                  RichardsonPy adapter
│   ├── coupling.py                 deterministic Gate-4 audit only
│   ├── behaviour_assumptions.csv   frozen values and sources
│   └── data/reference/             occupant distribution and provenance
└── data/behaviour/                 generated, reproducible audit artifacts
```

This is preferable to a top-level `stochastic_wrapper/` folder because the
code produces boundary conditions for this thermal model. It is nevertheless
isolated from `core.py`, and RichardsonPy is imported only when a profile is
generated. The future Monte Carlo driver belongs outside this package: it will
select weather members and seeds, then call this wrapper and the unchanged
thermal core.

Only four columns cross the boundary into `SimulationInput.schedules`:

| Variable | Column | Unit | Meaning |
|---|---|---:|---|
| $t$ | `timestamp_utc` | UTC | Complete, ordered hourly index |
| $\Phi_{int}(t)$ | `Phi_int_W` | W | Total sensible internal heat gain |
| $\theta_{set,H}(t)$ | `theta_set_heat_C` | °C | Heating setpoint |
| $\theta_{set,C}(t)$ | `theta_set_cool_C` | °C | Cooling setpoint |

The core does not receive occupancy, appliance, lighting or activity states.
It therefore cannot depend on the method used to construct the four-column
schedule. It still performs the ISO air/mass/surface gain allocation documented
in `THERMAL_CORE.md`; the wrapper applies no second radiant/convective split.

## 2. Reproduction

Install the optional, pinned behavioural dependency:

```bash
python3 -m pip install -r thermal_model/behaviour/requirements.txt
```

Regenerate the fixed-seed profiles, 75-cell comparison and two-archetype effect
decomposition:

```bash
python3 -m thermal_model.behaviour.coupling
```

Run the complete thermal-model test suite:

```bash
python3 -m pytest thermal_model/tests -q
```

The central Gate-4 seed is `20250805`. The reference weather rule remains the
Gate-3 rule and selects 2015. Neither value is inferred from the output.

## 3. Inputs and behavioural contract

### 3.1 Request

`BehaviourRequest` requires:

| Field | Type | Rule |
|---|---|---|
| `dwelling_type` | string | Exact stock-matrix dwelling type |
| `weather` | DataFrame | Complete 8,760- or 8,784-hour climate frame |
| `weather_member_id` | string | Non-empty provenance identifier |
| `seed` | integer | $0\le seed\le2^{32}-1$ |
| `occupant_count` | integer or `None` | Optional fixed count from 1 to 5; otherwise sampled |

The weather frame must satisfy the thermal weather contract and additionally
contain:

| Column | Unit | Use |
|---|---:|---|
| `I_beam_horizontal_W_m2` | W/m² | Direct horizontal input to RichardsonPy lighting |
| `I_diffuse_horizontal_W_m2` | W/m² | Diffuse horizontal input to RichardsonPy lighting |
| `I_solar_W_m2` | W/m² | Auditable total; must equal beam plus diffuse |

All irradiance must be finite and non-negative. Timestamps must align exactly
with the façade-transposed irradiance consumed by the thermal core. Radiation
is not transposed inside the behaviour layer.

### 3.2 Frozen central assumptions

`behaviour/behaviour_assumptions.csv` is executable: missing, extra, duplicated
or malformed assumption rows stop generation. Its complete SHA-256 is carried
into every result. The central choices are:

| Quantity | Value | Reason |
|---|---:|---|
| RichardsonPy | 0.2.2 | Exact dependency is pinned |
| Native timestep | 60 s | Package-native electricity resolution |
| Annual appliance + lighting electricity | 3,500 kWh/dwelling | Belgian CREG residential comparison benchmark |
| `prev_heat_dev` | `True` | Excludes electric space-heating and DHW devices |
| Appliance randomisation | `True` | Retains seeded household diversity |
| Seasonal lighting modifier | `False` | Lighting already uses the selected climate forcing |
| Appliance sensible fraction | 1.0 | Central single-conditioned-zone assumption |
| Lighting sensible fraction | 1.0 | Lighting electricity becomes indoor sensible heat |
| Active-person sensible gain | 70 W/person | ISO 13790 Annex G average |
| Active heating setpoint | 20°C | Same central value as Gate 3 |
| Inactive heating setpoint | 18°C | Modest 2 K setback |
| Cooling setpoint | 26°C | Fixed; cooling adoption is not invented here |

The 3,500 kWh value is the benchmark used in CREG study F2223. It is a
regulatory comparison case, not an empirical conditional mean for every
household size or dwelling type. Using one value keeps this gate simple and
isolates profile timing. A later sensitivity or Monte Carlo specification may
replace it with a documented conditional distribution, but this deterministic
gate does not silently do so.

### 3.3 Clock convention

RichardsonPy generates repeated 24-hour days. The wrapper therefore uses fixed
Belgian standard time, CET = UTC+1, without daylight-saving transitions. UTC
weather is cyclically shifted into fixed CET before generation, and the output
is shifted back to the exact UTC weather index. The cyclic boundary is
consistent with the thermal solver's periodic-year convergence.

This deliberately avoids missing or duplicated civil-time hours and makes leap
years reproducible. It does not reproduce daylight-saving clock changes, which
is a documented behavioural timing limitation.

## 4. Occupant sampling

Household size is conditioned on the broad building class:

- `Detached house`, `Semi-detached house`, `Terraced house` $\rightarrow$ SFH.
- `Apartment, enclosed`, `Apartment, exposed` $\rightarrow$ MFH.

The probabilities are derived from Eurostat Census 2021 table
`cens_21dwbno_r3` for Belgium. SFH is one-dwelling residential buildings;
MFH combines two-dwelling and three-or-more-dwelling buildings. RichardsonPy
supports no more than five occupants, so exact five-person and six-or-more
counts are combined into a transparent `5_or_more` source bin and simulated as
five.

| Simulated occupants | SFH probability | MFH probability |
|---:|---:|---:|
| 1 | 0.247308 | 0.523603 |
| 2 | 0.347018 | 0.271309 |
| 3 | 0.164868 | 0.097202 |
| 4 | 0.158111 | 0.061162 |
| 5 (`5_or_more`) | 0.082695 | 0.046724 |

The exact source counts, dataset DOI, API URL, retrieval date and
transformations are stored in
`behaviour/data/reference/occupant_distribution.provenance.json`.

The supplied seed is split deterministically with NumPy `SeedSequence` into:

1. An occupant-count stream.
2. A RichardsonPy stream.

RichardsonPy 0.2.2 uses module-global random-number generators. The wrapper
saves the Python and NumPy states, seeds them inside a process lock, generates
the profile, and restores both states. This makes repeated calls reproducible,
prevents leakage to calling code and prevents concurrent calls from racing.

## 5. RichardsonPy generation and aggregation

### 5.1 Native outputs

For one household, RichardsonPy generates:

- Active occupancy at ten-minute resolution.
- Appliance power at one-minute resolution.
- Lighting power at one-minute resolution.

`prev_heat_dev=True` excludes electric space-heating and hot-water appliances.
The package's `do_normalization=True` scales the combined appliance and lighting
profile to the annual reference while retaining its stochastic shape.

The wrapper provides horizontal beam and diffuse irradiance in kW/m², as
required by RichardsonPy:

$$
I_{beam,kW/m^2}(m)=\frac{I_{beam,W/m^2}(h)}{1000}
$$

$$
I_{diffuse,kW/m^2}(m)=\frac{I_{diffuse,W/m^2}(h)}{1000}
$$

where each hourly value is repeated for its 60 constituent minutes. This is the
same weather realisation used for the thermal façade gains; only the required
orientation and unit differ.

### 5.2 Compatibility correction

In RichardsonPy 0.2.2, `ElectricLoad` passes a zero-based day number to a helper
that compares it with one-based cumulative month-end days, and it does not pass
the leap-year flag. The wrapper applies a local, locked correction:

$$
d_{helper}=d_{zero-based}+1
$$

and supplies the actual calendar leap status. The original dependency method
is restored immediately after generation. The test suite verifies January,
February and leap-day boundaries. This correction is version-specific and must
be reviewed before changing the pinned package version.

### 5.3 Energy-conserving hourly aggregation

Let $P_{app,m}$ and $P_{light,m}$ be minute-average powers. Hourly values are:

$$
P_{app,h}=\frac{1}{60}\sum_{m=1}^{60}P_{app,m}
$$

$$
P_{light,h}=\frac{1}{60}\sum_{m=1}^{60}P_{light,m}
$$

Because every hourly interval lasts one hour, annual electrical energy is:

$$
E_{el}=\frac{1}{1000}\sum_h(P_{app,h}+P_{light,h})
$$

and must equal 3,500 kWh within $10^{-6}$ kWh. Values are averaged, not summed,
so the hourly series remains power in watts.

Let $N_{h,j}$ be RichardsonPy's number of active occupants in ten-minute period
$j$. Two hourly occupancy variables are retained:

$$
\bar N_{active,h}=\frac{1}{6}\sum_{j=1}^{6}N_{h,j}
$$

$$
A_h=\mathbb{1}\left(\max_j N_{h,j}>0\right)
$$

$\bar N_{active,h}$ drives metabolic gains. $A_h$ drives the simple thermostat
schedule. “Active” is the Richardson activity state, not a complete inference
of physical presence; sleeping and inactive presence are not distinguished.

## 6. Conversion to sensible internal heat

The wrapper retains the components for audit and sends only their sum to the
thermal core:

$$
\Phi_{occ,h}=70\bar N_{active,h}
$$

$$
\Phi_{app,h}=f_{app}P_{app,h}
$$

$$
\Phi_{light,h}=f_{light}P_{light,h}
$$

$$
\Phi_{int,h}=\Phi_{occ,h}+\Phi_{app,h}+\Phi_{light,h}
$$

with $f_{app}=f_{light}=1$ centrally. Included heat is active-occupant sensible
metabolic heat and generated appliance/lighting electricity released in the
conditioned zone. The following are excluded:

- Latent metabolic and cooking gains.
- Drain losses and DHW heat.
- Electric space heating.
- Appliances outside the conditioned zone.
- Heat fractions that RichardsonPy cannot identify by appliance end use.

The 100% appliance fraction is consequently a transparent central
simplification, not a claim that every real appliance transfers all annual
electricity sensibly to indoor air at the same moment.

## 7. Setpoint construction

The heating schedule is deliberately simple:

$$
\theta_{set,H,h}=\begin{cases}
20^\circ\mathrm{C}, & A_h=1\\
18^\circ\mathrm{C}, & A_h=0
\end{cases}
$$

$$
\theta_{set,C,h}=26^\circ\mathrm{C}
$$

There is no thermostat hysteresis, preheating, adaptive comfort, window opening
or stochastic cooling adoption. These choices retain the supervisor-approved
ideal-controller scope and isolate behavioural timing from HVAC-system sizing.

## 8. Fail-fast sanity checks

Every generated household must pass all of the following before coupling:

- Exact timestamp equality with the weather frame.
- A complete 8,760- or 8,784-hour series.
- Finite, non-negative electricity and internal-gain components.
- Appliance plus lighting power equals total electricity.
- Occupant, appliance and lighting gains equal $\Phi_{int}$.
- Active occupants never exceed sampled household size.
- Both active and inactive hours occur.
- Heating setpoints follow the declared activity rule and never exceed cooling.
- Annual electricity matches the normalization target.
- Electricity is temporally variable.
- Identical inputs and seeds reproduce the profile exactly.

Diagnostics record occupant count, active-person hours, active hours, annual and
peak electricity, annual and peak gains, day/night means, weekday/weekend means,
the two derived seeds, package version, weather-member identifier and SHA-256
checksums for all behavioural forcing and assumptions.

## 9. Deterministic coupling audit

### 9.1 Design

One fixed realization is generated for each broad household class and reused
across every corresponding physical archetype/state. This controls behaviour
while testing all 75 thermal combinations:

| Class | Seeded occupants | Annual appliance electricity | Annual lighting electricity | Active hours | Mean internal gain |
|---|---:|---:|---:|---:|---:|
| SFH | 2 | 2,989.381 kWh | 510.619 kWh | 4,963 h | 448.844 W |
| MFH | 1 | 3,098.220 kWh | 401.780 kWh | 4,183 h | 426.148 W |

These are individual reproducible draws. They are checked against distribution
properties but are not expected to equal a population mean.

Each cell is simulated twice under the same 2015 weather:

1. Gate-3 reference: constant 3 W/m² gains and 20/26°C setpoints.
2. Gate-4 profile: stochastic gains and 18/20°C activity-based heating setpoint,
   with fixed 26°C cooling.

The audit passes all hard numerical checks:

| Check | Result |
|---|---:|
| Archetype/state cells | 75 |
| Maximum energy-balance residual | $1.43\times10^{-10}$ W |
| Maximum controlled-temperature error | $6.39\times10^{-14}$ K |
| Cells with simultaneous heating and cooling | 0 |

The median heating change is −4.676 kWh/m², with a range from −41.219 to
+4.825 kWh/m². A positive change is possible in large, efficient SFHs because
the fixed 3,500 kWh/dwelling profile gives less than 3 W/m² mean internal heat;
the inactive setback does not necessarily offset that loss. In compact MFHs,
the same household electricity target gives a higher gain density and generally
reduces heating more strongly. This is a physically interpretable consequence
of the frozen per-dwelling normalization and should not be calibrated away.

### 9.2 Coupling-effect decomposition

Effects are isolated for two middle-period existing archetypes:

- `BE_TABULA_11`: detached house, 1971–1990.
- `BE_TABULA_14`: enclosed apartment, 1971–1990.

Five simulations are used:

| Code | Gains | Heating setpoint | Lighting timing |
|---|---|---|---|
| A | Constant 3 W/m² | Constant 20°C | Not applicable |
| B | Behavioural annual mean | Constant 20°C | Flat within total mean |
| C | Dynamic behavioural | Constant 20°C | Weather-driven |
| D | Dynamic behavioural | 18/20°C | Weather-driven |
| E | Dynamic except lighting is flattened at equal annual energy | 18/20°C | Flat |

The signed heating effects, in kWh/m², are:

| Effect | Definition | Detached house | Enclosed apartment |
|---|---|---:|---:|
| Gain magnitude | B − A | +7.747 | −7.461 |
| Gain timing | C − B | +0.112 | +0.622 |
| Setpoint setback | D − C | −19.378 | −9.072 |
| Lighting-weather timing | D − E | −0.023 | −0.013 |

The decomposition is diagnostic rather than additive: the lighting comparison
shares the full behavioural setpoint and other dynamic gains, while gain timing
and setback are sequential contrasts. It demonstrates that intermittent gains,
setpoint setbacks and climate-driven lighting all reach the thermal solver
through schedules while the physical core remains unchanged.

## 10. Generated artifacts

`thermal_model/data/behaviour/` contains:

| Artifact | Contents |
|---|---|
| `fixed_profile_sfh.csv` | Complete fixed-seed SFH profile and gain components |
| `fixed_profile_mfh.csv` | Complete fixed-seed MFH profile and gain components |
| `fixed_profile_diagnostics.csv` | Household sanity metrics and provenance |
| `deterministic_coupling_comparison.csv` | 75 constant-versus-behavioural comparisons |
| `coupling_effect_decomposition.csv` | Scenario totals and four isolated contrasts |
| `coupling_summary.json` | Machine-readable status, checks and provenance |

## 11. Interpretation and limits

Gate 4 verifies the coupling and provides a deterministic impact audit; it does
not validate Belgian occupant time use. RichardsonPy's activity, appliance and
lighting mechanisms originate from UK evidence. Belgium enters through the
annual CREG electricity benchmark, the Belgian Census household-size
distribution and the project's Belgian climate forcing.

The central model also does not represent DHW, latent gains, room-by-room
occupancy, rebound, window opening, HVAC capacity, thermostat hysteresis,
cooling ownership or adaptive comfort. Those omissions preserve the declared
single-zone sensible heating/cooling scope. The Monte Carlo gate may vary the
seed and sample weather members, but it must not reinterpret these omissions as
uncertainty without a separately sourced contract.

## 12. Principal sources

- RWTH-EBC, [RichardsonPy source and package documentation](https://github.com/RWTH-EBC/richardsonpy), version 0.2.2.
- Richardson et al. (2008), [stochastic active-occupancy model](https://doi.org/10.1016/j.enbuild.2008.02.006).
- Richardson et al. (2010), [high-resolution domestic electricity model](https://doi.org/10.1016/j.enbuild.2010.05.023).
- CREG, [Study F2223](https://www.creg.be/fr/publications/etude-f2223), source of the 3,500 kWh residential comparison case.
- Eurostat, [Census 2021 dwelling/occupant dataset](https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/cens_21dwbno_r3?geo=BE&lang=en), DOI 10.2908/CENS_21DWBNO_R3.
- ISO, [ISO 13790:2008](https://www.iso.org/standard/41974.html), source of the thermal-core method and 70 W/person sensible-gain reference.
