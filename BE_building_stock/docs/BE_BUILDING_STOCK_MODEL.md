# Belgian residential building-stock model

## Scope

This repository represents the Belgian residential stock with 25
TABULA/VITO archetypes, three regions and three physical envelope states. The
modeled stock contains the 5,537,385 dwellings in Statbel building categories
R1–R4 on 1 January 2025. The 290,438 dwellings in R5–R6 are retained as an
excluded residual. The modeled share equals 95.0% of the 5,827,823 Statbel
dwellings.

The 2050 files describe one renovation-state projection of this fixed 2025
stock.
Demolition, new construction and changes in household numbers can be added as a
separate stock-evolution module after the thesis scope is confirmed.

The core building-stock pipeline ends with the renovation-state projection. The
heating, cooling and PV assignment layer is isolated as an optional,
2024-vintage module in `docs/OPTIONAL_TECHNOLOGY_LAYER.md`. Energy demand,
system efficiency, COP, SEER, emissions and electricity generation belong to
the downstream thermal-demand model.

## Source hierarchy and evidence

| Input | Use | Evidence boundary |
| --- | --- | --- |
| [Statbel 2025 cadastral building stock](https://statbel.fgov.be/en/open-data/cadastral-statistics-building-stock-24) | Regional R1–R4 dwelling totals and joint R1–R3 type-by-period profiles | R1–R3 period profiles count buildings and are scaled to the matching dwelling total. |
| [Statbel Census 2021 HC37](https://statbel.fgov.be/nl/open-data/census-2021-conventionele-woningen-naar-verblijfplaats-arrondissement-type-gebouw) | Regional apartment construction-period profiles | Occupied and unoccupied conventional dwellings in buildings with two or at least three dwellings are scaled to the 2025 R4 total. |
| [Belgian TABULA/VITO scientific report](https://episcope.eu/fileadmin/tabula/public/docs/scientific/BE_TABULA_ScientificReport_VITO.pdf) | Geometry, existing envelope parameters, standard and advanced packages | Representative archetype parameters from 2011. Thermal bridges are outside the TABULA values used here. |
| [TABULA final report](https://episcope.eu/fileadmin/tabula/public/docs/report/TABULA_FinalReport.pdf) | Three-state conceptual basis | Page 9 defines Existing State, Standard Measures and Advanced Measures. |
| [Flemish Housing and Energy Policy Note 2024–2029](https://publicaties.vlaanderen.be/view-file/70827) | Flemish 2025 state calibration | Page 21 of the PDF reports 9% EPC A and 22% EPC B at the beginning of 2024; footnote 3 describes a weighted pre/post-2006 distribution with reference date 1 January 2024. |
| [Brussels residential PEB statistics 2024](https://document.environnement.brussels/opac_css/elecfile/Statistiques_certificatsPEB_residentiel_donnees_2024.pdf) | Brussels 2025 state calibration | Page 8 reports 1.55% in the A family and 5.62% in the B family among 406,785 certificates established through 1 January 2025. The set includes existing and new dwellings and expired certificates. |
| [Walloon environmental indicator MEN 10](https://etat.environnement.wallonie.be/contents/indicatorsheets/MEN%2010.html) | Walloon 2025 state calibration | The indicator reports 1.2% A/A+/A++ and 11.0% B among 790,073 cumulative certificates for pre-May-2010 dwellings at 20 August 2024. |
| [European Commission renovation study](https://energy.ec.europa.eu/document/download/2b58c118-89c1-46b5-a450-0f2d5d215e2c_en?filename=1.final_report.pdf) | Historical literature context | Table 2 on PDF page 16 / printed page 15 reports a 0.2% Belgian residential deep-renovation rate for 2012–2016, based on floor area and primary-energy savings above 60%. The observation does not parameterise the 2050 projection. |
| [Scenarios for a Climate Neutral Belgium by 2050](https://climat.be/doc/climate-neutral-belgium-by-2050-report.pdf) | National ARR and renovation-depth calibration | Appendix 1, Table 1 on PDF and printed page 49 reports 2.8% annual residential renovation by 2025 and a 40% shallow, 50% medium and 10% deep mix in CORE-95, Behaviour and Technology. |
| [European Commission NBRP register](https://energy.ec.europa.eu/topics/energy-efficiency/energy-performance-buildings/national-building-renovation-plans_en) | Current Walloon policy context | The draft Walloon NBRP was submitted on 23 December 2025, Commission feedback was published on 14 July 2026, and the final plan is due by 31 December 2026. |
| [Meier, “Infiltration: Just ACH50 Divided by 20?”](https://www.aivc.org/sites/default/files/airbase_7556.pdf) | Conversion from blower-door airflow to annual-average infiltration | The paper reviews the rule-of-20 and its climate, height, shielding and leakage-distribution limitations; the rule's original attribution is uncertain. |

Access dates and precise page locators are recorded in the assumption and
verification CSV files. Immutable downloaded documents are identified by their
retrieved SHA-256 values. Mutable HTML pages can change byte-for-byte through
page-generation metadata while the published evidence remains unchanged. The
Walloon calibration therefore uses a stable extracted-count snapshot in
`data/raw/provenance/wallonia_men10_certificate_counts_2024.csv`; its local
SHA-256 is enforced by `scripts/verify_source_integrity.py`, while the HTML
response hash records the exact retrieval instance.

## Stock weights

The regional stock calculation uses a joint type-period profile:

```text
N(r,a,2025)
  = Statbel_T8_dwellings(r,type(a))
    × conditional_period_share(period(a) | r,type(a))
```

For R1–R3, the conditional period share comes from the 2025 cadastral
type-by-period counts. The cadastral 1982–1991 and 2002–2011 bands are allocated
uniformly by calendar year at the 1990 and 2005 TABULA cut-offs. For R4, HC37
provides separate 2001–2005 and 2006–2010 counts. Unknown construction periods
are distributed pro rata through normalisation of the known-period profile.

Within R1–R3, the construction-period distribution of buildings is used as the
construction-period distribution of dwellings in the matching category. The
regional T8 dwelling total is distributed over that profile. Multi-dwelling
house-category buildings therefore retain the whole-house TABULA geometry of
their registered category. National absolute heat-demand and capacity results
are conditional on this geometry mapping.

Statbel R4 covers dwellings in apartment buildings. Each regional R4
type-period cell is split equally between enclosed and exposed TABULA apartment
archetypes. The 50/50 split is a balancing assumption because apartment position
is absent from the Statbel inputs.

## Three physical states

The state identifiers are fixed throughout every CSV and script:

| State identifier | TABULA basis | Physical parameters |
| --- | --- | --- |
| `TABULA_existing` | Existing State | Age- and type-specific geometry, U-values and `v50` from the base archetype matrix. |
| `TABULA_standard_B_proxy` | EPB2010 / Standard Measures | Facade 0.40, roof 0.30, floor 0.40, window 2.00, door 2.90 W/(m²K); `v50=6.0` m³/(h·m²); exhaust-air ventilation. |
| `TABULA_advanced_A_proxy` | Low Energy / Advanced Measures | Facade 0.25, roof 0.15, floor 0.25, window 1.60, door 1.60 W/(m²K); `v50=2.5` m³/(h·m²); balanced ventilation with `hrv_eta=0.80` and summer bypass. |

TABULA reports that the post-2005 class already achieves the EPB2010 envelope
values. The standard proxy therefore has the same envelope performance as the
post-2005 existing archetypes. The complete Standard cells remain in the 2025
calibration matrix. Post-2005 Existing cells are ineligible for medium
transitions and remain eligible for Advanced transitions.

The letters B and A identify regional calibration anchors. Regional EPC/PEB
methods also include standardised system and energy-performance effects, while
the three model states contain physical envelope packages. The term “proxy”
makes this evidence bridge explicit.

## Calibrated 2025 state shares

| Region | `TABULA_existing` | `TABULA_standard_B_proxy` | `TABULA_advanced_A_proxy` |
| --- | ---: | ---: | ---: |
| Flanders | 69.00% [68.00–70.00%] | 22.00% [21.50–22.50%] | 9.00% [8.50–9.50%] |
| Wallonia | 87.80% [87.70–87.90%] | 11.00% [10.95–11.05%] | 1.20% [1.15–1.25%] |
| Brussels | 92.83% [92.82–92.84%] | 5.62% [5.615–5.625%] | 1.55% [1.545–1.555%] |

The shares are stored in
`data/assumptions/renovation/regional_state_shares_2025.csv` together with
reference dates, populations, locators, limitations and source-precision
intervals. Flemish reported A and B shares receive a ±0.5 percentage-point
interval, Brussels ±0.005 percentage points and Wallonia ±0.05 percentage
points. The residual existing interval follows from the A and B bounds:

```text
existing_lower = 1 - standard_upper - advanced_upper
existing_upper = 1 - standard_lower - advanced_lower
```

The three bounds are correlated. An uncertainty draw samples standard and
advanced shares within their reported intervals and calculates the existing
share as `1 - standard - advanced`. The midpoint values drive the deterministic
2050 files; the bounds are available for later Monte Carlo sampling.

Regional certificate distributions are used as indicative estimates of the
2025 energy-performance state. The distributions are not fully harmonised
because Flanders reports a weighted EPC distribution for houses and apartments
at 1 January 2024, Wallonia reports cumulative PEB certificates for existing
dwellings with a pre-May-2010 permit through 20 August 2024, and Brussels
reports residential PEB certificates established through 1 January 2025,
including existing and new dwellings and expired certificates. Certificate
coverage, triggering events, validity, and regional calculation conventions
also differ.

### Independence/null assumption

A representative joint table of region × archetype × physical state is
unavailable. The model therefore applies each regional state marginal uniformly
to every archetype:

```text
P(S=s | region=r, archetype=a) = P(S=s | region=r) = p(r,s)
N(r,a,s,2025) = N(r,a,2025) × p(r,s)
```

Here `r` denotes region, `a` archetype, `s` physical state, `p(r,s)` the
regional share assigned to state `s`, and `N` a dwelling count.

This is the row–column independence form used for categorical contingency
tables. Günel and Dickey (1974) describe row–column independence as the standard
null hypothesis for a two-way table. Verellen and Allacker (2022) document the
data gaps encountered in Belgian building-stock modelling and recommend
transparent reporting of data quality and assumptions. The current
implementation follows that reporting principle by storing the formula and
evidence status on every state row.

The assumption preserves all regional state totals and all Statbel archetype
totals. Construction period, dwelling type, tenure, income and transaction
status correlations remain uncertainty dimensions. The deterministic heat-loss
ranking supplies a physical ordering for future renovations while leaving the
calibrated 2025 marginals unchanged.

Regional evidence quality differs:

- Flanders uses a weighted regional EPC distribution and provides the strongest
  stock-level anchor of the three sources. Its A and B shares are reported as
  whole percentages, so their rounding uncertainty carries into the residual
  existing share.
- Brussels uses all certificates established through 1 January 2025, including
  expired certificates and both existing and new dwellings. The source discusses
  transaction timing and conservative conventional inputs when documentation is
  missing.
- Wallonia uses cumulative transaction and audit certificates for dwellings with
  a planning permit before May 2010. The source discusses selection effects and
  the lag between certificate date and later renovation work. Its A-family and
  B shares are reported to one decimal.

## Infiltration conversion

TABULA/VITO reports envelope air permeability at 50 Pa:

```text
q50 = v50 × A_env                      [m³/h]
n50 = q50 / V                          [h⁻¹]
n_inf = n50 / 20                       [h⁻¹]
Vdot_inf = q50 / 20                    [m³/h]
H_inf = rho_air × cp_air × Vdot_inf / 3600  [W/K]
```

The constants are:

```text
infiltration_n_factor = 20
rho_air = 1.2 kg/m³
cp_air = 1005 J/(kg K)
```

The factor 20 is a rule-of-thumb conversion from blower-door leakage to an
annual-average natural-infiltration proxy under normal pressure conditions.
Published reviews caution that the appropriate divisor depends on climate,
height, shielding and leakage distribution, so the fixed factor supplies only a
simple and consistent national screening conversion. Weather-driven airflow,
building height, wind exposure, leakage distribution and occupant window
opening remain uncertainty dimensions for the thermal simulation.

## Specific heat-loss priority

For each archetype-state cell:

```text
H_tr =
    U_roof × A_roof
  + U_facade × (A_wall + A_wall_unheated)
  + U_floor × (A_floor_soil + A_floor_unheated)
  + U_window × A_window
  + U_door × A_door

z = (H_tr + H_inf) / A_floor           [W/(m² K)]
```

`z`, written as `z_a` when the archetype index is emphasised, is a simplified
archetype-level ranking metric. A larger value represents more transmission and
screening infiltration loss per square metre for the same indoor–outdoor
temperature difference. The later thermal-demand model adds ground and
unheated-space boundary factors, climate, solar and internal gains, controls,
thermal mass and detailed ventilation; thermal bridges remain omitted because
the TABULA source provides no archetype-level junction data.

Within each region, eligible current-state cells are sorted by descending `z`.
TABULA type number resolves ties. Medium transitions rank `TABULA_existing`
cells. Advanced transitions rank `TABULA_existing` and
`TABULA_standard_B_proxy` cells.

## National renovation projection

`ARR` denotes annual renovation activity as a share of the fixed Belgian 2025
R1–R4 stock. The projection uses `ARR = 0.028/year` and a 40/50/10
shallow–medium–advanced depth distribution. Both parameters come from Appendix
1, Table 1 of *Scenarios for a Climate Neutral Belgium by 2050*, which reports
these settings for the CORE-95, Behaviour and Technology pathways by 2025. The
model holds them constant over the 25 annual steps from 2026 through 2050. This
post-2025 continuation is a model calibration because the source does not
provide a harmonised Belgian depth trajectory for those years.

The same federal table associates shallow, medium and deep renovation with
approximately 85, 64 and 25 kWh/(m²·year). These whole-building
energy-intensity categories are represented through TABULA envelope packages:

| Federal renovation depth | Share | TABULA representation | Physical effect |
| --- | ---: | --- | --- |
| Shallow | 40% | Existing/as-is | No envelope-state change |
| Medium | 50% | Standard refurbishment | Intermediate envelope improvement; eligible pre-2006 Existing cells move to Standard |
| Advanced (federal deep category) | 10% | Low-energy refurbishment | Deep envelope improvement; eligible Existing or Standard cells move to Advanced |

The terms *medium* and *advanced* identify the transition depths in the model.
The federal report names the 10% category *deep*, while the VITO package used
for its physical representation is called *Low Energy*.

The European Commission renovation study reports 0.2% deep renovation of
Belgian residential floor area during 2012–2016 for work achieving more than
60% primary-energy savings. This historical observation is retained in the
literature review and provenance register as context for past activity.

The national activity quota for depth `d` is proportionally disaggregated over
the regions using their modeled R1–R4 dwelling counts. Consequently, every
region receives the same ARR and depth distribution while retaining its own
2025 state calibration, archetype composition and within-region priority
ranking:

```text
annualRenovations(BE,d)
    = ARR × depthShare(d) × N(BE,2025)
annualRenovations(r,d)
    = N(r,2025) / N(BE,2025) × annualRenovations(BE,d)
    = ARR × depthShare(d) × N(r,2025)
nominalRenovationsTo2050(r,d)
    = annualRenovations(r,d) × 25
appliedRenovationsTo2050(r,d)
    = sum of annual transitions after eligibility limits
```

The engine runs 25 annual steps from 2026 through 2050. Advanced transitions
are allocated first from current existing and standard cells. Medium
transitions then use the remaining pre-2006 current existing stock. Post-2005
Existing cells already represent the VITO EPB2010 standard package and can move
to Advanced. This ordering gives each dwelling at most one modeled transition
per year and allows a standard dwelling to reach advanced in a later year.
Unused medium and advanced quotas remain separated. Shallow activity appears in
the activity-accounting output and leaves the three physical state counts
unchanged.

Allocation totals count state-transition events. One dwelling can contribute a
medium event and an advanced event in different years. The aggregated Standard
state does not preserve complete cohort identity, so the output reports a
guaranteed lower bound on repeated transitions:

```text
minimumRepeatEvents
  = sum over archetypes max(0,
      standardToAdvancedEvents - initialStandardDwellings)

maximumUniqueDwellingsWithPhysicalTransition
  = physicalTransitionEvents - minimumRepeatEvents
```

The unique-dwelling quantity is an upper bound. Figures use the event metric and
label it explicitly. In the period-priority figure, the zero post-2005 bar
results from the descending-`z` allocation exhausting the modeled quotas among
higher-loss cohorts. Post-2005 Existing cells remain eligible for Advanced.

The resulting physical-state shares in 2050 are:

| Region | Existing | Standard | Advanced |
| --- | ---: | ---: | ---: |
| Belgium | 35.30% | 51.89% | 12.81% |
| Flanders | 27.00% | 57.00% | 16.00% |
| Wallonia | 45.80% | 46.00% | 8.20% |
| Brussels | 50.83% | 40.62% | 8.55% |

## Reproducible pipeline

Create a conventional environment in `BE_building_stock`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Run the complete chain from the repository root:

```bash
python BE_building_stock/scripts/size_type_composition.py
python BE_building_stock/scripts/construction_period.py
python BE_building_stock/scripts/archetype_matrix.py \
  --tabula-pdf BE_building_stock/data/inputs/physical/BE_TABULA_ScientificReport_VITO.pdf
python BE_building_stock/scripts/regional_pipeline.py
node BE_building_stock/scripts/renovation_scenarios_2050.mjs
python scripts/make_archetype_table.py
python scripts/plot_stock_composition.py
python scripts/plot_renovation_states.py
python scripts/plot_thermal_parameters.py
python BE_building_stock/scripts/verify_source_integrity.py
```

`requirements.txt` states compatible direct-dependency ranges.
`requirements-lock.txt` records the exact tested Python environment and
`.python-version` records Python 3.12.13. Node.js code uses built-ins only;
`.nvmrc` records the tested Node.js runtime. The optional technology generator
has its own execution instructions in `docs/OPTIONAL_TECHNOLOGY_LAYER.md`.

## Generated outputs

| Output | Rows | Meaning |
| --- | ---: | --- |
| `data/matrices/national/base_physical_archetype_matrix.csv` | 25 | TABULA physical archetypes with converted infiltration and `z`. |
| `data/matrices/national/stock_weighted_archetype_matrix.csv` | 25 | National R1–R4 weights. |
| `data/matrices/regional/regional_stock_weighted_archetype_matrix.csv` | 75 | Three regions × 25 archetypes. |
| `data/scenarios/renovation/renovation_state_layer.csv` | 225 | Calibrated 2025 region × archetype × state matrix with source-precision bounds. |
| `data/scenarios/renovation/renovation_state_layer_with_allocation.csv` | 150 | Advanced-eligible source-state cells; 20 pre-2006 Existing cells per region also receive medium ranks. |
| `data/scenarios/renovation/archetype_matrix_2050_renovation_scenarios.csv` | 225 | One projection × three regions × 25 archetypes × three states. The plural filename is retained for interface compatibility. |
| `data/scenarios/renovation/renovation_priority_allocation_2050.csv` | 210 | Depth-specific cumulative transition-event audit trail for the national projection. |
| `data/scenarios/renovation/renovation_scenario_policy_context_2050.csv` | 3 | Regional activity summaries, transition-event accounting and current policy context. The compatibility filename is retained. |
| `data/scenarios/renovation/renovation_projection_national_summary_2050.csv` | 1 | National inputs, activity balance, applied transitions and 2025/2050 state shares. |
| `data/scenarios/renovation/renovation_state_trajectory_2025_2050.csv` | 78 | Annual 2025–2050 physical-state counts and shares for the three regions. |
| `figures/fig_renovation_state_projection_2050.{png,pdf}` | — | National and regional 2025/2050 physical-state shares. |
| `figures/fig_renovation_state_composition_be_2025_2050.{png,pdf}` | — | Belgian renovation-state composition in 2025 and 2050. |
| `figures/fig_renovation_improved_share_by_region_2025_2050.{png,pdf}` | — | Annual regional Standard plus Low Energy share. |
| `figures/fig_renovation_priority_by_period.{png,pdf}` | — | Medium and advanced transition events by construction period. |

## Validation

The scripts enforce:

- 25 complete TABULA type-period archetypes;
- automated agreement with TABULA/VITO Tables 9, 10 and 19;
- automated agreement of the standard and advanced state packages with the
  TABULA/VITO report;
- 5,537,385 modeled R1–R4 dwellings and a 290,438-dwelling R5–R6 residual;
- regional archetype and state shares summing to one;
- every archetype’s three state counts reconstructing its Statbel stock;
- `q50=v50×A_env`, `n50=q50/V`, `Vdot_inf=q50/20` and positive `z`;
- 50 advanced and 20 pre-2006 medium priority ranks per region;
- source-precision bounds and correlated residual identities;
- ARR and depth-share quota identities;
- annual eligibility, transition ordering and separated unused quotas;
- annual trajectory conservation, endpoint agreement and non-decreasing
  Standard-plus-Advanced shares;
- national, regional and archetype stock preservation through the 2050 projection;
- agreement between the national summary and the three regional activity
  records;
- temporary output validation before atomic file replacement.

The optional technology module separately validates its variant shares and
reconstruction identities.

## Main caveats

- The fixed 2025 denominator represents a renovation-state projection. A full 2050
  stock projection requires demolition, new construction and new cohort
  archetypes.
- R1–R3 building-age profiles are applied to dwelling totals, and every
  house-category dwelling receives the whole-house TABULA geometry. National
  absolute heat-demand and capacity results are conditional on this mapping.
- The regional state sources have different populations, reference dates,
  certificate triggers and validity coverage.
- EPC/PEB labels provide calibration anchors for physical TABULA proxies.
- The independence/null assumption leaves archetype-state correlations as an
  uncertainty dimension in the 2025 calibration.
- The fixed infiltration n-factor represents an annual-average screening value.
- The heat-loss ranking is a simplified archetype-level metric; the thermal
  model adds boundary factors, climate, gains and controls but continues to
  omit thermal bridges.
- Shallow activity has no separate TABULA physical state.
- The 2.8% ARR and 40/50/10 depth mix are official pathway settings reported by
  2025 and are held constant through 2050 as a model calibration.
- Fractional dwelling counts are expected values. A Monte Carlo implementation
  can sample integer buildings or dwellings later.
- Heating, cooling and PV allocations are isolated optional stock assumptions
  with a 2024 evidence vintage. They are excluded from the core building-stock
  pipeline.
