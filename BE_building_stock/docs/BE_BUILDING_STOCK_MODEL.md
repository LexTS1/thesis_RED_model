# Belgian building-stock model

## Purpose and scope

This repository constructs a **Belgian residential dwelling-stock
representation** for later building-energy and renovation analysis. Its core is
a set of 25 residential archetypes with physical characteristics and national
stock weights. A regional extension applies those archetypes to the modeled
R1-R4 stock, a renovation generator creates 450 scenario-state rows, and the
technology layer assigns stock-only heating, cooling, and PV variants.

The model does **not** calculate useful or delivered energy, emissions,
heating-system efficiency, heat-pump COP, cooling SEER, PV generation, grid
demand, demolition, or new construction. The technology files describe which
systems or installations are assigned to how many dwellings; a downstream
simulation must calculate performance.

## Repository structure

~~~text
data/
  raw/statbel/                 # downloaded Statbel workbooks and metadata
  raw/technology/              # frozen technology-source values and locators
  inputs/physical/             # curated TABULA/VITO parameter transcriptions
  derived/                     # reproducible Statbel-derived tables
  matrices/national/           # the two national archetype matrices
  derived/regional_stock/      # regional supporting data, outside core scope
  matrices/regional/           # regional matrix, outside core scope
  assumptions/renovation/      # renovation-policy and allocation choices
  assumptions/technology/      # normalized technology inputs and metadata
  scenarios/renovation/        # renovation-layer outputs
  scenarios/technology/        # independent heating, cooling, and PV outputs
  legacy/                      # retained non-authoritative legacy material
scripts/                       # reproducible data-processing scripts
docs/                          # model documentation
~~~

The former source_register.xlsx is preserved in data/legacy/. It currently
contains placeholder text only and is not a usable source register; this guide
therefore provides the working provenance record.

## Source provenance

| Source | Local input and fields used | Role in the model | Recorded reference |
| --- | --- | --- | --- |
| Statbel cadastral building-stock open data, 2025 | data/raw/statbel/building_stock_open_data_2025.xlsx, sheet 2025; CD_REFNIS, CD_STAT_TYPE, CD_BUILDING_TYPE, and MS_VALUE | National dwelling counts by building type; also produces supporting regional breakdowns | Dataset landing page, licence, download date, and preferred thesis citation are not recorded locally. |
| Statbel building-stock metadata | data/raw/statbel/TF_Building_stock_metadata variables.xlsx | Definitions of the coded Statbel variables | Metadata workbook supplied with the data. |
| Statbel Census table T04, 2021 | data/raw/statbel/T04_POC_BE_NL.xlsx, sheet CENSUS_T04_21_BE_POC_CL_2021 | Construction-period distribution of conventional dwellings | Dataset landing page, licence, download date, and preferred thesis citation are not recorded locally. |
| Belgian TABULA/VITO scientific report | Three CSVs in data/inputs/physical/ | Geometry, U-values, and in/exfiltration rates for the 25 archetypes | Tables 9, 10, and 19 of the [Belgian TABULA/VITO report](https://episcope.eu/fileadmin/tabula/public/docs/scientific/BE_TABULA_ScientificReport_VITO.pdf). The PDF is not stored in this repository. |
| Belgian Social Climate Plan delivery report | `data/raw/technology/technology_source_snapshot_2024.csv` | Regional as-is heating-carrier shares | Table 3-2 of the [published report](https://klimaat.be/doc/be-dlv2-final-report-clean-1.pdf), based on BE-SILC 2022. |
| Statbel Household Budget Survey 2024 | Same technology snapshot | Regional integrated-air-conditioning shares | [Statbel release](https://statbel.fgov.be/fr/nouvelles/le-logement-premier-poste-de-depenses-des-belges). |
| European Commission EPBD guidance | Metadata source registry | 45°C classification for low-temperature emitters | [Commission guidance](https://eur-lex.europa.eu/legal-content/en/ALL/?uri=CELEX%3A52025XC06438). |
| VEKA/Fluvius, BRUGEL/Sibelga, ORES/SPW/CWaPE | Technology snapshot and normalized PV inputs | End-2024 small-PV proxy, regional trajectories, yields, and rooftop caps | Exact URLs and extraction notes are recorded per value in the snapshot and in `technical_systems_metadata.yaml`. |

The physical-input CSVs preserve their source-table names and report URL in
their source_table and source_url columns. The matrix builder can verify these
transcriptions against a local copy of the TABULA PDF before it writes outputs.
Technology-source values are preserved separately from transformations. The
metadata classifies fields as observed, derived, proxy, model assumption, or
scenario assumption and documents the less-than-or-equal-to-10 kW/kVA PV
extraction method for each region.

## Archetype definition

The national model contains **25 archetypes**: five dwelling types crossed with
five construction periods.

### Dwelling types

| Model dwelling type | Statbel building-type treatment |
| --- | --- |
| Terraced house | R1, closed/terraced houses |
| Semi-detached house | R2 |
| Detached house | R3, including the source category's open houses, farms, and castles |
| Apartment, enclosed | One half of R4 dwelling stock |
| Apartment, exposed | One half of R4 dwelling stock |

R5 and R6 are not represented by their own TABULA dwelling archetypes. Their
treatment in the national matrix is documented under assumptions below.

### Construction periods

| Model period | Census periods used |
| --- | --- |
| pre-1946 | Before 1919 + 1919–1945 |
| 1946-1970 | 1946–1960 + 1961–1970 |
| 1971-1990 | 1971–1980 + 1981–1990 |
| 1991-2005 | 1991–2000 + half of 2001–2010 |
| post-2005 | half of 2001–2010 + 2011 onwards |

The split of 2001–2010 is necessary because the supplied Census category spans
the 2005 TABULA boundary.

### Physical characteristics

Each archetype combines the following TABULA/VITO inputs:

- Geometry from Table 19: floor area, protected volume, envelope components,
  door area, and orientation-specific window areas, in m² or m³ as named in
  the columns.
- Envelope U-values from Table 10, in W/m²K.
- In/exfiltration at 50 Pa from Table 9, in m³/(h·m²), stored as
  v50_m3_h_m2.

data/matrices/national/base_physical_archetype_matrix.csv is the unweighted
25-row physical matrix. Its archetype_id values (BE_TABULA_01 through
BE_TABULA_25) identify the TABULA type numbers used throughout the core model.

## National stock construction

The workflow is:

~~~text
Statbel 2025 building-stock workbook ──> dwelling-type composition
Statbel Census 2021 T04 workbook ──────> construction-period composition
TABULA/VITO Tables 9, 10, 19 ──────────> physical archetype matrix
                                             │
                                             └──> national stock-weighted matrix
~~~

1. scripts/size_type_composition.py reads the 2025 Statbel workbook. It uses
   T1 for building counts and T8 for dwelling counts, validates the supplied
   national totals (4,668,425 buildings and 5,827,823 dwellings), and writes
   four supporting CSVs to data/derived/stock_composition/.
2. scripts/construction_period.py reads the 2021 Census T04 sheet and
   aggregates its detailed construction bands into the seven intermediate
   periods in
   data/derived/construction_periods/dwellings_by_construction_period.csv.
   Percentages are calculated among dwellings with a known construction period.
3. scripts/archetype_matrix.py merges the three TABULA/VITO input tables into
   the physical matrix, then applies national dwelling-type and construction-
   period shares to calculate national_share and number_of_dwellings.

For archetype *a* with dwelling type *d* and construction period *p*, the
national share is:

~~~text
national_share(a) = type_share(d) × period_share(p)
number_of_dwellings(a) = 5,827,823 × national_share(a)
~~~

The resulting file is
data/matrices/national/stock_weighted_archetype_matrix.csv. It retains every
physical field and adds number_of_dwellings, national_share, regional_share,
and weighting_method.

## Regional renovation and technology layers

The regional matrix contains 75 rows: the 25 archetypes repeated for Flanders,
Wallonia, and Brussels. Only Statbel residential types R1-R4 are retained in
the regional denominator. R5/R6 residual dwellings remain outside this part of
the model. The renovation generator combines three renovation scenarios with
two states per regional archetype, producing 450 rows in
`data/scenarios/renovation/archetype_matrix_2050_renovation_scenarios.csv`.

The renovated physical package now uses `hrv_eta=0.80` and
`summer_bypass=true`; the obsolete `heat_recovery_efficiency` field has been
removed. The technology layer reads those 450 state rows and produces three
independent variant tables. It never materializes a heating × cooling × PV
Cartesian product.

### Heating systems

Every as-is state row is split into gas, oil, electricity, and other variants
using regional Social Climate Plan shares. Shares are uniform across
archetypes within a region. The published Walloon values are retained in the
raw snapshot with their 0.997 total, then divided by 0.997 for model use. Every
renovated state row receives one package: air-water heat pump, heat-pump DHW,
45°C low-temperature radiators, balanced HRV with 0.80 heat recovery and summer
bypass, and reversible-cooling capability.

The resulting heating table has 1,125 rows: 225 as-is rows × four variants plus
225 renovated rows × one variant. Capability indicates what the renovated heat
pump can technically do; it does not mean cooling is active.

### Cooling adoption

Cooling is represented by active and inactive rows for every renovation-state
row under three cases. `no_active_cooling` assigns 0% active cooling;
`current_regional` assigns 13% in Flanders, 7% in Wallonia, and 4% in Brussels;
and `higher_uptake` doubles those shares. Zero-dwelling active variants are
retained. The 2,700-row output explicitly separates `active_cooling` from
`heating_system_reversible_cooling_capable`.

### PV assignment and projection

The end-2024 PV proxy uses only source extracts that can support a ≤10 kW/kVA
threshold. Flanders is a direct VEKA/Fluvius extract. Brussels combines the
exact BRUGEL threshold shares with Sibelga end-year totals. Wallonia preserves
the ORES ≤10 kVA mean system size and uses the published SPW small-capacity
share to form a regional proxy; this limitation is flagged in metadata. A
regional extract with `threshold_supported=false` makes generation fail.

For each region, the model calculates the mean small-system size from the 2024
count and capacity. The official 2030 energy target is converted to capacity
using the recorded regional reference yield. The small-system share of the
2024 capacity increment is carried into the 2030 value, and that line is
extended from 2024 through 2030 to 2050. The 2050 result is capped by both the
recorded official rooftop-potential proxy and the maximum capacity that the
eligible modeled stock can host.

Current PV is assigned to houses only. For the 2050 basis, apartment
participation is 25% of the regional house participation rate. Shared-system
capacity per participating apartment dwelling equals the regional mean small
system divided by Statbel's regional average R4 dwellings per apartment
building. The solved participation rates preserve the capped regional
small-residential capacity.

As-is state rows use the 2024 assignment, while renovated rows use the 2050
projection. The 900-row output retains both PV and no-PV variants for every
state, including current apartment PV variants with zero dwellings.

For every module, the stock identity is:

~~~text
variant_dwellings = state_dwellings × variant_share
sum(variant_share within a state/module case) = 1
sum(variant_dwellings within a state/module case) = state_dwellings
~~~

### Output schemas and joins

| Output | Rows | Unique variant key | Module-specific fields |
| --- | ---: | --- | --- |
| `archetype_heating_system_layer_2050.csv` | 1,125 | base state key + `heating_variant_id` | carrier, heat and DHW systems, emitter, supply temperature, ventilation, HRV, bypass, reversible capability |
| `archetype_cooling_adoption_layer_2050.csv` | 2,700 | base state key + `cooling_case` + `cooling_variant_id` | installed/active cooling, cooling system, reversible heat-pump capability |
| `archetype_pv_assignment_layer_2050.csv` | 900 | base state key + `pv_variant_id` | basis year, participation, assigned capacity per participating dwelling, allocated and capped regional capacities |

The base state join key is `scenario`, `region`, `archetype_id`, and
`renovation_state`. Downstream simulations should join one module at a time or
aggregate modules before joining. Joining all variant tables directly on the
base state key deliberately creates a large Cartesian product and is not an
output of this repository.

## Current modelling assumptions and limitations

The following are current modelling choices. They make the available sources
usable, but should not be interpreted as direct observations unless supporting
evidence is added.

| Assumption | Current treatment and rationale | Consequence / limitation |
| --- | --- | --- |
| 2021-to-2025 proxy | The 2021 Census construction-period distribution is applied to the 2025 Statbel dwelling total because the supplied files do not provide a compatible 2025 age-by-type cross-tabulation. | This combines two reference years and two source definitions: 2021 Census conventional dwellings versus 2025 cadastral T8 dwellings. It is a proxy, not a measured 2025 construction-period distribution. |
| Type-period independence | Each archetype's stock share is the product of its type share and period share. | The available inputs do not provide a dwelling-type × construction-period cross-tabulation, so real correlations between type and age are not represented. |
| Apartment position | R4 dwellings are divided equally between enclosed and exposed apartment archetypes. | Statbel R4 does not identify apartment position; the 50/50 split is an explicit balancing assumption. |
| 2001–2010 split | The aggregated Census class is divided equally between 1991–2005 and post-2005. | The split is a practical approximation around the TABULA boundary, not an observed annual distribution. |
| Unknown construction years | Dwellings without a reported construction period are excluded before period shares are normalised. | Known-period shares sum to one, but the unknown-period group is implicitly distributed proportionally rather than represented explicitly. |
| R5/R6 residual | The model first normalises modeled R1–R4 type shares to one, then applies them to the all-type national T8 total. | R5/R6 dwellings are allocated pro rata to the five residential archetype types. The national matrix therefore retains the full 2025 dwelling total, but its modeled type composition is not an observed R5/R6 classification. |
| TABULA-as-is baseline | The physical parameters are taken from TABULA/VITO reference archetypes without a calibration step to a separate Belgian measured stock dataset. | Archetypes are representative parameter sets, not individual-building observations. |
| Uniform as-is technology | Heating-carrier and active-cooling shares are repeated across all archetypes and renovation states within a region. | Correlations with building type, age, income, tenure, and renovation status are not represented. |
| Lumped as-is packages | The four heating carriers map to deliberately broad system descriptions. | Boiler age, distribution temperature, system efficiency, and mixed or secondary heating systems are unknown. |
| Renovated package | Every renovated dwelling receives the same air-water heat-pump, heat-pump DHW, 45°C radiator, and HRV package. | Feasibility, sizing, noise, electrical connection, and dwelling-specific emitter constraints are not assessed. |
| Cooling ownership proxy | Integrated-air-conditioning ownership is used as active cooling, independently of renovation state. | Usage intensity and portable cooling are not measured; reversible capability does not imply operation. |
| PV small-stock proxy | Brussels and Wallonia require documented source-derived scaling to obtain end-year ≤10 kW/kVA regional values. | These inputs are reproducible proxies, not direct end-year microdata extracts for the full regions. |
| PV projection | A linear 2024-2030 trajectory is extended to 2050 and capped by heterogeneous official rooftop-potential estimates and eligible stock. | It is a scenario allocation, not a forecast; roof suitability, ownership, grid constraints, degradation, replacement, and electricity output are outside scope. |
| Apartment shared PV | Current apartment PV is zero; 2050 participation is 25% of the house rate and one regional mean system is divided by average R4 dwellings per building. | Real condominium participation, roof area, metering, and building-size distributions are simplified. |

## Reproducibility and validation

The core scripts can be run from any working directory because they resolve
their default paths relative to the repository root:

~~~bash
python scripts/size_type_composition.py
python scripts/construction_period.py
python scripts/archetype_matrix.py --tabula-pdf /path/to/BE_TABULA_ScientificReport_VITO.pdf
node scripts/renovation_scenarios_2050.mjs
node scripts/technology_layer_2050.mjs
~~~

All three scripts require pandas; Excel input also requires openpyxl; PDF
verification requires pdfplumber. The TABULA PDF is required for the final
Python command because the script verifies Tables 9, 10, and 19 before writing
its two matrices. The two JavaScript generators require
`@oai/artifact-tool`; when using the Codex bundled runtime, set
`CODEX_NODE_MODULES` to its `node_modules` directory.

The implemented checks include:

- expected Statbel 2025 national totals in the size/type builder;
- required columns, unique keys, complete 5 × 5 type-period coverage, and
  non-missing physical parameters in the archetype builder;
- a national-share sum of one and a matrix dwelling total equal to the Statbel
  dwelling total; and
- TABULA table verification against a local report copy before the matrix
  builder writes output;
- 450 renovation-state rows with no obsolete `heat_recovery_efficiency`
  column;
- exact technology row counts and unique composite keys;
- valid booleans and all shares within [0,1];
- variant shares summing to one and variant dwellings reconstructing every
  source state, including retained zero-dwelling variants;
- regional R1-R4 totals preserved for each renovation scenario and, for
  cooling, each cooling case;
- a raw Walloon heating-share total of 0.997 and normalized modeling total of
  one;
- current apartment PV equal to zero and projected apartment participation
  equal to 25% of the corresponding house rate;
- PV assignment capacity reconciled with the 2024 proxy and capped 2050
  regional trajectory; and
- temporary-write validation followed by atomic replacement and SHA-256 output
  hashes.

The technology generator has been run twice against the committed inputs and
produced byte-identical heating, cooling, PV, and regional PV-assignment files.
The physical and national stock matrices reproduce in memory with 25
archetypes, a national-share sum of 1.0, and 5,827,823 modeled dwellings. TABULA
PDF verification remains pending because the authoritative PDF is not stored
locally.

## Information needed to complete thesis citations

Please provide, where available:

1. Preferred bibliographic citations, landing-page URLs, download/access dates,
   and licence details for both Statbel workbooks.
2. The authoritative TABULA/VITO report version and publication details, plus a
   decision on whether its PDF should be stored locally for reproducible
   verification.
3. A thesis justification or supporting source for using the 2021
   construction-period distribution as a proxy for the 2025 stock.
4. Confirmation that the national pro-rata treatment of R5/R6 dwellings is the
   intended scope choice, rather than a temporary convenience.
5. The location of any companion model that calculates energy, emissions, or
   technology performance, if this guide should eventually link to it.
