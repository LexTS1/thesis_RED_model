# Gate 5 structural-sensitivity screen

## Purpose and boundary

This workflow measures how much a small set of declared modelling assumptions changes
heating and cooling results. It is an **epistemic sensitivity screen**: each scenario is
an alternative, defensible model assumption. The scenarios are not random draws and do
not have probabilities.

The screen is intentionally separate from the authoritative central Monte Carlo run.
It reuses the validated thermal model, behavioural wrapper, stock weights, weather
members, seed generator, streaming runner and authenticated post-processor, but it has
its own frozen design and stopping rule in
`thermal_model/monte_carlo/sensitivity.py`.

The separation prevents three common errors:

1. a structural assumption cannot be hidden inside an occupant seed;
2. structural effects are evaluated on exact common random numbers rather than on
   unrelated households or weather years; and
3. a reduced weather screen can never be mistaken for the complete production run.

No screen simulation was launched when this workflow was implemented. Preparation,
execution and evaluation are explicit user actions.

## Qualification is deliberately not production `PASS`

The screen includes every one of the 75 weighted physical archetype/renovation cells,
but only six of the eighteen weather years per RCP. Its seed count is governed by a
separate paired-delta stability rule rather than the production occupant-convergence
evidence. The shared stock runner is therefore called with:

- `require_full_stock=True`, because all 75 authoritative stock cells and weights are
  required; and
- `require_convergence_evidence=False`, because the structural screen has its own
  declared stability decision.

Consequently, the shared execution must finish as `WORKFLOW_CHECK_ONLY`. The wrapper
adds the purpose `STRUCTURAL_SENSITIVITY_SCREEN` and the permanent qualification
`NOT_A_PRODUCTION_PASS`. It never changes the production qualification logic in
`runner.py` and never promotes a reduced weather design to `PASS`.

The three coverage fields must be reported separately:

| Axis | Frozen coverage label |
|---|---|
| Building stock | `AUTHORITATIVE_2050_WEIGHTS` |
| Weather | `STRATIFIED_6_OF_18_PER_RCP` |
| Occupants | `N40_OF_160`, `N80_OF_160` or `N160_OF_160` |

“Full stock” therefore means full coverage of the weighted building cells. It does not
mean full weather or occupant coverage.

## Frozen experimental design

### Physical stock

Every stage contains all 75 combinations of TABULA dwelling archetype and physical
renovation state used by the authoritative 2050 stock-weight contract. The screen does
not average those cells equally when producing stock summaries: the shared aggregation
layer retains the authoritative regional and renovation-state dwelling weights.

### Structural scenarios

The local screen contains `central` plus all five registered alternatives:

| Scenario ID | Axis | Declared change |
|---|---|---|
| `central` | central | authoritative `thermal_assumptions.csv` |
| `infiltration_half` | infiltration | multiply all mutually constrained normal-pressure leakage fields by 0.5 |
| `infiltration_one_and_half` | infiltration | multiply those leakage fields by 1.5 |
| `mass_light` | thermal mass | ISO light class: 110,000 J/(m² K), effective mass-area ratio 2.5 |
| `mass_heavy` | thermal mass | ISO heavy class: 260,000 J/(m² K), effective mass-area ratio 3.0 |
| `shading_unshaded` | fixed shading | set the fixed vertical shading factor to 1.0 |

The central case is intentionally repeated inside this screen. A structural result is
always calculated against that local central run with the same physical cell, RCP,
weather member and occupant seed. Overlap with a separately executed authoritative
central run is a reproducibility check only; results are not spliced across designs.

### Weather stratification

Weather selection was frozen before any structural-model result was examined. Selection
used only the 18 RCP4.5 weather forcings. The six observed PVGIS years were then copied
unchanged across RCP2.6, RCP4.5 and RCP8.5, giving 18 weather members in total.

For hourly outdoor temperature $T_t$, the selection metrics are:

$$
HDD_{20}=\frac{1}{24}\sum_t\max(20-T_t,0)
$$

$$
CDD_{26}=\frac{1}{24}\sum_t\max(T_t-26,0)
$$

For the already transposed facade irradiance, the orientation-sum metric is:

$$
I_{4f}=\sum_t\left(I_{S,t}+I_{E,t}+I_{W,t}+I_{N,t}\right)
$$

This last quantity is a selection index summed over four differently oriented planes;
it is not the irradiation of one physical surface.

| PVGIS year | Frozen role | RCP4.5 selection value |
|---:|---|---:|
| 2010 | maximum $HDD_{20}$ | 3,770.134 K·day |
| 2013 | minimum $I_{4f}$ | 2,508,110.579 Wh/m² orientation-sum |
| 2015 | four-metric medoid | standardized distance 0.753509 |
| 2019 | maximum hourly outdoor temperature | 38.181 °C |
| 2020 | maximum $CDD_{26}$ | 34.043 K·day |
| 2022 | maximum $I_{4f}$ | 2,930,206.157 Wh/m² orientation-sum |

The medoid uses $HDD_{20}$, $CDD_{26}$, maximum hourly outdoor temperature and
$I_{4f}$. Each metric is standardized across the 18 RCP4.5 years using the population
standard deviation (`ddof=0`). The selected year has the smallest Euclidean distance to
the four-dimensional standardized centroid.

During `prepare`, the workflow reloads all 18 RCP4.5 candidates, recomputes these four
metrics and the medoid, verifies the six frozen selections and values, and stores every
candidate's weather-contract and forcing checksums in the screen contract. Thus a change
to an unselected candidate cannot silently invalidate the original extremum or medoid
claim.

The six members are **strata**, not a probability sample. They are not multiplied by
weather probabilities and must not be used to claim an exhaustive weather distribution.

### Occupant seeds and common random numbers

All stages use prefixes of one deterministic seed bank:

- master seed: `20250808`;
- prospective checkpoints: 10, 20, 40, 80 and 160;
- allowed persisted stages: 40, 80 and 160 only;
- first n=40 seed: `1203443498`;
- fortieth seed: `1182233951`; and
- ordered n=40 prefix SHA-256:
  `d9c1d0f471fb97f0efab0c100463285cd55e4de7718f288f7be26835b3adb0fe`.

The seed order is part of the contract because every checkpoint is a nested prefix. It
must not be sorted by numeric seed value.

Exactly the same seed set is used for every weather member and every structural
scenario. Thus a structural delta is:

$$
\Delta Y_{a,s,w,j,m}
=
Y_{a,s,w,j,m}-Y_{a,s,w,j,central}
$$

where:

- $a$ is the physical archetype;
- $s$ is the renovation state;
- $w$ is the weather member;
- $j$ is the occupant seed; and
- $m$ is the structural scenario.

Pairing removes unrelated occupant and weather draws from the contrast. It does not
remove interaction: an assumption can still have a different effect under a different
weather member, household or physical cell.

## Run counts and partitions

The initial n=40 stage contains:

$$
75\ \text{cells}
\times18\ \text{weather members}
\times40\ \text{seeds}
\times6\ \text{model scenarios}
=324{,}000\ \text{dwelling-years}
$$

The runner creates one partition per weather member and model scenario:

$$
18\times6=108\ \text{partitions}
$$

Each n=40 partition contains $75\times40=3{,}000$ dwelling-year runs. The n=80
and n=160 stage totals, if prospectively authorized by the rule below, are 648,000 and
1,296,000 dwelling-years respectively.

The full blind cross-product over all 18 weather years per RCP would be much larger. The
screen is the declared compromise: all physics cells and all structural endpoints are
retained while weather is deliberately stratified.

## Prospective paired-delta stability rule

### Representative decision panel

Running all 75 physical cells is necessary for stock reporting. The decision about the
number of occupant seeds is made on the same low/medium/high deterministic-demand panel
used by the occupant convergence experiment:

| Demand role | Archetype | Renovation state |
|---|---|---|
| low | `BE_TABULA_14` | `TABULA_advanced_A_proxy` |
| medium | `BE_TABULA_13` | `TABULA_standard_B_proxy` |
| high | `BE_TABULA_11` | `TABULA_existing` |

For each panel cell, RCP and non-central structural scenario, the evaluator pools the
six selected weather members and the active seed prefix. It evaluates four paired-delta
metrics:

| Metric | Practical absolute floor |
|---|---:|
| heating intensity | 1 kWh/(m²·year) |
| cooling intensity | 1 kWh/(m²·year) |
| peak heating power | 100 W |
| peak cooling power | 100 W |

For every metric it calculates the mean, median and empirical 95th percentile. Let
$S_n$ be one statistic at seed prefix $n$, and let $f$ be its practical floor. Change
from the preceding declared checkpoint is:

$$
d_n=\frac{|S_n-S_{n^-}|}{\max(|S_n|,f)}
$$

where $n^-$ is the previous checkpoint. One criterion passes when:

$$
d_n\leq0.05
$$

There is an additional direction guard for the median. If both adjacent median effects
are larger than the metric floor in magnitude and their signs differ, that criterion
fails even if its normalized change would otherwise pass.

An expansion passes only when every statistic for every declared metric, structural
scenario, RCP and representative physical cell passes. Stability requires two
consecutive passing expansions. Therefore:

- n=10 establishes the first estimates and cannot pass;
- n=20 can be the first passing expansion;
- n=40 is the earliest possible stable decision;
- n=80 is allowed only if the authenticated n=40 result explicitly requests it; and
- n=160 is allowed only if the authenticated n=80 result explicitly requests it.

The tolerance, floors, statistics, panel and allowed checkpoints are frozen in the
contract before simulation. They are not adjusted after inspecting results.

## Extension-safe outcome

Each permitted stage has a distinct default directory:

- `thermal_model/data/monte_carlo/structural_sensitivity_screen_n040`;
- `thermal_model/data/monte_carlo/structural_sensitivity_screen_n080`; and
- `thermal_model/data/monte_carlo/structural_sensitivity_screen_n160`.

An extension never edits the predecessor. Its contract binds the predecessor contract
checksum, decision-summary checksum and seed-independent design-basis checksum. The new
stage must use the same stock, weather, physics, behavioural contracts and structural
scenarios, plus the longer prefix of the same seed bank.

Possible outcomes are:

| Outcome | Meaning |
|---|---|
| `STRUCTURAL_SENSITIVITY_SCREEN_STABLE_AT_N40` | n=20 and n=40 expansions both pass |
| `STRUCTURAL_SENSITIVITY_SCREEN_NOT_STABLE_AT_N40` / `EXTEND_TO_N80` | n=40 is not yet confirmed; a separate n=80 stage is permitted |
| `STRUCTURAL_SENSITIVITY_SCREEN_STABLE_AT_N80` | the allowed n=80 extension supplies the second consecutive pass |
| `STRUCTURAL_SENSITIVITY_SCREEN_NOT_STABLE_AT_N80` / `EXTEND_TO_N160` | a separate n=160 confirmation is permitted |
| `STRUCTURAL_SENSITIVITY_SCREEN_STABLE_AT_N160` | stability is first confirmed at n=160 |
| `TERMINAL_NOT_STABLE_AT_DECLARED_MAXIMUM` | n=160 is reached without two consecutive passes; report non-stabilisation rather than extending ad hoc |

If the maximum does not stabilize, the structural results still describe the completed
declared experiment, but statistics sensitive to seed count must be labelled as such.
No automatic n>160 search is authorized.

## Authenticated artifacts

Preparation writes:

- `streaming_design_contract.json`, owned by the shared stock runner; and
- `structural_sensitivity_contract.json`, which binds the exact screen selection, seed
  prefix, all 18 RCP4.5 selection-candidate checksums, scenario registry, shared design
  checksum and extension lineage.

Execution retains the runner's restartable, checksum-indexed partitions. A completed
screen then uses the bounded-memory post-processor to create, among other outputs,
`paired_model_scenario_deltas.csv`. The evaluator reads only the authenticated paired
deltas needed for the fixed stability panel; it does not read dwelling-hour files.

Evaluation writes:

- `structural_sensitivity_stability.csv`, one auditable row for every checkpoint,
  physical panel cell, RCP, structural scenario, metric and statistic; and
- `structural_sensitivity_summary.json`, the authenticated stage decision and any
  authorized next stage.

The decision summary binds the screen contract, post-processing summary and stability
CSV checksums. `status` is read-only and returns `INVALID` if those links no longer
authenticate.

## Commands

Inspect the untouched initial location:

```bash
python -c "from thermal_model.monte_carlo.runner import main; raise SystemExit(main(['sensitivity', '--status']))"
```

Prepare the n=40 contract without running simulations:

```bash
python -c "from thermal_model.monte_carlo.runner import main; raise SystemExit(main(['sensitivity', '--prepare-only', '--stage', '40', '--workers', '4']))"
```

Run or resume the prepared n=40 stage, then authenticate, post-process and evaluate it:

```bash
python -c "from thermal_model.monte_carlo.runner import main; raise SystemExit(main(['sensitivity', '--stage', '40', '--workers', '4']))"
```

Only if n=40 explicitly returns `EXTEND_TO_N80`, prepare n=80 in its separate default
directory and bind the predecessor:

```bash
python -c "from thermal_model.monte_carlo.runner import main; raise SystemExit(main(['sensitivity', '--prepare-only', '--stage', '80', '--previous-stage-dir', 'thermal_model/data/monte_carlo/structural_sensitivity_screen_n040']))"
```

The equivalent n=160 command must cite the completed n=80 predecessor. After an
extension has been prepared, a normal resume can rely on the predecessor lineage already
stored in its immutable contract.

`--output-dir` may be supplied for an explicit stage-specific destination.
`--chunk-rows` bounds post-processing reads. `--workers` is bounded by the shared runner.
`--prepare-only` and `--status` are mutually exclusive.

## Reporting rules

Report the central result and each paired structural delta separately. Good statements
include “under the unshaded endpoint, the median paired cooling change over the declared
screen was …” or “the infiltration endpoint changed the coincident stock peak by …”.

Do not:

- pool the five structural scenarios into a probability distribution;
- label their range a confidence or prediction interval;
- assign equal or implicit probabilities to structural endpoints;
- treat six selected weather strata as all weather uncertainty;
- average archetypes equally for stock totals; or
- combine the sum of dwelling-specific peaks with the coincident aggregated peak.

The resulting uncertainty statement is conditional: it covers the declared structural
endpoints, six weather strata per RCP, the stabilized or maximum declared occupant-seed
prefix, and the represented 2050 stock. Unmodelled uncertainty remains outside it.
