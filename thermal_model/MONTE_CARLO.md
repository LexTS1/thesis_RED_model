# Gate 5: Monte Carlo wrapper, uncertainty experiment, and stock aggregation

## 1. Purpose and modelling boundary

Gate 5 turns the verified thermal core and the isolated behavioural generator
into a reproducible uncertainty experiment. It does not change the ISO 13790
5R1C equations, recalibrate an archetype, or add HVAC-system performance. The
model still returns **useful ideal space-heating demand** and **useful ideal
sensible-cooling demand**. It does not return fuel use, electricity use, final
energy, costs, emissions, heat-pump load, or installed capacity.

The experiment is called Monte Carlo because the RichardsonPy occupant process
is sampled repeatedly. Its complete structure is more accurately described as
a balanced crossed design:

```text
physical archetype/state
    x climate pathway
    x paired empirical weather year
    x occupant seed
    x declared structural sensitivity scenario
```

These axes retain different interpretations:

- Occupant profiles are stochastic realizations generated from a declared
  pseudo-random seed.
- Weather members are a deterministic empirical ensemble of morphed observed
  years, not independent random climate-model draws.
- Archetypes and renovation states are a weighted representation of the stock,
  not random building draws.
- Structural scenarios are deliberate sensitivity cases, not samples from
  probability distributions.

Consequently, a Gate-5 interval describes variation over the dimensions that
were explicitly run. It is not a complete prediction interval for Belgian
residential energy demand.

## 2. Component architecture

Gate 5 is an orchestration layer around completed components:

```text
Belgian stock layer                       climate layer
  archetype geometry                       54-member manifest
  physical renovation state                horizontal hourly member
  regional/state dwelling weight           on-demand facade adapter
              |                                      |
              +------------------+-------------------+
                                 |
                        Monte Carlo wrapper
                     run manifest and provenance
                       /                    \
              behavioural wrapper       scenario overlay
              occupancy, gains,          central or declared
              setpoints                  structural sensitivity
                       \                    /
                         deterministic 5R1C core
                                 |
                  hourly result + annual diagnostics
                                 |
            distributions, variance audit, and weighted stock totals
```

The dependency direction is one-way. `thermal_model/core.py` has no climate,
RichardsonPy, Monte Carlo, or stock-weighting logic. The behaviour layer still
passes only total sensible internal gains and setpoints to the core. The stock
weights are applied after dwelling simulation and cannot alter the physics of a
representative dwelling.

## 3. Stable simulation interface

### 3.1 Public call

The stable conceptual interface is:

```python
simulate(
    archetype_state,
    weather_member,
    occupant_seed,
    model_scenario,
) -> hourly_results, diagnostics
```

`archetype_state` may be the validated typed object or a complete field mapping;
`weather_member` may be a validated typed object or an authoritative manifest
ID; and `model_scenario` may be a registered object or its ID. The concrete
`MonteCarloResult` contains `.hourly` and `.diagnostics` rather than relying on
an unlabelled tuple. `.hourly` is a new DataFrame and `.diagnostics` is an
immutable annual/provenance record.

The function is side-effect-free with respect to project data and calling-code
random state:

- It does not write an output file or edit an input object.
- It never edits `thermal_assumptions.csv` or a stock/climate source file.
- The RichardsonPy adapter saves and restores Python and NumPy global random
  states and performs generation under its existing process lock.
- An identical validated input and seed must return an identical profile,
  thermal result, and diagnostic record.
- Batch persistence is owned by the runner, outside `simulate`.

Fail-fast validation remains in force. Unknown scenario identifiers, missing
weather members, checksum disagreement, incomplete calendars, misaligned
timestamps, an invalid seed, or a physically invalid scenario overlay raise an
exception. Inputs are not clipped, filled, silently reordered, or repaired.

### 3.2 `archetype_state`

The archetype-state payload identifies one of the 75 distinct physical cells:

- 25 `archetype_id` values;
- five dwelling types and five construction periods represented by those
  archetypes; and
- three physical states: `TABULA_existing`,
  `TABULA_standard_B_proxy`, and `TABULA_advanced_A_proxy`.

It contains the base geometry and the state-specific envelope, infiltration,
ventilation, HRV, and bypass fields required by
`assemble_archetype_state()`. The 5R1C preprocessor then derives areas,
capacitance, solar properties, and network conductances under the validated
thermal assumptions contract. The stock-layer screening field
`transmission_heat_loss_H_tr_W_K` remains a diagnostic only and is not used as
the hourly model's transmission conductance.

Region is deliberately absent from the physical calculation because the three
regional copies of a given archetype/state have identical physics. Region is
reintroduced through the 225 stock-weight rows during aggregation. This avoids
running the same dwelling three times while preserving regional totals.

### 3.3 `weather_member`

The weather contract carries:

- `member_id`;
- RCP pathway (`rcp_2_6`, `rcp_4_5`, or `rcp_8_5`);
- observed PVGIS anchor year;
- the 2041–2060 climate target identifier;
- 8,760 or 8,784 ordered UTC timestamps;
- outdoor dry-bulb temperature;
- horizontal beam, diffuse, and total irradiance for the behaviour layer;
- north, east, south, and west facade irradiance for the thermal layer; and
- source/member/metadata checksums.

The authoritative manifest is:

```text
climate/data/processed/ensemble_2050/ensemble_2050_manifest.csv
```

It contains 54 members: three pathways by the same 18 PVGIS anchor years
(2006–2023). Years 2008, 2012, 2016, and 2020 contain 8,784 hours; leap days
are retained.

The canonical member files contain horizontal irradiance. The climate-layer
adapter appends facade irradiance on demand using the same pathway's monthly
solar factor and the aligned PVGIS facade templates. The Monte Carlo layer does
not implement a second transposition. The behaviour wrapper receives the
horizontal beam and diffuse series from that same member, so lighting and
window solar gains refer to one weather realization.

The behaviour wrapper converts UTC forcing to its documented periodic fixed
CET clock (UTC+1, no daylight-saving transition), generates the profile, and
maps the output back to the exact UTC member index. Therefore weather,
behaviour, setpoints, and thermal results retain exact timestamp equality.

### 3.4 `occupant_seed`

`occupant_seed` is an unsigned 32-bit integer. Gate 4 splits it
deterministically into an occupant-count stream and a RichardsonPy stream.
Occupant count is conditioned on the dwelling's SFH/MFH class using the frozen
Belgian Census distribution; it is not sampled independently of dwelling type.

The seed identifies occupant variability only. It must never choose weather,
renovation state, stock weight, thermal mass, shading, infiltration, or another
structural assumption.

### 3.5 `model_scenario`

`model_scenario` selects an immutable overlay from a closed registry. The
authoritative central assumptions remain in `thermal_assumptions.csv`; each
overlay is applied to an in-memory copy of either the assumption contract or
the raw archetype-state fields before preprocessing. It does not edit the CSV
or mutate the supplied archetype.

The declared Gate-5 registry is:

| Scenario identifier | Axis | Effective change |
|---|---|---|
| `central` | none | Unchanged validated central contract |
| `mass_light` | thermal mass | $C_m/A_f=110{,}000$ J/(m² K), $A_m/A_f=2.5$ |
| `mass_heavy` | thermal mass | $C_m/A_f=260{,}000$ J/(m² K), $A_m/A_f=3.0$ |
| `shading_unshaded` | fixed facade shading | $F_{sh}=1.0$ instead of the central 0.6 |
| `infiltration_half` | infiltration | Normal-pressure infiltration airflow multiplied by 0.5 |
| `infiltration_one_and_half` | infiltration | Normal-pressure infiltration airflow multiplied by 1.5 |

The mass overlays recalculate $C_m$, $A_m$, $H_{tr,ms}$, and $H_{tr,em}$ and
rerun the prepared-archetype validity checks. The infiltration overlays scale
`q50_m3_h`, `n50_h_1`, `infiltration_airflow_normal_m3_h`, and
`infiltration_ach_normal_h_1` together so the raw leakage identities remain
valid. Only the infiltration term in $H_{ve}$ is thereby changed; heat recovery
remains restricted to the mechanical-ventilation stream. The shading overlay
replaces the fixed factor once; it does not add dynamic blinds or a second
shading multiplier.

These values reproduce a deliberately small subset of the predeclared Gate-3
sensitivity screen. They are not assigned probabilities and must be reported
separately from the central run. Adding a scenario requires a named registry
entry, explicit parameters, tests, and a new scenario-contract checksum.

### 3.6 Provenance identity

Each run receives a deterministic `run_id` derived from its complete input
contract. The identifier binds a canonical SHA-256 of every validated
archetype-state field rather than relying only on the four human-readable stock
labels. Changing a U-value, area, airflow, or other physical field while
retaining the same labels therefore creates a different run. Diagnostics
preserve, at minimum:

- archetype, dwelling type, construction period, and physical-state IDs plus
  the complete archetype-state checksum;
- RCP pathway, weather-member ID, PVGIS anchor year, and climate target;
- occupant seed, sampled occupant count, and derived RichardsonPy seed;
- model-scenario ID and structural axis;
- SHA-256 of `thermal_assumptions.csv`;
- effective model-scenario checksum;
- SHA-256 of the behavioural assumptions and occupant distribution;
- climate member and metadata checksums; and
- checksum of the complete weather/behaviour forcing where applicable.

The effective scenario checksum binds the central contract checksum to a
canonical serialization of the selected overlay. It distinguishes a changed
sensitivity definition from a changed central assumptions file.

## 4. Simulation outputs and definitions

### 4.1 Hourly results

Every accepted run returns one complete UTC year with:

| Column | Unit | Meaning |
|---|---:|---|
| `timestamp_utc` | UTC | Exact weather/schedule timestamp |
| `T_out_C` | °C | Outdoor dry-bulb temperature |
| `theta_air_free_running_C` | °C | Air temperature under the no-HVAC trial for that hour |
| `theta_air_C` | °C | Final controlled indoor-air temperature |
| `theta_surface_C` | °C | Final mean internal-surface temperature |
| `theta_mass_C` | °C | Final mean thermal-mass temperature |
| `theta_operative_C` | °C | $0.3\theta_{air}+0.7\theta_s$ |
| `Phi_internal_W` | W | Total sensible internal gain supplied by Gate 4 |
| `Phi_solar_W` | W | Total transmitted window solar gain |
| `heating_demand_W` | W | Non-negative ideal useful heating power |
| `cooling_demand_W` | W | Non-negative ideal useful sensible-cooling power |
| `theta_set_heat_C` | °C | Active hourly heating setpoint |
| `theta_set_cool_C` | °C | Active hourly cooling setpoint |
| `H_ve_W_K` | W/K | Effective infiltration plus recoverable-ventilation conductance |
| `hrv_bypass_active` | boolean | Whether the documented summer-bypass rule is active |

`theta_air_free_running_C` is diagnostic. It is the no-load trial used by the
ideal-load procedure, not the temperature that an ideally conditioned dwelling
experiences after control.

### 4.2 Annual energy, intensity, and peaks

Hourly powers are mean watts over one-hour intervals. Annual useful energy is:

$$
E_H=\frac{1}{1000}\sum_t\Phi_{H,t}
$$

$$
E_C=\frac{1}{1000}\sum_t\Phi_{C,t}
$$

where $E_H$ and $E_C$ are in kWh. Intensities use conditioned floor area:

$$
e_H=\frac{E_H}{A_f},\qquad e_C=\frac{E_C}{A_f}
$$

Peak powers are the maximum hourly heating and cooling values. Full-load
equivalent hours are:

$$
t_{FL,H}=\frac{E_H}{\Phi_{H,peak}/1000}
$$

$$
t_{FL,C}=\frac{E_C}{\Phi_{C,peak}/1000}
$$

A zero peak produces zero full-load hours rather than a division by zero.

### 4.3 Control and numerical diagnostics

Annual diagnostics additionally include:

- hours scheduled at every distinct heating and cooling setpoint;
- hours with ideal heating and hours with ideal cooling;
- ISO no-load trial hours above the cooling setpoint, defined by
  `theta_air_free_running_C > theta_set_cool_C + 1e-9 K`;
- the analogous no-load trial count below the heating setpoint;
- annual internal sensible gains, household electricity, and HRV-bypass hours;
- maximum absolute energy-balance residual in watts;
- periodic warm-up cycles required to meet the 0.01 K start/end mass-node
  criterion; and
- all identifiers and checksums listed in Section 3.6.

Setpoint-hour counts describe the schedule, not necessarily hours of active
heating or cooling. The ISO free-running value is the no-load trial at each
controlled timestep, starting from the mass state produced by preceding
controlled hours. Its threshold count is therefore a pre-control diagnostic,
not a separate, periodically converged, all-year no-HVAC overheating model and
not an occupied discomfort metric. The annual fields are therefore named
`iso_no_load_trial_above_cooling_setpoint_hours` and
`iso_no_load_trial_below_heating_setpoint_hours`; Gate 5 does not label these
counts as an annual free-running building trajectory.

The lower-level core contract reconciles annual energies, intensities, and
peaks against the returned hourly table before Gate 5 adds its diagnostics. Any
disagreement fails the run.

## 5. Uncertainty structure

### 5.1 Weather uncertainty represented here

Within each RCP, the 18 members represent the observed interannual sequences of
2006–2023 after one pathway-specific monthly morph. The same anchor years are
paired across all three RCPs. The experiment must therefore:

- summarize each RCP separately;
- retain the 18-year empirical distribution within each pathway;
- pair equal anchor years for between-pathway contrasts; and
- avoid treating all 54 members as independent or equally probable climate
  futures.

The ensemble samples observed-year timing and the three pathway deltas. It does
not sample GCM choice, RCM choice, model member, future changes in sub-monthly
variance, future wind, or other structural climate uncertainty.

### 5.2 Occupant variability

Occupant variability includes the household count draw, Richardson active
occupancy, appliance timing, lighting timing, internal gains, and the associated
18/20°C heating schedule. The 3,500 kWh/dwelling annual electricity target and
26°C cooling setpoint remain fixed Gate-4 assumptions; the seed changes profile
composition and timing, not those central values.

### 5.3 Building-stock heterogeneity

Building heterogeneity is represented explicitly by archetype, construction
period, dwelling type, physical renovation state, and the regional 2050
dwelling weights. These are finite stock categories, not an additional random
seed.

### 5.4 Structural or epistemic uncertainty

Thermal mass, shading, and infiltration are kept as separately labelled
sensitivity scenarios. Their spread must not be pooled with occupant/weather
variation or presented as a probability-weighted interval. The central case is
always identifiable.

## 6. Balanced run design

### 6.1 Full factorial rule

For every selected physical archetype/state, RCP, and model scenario, run the
same Cartesian product of weather members and occupant seeds. A valid balanced
manifest has exactly one row for every requested combination and no duplicate
`run_id`. The manifest stores both the uint32 `occupant_seed` and its
one-based `occupant_seed_rank`, so a nested prefix remains recoverable after
sorting or restarting a run.

For all physical cells and all weather members, the central scenario requires:

$$
75\times54\times n_{seed}=4050n_{seed}\text{ dwelling-years}
$$

Running all six declared model scenarios would require:

$$
75\times54\times n_{seed}\times6=24300n_{seed}
\text{ dwelling-years}
$$

This count is stated before execution so computational subsampling remains
visible. A pilot or representative sample must be labelled as such and may not
be reported as the completed stock experiment.

### 6.2 Common random numbers

The identical ordered seed set is used:

- across weather members;
- across the three physical renovation states;
- across the structural scenarios; and
- where applicable, across climate pathways.

For a fixed archetype, weather member, and seed, renovation/scenario contrasts
therefore share the same behavioural boundary conditions. Differences are
attributable to the changed envelope or declared structural overlay rather than
an unrelated occupant draw. The same weather anchor year is retained across
RCP contrasts.

This is a variance-reduction design for comparisons, not evidence that real
households in different dwellings have identical behaviour. For stock peaks,
seed profiles are averaged within each archetype/state before applying millions
of dwelling weights, as described in Section 9.

### 6.3 Reusable behavioural calculations

A Richardson profile depends on dwelling class, weather member, and seed, but
not on envelope renovation state or thermal-mass/shading/infiltration overlay.
The runner may cache one validated profile for an identical behavioural key and
reuse it across the corresponding thermal cases. This is a performance
optimization only: cached and uncached results must be bit-for-bit identical,
and the stable `simulate` call remains free of persistent side effects.

### 6.4 Failure handling

The runner must not silently omit a failed cell and continue with an apparently
balanced summary. A failed run records its manifest identity and exception;
distribution, variance, and stock aggregation functions reject incomplete
groups unless the output is explicitly labelled an incomplete diagnostic.

The production path implements this requirement with
`execute_streaming_stock_design(...)`. It partitions work by one weather member
and one structural scenario. One lock-owning coordinator assigns at most one
worker to each partition, so separate processes never write the same checkpoint
path. It then commits progress after each **complete
occupant seed across every requested physical cell**. Two alternating
checkpoint slots hold the cumulative regional/national hourly arrays and the
completed diagnostic rows. The small `progress.json` pointer is replaced only
after both inactive-slot files have been written. A process interruption during
a write therefore leaves the preceding slot checksum-verifiable and usable.

On restart, the runner verifies the design checksum, the active checkpoint
checksums, the exact nested seed prefix, and the exact expected run-ID set. It
then resumes at the next seed; a partly completed seed is rerun. A caught
mid-seed exception is written atomically to `last_failure.json` with the run ID,
seed rank, physical cell, and exception before it is re-raised. The record is
marked `RECOVERED` after that seed is subsequently committed. A completed
partition is skipped only after all five partition artifacts and both its
manifest and diagnostics run-ID sets pass checksum/completeness checks.
The read-only `stock --status` command reports observational pointer counts;
the execution path itself revalidates slot checksums before resuming.
Before a long launch, `stock --prepare-only` runs the complete preflight,
persists the exact design contract and authenticated convergence evidence, and
returns the state, weather, seed, scenario, partition, and exact run counts. It
re-reads and hashes the persisted contract before returning and never calls a
partition worker. A subsequent execution must reproduce the same design hash or
it is rejected.

The production-default call also requires an authenticated seed-convergence
artifact:

```python
from thermal_model.monte_carlo import (
    execute_streaming_stock_design,
    load_convergence_continuation_selection,
)

selection = load_convergence_continuation_selection()

execute_streaming_stock_design(
    states,
    weather_members,
    selection.occupant_seeds,
    model_scenarios,
    convergence_results_path=selection.convergence_results_path,
    convergence_results_sha256=selection.convergence_results_sha256,
    convergence_rule=selection.convergence_rule,
    max_workers=4,
)
```

The selection loader refuses any status other than `CONVERGED` and
reconciles the execution-contract checksum, first converged checkpoint, exact
ordered seed prefix, authorized rule, evidence path, and evidence SHA-256.
Thus the production arguments cannot be copied independently from different
runs.

The runner first checks the CSV byte checksum, but it does not trust the stored
pass flags. It independently reconstructs every relative change, metric-level
decision, consecutive-pass count, group decision, and panel decision from the
persisted values. It requires the exact declared checkpoint sequence and the
complete metric-by-statistic cross-product at every checkpoint. The selected
seed count must equal the **first** independently reconstructed panel-wide
converged checkpoint and its SHA-256 must match the exact ordered seed prefix.
The authenticated CSV is then copied byte-for-byte to the production output
root.

For an authoritative full-stock run, the evidence must use the original
5/10/20/40/80 rule, its authorized prospective n=160 confirmation, or the
authorized n=320/n=640 continuation. All use the unchanged 2% tolerance and
two-consecutive-expansion requirement,
at least three representative physical cells, every selected RCP, and the same
weather panel within each RCP. Its archetype-state, structural-scenario,
weather-forcing, thermal, behavioural, occupant-distribution, and model
contract checksums must match the current execution inputs. This prevents a
validly formatted but stale convergence table from being reused after an input
contract changes. A deliberately altered convergence rule is permitted only in
a partial workflow or sensitivity check and cannot qualify a production run.

Missing, non-converged, inconsistent, reordered, stale, or incorrectly sized
evidence stops the run before simulation. `require_convergence_evidence=False`
exists only for tests and workflow checks. A full-stock run using that bypass
is labelled `WORKFLOW_CHECK_ONLY` and its convergence status is
`NOT_VERIFIED_BY_RUNNER`. Any call with `require_full_stock=False` is labelled
`PARTIAL_STOCK_WORKFLOW`, even if its seed-convergence evidence is valid. Only
authoritative full-stock coverage plus verified convergence can produce the
top-level status `PASS`.

## 7. Occupant-seed convergence experiment

The number of occupant seeds is selected by convergence, not convenience.
Seeds form one deterministic ordered sequence, and the predeclared checkpoints
are 5, 10, 20, 40, and 80 seeds. Each larger checkpoint uses the complete
prefix of the smaller checkpoint. This nested design ensures that movement
between checkpoints comes only from added households.

The default bank is generated once from master seed `20250808` with NumPy
`SeedSequence.spawn`. Every emitted occupant seed is a unique uint32 value.
Changing the master seed creates a different declared experiment and must be
recorded; the generated occupant-seed values, rather than only the master seed,
remain in the manifest.

At each checkpoint, track at least:

- mean, median, and 95th empirical percentile of annual heating intensity;
- mean, median, and 95th empirical percentile of annual cooling intensity;
- mean, median, and 95th empirical percentile of individual peak heating; and
- mean, median, and 95th empirical percentile of individual peak cooling.

For statistic $q$ at consecutive nested checkpoints $n_{j-1}$ and $n_j$, the
relative stabilization measure is:

$$
d_q(n_j)=
\frac{|q_{n_j}-q_{n_{j-1}}|}
{\max(|q_{n_j}|,\epsilon_q)}
$$

where $\epsilon_q$ is the predeclared scale floor for a near-zero metric. The
relative tolerance is 2%. The scale floor is 1 kWh/m² for annual intensity and
100 W for peak power. The selected seed count is the first checkpoint for which
every required statistic passes at two consecutive expansions. Annual-energy
and peak criteria are evaluated separately so a stable mean cannot conceal an
unstable upper tail.

The convergence table preserves the checkpoint and previous checkpoint, metric,
statistic, current and previous values, absolute floor, normalized change,
tolerance, individual pass flag, all-statistics pass flag, consecutive-pass
count, and per-group `converged_at_checkpoint` flag. Panel-level fields then
require every representative group to pass at the same checkpoint and apply
the same two-expansion rule. The selected count is the first checkpoint with
`panel_converged_at_checkpoint=true`. If the largest checkpoint does not pass,
the result is `NOT_CONVERGED`; the runner does not choose that number and call
it sufficient.

Every row also carries three seed-provenance fields:

- `occupant_seed_bank_count`: length of the complete ordered bank used in the
  convergence experiment;
- `occupant_seed_bank_sha256`: SHA-256 of that complete ordered uint32 list;
  and
- `occupant_seed_prefix_sha256`: SHA-256 of the exact ordered prefix at the
  row's checkpoint.

The prefix hash, rather than a set comparison or a sorted hash, binds a
production design to the convergence decision. This prevents the same seed
values in a different order from silently changing every nested checkpoint.

`evaluate_seed_convergence(...)` also carries the provenance already present
in the run diagnostics into its output. Global fields bind the Gate-5 model,
central thermal assumptions, behavioural assumptions, and occupant-count
distribution. Per-group fields bind the complete archetype-state physics and
registered structural scenario. A canonical `weather_panel_sha256` binds the
member IDs plus weather-contract and complete forcing checksums within each
RCP. The production runner compares these fields with the selected execution
inputs; users do not append checksum columns manually.

Convergence should be assessed over representative low-, medium-, and
high-demand cells and all selected RCPs. A seed count demonstrated only for one
archetype is not automatically valid for every tail statistic in the full
stock. The final thesis reports the experiment and its stopping rule alongside
the selected count.

### 7.1 Frozen representative panel

The production convergence panel is selected before inspecting stochastic
results from the deterministic Gate-3 validation table:

| Demand stratum | Physical cell | Description | Deterministic heating |
|---|---|---|---:|
| Low | `BE_TABULA_14` / `TABULA_advanced_A_proxy` | Enclosed apartment, 1971–1990 | 7.635 kWh/m² |
| Medium | `BE_TABULA_13` / `TABULA_standard_B_proxy` | Terraced house, 1971–1990 | 67.922 kWh/m² |
| High | `BE_TABULA_11` / `TABULA_existing` | Detached house, 1971–1990 | 202.678 kWh/m² |

These cells represent the minimum, exact median, and approximately 90th
percentile deterministic heating levels. They span all three renovation states,
both SFH/MFH behavioural classes, and enclosed/terraced/detached exposure while
holding the 1971–1990 construction period fixed. Each has positive 2050 stock
weight and is within its predeclared TABULA validation band. The selection is
bound to the deterministic validation CSV checksum; changing that source stops
the experiment until the panel is reviewed again.

### 7.2 Restartable adaptive execution

Run or resume the convergence experiment with:

```bash
python3 -m thermal_model.monte_carlo convergence --workers 4
```

The executor advances all 54 weather members to a common nested checkpoint,
evaluates the panel, and stops at the first verified checkpoint. If necessary,
it continues through 5, 10, 20, 40, and 80 seeds. It does not run a later
checkpoint merely because an earlier mean is stable: every declared mean,
median, and p95 annual/peak statistic across every panel group must satisfy the
two-expansion rule.

Work is partitioned by weather member. After one occupant seed has completed
for all three cells in that member, the complete diagnostics prefix is written
to the inactive one of two alternating slot files. The new slot is checked,
hashed, and only then selected by an atomically replaced `progress.json`
pointer. Restart reads only the slot named by that pointer, verifies its SHA-256,
design, weather member, run count, ordered seed prefix, and exact run-ID-to-seed
mapping, and resumes at the next incomplete seed. A slot written before an
interrupted pointer switch is uncommitted and is safely overwritten; a partly
executed seed is rerun. Parallel workers operate on distinct weather
partitions and therefore do not share RichardsonPy random state or output
files.

The execution contract also stores and rechecks the SHA-256 values of the
frozen panel and weather-selection CSVs. One coordinator lock prevents two
processes from writing the same output root. A coordinator exception is
recorded as `FAILED`; rerunning the identical command resumes from the last
authenticated partition prefix rather than accepting or silently repairing a
partial file.

The output root is:

```text
thermal_model/data/monte_carlo/convergence_panel/
  convergence_execution_contract.json
  panel_selection.csv
  weather_selection.csv
  convergence_summary.json
  run_manifest.csv                 # final evaluated prefix; selected if converged
  run_diagnostics.csv              # final evaluated prefix; selected if converged
  convergence_results.csv          # evidence consumed by production
  checkpoints/n005|n010|.../
  partitions/<weather_member_id>/
    run_manifest.csv
    partition_contract.json
    progress.json
    run_diagnostics.slot_a.csv
    run_diagnostics.slot_b.csv
```

`convergence_summary.json` is `CONVERGED` only after the stopping decision and
all final artifact checksums reconcile. `NOT_CONVERGED_AT_N80` does not select
80 seeds by default and therefore cannot authorize the production stock run.

### 7.3 Prospective n=160 confirmation

The completed original experiment evaluated 12,960 dwelling-years and ended
`NOT_CONVERGED_AT_N80`. At n=40, four heating-intensity statistics for the
low-demand advanced-apartment cell exceeded the 2% tolerance. At n=80, all 108
criteria passed, but this was only the first complete-panel pass; the rule
requires two consecutive passing expansions.

The follow-up is therefore a prospectively declared n=160 confirmation, not a
retroactive relaxation. It preserves the original tolerance, absolute floors,
metrics, statistics, representative cells, weather panel, master seed, and
two-pass requirement. Only checkpoint n=160 is appended:

```text
Original checkpoints: 5, 10, 20, 40, 80
Extension checkpoint: 160
Combined rule:        5, 10, 20, 40, 80, 160
```

The original `convergence_panel/` remains immutable. A separate extension
contract records its design, summary, n=80 checkpoint, root-artifact, panel,
weather, model, and seed-prefix checksums. Each weather partition imports the
authenticated first 80 seeds into a new alternating-slot checkpoint, verifies
the reassembled n=80 manifest, diagnostics, and recomputed convergence evidence
against the original, and only then simulates seeds 81–160. Historical
checkpoint statistics are recomputed from raw diagnostics and must reproduce
the original decisions.

Prepare, run/resume, or inspect the extension with:

```bash
python3 -m thermal_model.monte_carlo convergence-extension --prepare-only
python3 -m thermal_model.monte_carlo convergence-extension --workers 4
python3 -m thermal_model.monte_carlo convergence-extension --status
```

The live status command is deliberately observational: its progress histogram
reports atomically committed pointer counts but does not reread and hash every
large active diagnostics slot. Import, resume, checkpoint evaluation, and final
selection do perform the full identity and checksum validation.

Outputs are written separately under:

```text
thermal_model/data/monte_carlo/convergence_panel_n160_extension/
  convergence_extension_contract.json
  convergence_extension_summary.json
  panel_selection.csv
  weather_selection.csv
  run_manifest.csv
  run_diagnostics.csv
  convergence_results.csv
  checkpoints/n160/
  partitions/<weather_member_id>/
```

The executed n=160 experiment ended `NOT_CONVERGED_AT_N160`. Two p95 heating-
intensity criteria for the low-demand advanced-apartment cell exceeded the 2%
tolerance: 2.004% under RCP2.6 and 2.433% under RCP8.5. The exact 160-seed
prefix is therefore not a production selection.

### 7.4 Prospective n=320/n=640 continuation

Because n=160 failed rather than supplying a first new pass, two further
checkpoints were declared together before either was simulated:

```text
Authenticated history: 5, 10, 20, 40, 80, 160
New checkpoints:        320, 640
Combined rule:          5, 10, 20, 40, 80, 160, 320, 640
```

Every numerical criterion remains unchanged: 2% relative tolerance, the same
1 kWh/m² and 100 W scale floors, mean/median/p95 statistics, the same panel and
weather members, and two consecutive complete-panel passes. This is one frozen
continuation, not an adaptive choice after viewing n=320. Consequently:

- n=320 is evaluation-only and can never select a production seed count;
- n=640 is always evaluated, even when n=320 fails; and
- 640 is selected only if every criterion passes at both n=320 and n=640.

The continuation output is a third non-overlapping tree. Its root contract and
per-partition receipts authenticate the terminal n=160 summary, root and n=160
checkpoint artifacts, source design, all 54 partition manifests/progress
pointers/active diagnostics, the exact ordered 160-seed prefix, and the fully
recomputed historical 5--160 evidence. Each new partition imports that prefix
into its own alternating checkpoint slots. Execution then simulates only ranks
161--320 and 321--640, committing after each complete seed. Restart never
resimulates an authenticated source seed and resumes at the next committed
rank.

Prepare, run/resume, or inspect this protocol with:

```bash
python3 -m thermal_model.monte_carlo convergence-continuation --prepare-only
python3 -m thermal_model.monte_carlo convergence-continuation --workers 4
python3 -m thermal_model.monte_carlo convergence-continuation --status
```

The frozen run counts are 25,920 imported dwelling-years, 25,920 new runs for
ranks 161--320, 51,840 new runs for ranks 321--640, and 103,680 total rows at
n=640. `--prepare-only` writes the contract and selection CSVs but launches no
simulation. Status is observational while running; once terminal, it fully
reauthenticates both `checkpoints/n320/` and `checkpoints/n640/`, the root copy,
summary ledgers, historical evidence, and source lineage.

```text
thermal_model/data/monte_carlo/convergence_panel_n640_continuation/
  convergence_continuation_contract.json
  convergence_continuation_summary.json
  panel_selection.csv
  weather_selection.csv
  run_manifest.csv
  run_diagnostics.csv
  convergence_results.csv
  checkpoints/n320/
  checkpoints/n640/
  partitions/<weather_member_id>/
```

Terminal status is `CONVERGED` only for joint n=320/n=640 passes, in which case
the selection loader returns the exact 640-seed prefix. Every other outcome is
`NOT_CONVERGED_AT_N640` with null selection fields and cannot authorize stock
production.

## 8. Distribution summaries and variability attribution

### 8.1 Empirical summaries

For each RCP, archetype/state, and model scenario, report the number of valid
weather years and seeds plus the mean, standard deviation, median, and declared
5th, 25th, 75th, and 95th empirical percentiles of annual heating/cooling
energy, intensity, peak power, and full-load equivalent hours. RCPs and
structural scenarios remain separate grouping columns.

Quantiles are descriptive order statistics over the executed design. They are
not confidence limits, probabilities assigned to RCPs, or a model-error band.

### 8.2 Crossed weather/occupant decomposition

Because every weather member uses the same seed set, weather and occupant
contributions can be audited with an exact two-factor sum-of-squares
decomposition. For one scalar result $y_{ws}$ in a fixed
archetype/state/RCP/model-scenario group, let:

$$
\bar y=\frac{1}{n_wn_s}\sum_w\sum_s y_{ws}
$$

$$
\bar y_w=\frac{1}{n_s}\sum_s y_{ws},\qquad
\bar y_s=\frac{1}{n_w}\sum_w y_{ws}
$$

Then:

$$
SS_W=n_s\sum_w(\bar y_w-\bar y)^2
$$

$$
SS_S=n_w\sum_s(\bar y_s-\bar y)^2
$$

$$
SS_{W\times S}=
\sum_w\sum_s
(y_{ws}-\bar y_w-\bar y_s+\bar y)^2
$$

and:

$$
SS_T=SS_W+SS_S+SS_{W\times S}
$$

The reported shares are each component divided by $SS_T$ when $SS_T>0$.
`weather`, `occupant`, and `weather_x_occupant` are therefore descriptive
contributions within this balanced finite design. With one result per
weather-seed cell, the interaction is not separable from an independent error
term. It also contains genuine coupling such as weather-sensitive lighting and
nonlinear control response. These shares are not universal Sobol indices and
must not be extrapolated beyond the represented ensemble.

Structural scenarios are compared by paired contrasts under identical
weather/seed cells. They are not inserted as a third random variance component.
The persisted contrast is comparison-minus-central, so a negative heating
delta means that the declared structural case reduces useful heating demand.

Renovation effects use the same rule. For an identical archetype, weather
member, occupant seed, and model scenario:

$$
\Delta y_{w,s}
=y_{w,s}^{renovated}-y_{w,s}^{existing}
$$

The empirical distribution of $\Delta y$ is summarized directly. Subtracting
two independently calculated medians or upper quantiles would discard the
common-random-number pairing and is not an acceptable substitute.

## 9. Stock aggregation

### 9.1 Authoritative weights and join

The authoritative 2050 weight table is:

```text
BE_building_stock/data/scenarios/renovation/
archetype_matrix_2050_renovation_scenarios.csv
```

For its single `central` stock projection it contains 225 unique rows:

```text
3 regions x 25 archetypes x 3 physical states
```

The aggregation key is `(scenario, region, archetype_id, state_id)` and the
weight is `state_dwellings_2050`. The compatibility field `state_dwellings`
duplicates that value and is not added a second time. Regional total columns
repeat on many rows and must never be used as row weights.

The stock `scenario="central"`, climate `climate_scenario_id="rcp_*"`, and
thermal `model_scenario_id` are three different identifiers and remain three
different result columns.

The weights sum to 5,537,385 modelled R1–R4 dwellings, subject only to stored
floating-point precision. Zero-weight cells are valid stock cells and
contribute zero; they are not evidence of missing simulations. There are 23
zero-weight regional rows (six physical archetype/state cells are zero after
national grouping). The source-file checksum is stored with every aggregate
artifact. The audited Gate-5 input has
SHA-256
`710a8d6d9d250bb487349638f7e14dff681a9856a083d6f66ac1f239f9475594`;
the loader recomputes it rather than trusting this documentation string.

### 9.2 Per-dwelling distributions

Unweighted run-level results are retained so weather and occupant distributions
remain visible for every representative dwelling. An archetype should not be
duplicated into integer dwellings in memory, and archetypes must not be averaged
equally when estimating the Belgian stock.

### 9.3 Expected hourly stock series

For physical cell $(a,s)$, weather member $w$, and structural scenario $m$,
first average the hourly power over its converged seed set:

$$
\bar P_{a,s,w,m}(t)=
\frac{1}{n_s}\sum_{j=1}^{n_s}P_{a,s,w,j,m}(t)
$$

For region $r$ with weight $N_{r,a,s}$:

$$
P_{r,w,m}(t)=
\sum_{a,s}N_{r,a,s}\bar P_{a,s,w,m}(t)
$$

and the national series is the sum over regions. Heating and cooling are
aggregated separately. This seed-mean profile estimates a diversified class
load shape. It avoids multiplying one synthetic household realization by every
dwelling in its cell or imposing a single seed as perfectly synchronized
behaviour across the country.

The calculation does not explicitly simulate 5.5 million independent
households. Its coincident peak is conditional on the estimated mean class
profiles, the common Brussels weather realization, and the finite seed count.

### 9.4 Annual stock energy and intensity

For annual per-dwelling energy $E_{a,s,w,j,m}$:

$$
E_{r,w,m}=
\sum_{a,s}N_{r,a,s}
\left(\frac{1}{n_s}\sum_jE_{a,s,w,j,m}\right)
$$

Stock-total useful energy is reported in GWh. Stock intensity uses weighted
conditioned floor area, not the arithmetic mean of archetype intensities:

$$
e_{r,w,m}=
\frac{E_{r,w,m}}
{\sum_{a,s}N_{r,a,s}A_{f,a}}
$$

This preserves both the total energy implication and the per-square-metre
interpretation.

### 9.5 Two different peak metrics

The **sum of individual dwelling peaks** is:

$$
P_{sum\ of\ peaks}=
\sum_{r,a,s}N_{r,a,s}
\left(\frac{1}{n_s}\sum_j\max_t P_{a,s,w,j,m}(t)\right)
$$

This is an upper aggregation diagnostic because each dwelling's maximum may
occur at a different hour.

The **coincident stock peak** is:

$$
P_{coincident}=\max_t\sum_rP_{r,w,m}(t)
$$

Only the coincident value represents the peak of the aggregated hourly demand
series. The output stores both, their timestamps, and the diversity ratio:

$$
f_{diversity}=\frac{P_{coincident}}{P_{sum\ of\ peaks}}
$$

Heating and cooling metrics remain separate. No subtraction is used to create
a misleading net peak.

### 9.6 Regional and national reporting

For each RCP, weather member, and model scenario, aggregate:

- regional and national annual heating/cooling energy;
- weighted floor area and stock intensity;
- coincident heating/cooling peak and its UTC timestamp;
- weighted sum of individual dwelling peaks;
- diversity ratio; and
- stock contribution by region, dwelling type, period, and renovation state.

Weather-member aggregation is performed after the hourly stock series is
formed. Averaging individual peak hours before aggregation cannot recover a
system peak.

## 10. Output artifacts

Gate-5 artifacts belong under:

```text
thermal_model/data/monte_carlo/
```

The output contract separates input design, dwelling results, convergence,
variability, and stock results. The following is the complete Gate-5 artifact
vocabulary; no single execution path currently writes every row in this table:

| Artifact | Purpose |
|---|---|
| `run_manifest.csv` | Complete requested Cartesian design and deterministic run IDs |
| `run_diagnostics.csv` | One row per completed dwelling-year with annual metrics and provenance |
| `distribution_summary.csv` | Grouped empirical annual/peak summaries |
| `variance_contributions.csv` | Weather, occupant, and interaction sums of squares and shares |
| `paired_renovation_deltas.csv` | Direct common-weather/common-seed renovation contrasts |
| `paired_model_scenario_deltas.csv` | Direct common-weather/common-seed structural-scenario-minus-central contrasts |
| `convergence_results.csv` | Nested-checkpoint statistics and stopping-rule audit |
| `stock_aggregation.csv` | Regional/national useful energy, intensity, peak, and diversity metrics |
| `stock_contributions.csv` | Annual contributions by region, dwelling type, period, and state |
| `stock_distribution_summary.csv` | RCP-separated empirical weather-member distributions of stock annual energy and coincident peaks |
| `stock_weighted_distribution_summary.csv` | Regional/national dwelling-count-weighted empirical run distributions and intensity quantiles |
| `postprocessing_summary.json` | Authenticated source/output checksum chain and bounded post-processing diagnostics |
| `monte_carlo_summary.json` | Configuration, counts, checksums, statuses, warnings, and artifact hashes |
| `hourly/` | Optional run-level or aggregated hourly files when explicitly requested |

In particular, the current bounded-memory
`execute_streaming_stock_design(...)` runner does **not** consolidate a root
`run_manifest.csv`, root `run_diagnostics.csv`, or the run-level
`distribution_summary.csv`, `variance_contributions.csv`, and paired-delta
tables. Manifests and diagnostics remain checksum-indexed inside each
weather/scenario partition. The root files it does write are
`streaming_design_contract.json`, authenticated `convergence_results.csv`,
`partition_index.csv`, `stock_aggregation.csv`, `stock_contributions.csv`,
`stock_distribution_summary.csv`, and `monte_carlo_summary.json`. Run-level
distribution, ANOVA, and paired-delta tables require a separate bounded
post-processing pass over the partition diagnostic files. They are not stock
runner outputs and may be claimed only when the authenticated post-processing
commit described below is present and valid.

### 10.1 Authenticated bounded post-processing

After every stock partition has completed, the missing root run-level artifacts
are produced by a separate, deterministic and restartable analysis pass:

```bash
python3 -m thermal_model.monte_carlo postprocess \
  --output-dir thermal_model/data/monte_carlo/production
```

A read-only committed-artifact check is available as:

```bash
python3 -m thermal_model.monte_carlo postprocess \
  --output-dir thermal_model/data/monte_carlo/production \
  --status
```

The postprocessor authenticates the root design and completion summary, the
partition-index checksum, every partition-completion ledger, and every manifest
and diagnostics checksum. It independently checks exact partition coverage,
run-ID equality between each manifest and diagnostics file, ordered seed-bank
coverage, physical-cell coverage, model/weather identities, and thermal,
behavioural, weather, scenario, and cell-metadata provenance. The authoritative
stock-weight content and source checksums must equal the completed design.

Diagnostics are read in bounded chunks and reduced into temporary skinny files
for one archetype/state/RCP/model-scenario group. Exact quantiles and balanced
weather-by-occupant ANOVA are therefore calculated without loading the complete
run table. Renovation and structural-scenario deltas are written incrementally;
the structural delta file is present only when the completed design contains
more than one model scenario. Neither dwelling-hour traces nor the aggregated
`stock_hourly.csv` files are opened by this pass.

The committed outputs are:

- `distribution_summary.csv`: unweighted empirical distributions kept separate
  for every archetype, renovation state, RCP, and model scenario;
- `variance_contributions.csv`: balanced weather, occupant-seed, and interaction
  sums of squares for the four declared demand/peak metrics;
- `paired_renovation_deltas.csv`: renovated-minus-existing differences on exact
  common weather/seed draws;
- `paired_model_scenario_deltas.csv`: sensitivity-minus-central differences on
  exact common draws, when applicable;
- `stock_weighted_distribution_summary.csv`: inverse weighted empirical-CDF
  quantiles using 2050 dwelling counts, separately for each region and the
  modelled national stock; and
- `postprocessing_summary.json`: the source checksum chain, boundedness
  diagnostics, output row counts, and output SHA-256 ledger.

The stock-weighted intensity quantiles are distributions among dwellings: their
weights are dwelling counts, not conditioned floor area. They must not be
confused with either the unweighted per-archetype quantiles in
`distribution_summary.csv` or the aggregate stock intensity calculated as total
energy divided by total conditioned floor area.

All files are staged before atomic promotion, and the JSON summary is written
last as the commit marker. An interrupted pass can therefore be safely rerun.
The same authenticated inputs produce byte-identical CSVs and checksums.

Hourly dwelling output is not duplicated by default for every full-design run
because it is much larger than the annual table. A storage policy may stream
hourly results into stock accumulators or persist selected trace cases, but it
must never calculate a coincident peak from annual diagnostics alone. Any
stored hourly filename includes the deterministic `run_id`.

For production stock execution, the concrete persisted layout is deliberately
partitioned rather than one multi-billion-row dwelling-hour table:

```text
production/
  streaming_design_contract.json
  convergence_results.csv
  partition_index.csv
  stock_aggregation.csv
  stock_contributions.csv
  stock_distribution_summary.csv
  monte_carlo_summary.json
  partitions/<weather-member-and-model-scenario>/
    run_manifest.csv
    run_diagnostics.csv
    stock_aggregation.csv
    stock_contributions.csv
    stock_hourly.csv
    partition_complete.json
    progress.json
    progress_slot_{0,1}_arrays.npz
    progress_slot_{0,1}_diagnostics.csv
    last_failure.json                 # only after a caught failure/recovery
```

Each dwelling result is reconciled against its annual energy and peak, added
directly to seed-mean regional and Belgian stock accumulators with the declared
dwelling weights, and then released. Memory therefore scales with the number of
regions times the 8,760/8,784 hourly values, plus the partition diagnostic
ledger; it does not scale with the number of dwelling-hour simulations.
`partition_index.csv` gives the relative path, row count, and SHA-256 checksum
of every partition-level stock-hour file. Coincident peaks are still computed
from those accumulated hourly series, never reconstructed from annual data.

Stock weights are revalidated on every aggregation call. `stock_weights_sha256`
is a canonical checksum of normalized contract content, while
`stock_weights_source_sha256` retains the authoritative CSV byte checksum when
available. A stored content checksum that no longer matches the supplied table
is rejected. Stock outputs also carry the ordered seed values, seed-bank
checksum, per-cell archetype-physics checksum map, and thermal, behavioural,
occupant-distribution, scenario, weather, and forcing provenance.

That verbose provenance is deliberately **not** repeated on every one of the
8,760/8,784 stock-hour rows. Each partition-level `stock_aggregation.csv` (and
`stock_contributions.csv`) stores the verbose record once per aggregated row,
together with `stock_partition_provenance_contract_version` and
`stock_partition_provenance_sha256`. The matching `stock_hourly.csv` carries
only those two compact fields, the partition/group identity, timestamp, region,
and load values. The hash is calculated from a declared, versioned field list;
before weather-distribution reporting it is recomputed from the verbose annual
row and any mismatch is rejected. A consumer therefore resolves an hourly hash
to the verbose record in the same partition's `stock_aggregation.csv`, or in
the consolidated root `stock_aggregation.csv` located through
`partition_index.csv`, without broadcasting large JSON and checksum fields
hourly.

The stock distribution artifact keeps climate pathways and structural
scenarios separate. For annual heating, potential sensible cooling, and both
coincident peaks it reports descriptive minimum, p05, median, mean, p95,
maximum, and standard deviation across the included paired weather members.
Its member-provenance ledger retains member ID, weather-pair ID, observed anchor
year, climate target, and forcing/contract checksums. These are empirical
intervals over included members, not complete prediction intervals.

`monte_carlo_summary.json` records the convergence evidence status, source and
persisted SHA-256, first panel-converged checkpoint, selected ordered-prefix
hash, and complete convergence-bank hash. Its top-level status is `PASS` only
when this evidence is verified **and** `require_full_stock=True`. An explicit
evidence bypass on a full-stock workflow produces `WORKFLOW_CHECK_ONLY` with
`NOT_VERIFIED_BY_RUNNER` in the nested convergence ledger. A subset remains
`PARTIAL_STOCK_WORKFLOW` regardless of convergence status.

Run manifests and diagnostics retain the climate pathway, member, anchor year,
seed, archetype/state, model scenario, and relevant checksums. Aggregated stock
tables retain their applicable grouping axes and use the versioned compact
provenance link described above. Values are written with stable ordering and
explicit numeric precision so an identical run is reproducible.

## 11. Reproducible execution

Install the optional pinned behavioural dependency before any stochastic run:

```bash
python3 -m pip install -r thermal_model/behaviour/requirements.txt
```

Run the complete thermal/behaviour/Monte-Carlo test suite from the repository
root:

```bash
python3 -m pytest thermal_model/tests -q
```

Inspect the validated climate-member and model-scenario catalog:

```bash
python3 -m thermal_model.monte_carlo catalog
```

Run the bounded representative pilot:

```bash
python3 -m thermal_model.monte_carlo pilot
```

Run/resume the declared convergence panel, or inspect it without starting a
second coordinator:

```bash
python3 -m thermal_model.monte_carlo convergence --workers 4
python3 -m thermal_model.monte_carlo convergence --status
```

The command-line layer exposes the catalog, bounded pilot, and restartable
convergence and full-stock experiments. Before stock execution, run or inspect
the declared continuation:

```bash
python3 -m thermal_model.monte_carlo convergence-continuation --prepare-only
python3 -m thermal_model.monte_carlo convergence-continuation --workers 4
python3 -m thermal_model.monte_carlo convergence-continuation --status
```

The `stock` command loads the complete
75-cell physics matrix, all 54 paired weather members, the pinned authoritative
2050 stock weights, and the authenticated n=640 selection. Central production
can first be frozen and inspected without starting a simulation worker, then
run or resumed with:

```bash
python3 -m thermal_model.monte_carlo stock --prepare-only --workers 4 \
  --model-scenarios central
python3 -m thermal_model.monte_carlo stock --status
python3 -m thermal_model.monte_carlo stock --workers 4 --model-scenarios central
python3 -m thermal_model.monte_carlo stock --status
```

Structural cases must be declared when a new output design is created. To run
the complete registered set in one common-random-number design, use:

```bash
python3 -m thermal_model.monte_carlo stock --workers 4 --model-scenarios \
  central mass_light mass_heavy shading_unshaded \
  infiltration_half infiltration_one_and_half
```

The output directory is design-locked: a later invocation with a different
scenario set, weather inventory, seed bank, or model checksum is rejected.
The first command refuses to start unless the n=320/n=640 continuation status
is `CONVERGED` and both new checkpoints authenticate. Inspect the exact
installed interface with:

```bash
python3 -m thermal_model.monte_carlo --help
```

A full production command must be copied verbatim into the thesis run record
together with the generated summary checksum. A pilot command is an
implementation smoke test only and must remain labelled `pilot`; it cannot
establish the final occupant-seed count or produce thesis stock intervals.

The checked-in pilot under `thermal_model/data/monte_carlo/pilot/` contains 24
runs: one archetype, existing and advanced states, two RCP4.5 weather members,
three common seeds, and central/heavy-mass cases. It reuses six behavioural
profiles and retains one representative hourly trace. Its summary records
`seed_convergence=NOT_EVALUATED` because three seeds do not reach the first
predeclared checkpoint, and `stock_aggregation=NOT_EVALUATED` because two of 75
physical cells cannot represent the stock. Those statuses are safeguards, not
failed numerical checks.

Before Gate-5 results are accepted, rerun the upstream audits whose checksums
are consumed here:

```bash
python3 -m climate.src.validate --config climate/config.yaml
python3 -m thermal_model.validation
python3 -m thermal_model.behaviour.coupling
python3 -m pytest thermal_model/tests -q
```

## 12. Acceptance checks

### 12.1 Stable-interface checks

- Identical inputs and seed reproduce hourly values and diagnostics exactly.
- `simulate` performs no file write and does not alter its arguments or calling
  RNG state.
- Changing only the occupant seed cannot change a weather, archetype, stock, or
  structural identifier.
- Changing only the model scenario cannot change the occupant count or
  Richardson seed.
- Unknown identifiers and checksum mismatches fail explicitly.

### 12.2 Design checks

- The manifest is the exact requested Cartesian product.
- Every weather member receives the same ordered seed set.
- Renovation and model-scenario contrasts contain the same weather/seed keys.
- Equal PVGIS anchor years remain paired across RCPs.
- Missing or duplicate runs prevent balanced summaries.

### 12.3 Result checks

- Hourly input columns reconcile with the weather and behaviour contracts.
- Heating and cooling are non-negative and never simultaneous.
- Annual values and peaks recompute exactly from hourly output.
- Full-load hours are finite and zero-safe.
- Setpoint-hour counts sum to the calendar length for each schedule.
- Maximum energy-balance and setpoint errors remain inside the Gate-3 numerical
  tolerances.
- Assumptions, behaviour, weather, scenario, and stock hashes are present.

### 12.4 Aggregation checks

- The stock table has 225 unique region/archetype/state keys.
- `state_dwellings_2050` reconstructs regional and national totals.
- Weighted floor area and useful energy reconcile with their contribution
  tables.
- Coincident peak is calculated from the aggregated hourly series.
- Coincident peak does not exceed the weighted sum of individual peaks, apart
  from floating-point tolerance.
- National hourly power equals the sum of regional hourly power.
- Equal archetype weighting is never used for a stock result.

## 13. Reporting rules

For each major climate pathway and renovation comparison, report:

- median and empirical interval of per-dwelling annual heating and cooling;
- distribution of individual peak heating and cooling;
- weather, occupant, and interaction contributions from the crossed audit;
- paired differences between physical renovation states;
- regional, dwelling-type, construction-period, and state contributions;
- regional and national coincident peaks;
- sum-of-individual-peaks diagnostics and diversity ratios; and
- paired changes under each declared structural scenario.

Always state the run count, weather-member count, occupant-seed count,
convergence status, stock-weight vintage, and model/behaviour/climate contract
checksums. Report RCPs separately rather than pooling them into one interval.
Label cooling as universal ideal sensible-cooling **potential**, unless a
separate adoption and equipment layer is explicitly applied.

Recommended wording is “empirical range/quantile over the represented weather
members and occupant seeds.” Avoid “95% prediction interval,” “probability of
the RCP,” or “all uncertainty,” because those descriptions exceed the design.

### 13.1 Authenticated production report

The final reporting pass is deliberately read-only with respect to simulation
data. It neither launches model runs nor repeats post-processing. It will run
only after the central stock execution has the top-level status `PASS`, declares
`AUTHORITATIVE_FULL_STOCK`, carries verified convergence evidence, covers all 75
physics cells and all 54 weather members, and has a valid post-processing
commit. The exact central-only design is required; structural sensitivities are
reported through their separate sensitivity workflow rather than mixed into
the central interval.

Generate the Markdown report and figures with:

```bash
python3 -m thermal_model.monte_carlo report \
  --output-dir thermal_model/data/monte_carlo/production \
  --figure-dir thermal_model/figures
```

Verify an existing reporting commit without rewriting it with:

```bash
python3 -m thermal_model.monte_carlo report \
  --output-dir thermal_model/data/monte_carlo/production \
  --status
```

Before plotting, the reporter recomputes the streaming-design checksum, checks
the completed/expected run counts, verifies every root artifact checksum,
matches the exact partition index to the frozen design, and authenticates every
partition-completion ledger and all five partition artifacts. Manifest and
diagnostic run-ID sets must match exactly. It then calls the committed
post-processing status check, verifies every post-processing output checksum
and row count, and independently checks that stock distributions reproduce the
weather-member stock table, contribution tables close to national totals, ANOVA
sums of squares close, and paired renovation deltas reproduce their arithmetic.
Missing, partial, stale, sensitivity-mixed, or tampered inputs stop reporting.

The default outputs are:

- `thermal_model/data/monte_carlo/production/RESULTS.md` — concise numerical
  results, limitations, and full contract checksums;
- `thermal_model/data/monte_carlo/production/results_reporting_summary.json` —
  the self-hashed commit marker containing the SHA-256 of every source artifact,
  the report, and every figure; and
- nine independent figures in `thermal_model/figures/`, each exported as both a
  300-dpi PNG and a vector PDF, together with
  `mc_results_figure_provenance.json`, whose self-hashed ledger repeats every
  authenticated source checksum and every figure checksum.

The figures cover national annual ideal useful heating, national potential
sensible cooling, national coincident heating peak, heating contributions by
region, dwelling type, construction period and renovation state, balanced
weather-versus-occupant variance attribution, and exact-pair renovation
effects. A basename always represents one chart; different plots are never
combined into one PNG or PDF.

Every report and figure explicitly retains the interpretation boundaries:

- energy is ideal **useful** demand rather than delivered energy or electricity;
- cooling is universal ideal sensible-cooling **potential**, not adoption or AC
  electricity;
- totals cover the 5,537,385 R1–R4 modelled dwellings, with R5–R6 excluded;
- all regions share the Brussels weather trace, so regional climate gradients
  are absent and national coincidence is conditional on common weather; and
- p05–p95 ranges are empirical ranges over represented inputs, not complete
  prediction intervals.

The national stock energy and coincident-peak intervals span weather members
after occupant seeds have been averaged within every physical stock cell. The
occupant axis is shown separately in the crossed ANOVA and paired-dwelling
outputs; it must not be misdescribed as part of the national weather-only
interval.

## 14. Interpretation and limitations

### 14.1 Climate coverage

- The three RCPs use one GCM–RCM–member chain:
  CNRM-CERFACS-CNRM-CM5 / CNRM-ALADIN63 / r1i1p1. GCM, RCM, and climate-member
  structural uncertainty are absent.
- All members use the Brussels-area CORDEX cell and PVGIS point. Applying that
  forcing to Flanders, Wallonia, and Brussels omits Belgian spatial climate
  gradients and weather diversity. National coincident peaks are therefore
  conditional on one common Brussels weather trace and may overstate geographic
  coincidence.
- The 18 anchor years contain empirical variability and an observed-period
  trend. They are not stationary independent samples and do not carry
  probabilities.
- Monthly morphing preserves historical sub-monthly sequencing; it does not
  project changes in extremes, persistence, diurnal range, wind, humidity, or
  future cloud-regime structure.

### 14.2 Behaviour and comfort

- RichardsonPy behaviour is based mainly on UK evidence. Belgium enters through
  the household-size distribution, annual electricity benchmark, weather, and
  declared setpoints.
- The 3,500 kWh annual electricity normalization is a comparison reference, not
  a household-size-conditional empirical distribution.
- Fixed CET avoids duplicate/missing daylight-saving hours but does not model
  Belgian civil-time clock changes.
- Active occupancy does not identify sleeping or every physically present
  person. Adaptive comfort, window opening, thermostat hysteresis, preheating,
  rebound, and dynamic blind operation are absent.

### 14.3 Building and stock coverage

- The thermal model is a single-zone 5R1C representation. DHW, humidity, latent
  cooling, thermal bridges, HVAC efficiency/capacity, distribution losses, and
  auxiliaries are excluded.
- Cooling is calculated with universal unlimited ideal control. Because Belgian
  cooling ownership and use are not applied, the output is a technical useful
  cooling-demand potential, not current or future cooling electricity demand.
- The stock weights cover 5,537,385 R1–R4 dwellings. The 290,438 R5–R6
  residual equals 4.9836% of the 5,827,823 Statbel dwelling total and is
  excluded. National totals are therefore totals for the modelled stock, not all
  Belgian dwellings.
- The 2050 projection holds the 2025 modelled stock denominator fixed. It does
  not include demolition, new construction, new archetype cohorts, or changes
  in household count.
- Stock-composition uncertainty is not sampled. Regional state shares,
  archetype/state independence, the 50/50 apartment exposure split, whole-house
  geometry mapping for house categories, and fractional dwelling weights remain
  fixed upstream assumptions.
- Applying one representative dwelling profile to a weighted class is an
  archetype expectation, not a synthetic population of individually located
  buildings.

### 14.4 Statistical interpretation

- Structural scenario endpoints are sensitivity contrasts, not credible
  intervals or probability distributions.
- The crossed sum-of-squares shares are conditional descriptive attributions,
  not causal variance components for the real Belgian population.
- Finite-seed convergence establishes numerical stability of selected summary
  statistics only. It does not validate RichardsonPy behaviour or reduce model
  bias.
- Monte Carlo spread must not be used to conceal deterministic Gate-3 benchmark
  deviations.

## 15. Gate-completion criterion

The Gate-5 implementation is complete when its interfaces, analyses, tests and
bounded pilot exist. The thesis's **production Monte Carlo experiment** is
complete only when:

1. The stable wrapper and every declared scenario pass their contract tests.
2. The balanced-design and common-random-number identities pass.
3. A nested seed-convergence experiment reaches its predeclared criterion, or
   the non-convergence is reported without selecting an arbitrary seed count.
4. Distribution and crossed-variability outputs reconcile with run diagnostics.
5. Stock aggregation reconstructs all 225 weights and calculates coincident
   peaks from hourly aggregated series.
6. A representative pilot completes as a smoke test without being mislabelled
   a production result.
7. All generated artifacts carry their input identifiers and checksums.
8. The full thermal-model regression suite still passes.

Passing Gate 5 establishes a reproducible uncertainty and aggregation pipeline.
It does not turn the represented uncertainty dimensions into a complete
probabilistic forecast.

## 16. Related documentation

- `thermal_model/CONTRACT.md`: frozen thermal input/output contracts.
- `thermal_model/THERMAL_CORE.md`: ISO 13790 equations and variable dictionary.
- `thermal_model/GATE3_VERIFICATION_VALIDATION.md`: deterministic verification,
  Belgian validation, and sensitivity screen.
- `thermal_model/BEHAVIOURAL_WRAPPER.md`: RichardsonPy generation, seed split,
  fixed-CET alignment, internal gains, and setpoints.
- `climate/CLIMATE_ENSEMBLE.md`: climate-member provenance, pairing, facade
  adapter, validation, and climate limitations.
- `BE_building_stock/docs/BE_BUILDING_STOCK_MODEL.md`: archetype/state stock
  weights, 2050 projection, excluded residual, and stock caveats.

## 17. Deadline-fixed n=160 supervisor-results route

The supervisor-results route is a deliberately narrower, preliminary experiment for
obtaining interpretable weighted-stock results before the thesis deadline. It does not
replace or revise the original convergence result.

The immutable contract is
`data/monte_carlo/deadline_n160_supervisor_results/deadline_n160_supervisor_contract.json`
(SHA-256
`8a3599e8a09a6bd58016d62e49f94785dc25a3fe0db722cd2b821d97bc4713c8`).
It was declared after the n=160 result and therefore describes an administrative fixed
computational budget, not a prospective convergence stopping rule. Its only authorized
selection status is `DEADLINE_FIXED_BUDGET_SELECTED_AT_N160`; neither `CONVERGED` nor
`PASS` is permitted.

The original 54-member convergence panel remains
`NOT_CONVERGED_AT_N160`. At the n=80-to-n=160 comparison, 106 of 108 criteria were
within the unchanged 2% tolerance. The two misses were p95 heating intensity for the
low-demand advanced enclosed-apartment panel cell under RCP2.6 (2.004%) and RCP8.5
(2.433%). They remain disclosed in the selection contract and summary.

The fixed budget is the exact first 160 seeds from master seed 20250808, with ordered
prefix SHA-256
`a1e4b94e5be0a9552f952c6d8f0bda430f9bfdab3c397079192588f75a8aba46`.
The loader authenticates the completed n=160 source contract, summary, checkpoint,
manifest, diagnostics and convergence evidence before returning this prefix. No n=80
fallback, threshold retuning or alternative seed order is allowed.

### 17.1 Representative paired weather

The reduced weather design reuses the outcome-independent selection already
predeclared for the structural-sensitivity screen. The 2050-morphed RCP4.5 member
based on PVGIS historical year 2015 is the four-metric medoid: the minimum Euclidean
distance to the standardized centroid of HDD20, CDD26, maximum outdoor temperature
and the annual four-facade irradiance sum. The same PVGIS base year is retained while
applying each pathway-specific 2050 climate morph:

- `weather_2050_rcp_2_6_pvgis_2015`
- `weather_2050_rcp_4_5_pvgis_2015`
- `weather_2050_rcp_8_5_pvgis_2015`

Before a stock selection is loaded, all 18 RCP4.5 candidates are reloaded and the
predeclared medoid audit is recomputed. The three selected members' forcing and
weather-contract hashes are also checked exactly. This prevents demand outcomes from
influencing weather selection and preserves paired RCP comparisons.

As a descriptive check specific to this reduced subset, the same n=80-to-n=160
calculation has 105 of 108 criteria within 2%. The three misses are p95 cooling peak
for the low-demand advanced enclosed apartment: 3.632% (RCP2.6), 4.278% (RCP4.5)
and 2.334% (RCP8.5). This is reported as a stability diagnostic only; it does not
change the original non-convergence status.

### 17.2 Exact preliminary design and interpretation

The authorized run identity is:

$$
75\ \text{archetype states}
\times 3\ \text{paired representative weather members}
\times 160\ \text{occupant seeds}
\times 1\ \text{central scenario}
=36{,}000\ \text{runs}.
$$

All 75 physical archetype/renovation states and their 225 regional stock-weight rows
are included. The completed top-level status is
`PRELIMINARY_REPRESENTATIVE_WEATHER_STOCK_COMPLETE`, and the coverage label is
`AUTHORITATIVE_BUILDING_STOCK_REPRESENTATIVE_WEATHER_ONLY`. Both the design and final
summary carry the fixed-budget contract hash, selection evidence hash, exact seed and
weather identities, the original `NOT_CONVERGED_AT_N160` disclosure, and the future-work
pause receipts. Individual partitions use technical completion status `PASS`, but their
coverage is also explicitly limited to
`AUTHORITATIVE_BUILDING_STOCK_REPRESENTATIVE_WEATHER_ONLY`; partition metadata never
claims full weather-stock coverage.

The distributions quantify occupant-seed variability conditional on the three paired
2050-morphed weather members, each based on PVGIS 2015. They exclude within-RCP
weather-member variability and structural or epistemic scenario uncertainty. They
cannot support weather-variance attribution, extreme-weather-year demand,
system-reliability adequacy or complete prediction-interval claims. Differences
between the three RCP rows compare pathway morphs using the same historical base year;
they are not within-pathway weather distributions.

The n=320/n=640 continuation is preserved but paused and incomplete for future work.
Its pause receipt has SHA-256
`d5a59fa816d969cdee5869e19a8b3abf589b26b7f783440177f378a3335bd404`.
The earlier, unevaluated n=320 deadline amendment is retained only as superseded
methodological history (SHA-256
`a04a4319679afd7d8d5164248cd0159f832817fac817bc6e5b5bb8e4b5059158`).

### 17.3 Commands and artifacts

```bash
# Authenticate n=160, the weather medoid, and write the 105/108 evidence.
python3 -m thermal_model.monte_carlo supervisor-results --evaluate-only

# Freeze and inspect the exact 36,000-run design without simulating.
python3 -m thermal_model.monte_carlo supervisor-results --prepare-only --workers 4

# Run or resume the three independent weather/RCP partitions.
python3 -m thermal_model.monte_carlo supervisor-results --workers 4

# Read-only selection and stock progress.
python3 -m thermal_model.monte_carlo supervisor-results --status

# After all three partitions complete, publish truthful reduced-weather summaries.
python3 -m thermal_model.monte_carlo supervisor-postprocess

# Publish the supervisor-ready Markdown report and one PNG/PDF pair per figure.
python3 -m thermal_model.monte_carlo supervisor-report

# Reauthenticate the two downstream commits without rewriting them.
python3 -m thermal_model.monte_carlo supervisor-postprocess --status
python3 -m thermal_model.monte_carlo supervisor-report --status
```

The generic `postprocess` and `report` routes retain their authoritative-production
guards. In particular, the standard reporter still requires verified convergence,
18 weather members per RCP, `PASS`, and `AUTHORITATIVE_FULL_STOCK`; it must reject this
preliminary experiment. The separate supervisor downstream statuses are
`PRELIMINARY_REPRESENTATIVE_WEATHER_POSTPROCESS_COMPLETE` and
`PRELIMINARY_REPRESENTATIVE_WEATHER_REPORT_COMPLETE`, never `PASS` or `CONVERGED`.

The supervisor postprocessor reconstructs annual regional and national stock totals
for each of the 160 common seed realizations. This supports conditional occupant-seed
p05–p95 ranges for annual heating and cooling and exact paired RCP contrasts. Because
the same seed is used across archetype cells as a common-random-number design, these
ranges describe common-seed scenario variability; they are not the sampling
uncertainty of millions of independent households. Seed-specific stock-hourly arrays
were not retained, so the coincident system peak is reported only as the peak of the
seed-averaged aggregated profile. No seed-specific coincident-peak interval is
constructed.

Before calculation and again whenever status is requested, the supervisor route
reauthenticates the root design and summary, all three partition completion ledgers,
the exact 12,000-run manifest/diagnostic identity in each partition, and every
partition artifact checksum and row count. A later change to any completed source
therefore invalidates the downstream commit rather than leaving a stale completed
status.

Weather/occupant ANOVA and generic weather p05–p95 figures are deliberately omitted:
one member per RCP cannot identify within-RCP weather variance or a weather–occupant
interaction. Dwelling-level weighted distributions combine the represented physical
stock and occupant seeds conditional on the selected weather; they are not uncertainty
intervals for the national total. All figures are separate files under
`figures/supervisor_results/` by default.

The primary artifacts are:

- `data/monte_carlo/deadline_n160_supervisor_results/deadline_n160_selection_summary.json`
- `data/monte_carlo/deadline_n160_supervisor_results/representative_weather_n160_stability.csv`
- `data/monte_carlo/supervisor_results_preliminary/streaming_design_contract.json`
- byte-identical copies of the frozen supervisor contract, selection summary and
  stability evidence in `data/monte_carlo/supervisor_results_preliminary/`, providing
  a local record of the selection inputs; authentication still follows their frozen
  provenance back to the original n=160 source, pause receipt and weather catalog, so
  this directory is not independently portable
- `data/monte_carlo/supervisor_results_preliminary/monte_carlo_summary.json` after completion
- the root stock aggregation, contribution and distribution CSVs after completion
- `data/monte_carlo/supervisor_results_preliminary/supervisor_postprocessing_summary.json`
- `data/monte_carlo/supervisor_results_preliminary/supervisor_distribution_summary.csv`
- `data/monte_carlo/supervisor_results_preliminary/supervisor_stock_weighted_distribution_summary.csv`
- `data/monte_carlo/supervisor_results_preliminary/supervisor_paired_renovation_deltas.csv`
- `data/monte_carlo/supervisor_results_preliminary/supervisor_stock_annual_by_seed.csv`
- `data/monte_carlo/supervisor_results_preliminary/supervisor_stock_annual_seed_distribution.csv`
- `data/monte_carlo/supervisor_results_preliminary/supervisor_paired_rcp_annual_deltas.csv`
- `data/monte_carlo/supervisor_results_preliminary/SUPERVISOR_RESULTS.md`
- `data/monte_carlo/supervisor_results_preliminary/supervisor_results_reporting_summary.json`
- `figures/supervisor_results/supervisor_results_figure_provenance.json`
- 12 independent figure basenames under `figures/supervisor_results/`, each exported
  separately as PNG and PDF
