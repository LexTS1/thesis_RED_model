# Current-versus-2050 paired factorial experiment

## Purpose

The original weighted-stock results describe the projected 2050 stock under
2050-morphed weather, but do not by themselves say how much demand changes from
the current situation. This experiment supplies that counterfactual using a
two-factor, two-level design. The factors are:

- building-stock composition: 2025 or projected 2050 renovation-state weights;
- climate forcing: unmorphed observed reference weather or the corresponding
  2041–2060 morph.

The four cases are:

| Case | Stock weights | Weather forcing |
|---|---|---|
| Q00 | 2025 | unmorphed PVGIS 2015 |
| Q10 | 2050 | unmorphed PVGIS 2015 |
| Q01 | 2025 | PVGIS-2015-based 2050 morph for the RCP |
| Q11 | 2050 | PVGIS-2015-based 2050 morph for the RCP |

Q00 is a reference-year counterfactual, not a reconstruction of meteorological
year 2025. The 2025 stock weights are deliberately combined with the selected
observed 2015 chronology so stock and climate can be changed independently.

## Matched inputs

All four cases use the same 75 physical TABULA archetype/renovation-state
combinations, central model assumptions, and exact authenticated prefix of 160
occupant seeds. A seed therefore represents the same behavioural draw wherever
it appears. The observed and future weather members also share the same 2015
hour-of-year chronology. This common-random-number design reduces irrelevant
Monte Carlo noise in differences.

Each of the 75 × 160 dwelling-years is simulated once for each of four weather
forcings: one observed reference and three RCP morphs. Every resulting hourly
profile is accumulated simultaneously with the 2025 and 2050 regional stock
weights. The experiment therefore contains 48,000 thermal-model evaluations
and 96,000 weighted accumulations. Dual accumulation avoids duplicating the
physics calculations and preserves the true coincident regional and national
peak. Individual dwelling peaks are never summed and presented as a system
peak.

## Effect definitions

For an annual energy, intensity, or peak metric $Q$:

$$
\Delta Q_{\mathrm{renovation}} = Q_{10} - Q_{00}
$$

$$
\Delta Q_{\mathrm{climate}} = Q_{01} - Q_{00}
$$

$$
\Delta Q_{\mathrm{interaction}}
= Q_{11} - Q_{10} - Q_{01} + Q_{00}
$$

$$
\Delta Q_{\mathrm{combined}} = Q_{11} - Q_{00}
$$

Consequently:

$$
\Delta Q_{\mathrm{combined}}
= \Delta Q_{\mathrm{renovation}}
+ \Delta Q_{\mathrm{climate}}
+ \Delta Q_{\mathrm{interaction}}.
$$

This identity is verified for every occupant seed before distributions are
summarised. Negative effects denote reduced demand. Effects are also reported
as a percentage of Q00 for the same seed, then summarised across seeds.

## Outputs and uncertainty interpretation

Annual demand is calculated for every seed and reported by its median and
p05–p95 interval. This interval is conditional occupant variability, not a
complete prediction interval. Coincident peaks are extracted from the
stock-level hourly profile averaged across the 160 behavioural realisations;
they are point estimates conditional on the selected chronology.

The output directory is
`thermal_model/data/monte_carlo/current_vs_2050/`. It contains the immutable
design contract, restartable partition checkpoints, per-run diagnostics,
annual cases and effects by seed, distribution summaries, coincident-peak
tables, all dual-weight stock-hour profiles, an authenticated report, and an
artifact checksum ledger. Figures are stored separately under
`thermal_model/figures/current_vs_2050/`, with one chart per PNG/PDF pair.

## Qualification and limitations

The experiment remains preliminary and representative-weather. The n=160
sample is an explicitly chosen fixed computational budget and the original
status remains `NOT_CONVERGED_AT_N160`; it must not be relabelled as converged.
Only one PVGIS-2015 chronology is represented per RCP, so within-RCP
weather-year variability is excluded. The proposed all-54-member n=20 screen is
a separate follow-up and must not be merged silently with these n=160
conditional distributions.

Heating is ideal useful space-heating demand. Cooling is potential useful
sensible cooling under universal ideal control at the modelled cooling
setpoint, not actual cooling-appliance electricity or an adoption forecast.
Converting thermal peaks into generation, network, or installed heat-pump
capacity requires separate technology and energy-system assumptions.

## Authentication checks

The workflow stops if the stock totals, cell coverage, seed identity, weather
checksums, timestamp alignment, run completeness, hourly-to-annual energy
balance, or four-term effect identity fails. The rerun Q11 cases must also
reproduce the already authenticated supervisor annual and hourly 2050 results
within tight numerical tolerances. This last check links the new counterfactual
directly to the previously reported result chain.
