# CORDEX/PVGIS 2050-centred climate ensemble

This component converts daily CORDEX climate projections into monthly climate-
change parameters, applies those parameters to complete hourly PVGIS weather
years, and validates the resulting 54-member ensemble. The canonical weather
files remain horizontal; façade irradiance is delivered by a separate on-demand
adapter.

## Definition of the climate target

“2050 climate” does not mean the single simulated calendar year 2050. A climate
model year is one realization of internal variability. This component uses the
complete **2041–2060** period: the 20-year IPCC mid-term period centred between
2050 and 2051. It is therefore described as the **2050-centred mid-term climate
period**, not as a 2050 forecast or a 30-year WMO climatological normal.

Each RCP target is compared with its matching **2006–2023 CORDEX scenario
branch**. The canonical hourly PVGIS period is also 2006–2023, so the temporal
anchors now match exactly. The immutable PVGIS source downloads still contain
2005 for provenance, but the cleaner excludes that year before any morphing,
validation, or façade alignment.

## Data layout

```text
data/
  raw/
    cordex/cds/
      rcp_2_6|rcp_4_5|rcp_8_5/
        baseline_2006_2023/  # downloaded tas/rsds NetCDF + CDS provenance
        target_2041_2060/    # downloaded tas/rsds NetCDF + CDS provenance
    observed/                # original PVGIS horizontal and façade exports
    reference/               # fixed Eurostat BE100 comparison snapshot
  processed/
    cordex/daily/            # extracted Brussels daily series + sidecars
    deltas/                  # monthly climatologies, deltas, variability
    observed/                # cleaned hourly PVGIS series
    ensemble_2050/           # 54 horizontal morphed members + manifests
    validation/              # validation tables, reports, and figures
```

Raw NetCDF, provider provenance JSON, and provider provenance images remain
under `data/raw/cordex/cds`. The extracted daily CSVs are derived artifacts and
therefore live under `data/processed/cordex/daily`.

## CORDEX provenance and preparation

The raw inputs came directly from the Copernicus Climate Data Store dataset
[`projections-cordex-domains-single-levels`](https://cds.climate.copernicus.eu/datasets/projections-cordex-domains-single-levels),
DOI [`10.24381/cds.bc91edc3`](https://doi.org/10.24381/cds.bc91edc3).

The configured chain is:

- CORDEX EUR-11, 0.11° horizontal resolution, Gregorian daily means;
- GCM `CNRM-CERFACS-CNRM-CM5`;
- RCM `CNRM-ALADIN63`, member `r1i1p1`, corrected file version `v2`;
- RCP2.6, RCP4.5, and RCP8.5;
- near-surface air temperature `tas` in K and surface downward shortwave
  radiation `rsds` in W/m²;
- CDS area subset west/south/east/north = `2.4/49.4/6.5/51.6`;
- nearest cell to `(50.85, 4.35)`: `(50.79848111186651,
  4.269705300227394)`, subset indices `y=13, x=11` (full-domain indices
  `y=239, x=191`).

Every NetCDF and its CDS/Rook provenance JSON and image has an expected SHA-256
hash in `config.yaml`. The provenance records preserve the upstream CORDEX
catalogue entity, CDS workflow start/end time, requested years and area, and the
Rook/clisops software versions. NetCDF global attributes—including experiment,
tracking ID, model IDs, member, version, calendar, cell method, and units—are
validated during extraction.

Prepare the daily point series with:

```bash
python3 -m climate.src.prepare_cordex --config climate/config.yaml
```

This command verifies all raw hashes, checks each NetCDF, selects the configured
cell, converts `tas` using `T_out_C = tas_K - 273.15`, retains `rsds` unchanged,
and writes six processed series:

- three 2006–2023 scenario-matched baselines, each with 6,574 days;
- three 2041–2060 targets, each with 7,305 days.

`data/processed/cordex/cordex_daily_manifest.json` links every processed hash to
the raw NetCDF and provider provenance hashes and records the processing runtime.

## Monthly climate-change parameters

For each scenario and calendar month `m`:

```text
ΔT_m = mean(T_2041–2060,m) - mean(T_2006–2023,m)
α_raw,m = mean(rsds_2041–2060,m) / mean(rsds_2006–2023,m)
α_applied,m = clip(α_raw,m, 0.7, 1.3)
```

Both sides come from the same RCP branch. Leap days participate normally. The
0.7–1.3 α interval is only a numerical safety net: activation logs the scenario,
month, raw value, and clipped value. It does not activate for the current data;
the real range is `0.9552029895` (RCP2.6, January) to `1.1007481057`
(RCP2.6, September).

Temperature variance stretching is not applied. The morph consequently
preserves the observed within-month variance, diurnal range, and sequence of
weather events while changing monthly means.

Build the monthly artifacts with:

```bash
python3 -m climate.src.deltas --config climate/config.yaml
```

The current monthly parameters are:

| Month | ΔT RCP2.6 (°C) | α RCP2.6 | ΔT RCP4.5 (°C) | α RCP4.5 | ΔT RCP8.5 (°C) | α RCP8.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jan | 0.899 | 0.955 | 1.181 | 1.027 | 1.431 | 1.009 |
| Feb | 0.305 | 1.013 | 0.218 | 1.060 | 1.675 | 0.956 |
| Mar | 0.485 | 1.038 | 0.286 | 0.990 | 1.438 | 0.989 |
| Apr | 0.440 | 1.000 | 0.324 | 1.000 | 0.808 | 1.041 |
| May | 0.443 | 1.018 | 1.407 | 1.041 | 0.176 | 0.990 |
| Jun | 1.476 | 1.069 | 1.594 | 1.036 | 0.530 | 0.987 |
| Jul | 0.986 | 1.046 | 0.691 | 1.017 | 0.966 | 1.008 |
| Aug | 1.015 | 1.049 | 0.449 | 1.017 | 0.880 | 1.023 |
| Sep | 1.658 | 1.101 | 0.190 | 1.043 | 0.983 | 1.046 |
| Oct | 1.339 | 0.992 | 0.991 | 1.020 | 1.625 | 0.980 |
| Nov | 1.092 | 1.006 | 1.329 | 1.003 | 1.340 | 1.012 |
| Dec | 0.818 | 0.985 | 0.996 | 1.007 | 1.752 | 1.040 |

Day-weighted annual anchors are:

| Scenario | Mean-temperature change | Mean-solar ratio |
| --- | ---: | ---: |
| RCP2.6 | +0.9143 °C | 1.03587 |
| RCP4.5 | +0.8080 °C | 1.02305 |
| RCP8.5 | +1.1314 °C | 1.00654 |

RCP ordering need not be monotonic in every variable or month in one 20-year
realization. In particular, RCP4.5 is slightly cooler than RCP2.6 in this
single-chain period. That is retained as model output rather than adjusted.

The generated delta outputs are:

- `monthly_climatologies.csv`: 72 rows, six sources × twelve months;
- `monthly_deltas_2050.csv`: 36 scenario-month morph parameters;
- `year_month_variability_2050.csv`: 720 rows, three scenarios × twenty target
  years × twelve months;
- `climate_deltas_2050.provenance.json`: formulas, periods, raw/processed
  lineage, hashes, diagnostics, and caveats.

The year-month variability table is diagnostic. Its anomalies are not applied
to the hourly PVGIS morph.

## PVGIS hourly observations

The canonical hourly template covers 2006–2023 at `(50.830, 4.350)`, elevation
61 m. The five immutable PVGIS source exports cover 2005–2023; the cleaner
validates that full source coverage and then selects 2006–2023. Their exact
PVGIS v5.3 API request URLs and raw hashes are recorded in `config.yaml`:
horizontal plus south, east, west, and north 90° planes, using PVGIS-SARAH3,
component output, and the PVGIS horizon.

PVGIS documentation describes SARAH3 irradiance as satellite-derived values at
the scan timestamp. Here that timestamp is `HH:10`; PVGIS treats it as
representative of the hour. Temperature and wind are hourly reanalysis values,
not direct station observations. The cleaner floors `HH:10` to `HH:00` UTC, an
explicit ten-minute timing approximation that should be revisited if the demand
model proves sensitive to solar timing.

Build the cleaned horizontal series with:

```bash
python3 -m climate.src.load_observed --config climate/config.yaml
```

The output contains 157,776 continuous hours and these fields:

```text
timestamp_utc, T_out_C,
I_beam_horizontal_W_m2, I_diffuse_horizontal_W_m2, I_solar_W_m2,
wind_speed_10m_m_s, sun_height_deg, pvgis_reconstructed
```

`I_solar_W_m2 = beam + diffuse`. Horizontal reflected irradiance is verified as
zero. The raw source has no gaps, duplicates, missing values, negative
irradiance, or reconstructed values. `pvgis_reconstructed` is retained as an
explicit integrity flag even though all current values are false.

## Hourly morph and ensemble design

For every complete PVGIS year and scenario:

```text
T_morph(h)       = T_observed(h) + ΔT_month
beam_morph(h)    = beam_observed(h) × α_applied,month
diffuse_morph(h) = diffuse_observed(h) × α_applied,month
GHI_morph(h)     = beam_morph(h) + diffuse_morph(h)
```

Wind speed, sun height, timestamps, and reconstruction flags remain unchanged.
Build the ensemble with:

```bash
python3 -m climate.src.build_ensemble --config climate/config.yaml
```

The 54 members are a **paired deterministic empirical weather-year ensemble**:
the same 18 PVGIS years occur under each of three RCPs. They are not 54
independent Monte Carlo samples and do not define probabilities. Comparisons
between RCPs should preserve the weather-year pairing, with `n=18` weather years
per scenario.

The current ensemble contains 473,328 member-hours. Its physical envelope is
`−13.0843–38.4763 °C` and `0–998.8494 W/m²` horizontal GHI. The preserved
calendar includes 8,784-hour leap years in 2008, 2012, 2016, and 2020.

## Façade irradiance

`climate.src.transpose_facades` aligns the PVGIS south, east, west, and north
templates with a selected member and applies the same monthly α to each plane's
beam, diffuse, and reflected components. Façade values are intentionally not
materialized in the 54 canonical horizontal files.

These values are external plane-of-array irradiance, not transmitted window
heat gains. Before demand-model integration, the supervisor/model interface
must settle glazing area, g-value, frame fraction, shading and horizon handling,
incidence-angle treatment, and whether external irradiance or already-
transmitted solar gain is expected.

## Validation

Audit the existing artifacts without rebuilding them:

```bash
python3 -m climate.src.validate --config climate/config.yaml
```

Hard checks cover hashes, sidecars, schemas, complete UTC calendars, duplicate
or missing values, physical bounds, GHI composition, zero sub-horizon
irradiance, invariant fields, and monthly ΔT/α identities. The current run has
54 passing members, 473,328 passing hours, 648 passing month checks, zero hard
errors, and zero plausibility warnings.

Eurostat-compatible diagnostics use UTC daily means:

```text
HDD_C_days = Σ(18 - T_daily) when T_daily <= 15 °C; otherwise 0
CDD_C_days = Σ(T_daily - 21) when T_daily >= 24 °C; otherwise 0
annual solar [kWh/m²] = Σ hourly GHI [W/m²] / 1000
```

The stored Eurostat `nrg_chddr2_a`, BE100, 2005–2023 snapshot remains an
immutable source record and matches the official values for that period. The
comparison uses only its 2006–2023 rows to match the canonical PVGIS period.
BE100 is a regional NUTS-3 series rather than the PVGIS point, so this tests
interannual consistency rather than site identity. Over the matched period,
correlations are 0.988435 for HDD and 0.937998 for CDD; the PVGIS-minus-BE100
mean biases are +151.755208 and −3.531481 °C-days.

Direct scenario-matched CORDEX changes and mean paired morph changes are:

| Scenario | CORDEX ΔHDD | Morph ΔHDD | CORDEX ΔCDD | Morph ΔCDD |
| --- | ---: | ---: | ---: | ---: |
| RCP2.6 | −269.5240 | −257.6130 | +22.0550 | +16.4298 |
| RCP4.5 | −264.0022 | −247.6597 | +7.7714 | +11.7370 |
| RCP8.5 | −352.2434 | −364.4534 | +24.8758 | +11.7911 |

Exact equality is not expected because degree days are nonlinear threshold
indices and morphing deliberately retains the PVGIS temperature distribution.
The HDD/CDD/solar plausibility bands are broad, pre-specified engineering smoke
tests. They warn but do not prove climate validity or building-load validity.

## Validation figures

Generate figures after validation:

```bash
python3 -m climate.src.plot_validation --config climate/config.yaml
```

- `monthly_climate_change_parameters` shows the monthly ΔT and α supplied to
  the morph. Temperature and solar need not move together.
- `morph_invariant_residual_heatmaps` shows recovered-minus-expected identities.
  Uniform panels mean all serialized residuals are zero, not missing data.
- `temperature_duration_curves` shows median annual duration curves with the
  empirical 5th–95th percentile envelope across 18 weather years. That envelope
  is not a confidence interval. The hottest/coldest 2% panels are descriptive;
  they are not asserted to be capacity-design criteria.

Every PNG and vector PDF is hashed in
`validation_figures.provenance.json` together with its input hashes.

## Reproducible full build

```bash
python3 -m pip install -r climate/requirements.txt
python3 -m climate.src.prepare_cordex --config climate/config.yaml
python3 -m climate.src.deltas --config climate/config.yaml
python3 -m climate.src.build_ensemble --config climate/config.yaml
python3 -m climate.src.validate --config climate/config.yaml
python3 -m climate.src.plot_validation --config climate/config.yaml
python3 -m pytest climate/tests -q
```

Dependency versions are pinned because CSV, PDF, and PNG byte hashes can change
between numerical and rendering-library versions.

## Thesis interpretation and remaining limitations

- This is a **single GCM–RCM–member sensitivity study**. It samples RCP forcing
  and paired observed weather years but not structural GCM/RCM uncertainty. It
  must not be presented as a probabilistic climate projection.
- The PVGIS years 2006–2023 contain both weather variability and an observed-
  period climate trend. They are retained as an empirical range, not assumed to
  be stationary random draws.
- Additive monthly ΔT preserves observed variance, extremes, and hot/cold-spell
  persistence. It does not project future changes in those properties. Peak-
  capacity conclusions require later load-model sensitivity checks.
- A common α for beam, diffuse, and reflected radiation preserves the observed
  component fractions. Future cloud-regime, diffuse-fraction, snow/albedo, and
  sub-daily solar-shape changes are not represented.
- Future wind is not modelled; wind and sun geometry remain those of the source
  PVGIS year.
- The CORDEX cell and PVGIS point are not colocated. Delta morphing reduces
  sensitivity to absolute climate-model bias but does not remove urban,
  elevation, horizon, or sub-grid differences.
- UTC weather must be reconciled with Europe/Brussels occupancy and control
  schedules. Cooling forcing currently has no humidity, so it supports sensible
  loads only unless humidity/dew point is added later.

## Core references

- Belcher, S. E., Hacker, J. N., & Powell, D. S. (2005). Constructing design
  weather data for future climates. *Building Services Engineering Research and
  Technology, 26*(1), 49–61. <https://doi.org/10.1191/0143624405bt112oa>
- IPCC AR6 WGI Chapter 1, assessment-period definitions:
  <https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-1/>
- Copernicus Climate Data Store CORDEX dataset and quality documentation:
  <https://cds.climate.copernicus.eu/datasets/projections-cordex-domains-single-levels>
- PVGIS 5 user manual:
  <https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-user-manual_en>
- Eurostat heating and cooling degree-day methodology:
  <https://ec.europa.eu/eurostat/cache/metadata/en/nrg_chdd_esms.htm>
