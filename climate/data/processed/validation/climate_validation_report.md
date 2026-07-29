# Climate ensemble validation report

**Overall status: PASS**

The persisted PVGIS baseline and 2050-centred ensemble were audited without regenerating any weather member.

## Coverage and hard checks

- Ensemble members: 54
- Total member-hours: 473,328
- Monthly morph checks: 648
- Every member is a complete 8,760- or 8,784-hour UTC calendar year.
- Hashes, schemas, sidecars, physical bounds, GHI composition, night irradiance, unchanged fields, and monthly morph identities passed.

## Climate diagnostics

Degree days use UTC daily-mean temperature:

- `HDD = 18 - T_daily_mean` when `T_daily_mean <= 15 degC`; otherwise 0.
- `CDD = T_daily_mean - 21` when `T_daily_mean >= 24 degC`; otherwise 0.
- `annual solar = sum(GHI_hourly) / 1000` in kWh/m2/year

- Paired HDD ratio: 0.831742–0.923741
- Paired CDD change: 0.000000–40.141013 degC-days
- Paired annual-solar ratio: 1.006207–1.039224
- Observed annual solar: 1040.578990–1231.262550 kWh/m2/year
- Morphed annual solar: 1047.716611–1276.183577 kWh/m2/year

## Official Brussels reference comparison

The same degree-day formulas were applied to PVGIS 2006–2023 and paired by year with the official Eurostat BE100 series.

- HDD Pearson correlation: 0.988435
- HDD mean bias (PVGIS - BE100): 151.755208 degC-days
- HDD mean absolute error: 151.755208 degC-days
- CDD Pearson correlation: 0.937998
- CDD mean bias (PVGIS - BE100): -3.531481 degC-days
- CDD mean absolute error: 5.102130 degC-days

## Direct CORDEX comparison

Identical degree-day definitions were calculated for every scenario-matched 2006–2023 baseline and 2041–2060 CORDEX year. The comparison below contrasts the direct CORDEX ensemble-mean change with the mean paired change in the morphed PVGIS ensemble. Exact equality is not required because degree days are non-linear threshold indicators and the morph intentionally retains the observed PVGIS baseline distribution.

- rcp_2_6: HDD change CORDEX -269.523994, morph -257.612965; CDD change CORDEX +22.054987, morph +16.429751 degC-days.
- rcp_4_5: HDD change CORDEX -264.002245, morph -247.659702; CDD change CORDEX +7.771449, morph +11.736953 degC-days.
- rcp_8_5: HDD change CORDEX -352.243403, morph -364.453391; CDD change CORDEX +24.875836, morph +11.791134 degC-days.

## Warning-only plausibility screening

- Warning count: 0
- HDD ratio band: [0.65, 1.0]
- CDD change band: [0.0, 150.0] degC-days
- Annual-solar ratio band: [0.95, 1.15]

A plausibility warning does not invalidate mathematically correct morphing. This canonical run produced no warnings.

## Interpretation and caveats

The temperature invariant compares the monthly mean change `mean(T_morph - T_observed)` with CORDEX `delta_T_C`. It deliberately does not require the morphed absolute mean to equal the CORDEX future mean; retaining the observed hourly baseline is the bias-cancellation step.

- This is a single GCM-RCM-member sensitivity study. It samples RCP forcing and paired observed weather years, but it does not sample structural climate-model uncertainty and is not a probabilistic climate projection.
- Each 2041-2060 scenario climatology is differenced against its matching 2006-2023 CORDEX scenario branch and applied to the same 2006-2023 PVGIS period. Matching the periods removes the former one-year temporal-anchor mismatch, while delta morphing still retains the observed point-weather distribution rather than reproducing the CORDEX absolute baseline.
- The 54 members form a paired deterministic empirical weather-year ensemble (18 PVGIS years repeated across 3 RCPs), not 54 independent Monte Carlo samples.
