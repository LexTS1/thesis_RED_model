# Climate ensemble validation report

**Overall status: PASS**

The persisted PVGIS baseline and 2050 ensemble were audited without regenerating any weather member.

## Coverage and hard checks

- Ensemble members: 57
- Total member-hours: 499,608
- Monthly morph checks: 684
- Every member is a complete 8,760- or 8,784-hour UTC calendar year.
- Hashes, schemas, sidecars, physical bounds, GHI composition, night irradiance, unchanged fields, and monthly morph identities passed.

## Climate diagnostics

Degree days use UTC daily-mean temperature:

- `HDD = 18 - T_daily_mean` when `T_daily_mean <= 15 degC`; otherwise 0.
- `CDD = T_daily_mean - 21` when `T_daily_mean >= 24 degC`; otherwise 0.
- `annual solar = sum(GHI_hourly) / 1000` in kWh/m2/year

- Paired HDD ratio: 0.734115–0.904866
- Paired CDD change: 0.000000–83.502933 degC-days
- Paired annual-solar ratio: 1.061137–1.073875
- Observed annual solar: 1040.578990–1231.262550 kWh/m2/year
- Morphed annual solar: 1105.446984–1317.039297 kWh/m2/year

## Official Brussels reference comparison

The same degree-day formulas were applied to PVGIS 2005–2023 and paired by year with the official Eurostat BE100 series.

- HDD Pearson correlation: 0.986929
- HDD mean bias (PVGIS - BE100): 154.981776 degC-days
- HDD mean absolute error: 154.981776 degC-days
- CDD Pearson correlation: 0.937705
- CDD mean bias (PVGIS - BE100): -3.736513 degC-days
- CDD mean absolute error: 5.224496 degC-days

## Direct CORDEX comparison

Identical degree-day definitions were calculated for every historical and future CORDEX year. The comparison below contrasts the direct CORDEX ensemble-mean change with the mean paired change in the morphed PVGIS ensemble. Exact equality is not required because degree days are non-linear threshold indicators and the morph intentionally retains the observed PVGIS baseline distribution.

- rcp_2_6: HDD change CORDEX -327.459504, morph -300.377007; CDD change CORDEX +17.398734, morph +18.241380 degC-days.
- rcp_4_5: HDD change CORDEX -503.087916, morph -452.847443; CDD change CORDEX +39.882122, morph +30.030986 degC-days.
- rcp_8_5: HDD change CORDEX -624.521745, morph -572.830194; CDD change CORDEX +48.100010, morph +38.922412 degC-days.

## Warning-only plausibility screening

- Warning count: 0
- HDD ratio band: [0.65, 1.0]
- CDD change band: [0.0, 150.0] degC-days
- Annual-solar ratio band: [0.95, 1.15]

A plausibility warning does not invalidate mathematically correct morphing. This canonical run produced no warnings.

## Interpretation and caveats

The temperature invariant compares the monthly mean change `mean(T_morph - T_observed)` with CORDEX `delta_T_C`. It deliberately does not require the morphed absolute mean to equal the CORDEX future mean; retaining the observed hourly baseline is the bias-cancellation step.

- Climate-model uncertainty is not sampled because all scenarios use one GCM-RCM-member chain.
- Applying the full 1981-2005 to 2050-2070 delta to PVGIS 2005-2023 weather would slightly double-count warming since the historical CORDEX reference period.
