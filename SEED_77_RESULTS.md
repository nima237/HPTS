# HPTS results with random seed 77

All final-package analyses were rerun from the model-ready data with `SEED = 77`. The seed controls stochastic competitors, moving-block bootstrap inference, and the DVOL date-permutation placebo. The hierarchical ridge forecasts are deterministic.

## Primary holdout result

- HPTS-Final versus per-asset HAR: 6.9925% MSE gain; Newey-West t = 5.1605; two-sided p = 2.462e-7; 28-day block-bootstrap 95% interval [5.0420%, 9.3685%].
- Assets without listed options: 6.686% MSE gain versus HAR.
- HPTS-Final versus HPTS-NoIV: 2.1026% gain; t = 2.4361; nominal p = 0.01485; broad-family Holm p = 0.07423; bootstrap interval [0.4948%, 3.8030%].
- The gain is positive for 16 of 17 assets and Benjamini-Hochberg significant for 13 of 17.
- Raw-realised-volatility diagnostic: 2.82% gain; t = 1.21; p = 0.228.

## Seed-dependent extensions

- Pooled histogram gradient boosting versus HAR: 6.0246% gain; t = 4.0561; p = 4.99e-5.
- HPTS-Final versus pooled histogram gradient boosting: 1.0300% gain; t = 0.9756; p = 0.3293.
- Date-permuted DVOL versus HAR: 5.4837% gain.
- Correctly timed HPTS versus date-permuted DVOL: 1.5963% gain; t = 1.6267; nominal p = 0.1038; seven-test Holm p = 0.3114.
- Date-permuted DVOL versus HPTS-NoIV: 0.5145% gain; t = 1.1485; p = 0.2508.

## Predictor-block attribution

- XSEC restored: 2.8025% gain; Holm p = 1.052e-5.
- IV restored: 2.1026% gain; Holm p = 0.04454.
- TAIL restored: 1.5856% gain; Holm p = 0.04454.
- HAR restored: 0.3623% gain; Holm p = 0.1486.

## Interpretation

The primary hierarchical-ridge result is unchanged because it is deterministic. Seed 77 changes the bootstrap interval slightly and changes the stochastic nonlinear and permutation-placebo estimates. Under seed 77, the current-versus-permuted DVOL contrast is not nominally significant, so the option-information claim should remain supportive and exploratory rather than definitive. The tested volatility-targeting strategy remains negative and does not justify an economic-value claim.

## Added benchmark and horizon analyses

- HPTS versus ARMA(1,1): 14.99% MSE gain; Newey-West t = 8.83.
- HPTS versus GARCH(1,1)-t: 30.41% MSE gain; t = 9.68.
- HPTS versus two-step ARFIMA(1,d,1): 62.76% MSE gain; t = 15.23. The unstable two-step ARFIMA estimate is interpreted cautiously.
- HPTS versus HAR at 1, 3 and 7 days: 6.99%, 6.79% and 6.46% MSE gains.
- IV increment over HPTS-NoIV at 1, 3 and 7 days: 2.10% (p=0.0148), 3.23% (p=0.0064), and 3.45% (p=0.0584).
- In a complementary 1,000-draw locked-model joint-date permutation test, permuted IV worsens MSE by 4.80% versus NoIV on average; empirical one-sided p=0.001. This test asks a different question from the re-estimated single placebo and is reported alongside it.
