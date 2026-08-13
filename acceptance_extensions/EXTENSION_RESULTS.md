# Acceptance-strengthening extensions

Protocol: all model selection uses calendar 2023. The evaluation period remains the locked 2024-2026 holdout. HPTS hyperparameters used in the placebo tests are the original pre-holdout choices (alpha=100, pool=10).

## 1. Exploratory economic value

The tested long-only volatility-targeting rule does not support an economic-performance claim. HPTS has slightly lower turnover than HAR, but lower return, lower Sharpe ratio, lower certainty-equivalent return, and slightly worse variance-target tracking at every tested transaction-cost level (0, 5, 10, and 20 bps). The paired utility difference is negative and not significant (NW t from -1.53 to -1.28).

Decision: do not add this strategy as positive evidence. It may be disclosed as an exploratory negative result, but it should not become a headline contribution.

## 2. Pooled nonlinear benchmark

The validation-selected pooled histogram gradient boosting model improves MSE over HAR by 6.025% (NW t=4.056, p=4.99e-5). HPTS-Final remains the best point estimate and improves on pooled HGB by 1.030%, but that direct difference is not significant (NW t=0.976, p=0.329).

Decision: add pooled HGB as a strong nonlinear panel benchmark. Phrase the conclusion as best pooled point estimate, not statistically established dominance over the nonlinear benchmark.

## 3. DVOL timing and placebo tests

Current-open HPTS improves on HAR by 6.992%. Stale-DVOL versions remain above HAR because the spot and cross-sectional blocks remain informative and DVOL is persistent, but current-open HPTS is better than every stale specification by 1.04%-2.17%.

The cleaner joint-state placebo uses one common date permutation for the public DVOL variables and then reconstructs the IV-RV gap from permuted DVOL and current realised history. With seed 77, permuted DVOL produces a 5.484% gain versus HAR and remains statistically indistinguishable from HPTS-NoIV (0.515% relative gain, p=0.251). Current correctly timed DVOL improves on permuted DVOL by 1.596% (NW t=1.627, nominal p=0.104; Holm p=0.311 across the seven direct comparisons).

Decision: add the date-permutation placebo only as exploratory robustness evidence. Its strongest implication is that permuted DVOL remains statistically indistinguishable from NoIV performance; the direct current-versus-permuted contrast is not significant under seed 77. Report the stale-lag results as supportive but non-monotonic, not as a clean decay curve.

## 4. Predictor-block attribution and coefficients

Relative to separately re-tuned ablations, HPTS-Final gains 2.803% when XSEC is restored, 2.103% when IV is restored, 1.586% when TAIL is restored, and 0.362% when HAR is restored. Across the four block tests, Holm-adjusted p-values are 0.000011 (XSEC), 0.0445 (IV), 0.0445 (TAIL), and 0.149 (HAR).

Mean standardised coefficient RMS across expanding-window refits ranks the blocks HAR (0.130), IV (0.085), TAIL (0.063), and XSEC (0.053). Coefficient magnitude is descriptive and not directly comparable to ablation value under collinearity; the ablation tests are the primary attribution evidence.

Decision: include the block-ablation table in the main text or supplement and use the coefficient plot as an interpretability figure. Because these extensions were designed after inspecting the original holdout results, label the four-test Holm family as an exploratory structural diagnostic; do not present it as pre-specified or mix it silently with the broader model-comparison family already reported in the manuscript.

## Overall decision

Three additions improve the paper: pooled nonlinear HGB, the date-permuted DVOL placebo, and block-level attribution. The tested economic strategy does not improve the case and should not be presented as supportive evidence.
