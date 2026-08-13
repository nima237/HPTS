# Final leakage-safe HPTS package

This directory is the authoritative, standalone analysis package for the paper.
The study forecasts **log realised variance** over the exact next 24 hours.

## Authoritative files

- `model_data.csv`: 31,965 model-ready asset-day observations for 17 assets.
- `data_dictionary.csv`: corrected variable definitions and missing-value audit.
- `run_final_analysis.py`: one-command final pipeline.
- `FINAL_HPTS_analysis.ipynb`: complete runnable notebook containing the primary and added analyses.
- `run_requested_tests.py`: Hoang-Baur baselines, 1/3/7-day horizons, and 1,000-draw IV permutation test.
- `requested_tests/`: saved predictions and numerical results for the added analyses.
- `FINAL_RESULTS.md`: concise interpretation of the locked results.
- `final_results/`: paper tables, supplementary checks, predictions, metadata and figures.
- `requirements.txt`: required Python packages.

Older files in this directory are retained only for traceability. For the paper,
use `FINAL_HPTS_analysis.ipynb`, `run_final_analysis.py`, `FINAL_RESULTS.md`, and
`final_results/`.

## Locked protocol

- Train: observations before 2023-01-01.
- Hyperparameter validation: calendar 2023 only.
- Untouched holdout: 2024-01-01 onward.
- Forecast horizon: exact next 24 hours.
- Target: `log(sum of squared hourly log returns)` over the target window.
- Primary benchmark: per-asset HAR.
- Primary loss: MSE on log realised variance.
- Secondary loss: QLIKE on realised variance.
- Refit cadence: 60 days.
- Option timing: current UTC day's BTC/ETH DVOL open only.
- Proposed model: one hierarchical ridge regression; it is not an ensemble.
- Seed: 77 for bootstrap, stochastic competitors and permutation inference; ridge is deterministic.

## Final model

`HPTS_Final` contains common effects, asset fixed effects and regularised
asset-specific deviations for HAR, cross-sectional, tail/path/liquidity and
causally timed BTC/ETH option-implied predictors. The BTC--ETH IV spread is
omitted because the no-spread specification had lower 2023 validation MSE.

## Run

Create an environment and install the locked dependencies:

```bash
python -m pip install -r requirements.txt
python run_final_analysis.py
python run_requested_tests.py all
```

The script recreates `final_results/`. The notebook contains the same pipeline
and can be run from this directory. A complete run includes 17 leave-one-asset-
out refits and can take several minutes.

`model_data.csv` also contains `y_3d` and `y_7d`, exact forward 72-hour and
168-hour log-realised-variance targets. The longer-horizon code censors training
origins whose target window has not ended at the refit cutoff.

## Scope of claims

The main paper is a forecasting study. Risk-management results are not part of
the headline contribution because paired VaR/ES loss improvements were not
statistically decisive. Raw realised-volatility performance is reported only as
a scale diagnostic; the primary result concerns log realised variance.
