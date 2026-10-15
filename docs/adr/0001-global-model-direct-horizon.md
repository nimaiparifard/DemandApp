# ADR 0001 - One global LightGBM with direct 28-day features

**Status:** accepted

## Context
The task is a daily 28-day forecast for 30,490 series. The star of the project is the platform, so the model
must be simple, cheap to retrain and cheap to serve.

## Decision
* One global LightGBM (tweedie objective, variance power 1.05) trained on all series.
* **Direct** strategy: all lags/rolling windows are >= 28 days old (`lag_28/35/42`, rolling windows shifted by 28).

## Consequences
* (+) No recursion, so no error accumulation and one model call per request.
* (+) Features for the whole horizon exist the day the data lands, so they can be precomputed, stored in a feature
  store and looked up online; training and serving share `build_features`.
* (+) Leakage is testable (see `test_no_leakage_...`).
* (-) The spec lists lags 7/14. These are unusable in a direct 28-day model (they would be unknown at prediction
  time for days 8-28), so they were replaced by 35/42 (same weekday). Short-lag information is lost; a recursive or
  per-horizon model could recover some accuracy at higher cost.
* Tweedie power 1.05 instead of 1.1: a sweep on a 300-series sample cut the systematic under-forecast
  (bias -4.8 % -> -1.8 %) and improved WRMSSE, which matters because the promotion gate requires |bias| < 5 %.

## Addendum - level correction (bias_scale)
After the price-mean feature became a fixed 365-day window, a clean-slate run produced a first model with -5.5 % bias,
which the |bias| < 5 % gate (correctly) refused to promote, leaving the platform without a champion.
Demand in M5 trends upward, so every model trained on the past under-forecasts. The model now multiplies its output by
`sum(actual) / sum(raw forecast)` measured on **earlier** backtest folds (clipped to [0.85, 1.20]); the correction
used for each fold's score is learned only from forecasts that were already matured at that fold's cutoff.
Effect on the 3,000-series backtest: bias -2.66 % -> -1.63 %, WRMSSE 0.6649 -> 0.6564 (the price-feature change is
included in that comparison, so the two effects are not separated).
