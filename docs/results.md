# Results

All numbers below were produced by this repository on the real M5 data (Kaggle `m5-forecasting-accuracy`,
`sales_train_evaluation.csv`, 1,941 days). Nothing is copied from papers or estimated.

## Forecast accuracy (rolling-origin backtest)

Protocol: 3 folds; fold *k* trains on days <= cutoff and predicts the next 28 days; cutoffs d_1857, d_1885, d_1913.
Reported values are the mean over folds. Reproduce with `demandops evaluate --n-series N`.
Series are a deterministic random sample (seed 42).

**3,000 series (about 10 % of the data)**

| Model | WRMSSE | MAE | Bias | Fit + predict time (3 folds) |
|---|---|---|---|---|
| Seasonal naive | 0.9094 | 1.2062 | -4.52 % | 0.1 s |
| AutoETS (store x dept, top-down) | 0.7523 | 1.0141 | -3.71 % | 46.5 s |
| LightGBM (global, tweedie, level-corrected) | **0.6564** | 1.0384 | -1.63 % | 45.7 s |

**300 series**

| Model | WRMSSE | MAE | Bias | Fit + predict time (3 folds) |
|---|---|---|---|---|
| Seasonal naive | 0.9672 | 1.0594 | -3.85 % | 0.0 s |
| AutoETS (store x dept, top-down) | 0.7579 | 0.8727 | -1.60 % | 43.5 s |
| LightGBM (global, tweedie, level-corrected) | 0.7471 | 0.8998 | -1.41 % | 5.1 s |

How to read these honestly:

* **WRMSSE is computed on the sampled series** (aggregates are those of the sample, weights from the last 28
  training days), so it is **not comparable to the official M5 leaderboard**, where a solid LightGBM is ~0.55-0.65.
  It is comparable *between the rows of one table*.
* A global model profits from more series: LightGBM goes from a tie with AutoETS at 300 series to a clear win at
  3,000. A backtest on all 30,490 series has **not** been run (see "Scale" below for why).
* On MAE, AutoETS is slightly better than LightGBM in both tables; the LightGBM advantage is in the WRMSSE
  (weighted, scaled) metric.
* All models under-forecast a little (negative bias) because M5 demand grows over time. LightGBM carries a **level
  correction** = sum(actual)/sum(raw forecast) over earlier folds, clipped to [0.85, 1.20] (`bias_scale`, ADR 0001).
  It is learned only from forecasts whose actuals are already known at each fold's cutoff (fold 1 is uncorrected,
  fold 2 uses fold 1, ...), so the backtest bias the promotion gate sees is what production does. Without it a
  fresh-stack run produced a first model at -5.5 % bias and the |bias| < 5 % gate refused to promote it.
  The baselines are reported raw (uncorrected).
* The 28-day lags 7/14 of the brief were replaced by 35/42 (ADR 0001); this costs some accuracy.

## Business simulation: reorder policy (out of sample)

Order-up-to policy, replayed over the last 28 observed days of the 3,000-series sample
(`demandops reorder-sim`). The forecast-driven policy orders to cover forecast demand over the lead time plus a
z = 1.28 safety stock; the naive policy uses the trailing 28-day mean. The model is **fitted before the replayed
window** (a first version replayed with the live champion, which had already seen those days, and looked better:
stock-out 8.65 % - that number was in-sample and is discarded).

| Lead time | Policy | Stock-out rate (units not served) | Avg inventory | Overstock cost |
|---|---|---|---|---|
| 3 days | forecast-driven | **10.20 %** | 3.29 | 4,292 |
| 3 days | naive | 10.90 % | 3.31 | 4,294 |
| 7 days | forecast-driven | 9.29 % | 6.05 | 8,618 |
| 7 days | naive | 9.30 % | 5.99 | 8,491 |

Reading: at equal inventory the forecast policy serves ~0.7 points more demand at a 3-day lead time (a ~6 % relative
cut in lost sales) and is a tie at 7 days. The gain is modest because the series are intermittent and the policy is
simple; do not read it as a large saving. `sigma` is the std of demand in the 28 days before the window, a proxy for
forecast-error spread. Overstock cost = leftover-stock unit-days x 0.02.

## Scale (all 30,490 series, real M5 data, laptop: 12 cores / 16 GB RAM)

| Step | Result |
|---|---|
| Nightly feature publish (`build_horizon_table`, 450-day window, chunks of 5,000 series) | 853,720 rows (30,490 x 28 days) in **49 s**, peak **3.2 GB** |
| Nightly batch forecast (score + write to the forecast table, SQLite here) | 853,720 rows in **6 s**, peak 3.2 GB (model trained on the 300-series sample: this measures throughput, not accuracy) |
| Training memory (one global LightGBM, 900-day window) | **6.6 GB peak at 10,000 series** (about 0.66 GB per 1,000 series), so all series would need ~20 GB. Not run. |

Consequences: ingest, batch forecast and monitoring fit a 16 GB VPS at full scale; **retraining on every series does
not** (needs ~20 GB at 900 days; use `HISTORY_DAYS`/`train_days` to shrink the window, sample series, or give the
`retrain` pod a 24-32 GB node). The chunked feature builder produces output identical to the one-shot builder
(`test_chunked_horizon_table_equals_the_one_shot_table`).

## Platform measurements

| Measurement | Value | Conditions |
|---|---|---|
| Inference p95 at 10-25 req/s | 12 ms | before request caching, 1 container / 1 uvicorn worker |
| Inference ceiling | ~58 req/s | same, before optimisation (CPU bound: pandas matrix building) |
| Inference p95 at 50 req/s | **7.8 ms** (median 4.5 ms), 0 errors / 1,501 | after caching + date-lookup fix, 300 distinct series, 30 s |
| Inference p95 at 120 req/s | **6.3 ms**, 0 errors / 3,600 | same |
| Inference ceiling | ~200 req/s per worker (latency explodes beyond: 300 req/s offered -> 202 served, p95 1.1 s) | same |
| Gate-style in-process p95 (cold path, no cache) | 3-9 ms | `retrain` DAG measurement |
| Reproducibility | WRMSSE identical to 7 digits (0.7862368) between the laptop run and the containerised run | same seed and data |
| Drift -> alert | A 4x demand shock injected through the simulator (19 days) gave forecast bias -74 % and sales PSI 0.26; Prometheus fired `DemandForecastBias` and `DemandTargetDrift` (label `action=retrain`) within one rule evaluation after the monitoring DAG pushed its gauges | compose stack, monitoring DAG run by Airflow 3.2.2; the old triggers (feature share, WAPE ratio 1.11) did **not** fire - see architecture.md |
| Retrain DAG on that shock | gates answered "kept champion": challenger WRMSSE on the shared holdout window -1.4 % vs champion, |bias| 24 % | neither model can foresee a shock through 28-day-old features; the alerts are what flags it |
| Data -> promoted model | ~50 s for 300 series (laptop, `demandops retrain`: train + backtest + gates + promote) | **not measured on the cluster** with Airflow and the full data |
| 1 / 3 / 5 pods (HPA) | **not measured** | needs the k3d/VPS cluster; compose runs a single container |
| Canary rollback time | **not measured** | needs a cluster with Argo Rollouts |
| Chaos-lite (kill a pod under load) | **not measured** | needs >= 2 replicas |
| VPS monthly cost | **not measured** | no VPS was provisioned |

k6 script: `tests/load/infer.js`. Numbers above were taken with `grafana/k6` inside the compose network,
so they exclude the host network path.
