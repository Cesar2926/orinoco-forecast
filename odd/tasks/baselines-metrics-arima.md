# Feature: baselines, metrics, and ARIMA

Objective: deliver phases 4-6 of `docs/plan_entrenamiento.md` (Naive/SeasonalNaive baselines, MAE/RMSE/sMAPE/MASE metrics, and an ARIMA adapter). Branch: `main`. Delivery: `exception-ok` (local work-unit commits only; no PR requested and no push). Test runner: `uv run pytest`; lint: `uv run ruff check .`.

## Specs

- **S1 — Baselines.** Implement `Naive` and `SeasonalNaive` in `src/orinoco_forecast/models/statistical/baselines.py`. Plan quotes: "**Naive**: repite el último nivel observado." and "**Seasonal Naive**: repite el nivel de hace 365 días." Both must use exactly the same cutoffs and horizon as every trainable model, and return exactly `horizon` prediction rows per station with `model, unique_id, cutoff, ds, horizon_step, y_pred`.
- **S2 — Metrics.** Complete `src/orinoco_forecast/evaluation/metrics.py` with MAE, RMSE, sMAPE and MASE with seasonality 365. Plan quote: "MAPE solo debe reportarse si se define claramente el tratamiento de valores cercanos a cero." MAPE is implemented with an explicit guard: it raises when any `y_true` is zero instead of dividing by zero. MASE scaling is the mean absolute seasonal difference computed on the training history only.
- **S3 — ARIMA.** Complete `src/orinoco_forecast/models/statistical/arima.py`. Plan quote: "recibir solo el período de entrenamiento de cada corte; producir exactamente 14 predicciones; guardar predicciones con su fecha, estación, modelo y corte; no utilizar el período de test durante la selección." The adapter fits one univariate model per station with only `ds <= cutoff`, supports the fixed `order`, `seasonal_order` and `auto_order` from `configs/models/arima.yaml`, and returns exactly `horizon` rows per station carrying the identity columns. Persisting to `data/predictions/<modelo>/` is phase 9 work and is intentionally out of scope.
- **S4 — Reserved test.** No fit, selection or validation uses observations on or after `test_start = 2023-01-01`. `fit` trims to the cutoffs produced by `walk_forward_splits`, and forecasts are dated `cutoff + 1 .. cutoff + horizon`.
- **S5 — Checks.** `uv run pytest` and `uv run ruff check .` finish clean; a real-data smoke check confirms ARIMA produces 14 finite predictions per station at a validation cutoff.

## Tasks

- T1 — Forecasting protocol plus Naive and SeasonalNaive baselines with tests. Route: inline. Commit: `251faac`.
- T2 — MAE, RMSE, sMAPE, MAPE and MASE metrics with tests. Route: inline. Commit: `8df779a`.
- T3 — Model registry plus ARIMA adapter (fixed order and optional AutoARIMA) with tests. Route: inline. Commit: `f4b74b9`.
- T4 — Full suite, lint, and real-data ARIMA smoke check at a validation cutoff. Route: inline. Verification: complete; evidence in L7-L8.

## Log

- L1 — User request (verbatim): "avanza hasta arima".
- L2 — Scope decision: phases 4, 5 and 6 only. Phase 9 (`evaluation/backtest.py` and persistence under `data/predictions/`) is excluded; predictions already carry `model`, `unique_id`, `cutoff`, `ds` and `horizon_step` so phase 9 only adds `y_true` and writes the files.
- L3 — Environment probe: `statsmodels.tsa.arima.model.ARIMA` accepts NaN endog (Kalman filter), `pmdarima 2.1.1` imports, and an 18k-point fixed-order fit takes about 3.5 s. ARIMA can therefore consume `palua` with its known leap-day nulls without imputation.
- L4 — T1 evidence: baseline tests cover last observed value, trailing nulls, cutoff trimming (tampering after the cutoff leaves predictions unchanged), 365-day seasonal lookup, missing seasonal values, multi-station stacking, unfitted use, invalid horizon, duplicate dates and empty history. Full suite: 54 passed, 1 skipped (metrics placeholder at that point); lint clean. Commit `251faac`.
- L5 — T2 evidence: metric tests pin MAE, RMSE, sMAPE (0/0 term), MAPE (0.375 case and near-zero rejection), MASE against a 365-day scale, zero-scale rejection, null/mismatched/non-finite inputs and duplicate dates. `tests/test_metrics.py`: 11 passed. Commit `8df779a`.
- L6 — T3 evidence: ARIMA tests cover the prediction schema and 14-step horizon, cutoff trimming (tampering after the cutoff leaves predictions unchanged), NaN endog for `palua`, multi-station stacking, AutoARIMA on a short clean series, AutoARIMA refusing NaN, all-null stations, unfitted use and order validation. Registry tests cover the two baselines, the ARIMA YAML config (`order=(2,1,2)`, `seasonal_order=(0,0,0,0)`, `auto_order=False`), unknown names and missing config files. Full suite: 79 passed in 9.9 s; lint clean.
- L7 — T4 real-data smoke (last validation cutoff `2022-12-17`, validation `2022-12-18..2022-12-31`, no test data touched). `ciudad_bolivar`: Naive MAE 1.656, SeasonalNaive 0.776, ARIMA 0.737 (5.12 s fit); `palua`: Naive 1.706, SeasonalNaive 0.412, ARIMA 1.046 (7.55 s fit). Every model returned 14 finite predictions. statsmodels emitted non-stationarity/invertibility and convergence warnings for the configured `(2,1,2)` order; predictions remained finite and the warnings are reported as a model-quality observation, not hidden.
- L8 — Final verification: `uv run pytest` 79 passed; `uv run ruff check .` clean; no writes to `data/raw/` or `data/predictions/` (persistence is phase 9). The test period starting at `2023-01-01` was never used for fitting or selection.
- L9 — Post-review fix: a full run over the 24 validation windows revealed that `SeasonalNaive` failed for `palua` at cutoffs whose target maps to the null `2020-02-29`. Policy: use the last non-null level at or before the seasonal date and emit `DataQualityWarning`; no global imputation. Added `test_seasonal_naive_falls_back_to_the_last_available_level`. The full flow then completed: baselines over 24 windows and ARIMA over the last 2 (for speed). Two-window ARIMA MAE `ciudad_bolivar` 0.424, `palua` 0.877; 24-window Naive MAE 0.567 and 0.488; 24-window SeasonalNaive MAE 1.825 and 1.543. Window counts differ, so this is a smoke, not a model comparison.
