# Feature: baselines, metrics, and ARIMA

Objective: deliver phases 4-6 of `docs/plan_entrenamiento.md` (Naive/SeasonalNaive baselines, MAE/RMSE/sMAPE/MASE metrics, and an ARIMA adapter). Branch: `main`. Delivery: `exception-ok` (local work-unit commits only; no PR requested and no push). Test runner: `uv run pytest`; lint: `uv run ruff check .`.

## Specs

- **S1 — Baselines.** Implement `Naive` and `SeasonalNaive` in `src/orinoco_forecast/models/statistical/baselines.py`. Plan quotes: "**Naive**: repite el último nivel observado." and "**Seasonal Naive**: repite el nivel de hace 365 días." Both must use exactly the same cutoffs and horizon as every trainable model, and return exactly `horizon` prediction rows per station with `model, unique_id, cutoff, ds, horizon_step, y_pred`.
- **S2 — Metrics.** Complete `src/orinoco_forecast/evaluation/metrics.py` with MAE, RMSE, sMAPE and MASE with seasonality 365. Plan quote: "MAPE solo debe reportarse si se define claramente el tratamiento de valores cercanos a cero." MAPE is implemented with an explicit guard: it raises when any `y_true` is zero instead of dividing by zero. MASE scaling is the mean absolute seasonal difference computed on the training history only.
- **S3 — ARIMA.** Complete `src/orinoco_forecast/models/statistical/arima.py`. Plan quote: "recibir solo el período de entrenamiento de cada corte; producir exactamente 14 predicciones; guardar predicciones con su fecha, estación, modelo y corte; no utilizar el período de test durante la selección." The adapter fits one univariate model per station with only `ds <= cutoff`, supports the fixed `order`, `seasonal_order` and `auto_order` from `configs/models/arima.yaml`, and returns exactly `horizon` rows per station carrying the identity columns. Persisting to `data/predictions/<modelo>/` is phase 9 work and is intentionally out of scope.
- **S4 — Reserved test.** No fit, selection or validation uses observations on or after `test_start = 2023-01-01`. `fit` trims to the cutoffs produced by `walk_forward_splits`, and forecasts are dated `cutoff + 1 .. cutoff + horizon`.
- **S5 — Checks.** `uv run pytest` and `uv run ruff check .` finish clean; a real-data smoke check confirms ARIMA produces 14 finite predictions per station at a validation cutoff.

## Tasks

- T1 — Forecasting protocol plus Naive and SeasonalNaive baselines with tests. Route: inline. Commit: pending.
- T2 — MAE, RMSE, sMAPE, MAPE and MASE metrics with tests. Route: inline. Commit: pending.
- T3 — Model registry plus ARIMA adapter (fixed order and optional AutoARIMA) with tests. Route: inline. Commit: pending.
- T4 — Full suite, lint, and real-data ARIMA smoke check at a validation cutoff. Route: inline.

## Log

- L1 — User request (verbatim): "avanza hasta arima".
- L2 — Scope decision: phases 4, 5 and 6 only. Phase 9 (`evaluation/backtest.py` and persistence under `data/predictions/`) is excluded; predictions already carry `model`, `unique_id`, `cutoff`, `ds` and `horizon_step` so phase 9 only adds `y_true` and writes the files.
- L3 — Environment probe: `statsmodels.tsa.arima.model.ARIMA` accepts NaN endog (Kalman filter), `pmdarima 2.1.1` imports, and an 18k-point fixed-order fit takes about 3.5 s. ARIMA can therefore consume `palua` with its known leap-day nulls without imputation.
