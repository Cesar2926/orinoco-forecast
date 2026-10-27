# Feature: walk-forward baseline comparison

Objective: evaluate a model and its comparison runner (the baseline) on the same walk-forward folds, with per-fold statistics on each split's held-out segment plus cross-fold dispersion and improvement. Branch: `main`. Delivery: `exception-ok` (local work-unit commits only; no PR requested and no push). Test runner: `uv run pytest`; lint: `uv run ruff check .`.

## Specs

- **S1 — Same folds for both runners.** Quote: "debes evaluar el baseline utilizando los mismos pliegues (folds) de validación del walk-forward split que utilizas para tu modelo complejo" and "Comparar ambos enfoques sobre las mismas ventanas garantiza que evalúas bajo las mismas condiciones". The model and the comparison runner are fitted and scored on the same `cutoff`, the same training history, the same supervised windows and the same validation segment of every fold.
- **S2 — The comparison runner is the baseline.** Quote: "tomando el modelo con que se compara como el baseline". `--against` (default `naive`) supplies the reference; `rel_mae = mae / mae_against` is the per-fold ratio and improvement is derived from the paired errors.
- **S3 — Per-fold improvement.** Quote: "Mejora \% = ((Error_baseline - Error_modelo) / Error_baseline) x 100". Reported per fold and per station for MAE and RMSE, which are non-negative and share units; a zero baseline error yields a null improvement instead of `inf`.
- **S4 — Statistics on the split's held-out data.** Quote: "agregando la estadsiticas sobre la data de test del split". Every fold reports `mae, rmse, mase, nse, kge, rel_mae` per station, per horizon step (`1..14`) and for `all`, scored only on `cutoff + 1 .. cutoff + horizon`.
- **S5 — Media and desviación across folds.** Quote: "si el modelo avanzado no supera consistentemente al baseline en la media y la desviación estándar de los errores". The comparison reports, per station x model x horizon, mean/std of both runners' errors, the paired difference, mean/std/min/max of the improvement percentage, wins/ties/losses, `win_rate` and an exact two-sided sign test.
- **S6 — No reserved-test contamination.** Every cutoff satisfies `validation_end < test_start = 2023-01-01`; the reserved block is never read by the sweep. Quote: "no utilizar el período de test durante la selección".
- **S7 — Persistence without breaking existing artifacts.** Outputs go to `data/predictions/<model>/backtest/<against>/{fold_predictions,fold_statistics,comparison}.parquet`. `predictions.parquet` and `statistics.parquet` keep their current single-cutoff schema because the dashboard and `validate_predictions` read them by exact name.
- **S8 — Checks.** `uv run pytest` and `uv run ruff check .` finish clean; a real-data smoke run reports 24 folds, `n = 14` per fold and station, and no row after 2022-12-31.

## Tasks

- T1 — Protocol config, walk-forward splits builder and the evaluation import cycle. Route: inline. Status: done; 7 backtest tests, 24 neighbouring evaluation tests pass.
- T2 — `run_backtest` over the existing `run_runners` path plus `BacktestResult`. Route: inline.
- T3 — `summarize_comparison` aggregation and sign test. Route: inline.
- T4 — Backtest persistence and the `--folds` CLI. Route: inline.
- T5 — Documentation. Route: inline.
- T6 — Full suite, lint and real-data smoke run. Route: inline.

## Log

- L1 — User request (verbatim): "crea un plan para agregar esta comparacion

Para que la comparación sea justa y metodológicamente sólida, debes evaluar el baseline utilizando los mismos pliegues (folds) de validación del walk-forward split que utilizas para tu modelo complejo.¿Por qué evaluarlo sobre los mismos splits?Comparación en condiciones idénticas (Apple-to-apples): En series temporales, algunos períodos son mucho más difíciles de predecir que otros debido a volatilidad, cambios de tendencia o eventos atípicos. Comparar ambos enfoques sobre las mismas ventanas garantiza que evalúas bajo las mismas condiciones del mercado o del entorno.Consistencia temporal: Te permite comprobar si tu modelo avanzado supera al baseline de forma consistente en el tiempo o si solo fue superior en un período específico.Medición del valor agregado: Puedes calcular la mejora porcentual en cada ventana $k$:$$\text{Mejora \%} = \left(\frac{\text{Error}_{\text{baseline}} - \text{Error}_{\text{modelo}}}{\text{Error}_{\text{baseline}}}\right) \times 100$$Flujo de trabajo en el bucleEn cada iteración $k$ del walk-forward:Tomar el split actual: Extraer $X_{train}^{(k)}, y_{train}^{(k)}$ y $X_{val}^{(k)}, y_{val}^{(k)}$.Generar la predicción del Baseline:Si es un baseline ilativo/Persistence ($y_{t+1} = y_t$), predices el último valor conocido de $y_{train}^{(k)}$.Si es un baseline estadístico (ej. media), calculas la media de $y_{train}^{(k)}$ y la aplicas a $X_{val}^{(k)}$.Generar la predicción del modelo complejo: Entrenas tu algoritmo y predices sobre $X_{val}^{(k)}$.Registrar métricas: Calculas el error (MAE, RMSE, etc.) de ambos sobre $y_{val}^{(k)}$ y guardas los dos valores.Al finalizar el recorrido, si el modelo avanzado no supera consistentemente al baseline en la media y la desviación estándar de los errores, significa que la complejidad adicional del modelo no está aportando un valor predictivo real.¿De qué naturaleza son tus datos o qué regla lógica estabas pensando usar como baseline?

tomando el modelo con que se compara como el baseline y agregando la estadsiticas sobre la data de test del split"
- L2 — User follow-up (verbatim): "relee el codigo y ajusta el plan".
- L3 — User authorization (verbatim): "implementa".
- L4 — Design from the code review: `run_runners` already builds one window table per cutoff and hands it to both runners, so the sweep reuses it instead of refactoring `execution.py`. `StatisticsResult` selects only `STATISTICS_COLUMNS`, so fold tags are added in the backtest layer. The `execution` -> `backtest` import becomes local because `backtest` now imports `run_runners`.
- L5 — Protocol values: 24 windows, 30-day step, last cutoff 2022-12-17, first cutoff 2021-01-26; SeasonalNaive has history back to 2020-01-27. Data spans 2009-01-01..2025-03-31, so the reserved test block is 2023-01-01..2025-03-31 (821 days) and stays untouched.
