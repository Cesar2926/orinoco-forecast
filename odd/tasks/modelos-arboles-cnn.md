# Feature: modelos-arboles-cnn

## Objective

Implement the two model families assigned to this team member (tree ensemble with
XGBoost and convolutional network with TCN), the shared evaluation harness they need
(metrics, model-ready dataset, baselines, cross-validation), and a Streamlit dashboard
to inspect data and model behaviour.

## Problem / why

The models must be evaluated under the protocol in the README (h=14, 24 windows every
30 days before `test_start`, final test used once). Teammates' models will reuse the
same harness, so it must be model-agnostic. Daniel's `add_exogenous` already joins rain
and ONI onto a long table; it is reused, not duplicated.

## Scope

- In: dependencies (statsforecast, mlforecast, neuralforecast, utilsforecast, streamlit),
  `evaluation/metrics.py`, `features/` model frame, `evaluation/backtest.py`,
  `scripts/run_model.py`, `models/trees/tree.py`, `models/neural/cnn.py`, model configs,
  Streamlit dashboard, tests.
- Out: ARIMA, LSTM, TFT, TimesFM (teammates), final test evaluation (run only when the
  team decides), hyperparameter search (later slice), changes to Daniel's script.

## Constraints

- Polars for data handling; convert to pandas only at the neuralforecast boundary.
- `data/raw/` read-only. Predictions under `data/predictions/<model>/`.
- No evaluation on the final test block unless an explicit flag is passed.
- Same windows and refit policy for every model.

## Acceptance criteria

- Metrics MAE, RMSE, MASE, NSE, KGE tested against hand-computed values.
- Baselines, XGBoost and TCN produce cross-validation predictions and per-horizon
  metrics for `ciudad_bolivar` and `palua`.
- Dashboard shows series, quality labels, predictions vs actuals per cutoff and error by
  horizon for all models with saved predictions.
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` pass.

## Delivery strategy

stacked-to-main (user choice). Forecast: ~1200 authored lines across 4 slices.

| Slice | Branch | Content |
|---|---|---|
| 1 | `feat/modelos-1-base` | deps, metrics, model frame, backtest harness, baselines |
| 2 | `feat/modelos-2-xgboost` | XGBoost via mlforecast |
| 3 | `feat/modelos-3-tcn` | TCN via neuralforecast |
| 4 | `feat/modelos-4-dashboard` | Streamlit dashboard |

## Tasks

- [x] T1 Add dependencies — route: delegated writer (slice 1, 2+ non-trivial files).
      Commit `895cd4a`.
- [x] T2 Metrics with tests — delegated writer. Commit `e95eb86`.
- [x] T3 Model frame (cleaning, upstream covariates, Fourier terms, leakage guard) —
      delegated writer. Commit `efebf9c`.
- [x] T4 Backtest harness + baselines + `run_model.py` — delegated writer.
      Commit `4685e04`.
- [x] T5 XGBoost (slice 2) — delegated writer. Commit `1263378`.
- [x] T6 TCN (slice 3) — delegated writer. Commit `d3f8502` on `feat/modelos-3-tcn`.
- [x] T7 Streamlit dashboard (slice 4) — delegated writer. Commits `ddad677`, `7239ccd`, `879e427`.

## Progress / evidence

- Branch `feat/modelos-1-base` created from `main` (e3d301f).
- Slice 1 done by a delegated writer (route: delegated direct, writer trigger: 2+
  non-trivial files per task).
  - T1: statsforecast 2.1.1, mlforecast 1.1.0, neuralforecast 3.3.0, utilsforecast
    0.2.17; 47 packages added (ray 2.59, optuna 5.0, pytorch-lightning 2.6.6,
    pyarrow 26, fugue). pandas downgraded 3.0.6 -> 2.3.3 (statsforecast/mlforecast pin
    `pandas<3`); numpy 2.5.3, polars 2.0.0, torch 2.14.1 unchanged.
  - T2: RED = ImportError at collection; GREEN = 7 passed.
  - T3: RED = ModuleNotFoundError (conftest); GREEN = 9 passed. Stage A: 37,366 rows,
    1974-01-01..2025-02-24, 108 imputed rows. Stage B: 32,252 rows from 1981-01-01.
  - T4: RED = ImportError at collection; GREEN = 5 passed.
- Checks: `uv run pytest -q` 49 passed; `ruff check` and `ruff format --check` clean.
- Follow-up (parent decision): MASE is horizon-matched. MASE_h = MAE_h / in-sample MAE
  of lag-h persistence; "all" = mean of MASE_h. `mase_season` removed. Commits
  `6d5b954` (fix, RED = TypeError on removed `season` arg, GREEN = 8 passed; includes
  the user-approved one-line edit in `evaluation/backtest.py`) and `7c80eb6` (docs:
  README MASE row and leakage rule 3). Full suite: 50 passed.
- Baselines, stage A, 24 windows, all horizons (horizon-matched MASE):

  | Station | Model | MAE | RMSE | MASE | NSE | KGE |
  |---|---|---|---|---|---|---|
  | ciudad_bolivar | Naive | 0.5673 | 0.7675 | 1.0007 | 0.9711 | 0.9797 |
  | ciudad_bolivar | SeasonalNaive | 1.8237 | 2.2505 | 5.5111 | 0.7513 | 0.8540 |
  | palua | Naive | 0.4883 | 0.7016 | 1.1613 | 0.9592 | 0.9686 |
  | palua | SeasonalNaive | 1.5426 | 1.9406 | 5.6664 | 0.6881 | 0.8148 |

  Naive MASE: CB h=1 1.046, h=14 1.025; Palua h=1 1.041, h=14 1.185.
- Decisions: interpolation is global, guarded by `assert_no_imputation_near_cutoffs`
  (documented in README rule 3 as equivalent to per-window imputation).

- Slice 1 review: full range (e3d301f..HEAD) refused with `lens_context_budget_exceeded`
  (uv.lock). Reduced scope to code commits (895cd4a..7c80eb6, 14 files, 1001 lines):
  medium risk, consent granted, reliability lens approved, acknowledged
  (lineage review-a81624af957f2067). Commit 895cd4a (deps/uv.lock) not reviewed.
- Open: Palúa Naive MASE = 1.16 on validation windows, so "MASE < 1 per station" can
  reject a model that does beat persistence there; proposed criterion: MAE relative to
  Naive on the same windows.

- Slice 2 / T5 (route: delegated direct, writer trigger: runner + config + script +
  registry + tests). RED = ImportError at collection (`XGBoostRunner`); GREEN = 7
  passed in tests/test_tree.py; full suite 57 passed; ruff check/format clean.
  - Verified in mlforecast 1.1.0: polars input with `freq="1d"`; direct strategy
    aligns dynamic exogenous columns to the target date (t+h) in training and in
    prediction, so covariate lags >= horizon are leak-free (runner raises otherwise);
    `refit=False` fits once and predicts later windows with `new_df=train`;
    `models_["XGBoost"]` is a dict of 14 boosters (0-indexed horizons).
  - Covariates: Fourier (`sin_*`, `cos_*`) future; every other covariate (upstream,
    and rain/ONI in stage B) pre-lagged 14/21/28/56 days; first 56 rows per series
    dropped (warm-up). Feature importance: XGBoost total_gain normalised per horizon
    model, averaged over the 14 models; written to
    `data/predictions/xgboost/feature_importance.parquet` (development rows only).
  - Stage A, 24 windows, wall time 64 s (CV + importance fit):

    | Station | Model | MAE | RMSE | MASE | NSE | KGE | MAE h=1 | h=7 | h=14 |
    |---|---|---|---|---|---|---|---|---|---|
    | ciudad_bolivar | Naive | 0.5673 | 0.7675 | 1.0007 | 0.9711 | 0.9797 | 0.0854 | 0.5125 | 1.0633 |
    | ciudad_bolivar | XGBoost | 0.2706 | 0.3709 | 0.4880 | 0.9932 | 0.9951 | 0.0385 | 0.2754 | 0.4398 |
    | palua | Naive | 0.4883 | 0.7016 | 1.1613 | 0.9592 | 0.9686 | 0.0771 | 0.4346 | 0.8908 |
    | palua | XGBoost | 0.3288 | 0.4478 | 0.8178 | 0.9834 | 0.9901 | 0.0657 | 0.3559 | 0.4954 |

  - Top importances: rolling_mean_lag1_window_size7 0.180, cos_1 0.175, lag1 0.072,
    caicara_lag56 0.056, rolling_mean_lag1_window_size30 0.048.
  - Pending: native review for commit `1263378` (parent decides).

- Accepted change (user decision): minimum selection requirement is now MAE relative to
  persistence (model MAE / Naive MAE on the same validation windows) < 1 per station;
  MASE stays reported. Reason: Naive scores MASE 1.16 at Palua on 2021-2022, so the
  MASE threshold depends on how hard the validation period was. Route: delegated
  writer on `feat/modelos-2-xgboost`. RED = ImportError (`relative_to_baseline`);
  GREEN = 11 passed in tests/test_metrics.py; full suite 60 passed. Commits `e8698cb`
  (feat: `relative_to_baseline` + `run_model.py` prints `rel_mae`) and `c9ea55c`
  (docs: README criterion item 1 + `rel_mae` metrics row). `run_cv` frames carry no
  cutoffs, so `relative_to_baseline` checks equal `n` per unique_id/horizon and
  assumes the same `configs/backtest.yaml`.
  XGBoost rel_mae (all horizons): ciudad_bolivar 0.4770, palua 0.6735 (h=1/7/14:
  CB 0.451/0.538/0.414; Palua 0.853/0.819/0.556).

- Slice 3 / T6 (route: delegated direct, writer trigger: runner + config + registry +
  CLI + tests). Branch `feat/modelos-3-tcn` from `c9ea55c`. RED = ImportError at
  collection (`TCNRunner`); GREEN = 7 passed in tests/test_cnn.py (incl. an empirical
  leakage test); full suite 67 passed; ruff check/format clean.
  - Verified in neuralforecast 3.3.0: polars input with `freq="1d"`; `TCN` is not
    recurrent, so `_parse_windows` slices `hist_exog` to `[:, :input_size]` (only
    days up to each cutoff); `TCN.forward` uses `futr_exog` after the cutoff only for
    Fourier; robust scaler masks the last h days. Empirical check: shocking upstream
    values after the last cutoff leaves forecasts identical; shocking the last cutoff
    day changes only the last window.
  - `cross_validation(val_size=365, refit=False)`: one fit on data up to first cutoff
    minus 365 days; the 365 days ending at the first cutoff are only for early stopping
    (CV path does not check val_size>0, so the runner raises if patience>0 and
    val_size<h). Trainer: accelerator cpu/gpu by CUDA, progress bar/logger/summary off,
    deterministic "warn".
  - Seeds [1,2,3] as three aliased TCN models in one NeuralForecast (same windows);
    `run_model.py` prints per-seed rows plus mean ± std; `--max-steps` override.
  - Stage A, 24 windows, max_steps 1000 (early stopping never fired), wall time 4m30s
    on CPU:

    | Station | Model | MAE | rel_mae | MASE | NSE | KGE |
    |---|---|---|---|---|---|---|
    | ciudad_bolivar | TCN s1/s2/s3 | 0.2880/0.2829/0.3441 | 0.508/0.499/0.607 | 0.649/0.626/0.767 | | |
    | ciudad_bolivar | TCN mean ± std | 0.3050 ± 0.0340 | 0.5377 ± 0.0599 | 0.6808 ± 0.0759 | 0.9917 ± 0.0017 | 0.9875 ± 0.0046 |
    | palua | TCN s1/s2/s3 | 0.4260/0.3868/0.4156 | 0.873/0.792/0.851 | 1.116/1.026/1.136 | | |
    | palua | TCN mean ± std | 0.4095 ± 0.0203 | 0.8386 ± 0.0416 | 1.0925 ± 0.0588 | 0.9749 ± 0.0012 | 0.9760 ± 0.0040 |

    Mean MAE h=1/7/14: CB 0.157/0.291/0.498 (Naive 0.085/0.513/1.063); Palua
    0.145/0.435/0.613 (Naive 0.077/0.435/0.891). TCN loses to persistence at h=1
    (rel ~1.85) and ties at Palua h=7.
  - Pending: native review for `e8698cb..c9ea55c` and `d3f8502` (parent decides).

- T7 Streamlit dashboard (branch `feat/modelos-4-dashboard`, route: delegated writer;
  trigger: 5+ new files incl. dependency change).
  - Commits: `ddad677` build: add streamlit (1.66.1; adds altair 6.3.0, starlette,
    uvicorn, pydeck, watchdog...; numpy 2.5.3 / polars 2.0.0 / torch 2.14.1 / pandas
    2.3.3 unchanged), `7239ccd` feat (dashboard + tests + `make dashboard`), `879e427`
    docs (README "Dashboard").
  - Layout: pure shaping in `src/orinoco_forecast/dashboard/data.py`, Altair builders in
    `charts.py` (Altair takes Polars frames directly), thin UI in `app/dashboard.py` with
    `st.cache_data` (prediction cache keyed by parquet mtimes). Read-only; reads only
    `data/predictions/<model>/{cv,metrics}.parquet` (never `final_test/`).
    `ORINOCO_PREDICTIONS_DIR` overrides the predictions root.
  - Tabs: Datos (levels + quality highlights, missing as bottom ticks, quality counts,
    optional rain/ONI), Validación (h=1..14 table with rel_mae<1 checkbox, seed mean ±
    std, selection table, MAE by horizon), Pronóstico por ventana (60 d history + 14 d
    after cutoff + forecasts, seed mean only visual, per-window MAE), Importancia (top 15
    XGBoost gain).
  - RED: `uv run pytest tests/test_dashboard.py` -> collection error
    `ModuleNotFoundError: orinoco_forecast.dashboard`. GREEN: 11 passed. Full suite
    78 passed; ruff check/format clean; headless server `/_stcore/health` = ok; browser
    render checked (no console errors).
  - Real-repo AppTest skips when predictions are absent (they are gitignored).
  - Run: `uv run streamlit run app/dashboard.py` or `make dashboard`.

## Phase 2 (user request 2026-10-25): complete the models on ONE branch

Branch `feat/arboles-cnn-final` (from `feat/modelos-4-dashboard`), several commits, one PR.
Final test is NOT evaluated: the team selects among all six models first.

- [x] T8 Tuning harness: Optuna search with equal budget per model on tuning windows
      placed BEFORE the validation windows (no tuning on the windows that decide the
      comparison); TCN option to model daily differences — delegated writer.
      Commits `ca3a3f8` (TCN difference), `62e1719` (tuning), `cc1e4a4` (fix: tuning
      grid 12x23), `23e97ff` (fix: TCN max_steps {500, 1000}).
- [x] T9 Run tuning for XGBoost and TCN; write tuned configs — delegated writer.
      Commits `0f0746f` (fix: reseed sampler on resume), `9b9d757` (tuned configs +
      reports/tuning CSVs).
- [x] T10 Final validation runs (stage A and B; baselines too) with tuned configs, TCN
      3 seeds; dashboard stage selector — delegated writer. Commits `25d4f2e` (build:
      optuna), `07e30ee` (feat: stage folders + selector).
- [x] T11 Results document `docs/modelos_arboles_cnn.md` for the report — delegated
      writer. Commit `998f9be`.

### T8 evidence (route: delegated writer; trigger: 5 non-trivial files)

- RED: `tests/test_tuning.py` and `tests/test_cnn.py` -> ImportError at collection
  (`predict_cv`, `prepare_frame`). GREEN: 7 tuning + 10 cnn passed; then one more
  reconstruction bug caught by RED (missing `.alias`, sum written to `_y_cut`).
  Full suite 88 passed; ruff check/format clean.
- Verified Optuna 5.0.0 (transitive dep of neuralforecast, not declared in
  pyproject): `create_study(storage, sampler, study_name, direction, load_if_exists)`,
  `TPESampler(seed=)`, `TrialPruned`, `study.MaxTrialsCallback(n, states=(COMPLETE,))`,
  `trial.FixedTrial`; SQLAlchemy 2.1.6 for sqlite storage. `artifacts/` gitignored.
- Budget counts COMPLETE trials only (pruned TCN configs are invalid, ~0.1 s); cap of
  10x attempts so a pruning-only space cannot loop.
- `backtest.predict_cv` = guard + runner CV + cutoff check without writing; `run_cv`
  reuses it, so the leakage guard runs on tuning cutoffs too.
- Deviation from the planned 12x30 tuning grid: the guard rejected it (imputed target
  gap 2020-03-25..2020-04-12, exclusion zone ~47 days > 30). Any 30-day grid ending at
  the first validation cutoff (2021-01-26) with >9 windows hits it. Chosen 12 x 23 days:
  cutoffs 2020-05-04..2021-01-12, targets up to 2021-01-26, no overlap. Regression test
  on the real stage A frame. Stage B has the same gap and passes too.
- Deviation: TCN `max_steps` choices {500, 1000}. One trial (input 112, 128 filters,
  1000 steps, 1 seed) took 524 s on CPU, so {1000, 2000} projected > 2 h for 20
  trials. First study moved to `artifacts/optuna/tcn_maxsteps_1000_2000.db`.
- XGBoost: one trial 110 s.

### T9 evidence (route: delegated writer; long runs in background)

- Studies (stage A, tuning windows 2020-05-04..2021-01-26, objective mean rel_mae):
  - XGBoost: 20 COMPLETE, 17.4 min of trial time (one trial in a first process +
    19 in 15m43s). Best #18 = 0.4864 (CB 0.3605, Palua 0.6123); untuned config on the
    same windows = 0.4936 (CB 0.3643, Palua 0.6229); median trial 0.5186.
    Best: n_estimators 900, learning_rate 0.1201, max_depth 6, subsample 0.8839,
    colsample_bytree 0.5034, min_child_weight 1.379, reg_lambda 0.2729,
    covariate_lags 14-21-28-56.
  - Defect found: trial 1 duplicated trial 0 (resume re-seeded TPE with 42 and redrew
    the first random point), so XGBoost had 19 distinct configurations. Fixed in
    `0f0746f` (seed offset by stored trials; RED = params equal, GREEN). Study not
    rerun (parent decides).
  - TCN: 20 COMPLETE + 9 PRUNED (receptive field > input), 34m10s wall, one seed.
    Best #25 = 0.5710 (CB 0.4562, Palua 0.6859): difference, input_size 112,
    kernel 2, dilations 1-16, encoder_hidden_size 32, learning_rate 0.002656,
    max_steps 1000. Difference beat level clearly: 16 difference trials best 0.571 /
    median 0.684; 4 level trials best 0.863 / median 1.345.
- Validation (stage A, 24 windows) before -> after tuning:

  | Station | Model | MAE | rel_mae | MAE h=1 / h=7 / h=14 |
  |---|---|---|---|---|
  | CB | XGBoost untuned | 0.2706 | 0.4770 | 0.0385 / 0.2754 / 0.4398 |
  | CB | XGBoost tuned | 0.2867 | 0.5055 | 0.0433 / 0.3025 / 0.4600 |
  | Palua | XGBoost untuned | 0.3288 | 0.6735 | 0.0657 / 0.3559 / 0.4954 |
  | Palua | XGBoost tuned | 0.3397 | 0.6956 | 0.0645 / 0.3656 / 0.5094 |
  | CB | TCN untuned (3 seeds) | 0.3050 ± 0.0340 | 0.5377 ± 0.0599 | 0.157 / 0.291 / 0.498 |
  | CB | TCN tuned (3 seeds) | 0.2959 ± 0.0171 | 0.5216 ± 0.0301 | 0.045 / 0.295 / 0.485 |
  | Palua | TCN untuned | 0.4095 ± 0.0203 | 0.8386 ± 0.0416 | 0.145 / 0.436 / 0.613 |
  | Palua | TCN tuned | 0.3750 ± 0.0213 | 0.7679 ± 0.0436 | 0.072 / 0.418 / 0.556 |

  Runtimes: run_model xgboost 39 s; run_model tcn 3m30s (CPU).
  TCN h=1 rel_mae: CB 1.83 -> 0.53, Palua 1.88 -> 0.94 (now beats persistence).
- Finding: tuned XGBoost is WORSE on validation (+6% rel_mae CB, +3% Palua). Its gain on
  the tuning windows (0.4936 -> 0.4864, 1.5%) was within noise and did not transfer.
  Options for the parent: keep tuned (protocol-consistent, no validation peeking) or
  revert to defaults; choosing by validation result would be selection on validation.
- `artifacts/optuna/tcn_maxsteps_1000_2000.db`: abandoned first TCN study (1 complete
  trial, 2 pruned, max_steps {1000, 2000}); not used; gitignored; safe to delete.
- Checks: `uv run pytest -q` 90 passed; ruff check/format clean.

### T10 evidence (route: delegated writer; trigger: 6 non-trivial files + runs)

- `uv add "optuna>=5.0.0,<6"`: lock adds only the requirement; no version changes.
- Outputs now `data/predictions/<model>/stage_<A|B>/` (final test under
  `stage_<X>/final_test/`); `stage_baseline_metrics` refuses other-stage or root-level
  baselines; `run_model.py` fails before training if same-stage baselines are missing
  (verified: `xgboost --stage B` before baselines B -> Spanish error). Importance per stage.
- RED: test_backtest collection ImportError (`stage_baseline_metrics`, `stage_dir`);
  test_dashboard 13 failed (TypeError on stage args). GREEN: 8 backtest, 14+1 skip
  dashboard; full suite with real predictions 97 passed.
- Stage B frame 32,252 rows 1981-01-01..2025-02-24 (rain_* x4 + oni); 24 cutoffs
  2021-01-26..2022-12-17 identical in both stages. Stage B uses stage A tuned params.
- Wall times (CPU): baselines 5 s / 5 s; xgboost A 43 s, B 139 s; tcn A 534 s, B 511 s.
- Stage A reproduces T9 tuned numbers exactly. Stage B: XGBoost CB 0.2850 (rel 0.502),
  Palua 0.3519 (0.721); TCN CB 0.3895 ± 0.0452 (0.687), Palua 0.3730 ± 0.0447 (0.764).
  A->B: XGBoost CB -0.6%, Palua +3.6%; TCN CB +31.6%, Palua -0.5%. Rain+ONI = 0.215 of
  XGBoost gain, best rank 16. Naive MAE identical across stages, MASE differs.
- Legacy root-level files left in place (parent decides): baselines/{cv,metrics},
  tcn/{cv,metrics}, xgboost/{cv,metrics,feature_importance}.parquet.
- Dashboard: sidebar radio (key `stage`), A vs B table in Validación; headless
  `/_stcore/health` ok on port 8599, stopped.

### T11 evidence

- `docs/modelos_arboles_cnn.md` (10 sections, numbers from stage_A/B parquet files and
  reports/tuning CSVs); README "Modelos" link + dashboard section per stage.

## Next step

Parent/user: delete legacy root-level prediction files; native review of
`25d4f2e..998f9be`; push/PR. Open: rerun XGBoost study (19 configs); stage B confound
(fewer years + new covariates); Palua upstream covariates vs Daniel's ARIMAX.

Previous: decide on the TCN short-horizon weakness (e.g. predict differences / larger input_size
in the tuning slice). T7 dashboard done; pending native review of `ddad677..879e427`
(parent decides) and push/PR (user decides).
