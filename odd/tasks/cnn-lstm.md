# Feature: cnn-lstm

## Objective

Add a hybrid CNN-LSTM runner (`cnn_lstm`) that forecasts Ciudad Bolivar and Palua at
1..14 days under the same protocol as the booster and the TCN, so the team can compare
it with `run_model.py cnn_lstm --against naive --folds`.

## Problem / why

The team now wants a convolutional + LSTM hybrid instead of (or besides) the pure TCN.
`neuralforecast` 3.3.0 has `TCN` and `LSTM` but no hybrid, so the network is a custom
PyTorch module wrapped in the existing `Runner` contract.

## Scope

- In: `models/neural/cnn_lstm.py` (module + `CnnLstmRunner`), `configs/models/cnn_lstm.yaml`,
  registry entry `cnn_lstm`, tests, docs section, Optuna search (own script).
- Out: Daniel's LSTM stub (`nsmt`), TCN code, data pipeline, runner contract, `--folds`.

## Design

CNN-LSTM, not ConvLSTM (there is no spatial grid; four stations along one river).
Causal Conv1d blocks extract features from the input window, an LSTM reads them in order,
a linear head predicts the 14 daily changes; levels are rebuilt from the level at the
cutoff (same difference target and reconstruction as the TCN). Inputs: station levels via
`drop_covariates` (rain/ONI hidden, as for the TCN), 3 seeds, early stopping on the last
`val_size` days before the first cutoff. Scaling fitted on the input window only.

## Constraints

- No leakage: nothing after `cutoff` reaches fit or predict; test block (2023+) untouched.
- Same runner contract (`fit/predict`, `PredictionResult`, model columns `CNNLSTM_s<seed>`).
- Do not edit Daniel's files except registry (one factory).
- Equal Optuna budget (20 trials) as the other models.

## Open decisions (team)

- Replace the TCN or keep it as ablation (recommended: keep as reference).
- Who owns the pure LSTM (`nsmt`).
- Custom PyTorch accepted despite the "Nixtla only" rule (no hybrid in Nixtla).

## Tasks

- [x] T1 Module + runner + config + registry + tests (RED first). Delegated fork (route: delegated direct; trigger: 2+ non-trivial files). Commit: see Progress.
- [x] T2 Optuna search (20 trials) and apply best params. Delegated fork (route: delegated direct; trigger: new script + runner fix + 1 h 47 min of trials). Commits: see Progress.
- [x] T3 Validation with `--folds` vs naive, booster, TCN and ARIMAX; docs and notebooks updated. Commits: see Progress.

## Delivery strategy

single-pr on `feat/cnn-lstm` (from main ff458d1), separate commits per task.

## Progress / evidence

- Branch created from main ff458d1 (PR #8 merged).
- T1 RED: `tests/test_runner_cnn_lstm.py` -> ModuleNotFoundError (module absent). GREEN: 12 new tests
  + catalog test, 21 passed; full suite 274 passed, 2 skipped; ruff check clean.
- T1 design: `models/neural/cnn_lstm.py` (CnnLstm: left-padded Conv1d+ReLU blocks -> LSTM -> linear head;
  `CnnLstmRunner`, `reconstruct_levels`, `robust_scale`, `make_windows`), `configs/models/cnn_lstm.yaml`,
  registry factory `cnn_lstm`. Per-window robust scaling (median/IQR of the input part), L1 loss, seeded
  shuffle of training windows, early stopping on windows whose target lies entirely in the last `val_size` days.
  No future covariates (the parquet has no Fourier terms). Defaults not tuned (T2).
- T1 smoke (not validation): `run_model.py cnn_lstm --against naive --cutoff 2022-12-17`, 3 seeds, 1 m 47 s
  CPU incl. dataset write; MAE all steps CB 0.68-1.00, Palua 0.66-0.90 (booster same cutoff: 0.83 / 1.09).
  One cutoff only; a 24-fold run would take about 40 min.

- T2 RED: `tests/test_tune_cnn_lstm.py` -> FileNotFoundError (script absent); dropout test -> TypeError. GREEN: 281 passed,
  2 skipped; ruff check clean. Runner change: `dropout` parameter (before the head and between stacked LSTM layers).
- T2 protocol: 12 tuning cutoffs every 23 days (2020-05-04..2021-01-12), fit once at 2020-05-04, predict all 12, 3 seeds,
  objective = mean rel_mae over CB and Palua vs Naive (all steps); imputation guard passed (no imputed day within 14 days).
  Study `artifacts/optuna/cnn_lstm_2009.db` (gitignored), TPE seed 42, 20 COMPLETE trials, none pruned by validity.
- T2 result: best trial #13, rel_mae 0.700 (CB 0.497, Palua 0.903, mean MAE 0.262): input_size 112, conv_channels 32,
  kernel 3, conv_layers 1, lstm_hidden 32, lstm_layers 2, dropout 0.0378, lr 0.000439, batch 256. Runner-up #6: 0.711.
  Palua is the weak point (0.90 vs 0.50 in CB), as with the TCN. Trial durations 0.5-19 min; trials sum 1.79 h.
- T2 trim (declared): trial #17 (LSTM 128 x 2 layers, window 112) ran over 30 min on CPU and was killed (stays RUNNING in the
  study and in the CSV, 21 rows); from then on that combination is pruned in `suggest` (tested). Trials 1, 2 and 19 used
  LSTM 128 x 1 layer (6.5 and 19 min). Tuning is a single-seed-set, 12-window estimate; the 24-fold validation is T3.
- T3 result (24 folds, MAE of days 1-14, mean of 3 seeds): CNN-LSTM on Colab GPU CB 0.337 / Palua 0.364
  (rel_mae 0.59 / 0.75, mean over stations 0.350); on local CPU CB 0.331 / Palua 0.362 (mean 0.346), same
  command `run_model.py cnn_lstm --against naive --folds` (2 h 16 min to 2 h 28 min on CPU). The first local
  attempt failed at the end on the per-seed model-name check (fixed in PR #10) and the GPU run first failed on
  a CPU-only quantile tensor (fixed on main, commit 4539c69).
- Comparison with the other models (same 24 folds; mean over stations of days 1-14): booster 0.309 (CPU), TCN 0.317
  (GPU), CNN-LSTM 0.350 (GPU), ARIMAX 0.351 (CPU, 583 s). The hybrid does not improve on the TCN and costs far more
  compute than ARIMAX; all four have rel_mae < 1 everywhere. Palua day 7 stays the weak point.
- CPU vs GPU criterion (agreed before looking): MAE difference < largest seed std and same sign-test verdict at 0.05.
  MAE differences 0.000-0.009 in all 6 cells (all below the seed std); verdicts agree in 5 of 6; Palua day 14 differs
  (median p 0.064 CPU vs 0.023 GPU, identical MAE 0.613). Open question: the TCN was not rerun on CPU with `--folds`
  (old CPU figure from the retired script: Palua 0.383 vs 0.342 on GPU).

## Next step

Agree with the team on the final-evaluation protocol (single run on 2023-2024 with all models; select on validation and
use the test only to report) and close the review warnings with a tests commit.
