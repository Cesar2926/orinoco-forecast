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
- [ ] T2 Optuna search (20 trials) and apply best params.
- [ ] T3 Validation with `--folds` vs naive, booster and TCN; docs section.

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

## Next step

T2 (Optuna search).
