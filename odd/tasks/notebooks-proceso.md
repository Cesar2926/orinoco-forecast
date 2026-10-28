# Feature: notebooks-proceso

## Objective

Three Jupyter notebooks that tell the creation process of each model (XGBoost booster, TCN,
CNN-LSTM): data, partition, design decisions, Optuna tuning, validation results and
limitations, plus an index (`notebooks/README.md`).

## Problem / why

The process (decisions, data use, training) lives only in long Markdown docs. The team asked
for notebooks that walk through it and can be read or re-run by teammates.

## Scope

- In: `notebooks/01_xgboost_booster.ipynb`, `notebooks/02_tcn.ipynb`,
  `notebooks/03_cnn_lstm.ipynb`, `notebooks/README.md`, this feature document.
- Out: any training, `--folds` run, change in `data/`, `artifacts/` or `data/predictions/`,
  model code, configs, `notebooks/colab_folds.ipynb` (read only for style).

## Constraints

- A heavy validation (`run_model.py cnn_lstm --folds`) runs in the background: code cells are
  light reads (polars, yaml, tiny tensors) and fail with a message when a file is missing.
- Every figure cited must match `docs/` or `reports/tuning/*.csv`.
- The CNN-LSTM 24-fold validation is pending: marked as such, no invented numbers.
- No saved outputs (`execution_count` null).

## Tasks

- [x] T1 `01_xgboost_booster.ipynb`. Commit `5321609`.
- [x] T2 `02_tcn.ipynb`. Commit `6d9da1b`.
- [x] T3 `03_cnn_lstm.ipynb` (compares with 01 and 02). Commit `3b7c9bb`.
- [x] T4 `notebooks/README.md` index and this document updated with commits.

## Delivery strategy

single-pr on `feat/cnn-lstm`, one commit per notebook plus one for the index. Docs only.

## Progress / evidence

- Route: delegated direct (fork); trigger: 3 notebooks + index, each non-trivial.
- Generated from a script kept outside the repo (scratchpad) so the JSON stays valid; no
  outputs saved (`execution_count` null).
- Verification: every code cell of the three notebooks executed in sequence with `uv run python`
  from the repo root, no exception (7, 8 and 9 code cells). Cells that read missing files print a
  message: `data/predictions/{booster,tcn,cnn_lstm}/backtest/naive/comparison.parquet` do not exist
  locally, so the validation tables come from the documented figures.
- Cited figures checked with `grep -F` against `docs/modelos_arboles_cnn.md` (24-fold MAE and
  `rel_mae`, sign tests, covariate experiments) and read from `reports/tuning/*.csv` by the cells
  themselves (best trials: XGBoost #14 0.4902, TCN #25 0.6058, CNN-LSTM #13 0.7000).
- The CNN-LSTM 24-fold validation is pending and marked so; no figure invented.
- No training, no `--folds` run, no write to `data/`, `artifacts/` or `data/predictions/`.

## Next step

When `run_model.py cnn_lstm --against naive --folds` finishes, fill the pending rows of the table in
section 6 of `03_cnn_lstm.ipynb` and re-check the comparison with 01 and 02.
