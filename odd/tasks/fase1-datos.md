# Feature: fase1-datos

## Objective

Reproducible extraction of the raw river-level sources into a long-format table with an
explicit per-value `quality` label, plus the shared configuration agreed in the README.

## Problem / why

The raw sources disagree (Ciudad Bolívar 1987 and 1997), the primary source is already
imputed, and there are disguised missing values (flat runs). Models must not consume these
silently, and the hand-made `Nivel Orinoco Formateado.csv` is not reproducible.

## Scope

- In: `configs/backtest.yaml`, `configs/data.yaml`, AGENTS.md config convention line,
  `src/orinoco_forecast/data/loading.py`, new `src/orinoco_forecast/data/quality.py`,
  `scripts/build_dataset.py`, tests.
- Out: Nixtla dependencies and model configs (model phase), imputation (done per fold in
  the backtest), external data (stage B).

## Constraints

- `data/raw/` is read-only; outputs go to `data/interim/`.
- Polars only (no pandas) per AGENTS.md; load `.agents/skills/polars-best-practices/SKILL.md`.
- Anomalies are flagged with warnings, never corrected silently.
- `find_flat_runs` is left as a bounded exercise for the team (tests written, skipped until
  implemented); the build script must degrade gracefully while it is not implemented.

## Acceptance criteria

- Primary source loads to `unique_id, ds, y` for 4 stations, 18 683 days each.
- Historical Ciudad Bolívar loads 1930-01-01..2025-12-31 with Feb 29 rows present as null.
- `quality` labels: `observed`, `missing`, `suspect_flat`, `suspect_jump`, `source_conflict`.
- Palúa has exactly 17 `missing` in the primary source; 1987 and 1997 contain
  `source_conflict` rows for Ciudad Bolívar.
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` pass.

## Delivery strategy

ask-on-risk. Forecast: ~350 authored lines.

## Tasks

- [x] T1 Update shared configs (`backtest.yaml`: horizon 14, step_size 30, n_windows 24,
      test_start 2023-01-01; `data.yaml`: source roles and quality thresholds) — route:
      delegated writer (writer trigger: 2+ non-trivial files across T1–T4).
- [x] T2 Loaders for primary and historical sources to long format, with tests — route:
      delegated writer.
- [x] T3 Quality labelling (`missing`, `suspect_jump`, `source_conflict`; `suspect_flat`
      skeleton + skipped tests as exercise), with tests — route: delegated writer.
- [x] T4 `scripts/build_dataset.py` writing `data/interim/levels_long.parquet` and a
      quality summary — route: delegated writer.

## Progress / evidence

- T1 `fff3eca` chore: backtest.yaml (horizon 14, step_size 30, n_windows 24,
  test_start 2023-01-01, seasonality 365); data.yaml roles, targets, quality block;
  AGENTS.md convention line. Check: YAML parses.
- T2 `37ea21b` feat: loaders (`load_config`, `load_primary`, `load_historical`,
  `load_source`, `load_raw_excel`, `validate_daily_dates`, `DataQualityWarning`);
  data.yaml gained `unique_id: ciudad_bolivar` on the historical source.
  RED: ImportError on `load_historical`; GREEN: 12 passed.
- T3 `b9004bc` feat: `quality.py` (labels, precedence, `find_flat_runs` exercise stub).
  RED: collection ImportError; GREEN: 11 passed, 4 skipped. The skipped flat tests
  were checked once against a scratch reference solution outside the repo (pass).
- T4 `929c28c` feat: `scripts/build_dataset.py` -> data/interim/levels_long.parquet
  (75 042 rows) and quality_summary.csv (256 rows). CB: 656 source_conflict
  (1987: 201, 1997: 353, 1974: 25, rest scattered), 2 missing in the 2025 extension.
- Final: `uv run pytest -q` 23 passed, 5 skipped; `ruff check` clean;
  `ruff format --check` clean.

- Review (gentle-ai, RDD on): assess medium (`slice_budget_reached`, AGENTS.md executable
  change); consent granted; reliability lens found 2 CRITICAL findings: `find_flat_runs`
  stub silently degraded `suspect_flat` to `observed` in the built artifact (deterministic),
  and a stale AGENTS.md note claimed data/raw workbooks were uncommitted (inferential; files
  are tracked). Correction `4fa789c`: implemented `find_flat_runs` (rle_id), activated its
  tests, added flat precedence test, replaced the AGENTS.md note. The flat-run exercise for
  the team is withdrawn. `92aaffe`: ASCII table borders (cp1252 console crash).
- Targeted validator returned `invalid_request` (could not inspect frozen trees, no
  verdict). User chose "continue without reporting"; the captured decline invocation was
  run once and refused with `stale_target_identity` (candidate moved after correction).
  Review state preserved, not approved. Delivery under ordinary repository policy.
- Final checks: `uv run pytest -q` 28 passed, 1 skipped (metrics placeholder); ruff check
  and format clean; build script exit 0. suspect_flat counts: ayacucho 12, caicara 10,
  ciudad_bolivar 19, palua 28.

## Next step

Model phase: metrics (MAE, RMSE, MASE, NSE, KGE) and baselines.
