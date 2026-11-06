# Agent notes: explainability for the 83-input model

Same content as `README.md`, in the form an agent needs. Read `README.md` for results and caveats.

## File map

- `scripts/explain/xgb83_explain.py`: stages `train`, `explain`, `report`. Writes tables and figures here.
  - `train`: refits the 83-input model on CPU, checks validation log loss (0.450210) and predictions against the sprint's saved predictions.
  - `explain`: TreeSHAP on 100,000 validation seconds (`shap_sample.npz`, `xs.npy`) and permutation importance on all validation rows.
  - `report`: bundle and input tables, figures.
- `scripts/explain/feature_curves.py`, `feature_curve_figs.py`: per-input curves and slopes (needs `explain`).
- `scripts/explain/xgb83_local.py`: one-prediction waterfalls, contrasts, interactions. Writes `local/`. Needs `explain`. About 5 seconds.
- `scripts/explain/xgb57_explain.py`: explains the saved v6 booster (`models/winprob_v6_drop_heavy.json`) on the same seconds. Writes `xgb57/tables/`.
- `scripts/explain/docx/`: `prep_docx_data.py` (collects tables and figures), `build_docx.js` (writes `XGB_explainability_report.docx`).

## Environment and paths

- Python: `G:\pureskill\csds-cs2\.pixi\envs\default\python.exe` (xgboost 3.4.2, CPU only; polars; numpy; matplotlib). System Python has none of these.
- `CS2_EXPLAIN_MATRIX`: matrix directory (default `C:/Users/13144/sprint/matrix_trainctr`). Holds `X.npy` (5.1 GB), `y.npy`, `meta.json`, `val_keys.parquet`. Not in git.
- `CS2_EXPLAIN_WORK`: work directory (default `C:/Users/13144/sprint/explain83`). Holds the booster `xgb83_s0.json`, `train_meta.json`, `shap_sample.npz`, `xs.npy`, `perm_importance.csv`. Not in git.
- Sprint comparison files are on `G:/pureskill/csds-cs2/data/sprint/run/gpu_fast/`. Only `train` reads them.
- `CS2_ROOT` (default `G:/pureskill/csds-cs2`) is the data root for `sprint_prepare.py`.

## Invariants (asserted in code; keep them)

- The 83 inputs are covered exactly once by `GROUPS` (`xgb83_explain.py`).
- Base logit + sum of TreeSHAP contributions = z, to 3e-5 or better.
- Bundle effects are sums of member contributions, so they are exact.
- `local/contrast_*.csv`: per-input differences sum to the logit gap.
- `local/interactions_top_pairs.csv`: interaction rows sum to SHAP values (max error 9e-6 on 2,000 seconds).
- The matrix row order equals `val_keys.parquet` row order. `train` checks this against the sprint's predictions.

## Data facts (checked)

- Training pool: `cache_snapshot_2026.parquet`, 14,804 matches, dated 2026-08-02 to 2026-10-13. No match on or after 2026-10-26. Zero overlap with the Oct 26–31 holdout keys (`C:/Users/13144/sprint/holdout_oct26_31/keys.parquet`).
- Validation: 1,168 matches, 1,515,228 seconds.
- Per-day match counts in the pool: Oct 9 has 2,214, Oct 10 has 2,651. The handoff (`docs/handoff/2026-11-04_tomes_to_model.md`) says Oct 9–10 are unreconciled and gives 2,598 matches for them. This disagrees with the pool. Do not assume either number without reconciling.

## Verification status

- Done: `train` reproduces the sprint's validation log loss within 0.000018. Per-row predictions differ (max 0.25, mean 0.004).
- Done: `xgb83_local.py` runs end to end on the existing artifacts.
- Not done: `sprint_prepare.py` was not rerun, so the 5.1 GB matrix was not rebuilt. Reproducibility from raw tomes is untested.
- Not located: `frozen_test_keys.json` (needed by `sprint_prepare.py`; not found under `G:/pureskill/csds-cs2` to depth 5).
- Not identified: the builder of `cache_pos_2026_trainctr.parquet` (1.25 GB, in `data/combined/`, dated Nov 2).

## Do not

- Do not commit `*.npy`, `*.npz` or `*.parquet`. `.gitignore` covers most of them. The 83-input booster is committed as `models/winprob_xgb83_final.json`, byte-identical to `xgb83_s0.json` in the work folder.
- Do not run any holdout scoring. The Oct 26–31 holdout has one recorded read (the 83-input candidate). A new read needs a pre-registration pin and an explicit go-ahead.
- Do not change the sprint's split or the matrix in place. Write new matrices to new directories.
- Do not state causal effects. These are model explanations.

## Open items

See `handover/explainability/README.md`.
