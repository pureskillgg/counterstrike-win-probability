# Agent notes: explainability for the 83-input model

As of Nov 6, 2026. Same results as [REPORT.md](REPORT.md) (the full write-up; [README.md](README.md) is the short human summary), in the form an agent needs. Read REPORT.md for the findings and caveats.

## File map

Scripts are in [scripts/explain/](../../scripts/explain/).

| Script | Stage | Writes |
|---|---|---|
| `xgb83_explain.py --stage train` | refit the 83-input model on CPU; check against the sprint's predictions | work folder (not in git) |
| `xgb83_explain.py --stage explain` | TreeSHAP on 100,000 validation seconds; permutation on all validation rows | work folder |
| `xgb83_explain.py --stage report` | bundle and input tables, figures | [tables/](tables/), [figures/](figures/) |
| `feature_curves.py`, `feature_curve_figs.py` | per-input curves and slopes (needs `explain`) | [tables/](tables/), [figures/](figures/) |
| `xgb83_local.py` | waterfalls, contrasts, SHAP interactions (about 5 s; needs `explain`) | [local/](local/) |
| `xgb83_calibration.py` | shipped booster on all validation seconds; deciles; 1-versus-5 per round | [tables/calibration_*](tables/) |
| `xgb83_drop_ablation.py` | nine drop-group sets × three seeds (WSL, GPU) | work folder `explain83_ablation/` |
| `xgb83_gdt.py` | population rows (295 matches) and the team's scorer, per seed | work folder `gdt/` |
| `xgb83_ablation_report.py` | log-loss and GDT ablation tables | [tables/drop_ablation_*](tables/) |
| `xgb57_explain.py` | explains the saved v6 booster on the same seconds | [xgb57/tables/](xgb57/tables/) |
| `readme_figs.py` | README figures: GDT ablations, calibration deciles, the largest-loss waterfall (reads the tables and [local/](local/)) | [figures/](figures/) |
| `docx/prep_docx_data.py`, `docx/build_docx.js` | the Word report | [XGB_explainability_report.docx](XGB_explainability_report.docx) |

## Environment and paths

- Windows Python: `G:\pureskill\csds-cs2\.pixi\envs\default\python.exe` (xgboost 3.4.2, CPU; polars; numpy; matplotlib). System Python has none of these.
- WSL Ubuntu 22.04, Python 3.10, `xgboost` 3.2.0 with CUDA, on the RTX 3080. The ablation needs the matrix on WSL's own disk, at `/home/<user>/sprint/matrix_trainctr`. Mapping it over `/mnt/c` is far too slow.
- `CS2_EXPLAIN_MATRIX`: matrix directory (default `C:/Users/<user>/sprint/matrix_trainctr`). Holds `X.npy` (5.1 GB), `y.npy`, `meta.json`, `val_keys.parquet`. Not in git.
- `CS2_EXPLAIN_WORK`: work directory (default `C:/Users/<user>/sprint/explain83`). Holds `train_meta.json`, `shap_sample.npz`, `xs.npy`, `perm_importance.csv`, `p_val_shipped.npy`. Not in git.
- The shipped booster is in git: [models/winprob_xgb83_final.json](../../models/winprob_xgb83_final.json). It is byte-identical to `explain83/xgb83_s0.json`.
- `CS2_ROOT` (default `G:/pureskill/csds-cs2`) is the data root for `sprint_prepare.py`.
- GDT scorer: [evaluation/intern_gdt/score_gdt.py](../../evaluation/intern_gdt/score_gdt.py). Its README is [evaluation/intern_gdt/README.md](../../evaluation/intern_gdt/README.md).

## Invariants (asserted in code; keep them)

- The 83 inputs are covered exactly once by `GROUPS` in `xgb83_explain.py`.
- Base logit plus the sum of TreeSHAP contributions equals z, to 3e-5 or better.
- Bundle effects are exact sums of their members' contributions.
- `local/contrast_*.csv`: per-input differences sum to the logit gap.
- `local/interactions_top_pairs.csv`: interaction rows sum to the SHAP values (max error 9e-6 on 2,000 seconds).
- The matrix row order equals `val_keys.parquet` row order. `train` checks this against the sprint's predictions.
- `xgb83_drop_ablation.py` uses the sprint's fit path (depth 4, eta 0.03, min_child_weight 20, subsample 0.8, hist, max_bin 256, early stopping 100, 3,000-round cap, monotone alive constraints). The full set reproduces the sprint's saved seed results to within 0.00005 log loss (`tables/drop_ablation_reproduction.csv`).
- GDT population: matches dated 2026-09-01 to 2026-10-08, 5-against-5 starts, 295 matches. The README excludes Oct 9–13.

## Data facts (checked)

- Training pool: `cache_snapshot_2026.parquet`, 14,804 matches, dated 2026-08-02 to 2026-10-13. No match on or after 2026-10-26. Zero overlap with the Oct 26–31 holdout keys (`C:/Users/<user>/sprint/holdout_oct26_31/keys.parquet`).
- Validation: 1,168 matches, 1,515,228 seconds.
- Day counts in the pool: Oct 9 has 2,214 matches and Oct 10 has 2,651. The handoff ([docs/handoff/2026-11-04_tomes_to_model.md](../handoff/2026-11-04_tomes_to_model.md)) gives 2,598 for the unreconciled Oct 9–10. The two numbers disagree, and neither should be assumed without reconciling.

## Verification status

- Done: `train` reproduces the sprint's validation log loss to within 0.000018. Per-row predictions differ (max 0.25, mean 0.004).
- Done: the shipped booster's validation log loss is 0.450210 on all 1,515,228 seconds.
- Done: the 1-versus-5 check, per round. The one CT win is the largest miss's round.
- Done: nine drop-group sets, three seeds, scored on GDT. The full model's GDT is 0.0254, matching the team's documented figure.
- Not done: `sprint_prepare.py` has not been rerun. The 5.1 GB matrix was not rebuilt from raw tomes, so reproducibility from raw data is untested.
- Not located: `frozen_test_keys.json`, which `sprint_prepare.py` needs. It is not under `G:/pureskill/csds-cs2` to depth 5.
- Not identified: the builder of `cache_pos_2026_trainctr.parquet` (1.25 GB, in `data/combined/`, dated Nov 2).

## Do not

- Do not commit `*.npy`, `*.npz` or `*.parquet`. `.gitignore` covers most of them.
- Do not run holdout or frozen-test scoring. The Oct 26–31 holdout has one recorded read (the 83-input candidate). A new read needs a pre-registration pin and an explicit go-ahead.
- Do not change the sprint's split or the matrix in place. Write new matrices to new directories.
- Do not state causal effects. These are model explanations.
- Do not report log loss as the headline result. The team's metric is GDT, and log loss goes alongside it.

## Open items

See [handover/explainability/README.md](../../handover/explainability/README.md#open-items).
