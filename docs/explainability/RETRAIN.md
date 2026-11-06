# Retraining the explainability models

As of Nov 6, 2026. This says what data each model used, which code trains it, and what is still missing. The agent version is in [README.agents.md](README.agents.md) if you need it; the findings are in [README.md](README.md).

## 1. Environment

- `pixi.toml` and `pixi.lock` are at the repo root. They are the environment this work ran in (copied from the `csds-cs2` project). Use `pixi install`, then `pixi run python <script>`. The default environment has xgboost 3.4.2, polars 2.0 and numpy 2.x. matplotlib and pillow are in the lock.
- The GPU ablations ran in WSL Ubuntu, with xgboost 3.2.0 (CUDA) on an RTX 3080. That is not in `pixi.lock`. Install it with `python3 -m pip install --user xgboost numpy`.
- Scripts default to paths on this machine (`C:/Users/<user>/sprint/...`, `G:/pureskill/csds-cs2`). Override them with `CS2_EXPLAIN_MATRIX`, `CS2_EXPLAIN_WORK` and `CS2_ROOT`.

## 2. Data

| File | Where | What it holds | Status |
|---|---|---|---|
| `cache_snapshot_2026.parquet` | `G:/pureskill/csds-cs2/data/combined/` | The training cache: 14,804 matches dated 2 Aug to 13 Oct 2026. 182,283,341 bytes. | The code that writes it was **not found** in the repo. |
| `cache_pos_2026_trainctr.parquet` | `G:/pureskill/csds-cs2/data/combined/` | Position features per second. Site centres fitted on training matches only. 1,250,708,888 bytes. | **Builder not identified.** `scripts/data/make_pos_features.py` builds `cache_pos_2026.parquet` from the raw position and status tomes. It has no training-only option, so it does not produce this variant as written. |
| `frozen_test_keys.json` | `data/benchmark/` (in the repo) | 3,059 match keys, the frozen test. | Found here Nov 6. The handover said it was missing. |
| `split_manifest.json` | `data/benchmark/` (in the repo) | Each match's train, val or test assignment. Rule: per-era shuffle with seed 0, later matches by sha256 of seed and key. sha256 `6a37070a…`. Created 15 Oct 2026. | Present. Not yet checked against the matrix split. |
| `holdout_22_25_test_keys.json` | `G:/pureskill/csds-cs2/data/combined/` | Holdout B keys, excluded from every fit. | Not in the repo. |
| Oct 26 to 31 holdout keys | `C:/Users/<user>/sprint/holdout_oct26_31/keys.parquet` | Not scored. One earlier candidate was read once. | Not in the repo. |

## 3. Dates

- **Training pool:** matches dated 2 Aug to 13 Oct 2026, 14,804 in all. The split is by match: fit 12,355,820 rows, early stopping 1,403,081, validation 1,515,228 seconds (1,168 matches).
- **GDT population:** 295 validation matches dated 1 Sep to 8 Oct, rounds that start 5 against 5.
- **Frozen test:** uploads of 14 to 25 Oct (per the scorer's README). Not read.
- **9 and 10 Oct:** unresolved. The cache holds 4,865 matches on these days; the handoff gives 2,598.

## 4. Commands, in order

1. **Design matrix** (needs the cache, the position table and the frozen keys):
   `python experiments/sprint/sprint_prepare.py --pos <cache_pos_2026_trainctr.parquet> --frozen data/benchmark/frozen_test_keys.json --out <matrix dir>`
   Check `meta.json`: 15,274,129 rows, 83 inputs, n_fit 12,355,820, n_es 1,403,081, n_val 1,515,228. Last run Nov 3; not rerun since.
2. **Row keys** (optional check): `python scripts/explain/train_keys.py`. The split counts match `meta.json`. The validation keys have the same rows as `val_keys.parquet` but in a different order, so the script writes nothing.
3. **83-input CPU refit:** `python scripts/explain/xgb83_explain.py --stage train`. Best iteration 2,695. The delivered file holds 2,796 boosted rounds, so predict with `iteration_range=(0, 2696)`. A plain predict uses all rounds and differs by up to 0.02.
4. **Explanations:** `--stage explain`, then `--stage report`; then `feature_curves.py`, `feature_curve_figs.py`, `xgb83_local.py`, `match_explain.py`, `xgb83_calibration.py`, `xgb57_explain.py`.
5. **Ablations** (WSL, GPU): `python3 scripts/explain/xgb83_drop_ablation.py --matrix /home/<user>/sprint/matrix_trainctr --out <dir> --device cuda [--only <sets>]`. Thirteen sets, three seeds: full, minus_alive, round_state_only, loadout_only, minus_round_state, minus_blind, minus_view, minus_armor_sums, minus_bomb_geo, no_position, checklist_77, minus_site_spread, minus_contact.
6. **GDT:** `python scripts/explain/xgb83_gdt.py --sets <the thirteen> --seeds 0 1 2`, then `xgb83_ablation_report.py` and `gdt_examples.py`.
7. **Figures and report:** `readme_figs.py`, then `docx/prep_docx_data.py`, then `docx/build_docx.js`.

## 5. Not yet reproducible

- The matrix has not been rebuilt from the raw tomes since Nov 3.
- The builder of the training-centre position table is not identified.
- The cache writer is not identified.
- No GDT-weighted model has been trained. That needs each training row's match and round, and those are not saved in the matrix.
- The 57-input v6 model's training code is not verified in this repo. Its explanation script is `xgb57_explain.py`.
