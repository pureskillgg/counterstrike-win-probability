# Handover: explainability for the 83-input model

Status as of Nov 5, 2026. Everything here is either in the repo or listed below with its location. Nothing is running in the background.

Start with `docs/explainability/README.md` for the results and caveats.

## Run order

Use the pixi Python: `G:\pureskill\csds-cs2\.pixi\envs\default\python.exe`. Set `CS2_EXPLAIN_MATRIX` and `CS2_EXPLAIN_WORK` if the matrix and work folder are not at their defaults.

```bash
python scripts/explain/xgb83_explain.py --stage train     # about 83 minutes on CPU; writes the booster to WORK
python scripts/explain/xgb83_explain.py --stage explain   # TreeSHAP on 100,000 seconds; permutation on all validation rows
python scripts/explain/xgb83_explain.py --stage report    # tables and figures under docs/explainability/
python scripts/explain/feature_curves.py
python scripts/explain/feature_curve_figs.py
python scripts/explain/xgb83_local.py                     # waterfalls, contrasts, interactions; about 5 seconds
python scripts/explain/xgb57_explain.py                  # the 57-input submission model on the same seconds
```

To rebuild the Word report, run `scripts/explain/docx/prep_docx_data.py`, then `build_docx.js`. Both are in the repo. `build_docx.js` takes the output path as its second argument.

## Inputs (not in git)

| Input | Where | Size | Note |
|---|---|---|---|
| Design matrix `X.npy`, `y.npy`, `meta.json`, `val_keys.parquet` | `C:/Users/13144/sprint/matrix_trainctr/` | 5.1 GB + 0.1 GB | Written by `experiments/sprint/sprint_prepare.py` on Nov 3, 2026 |
| Training cache `cache_snapshot_2026.parquet` | `G:/pureskill/csds-cs2/data/combined/` | 182 MB | 14,804 matches, Aug 2 to Oct 13, 2026 |
| Position table `cache_pos_2026_trainctr.parquet` | `G:/pureskill/csds-cs2/data/combined/` | 1.25 GB | Input to `sprint_prepare.py`. Builder not identified (see open items) |
| Holdout-B keys `holdout_22_25_test_keys.json` | `G:/pureskill/csds-cs2/data/combined/` | 42 KB | Excluded by `sprint_prepare.py` |
| Frozen test keys `frozen_test_keys.json` | not located | | Input to `sprint_prepare.py` (see open items) |
| Sprint predictions and results | `G:/pureskill/csds-cs2/data/sprint/run/gpu_fast/` | | Used by `train` to check the refit |
| Oct 26–31 holdout keys | `C:/Users/13144/sprint/holdout_oct26_31/keys.parquet` | | Do not score without a pin and a go-ahead (see below) |
| Booster | `models/winprob_xgb83_final.json` (in git) | 5.7 MB | Byte-identical to `explain83/xgb83_s0.json`; committed by Adam in `3936180` |
| Shap sample, sampled features | `C:/Users/13144/sprint/explain83/` | 35 MB sample, 33 MB features | Written by `explain`. Not in git |

## Open items

1. **Reproduce the matrix.** `sprint_prepare.py` has not been rerun on the raw data. Rebuild it into a new directory and check its `meta.json` against the existing one (rows 15,274,129; 83 inputs; n_fit 12,355,820; n_es 1,403,081; n_val 1,515,228).
2. **Find `frozen_test_keys.json`.** It is not under `G:/pureskill/csds-cs2` to depth 5. `sprint_prepare.py` needs it.
3. **Identify the builder of `cache_pos_2026_trainctr.parquet`.** Its name suggests the train-centre table used by the validation runs. Not confirmed in this session.
4. **Reconcile Oct 9–10.** The handoff (Nov 4) says these days are unreconciled and gives 2,598 matches. The training cache has 4,865 matches on those two days. Decide which count is right and whether these days stay in the pool.
5. **GPU retrain ablations: done.** Results and GDT are in `docs/explainability/README.md` ("Drop-group ablations"). Armour sums and blind add nothing on GDT. View and bomb geometry add about 0.0005 each.
6. **Alive-count question: done.** Dropping the alive counts costs nothing on GDT (+0.0002). Loadout only scores GDT 0.0098, round state only −0.0164. The loadout inputs carry the signal.
7. **1-versus-5 calibration: done, per round.** Across the full validation set, the 1-versus-5 states give 1.48 expected CT-win rounds against 1 observed (all states). The bomb-planted state gives 0.59 against 1. The one win is the largest miss's round, so the earlier "3 wins in 426 seconds" was the same round. Table: `docs/explainability/tables/calibration_1v5_shipped.csv`.
8. **Feature fixes from the team notes:** rebuild VIEW (sees-enemy and exposed), BLIND (from player blind events) and defuse slack (time to reach and defuse). Drop the armor sums. Heavy weapons are already dropped in v6. Then rerun the explanation.

## Holdout governance

The Oct 26–31 holdout has one recorded read, for the 83-input candidate (0.452295). No new holdout scoring has been run for this work, and none should run without a pre-registration pin and an explicit go-ahead.

## Not in this handover

- The I9 form and the email to Bill (his checklist, item 1). Not done.
- The deep model (item 4 of Bill's checklist). Its handoff is in [handover/deep-model/README.md](../deep-model/README.md). No deep-model training was run.
- The git push to `pureskillgg/counterstrike-win-probability`. Work is committed on `wrap-up-nov-2026` only, if the commit goes through. Not pushed.
