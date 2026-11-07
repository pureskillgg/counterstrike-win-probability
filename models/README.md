# Models

Every model file kept in git. Each XGBoost model is a JSON booster with a `.meta.json` next to it that lists its inputs and training settings. To retrain any of them, see [TRAINING.md](../TRAINING.md).

## Which model is served

The serving code, `scripts/serving/win_probability.py`, uses **`winprob_v6_drop_heavy`** (57 inputs) by default. It describes that model as "the submission, decided 19 October".

Two metadata files also say `served: true`: `winprob_v7_basic_scorelead` and `winprob_v8_ensemble`. Those flags predate the 19 October decision and conflict with the serving code. Treat the serving code as the record until the owner confirms.

## XGBoost models

| File | Inputs | Created | What it is |
|---|---|---|---|
| `winprob_v6_drop_heavy` | 57 | Oct 2026 | Serving default. The 59 snapshot inputs without the heavy-weapon group. |
| `winprob_v8_seed0` to `seed4` (+ `winprob_v8_ensemble.meta.json`) | 58 | Oct 2026 | Mean of five seeds: v6's inputs plus the score lead. |
| `winprob_v8_tuned` | 58 | Oct 2026 | v8 inputs with the selected hyperparameters. |
| `winprob_v7_basic_scorelead` | 14 | Oct 2026 | Event channels only, plus the score lead. |
| `winprob_v7_59_drop_heavy_scorelead` | 58 | Oct 2026 | Same inputs as v8, one seed. |
| `winprob_v6_selected` | 38 | Oct 2026 | Loadout inputs selected on validation. |
| `winprob_v5_59_scorelead` | 60 | 17 Oct | 59 snapshot inputs plus the score lead. |
| `winprob_v1` to `v4` | 59 | 14 to 16 Oct | Earlier 59-input models. |
| `winprob_v1_basic` to `v4_basic` | 13 | 14 to 16 Oct | Earlier event-only models. |
| `winprob_xgb83_final` | 83 | 5 Nov | 59 base inputs plus position, view and blind inputs. A CPU refit; the subject of the explainability report. Not served. |
| `xgb_primary_seed0` | 10 | 12 Oct | First model, on tome `2026-10-01_to_08` (1,521 matches). |
| `winprob_test` | 15 | 14 Oct | Pipeline test model. |
| `golden_winprob_v1` to `v3` (+ `_basic`) | | | Reference boosters with no metadata file. Their purpose isn't recorded. |

Every model from v2 onwards records `split: random` in its metadata: matches are assigned to train, validation or test at random by match.

## Sequence and temporal models

- `seq_gdt_weights/`: the 47 sequence models trained on the GDT objective (one `.pt` and one metrics JSON per arm and seed), plus four extra TCN runs in `weights_tcnr*/`. Results are in [sequence/README.md](../sequence/README.md).
- `temporal/`: 43 earlier temporal and graph models (GRU on loadouts, graph and navigation-graph variants, mostly on Mirage).
- `round_state/de_mirage_h3.pt`: an earlier round-state model.

## Not in git

These are in the temporary handover bucket (`s3://temp-model-transfer-0/adam/cs2-finished-models-v1/extra-weights/`):
- the v4 tuning variants (`es50`, `es200`, `nomono`, `s1`, `s2`, and the same for `_basic`)
- three prediction files (`*valpreds.pt`)

The bucket is temporary, so the original copies stay on `G:\pureskill\csds-cs2\models`.
