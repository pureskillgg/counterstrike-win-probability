# Models

Every model file kept in git. Each XGBoost model is a JSON booster. Most have a `.meta.json` next to them listing inputs and training settings. The five `winprob_v8_seed` files share `winprob_v8_ensemble.meta.json`, and `xgb_primary_seed0` uses `xgb_primary_seed0_meta.json`. To retrain them, see [TRAINING.md](../TRAINING.md); v1 to v3 can only be approximated.

## Which model is served

The serving code, `scripts/serving/win_probability.py`, uses **`winprob_v6_drop_heavy`** (57 inputs) by default. It describes that model as "the submission, decided 20 October" (pin aaecad58).

Served model: winprob_v6_drop_heavy (pin aaecad58, 20 Oct 2026; the serving code default). v8 was named production on 19 Oct and v7 was served earlier; both are superseded and their metadata now says served: false. The 83-input model is more accurate but is not servable by win_probability.py and was never pinned as the submission.

## XGBoost models

| File | Inputs | Created | What it is |
|---|---|---|---|
| `winprob_v6_drop_heavy` | 57 | Oct 2026 | Served model and serving default in code. The 59 snapshot inputs without the two heavy-weapon inputs. Its metadata says `served: true`. |
| `winprob_v8_seed0` to `seed4` (+ `winprob_v8_ensemble.meta.json`) | 58 | Oct 2026 | Mean of five seeds: v6's inputs plus the score lead. |
| `winprob_v8_tuned` | 58 | Oct 2026 | The same booster as `winprob_v8_seed0`: the hyperparameter search picked the baseline settings. |
| `winprob_v7_basic_scorelead` | 14 | Oct 2026 | Event channels only, plus the score lead. |
| `winprob_v7_59_drop_heavy_scorelead` | 58 | Oct 2026 | Same inputs as v8, one seed. |
| `winprob_v6_selected` | 38 | Oct 2026 | Loadout inputs selected on validation. |
| `winprob_v5_59_scorelead` | 60 | 17 Oct | 59 snapshot inputs plus the score lead. |
| `winprob_v1` to `v4` | 59 | 14 to 16 Oct | Earlier 59-input models. |
| `winprob_v1_basic` to `v4_basic` | 13 | 14 to 16 Oct | Earlier event-only models. |
| `winprob_xgb83_final` | 83 | 5 Nov | 59 base inputs plus position, view and blind inputs. A CPU refit; the subject of the explainability report. Not served. |
| `xgb_primary_seed0` | 10 | 12 Oct | Baseline model, on tome `2026-10-01_to_08` (1,521 matches). |
| `winprob_test` | 15 | 14 Oct | Early model with an older metadata format. Its purpose isn't recorded. |
| `golden_winprob_v1` to `v3` (+ `_basic`) | | | Not models: test fixtures written by `scripts/model/stress_test_winprob.py --write-golden`. |

Every model from v2 to v8 records `split: random`. Matches go to train, validation or test by a hash of the match key, using the split manifest present at training time ([TRAINING.md](../TRAINING.md) says which one). The 83-input model's metadata has no split key; it uses the same split through `make_split`.

## Sequence and temporal models

- `seq_gdt_weights/`: the 47 sequence models trained on the GDT objective (one `.pt` and one metrics JSON per arm and seed), plus four extra TCN runs in `weights_tcnr*/`. Results are in [sequence/README.md](../sequence/README.md).
- `temporal/`: 40 earlier temporal and graph models (GRU on loadouts, graph and navigation-graph variants, mostly on Mirage), with their run metrics in `temporal/metrics/`.
- `round_state/de_mirage_h3.pt`: an earlier round-state model.

## Not in git

These are in the temporary handover bucket (`s3://temp-model-transfer-0/adam/cs2-finished-models-v1/extra-weights/`):
- the v4 tuning variants (`es50`, `es200`, `nomono`, `s1`, `s2`, and the same for `_basic`)
- three prediction files (`*valpreds.pt`)

The bucket is temporary, so the original copies stay on `G:\pureskill\csds-cs2\models`.
