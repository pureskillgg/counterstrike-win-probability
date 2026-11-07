# Retraining the models

Every model in `models/` has a training recipe in this repo. Most can be retrained exactly; the exceptions are listed under "Things that will trip you up". The data isn't in git because it's too large. [DATA.md](DATA.md) says where it is. The exact command for every model is in [TRAINING.agents.md](TRAINING.agents.md).

## Before you start

1. **Get the data folder.** It's `G:\pureskill\csds-cs2\data` on the original laptop. There's a backup at `C:\Users\<user>\Backups`, and the training inputs are also on the RunPod volume. Set `CS2_ROOT` to the folder that contains `data/`. The shared helper defaults to `G:\pureskill\csds-cs2`. Some older scripts hard-code that path or look for `data/` beside the script, so run them from a tree where `data/` sits next to `scripts/`.
2. **Use the right split.** Match keys in this repo are pseudonyms ([DATA.md](DATA.md) says why), so restore the real keys before any script reads a key file. For the XGBoost models v4 to v8 and the 83-input model, copy `data/benchmark/split_manifest_a17f219b.json` to `<CS2_ROOT>/data/combined/split_manifest.json`, then run `py -3 scripts/data/restore_match_keys.py --keys <CS2_ROOT>/data/combined/grid.parquet --check-sha256 a17f219b <CS2_ROOT>/data/combined/split_manifest.json`. It should print `starts with a17f219b: YES`. The frozen-test, holdout B and centre-fit key files in `data/benchmark/` need the same restore step. That's the exact manifest those models recorded. It reproduces their counts: 10,577 train, 1,168 validation and 3,059 test matches. The newer manifest on the laptop gives a different split.
3. **Use the right environment.** XGBoost models use the pixi environment (`pixi.toml`). GPU XGBoost runs used a CUDA environment: WSL `~/venvs/csds`, or the `cs2gpu` environment named in `scripts/run_final.ps1`. Sequence and temporal models use PyTorch on a GPU. The sequence runs used RunPod RTX 4090s.

## What each model was trained on

Dates are upload days. They are read from the match key, which looks like `csds/YYYY/MM/DD/<id>/csds`, and match the tome folder names. The per-day counts are in [data/manifests/](data/manifests/README.md).

| Model | Train with | Upload days | Matches (train / val / test) |
|---|---|---|---|
| XGBoost v6 drop-heavy, 57 inputs (serving default) | `scripts/model/save_winprob_v6_drop_heavy.py` | 1 Sep to 13 Oct 2026 | 10,577 / 1,168 / 3,059 |
| XGBoost v8, 5-seed ensemble, 58 inputs | `exp_seed_avg58.py`, then `finalize_ensemble58.py` | same | same |
| XGBoost v7 (58 and 14 inputs) | `save_score_lead_candidates.py` | same; the 14-input one also uses Feb 2024 | same; 14-input: as v4 basic (inferred: identical test rows) |
| XGBoost v5 (60), v6 selected (38), v4 (59) | `train_final_59_scorelead.py`, `exp_loadout_select.py`, `train_final.py --model full` | 1 Sep to 13 Oct 2026 | 10,577 / 1,168 / 3,059 |
| XGBoost v4 basic (13 inputs) | `train_final.py --model basic` | Feb 2024 and Sep to Oct 2026 | 12,509 / 1,383 / 3,601 |
| XGBoost 83 inputs | the sprint scripts, then a CPU refit (see the agents doc) | 1 Sep to 13 Oct 2026 | 10,577 / 1,168; Oct 26-31 holdout scored once |
| XGBoost primary (10 inputs) | `scripts/model/xgb_primary_seed0.py` | tome `2026-10-01_to_08` | 1,095 / 122 / 304 |
| Sequence models on the GDT objective (47, plus 4 TCN reruns) | `scripts/model/train_phase1_pos24.py --objective gdt --save_weights` | 1 Sep to 21 Oct 2026 | 19,486 / 1,169 / not loaded |
| Regularisation test (GRU, 10 runs) | `train_phase1_pos24.py` via `sequence/pod/run_reg.sh` | same | same |
| Earlier temporal and graph models | `train_temporal_round_model.py`, `train_temporal_allmaps.py`, `train_thgnn_lite.py` | Mirage: 1 Sep to 13 Oct; Mirage `_ext` and all maps: 1 Sep to 21 Oct 2026 | Mirage: 2,455 / 274 / 687; Mirage `_ext`: 4,535 / 275 / 688; all maps: as the sequence models |

All models use seed 0 unless the name says otherwise.

## Things that will trip you up

- **The 57,811-match export wasn't used.** It's in `export_clean_v4` and `v5` on the RunPod volume. None of these models trained on it. The XGBoost models saw 10,577 training matches, and the sequence models 19,486.
- **v1, v2, v3, their basic versions and `winprob_test` can't be retrained exactly.** Their split manifests weren't kept, so their commands are reconstructed from run scripts.
- **The 83-input model is a CPU refit.** The validated numbers come from a GPU run. The two disagree by up to 0.25 on single predictions, and 0.004 on average. Copying the refit into `models/` is a manual step.
- **"Served" flags were reconciled by hand on 7 Nov 2026.** v6 is `served: true`; v7 and v8 are `served: false`. Every other model's metadata says `served: false`. `save_winprob_v6_drop_heavy.py` now writes `served: true` for v6, and `save_score_lead_candidates.py` and `finalize_ensemble58.py` write `served: false` for v7 and v8, so a rerun keeps these settings. The serving code uses v6 by default (see [models/README.md](models/README.md)).
- **`winprob_v8_tuned` is the same booster as `winprob_v8_seed0`.** The hyperparameter search picked the baseline settings.
- **The sequence weights are from the final step, not the best checkpoint.** Report `final_val_log_loss` from each metrics file. Seven runs had their best validation score at the last step (30,000), so they may be undertrained: gdn2 seed 0, mingru seed 3, xlstm_fast seed 3, xlstm_blk seed 3, xlstm_sig seed 3, and the TCN reruns for seeds 0 and 3.
- **The sequence split comes from the round-state arrays.** It isn't the XGBoost manifest, and whether the two agree hasn't been checked.
- **84 protected matches are in the sequence training data.** The 21-25 Oct batch (2,848 matches) needed owner sign-off before training use. The sequence models' training data includes 84 of them, all dated 21 Oct and all labelled train. The XGBoost training data includes none. See [DATA.md](DATA.md).
