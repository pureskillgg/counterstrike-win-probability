# TRAINING (agent notes)

Exact retraining recipes for every artifact in `models/`. Human summary: [TRAINING.md](TRAINING.md). Data locations and tables: [DATA.agents.md](DATA.agents.md).

Conventions used below:
- `<py>`: the pixi env python (`G:\pureskill\csds-cs2\.pixi\envs\default\python.exe`) for CPU XGBoost; `<gpu-py>`: the CUDA XGBoost env (WSL `~/venvs/csds` or the `cs2gpu` env named in `scripts/run_final.ps1`); `<torch-py>`: the torch env (`G:\pureskill\csds-cs2\.pixi\envs\torch\python.exe` locally, `/workspace/envs/cs2/.pixi/envs/torch/bin/python` on pods).
- Scripts resolve the data root from `CS2_ROOT` (`scripts/model/wp_common.py`, default `G:\pureskill\csds-cs2`). Several older scripts hard-code `G:/pureskill/csds-cs2` or `parents[2]` of the script; run them from a tree where `data/` sits beside `scripts/`.
- Outputs land in `<CS2_ROOT>/models` (XGBoost) or `models/temporal`, `data/temporal`, `data/diagnostics` under the script root. Copy finished artifacts into this repo's `models/` yourself.
- Never pass test rows into training, early stopping or selection. The frozen test is `data/benchmark/frozen_test_keys.json` (3,059 keys). Holdout B (Oct 22-25) is `data/combined/holdout_22_25_test_keys.json` on the data root. The Oct 26-31 holdout comes from `cache_snapshot_2026_oct23_31.parquet` (`scripts/final/holdout_population.py`).

## Split manifests (read first)

| File | sha256 (file bytes) | Used by |
|---|---|---|
| `data/benchmark/split_manifest_a17f219b.json` | `a17f219b5287ff1e…` | XGBoost v4, v4_basic, v5, v6_drop_heavy, v6_selected, v7 (both), v8 seeds/tuned/ensemble; xgb83 inherits it through `make_split` |
| `data/benchmark/split_manifest.json` | `7fa526c3…` | the v1-archive manifest (29,885 entries, digest 6a37070a); not the one any shipped model recorded |
| workspace `data/combined/split_manifest.json` | `eb937993…` | current file, extended on 4 Nov with a different rule; its internal `sha256` key does not verify |
| (missing) | `2088a5a6…` | winprob_v3 |
| (missing) | `96df8c8a…` (internal key) | winprob_v2 |

Check: `cache_snapshot_2026` keys under the a17f219b manifest give train 10,577 / val 1,168 / test 3,059, equal to the v4-v8 metadata. Restore it as `<CS2_ROOT>/data/combined/split_manifest.json` before any XGBoost retrain. Rule (`wp_common.py` lines 67-75): `u = sha256(seed:match_key)[:12] / 16^12`; test if u < 0.20, validation if u < 0.28, else train. Early stopping inside sprint runs: `crc32("es:"+key) % 100 < 10`.

## XGBoost

Common settings (`wp_common.fit_xgb`): `binary:logistic`, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, early stop 100, seed 0, monotone constraints from `winprob_v4.meta.json`. Inputs come from `<CS2_ROOT>/data/combined/cache_snapshot_2026.parquet` (built by `scripts/model/make_snapshot_cache.py --store combined`), score lead from `grid.parquet` (`score_lead_diag`).

| Artifact | Command | Notes |
|---|---|---|
| `winprob_v4` / `_basic` | `<gpu-py> -I -X utf8 scripts/model/train_final.py --model full --device cuda` / `--model basic` | script default device is cpu; meta says cuda. basic reads `grid.parquet` (all eras; meta grid hash 2f08dc8d, grid rewritten 2 Nov) |
| `winprob_v5_59_scorelead` | `<gpu-py> -I -X utf8 scripts/model/train_final_59_scorelead.py --device cuda` | 60 inputs |
| `winprob_v6_drop_heavy` | `<gpu-py> -I -X utf8 scripts/model/save_winprob_v6_drop_heavy.py` | no args; 57 inputs; serving default |
| `winprob_v6_selected` | `<gpu-py> -I -X utf8 scripts/model/exp_loadout_select.py` | no args; selection by validation |
| `winprob_v7_59_drop_heavy_scorelead`, `winprob_v7_basic_scorelead` | `<gpu-py> -I -X utf8 scripts/model/save_score_lead_candidates.py` | no args; writes both; rewrites meta with `served: false` |
| `winprob_v8_seed0..4` | `<gpu-py> -I -X utf8 scripts/model/exp_seed_avg58.py` | no args; seeds 0-4; no per-seed meta |
| `winprob_v8_ensemble.meta.json` | `<py> -I -X utf8 scripts/model/finalize_ensemble58.py` | needs the five seed boosters and `data/combined/exp_seed_avg58.json`; scoring grid: `scripts/model/export_ensemble_predictions.py` |
| `winprob_v8_tuned` | `<gpu-py> -I -X utf8 scripts/model/exp_hparams58.py` | 11 configs; the baseline wins; byte-identical to `winprob_v8_seed0` |
| `winprob_v3` | `<gpu-py> -I -X utf8 scripts/model/export_model.py --store combined --from-cache --sets map,status --eras 2026 --stride 2 --name winprob_v3 --split random --device cuda` | from `logs/pod_v3/pod_run.log`; manifest 2088a5a6 missing |
| `winprob_v2` | `make_snapshot_cache.py --store combined`; `export_model.py --store combined --from-cache --sets map,status --eras 2026 --name winprob_v2 --stride 2` | from `scripts/run_final.ps1`; split not recoverable |
| `winprob_v1`, `_v1_basic`, `_v2_basic`, `_v3_basic`, `winprob_test` | `export_model.py --store combined --sets <map,status | map | hp,map> [--eras 2026] --name <name> --stride <1|2>` | inferred from `scripts/run_pipeline.ps1` / `run_final.ps1`; older meta schema; not exactly reproducible |
| `xgb_primary_seed0` | `pixi run python -I scripts\model\xgb_primary_seed0.py` | reads `tomes/2026-10-01_to_08/tome/csds`; primary split (seed 0, test 0.2, val 0.1 of rest) |
| `golden_winprob_*` | `python -I -X utf8 scripts/model/stress_test_winprob.py --model models/<name>.json [--fallback models/<name>_basic.json] --write-golden` | test fixtures, not trained models |

### 83-input model (`winprob_xgb83_final`)

1. Position table with train-only site centres: `<py> -I -X utf8 scripts/data/make_pos_features.py --store combined --every 2 --centre_keys data/benchmark/centre_fit_train_keys.json` (writes the `_trainctr` variant; the every-second variant is `make_pos_features_every1.py` with the same flag). Centre keys: the 10,577 training matches.
2. Matrix: `<py> -I -X utf8 experiments/sprint/sprint_prepare.py --pos <CS2_ROOT>/data/combined/cache_pos_2026_trainctr.parquet --frozen data/benchmark/frozen_test_keys.json --out <matrix_dir>` (asserts holdout B and frozen test are out).
3. Queue: `<py> -I -X utf8 experiments/sprint/make_queues.py --out <run_dir>` (126 cells; recorded queue in `experiments/sprint/run_records/gpu_fast/`).
4. GPU cells: `<gpu-py> -I -X utf8 experiments/sprint/sprint_worker.py --queue <run_dir>/queue.json --matrix <matrix_dir> --out <run_dir> --device cuda`. The model's cell is `full_d4_e0.03_m20_s0` (eta 0.03, depth 4, mcw 20, subsample 0.8, max_bin 256, cap 3000, early stop 100). Seed means use s0, s1, s2.
5. Delivered booster (CPU refit, seed 0): set `CS2_EXPLAIN_MATRIX=<matrix_dir>` and run `<py> -I -X utf8 scripts/explain/xgb83_explain.py --stage train`. It writes `C:/Users/13144/sprint/explain83/xgb83_s0.json` (hard-coded WORK path). Copy it to `models/winprob_xgb83_final.json` and check sha256 `1a514b0d1aa27650…`. Predict with `iteration_range=(0, 2696)`.
Run records: `experiments/sprint/run_records/` (matrix meta, per-cell results, refit metadata). The refit differs from the GPU sprint predictions (max abs 0.246, mean 0.004).

## Sequence models (GDT objective, `models/seq_gdt_weights/`)

Trainer: `scripts/model/train_phase1_pos24.py` (identical to `sequence/model/`). Needs `scripts/model/gdt_grader.py` (verbatim intern grader) for `--objective gdt`.

```
<torch-py> -I -X utf8 -u scripts/model/train_phase1_pos24.py --arms <arm> --seeds <seeds> --steps 30000 --eval_every 1000 --dist --no_test_load --objective gdt --save_weights --out_dir data/diagnostics/<out>
```

On RunPod: `sequence/pod/setup_pod.sh`, then `sequence/pod/run_w.sh "<arms>" "<seeds>" <bin>` (writes `/workspace/results/weights_<bin>`).

| Arm | Seeds with weights | Bin |
|---|---|---|
| gru, xlstm_fast | 0-4 | b7 |
| tcn, xlstm_blk | 0-4 | b8 (tcn reruns 0,1,3,4 in `weights_tcnr*`) |
| xlstm_sig | 0-4 | b6 |
| mingru | 0-2 / 3-4 | b4 / x2 |
| rwkv7 | 0,1 / 2,3 / 4 | b1 / b2 / x1 |
| mamba2, minlstm | 0 | b1 |
| lru, lstm, xlstm_mh | 0 | b2 |
| gdn, tf | 0 | b4 |
| griffin, gru2 | 0 | x2 |
| xlstm | 0 | x1 |
| gdn2, tf2 | 0 | x3 |

Inputs (data root): `data/round_state/<map>_ext.npz`, `<map>_ext_loadout.npy`, `<map>_ext.json` for 7 maps; `data/combined/cache_snapshot_2026_ext.parquet`; `data/combined/rounds.parquet`; `data/diagnostics/allmaps_pos24_trainsites_grid/<map>.parquet`. Population 23,716 matches, uploads 2026-09-01 to 2026-10-21; split from the npz `split` array: train 19,486, val 1,169, test 3,061 (test not loaded). 84 protected-batch matches (dated 2026-10-21) are in train.

Settings: lookback 20 s, batch 2048, lr 1e-3, 30,000 steps (script default 12,000), eval every 1,000, final-step scoring, arm widths from the `ArmModel` defaults (no width flags passed). Load a weights file: `ArmModel(arch, state_dim=34, K=20, cfg={})`, then `load_state_dict(torch.load(path))`. Saved weights are the final step; `gdn2 s0`, `mingru s3`, `xlstm_fast s3`, `xlstm_blk s3`, `xlstm_sig s3` had best step 30,000. The 47 metrics files predate the `dropout`/`weight_decay` keys (trainer revision drift; current defaults should reproduce, unproven).

Site-grid input: `scripts/model/build_pos24_maps.py` builds `allmaps_pos24_trainsites/<map>.parquet` from `cache_pos_2026_trainctr.parquet` joined to `cache_snapshot_2026.parquet`. The `_grid` folder the trainer reads holds the same 24 inputs at every second; it was built the same way from the every-second table (`cache_pos_2026_trainctr_every1`), but that exact script version was not saved (inference).

### Regularisation test

```
arm A: ... train_phase1_pos24.py --arms gru --seeds <s> --steps 30000 --eval_every 1000 --dist --no_test_load --dropout 0.2 --weight_decay 0.0001 --out_dir <dir>
arm B: ... train_phase1_pos24.py --arms gru --seeds <s> --steps 25000 --eval_every 1000 --dist --no_test_load --out_dir <dir>
```
Seeds 0-4 each, log-loss objective, no weights saved. Pod form: `sequence/pod/run_reg.sh <seed> <A|B>`. Metrics: `sequence/analysis/reg_results/`; comparison: `sequence/analysis/reg_compare.py`.

## Earlier temporal and graph models (`models/temporal/`, `models/round_state/`)

Run from a tree with `data/` beside `scripts/` (trainers use `parents[2]`). Metrics: `models/temporal/metrics/`.

| Artifacts | Command |
|---|---|
| `de_mirage_gru` | `<torch-py> -I -X utf8 scripts/model/train_temporal_round_model.py --map de_mirage` (older run, no seed key) |
| `de_mirage_loadout_gru*` | `... --map de_mirage --loadout [--seed s --suffix _seed<s>] [--epochs 6 --suffix _ep6] [--epochs 6 --max_steps 8544 --eval_every 500 --suffix _ms8544] [--seed s --suffix _t3s<s>]` |
| `de_mirage_gat_loadout_gru*` | `... --map de_mirage --loadout --gat [--seed s --suffix _t3s<s>]` |
| `de_mirage_thgnn_lite*` | `<torch-py> -I -X utf8 -u scripts/model/train_thgnn_lite.py --map de_mirage [--tag thgnn_lite_bestval | --train_frac 0.25 --tag thgnn_lite_frac25 | --train_frac 0.5 --tag thgnn_lite_frac50 | --epochs 6 --max_steps 8544 --eval_every 500 --tag thgnn_lite_ms8544]` |
| `de_mirage_ext_thgnn_lite_ms8544*` | `... train_thgnn_lite.py --map de_mirage --stem de_mirage_ext --cache cache_snapshot_2026_ext.parquet --epochs 6 --max_steps 8544 --eval_every 500 --tag thgnn_lite_ms8544 [--seed s --tag thgnn_lite_ms8544_seed<s>]` |
| `de_mirage_ext_navgraph_ms8544*` | same plus `--graph data/round_state/de_mirage_ext_navgraph.npz --tag navgraph_ms8544`; graph from `scripts/model/build_nav_region_graph.py` |
| `allmaps_gru_loadout*` | `<torch-py> -I -X utf8 -u scripts/model/train_temporal_allmaps.py --seeds 0,1,2 [--dist | --walk]`; learning curves: `--seeds 0 --match-fraction 0.25 --subset-seed 0 --no-test --suffix _lc25` (0.5, `_lc50`); `--walk` needs `scripts/model/build_nav_walk_allmaps.py` |
| `round_state/de_mirage_h3.pt` | `<torch-py> -I -X utf8 scripts/model/train_round_state_encoder.py --map de_mirage --horizon 3` (state encoder, not a win-probability model) |

Round-state arrays: `scripts/model/build_round_state_map.py --map <map>` (split labels from the manifest present at build time; build metadata in `data/manifests/round_state_meta/`). Mirage-only arrays: 43 upload days (2026-09-01 to 10-13), 2,455 / 274 / 687 matches; `_ext` arrays: 51 days to 2026-10-21. `--dist` and the default `--walk` builder fit site centres on all matches (methodology caveat).

## Known gaps

- Manifests for v2 and v3 are lost; v1, v1_basic and winprob_test use an older meta schema.
- The `_grid` sequence position folder's exact builder version is not saved (see above).
- The xgb83 refit-to-`models/` copy is manual; the GPU sprint launch command was reconstructed from `make_queues.py`.
- `served` flags in `winprob_v7_basic_scorelead.meta.json` and `winprob_v8_ensemble.meta.json` were hand-edited; the serving default in code is `winprob_v6_drop_heavy`.
- The THGNN all-maps trainer (`train_thgnn_allmaps.py`) and `train_graph_transformer.py` have no saved run.
