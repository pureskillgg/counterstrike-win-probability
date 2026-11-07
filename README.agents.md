# Repo notes for agents

Human front page: [README.md](README.md). Detail docs: [TRAINING.agents.md](TRAINING.agents.md), [DATA.agents.md](DATA.agents.md), [docs/explainability/README.agents.md](docs/explainability/README.agents.md).

## Rules that apply everywhere

- Never use frozen-test (`data/benchmark/frozen_test_keys.json`), holdout B (`data/benchmark/holdout_22_25_test_keys.json`) or October 26-31 rows for any training, early stopping, selection or tuning. Score them once, after the choice is pinned.
- Restore the split manifest a model recorded before retraining it (`data/benchmark/split_manifest_a17f219b.json` for XGBoost v4-v8 and xgb83).
- Match keys in this repo are pseudonyms (`csds/YYYY/MM/DD/m_<24 hex>/csds`). Run `scripts/data/restore_match_keys.py --keys <CS2_ROOT>/data/combined/grid.parquet <file>` on a copy of any key file before a script reads it. Never commit raw match keys, emails, user-name paths, IPs or account numbers; `scripts/data/check_repo_privacy.py` scans the tree for them.
- Do not commit data: `*.parquet`, `*.npy`, `*.npz`, raw tomes, prediction files, credentials. `.pt` is allowed only under `models/`.
- Credentials (RunPod key, S3 keys, AWS profiles) live in local files outside the repo. Never print or commit them.
- Doc convention: `X.md` for people (short, readable); `X.agents.md` for agents (paths, commands, evidence).

## Environment

- Data root: `CS2_ROOT` (default `G:\pureskill\csds-cs2`), read by `scripts/model/wp_common.py`. Many scripts hard-code `G:/pureskill/csds-cs2` or `C:/Users/<user>/...`; override or edit before running elsewhere.
- Python: pixi (`pixi.toml`, `pixi.lock`) for data and CPU XGBoost; CUDA XGBoost in WSL (`~/venvs/csds`) or the `cs2gpu` env; PyTorch env for sequence/temporal models; RunPod pods use `/workspace/envs/cs2/.pixi/envs/torch/bin/python`.
- On Windows, XGBoost `device="cuda"` silently falls back to CPU.

## Entry points

| Task | Start with |
|---|---|
| Serve a prediction | `scripts/serving/win_probability.py` (`win_probability(data, model_name=...)`), parity test `scripts/serving/test_win_probability.py` |
| Retrain XGBoost | `scripts/model/save_winprob_v6_drop_heavy.py`, `exp_seed_avg58.py` + `finalize_ensemble58.py`, `train_final.py`; xgb83 via `experiments/sprint/` and `scripts/explain/xgb83_explain.py` |
| Retrain sequence models | `scripts/model/train_phase1_pos24.py`; pods: `sequence/pod/` |
| Rebuild data | `scripts/data/` and `scripts/model/make_snapshot_cache.py`, `build_round_state_map.py`, `split_manifest.py` |
| Score GDT | `evaluation/intern_gdt/score_gdt.py` (verbatim intern grader; copies at `scripts/model/gdt_grader.py`, `sequence/model/gdt_grader.py`) |
| Explainability | `scripts/explain/` (see `docs/explainability/README.agents.md`) |
| Day coverage of any table | `data/manifests/` |

## Where results and decisions are recorded

- `notes/REPORT_2026-11-06.md` (also `sequence/notes/`): sequence vs XGBoost results as of 6 Nov.
- `docs/final/`: final inventory (`inventory/code_inventory.md`, `pins_inventory.md`, `runs_inventory.csv`), `RUNS.md`.
- `notes/decision_register_2026-10-27/`, `notes/provenance_2026-11-06/`: decision and provenance records.
- Model metadata: `models/*.meta.json`; sequence metrics: `models/seq_gdt_weights/*/*_metrics.json`; temporal metrics: `models/temporal/metrics/`.
- Decision pins live in the Meridian project (ids such as `064d46c76`, `c807884d`, `a3ceeda7`); they are cited in the reports, not stored here.

## Known inconsistencies (do not "fix" silently)

- Served flags were reconciled on 7 Nov 2026: v6 is `served: true`; v7 and v8 were hand-edited to `served: true` before and are now `served: false`.
- `models/README.md` lists models left out of git (v4 tuning variants, `*valpreds.pt`).
- The sequence position folder `_grid` builder version was not saved.
- The 295 vs 297 match count for the GDT proxy differs between the explainability deck and the 6 Nov report.
