# counterstrike-win-probability

Work on the CS2 round win-probability model: how we evaluate it, and the experiments run on top of the shipped model (v6, `winprob_v6_drop_heavy`).

## Layout

- `evaluation/intern_gdt/`: the intern's grader (README, `score_gdt.py`, `table.csv`), copied unchanged. It grades a model by its gain over a fixed situation table (GDT, with a per-situation GDT_sit).
- `evaluation/build_gdt_rows.py`: builds the grader's per-second rows. Not yet runnable on our data; see `docs/RESULTS.md`.
- `analysis/`: loss by seconds into the round, and the split and upload-day check.
- `experiments/`: snapshots of the scripts behind each experiment (lag, money, GRU, E1 money and equipment, diagnostics, the pipeline runner, the retrain and the frozen-test guard). Paths still point at the local data tree; see below.
- `docs/EVALUATION.md`: the round-quality evaluation protocol. Read this first; it sets the headline metric, the windows and the rules.
- `evaluation/evaluate.py`: one entry point for the protocol. It writes a report to `docs/eval_reports/`.
- `docs/RESULTS.md`: recorded results with their sources, the grading status, and corrections.
- `scripts/data/` and `scripts/model/`: the pipeline and experiment code, copied from the project folder `G:\pureskill\csds-cs2`. The folder layout is kept, so imports between scripts still work. The model helpers read `CS2_ROOT` (default `G:\pureskill\csds-cs2`). Experiments that aren't used for the current results are kept as a record.
- `models/`: the v4 and v6 models and their metadata.
- `data/benchmark/`: the frozen test keys and the split manifest that every result depends on.
- Not in git: the data, the parquet caches, the prediction files and the experiment outputs in the scratchpad.

## Data (not in git)

- Data root: `G:\pureskill\csds-cs2` (tomes, the combined store, caches, models, predictions).
- Frozen benchmark: 3,059 test matches listed in the session scratchpad's `frozen_test_keys.json`. Never change it.
- Frozen cache: `data/combined/cache_snapshot_2026.parquet`. Never overwrite it; new caches get their own file name.

## Running

- Python: the pixi environment at `G:\pureskill\csds-cs2\.pixi\envs\default\python.exe` (XGBoost, CPU).
- GPU jobs run in WSL with `~/venvs/csds` (CUDA XGBoost). On Windows, `device="cuda"` silently falls back to CPU.
- The scripts in `experiments/` hard-code the local data and scratch paths. Moving them behind a shared path config is a follow-up.

## Conventions

- Compare models on the same rows and the same seed (a paired comparison).
- Report log loss and AUC together for any new model.
- Report per-situation results next to the pooled number, with the situations named in advance.
