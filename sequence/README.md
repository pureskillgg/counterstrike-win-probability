# Sequence-model screen: CS2 round win probability (Nov 2026)

Deep sequence models (GRU, LSTM, TCN, transformer, minGRU, minLSTM, LRU, xLSTM variants, RWKV-7, Mamba-2, Griffin, Gated DeltaNet) compared with XGBoost on per-second round state. Validation only for every choice; holdout and frozen test not used for any result here.

## Contents
- `model/`: the trainer (`train_phase1_pos24.py`), its loaders (`train_temporal_allmaps.py`, `train_temporal_round_model.py`), and `gdt_grader.py`, a verbatim copy of `evaluation/intern_gdt/score_gdt.py`.
- `analysis/`: `stageB_pooled.py` (GDT pooled over seeds, match-cluster bootstrap) and `failure_analysis.py` (per-segment gains against the GRU control).
- `pod/`: RunPod run scripts. They assume the network-volume layout and contain no keys or addresses.
- `notes/`: architecture rationale, the results report, the daily review, and provenance.
- `results/`: GDT pooled results, the seed-0 GDT summary, and `weights_manifest.csv` (47 final-step weight files with sizes and SHA-256).

## Not in git
- Data (parquet caches, round-state arrays, prediction files) and model weights (`*.pt`, ignored by `.gitignore`). Weights are on S3 at `models/gdt_objective/` and in the local models folder.
- Paths in the scripts assume the local data tree `G:\pureskill\csds-cs2` and the RunPod volume.

## Key settings
- Inputs: per-second state, loadout and 24 train-only position features (`--dist`), lookback 20 s.
- Training: lr 1e-3, batch 2048, 30,000 steps, final-step scoring, no checkpoint selection, `--no_test_load`.
- Objective: `--objective gdt` (GDT-equivalent per-second weights) or the default log loss. `--save_weights` writes final weights. `--dropout` and `--weight_decay` default to off.

## Pins
064d46c76 (GDT objective), 079f7189 (amendment), c807884d (GDT selection on log-loss runs), 28527ac3 (GDT-trained results), 51a86c98 (round-2 log-loss result), c1f31a9d (regularisation test), 0d97f8f6 (holdout candidate, not run).
