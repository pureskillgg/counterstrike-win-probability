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
- Training: lr 1e-3, batch 2048, 30,000 steps in the pod runs (the script default is 12,000; the pod runs pass `--steps 30000`), final-step scoring, no checkpoint selection, `--no_test_load`.
- Data: the round-state arrays for 7 maps, 23,716 matches uploaded 1 Sep to 21 Oct 2026. The split comes from the arrays: 19,486 train, 1,169 validation, 3,061 test (test not loaded). 84 matches from the protected 21-25 Oct batch are in train. Per-day counts: `data/manifests/round_state_days.csv`. Retraining: [TRAINING.agents.md](../TRAINING.agents.md).
- Objective: `--objective gdt` (GDT-equivalent per-second weights) or the default log loss. `--save_weights` writes final weights. `--dropout` and `--weight_decay` default to off.

## Results (validation only, as of 6 Nov 2026)

Final-step log loss, five seeds, 1,515,228 validation seconds. Source: `notes/REPORT_2026-11-06.md`, sections 3.1 to 3.3.

| Arm | Log-loss objective | GDT objective |
|---|---|---|
| xLSTM 7B-style block | 0.449940 | 0.450746 |
| xLSTM closed-form | 0.450123 | 0.450441 |
| MinGRU | 0.450138 | 0.450274 |
| RWKV-7 | 0.450168 | 0.450730 |
| xLSTM sigmoid-gate | 0.450337 | 0.450474 |
| TCN | 0.450735 | 0.450480 |
| GRU control | 0.451061 | 0.450392 |

**GDT (pooled over five seeds, against the GRU control).** No arm passes the pinned bar, which needs the whole 95% interval above zero and a positive situation-weighted estimate. MinGRU +0.00039 (interval −0.00014 to +0.00093); xLSTM sigmoid-gate +0.00019; all others are within ±0.001 with intervals that include zero. The earlier pick of xLSTM sigmoid-gate (pin c807884d) came from scoring the log-loss models on GDT and does not reproduce on the GDT-trained models.

**Regularisation (log-loss GRU, five seeds).** Reference, 30k steps: 0.451061. Dropout 0.2 with weight decay 1e-4 at 30k steps: 0.451266, 2 of 5 seeds better, not a lever. No regularisation, stopped at 25k steps: 0.449989, +0.00107 against the reference, 5 of 5 seeds better. This is a stopping-point effect on the GRU only; the other arms were scored at 30k steps. Rule: pin c1f31a9d; result: pin a3ceeda7 (Meridian).

**XGBoost (validation).** The 83-input model: 0.450252 (mean of three seeds, pin 1b7c86fd). The GRU beats the 57-input model on every seed (pin 921fa3dd). Holdout, one read (pin 56316f69): 0.452295 on 3,315 matches, not used for any choice.

**Reproduction.** 43 of 47 retrained GDT runs match their original validation log loss within 0.0001. The four TCN seeds differ by up to 0.0006; likely GPU nondeterminism, not yet confirmed.

**Failure analysis (GDT-trained proxy rows, 297 matches, validation only; `analysis/seg_check.py`).** Across the 12 segments with at least 20,000 seconds, no arm fails to beat the lookup table. The spread between five-seed GDT arms, measured against the GRU, has a median of 0.0011 and a widest value of 0.0016 (planted = 1).

**Weights.** 47 GDT-objective arm-seeds have weights. They are in S3 under `results/weights_b1`–`weights_b8` and `weights_x1`–`weights_x3`, and under `models/gdt_objective/`. All three copies match by SHA-256 (`results/weights_manifest.csv`). The four extra TCN weights (`results/weights_tcnr0`, `tcnr1`, `tcnr3`, `tcnr4`) are in S3 only. The log-loss runs and the regularisation runs saved no weights. The 83-input XGBoost model is `models/winprob_xgb83_final.json` (CPU refit, 5 Nov 2026). The pins inventory records the served model as `winprob_v7_basic_scorelead` (14 inputs) and the 58-input v8 ensemble as not yet servable, so which model ships needs confirming.

**Open.** PII review of the dataset is pending and the dataset is not cleared. The holdout read (pin 0d97f8f6) is on hold: no candidate passes the pinned rules. GDT is the agreed objective (confirmed with the project owner in chat on 7 Nov 2026). Match population: 295 in the explainability deck, 297 in the 6 November report; to reconcile. A 25k-step re-run of all arms is proposed and needs a new pin before any run.

## Pins
064d46c76 (GDT objective), 079f7189 (amendment), c807884d (GDT selection on log-loss runs), 28527ac3 (GDT-trained results), 51a86c98 (round-2 log-loss result), c1f31a9d (regularisation test), 0d97f8f6 (holdout candidate, not run).
