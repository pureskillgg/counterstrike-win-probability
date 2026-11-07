# Foundation models against the boosted trees: head-to-head, 2026-10-14

Question (from the user, after arXiv 2407.00956 TALENT and 2510.21204 Mitra): would a lightweight, fine-tunable, pretrained tabular foundation model
be a wise option, possibly better than XGBoost, and easy to run on a CPU?

## Setup (scripts/fm/)
* Same rows, 52 inputs, split (random match split, seed 0) as exp_model_families.py; 2026 snapshot table.
* Context: labelled seconds from TRAINING matches only, ONE second per round (the code meant 'at most two' but polars' cum_count is 1-based, so the cap of 2 kept only the first second of each round: 47,574 rounds -> 47,574 context rows; found 2026-10-15, fixed in prep_fm_data.py and finetune_*.py; the results below used the one-per-round context), random order (`prep_fm_data.py`; the first n rows are an
  n-row context). Test: seconds drawn at random from the 743 held-out matches. Reference: the final XGBoost settings fitted on the whole training set
  (CPU, stride 3, early stopping on validation matches) and scored on exactly the same seconds, so every difference is PAIRED, with a match-cluster
  bootstrap interval (1,000 resamples).
* Models: TabICL v2 (tabicl 2.2.0, checkpoint tabicl-classifier-v2-20260212, BSD-3-Clause code) and Mitra (autogluon.tabular 1.6.3 [mitra], Apache-2.0
  weights). No fine-tuning (the RTX 3080 went into an error state at ~22:15; nothing GPU-based could run). CPU: Ryzen 7 7840U, 16 threads, shared at
  ~100% load with two other fine-tuning jobs, so the speeds below are rough upper bounds on how slow, not benchmarks.

## Results (data/combined/exp_foundation.json)
| model | context rows | test seconds | log loss | XGBoost, same seconds | paired difference (95%) | AUC (XGB) | ECE | predict rate |
|---|---|---|---|---|---|---|---|---|
| TabICL v2, 4 members | 5,000 | 1,000 | 0.4630 | 0.4513 | +0.0118 (0.0048 to 0.0198) | 0.8518 (0.8573) | 0.031 | 6.7 rows/s |
| TabICL v2, 4 members | 10,000 | 1,000 | 0.4623 | 0.4513 | +0.0111 (0.0042 to 0.0186) | 0.8521 (0.8573) | 0.034 | 3.3 rows/s |
| TabICL v2, 8 members | 10,000 | 4,000 | 0.4686 | 0.4597 | +0.0089 (0.0046 to 0.0136) | 0.8485 (0.8526) | 0.025 | 5.3 rows/s |
| Mitra | 2,000 | 500 | 0.4978 | 0.4844 | +0.0134 (0.0010 to 0.0259) | 0.8264 (0.8320) | 0.049 | 4.4 rows/s (fit 322 s) |

XGBoost's own calibration error on the held-out matches is about 0.007; its prediction rate on a CPU is thousands of rows per second.

## Reading
* Every run is worse than XGBoost in log loss and the paired interval excludes zero in every run. AUC is within ~0.004-0.006 (ranking is nearly as good);
  the loss is in the probabilities (calibration error 3-7 times larger).
* More context did not help (TabICL 5,000 -> 10,000 rows: 0.4630 -> 0.4623). Mitra was tested only at 2,000 rows because AutoGluon estimates 18.6 GB of
  RAM for a 5,000-row context on this machine; TabICL at 30,000 rows held 12.9 GB. So "easy to run on a CPU" is not true on this hardware at useful
  context sizes.
* A live service needs about one row per second per match; the foundation models reached 3-7 rows/s on a loaded CPU, in batch, with no per-match state.
  With TabICL's kv_cache option this could improve (not tested).
* This matches the research: no 2025-2026 source documents a CPU real-time tabular foundation model beating tuned boosted trees on log loss with
  match-grouped splits (notes/research/RESEARCH_ROUND4_2026-10-14.md); TabPFN-2.5+ and LimiX-2 weights are non-commercial and were not used.

## What was NOT tested
Fine-tuning (full, LoRA, last layer) of either model; GPU inference; TabICL with kv_cache; Mitra above 2,000 context rows; distillation of a
foundation model into the trees (the cheapest remaining idea: a grouped out-of-fold teacher, a small gain at best according to the Pocket Foundation
Models paper, +0.001 AUC above 21 features).

## Decision
Keep the XGBoost snapshot model (+ alive-count fallback) for Friday. Re-test fine-tuning only if the GPU comes back and time allows; the expected gain is
small and the CPU serving cost (PyTorch dependency, context at prediction time, seconds per hundred rows) is large.


## Erratum and later results (2026-10-15)
* Context sampling: see the corrected line above (one second per round, not two). This does not bias the comparison against the foundation models (more distinct rounds per context row), but the write-up said otherwise.
* Fine-tuning arm and TabPFN v2 zero-shot: see `notes/research/FM_FINETUNE_2026-10-15.md` (written when those runs finish).
