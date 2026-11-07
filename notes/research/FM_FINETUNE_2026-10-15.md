# Fine-tuning a tabular foundation model on the round data (TabPFN v2): first result, 2026-10-15

The 2026-10-14 head-to-head (TabICL v2, Mitra) was zero-shot / in-context only. This is the follow-up the user asked for: does FINE-TUNING close the gap to the tuned XGBoost?

## Data used (all from the 3,716-match 2026 snapshot cache; the ~10,000 matches built on days 9-13 are NOT in these runs yet)
| part | matches | seconds | notes |
|---|---|---|---|
| all usable | 3,716 | 4,458,302 (65,923 rounds) | competitive 5v5, 7 maps, 2026 only (2024 has no player table) |
| fine-tune TRAIN | 2,676 | 142,722 | 3 seconds per round (47,574 training rounds), random order |
| validation (early stopping, log loss) | 297 | 10,500 | up to 2 seconds per round; an explicit match-grouped set, not a random split of training rows |
| TEST | 743 (736 present in the sample) | 10,000 | random seconds from the held-out matches; never used for fitting or stopping |
| XGBoost reference | same 2,676 train, same 297 val | 1,089,000 (every 3rd second) | scored on the same 10,000 test seconds; the comparison is PAIRED with a match-cluster bootstrap |

## Result (data/combined/exp_finetune.json; predictions in data/combined/fm/pred_*.npy)
| arm | log loss | 95% interval | paired difference to XGBoost | AUC | calibration error | time |
|---|---|---|---|---|---|---|
| XGBoost (reference, same seconds) | 0.4639 | 0.4535-0.4749 | 0 | 0.8517 | ~0.007 | |
| TabPFN v2 zero-shot, 10,000 context rows | 0.4693 | 0.4580-0.4808 | +0.0054 (0.0029 to 0.0080) | 0.8494 | 0.0126 | predict 81 s / 10k rows |
| TabPFN v2 fine-tuned, 6 epochs, lr 1e-5, 5,000-row meta-datasets, 10,000 context rows | 0.4685 | 0.4566-0.4805 | +0.0046 (0.0021 to 0.0074) | 0.8507 | 0.0240 | fit 2,552 s (GPU shared with another job), predict 62 s / 10k rows |

## Reading
* Fine-tuning moved log loss by about 0.0008 (0.4693 -> 0.4685): inside the noise of 10,000 seconds, and the paired difference to XGBoost is still significantly positive (+0.0046, interval excludes 0). Calibration got WORSE (0.0126 -> 0.0240).
* So with default-like settings (lr 1e-5, 174 optimiser steps) fine-tuning did not close the gap. That is a real but narrow statement: it does not cover higher learning rates, many more steps, larger meta-datasets, other models (TabICL v2 runs follow), or fine-tuning on the 10k newer matches.
* The zero-shot TabPFN v2 is already much closer to XGBoost (+0.0054) than TabICL v2 was (+0.009 to +0.012), with far better calibration (0.013 vs 0.025-0.034).
* Per-epoch validation loss was not recorded (no experiment logger was attached), so it is unknown whether validation loss was still falling at epoch 6. Next runs keep it.
* Not a real-time model: in-context, GPU, 62 s for 10,000 rows (160 rows/s) with a shared GPU; no CPU claim.
* Licence: TabPFN v2 weights, Prior Labs License v1.1 (Apache 2.0 + attribution, "Built with PriorLabs-TabPFN"); a distilled student of it carries a naming clause. TabICL v2 (BSD-3-Clause) has no such clause.

## Queued (scripts/fm/run_ft_suite.ps1, logs/ft_suite.log)
TabICL v2 zero-shot + fine-tune (lr 1e-5); TabPFN v2 fine-tune lr 5e-5 x 8 epochs; TabICL v2 fine-tune lr 5e-5 x 8 epochs.
