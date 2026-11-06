# Handover: deep model line

Status as of Nov 6, 2026. This is a handover, not a finished model. No deep-model training was run for this handover. Each run needs a pin before it is trained or read, and most need a GPU.

The planning list is [docs/final/MODELS_TO_RUN.md](../../docs/final/MODELS_TO_RUN.md). The metric definitions are in [docs/final/TRAINING_METRIC.md](../../docs/final/TRAINING_METRIC.md). Read both first.

## Terms used below

- **GDT**: the table's log loss minus the model's log loss, on the same sampled datapoints (one per round per situation). Positive means the model beats the fixed lookup table. It is a scoring metric, not a training objective.
- **Validation** means the 1,168 validation matches, or the 297-match GDT population (Sept 1 to Oct 8). These are different populations, so their log losses are not comparable.
- **Pin**: a recorded, dated commitment to a run's design, made before any training or read.

## Where each line stands

| # | Run | Status | To finish |
|---|---|---|---|
| 1 | 59-input control | Fitted, seed mean 0.453754 | Nothing |
| 2 | 75-input | Fitted, seed mean 0.450597 | Nothing |
| 3 | 83-input | Sprint fit exists (0.450252, 3-seed mean). **The ablation fit did not finish.** | Finish the fit (about 65 s per fit on GPU), then compute GDT on the 297 matches |
| 4 | 57-input (v6) | Booster saved; GDT done | Nothing |
| 5 | 83-input explanation | Done on CPU: [docs/explainability/](../../docs/explainability/README.md) | Nothing |
| 6 | Visit-weighted training (A1) | Proposed, **not pinned** | Pin, then one GPU A/B arm against plain log loss |
| 7 | Shrinkage (A2) | Proposed, only if A1 helps overall GDT | Waits on 6 |
| 8 | Full-archive XGBoost sweep on the 57,811-match export | Waits on per-second loadouts (row 15) | Row 15, then GPU fits |
| 9 | GRU control (lookback 20, 24 position inputs, lr 1e-3, 30k steps, 3 seeds) | Run before; **validation GDT not computed** | Compute GDT (GPU) |
| 10 | Architecture screen, pin 66bca815 | LSTM, TCN and transformer run before. **minGRU, minLSTM and LRU are not in the code** | Write the three architectures, then run the screen on GPU (one arm per pod is possible) |
| 11 | Round-so-far window (up to about 154 s of history) | **No script yet**, pin needed | Write the script, pin, run |
| 12 | Per-map models: de_mirage, de_dust2 | de_mirage earlier judged unlikely to help; de_dust2 not started | Pin and run de_dust2 if wanted |
| 13 | xLSTM, Mamba-2, RWKV-7, Griffin | Not implemented; each needs a pin | Implement, pin, run |
| 14 | Graph transformer (THGNN-lite) | Run before | Compute GDT if not done |
| 15 | Per-second loadout build (v3 export) | In progress on the space | Finish the build |
| 16 | Laptop position parity check (c050, c060) | Not run | Run (CPU) |
| 17 | 10-feature event-only learning curve (pin 238a70d6) | Done. The output file is lost; the result note is the record | Nothing |
| 18 | Odd-second positions | Deferred (about 36 million rows, about 1.2 GB) | Owner decision |

## Data that must not be read again without a go-ahead

- **Oct 26–31 holdout:** read once for the 83-input candidate (0.452295). Not clean for any new comparison. Its metric needs a pin, and the candidate must be named to Dr Freeman before any final read.
- **Frozen test:** read once for the 57-input package (test log loss 0.456411). Not clean.
- **Holdout B:** contaminated by the Nov 3 owner ruling.
- **Nov 1 to 4:** not built and never read. Usable only if pinned before any read.

## Compute and infrastructure

- Median GPU fit: about 29 s for the base arm and about 64 s for the full arm, on 15 million training rows.
- XGBoost fits in 24 GB of GPU memory. Sequence-model memory has not been measured.
- A RunPod test pod (RTX 4090, 24 GB, EU-RO-1, Secure Cloud, $0.89/hr) passed a CUDA check on Nov 5 and was then terminated.
- Stock is "Low" wherever it appears. Check again before each deploy.
- Copying data from the laptop to the space ran at about 5 MB/min. Plan for that before starting a pod.

## Decisions already made

- **Checklist deviation (owner decision, Nov 5):** the 83-input model keeps `st_armor_sum_ct`, `st_armor_sum_t`, and the sees-enemy and exposed inputs, against the checklist. This is disclosed in [models/winprob_xgb83_final.meta.json](../../models/winprob_xgb83_final.meta.json). The fallback is the 77-input variant, which drops the armour sums and the VIEW inputs. Use it only if a learned temporal model favours it on validation, pinned before the run.
- **Objective:** all XGBoost models were trained on log loss, not GDT. Visit weighting would be a separate pinned arm.

## What is needed from the owner

1. A pin for each run in rows 6, 10, 11, 12 and 13 before any training.
2. A go-ahead before any read of the frozen test, the holdout, or Nov 1 to 4.
3. A decision on row 18 (odd-second positions) and on whether row 12's de_dust2 is wanted.
