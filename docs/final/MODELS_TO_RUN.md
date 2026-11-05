# Models to train and test, and what each needs

Status as of 5 Nov 2026. This is a planning list, not a pin. Each run needs its own pin before it is trained or read. Validation only, unless a pin says otherwise.

## Explainable line: XGBoost, trained on log loss, reported with GDT

| # | Model | Status | Needs |
|---|---|---|---|
| 1 | 59-input control (sprint base arm) | Fitted. Seed mean 0.453754 on 1,168 validation matches | GPU, about 30 s per fit |
| 2 | 75-input (59 plus geometry) | Fitted. Seed mean 0.450597 | GPU, about 60 s per fit |
| 3 | 83-input (59 plus geometry, view and blindness) | Sprint fit exists (0.450252). The ablation fit did not finish | GPU, about 65 s per fit. Finish the fit, then GDT on the 297 matches |
| 4 | 57-input (winprob_v6_drop_heavy) | Booster saved. Validation predictions regenerated, GDT done | CPU is enough |
| 5 | Explainability for 83 (TreeSHAP and permutation) | Script written (`scripts/explain/xgb83_explain.py`), not run | CPU, or the local pixi environment |
| 6 | Visit-weighted training arm (A1) | Proposed, not pinned, not run | GPU, one pinned A/B arm |
| 7 | Shrinkage arm (A2) | Proposed, only if A1 helps overall GDT | GPU, same as 6 |
| 8 | Full-archive XGBoost sweep on the 57,811-match export | Waits on per-second loadouts (v3) | GPU for the fits; CPU for the loadout build |

## Black-box sequence line: comparison only, reported separately with GDT

| # | Model | Status | Needs |
|---|---|---|---|
| 9 | GRU control (lookback 20, 24 position inputs, lr 1e-3, 30k steps, 3 seeds) | Run before. Validation GDT not computed | GPU |
| 10 | Architecture screen, pin 66bca815: GRU, LSTM, minGRU, minLSTM, LRU, causal TCN, causal transformer, seed 0, 30k steps | LSTM, TCN and transformer were run before. minGRU, minLSTM and LRU are not in the code | GPU, one arm per pod is possible |
| 11 | Round-so-far window (up to about 154 s of history) | No script yet, pin needed | GPU |
| 12 | Per-map models: de_mirage (earlier verdict: unlikely to help) and de_dust2 (not started) | Not started | GPU |
| 13 | xLSTM, Mamba-2, RWKV-7, Griffin | Not implemented. Each needs a pin | GPU |
| 14 | Graph transformer (THGNN-lite) | Run before | GPU |

## Data builds and checks (CPU)

| # | Job | Status |
|---|---|---|
| 15 | Per-second loadout build (v3 export) | In progress on the space |
| 16 | Laptop-sourced position parity check (c050, c060) | Not run |
| 17 | Learning curve, 10-feature event-only (pin 238a70d6) | Done on the space. Its output file is lost; the result note is the record |
| 18 | Odd-second positions | Deferred. Would add about 36 million rows and about 1.2 GB |

## Final reads (never more than once per candidate)

- Oct 26 to 31 holdout: used once, for the 83 (0.452295). Not clean for any new comparison.
- Frozen test: used for the 57 package (test log loss 0.456411). Not clean.
- Holdout B: contaminated by the 3 Nov owner ruling.
- Nov 1 to 4: not built, never read. Usable only if pinned before any read.

## Compute notes

- The sprint report puts median GPU fit time at about 29 s for the base arm and about 64 s for the full arm, on 15 million training rows.
- GPU memory for XGBoost on this data fits in 24 GB. Sequence-model memory has not been measured here.
- A test pod on RunPod (RTX 4090, 24 GB, EU-RO-1, Secure Cloud, $0.89/hr) started and passed a CUDA check on 5 Nov, then was terminated.
- Stock is "Low" wherever it appears. Expect one or two pods per datacenter, and check again before each deploy.
- Each pod needs its data copied in, straight from the space or from the laptop. The laptop-to-space route ran at about 5 MB/min.
