# Provenance: log-loss results, GDT-objective runs, and volume uploads (6 Nov 2026)

Pins referenced below are in Meridian project fa6f6e09-513d-4788-9db5-43123df4ce0f.

## 1. Where the log-loss results are saved

**Laptop (durable):** `G:/pureskill/csds-cs2/data/diagnostics/`
- `pos24_screen_gru` (GRU seed 0), `pos24_screen_rest` (LSTM, TCN, transformer seed 0), `pos24_screen_new` (minGRU, minLSTM, LRU seed 0)
- `pos24_round2_screen_laptop` (xLSTM sigmoid, 7B-style block, GDN-2 seed 0)
- `pos24_gru_s12` (GRU seeds 1, 2), `pos24_sig_s12_laptop` (sigmoid seeds 1, 2), `pos24_sig_s4_laptop` (sigmoid seed 4), `pos24_xlstm_fast_s4` (closed-form xLSTM seed 4)
- `pos24_smoke` and `pos24_promo` are excluded (smoke test and empty folder)

**Network volume z7yvc40ykx (durable):** `/workspace/results/pos24_*`, with copies of laptop runs under `results/local_laptop/`
- `pos24_xlstmfast_s012` and `pos24_xlstmfast_s3` (closed-form xLSTM seeds 0 to 3), `pos24_blk_s12` and `pos24_blk_s34` (7B-style block seeds 1 to 4), `pos24_sig_s3` (sigmoid seed 3)
- `pos24_rwkv7_s12` and `pos24_rwkv7_s34` (RWKV-7 seeds 1 to 4), `pos24_mingru_s1234` (minGRU seeds 1 to 4), `pos24_tcn_s1234` (TCN seeds 1 to 4), `pos24_tf_s1234` (transformer seeds 1, 2)
- `pos24_gru_s34` (GRU seeds 3, 4), `pos24_screen_xr` (xLSTM loop seed 0, RWKV-7 seed 0), `pos24_screen_gm` (Griffin, Mamba-2 seed 0), `pos24_screen_lm` (GDN, multi-head xLSTM seed 0), `pos24_cap_s0b` (GRU 2-layer and transformer 2-layer, seed 0)

**Pins for the log-loss results:** 66bca815 (screen design), 6fb5578a, 19151b44, ffa4cfc4 (phase 1), ea145e3b, 7fe1d12b, c734a903, b8a83b32, 31f6825b (round 2 selection), c807884d (GDT selection, which also reports the log-loss numbers).

**Analysis files (copied here on 6 Nov, previously only in the session temp folder):**
- `round2_paired/paired_round2.json` and `paired_round2.py`: the round-2 paired log-loss test against GRU, 1,000 match-cluster resamples, RNG 0. Result: no candidate passes the pinned every-seed rule.
- `round2_paired/summary.json` and `tabulate.py`: per-arm final-step means for every log-loss run.
- `logloss_check/`: the reproduction check, GRU seed 0 at 30k steps, run locally on 6 Nov. Final validation log loss 0.4517706, saved value 0.451771. This shows the log-loss path of the trainer is unchanged.

**Not yet pinned:** the round-2 paired log-loss result itself. It's in `round2_paired/` and in chat only, so it needs a pin.

## 2. Volume items that are not results

| Item | Status |
|---|---|
| `pos24_sig_s12` | Stopped at step 6000 of seed 1. No metrics. The laptop reruns of sigmoid seeds 1 and 2 are the results. |
| `pos24_xlstm_s12` | Ended with no metrics. The log tail shows no training lines for the loop xLSTM. Cause not in the tail. |
| `pos24_xlstm_s3` | Ended at seed 3 start with no metrics. Cause not in the tail. |
| `pos24_cap_s0` | Failed with "unknown arm tf2". Replaced by `pos24_cap_s0b`. |
| `P59_s0.json` | XGBoost, 59 inputs, RTX 4090. Tree model, not a sequence result. |
| `X83_EXPLORATORY_s0.json` | XGBoost, 83 inputs, using all-match site centres. Labelled "EXPLORATORY, not for results." |

## 3. What is being uploaded now (6 Nov)

**Uploaded and checked:**
- `s3://z7yvc40ykx/pod_runner/ws_scripts/model/train_phase1_pos24.py`: overwritten 16:39 UTC with the version that has the `--objective gdt` flag. MD5 `e1b8db9e…` matches the laptop copy. The earlier log-loss-only copy is no longer on the volume. The change is additive, and the log-loss path reproduced (section 1). I haven't checked whether the old copy can be recovered from history.
- `s3://z7yvc40ykx/pod_runner/ws_scripts/model/gdt_grader.py`: new, a verbatim copy of `evaluation/intern_gdt/score_gdt.py`. MD5 `34f5b573…`.

**Running:** the five pods write outputs straight to `/workspace/results/pos24_gdt_p1` to `p5`.
- Pod 1 (`gdt-swarm-01`, id 754jopwug8bdvz, RTX 4090): setup done. First result: `gru_gdt_seed0`, final validation log loss 0.450469 (best 0.449778 at step 27000), 3,530,375 groups, zero test rows used.
- Pods 2 to 5: not yet confirmed. Their setup logs haven't appeared on the volume.
- Pod ids are in `scripts/pods_created.txt`. Each pod is $0.89/hr, about $4.45/hr in total.

**Not uploaded (laptop only):**
- `G:/pureskill/csds-cs2/data/diagnostics/gdt_seq/`: the 356,986-row GDT rows file, the Stage A and Stage B outputs, and `stageB_pooled_results.txt`.

## 4. Gaps

- The overwritten trainer. Recovery is unchecked.
- The round-2 paired log-loss result has no pin.
- The session scratchpad (`C:/Users/13144/AppData/Local/Temp/claude/…/scratchpad`) is temporary. Anything needed must live under `G:` or the project. The `final_results/vol` and `vol_preds` folders there hold temporary copies of volume predictions and can be deleted once the volume copies are confirmed.
- The GDT-objective runs are not yet pinned as results. The objective pin is 64d46c76.
