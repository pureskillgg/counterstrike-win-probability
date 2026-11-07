# Audit of the zero-shot foundation-model head-to-head (written 2026-10-15)

Source: workflow `wf_2d8c4773-548` (5 lenses, 131 agents, two skeptics per finding; raw output `notes/research/raw/w2sim6uo2.output`). It re-audited
`FM_HEAD_TO_HEAD_2026-10-14.md` after the first audit read nothing (G: was unplugged). Read-only; no number below was recomputed by me beyond what
the auditors report with their evidence. **The head-to-head note and the deck slide `m_foundation` overstate what the data shows; this file is the correction.**

## What must change in what we say
1. **"Foundation models are 3-7x worse calibrated" is withdrawn.** It compared ECE10 on 500-4,000 seconds with XGBoost's ECE10 on 889,123 seconds.
   ECE10 of a *perfectly calibrated* predictor is 0.042 at n=500, 0.030 at n=1000, 0.015 at n=4000 (0.007 at 20,000). On the same seconds XGBoost's
   ECE10 is 0.035 / 0.026 / 0.017, so the matched excess of the foundation models is about 1.1-1.3x, inside the noise. Compare calibration only on the
   same seconds with smooth ECE or an interval (the evaluator now returns an ECE10 interval and smooth ECE).
2. **"Every run worse than XGBoost" is one comparison, not four.** The runs are nested prefixes (500 inside 1,000 inside 4,000 inside 20,000) of one
   randomly ordered test pool; the TabICL 5k and 10k runs score identical seconds. One split, one reference, one pool. No multiplicity correction.
3. **It is a data-budget comparison, not a learner comparison.** The foundation models saw 2,000-10,000 context rows; the reference XGBoost was fitted
   on ~1.09 million seconds. XGBoost refit on the *same* 10,000-row context scores 0.46242 against TabICL's 0.46235 (n=1000), 0.46996 vs 0.46863
   (n=4000), 0.47183 vs TabPFN zero-shot 0.46928 (10,000 seconds); on Mitra's 2,000 rows 0.50379 vs 0.49779. So at equal data a foundation model is
   level with or slightly ahead of XGBoost (TabICL is 0.0079 better than a ridge logit), and XGBoost's lead is what ~100x more training rows buy.
   These matched comparisons have no paired intervals (predictions were not saved at the time).
4. **The reference was called "tuned" but was the project's standard setting** (depth 4, eta 0.05, mcw 20); the 52-input GPU search never completed.
5. **Stale statements:** "fine-tuning not tested" and "GPU unusable" are false as of 2026-10-15: TabPFN v2 was fine-tuned on the GPU
   (`exp_finetune.json`, commit 589c7d8: +0.0046 log loss vs XGBoost at lr 1e-5, 6 epochs; zero-shot +0.0054 on that run's pool). Mitra: one ensemble
   member and 2,000 context rows (AutoGluon's RAM formula refused 5,000), so it is not on the same footing as TabICL.
6. **Timings are not benchmarks:** single runs on a saturated machine, no warm-up, thread count and batch size not recorded; the 12.9 GB figure
   for 30,000 context rows has no artifact behind it (that run never finished); the XGBoost "thousands of rows per second" is asserted, not measured.
7. **Context really holds one second per round, not two** (polars `cum_count` is 1-based; erratum already in the repo). The saved prefixes are
   length-biased, so each context size is a different sample. `prep_fm_data.py` does not drop roster-defect rounds.
8. **Headline generalised from two zero-shot models** (TabICL v2, Mitra, plus a TabPFN v2 run) to "a pretrained tabular foundation model".
9. All intervals are conditional on one match split (seed 0) and one model seed; Mitra's paired interval [0.0010, 0.0259] only just excludes zero and
   was chosen after several aborted attempts on the same pool.

## What can still be said (supported)
* On held-out matches, with ~1 million training seconds, tuned-by-default XGBoost beats every foundation-model configuration we managed to run
  (zero-shot at 2k-10k context; TabPFN v2 fine-tuned on 143k seconds). The lead is small (0.005-0.02 log loss) and its size depends on the data budget.
* At the same context size a foundation model and XGBoost are about level; the foundation models do not scale to the full training set on this
  hardware (TabICL fine-tuning held 15-24 GB of host memory on 20k-143k rows).
* XGBoost on CPU serves a row in a fraction of a millisecond with exact tree SHAP; the foundation models re-read the context on every call.

## Open (not done)
* Fine-tuned arms on the new, larger store (see `FM_FINETUNE_2026-10-15.md` for the status of the runs); equal-data XGBoost with paired intervals;
  Mitra at 5k-10k context; split-to-split and seed-to-seed variation; distillation (teacher to XGBoost) only if a fine-tuned teacher clearly wins.
