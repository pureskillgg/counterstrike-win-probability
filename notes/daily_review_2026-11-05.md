# Daily review: Thursday 5 November 2026

Built from the Meridian pins, the volume upload log and laptop folder timestamps. Pin times are UTC converted to US Central (UTC minus 6). Laptop folder times are the machine clock. Anything decided in chat and not pinned is not in this review.

## Summary

- **Protocol set, then contested.** The selection protocol (pin 64b9d52d) was filed at 13:16 CST. It sets GDT as the selection metric and records log-loss training, as the user's report of Dr Freeman's approval. The user has since reported that Freeman said to train on GDT. The two accounts conflict and need the written message.
- **Features frozen for the tree model.** The 83-input model became the delivered tree model, and the defuse-margin input was dropped.
- **Data pipeline fixes.** Missing sequence inputs were uploaded to the volume, which unblocked the RunPod sequence screen.
- **Sequence screen queued.** Arms H to K were pinned late at night and the seed-0 screen ran locally.

## Timeline (CST)

**Afternoon: protocol and features**
- 13:16: protocol pin 64b9d52d (selection on GDT, log loss alongside, training on log loss).
- 13:56: pin 56ae8e7d, 77-input feature set and defuse-margin arm, validation only.
- 14:21: pin 0cf889ba, 83 inputs final, defuse margin dropped.
- 15:09: pin 425fca91, keep the 2,848 Oct 21–25 matches, freeze-side defect fixed in export v4 (283 rounds).
- 16:37: pin e53ed710, 83 stays the delivered tree model, 77 for temporal and explainability, compute local plus RunPod.

**Afternoon: data uploads**
- 13:54: `clone.log` to the volume.
- 14:57: `cache_snapshot_2026.parquet` and `split_manifest.json`.
- 15:19: `cache_pos_2026_allmatch_EXPLORATORY.parquet`, labelled exploratory.
- 15:54: `cache_pos_2026_trainctr.parquet`.
- 15:15 to 15:24: validation and frozen-test key files.
- 14:42 to 15:39: gzipped round-state and position staging for the pods.
- 17:10 to 17:15: per-map grid position tables.
- 17:21 to 17:23: `rounds.parquet` and `cache_snapshot_2026_ext.parquet`. These fixed the pod runs that had failed on a missing-file error.

**Evening: tree-model work and sequence screen**
- 19:46: pin 2d4e814e, include the Oct 21–25 batch in export-based training, with a before-and-after validation run.
- 19:57: pin f9053b41, local tree queue on the sprint matrix (validation only).
- 20:24: pin 7fe1d12b, amendment allowing RunPod for the sequence screen, plus the approval wording.
- 20:58 to 21:22: pins 82fd6b58, 91c606c5, 3f7fa640 and 74ac8cdf, tree sweeps, a RunPod pipeline check, and a seed check of the top tree candidates.
- 23:22: pins 6a674e1e, 6580d4fc, 4025d97d and d47a24f1 for sequence arms H to K (xLSTM mLSTM, RWKV-7, Griffin, Mamba-2).
- Laptop, machine clock 22:18 to 23:59: `pos24_smoke`, then the seed-0 screen runs `pos24_screen_gru`, `pos24_screen_rest` and `pos24_screen_new`. `pos24_promo` is an empty folder.

## Open items carried into Friday

1. **Training objective.** The protocol pin (13:16) records log-loss training. The user reports Freeman said GDT training at about noon. The pin was filed after noon, so either the pin misrecorded the message or the message isn't as described. Freeman's written words are needed.
2. **Approval dating.** Pin 7fe1d12b was filed at 20:24 CST Thursday, but its text says the approval was given on 6 November. Confirm the actual date.
3. **Unpinned Thursday decisions.** Any Thursday chat decisions not pinned above are missing from this review.
