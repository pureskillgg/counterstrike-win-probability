# Check B: batch holdout

Script: `scripts/model/sufficiency/batch_holdout.py`, seed 0. Batch = the `<DD>` folder of the match key. Header match_date is a cross-check only. Every number below was computed in this run (status VERIFIED).

## Counts

| Item | Value | Status |
|---|---|---|
| Kept set reproduces bakeoff.log (channel kept, dup removed, kept rounds, kept matches, states) | 1709, 47, 28683, 1521, 234265 | VERIFIED |
| Header matches by folder day | 01: 67, 02: 65, 03: 100, 04: 109, 05: 83, 06: 104, 07: 89, 08: 1135 | VERIFIED |
| Kept matches by folder day | 01: 50, 02: 48, 03: 85, 04: 78, 05: 58, 06: 77, 07: 68, 08: 1057 | VERIFIED |
| Header match_date equals folder day | 1730 of 1752 (all header rows); kept matches disagreeing: 18 | VERIFIED |

## Splits

| Run | Set | Batch days | Matches | Rounds | States | CT-win share (states) | Status |
|---|---|---|---:|---:|---:|---:|---|
| forward | train (pool minus in-batch minus validation) | 01: 38, 02: 35, 03: 61, 04: 56, 05: 38, 06: 54, 07: 52 | 334 | 5890 | 48748 | 0.5152 | VERIFIED |
| forward | validation (match-level 10%, early stopping) | 01: 2, 02: 4, 03: 8, 04: 5, 05: 5, 06: 9, 07: 4 | 37 | 588 | 4845 | 0.5011 | VERIFIED |
| forward | in-batch test (20%, held out) | 01: 10, 02: 9, 03: 16, 04: 17, 05: 15, 06: 14, 07: 12 | 93 | 1607 | 13251 | 0.5189 | VERIFIED |
| forward | out-of-batch test | 08: 1057 | 1057 | 20598 | 167421 | 0.5200 | VERIFIED |
| reverse | train (pool minus in-batch minus validation) | 08: 761 | 761 | 14859 | 120750 | 0.5195 | VERIFIED |
| reverse | validation (match-level 10%, early stopping) | 08: 85 | 85 | 1588 | 12999 | 0.4895 | VERIFIED |
| reverse | in-batch test (20%, held out) | 08: 211 | 211 | 4151 | 33672 | 0.5334 | VERIFIED |
| reverse | out-of-batch test | 01: 50, 02: 48, 03: 85, 04: 78, 05: 58, 06: 77, 07: 68 | 464 | 8085 | 66844 | 0.5149 | VERIFIED |

## Test metrics

| Run | Test set | States | Log loss | Brier | AUC | ECE (10 bins) | Status |
|---|---|---:|---:|---:|---:|---:|---|
| forward | in-batch: random 20% of days 01-07 | 13251 | 0.43803 | 0.14622 | 0.8714 | 0.0153 | VERIFIED |
| forward | out-of-batch: all of day 08 | 167421 | 0.43999 | 0.14764 | 0.8685 | 0.0168 | VERIFIED |
| forward | constant, in-batch (train rate 0.5152) | 13251 | 0.69246 | 0.24966 | 0.5000 | 0.0037 | VERIFIED |
| forward | constant, out-of-batch (train rate 0.5152) | 167421 | 0.69239 | 0.24962 | 0.5000 | 0.0048 | VERIFIED |
| forward | gap (out-of-batch minus in-batch) | - | +0.00196 | +0.00142 | -0.0029 | +0.0014 | VERIFIED |
| reverse | in-batch: random 20% of day 08 | 33672 | 0.42917 | 0.14366 | 0.8737 | 0.0085 | VERIFIED |
| reverse | out-of-batch: all of days 01-07 | 66844 | 0.44529 | 0.14921 | 0.8661 | 0.0115 | VERIFIED |
| reverse | constant, in-batch (train rate 0.5195) | 33672 | 0.69130 | 0.24908 | 0.5000 | 0.0139 | VERIFIED |
| reverse | constant, out-of-batch (train rate 0.5195) | 66844 | 0.69275 | 0.24980 | 0.5000 | 0.0046 | VERIFIED |
| reverse | gap (out-of-batch minus in-batch) | - | +0.01613 | +0.00555 | -0.0076 | +0.0030 | VERIFIED |

## Gap uncertainty (match bootstrap, 1000 resamples, seed 0)

| Run | Log loss gap 95% CI | Brier gap 95% CI | AUC gap 95% CI | Status |
|---|---|---|---|---|
| forward | [-0.01269, +0.01623] | [-0.00418, +0.00692] | [-0.0129, +0.0074] | VERIFIED |
| reverse | [+0.00384, +0.02833] | [+0.00080, +0.01019] | [-0.0158, +0.0007] | VERIFIED |

## Verdict

- NOT LARGE batch dependence (forward): log loss gap +0.00196, 95% CI [-0.01269, +0.01623]; rule: LARGE if the CI excludes 0 and |gap| >= 0.01.
- forward: device cpu, best_iteration 159, early-stop validation LL 0.44366 (VERIFIED).
- LARGE batch dependence (reverse): log loss gap +0.01613, 95% CI [+0.00384, +0.02833]; rule: LARGE if the CI excludes 0 and |gap| >= 0.01.
- reverse: device cpu, best_iteration 417, early-stop validation LL 0.43595 (VERIFIED).
- Device: cuda requested; xgboost reported no usable GPU and ran on cpu: [16:09:08] WARNING: D:\bld\xgboost-split_1789510239039\work\src\context.cc:218: Device is changed from GPU to CPU as we couldn't find any available GPU on the system.

