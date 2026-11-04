# Validation sprint report

Validation only: 1,168 matches, the split of experiments/positions/positions_validation.py. Holdout B and the frozen test were not read.
Selection on validation is optimistic for the selected setting. Nothing here is a test result.

## Reproduction (default setting, seed 0)

| Arm | Device | Log loss | Pinned | Gap | Position table | Predictions within 1e-5 of pinned |
|---|---|---|---|---|---|---|
| full | cuda | 0.45029 | 0.45032 | -0.00003 | cache_pos_2026_trainctr.parquet | no (max 4.60e-01) |
| base | cuda | 0.45378 | 0.45381 | -0.00003 | cache_pos_2026_trainctr.parquet | no (max 1.60e-01) |

## Grid: full minus base, paired on validation, seeds 0-2

| depth | eta | min_child_weight | full LL (seeds) | base LL (seeds) | full minus base (pooled) | verdict |
|---|---|---|---|---|---|---|
| 3 | 0.03 | 20 | 0.45036 (3/3) | 0.45374 (3/3) | -0.00338 | first arm better (bar met) |
| 3 | 0.03 | 100 | 0.45033 (3/3) | 0.45369 (3/3) | -0.00337 | first arm better (bar met) |
| 3 | 0.05 | 20 | 0.45029 (3/3) | 0.45377 (3/3) | -0.00348 | first arm better (bar met) |
| 3 | 0.05 | 100 | 0.45026 (3/3) | 0.45374 (3/3) | -0.00348 | first arm better (bar met) |
| 3 | 0.1 | 20 | 0.45033 (3/3) | 0.45379 (3/3) | -0.00346 | first arm better (bar met) |
| 3 | 0.1 | 100 | 0.45033 (3/3) | 0.45381 (3/3) | -0.00347 | first arm better (bar met) |
| 4 | 0.03 | 20 | 0.45025 (3/3) | 0.45375 (3/3) | -0.00350 | first arm better (bar met) |
| 4 | 0.03 | 100 | 0.45025 (3/3) | 0.45375 (3/3) | -0.00350 | first arm better (bar met) |
| 4 | 0.05 | 20 | 0.45032 (3/3) | 0.45377 (3/3) | -0.00344 | first arm better (bar met) |
| 4 | 0.05 | 100 | 0.45031 (3/3) | 0.45382 (3/3) | -0.00351 | first arm better (bar met) |
| 4 | 0.1 | 20 | 0.45053 (3/3) | 0.45395 (3/3) | -0.00342 | first arm better (bar met) |
| 4 | 0.1 | 100 | 0.45037 (3/3) | 0.45391 (3/3) | -0.00354 | first arm better (bar met) |
| 6 | 0.03 | 20 | 0.45043 (3/3) | 0.45387 (3/3) | -0.00344 | first arm better (bar met) |
| 6 | 0.03 | 100 | 0.45042 (3/3) | 0.45386 (3/3) | -0.00344 | first arm better (bar met) |
| 6 | 0.05 | 20 | 0.45049 (3/3) | 0.45391 (3/3) | -0.00342 | first arm better (bar met) |
| 6 | 0.05 | 100 | 0.45046 (3/3) | 0.45394 (3/3) | -0.00348 | first arm better (bar met) |
| 6 | 0.1 | 20 | 0.45060 (3/3) | 0.45406 (3/3) | -0.00345 | first arm better (bar met) |
| 6 | 0.1 | 100 | 0.45060 (3/3) | 0.45407 (3/3) | -0.00347 | first arm better (bar met) |

## Selected full setting

Selected by the pinned rule (lowest full LL over seeds 0-2): depth 4, eta 0.03, min_child_weight 20.

## Feature-group ablations at the default setting, seeds 0-2

Difference is ablation set minus full, per second. Negative: the ablation is better, so the dropped groups hurt. Positive: the full set is better, so the group helps.

| Set | Seeds | Set minus full (pooled) | Reading |
|---|---|---|---|
| base+GEO | 3/3 | +0.00032 | no claim |
| base+VIEW | 3/3 | +0.00278 | full better (group helps) |
| base+BLIND | 3/3 | +0.00343 | full better (group helps) |
| base+VIEW+BLIND | 3/3 | +0.00278 | full better (group helps) |
| base+GEO+BLIND | 3/3 | +0.00032 | no claim |
| base+GEO+VIEW | 3/3 | -0.00004 | no claim |

## Timing

Cells ok: 126; cells with errors or unreadable files: 0. Summed fit time 7,046 s; median fit 49.5 s.

Median fit seconds by set: base 29.1, base+BLIND 52.3, base+GEO 58.0, base+GEO+BLIND 69.6, base+GEO+VIEW 69.1, base+VIEW 49.8, base+VIEW+BLIND 66.1, full 64.1

