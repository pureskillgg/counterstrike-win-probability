# Positions validation, paired per-second log loss (pin 773d5c12)

Validation: 1,168 matches. Fit: 90% of the training matches; early stop: the other 10%. No test or holdout B rows used.
Interval: match-cluster bootstrap, 1,000 resamples, seed 0, from paired().

## full_trainctr minus base

| Seed | Difference | 95% interval |
|---|---|---|
| 0 | -0.00349 | -0.00396 to -0.00300 |
| 1 | -0.00352 | -0.00400 to -0.00303 |
| 2 | -0.00353 | -0.00400 to -0.00305 |
| 3 | -0.00349 | -0.00394 to -0.00302 |
| 4 | -0.00343 | -0.00389 to -0.00294 |

Pooled mean over five seeds: -0.00349 (seed sd 0.00004).

## full_oldctr minus base

| Seed | Difference | 95% interval |
|---|---|---|
| 0 | -0.00352 | -0.00399 to -0.00306 |
| 1 | -0.00355 | -0.00403 to -0.00305 |
| 2 | -0.00353 | -0.00401 to -0.00305 |
| 3 | -0.00348 | -0.00393 to -0.00299 |
| 4 | -0.00349 | -0.00391 to -0.00303 |

Pooled mean over five seeds: -0.00351 (seed sd 0.00003).

## full_trainctr minus full_oldctr

| Seed | Difference | 95% interval |
|---|---|---|
| 0 | +0.00004 | -0.00007 to +0.00015 |
| 1 | +0.00003 | -0.00010 to +0.00015 |
| 2 | +0.00000 | -0.00011 to +0.00013 |
| 3 | -0.00001 | -0.00014 to +0.00012 |
| 4 | +0.00007 | -0.00007 to +0.00021 |

Pooled mean over five seeds: +0.00002 (seed sd 0.00003).

Verdict, full_trainctr against base: beats base (pinned bar met).
