# Round-quality evaluation: intern_oct14_25_positions_vs_base

Scored population: holdout B, the 653 test-assigned matches added by the Oct 22 to 25 extension (upload days Oct 22 to 25; keys in data/combined/holdout_22_25_test_keys.json). Not the Oct 14 to 25 intern window. This run scored holdout B before any owner go (pin dd78df27 says unscored), so it is not a clean test. Reference is p_base, the 59-input control trained on the same rows as p_xgb83. Caveats: the six site-centre features are fitted on plant positions from all matches including this population; each arm is a single seed-0 fit, and fit_xgb subsamples 80% of rows per tree, so the seed matters.

Reference: `p_base`. Rows: 872,038 seconds, 653 matches, 12,645 rounds, 119,173 grader datapoints in 1,228 situations (44 with 500 or more). Seconds per round: 69.0. Seed 0, 1000 resamples of matches.

Positive differences mean the model beats the reference. Intervals resample matches.

## Headline: GDT against the table

| Model | GDT (95%) | GDT_sit (95%) | rGDT | Matches better than table | GDT minus reference (95%) | GDT_sit minus reference (95%) |
|---|---|---|---|---|---|---|
| `p_base` | 0.0183 (0.0162 to 0.0204) | 0.0149 (0.0124 to 0.0175) | 4.1% | 75.8% | reference | reference |
| `p_xgb83` | 0.0228 (0.0206 to 0.0249) | 0.0188 (0.0161 to 0.0215) | 5.1% | 80.4% | +0.0044 (+0.0037 to +0.0052) | +0.0038 (+0.0029 to +0.0049) |
| `p_v6` | 0.0184 (0.0162 to 0.0204) | 0.0148 (0.0123 to 0.0174) | 4.1% | 74.7% | +0.0000 (-0.0001 to +0.0002) | -0.0001 (-0.0003 to +0.0002) |

## Diagnostics (per second, all rows)

| Model | Log loss | AUC |
|---|---|---|
| `p_base` | 0.45198 | 0.86063 |
| `p_xgb83` | 0.44825 | 0.86267 |
| `p_v6` | 0.45200 | 0.86065 |

## By situation: difference from the reference (model better when positive)

### overall

| Group | Datapoints | Matches | `p_xgb83` | `p_v6` |
|---|---|---|---|---|
| all | 119,173 | 653 | +0.0044 (+0.0037 to +0.0052) | +0.0000 (-0.0001 to +0.0002) |

### phase and time left

| Group | Datapoints | Matches | `p_xgb83` | `p_v6` |
|---|---|---|---|---|
| post-plant little left | 3,632 | 608 | +0.0088 (+0.0048 to +0.0131) | +0.0002 (-0.0006 to +0.0010) |
| post-plant lot left | 13,443 | 650 | +0.0075 (+0.0049 to +0.0099) | -0.0001 (-0.0005 to +0.0004) |
| post-plant medium left | 8,750 | 649 | +0.0091 (+0.0057 to +0.0127) | +0.0005 (-0.0001 to +0.0010) |
| pre-plant little left | 5,355 | 609 | +0.0093 (+0.0058 to +0.0130) | -0.0004 (-0.0009 to +0.0002) |
| pre-plant lot left | 58,912 | 653 | +0.0023 (+0.0015 to +0.0031) | +0.0000 (-0.0002 to +0.0003) |
| pre-plant medium left | 29,081 | 653 | +0.0046 (+0.0033 to +0.0058) | -0.0001 (-0.0003 to +0.0002) |

### players alive

| Group | Datapoints | Matches | `p_xgb83` | `p_v6` |
|---|---|---|---|---|
| CT +1 | 23,316 | 652 | +0.0038 (+0.0024 to +0.0051) | +0.0001 (-0.0002 to +0.0005) |
| CT +2 or more | 19,808 | 649 | +0.0008 (-0.0000 to +0.0016) | +0.0002 (-0.0001 to +0.0004) |
| T +1 | 21,106 | 652 | +0.0066 (+0.0050 to +0.0083) | -0.0001 (-0.0004 to +0.0003) |
| T +2 or more | 17,250 | 648 | +0.0038 (+0.0027 to +0.0051) | -0.0000 (-0.0003 to +0.0002) |
| even | 37,693 | 653 | +0.0058 (+0.0047 to +0.0069) | -0.0001 (-0.0004 to +0.0002) |

### kit level

| Group | Datapoints | Matches | `p_xgb83` | `p_v6` |
|---|---|---|---|---|
| CT better kit | 29,673 | 650 | +0.0047 (+0.0034 to +0.0061) | +0.0003 (+0.0000 to +0.0007) |
| T better kit | 32,060 | 645 | +0.0045 (+0.0032 to +0.0058) | -0.0002 (-0.0006 to +0.0002) |
| same kit level | 57,440 | 653 | +0.0042 (+0.0032 to +0.0053) | -0.0001 (-0.0003 to +0.0002) |

### pre-plant: players alive

| Group | Datapoints | Matches | `p_xgb83` | `p_v6` |
|---|---|---|---|---|
| CT +1 | 19,637 | 652 | +0.0031 (+0.0018 to +0.0042) | -0.0000 (-0.0003 to +0.0002) |
| CT +2 or more | 16,987 | 649 | +0.0007 (-0.0001 to +0.0014) | +0.0001 (-0.0002 to +0.0003) |
| T +1 | 15,040 | 651 | +0.0052 (+0.0036 to +0.0071) | -0.0003 (-0.0007 to +0.0001) |
| T +2 or more | 9,707 | 648 | +0.0050 (+0.0034 to +0.0066) | +0.0001 (-0.0003 to +0.0005) |
| even | 31,977 | 653 | +0.0037 (+0.0027 to +0.0047) | -0.0000 (-0.0003 to +0.0003) |

### pre-plant: kit level

| Group | Datapoints | Matches | `p_xgb83` | `p_v6` |
|---|---|---|---|---|
| CT better kit | 24,058 | 649 | +0.0036 (+0.0022 to +0.0048) | +0.0001 (-0.0002 to +0.0005) |
| T better kit | 25,225 | 644 | +0.0041 (+0.0026 to +0.0056) | -0.0002 (-0.0006 to +0.0002) |
| same kit level | 44,065 | 653 | +0.0029 (+0.0019 to +0.0039) | -0.0000 (-0.0002 to +0.0002) |

### pre-plant, lot: kit level

| Group | Datapoints | Matches | `p_xgb83` | `p_v6` |
|---|---|---|---|---|
| CT better kit | 16,679 | 648 | +0.0031 (+0.0017 to +0.0045) | +0.0001 (-0.0003 to +0.0004) |
| T better kit | 16,255 | 640 | +0.0023 (+0.0009 to +0.0038) | -0.0001 (-0.0007 to +0.0004) |
| same kit level | 25,978 | 653 | +0.0017 (+0.0006 to +0.0028) | +0.0001 (-0.0001 to +0.0004) |

## Notes

- The table is fixed (`evaluation/intern_gdt/table.csv`, built on upload days Oct 9 to 13, 2026).
- Groups were defined before the run (`analysis/group_gdt.py`). Read any single interval with care: many groups are shown at once.
- This run's positions models predict every second (872,038 rows over 12,645 rounds, all seconds present). The every-other-second subsample described in the protocol does not apply here.
