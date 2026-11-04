# Round-quality evaluation: intern_oct14_25_positions

Scored population: holdout B, the 653 test-assigned matches added by the Oct 22 to 25 extension (upload days Oct 22 to 25, match dates Oct 21 to 25; keys in data/combined/holdout_22_25_test_keys.json). These are not the Oct 14 to 25 intern window. Holdout B was registered as unscored for the candidate-B read (pin dd78df27), and this run scored it before any owner go, so the numbers here are not a clean test. p_base and p_xgb83 are trained on the same 15.27M rows from the training-cache train and validation matches (10,577 and 1,168 matches); p_xgb83 adds 24 position, view and blindness inputs. Caveats: the six site-centre features are fitted on plant positions from all matches including this population; each arm is a single seed-0 fit.

Reference: `p_v6`. Rows: 872,038 seconds, 653 matches, 12,645 rounds, 119,173 grader datapoints in 1,228 situations (44 with 500 or more). Seconds per round: 69.0. Seed 0, 1000 resamples of matches.

Positive differences mean the model beats the reference. Intervals resample matches.

## Headline: GDT against the table

| Model | GDT (95%) | GDT_sit (95%) | rGDT | Matches better than table | GDT minus reference (95%) | GDT_sit minus reference (95%) |
|---|---|---|---|---|---|---|
| `p_v6` | 0.0184 (0.0162 to 0.0204) | 0.0148 (0.0123 to 0.0174) | 4.1% | 74.7% | reference | reference |
| `p_v4` | 0.0186 (0.0164 to 0.0206) | 0.0150 (0.0125 to 0.0176) | 4.2% | 75.5% | +0.0002 (+0.0000 to +0.0004) | +0.0001 (-0.0001 to +0.0004) |
| `p_exp` | 0.0188 (0.0167 to 0.0209) | 0.0151 (0.0125 to 0.0177) | 4.2% | 76.1% | +0.0004 (+0.0001 to +0.0008) | +0.0003 (-0.0001 to +0.0007) |
| `p_base` | 0.0183 (0.0162 to 0.0204) | 0.0149 (0.0124 to 0.0175) | 4.1% | 75.8% | -0.0000 (-0.0002 to +0.0001) | +0.0001 (-0.0002 to +0.0003) |
| `p_xgb83` | 0.0228 (0.0206 to 0.0249) | 0.0188 (0.0161 to 0.0215) | 5.1% | 80.4% | +0.0044 (+0.0038 to +0.0051) | +0.0039 (+0.0029 to +0.0050) |

## Diagnostics (per second, all rows)

| Model | Log loss | AUC |
|---|---|---|
| `p_v6` | 0.45200 | 0.86065 |
| `p_v4` | 0.45187 | 0.86068 |
| `p_exp` | 0.45154 | 0.86103 |
| `p_base` | 0.45198 | 0.86063 |
| `p_xgb83` | 0.44825 | 0.86267 |

## By situation: difference from the reference (model better when positive)

### overall

| Group | Datapoints | Matches | `p_v4` | `p_exp` | `p_base` | `p_xgb83` |
|---|---|---|---|---|---|---|
| all | 119,173 | 653 | +0.0002 (+0.0000 to +0.0004) | +0.0004 (+0.0001 to +0.0008) | -0.0000 (-0.0002 to +0.0001) | +0.0044 (+0.0038 to +0.0051) |

### phase and time left

| Group | Datapoints | Matches | `p_v4` | `p_exp` | `p_base` | `p_xgb83` |
|---|---|---|---|---|---|---|
| post-plant little left | 3,632 | 608 | +0.0021 (+0.0002 to +0.0040) | +0.0010 (-0.0005 to +0.0026) | -0.0002 (-0.0010 to +0.0006) | +0.0086 (+0.0044 to +0.0130) |
| post-plant lot left | 13,443 | 650 | +0.0001 (-0.0003 to +0.0006) | +0.0011 (+0.0003 to +0.0018) | +0.0001 (-0.0004 to +0.0005) | +0.0075 (+0.0049 to +0.0100) |
| post-plant medium left | 8,750 | 649 | -0.0000 (-0.0007 to +0.0006) | +0.0005 (-0.0006 to +0.0015) | -0.0005 (-0.0010 to +0.0001) | +0.0086 (+0.0053 to +0.0123) |
| pre-plant little left | 5,355 | 609 | -0.0003 (-0.0009 to +0.0004) | +0.0003 (-0.0009 to +0.0015) | +0.0004 (-0.0002 to +0.0009) | +0.0097 (+0.0062 to +0.0133) |
| pre-plant lot left | 58,912 | 653 | +0.0002 (-0.0001 to +0.0004) | +0.0003 (-0.0001 to +0.0008) | -0.0000 (-0.0003 to +0.0002) | +0.0022 (+0.0015 to +0.0031) |
| pre-plant medium left | 29,081 | 653 | +0.0003 (-0.0000 to +0.0005) | +0.0003 (-0.0003 to +0.0008) | +0.0001 (-0.0002 to +0.0003) | +0.0047 (+0.0034 to +0.0058) |

### players alive

| Group | Datapoints | Matches | `p_v4` | `p_exp` | `p_base` | `p_xgb83` |
|---|---|---|---|---|---|---|
| CT +1 | 23,316 | 652 | +0.0003 (-0.0001 to +0.0007) | +0.0006 (-0.0000 to +0.0012) | -0.0001 (-0.0005 to +0.0002) | +0.0037 (+0.0023 to +0.0049) |
| CT +2 or more | 19,808 | 649 | +0.0003 (-0.0000 to +0.0007) | +0.0005 (-0.0001 to +0.0011) | -0.0002 (-0.0004 to +0.0001) | +0.0007 (-0.0002 to +0.0015) |
| T +1 | 21,106 | 652 | +0.0002 (-0.0002 to +0.0006) | +0.0007 (-0.0000 to +0.0013) | +0.0001 (-0.0003 to +0.0004) | +0.0067 (+0.0051 to +0.0083) |
| T +2 or more | 17,250 | 648 | +0.0001 (-0.0002 to +0.0004) | -0.0004 (-0.0009 to +0.0002) | +0.0000 (-0.0002 to +0.0003) | +0.0039 (+0.0027 to +0.0052) |
| even | 37,693 | 653 | +0.0001 (-0.0002 to +0.0004) | +0.0006 (+0.0000 to +0.0011) | +0.0001 (-0.0002 to +0.0004) | +0.0059 (+0.0048 to +0.0070) |

### kit level

| Group | Datapoints | Matches | `p_v4` | `p_exp` | `p_base` | `p_xgb83` |
|---|---|---|---|---|---|---|
| CT better kit | 29,673 | 650 | +0.0002 (-0.0001 to +0.0005) | +0.0003 (-0.0004 to +0.0010) | -0.0003 (-0.0007 to -0.0000) | +0.0044 (+0.0031 to +0.0057) |
| T better kit | 32,060 | 645 | +0.0004 (-0.0000 to +0.0008) | +0.0000 (-0.0006 to +0.0007) | +0.0002 (-0.0002 to +0.0006) | +0.0047 (+0.0034 to +0.0060) |
| same kit level | 57,440 | 653 | +0.0001 (-0.0001 to +0.0003) | +0.0008 (+0.0004 to +0.0012) | +0.0001 (-0.0002 to +0.0003) | +0.0043 (+0.0033 to +0.0054) |

### pre-plant: players alive

| Group | Datapoints | Matches | `p_v4` | `p_exp` | `p_base` | `p_xgb83` |
|---|---|---|---|---|---|---|
| CT +1 | 19,637 | 652 | +0.0002 (-0.0001 to +0.0005) | +0.0005 (-0.0001 to +0.0011) | +0.0000 (-0.0002 to +0.0003) | +0.0031 (+0.0018 to +0.0043) |
| CT +2 or more | 16,987 | 649 | +0.0001 (-0.0002 to +0.0005) | +0.0004 (-0.0002 to +0.0009) | -0.0001 (-0.0003 to +0.0002) | +0.0006 (-0.0002 to +0.0013) |
| T +1 | 15,040 | 651 | +0.0003 (-0.0001 to +0.0008) | +0.0007 (-0.0001 to +0.0015) | +0.0003 (-0.0001 to +0.0007) | +0.0055 (+0.0038 to +0.0074) |
| T +2 or more | 9,707 | 648 | -0.0001 (-0.0005 to +0.0003) | -0.0006 (-0.0014 to +0.0003) | -0.0001 (-0.0005 to +0.0003) | +0.0049 (+0.0032 to +0.0066) |
| even | 31,977 | 653 | +0.0001 (-0.0002 to +0.0004) | +0.0003 (-0.0002 to +0.0008) | +0.0000 (-0.0003 to +0.0003) | +0.0037 (+0.0027 to +0.0047) |

### pre-plant: kit level

| Group | Datapoints | Matches | `p_v4` | `p_exp` | `p_base` | `p_xgb83` |
|---|---|---|---|---|---|---|
| CT better kit | 24,058 | 649 | +0.0001 (-0.0003 to +0.0004) | +0.0003 (-0.0004 to +0.0011) | -0.0001 (-0.0005 to +0.0002) | +0.0034 (+0.0021 to +0.0047) |
| T better kit | 25,225 | 644 | +0.0005 (+0.0000 to +0.0009) | +0.0001 (-0.0005 to +0.0008) | +0.0002 (-0.0002 to +0.0006) | +0.0043 (+0.0029 to +0.0058) |
| same kit level | 44,065 | 653 | +0.0000 (-0.0002 to +0.0003) | +0.0005 (+0.0000 to +0.0009) | +0.0000 (-0.0002 to +0.0002) | +0.0029 (+0.0019 to +0.0039) |

### pre-plant, lot: kit level

| Group | Datapoints | Matches | `p_v4` | `p_exp` | `p_base` | `p_xgb83` |
|---|---|---|---|---|---|---|
| CT better kit | 16,679 | 648 | +0.0001 (-0.0003 to +0.0005) | +0.0006 (-0.0002 to +0.0014) | -0.0001 (-0.0004 to +0.0003) | +0.0030 (+0.0017 to +0.0044) |
| T better kit | 16,255 | 640 | +0.0005 (+0.0000 to +0.0010) | +0.0002 (-0.0006 to +0.0009) | +0.0001 (-0.0004 to +0.0007) | +0.0025 (+0.0010 to +0.0039) |
| same kit level | 25,978 | 653 | -0.0000 (-0.0003 to +0.0002) | +0.0003 (-0.0001 to +0.0008) | -0.0001 (-0.0004 to +0.0001) | +0.0016 (+0.0005 to +0.0027) |

## Notes

- The table is fixed (`evaluation/intern_gdt/table.csv`, built on upload days Oct 9 to 13, 2026).
- Groups were defined before the run (`analysis/group_gdt.py`). Read any single interval with care: many groups are shown at once.
- This run's positions models predict every second (872,038 rows over 12,645 rounds, all seconds present). The every-other-second subsample described in the protocol does not apply here.
