# Round-quality evaluation: positions_validation_gdt

Scored population: the 1,168 validation matches of pin 773d5c12, kept after the intern's filters (in scope, max_rounds 24, seven maps, 5v5 round starts) and without the Oct 9 to 13, 2026 table-build days: 297 matches. Development read under pin 594c2815. Not the protocol's Oct 14 to 25 test days, not holdout B, not the frozen test. Seed-0 arms only.

Reference: `p_base`. Rows: 356,986 seconds, 297 matches, 5,275 rounds, 50,471 grader datapoints in 1,129 situations (10 with 500 or more). Seconds per round: 67.7. Seed 0, 1000 resamples of matches.

Positive differences mean the model beats the reference. Intervals resample matches.

## Headline: GDT against the table

| Model | GDT (95%) | GDT_sit (95%) | rGDT | Matches better than table | GDT minus reference (95%) | GDT_sit minus reference (95%) |
|---|---|---|---|---|---|---|
| `p_base` | 0.0204 (0.0170 to 0.0238) | 0.0117 (0.0076 to 0.0161) | 4.4% | 75.1% | reference | reference |
| `p_full_trainctr` | 0.0254 (0.0218 to 0.0291) | 0.0126 (0.0077 to 0.0175) | 5.5% | 79.5% | +0.0050 (+0.0036 to +0.0063) | +0.0009 (-0.0007 to +0.0025) |
| `p_full_oldctr` | 0.0254 (0.0218 to 0.0292) | 0.0125 (0.0076 to 0.0172) | 5.5% | 80.5% | +0.0050 (+0.0038 to +0.0062) | +0.0008 (-0.0008 to +0.0023) |

## Diagnostics (per second, all rows)

| Model | Log loss | AUC |
|---|---|---|
| `p_base` | 0.46474 | 0.85298 |
| `p_full_trainctr` | 0.46048 | 0.85552 |
| `p_full_oldctr` | 0.46046 | 0.85550 |

## By situation: difference from the reference (model better when positive)

### overall

| Group | Datapoints | Matches | `p_full_trainctr` | `p_full_oldctr` |
|---|---|---|---|---|
| all | 50,471 | 297 | +0.0050 (+0.0036 to +0.0063) | +0.0050 (+0.0038 to +0.0062) |

### phase and time left

| Group | Datapoints | Matches | `p_full_trainctr` | `p_full_oldctr` |
|---|---|---|---|---|
| post-plant little left | 1,381 | 275 | +0.0127 (+0.0062 to +0.0189) | +0.0121 (+0.0052 to +0.0190) |
| post-plant lot left | 5,794 | 293 | +0.0085 (+0.0047 to +0.0127) | +0.0087 (+0.0049 to +0.0126) |
| post-plant medium left | 3,758 | 292 | +0.0089 (+0.0031 to +0.0142) | +0.0092 (+0.0037 to +0.0146) |
| pre-plant little left | 1,783 | 265 | +0.0113 (+0.0024 to +0.0205) | +0.0112 (+0.0027 to +0.0202) |
| pre-plant lot left | 25,541 | 297 | +0.0029 (+0.0016 to +0.0042) | +0.0028 (+0.0016 to +0.0041) |
| pre-plant medium left | 12,214 | 295 | +0.0048 (+0.0026 to +0.0067) | +0.0048 (+0.0027 to +0.0069) |

### players alive

| Group | Datapoints | Matches | `p_full_trainctr` | `p_full_oldctr` |
|---|---|---|---|---|
| CT +1 | 9,648 | 297 | +0.0068 (+0.0044 to +0.0091) | +0.0068 (+0.0044 to +0.0091) |
| CT +2 or more | 8,091 | 292 | +0.0016 (-0.0004 to +0.0038) | +0.0017 (-0.0002 to +0.0038) |
| T +1 | 9,299 | 295 | +0.0067 (+0.0040 to +0.0092) | +0.0069 (+0.0042 to +0.0094) |
| T +2 or more | 7,236 | 291 | +0.0037 (+0.0013 to +0.0058) | +0.0041 (+0.0017 to +0.0063) |
| even | 16,197 | 297 | +0.0053 (+0.0033 to +0.0072) | +0.0050 (+0.0031 to +0.0069) |

### kit level

| Group | Datapoints | Matches | `p_full_trainctr` | `p_full_oldctr` |
|---|---|---|---|---|
| CT better kit | 13,484 | 288 | +0.0037 (+0.0014 to +0.0058) | +0.0041 (+0.0017 to +0.0062) |
| T better kit | 14,365 | 292 | +0.0045 (+0.0025 to +0.0067) | +0.0044 (+0.0025 to +0.0066) |
| same kit level | 22,622 | 297 | +0.0061 (+0.0041 to +0.0080) | +0.0060 (+0.0040 to +0.0078) |

### pre-plant: players alive

| Group | Datapoints | Matches | `p_full_trainctr` | `p_full_oldctr` |
|---|---|---|---|---|
| CT +1 | 7,962 | 295 | +0.0048 (+0.0025 to +0.0070) | +0.0048 (+0.0025 to +0.0069) |
| CT +2 or more | 6,820 | 291 | +0.0012 (-0.0004 to +0.0028) | +0.0012 (-0.0005 to +0.0028) |
| T +1 | 6,732 | 295 | +0.0060 (+0.0031 to +0.0088) | +0.0060 (+0.0034 to +0.0088) |
| T +2 or more | 4,325 | 289 | +0.0037 (+0.0007 to +0.0068) | +0.0043 (+0.0013 to +0.0072) |
| even | 13,699 | 297 | +0.0036 (+0.0019 to +0.0053) | +0.0034 (+0.0017 to +0.0051) |

### pre-plant: kit level

| Group | Datapoints | Matches | `p_full_trainctr` | `p_full_oldctr` |
|---|---|---|---|---|
| CT better kit | 10,790 | 285 | +0.0024 (+0.0004 to +0.0046) | +0.0028 (+0.0009 to +0.0049) |
| T better kit | 11,523 | 291 | +0.0035 (+0.0013 to +0.0056) | +0.0033 (+0.0012 to +0.0054) |
| same kit level | 17,225 | 297 | +0.0050 (+0.0031 to +0.0068) | +0.0049 (+0.0030 to +0.0065) |

### pre-plant, lot: kit level

| Group | Datapoints | Matches | `p_full_trainctr` | `p_full_oldctr` |
|---|---|---|---|---|
| CT better kit | 7,428 | 285 | +0.0014 (-0.0008 to +0.0036) | +0.0017 (-0.0005 to +0.0038) |
| T better kit | 7,694 | 287 | +0.0027 (+0.0003 to +0.0049) | +0.0026 (+0.0005 to +0.0049) |
| same kit level | 10,419 | 297 | +0.0041 (+0.0020 to +0.0061) | +0.0038 (+0.0016 to +0.0058) |

## Notes

- The table is fixed (`evaluation/intern_gdt/table.csv`, built on upload days Oct 9 to 13, 2026).
- Groups were defined before the run (`analysis/group_gdt.py`). Read any single interval with care: many groups are shown at once.
- This run's positions models predict every second the position table has. All 356,986 qualifying seconds have a prediction from every arm (100%), so the every-other-second subsample does not apply.
- The table-build days (Oct 9 to 13, 2026) are excluded as the README asks. That drops 871 of the 1,168 validation matches, so most of the validation set falls on the days the table was built on. The result is a 297-match read with wider intervals than the full validation set would give.
- GDT_sit counts only 10 situations with 500 or more datapoints, and both of its intervals span 0. The pinned bar uses its point estimate, which is +0.0009.
- Centre effect, full_trainctr minus full_oldctr, GDT: −0.0000 (−0.0003 to +0.0003); GDT_sit: +0.0002 (−0.0003 to +0.0007). Scored with `evaluation/intern_gdt/score_gdt.py`, reference p_full_oldctr; the same rows and seed.
