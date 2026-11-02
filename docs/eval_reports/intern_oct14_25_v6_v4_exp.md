# Round-quality evaluation: intern_oct14_25_v6_v4_exp

Intern protocol window: upload days Oct 14 to 25, 2026. 653 test-assigned matches (manifest test, not in the frozen benchmark), seven maps, in scope, max rounds 24. Built by evaluation/build_gdt_rows_cache.py.

Reference: `p_v6`. Rows: 872,038 seconds, 653 matches, 12,645 rounds, 119,173 grader datapoints in 1,228 situations (44 with 500 or more). Seconds per round: 69.0. Seed 0, 1000 resamples of matches.

Positive differences mean the model beats the reference. Intervals resample matches.

## Headline: GDT against the table

| Model | GDT (95%) | GDT_sit (95%) | rGDT | Matches better than table | GDT minus reference (95%) | GDT_sit minus reference (95%) |
|---|---|---|---|---|---|---|
| `p_v6` | 0.0184 (0.0162 to 0.0204) | 0.0148 (0.0123 to 0.0174) | 4.1% | 74.7% | reference | reference |
| `p_v4` | 0.0186 (0.0164 to 0.0206) | 0.0150 (0.0125 to 0.0176) | 4.2% | 75.5% | +0.0002 (+0.0000 to +0.0004) | +0.0001 (-0.0001 to +0.0004) |
| `p_exp` | 0.0188 (0.0167 to 0.0209) | 0.0151 (0.0125 to 0.0177) | 4.2% | 76.1% | +0.0004 (+0.0001 to +0.0008) | +0.0003 (-0.0001 to +0.0007) |

## Diagnostics (per second, all rows)

| Model | Log loss | AUC |
|---|---|---|
| `p_v6` | 0.45200 | 0.86065 |
| `p_v4` | 0.45187 | 0.86068 |
| `p_exp` | 0.45154 | 0.86103 |

## By situation: difference from the reference (model better when positive)

### overall

| Group | Datapoints | Matches | `p_v4` | `p_exp` |
|---|---|---|---|---|
| all | 119,173 | 653 | +0.0002 (+0.0000 to +0.0004) | +0.0004 (+0.0001 to +0.0008) |

### phase and time left

| Group | Datapoints | Matches | `p_v4` | `p_exp` |
|---|---|---|---|---|
| post-plant little left | 3,632 | 608 | +0.0021 (+0.0002 to +0.0040) | +0.0010 (-0.0005 to +0.0026) |
| post-plant lot left | 13,443 | 650 | +0.0001 (-0.0003 to +0.0006) | +0.0011 (+0.0003 to +0.0018) |
| post-plant medium left | 8,750 | 649 | -0.0000 (-0.0007 to +0.0006) | +0.0005 (-0.0006 to +0.0015) |
| pre-plant little left | 5,355 | 609 | -0.0003 (-0.0009 to +0.0004) | +0.0003 (-0.0009 to +0.0015) |
| pre-plant lot left | 58,912 | 653 | +0.0002 (-0.0001 to +0.0004) | +0.0003 (-0.0001 to +0.0008) |
| pre-plant medium left | 29,081 | 653 | +0.0003 (-0.0000 to +0.0005) | +0.0003 (-0.0003 to +0.0008) |

### players alive

| Group | Datapoints | Matches | `p_v4` | `p_exp` |
|---|---|---|---|---|
| CT +1 | 23,316 | 652 | +0.0003 (-0.0001 to +0.0007) | +0.0006 (-0.0000 to +0.0012) |
| CT +2 or more | 19,808 | 649 | +0.0003 (-0.0000 to +0.0007) | +0.0005 (-0.0001 to +0.0011) |
| T +1 | 21,106 | 652 | +0.0002 (-0.0002 to +0.0006) | +0.0007 (-0.0000 to +0.0013) |
| T +2 or more | 17,250 | 648 | +0.0001 (-0.0002 to +0.0004) | -0.0004 (-0.0009 to +0.0002) |
| even | 37,693 | 653 | +0.0001 (-0.0002 to +0.0004) | +0.0006 (+0.0000 to +0.0011) |

### kit level

| Group | Datapoints | Matches | `p_v4` | `p_exp` |
|---|---|---|---|---|
| CT better kit | 29,673 | 650 | +0.0002 (-0.0001 to +0.0005) | +0.0003 (-0.0004 to +0.0010) |
| T better kit | 32,060 | 645 | +0.0004 (-0.0000 to +0.0008) | +0.0000 (-0.0006 to +0.0007) |
| same kit level | 57,440 | 653 | +0.0001 (-0.0001 to +0.0003) | +0.0008 (+0.0004 to +0.0012) |

### pre-plant: players alive

| Group | Datapoints | Matches | `p_v4` | `p_exp` |
|---|---|---|---|---|
| CT +1 | 19,637 | 652 | +0.0002 (-0.0001 to +0.0005) | +0.0005 (-0.0001 to +0.0011) |
| CT +2 or more | 16,987 | 649 | +0.0001 (-0.0002 to +0.0005) | +0.0004 (-0.0002 to +0.0009) |
| T +1 | 15,040 | 651 | +0.0003 (-0.0001 to +0.0008) | +0.0007 (-0.0001 to +0.0015) |
| T +2 or more | 9,707 | 648 | -0.0001 (-0.0005 to +0.0003) | -0.0006 (-0.0014 to +0.0003) |
| even | 31,977 | 653 | +0.0001 (-0.0002 to +0.0004) | +0.0003 (-0.0002 to +0.0008) |

### pre-plant: kit level

| Group | Datapoints | Matches | `p_v4` | `p_exp` |
|---|---|---|---|---|
| CT better kit | 24,058 | 649 | +0.0001 (-0.0003 to +0.0004) | +0.0003 (-0.0004 to +0.0011) |
| T better kit | 25,225 | 644 | +0.0005 (+0.0000 to +0.0009) | +0.0001 (-0.0005 to +0.0008) |
| same kit level | 44,065 | 653 | +0.0000 (-0.0002 to +0.0003) | +0.0005 (+0.0000 to +0.0009) |

### pre-plant, lot: kit level

| Group | Datapoints | Matches | `p_v4` | `p_exp` |
|---|---|---|---|---|
| CT better kit | 16,679 | 648 | +0.0001 (-0.0003 to +0.0005) | +0.0006 (-0.0002 to +0.0014) |
| T better kit | 16,255 | 640 | +0.0005 (+0.0000 to +0.0010) | +0.0002 (-0.0006 to +0.0009) |
| same kit level | 25,978 | 653 | -0.0000 (-0.0003 to +0.0002) | +0.0003 (-0.0001 to +0.0008) |

## Notes

- The table is fixed (`evaluation/intern_gdt/table.csv`, built on upload days Oct 9 to 13, 2026).
- Groups were defined before the run (`analysis/group_gdt.py`). Read any single interval with care: many groups are shown at once.
- Models with predictions on only every other second (positions, graph, set) are scored on that subsample. The intern's protocol asks for every second.
