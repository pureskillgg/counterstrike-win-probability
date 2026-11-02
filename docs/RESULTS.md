# Results and evaluation status

As of 2026-11-02. Log loss in nats. "Paired" means the same rows scored by both models.

## Evaluation status

- The intern grader (GDT, `evaluation/intern_gdt/`) grades on upload days **Oct 14 to 25, 2026**. Its table was built on Oct 9 to 13.
- Our frozen test set has **no** Oct 14 to 25 matches. Its upload days run Sep 1 to Oct 13. About 2,300 of its 3,059 matches fall on Oct 9 to 13, the days the table was built from. Scoring it with the grader would favor the table, because the table was fit on those rows.
- Correction: the expanded cache (`cache_snapshot_2026_oct23_31.parquet`) does contain the test-assigned matches added after the freeze. 653 of them fall on Oct 14 to 25, and no model trained on them. That is the intern's protocol window, and it is graded below.
- `evaluation/build_gdt_rows.py` ran end to end on the frozen data and returned 0 rows, which is the expected result here.
- The earlier extended cache (`cache_snapshot_2026_ext.parquet`, Oct 27) has Oct 14 to 21 matches, all in the manifest's **train** split. It predates the test-assigned matches, so it cannot provide the window.
- **Clean window in use:** the frozen test matches from Sep 1 to Oct 8 (743 matches, 13,170 rounds). The table was built from Oct 9 to 13 and neither the table nor our models saw these matches. This is a deviation from the intern's protocol (Oct 14 to 25), and the numbers below are on this window, not that one.
- Grader run on that window: `docs/gdt_results/sep01_oct08_summary.json`, `docs/gdt_results/sep01_oct08_by_situation.csv`. Per-group differences: `docs/gdt_results/group_gdt_sep01_oct08.csv`, built by `analysis/group_gdt.py`.

## Intern protocol window (upload days Oct 14 to 25, 653 test-assigned matches)

Built by `evaluation/build_gdt_rows_cache.py` from the expanded cache, with every second of every 5v5 round. Reference is v6.

| Model | GDT (95%) | GDT_sit (95%) | GDT minus v6 (95%) |
|---|---|---|---|
| v6 (reference) | 0.0184 (0.0162 to 0.0204) | 0.0148 (0.0123 to 0.0174) | |
| v4 | 0.0186 (0.0164 to 0.0206) | 0.0150 (0.0125 to 0.0176) | +0.0002 (+0.0000 to +0.0004) |

The intern's reference for v6 on the full 12,566-match set is GDT 0.0180 (0.0175 to 0.0184). Our 0.0184 on 653 matches sits inside that, which supports the row build. Positions, graph, set, GRU and lag have no predictions on these matches yet, because their predictions exist only for the frozen matches. The expanded-cache model (`stream_oct23_31`) is the next candidate once its retrain finishes.

## Grader results on the clean window (Sep 1 to Oct 8, 743 matches)

Reference is v6, as the grader asks. Positive difference means the model beats v6. Intervals resample matches.

| Model | GDT (95%) | GDT_sit (95%) | GDT minus v6 (95%) | GDT_sit minus v6 (95%) |
|---|---|---|---|---|
| v6 (reference) | 0.0197 (0.0176 to 0.0218) | 0.0152 (0.0126 to 0.0176) | | |
| v4 (59 inputs) | 0.0197 (0.0175 to 0.0218) | 0.0150 (0.0125 to 0.0175) | −0.0001 (−0.0003 to +0.0002) | −0.0002 (−0.0005 to +0.0002) |
| xgb83 (positions added) | 0.0235 (0.0212 to 0.0257) | 0.0177 (0.0150 to 0.0203) | +0.0038 (+0.0029 to +0.0046) | +0.0026 (+0.0016 to +0.0036) |
| Graph model (gnn_hyb1) | 0.0237 (0.0214 to 0.0260) | 0.0174 (0.0143 to 0.0203) | +0.0040 (+0.0027 to +0.0053) | +0.0022 (+0.0004 to +0.0039) |
| Set model (gnn_set) | 0.0237 (0.0211 to 0.0261) | 0.0172 (0.0139 to 0.0203) | +0.0040 (+0.0025 to +0.0055) | +0.0020 (+0.0001 to +0.0042) |
| Causal GRU (full) | 0.0204 (0.0183 to 0.0227) | 0.0149 (0.0120 to 0.0174) | +0.0007 (−0.0003 to +0.0018) | −0.0003 (−0.0014 to +0.0009) |
| Lag features | 0.0206 (0.0184 to 0.0227) | 0.0155 (0.0130 to 0.0180) | +0.0009 (+0.0006 to +0.0012) | +0.0004 (−0.0001 to +0.0008) |

Our v6 reads 0.0197 here against the intern's 0.0180 on the Oct 14 to 25 days. That is close, so the row build looks sound, but the two windows are different.

## Where the gains land (`docs/gdt_results/group_gdt_sep01_oct08.csv`)

Difference in log loss against v6 on the grader's datapoints, by group. Positive is better.

| Group | Datapoints | xgb83 | Graph | Lag |
|---|---|---|---|---|
| Pre-plant, lot of time left | 59,339 | +0.0018 | +0.0022 | +0.0003 |
| Pre-plant, medium time left | 27,515 | +0.0025 | +0.0037 | +0.0010 |
| Pre-plant, little time left | 4,470 | +0.0123 | +0.0146 | +0.0005 |
| Post-plant, lot of time left | 12,561 | +0.0067 | +0.0066 | +0.0014 |
| Post-plant, medium time left | 8,201 | +0.0103 | +0.0084 | +0.0022 |
| Post-plant, little time left | 3,273 | +0.0110 | +0.0033 (CI includes 0) | +0.0063 |
| Even alive counts | 37,902 | +0.0046 | +0.0043 | +0.0007 |
| T one player up | 20,576 | +0.0055 | +0.0065 | +0.0011 |

The early-round group holds half of the grader's datapoints and gains about 0.002. The big gains sit in late-round groups, which have far fewer datapoints. Many groups are tested at once, so read any single interval with care.

## Recorded results (frozen test, 3,059 matches)

| Change | Paired change in log loss | 95% interval | Source |
|---|---|---|---|
| 24 position, view and blindness inputs (xgb83 vs xgb59) | −0.00336 | −0.00365 to −0.00306 | `data/combined/exp_gnn.json` (every other second; its xgb59 scores 0.45777, not v4's 0.45641) |
| Graph model (gnn_hyb1) vs xgb59 | −0.00392 | see file | `data/combined/exp_gnn.json` |
| Graph model vs xgb83 | about −0.0006 | not computed | by subtraction: 0.45385 vs 0.45441 |
| Set model (gnn_set, player tokens, no hand-built inputs) | not paired; absolute 0.45441 | 0.45130 to 0.45720 | `data/gnn/result_gnn_set.json` |
| Lag features (v4 base 0.45643, lag 0.45612) | −0.00032 | −0.00043 to −0.00020 | `lag_results.json` |
| Money on hand, combined | −0.00023 | −0.00039 to −0.00007 | money probe (`experiments/money/`) |
| Money or equipment alone | about −0.0001 each | not recorded here | E1 (`experiments/e1_and_diagnostics/`) |
| Causal GRU, full sequence | −0.00014 | −0.00053 to +0.00026 | `experiments/gru/` |
| Causal GRU, 20-second window | +0.00066 (worse) | +0.00020 to +0.00113 | `experiments/gru/` |
| Utility events (smoke, flash, HE, decoy; enemy-blind) | +0.00031 (worse) | interval includes 0 | `EXPERIMENTS_2026-10-19.md` (line 346) |
| Damage and shots events | +0.00016 (worse) | not recorded here | `exp_event_features.py` |

Outside the frozen test: a per-slot GRU with loadout (18 values per player) beat its own 57-input XGBoost arm by −0.00219 on the 2026 validation split (three seeds, each passing the pre-registered rule). That baseline is not v4's, so the number does not transfer directly. Source: `data/diagnostics/g1_gru_vs_xgb_validation.json`.

## Expanded training data (retrain on the frozen test, 3,059 matches)

The in-memory retrain ran out of memory on the expanded cache, so the reruns use `experiments/pipeline/retrain_stream.py`. Test rows are the frozen matches only.

| Run | Train rows | Test log loss | Test AUC | Best iteration |
|---|---|---|---|---|
| Original control (in-memory, seed 0) | 13,758,901 | 0.456435 | 0.857810 | 1059 |
| Streamed control (seed 0) | 13,758,901 | 0.456516 | 0.857758 | 985 |
| Expanded (seed 0) | 28,813,624 | 0.456026 | 0.858098 | 1038 |
| Streamed control (seed 1) | 13,758,901 | 0.456613 | 0.857701 | 733 |
| Expanded (seed 1) | 28,813,624 | 0.455989 | 0.858133 | 1031 |
| Streamed control (seed 2) | 13,758,901 | 0.456576 | 0.857708 | 847 |
| Expanded (seed 2) | 28,813,624 | 0.455982 | 0.858120 | 1085 |

The streamed and original controls differ by 0.00008 with identical settings, so that is the scale of run-to-run noise from row order alone. The paired gain of the expanded run over the streamed control is +0.00049, with a match-cluster interval of +0.00031 to +0.00065. That interval covers test-match sampling only, not training seed.

Across three seeds, the expanded run beats the streamed control every time: by 0.00049, 0.00062 and 0.00059 in log loss (mean 0.00057). The control's spread across seeds is about 0.00005, so the gain is well outside training noise. The table below is for seed 0. Seed-averaged intervals by situation are not computed yet.

Paired by situation (`docs/retrain_results/paired_stream_control_vs_expanded.csv`, positive is better for the expanded run):

| Group | Datapoints | Gain (95%) |
|---|---|---|
| Pre-plant, lot of time left | 2,148,905 | +0.00049 (+0.00030 to +0.00067) |
| Pre-plant, medium time left | 960,650 | +0.00044 (+0.00016 to +0.00071) |
| Pre-plant, little time left | 182,655 | +0.00034 (−0.00034 to +0.00100) |
| Post-plant, lot of time left | 341,566 | +0.00051 (+0.00009 to +0.00096) |
| Post-plant, medium time left | 221,105 | +0.00080 (+0.00017 to +0.00148) |
| Post-plant, little time left | 101,493 | +0.00044 (−0.00028 to +0.00126) |

On the intern window (653 test-assigned matches), the expanded model against v6: GDT +0.0004 (+0.0001 to +0.0008), GDT_sit +0.0003 (−0.0001 to +0.0007). Source: `docs/gdt_results/intern_oct14_25_exp_summary.json`.

## Corrections

- Earlier I said utility features had not been tested. They were. The utility experiment gave +0.00031 with an interval that includes zero.
- Earlier I listed a post-plant timer and defuse state as the largest unmeasured source. `seconds_since_plant`, `st_kit_n_*` and `st_defusing_*` are already among the 59 inputs (`data/gnn/meta.json`, `G59_names`). The upside from that source is smaller than I estimated.

## In-round information ceiling (workflow `wf_76bddfef-16d`)

- The steelman put 0.42 at the optimistic edge, with 0.43 to 0.44 as the realistic range once player and team identity are excluded. It estimated 0.41 needs a total cut of 0.046 and 0.40 needs 0.056. These are estimates, not measurements.
- The tome inventory found that the in-round sources we have are already tested (utility, blindness, damage and shots). Each moved log loss by about 0.0003 or less. Per-tick channels other than the ones used are stored only on the E: drive.

## Open items

1. Score the Oct 14 to 25 test-assigned matches with the grader once the expanded cache exists. Include v4 and v6, and report GDT and GDT_sit with the difference and its interval.
2. Ceiling experiment: add the in-round sources together on the frozen rows, and report the match-clustered change with its interval.
3. Learning curve over training matches (25, 50, 100 percent), to separate a data limit from an information limit.
4. Log loss at 0 to 9 seconds from a round-start-only model, to test whether the opening seconds are at their floor.
