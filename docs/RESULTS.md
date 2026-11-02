# Results and evaluation status

As of 2026-11-02. Log loss in nats. "Paired" means the same rows scored by both models.

## Evaluation status

- The intern grader (GDT, `evaluation/intern_gdt/`) grades on upload days **Oct 14 to 25, 2026**. Its table was built on Oct 9 to 13.
- Our frozen test set has **no** Oct 14 to 25 matches. Its upload days run Sep 1 to Oct 13. About 2,300 of its 3,059 matches fall on Oct 9 to 13, the days the table was built from. Scoring it with the grader would favor the table, because the table was fit on those rows.
- The test-assigned 2026 matches added after the freeze (Oct 14 to 27, 655 of them) are not in the frozen cache. A clean Oct 14 to 25 test window needs the expanded cache (`cache_snapshot_2026_oct23_31.parquet`), once the pipeline finishes. Confirm that it includes those matches before scoring.
- `evaluation/build_gdt_rows.py` ran end to end on the frozen data and returned 0 rows, which is the expected result here.
- The extended cache from Oct 27 (`cache_snapshot_2026_ext.parquet`) has Oct 14 to 21 matches, but every one is in the manifest's **train** split. Those days cannot be a test window for our models.
- **Clean window in use:** the frozen test matches from Sep 1 to Oct 8 (743 matches, 13,170 rounds). The table was built from Oct 9 to 13 and neither the table nor our models saw these matches. This is a deviation from the intern's protocol (Oct 14 to 25), and the numbers below are on this window, not that one.
- Grader run on that window: `docs/gdt_results/sep01_oct08_summary.json`, `docs/gdt_results/sep01_oct08_by_situation.csv`. Per-group differences: `docs/gdt_results/group_gdt_sep01_oct08.csv`, built by `analysis/group_gdt.py`.

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
