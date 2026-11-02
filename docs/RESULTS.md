# Results and evaluation status

As of 2026-11-02. Log loss in nats. "Paired" means the same rows scored by both models.

## Evaluation status

- The intern grader (GDT, `evaluation/intern_gdt/`) grades on upload days **Oct 14 to 25, 2026**. Its table was built on Oct 9 to 13.
- Our frozen test set has **no** Oct 14 to 25 matches. Its upload days run Sep 1 to Oct 13. About 2,300 of its 3,059 matches fall on Oct 9 to 13, the days the table was built from. Scoring it with the grader would favor the table, because the table was fit on those rows.
- The test-assigned 2026 matches added after the freeze (Oct 14 to 27, 655 of them) are not in the frozen cache. A clean Oct 14 to 25 test window needs the expanded cache (`cache_snapshot_2026_oct23_31.parquet`), once the pipeline finishes. Confirm that it includes those matches before scoring.
- `evaluation/build_gdt_rows.py` ran end to end on the frozen data and returned 0 rows, which is the expected result here.

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
