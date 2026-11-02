# Round-quality evaluation

Status: adopted 2026-11-02. This replaces the pooled per-second log loss as the headline. It follows the intern grader in `evaluation/intern_gdt/`.

## Why the change

Pooled per-second log loss averages every second of every round. Most experiments moved it by 0.0001 to 0.0005, and the average hides where a change helps. The intern's grader instead scores each model against a fixed table of situations, one datapoint per round per situation. Each situation then counts once per round, so a change in a particular situation shows up at its own size.

## Measures

- **Situation.** Alive players on each side, bomb planted or not, time left (thirds of the round before the plant and of the 40 s bomb after it), and each side's equipment per alive player (low, mid, high).
- **Table.** `evaluation/intern_gdt/table.csv`: the chance CT wins in each of 1,530 situations. Fixed. It was built on upload days Oct 9 to 13, 2026.
- **GDT.** Table log loss minus model log loss, averaged over the grader's datapoints. Positive means the model beats the table. Reported with a 95% interval from resampling matches.
- **GDT_sit.** The same, but each situation with 500 or more datapoints counts once. Reported beside GDT, never instead of it.
- **Datapoints.** One per round per situation, drawn at random with a fixed seed (0). Every model in a comparison is scored on the same datapoints.
- **Group differences.** For a group of situations, the mean of (reference log loss minus model log loss) over that group's datapoints, with a match-cluster interval.
- **Diagnostics.** Per-second log loss and AUC over all rows. They do not decide anything, but they are kept for comparison with earlier results.

## Windows

| Window | Matches | Use | Why |
|---|---|---|---|
| Intern protocol: upload days Oct 14 to 25, 2026, test-assigned, not in the frozen benchmark | 653 (12,645 rounds) | Claims about a model | Clean for the table and for our models. Matches the intern's test days. |
| Frozen development window: upload days Sep 1 to Oct 8, 2026 | 743 (13,170 rounds) | Comparing experiments that have saved predictions | Neither the table nor our models saw these matches. Predictions exist for every experiment, but only on every other second. |
| Table build days Oct 9 to 13 | — | Never for scoring | The table was fit on them. |
| Any match a model trained on | — | Never for scoring | The intern's rule. |

The protocol's reference is v6 (`p_v6`), and the intern's own v6 figure (GDT 0.0180 on 12,566 matches) is the check on our rows. On the intern window we get 0.0184 (0.0162 to 0.0204), which is inside that range.

## Rules

- A new model is better than v6 when its GDT difference from v6 is positive with the whole interval above 0. Report GDT_sit with the same interval.
- No score lead, or anything derived from it, as an input.
- The table is fixed. Do not refit or retune it to suit a model.
- The same rows and the same seed for every model compared.
- Name the groups before a run. Report every group, not the ones that moved.
- Report the seeds too. The intervals cover test-match sampling only. A training-seed spread is measured separately (see `docs/RESULTS.md`).

## Groups

Defined in `analysis/group_gdt.py` before any run:

- Phase and time left: pre-plant and post-plant, each with lot, medium and little time left.
- Players alive: even, CT plus 1, T plus 1, CT plus 2 or more, T plus 2 or more.
- Kit level: CT better, same, T better.
- Pre-plant by players alive; pre-plant by kit level; pre-plant lot by kit level.

## How to run

```
python evaluation/evaluate.py --rows <rows.parquet> --reference p_v6 --models p_v4,p_new --tag <name> --window "<which matches>"
```

The rows come from `evaluation/build_gdt_rows_cache.py` (the cache, for the intern window) or `evaluation/build_gdt_rows.py` (the frozen benchmark, with saved predictions). The report goes to `docs/eval_reports/<name>.md`, with the grader's `summary.json`, `by_situation.csv` and `groups.csv` beside it.

## Open gaps

1. **The intern window has predictions only for v4, v6 and the expanded retrain.** Positions, graph, set, GRU and lag need predictions on those 653 matches before they can be compared on the intern protocol. Positions need the position features for those days. The graph and set models need player tensors built for them.
2. **Coverage.** The positions, graph and set predictions cover every other second. The intern's README asks for every second, so their GDT is on a subsample. Rerunning them on every second would meet the protocol.
3. **Training seeds.** Two extra seeds for the control and expanded runs are running. Until they land, a gain of about 0.0005 in log loss is not separated from seed noise.
