# CHECK A: learning curve, 2026 kept set (seed 0)

**Verdict:** Yes, test log loss still improves from 50% to 100% of train matches (-0.00142, 95% CI -0.00230 to -0.00057), but the gain is small and shrinking: the step gains are -0.00352 (10% to 25%), -0.00204 (25% to 50%), -0.00142 (50% to 100%). More matches would still help a little, and the curve is flattening.

Label: VERIFIED. Every number below was computed by `scripts/model/sufficiency/learning_curve.py` (run output and JSON: `notes/model/sufficiency/learning_curve.json`). Rendered by `scripts/model/sufficiency/learning_curve_md.py` from that JSON.

## Data counts

- Kept set (bake-off filters, rebuilt and checked against `logs/bakeoff.log`): 1521 matches, 28683 rounds, 234265 states. Channel-kept matches 1709, duplicate matches removed 47.

| split | matches | rounds | states | CT-win share (states) |
|---|---:|---:|---:|---:|
| train pool (90% of the 80%) | 1095 | 20470 | 167379 | 0.5214 |
| validation (10% of the 80%) | 122 | 2364 | 19170 | 0.5190 |
| test (20%) | 304 | 5849 | 47716 | 0.5084 |

Splits are the bake-off primary split (`bakeoff.make_splits`, seed 0). Its train, validation and test counts match `logs/bakeoff.log` exactly.

## Learning curve (XGBoost, test set fixed at 304 matches)

| train fraction | train matches | train rounds | train states | best iter | test log loss | test Brier | test AUC | test ECE | device |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10% | 110 | 2071 | 16975 | 150 | 0.44357 | 0.14861 | 0.8666 | 0.0131 | cpu (cuda requested; XGBoost found no GPU and used CPU) |
| 25% | 274 | 5211 | 42588 | 224 | 0.44004 | 0.14759 | 0.8687 | 0.0134 | cpu (cuda requested; XGBoost found no GPU and used CPU) |
| 50% | 548 | 10180 | 83316 | 277 | 0.43800 | 0.14693 | 0.8695 | 0.0097 | cpu (cuda requested; XGBoost found no GPU and used CPU) |
| 100% | 1095 | 20470 | 167379 | 261 | 0.43659 | 0.14635 | 0.8704 | 0.0088 | cpu (cuda requested; XGBoost found no GPU and used CPU) |

Metrics are on states, from `bakeoff.metrics()`. Fractions are of the train pool (see notes).

## Change in test log loss between fractions (paired cluster bootstrap over test matches)

| change | test LL change | 95% CI | share of resamples better |
|---|---:|---|---:|
| 10% to 25% | -0.00352 | [-0.00544, -0.00143] | 1.0000 |
| 25% to 50% | -0.00204 | [-0.00299, -0.00104] | 1.0000 |
| 50% to 100% | -0.00142 | [-0.00230, -0.00057] | 0.9985 |

## Cross-check against the bake-off log

- Bake-off log, XGBoost raw, primary test: log loss 0.43660, Brier 0.14637.
- Here, 100% row: log loss 0.43659, Brier 0.14635, best iteration 261 (the log also reports 261).
- Match at 5 decimal places: False. The gap is 0.000014 in log loss. The likely cause is the script's `n_jobs=4` against the bake-off's default threading. This was not tested, so treat it as a cause for investigation, not a confirmed one. The split and tree count are identical.

## Notes and deviations

- Fractions 10, 25, 50, 100% are of the train pool, not of the full 80%: validation is 10% of the 80% (the bake-off's early-stopping slice), so the 100% row trains on 1095 of 1521 kept matches (72%). A literal 100% of the 80% would leave no early-stopping slice. Fractions are nested by a seeded shuffle.
- GPU: the request was for `device="cuda"`. The pixi `xgboost 3.4.2` build reports `USE_CUDA False`, so every fit ran on CPU. The GPU was not used.
- Only XGBoost was fitted. No hyperparameter search. Settings as in `bakeoff.py`: hist, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, early stopping 100 on validation, bake-off monotone constraints, features `bk.FEATS`.
- 50% to 100% change: point estimate -0.00142; the 100% model's log loss is 0.43659 and the 50% model's is 0.43800.
- Test log loss differences of about 0.001 are small against the test set's 304 matches. The CI is the guide, not the point estimate alone.

