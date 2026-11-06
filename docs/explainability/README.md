# Explainability for the 83-input win-probability model

This folder explains the 83-input XGBoost model that gives the probability that CT wins a CS2 round, second by second.
It answers three questions:

1. **Which inputs matter, and how much?** Global view, by bundle of inputs.
2. **Where did one prediction come from?** Local view: what each input contributed to a single second.
3. **How do two predictions differ, and which inputs work together?** Contrasts and interactions.

Explanation and importance are different things and are reported separately:
- *Explanation* (TreeSHAP) says where each prediction came from. Each input's contribution adds up exactly to the prediction.
- *Importance* (permutation) is black-box. It says how much the model's log loss rises when one input is shuffled.

## Where to look

| You want | Go to |
|---|---|
| The short write-up with figures | `XGB_explainability_report.docx` (9 pages) |
| Global results by bundle | `tables/bundle_importance_and_effects.csv`, `figures/bundle_effects.png` |
| Each input: effect, slope, and curve | `tables/feature_slopes_all83.csv`, `tables/feature_curves_all83.csv`, `figures/curves_*.png` |
| One prediction, a contrast, interactions | `local/` (below) |
| The 57-input submission model on the same seconds | `xgb57/tables/` |
| How to rerun everything | "Reproduce" below, and `../../handover/explainability/README.md` |

## One prediction: four example seconds

Four seconds from the 100,000-second validation sample were chosen by fixed rules (recorded in `local/local_explain_meta.json`).

| Example | Match (date), round, second | Label | P(CT wins) | Why it was chosen |
|---|---|---|---|---|
| `largest_loss` | 706c83dd… (Oct 9), round 2, second 52 | CT wins | 0.1% | Largest log loss in the sample. A 1-versus-5 clutch: the model gave CT almost no chance and CT won. |
| `confident_ct_win` | f68a407f… (Oct 8), round 23, second 100 | CT wins | 99.999% | The most confident correct call for CT. |
| `confident_t_win` | 71e89d12… (Oct 13), round 2, second 71 | T wins | 0.00002% | The most confident correct call for T. |
| `close_call` | 698d7562… (Oct 12), round 18, second 49 | T wins | 50.0% | The second closest to a coin toss. |

Full match keys are in `local/examples.csv`.

**Reading a waterfall** (`local/waterfall_<example>.csv`). The base value is the model's average logit, which is P = 51.3%. Each input adds a contribution on the logit scale, largest first. The base plus the contributions equals the prediction's logit exactly (checked to 1e-3). For the `largest_loss` second the biggest pushes against CT were:
- team HP difference of −270 (CT has 100 HP and T has 370), about the 1st percentile of the sample: −1.65 logit;
- CT alive minus T alive of −4 (1 against 5), about the 1st percentile: −1.36 logit;
- armored-player difference of −5: −0.95 logit;
- equipment difference of −$13,850: −0.51 logit.

**Calibration check for that state.** The state is 1 CT alive against 5 T. On the full validation set (1,515,228 seconds), 1,033 rounds reach it. Summing each round's mean predicted P(CT wins) gives 1.48 expected CT-win rounds. One occurred, and it is the largest miss itself: all of the CT-win seconds in the bomb-planted version of the state come from that one round. With the bomb planted, 580 rounds give 0.59 expected against 1 observed. With no bomb, 724 rounds give 1.30 expected against 0 observed. So the miss is one rare round, and this data does not show a calibration problem in this state. The earlier "3 wins in 426 seconds" was the same round, counted per second. Seconds within a round are not independent, so the per-second counts overstate the evidence.

**Overall calibration.** Deciles of predicted probability over all 1,515,228 validation seconds match the observed CT-win rate to within 1 percentage point in every decile (`tables/calibration_deciles_shipped.csv`). The intervals in that table treat seconds as independent, so they are too narrow.

## Two predictions: contrasts

`local/contrast_confident_ct_vs_t.csv` compares the two confident calls (P = 99.999% against P = 0.00002%). The logit gap is 27.1, and the per-input differences add up to it exactly. The largest contributors to the gap:

| Input | CT-call value | T-call value | Difference in logit |
|---|---|---|---|
| defuse slack (seconds) | 24.1 | −20.4 | +4.59 |
| team HP difference | +393 | −326 | +2.95 |
| CT alive minus T alive | +4 | −4 | +2.72 |
| CT distance to nearest bomb spot | 6.5 | 2,431 | +2.05 |
| armored-player difference | +4 | −5 | +1.94 |
| seconds since plant | 10.2 | 40.7 | +1.85 |

`local/contrast_largest_loss_vs_neighbour.csv` compares the `largest_loss` second with its nearest neighbour on the top 10 inputs, among seconds with the same state. The neighbour's prediction is almost the same (0.11% against 0.11%), so this contrast shows that the model treats those seconds alike. It is more useful as the calibration check above than as an explanation.

## Interactions

`local/interactions_top_pairs.csv` lists the strongest input pairs by mean absolute SHAP interaction value, computed on 2,000 of the sample seconds. The row sums match the SHAP values to 9e-6.

- The largest pair is seconds since plant with defuse slack (0.039 logit). Both are about the bomb timeline.
- Next are team HP difference with armored-player difference (0.034), and alive difference with armored-player difference (0.033).

These are small compared with the main effects. The inputs in these pairs contribute 0.06 to 0.48 logit each on average, and the interaction values are 0.03 to 0.04. The file's convention is xgboost's off-diagonal entry. The standard total pair effect is twice that value.

## Global results

Average absolute effect on P(CT wins), in percentage points per second, from TreeSHAP on 100,000 validation seconds:

| Bundle | Inputs | Average effect (pp) |
|---|---|---|
| Loadout difference, CT minus T | 6 | 13.29 |
| Round state | 6 | 3.11 |
| Bomb distance and defuse slack | 4 | 2.56 |
| Site and spread geometry | 9 | 1.83 |
| T health and armor | 6 | 1.46 |
| Map | 7 | 1.10 |

Position (24 inputs) totals 3.50 pp. Visibility (4 inputs) is 0.63 pp and blindness (4 inputs) is 0.03 pp. The retrain ablation shows the same ordering: adding all position groups to the 59-input base lowers validation log loss by 0.00344. The report has the per-input curves, the permutation comparison, and the flagged inputs.

The 57-input submission model (no position inputs) gives round state 4.72 pp (against 3.11 pp), and its bomb-timing inputs take over what defuse slack and bomb distance carried. Validation log loss is 0.453557 for 57 inputs, against 0.450210 for 83.

## Model and data

- **Model.** XGBoost, binary logistic. 83 inputs: 59 base, plus GEO (16), VIEW (4) and BLIND (4). Depth 4, eta 0.03, min_child_weight 20, subsample 0.8, max_bin 256, seed 0. Early stopping after 100 rounds on a 10% split, with a 3,000-round cap. Best iteration 2,695 (the sprint's was 2,555). Monotone constraints on the alive counts.
- **Training pool.** `cache_snapshot_2026.parquet`: 14,804 matches dated 2026-08-02 to 2026-10-13. The validation sample has 1,168 matches and 1,515,228 seconds. Checked on Nov 5, 2026: the pool has no match on or after Oct 26, and no match in common with the Oct 26–31 holdout keys.
- **Fit.** The CPU refit gives validation log loss 0.450210. The sprint's GPU run gave 0.450228. Per-row predictions differ from the sprint's by up to 0.25 (mean 0.004), so the refit is not identical to the GPU run.

## Drop-group ablations (retrained, scored on GDT)

Each set below is the full 83-input model with one group removed, or one group alone. Same matrix, split and fit settings as the sprint's full cell, three seeds each, refit in one environment (xgboost 3.2.0 on the WSL RTX 3080). GDT is the team's metric (`evaluation/intern_gdt/score_gdt.py`, the 295-match population from Sept 1 to Oct 8, 5-against-5 starts). Paired differences are against the full model with the same seed, on the same datapoints. Negative means the set is worse.

| Set (inputs) | GDT, mean of 3 seeds | Paired GDT vs full (range over seeds) | Log loss, all validation seconds | Paired log loss vs full |
|---|---|---|---|---|
| full (83) | 0.0254 | — | 0.450270 | — |
| minus alive counts (80) | 0.0256 | +0.0002 (+0.0000 to +0.0003) | 0.450536 | +0.00027 |
| minus armour sums (81) | 0.0255 | +0.0001 (−0.0000 to +0.0004) | 0.450259 | −0.00001 |
| minus blind (79) | 0.0254 | +0.0000 (−0.0001 to +0.0002) | 0.450255 | −0.00002 |
| minus round state (77) | 0.0253 | −0.0001 (−0.0002 to +0.0001) | 0.451096 | +0.00083 |
| minus view (79) | 0.0249 | −0.0005 (−0.0006 to −0.0004) | 0.450552 | +0.00028 |
| minus bomb geometry (79) | 0.0249 | −0.0005 (−0.0007 to −0.0004) | 0.450614 | +0.00034 |
| loadout only (46) | 0.0098 | −0.0156 (−0.0158 to −0.0154) | 0.463166 | +0.01290 |
| round state only (6) | −0.0164 | −0.0418 (−0.0418 to −0.0416) | 0.497067 | +0.04680 |

What the table says:
- **Alive counts:** removing them does not reduce GDT. The loadout inputs carry the alive signal.
- **Round state:** removing it from the full model costs nothing on GDT. Alone, round state scores below the lookup table.
- **Loadout only** beats the table (GDT 0.0098), but the position inputs add about 0.016 GDT on top of it.
- **Armour sums and blind inputs** have no measurable effect on GDT. These are the inputs the team's notes asked to drop (armour sums) or rebuild (blind).
- **View and bomb geometry** each give a small, consistent gain: about 0.0005 GDT. For each, the paired interval excludes zero in two seeds and touches it in the third.

The full model's scorer numbers match the team's documented 83-input figures: GDT 0.0254 against 0.0254 for the 297-match population in `docs/final/TRAINING_METRIC.md`. The population here is 295 matches. The full-model log loss matches the sprint's saved values on each seed to within 0.00005 (`tables/drop_ablation_reproduction.csv`).

Tables: `tables/drop_ablation_83_gdt.csv`, `tables/drop_ablation_83.csv`, `tables/drop_ablation_83_per_seed.csv`, and the three scorer summaries `tables/gdt_ablation_summary_seed{0,1,2}.json`. Built by `scripts/explain/xgb83_drop_ablation.py`, `scripts/explain/xgb83_gdt.py` and `scripts/explain/xgb83_ablation_report.py`.

## Caveats

- **Oct 9 and Oct 10.** The handoff of Nov 4 marks these days as unreconciled and gives 2,598 matches for them. The training cache holds 4,865 matches dated Oct 9–10 (2,214 and 2,651). The two counts disagree. Reconcile them before relying on those days. The `largest_loss` example is from Oct 9.
- **Missing and sentinel values.** `defuse_slack` is missing when no bomb is planted, and `seconds_since_plant` is −1. Both inflate permutation importance.
- **Correlated inputs.** TreeSHAP splits credit between correlated inputs. The alive-count inputs get little direct credit, and the retrains below show why: dropping them costs nothing on GDT. The loadout inputs carry the signal.
- **Not causal.** These are explanations of the model's behavior. They are not causal effects.
- **Clustered outcomes.** Seconds in a round share one outcome, so per-second statistics overstate how much evidence there is. The 1-versus-5 check is therefore done per round.
- **Small samples.** The interaction values use 2,000 seconds.

## Reproduce

Use the pixi environment that has xgboost, polars, numpy and matplotlib:
`G:\pureskill\csds-cs2\.pixi\envs\default\python.exe`

The data is not in git. The file list and locations are in `../../handover/explainability/README.md`.

1. **Design matrix** (5.1 GB, not in git). Written by `experiments/sprint/sprint_prepare.py`, which reads the training cache, the position table `cache_pos_2026_trainctr.parquet`, and `frozen_test_keys.json`, and excludes the holdout-B keys. Rebuilding it was not run in this session. The existing matrix is from Nov 3, 2026.
2. **Explanation pipeline** (run in order; set `CS2_EXPLAIN_MATRIX` and `CS2_EXPLAIN_WORK` if the matrix and the work folder are not at the defaults):
   - `python scripts/explain/xgb83_explain.py --stage train` (refits on CPU, about 83 minutes)
   - `python scripts/explain/xgb83_explain.py --stage explain`
   - `python scripts/explain/xgb83_explain.py --stage report`
   - `python scripts/explain/feature_curves.py` then `python scripts/explain/feature_curve_figs.py`
   - `python scripts/explain/xgb83_local.py` (about 5 seconds, after `explain`)
   - `python scripts/explain/xgb57_explain.py`
3. **Word report.** `scripts/explain/docx/prep_docx_data.py` collects the tables and figures; `build_docx.js` builds the `.docx` with the `docx` package. It takes the output path as its second argument.

## Files

- `README.md`: this file. `README.agents.md` is the same content in the form agents need.
- `XGB_explainability_report.docx`: the 9-page Word report.
- `tables/`: global tables (bundles, inputs, slopes, curves, dependence).
- `figures/`: PNG figures used in the report.
- `local/`: one-prediction waterfalls, contrasts, interactions, and their metadata.
- `xgb57/tables/`: the 57-input model's effects, permutation, and comparison with the 83-input model.
