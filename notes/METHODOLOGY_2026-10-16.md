# How the round win-probability model works

Date: 2026-10-16. Data: PureSkill CSDS, 2026 competitive matches. Model: XGBoost on per-second snapshots of a live round.

## 1. What it predicts

At every second of a live round, the model gives the probability that the CT side wins the round. A round ends with exactly one winner, so the T probability is 100 minus the CT probability. The function returns whichever side was asked for, as a percentage.

## 2. The objective

**What we optimise:** binary log loss, the average of −log(probability given to the true result), over every second of the held-out matches. Lower is better.

**Why this one:**

- It scores the whole probability, not only which side won. A model that says 90% and loses is penalised more than one that says 55% and loses.
- It is a proper scoring rule. The best model in expectation is also the one whose probabilities are right on average.
- We score every second because the product gives a probability at every second, not one per round.

**Reported with it:** round-weighted log loss, which gives each round equal weight instead of each second. It is a second view, not the headline. In the full comparison it ranks the models almost the same way.

**Uncertainty:** 95% intervals resample whole matches. Seconds inside one round are strongly correlated, so counting seconds as independent would make the intervals 6.1 times too narrow.

## 3. The data and the split

- **Matches:** 14,804 of the 2026 matches have the snapshot features. The usable count is 14,814, so 10 matches are not in the split. I have not traced those 10 yet.
- **Split:** random by match, fixed once in a manifest. New matches get their split by a hash rule, so later results stay comparable.
- **Sizes:** 10,577 training, 1,168 validation, 3,059 test matches (71.4%, 7.9%, 20.7%).
- **Why by match:** seconds from one match are almost identical. Splitting by second would leak the answer into the test set.
- **Why random and not chronological:** the chronological split puts 77% FACEIT-server matches into the test set and almost none into training. It measures a shift in source, not the model. A later-day test is reported separately below.

Validation chooses the settings. The test set is reported once the settings are fixed.

## 4. Which seconds are in each set

For each round, the model gets one row per second of live play:

- Rows are at tick = freeze end + 64 × k for k = 0, 1, 2, …
- The rows stop before the round's effective end, which is the decisive kill or bomb event. If no bomb was planted, the rows are capped at 116 seconds after freeze end.
- Rounds marked unused in the rounds table are dropped, and so are the 0.4% of rounds with a missing player, and rounds where the player table and event log disagree.
- Training, validation and test all use the same rule. The full comparison uses every second in all three sets. The v3 export used every other second in training.

**Worked example.** Test match `csds/2026/09/01/m_01f731acb3f44b10cb38244f/csds` has 13 rounds and 810 seconds in total.

| Round | Seconds | First tick (second 0) | Last tick | Bomb planted |
|---|---|---|---|---|
| 1 | 46 | 1441 | 4321 | no |
| 5 | 129 | 18203 | 26395 | yes |

Round 1 has 46 rows, one every 64 ticks from tick 1441 to tick 4321. Nothing in between is used.

Full counts: training 13,758,901 seconds, validation 1,515,228, test 3,956,374.

## 5. The inputs (59)

- **Round state:** CT alive, T alive, and their difference.
- **Clock:** seconds since live play started, whether the bomb is planted, seconds since the plant.
- **Map:** seven one-hot columns (ancient, anubis, cache, dust2, inferno, mirage, nuke).
- **Loadout, per side:** health, armour (count and total), helmets, defuse kits, defusing, rifle, AK, M4, AWP, sniper, SMG and heavy counts, no primary, grenades by type, equipment value.
- **Differences, CT minus T:** health, rifle, equipment, armour count, AWP and SMG.

Not used: positions, facing direction, smokes, fire, blindness, and score lead. Positions give a paired gain of 0.0033 log loss (`exp_positions.json`), but that run used only every other second of the test matches (1,992,866 seconds, base 0.45761), not the full test set. A full-second check is needed before quoting the gain against the 0.45641 baseline. Positions are also left out because the event tables do not carry per-tick positions, so the live function cannot compute them yet.

## 6. The model

Gradient-boosted decision trees (XGBoost). Each tree corrects the errors of the ones before it, and the sum of the trees is passed through a logistic function to give the probability.

Settings:

- Tree depth 4, learning rate 0.05, minimum child weight 20, subsample 0.8.
- Early stopping after 100 rounds without improvement on the validation matches.
- Monotone constraints on the alive counts: more CT players alive never lowers the CT chance, and more T players alive never raises it.
- No calibration step. Isotonic and Platt scaling were tested on validation and did not improve the test log loss.

Trained on every second of the training matches, the best validation iteration was 1,051 trees.

## 7. How the settings were chosen

- The settings came from the standard bake-off settings, set before any tuning.
- A sweep over depth (3, 4, 6, 8) and minimum child weight (20, 100) used the same split. Validation picks depth 3 by 0.00007 log loss, which is noise. On test the depths 3, 4 and 6 are indistinguishable. Depth 8 is about 0.0002 worse. Depth 4 was kept because it was set first.
- The 40-trial random search exists as a script but has never been run. The hyperparameter study is an open item.

## 8. Results

Test: 3,059 matches, 3,956,374 seconds. Every model is trained on every second of the 10,577 training matches and scored on the same test seconds.

| Rank | Model | Log loss per second | 95% interval | Difference vs XGBoost | Round-weighted (rank) |
|---|---|---|---|---|---|
| 1 | XGBoost | 0.45641 | [0.45333, 0.45914] | — | 0.43403 (1) |
| 2 | CatBoost | 0.45677 | [0.45368, 0.45950] | +0.00036 | 0.43445 (4) |
| 3 | LightGBM | 0.45678 | [0.45363, 0.45951] | +0.00037 | 0.43432 (2) |
| 4 | Histogram boosting | 0.45691 | [0.45378, 0.45963] | +0.00049 | 0.43443 (3) |
| 5 | Random forest (10% of rows) | 0.46034 | [0.45726, 0.46301] | +0.00393 | 0.43781 (5) |
| 6 | ExtraTrees (10% of rows) | 0.46168 | [0.45880, 0.46424] | +0.00527 | 0.43953 (6) |
| 7 | MLP (25% of rows) | 0.46468 | [0.46154, 0.46761] | +0.00827 | 0.44188 (7) |
| 8 | Logistic regression | 0.46741 | [0.46429, 0.47036] | +0.01100 | 0.44321 (8) |

Reading the table:

- XGBoost is best on the chosen objective and on the round-weighted view.
- The three tree models close behind are worse by 0.0004 to 0.0005. The gaps are real, but they are below the 0.005 bar we set for adopting a more complex model.
- Random forest, ExtraTrees and the MLP fit on sampled rows, so their numbers are provisional.
- CatBoost calibrates worse: 10-bin calibration error 0.0072, against 0.0032 for XGBoost.

The deployed v3 model (every other second in training) scores 0.45647 per second on the same test set. Its round-weighted log loss is 0.43408, its AUC 0.858, and its calibration error 0.0034.

## 9. Limits

- One year and one source (2026 PureSkill competitive). Random split within it.
- Later-day test: a model trained up to 11 October scored 0.00028 worse on 644 later matches (interval [0.00006, 0.00052]).
- The seven maps are the only ones tested. Scores on a map the model has never seen are not measured.
- 2,452 of the 3,059 test matches have no match type (FACEIT-style). Training on all matches beats valve-only by 0.0014 and FACEIT-only by 0.0007 on the 574 Valve test matches.
- One seed for the final fit. Seed variance is not measured.
- Only competitive 5v5 on the seven maps is in scope. Wingman, hostage and other modes are excluded by design.

## 10. Open decisions

1. **Which XGBoost to ship.** The comparison used every second (0.45641). The shipped v3 used every other second (0.45647). The difference is 0.00006. Shipping the full-second version keeps the deliverable consistent with the comparison; it needs a new export.
2. **Grid resolution.** One-second against quarter-second rows, judged on validation (open item).
3. **Hyperparameter study.** Run the 40-trial search and report validation and test side by side (open item).
4. **Positions.** Known gain of 0.0033, not servable yet. Revisit once the event tables or the live feed can supply positions.
5. **Deep models.** Next week, with the same split and the same 0.005 bar.
