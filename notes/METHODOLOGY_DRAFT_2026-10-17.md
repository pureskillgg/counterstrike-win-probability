# Methodology draft: CS2 round win-probability model

Draft of 2026-10-17, built from the four notes of 2026-10-16. Text in square brackets marked HOLE is evidence that is missing or a run that is pending. The holes are visible on purpose.

## 1. What the model predicts, and how it is scored

At every second of a live competitive round, the model gives the probability that the CT side wins the round. A round has one winner, so the T probability is 100 minus the CT probability. The served function is `win_probability(data, team="CT", unit="percent")`, which returns a percentage for each tick. Scope is competitive 5v5 on seven maps (ancient, anubis, cache, dust2, inferno, mirage, nuke); wingman and hostage are excluded. The map list is the scope we chose, not a checked copy of the current Active Duty pool.

The score is per-second log loss: the average of −log(probability given to the true result) over every second of held-out matches. Lower is better. We use it because it scores the whole probability (saying 90% and losing costs more than saying 55% and losing), it is a proper scoring rule, and the product gives a probability every second. Round-weighted log loss, where each round counts equally, is a second view. The 95% intervals resample whole matches, because seconds inside one round are strongly correlated; treating seconds as independent would make the intervals 6.1 times too narrow.

## 2. Data and split

The data are 2026 PureSkill CSDS matches, plus 2024 matches for the served model. In 2026, 14,804 matches have the snapshot features and 14,814 are usable. [HOLE: the 10 matches that differ are untraced; the final notes also mention about 10 manifest matches with no grid rows, and nobody has checked whether these are the same ten.]

Matches are assigned to training, validation and test at random by match, fixed once in `split_manifest.json` (sha256 starts a17f219b). New matches get their split from a hash rule, so later results stay comparable. The split is by match because seconds from one match are almost identical, and splitting by second would leak the answer into the test set. It is random rather than chronological because a chronological split puts 77% FACEIT-server matches in the test set and almost none in training, which measures a shift in source rather than the model. As a separate check, an earlier model trained up to 11 October scored 0.00028 worse on 644 later matches (interval [0.00006, 0.00052]).

| | Train | Validation | Test |
|---|---|---|---|
| 59-input model (2026): matches | 10,577 | 1,168 | 3,059 |
| 59-input model: seconds | 13,758,901 | 1,515,228 | 3,956,374 |
| 13-input model (2024 + 2026): matches | 12,509 | 1,383 | 3,601 |
| 13-input model: seconds | 15,933,894 | 1,757,999 | 4,568,043 |

The 13-input counts sit below the manifest's 12,563 / 1,389 / 3,611 because matches with no usable rounds are dropped. [HOLE: confirm that the 59-input model's 3,059 test matches are the 2026 part of the current manifest; the earlier deliverable saw 3,061 against 3,059 and never checked.] The two models are scored on different test matches, so their headline numbers are not like-for-like. [HOLE: score `winprob_v4_basic` on the 2026 test matches only; for the older v3 pair that like-for-like gap was 0.043.]

## 3. Which seconds are used

Each round gives one row per second of live play, at tick = freeze end + 64 × k for k = 0, 1, 2, and so on. Rows stop at the round's effective end (the decisive kill or bomb event), because after it the winner is already known. If no bomb was planted, rows are capped at 116 seconds after freeze end. [HOLE: the reason for the 116 s cap is not written down in the notes.] Dropped: rounds marked unused, the 0.4% of rounds with a missing player, and rounds where the player table and the event log disagree. Training, validation and test use the same rule.

Worked example: test match `csds/2026/09/01/10b093be-5095-4def-b6ab-660f09705c5d/csds` has 13 rounds and 810 seconds. Round 1 has 46 rows, from tick 1441 to tick 4321, with no plant. Round 5 has 129 rows, from tick 18203 to tick 26395, with a plant. We train on every second, on one-second rows; section 7 gives the reasons.

## 4. The inputs

**Served model, 13 inputs.** CT alive, T alive, their difference, bomb planted, seconds since the round went live, seconds since the plant, and seven map one-hots. It uses only the event channels the DSDK tome carries, so it runs on any tome. Its output matches the same booster applied to the grid rows to 0.0001 percentage points on three test matches.

**Best model, 59 inputs.** The 13 above, plus 40 per-side loadout totals (health, armour, helmets, defuse kits, defusing, weapon-class counts, grenades by type, equipment value) and 6 CT-minus-T differences. It is not served: the loadout needs per-tick player status, and the 2026 tome has no such channel.

**Monotone constraints** on the alive counts: more CT players alive never lowers the CT chance, and more T alive never raises it. The reason: this is true of the game, so the trees should not learn noise against it. Ablation, paired on the same seconds: dropping the constraints lowers log loss on the 13-input model by 0.00019 on validation (interval [−0.00036, −0.00002]) and by 0.00044 on test ([−0.00059, −0.00031]). For the 59-input model the two differ by less than 0.0001. The served model still has the constraints. [HOLE: decision open, keep or drop the constraints in the served model; see FINAL_RESULTS section 4c.]

**Left out:** positions, facing direction, smokes, fire, blindness and score lead. Only positions were tested (section 7). [HOLE: no recorded test or stated reason for facing, smokes, fire, blindness or score lead.] [HOLE: event-derived features the tome does carry (grenades, weapon fire, player hurt, deaths) are unmeasured; they are the stated route to narrowing the gap to the 59-input model.]

## 5. The model

XGBoost: gradient-boosted decision trees, each tree correcting the ones before it, with the sum passed through a logistic function. Settings for both models: depth 4, learning rate 0.05, minimum child weight 20, subsample 0.8, early stopping after 100 rounds without validation improvement, seed 0. The best iteration was 1,051 trees for the 59-input model and 443 for the 13-input model. There is no calibration step: isotonic and Platt scaling were tried on validation and did not improve test log loss. A CPU run of the 13-input settings gave test 0.50224 against 0.50220 on the GPU, so the device does not matter.

Seed variance over seeds 0, 1 and 2, on the same split: the 59-input model scores 0.45359 to 0.45369 on validation and 0.45641 to 0.45651 on test (spread 0.00010 on each). The 13-input model scores 0.49960 to 0.49963 on validation and 0.50219 to 0.50221 on test (spreads 0.00003 and 0.00002). Early stopping: patience 50 costs the 59-input model 0.00002 on validation, and patience 200 changes nothing. The 13-input model stops at iteration 443 under all three patience settings.

## 6. How settings were chosen

The baseline is the standard bake-off setting, fixed before any tuning. An early sweep over depth (3, 4, 6, 8) and minimum child weight (20, 100) picked depth 3 on validation by 0.00007, which is noise. On test, depths 3, 4 and 6 were indistinguishable and depth 8 was about 0.0002 worse, so depth 4 stayed because it was set first.

The proper tuning run came with a rule fixed beforehand: 40 random trials plus the baseline, judged on validation only, adopting a tuned setting only if it beat the baseline by at least 0.0005. The baseline scored 0.45584 on validation and the best tuned setting 0.45569, a gain of 0.00015. The baseline was kept. On test, the top three tuned settings were no better; the largest gain, −0.00015 [−0.00025, −0.00004], was not the validation pick and is below the bar. The run trained on every third second, so its validation figures are not comparable with the final model's 0.45369. [HOLE: the tuning run covered the 59-input model only; there is no tuning evidence for the served 13-input model.]

The local `grillme` audit skill is a reconstruction; none of the four public versions found has its seven-field table, so treat it as unverified. [HOLE: no per-choice audit has been run on this document.]

## 7. What was tried and rejected

- **Round-type splits and flags.** The groups differ in difficulty: the 59-input model scores 0.52038 on pistol rounds (1, 13), 0.31462 after pistol (2, 14) and 0.46450 on the rest (test, 3,059 matches). Three things were tried. (a) A pooled model with round-type flags as inputs (pistol, after pistol, round number, overtime): on validation (1,168 matches) it is +0.00015 [−0.00014, +0.00045] against one model for every round, so no gain; on test it is −0.00018 [−0.00036, −0.000006], a small gain the validation set does not confirm. (b) Separate models trained only on pistol rounds: on test, the pistol group is 0.00082 worse [−0.00010, +0.00183]. (c) Separate model for rounds after pistol: on test, that group is 0.00251 worse [0.00105, 0.00390]. The separate models have far fewer training seconds (pistol 1.02 million, after pistol 1.10 million, rest 11.6 million), which is the likely cause. Round-type flags were kept out of the final model because validation shows no gain. [HOLE: validation numbers for the separate models (b) and (c); only test was recorded.]
- **Tree families.** On test (3,059 matches), CatBoost, LightGBM and histogram boosting were worse than XGBoost by 0.00036 to 0.00049. Random forest, ExtraTrees, an MLP and logistic regression were worse by 0.00393 to 0.01100. The first three gaps are real but below the 0.005 bar set for adopting a more complex model, and CatBoost's calibration error is 0.0072 against 0.0032. Random forest, ExtraTrees and the MLP were fit on sampled rows, so they are provisional. [HOLE: validation numbers for the eight-model comparison; the table is test-only, so the family choice rests on test.] Deep models are planned for next week under the same split and bar.
- **Sampling fewer seconds.** One sampled second per round was 0.00235 worse than all seconds and 64 per round was 0.00026 worse, so we use every second. Each setting ran with 2 seeds. [HOLE: validation numbers for the sweep; the rule says validation chooses, and this decision currently rests on test.] [HOLE: three test figures for the same 59-input setup differ at the 0.0001 level: 0.45641 (final model), 0.45647 (every-second row of the sweep, which equals the old v3 figure exactly) and 0.45654 (base of the positions run). Reconcile.]
- **Grid resolution.** The 13-input model was trained on one-second or quarter-second rows (89,657,037 rows). Rule fixed first: adopt quarter-second only if validation improves by at least 0.0005. Validation (1,389 matches): 0.49851 against 0.49854, difference +0.00003 [0.00000, 0.00006]. Test (3,611 matches): 0.50110 against 0.50112. One second kept. These scores use quarter-second rows, so they are not comparable with 0.50220. An earlier run reported 1.10 for the quarter-second model because the two models read rows in different orders; the script now sorts and asserts identical rows.
- **Positions.** Adding 24 position inputs (83 in all) improved the full test set (3,956,374 seconds) from 0.45654 to 0.45326, a difference of −0.00328 [−0.00358, −0.00296], and it helped in every round group. They are rejected for serving, not on merit: the tome carries no per-tick positions.

## 8. Final numbers

| Model | Inputs | Validation log loss (matches) | Test log loss, 95% interval (matches) | Test AUC | Test calibration error (10 bins) |
|---|---|---|---|---|---|
| `winprob_v4_basic` (served) | 13 | 0.49962 (1,383) | 0.50220 [0.49980, 0.50459] (3,601) | 0.8142 | 0.0043 |
| `winprob_v4` (best, not served) | 59 | 0.45369 (1,168) | 0.45641 [0.45333, 0.45914] (3,059) | 0.8578 | 0.0032 |

The DSDK test suite (commit `e8d0e24`) passes 372 tests. Hashes are in `FINAL_RESULTS_2026-10-16.md`. All v4 `.meta.json` files record manifest a17f219b; the v3 files recorded 2088a5a6.

## 9. Limits and open items

One year, one source, seven maps; scores on an unseen map are not measured. Of the 3,059 test matches, 2,452 have no match type (FACEIT-style). On the 574 Valve test matches, training on all matches beat Valve-only by 0.0014 and FACEIT-only by 0.0007.
[HOLE: email of the models to the user. Not sent; it needs the user's explicit approval for each send.]
