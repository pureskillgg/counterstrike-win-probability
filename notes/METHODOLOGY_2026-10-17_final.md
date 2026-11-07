# Round win probability for CS2: methodology

Draft dated 2026-10-17, drafted from the project notes by a Sonnet subagent (`notes/METHODOLOGY_DRAFT_2026-10-17.md`), then polished by a Haiku subagent. Every number here comes from a file named in `notes/FINAL_RESULTS_2026-10-16.md`. Remaining holes are marked **HOLE** and collected at the end.

## 1. What the model predicts, and how it is scored

At every second of a live competitive round, the model gives the probability that the CT side wins. A round has one winner, so the T probability is 100 minus the CT probability. The served function is `win_probability(data, team="CT", unit="percent")`. It returns a percentage for each tick.

Scope: competitive 5v5 on seven maps (ancient, anubis, cache, dust2, inferno, mirage, nuke). Wingman and hostage are excluded. The map list is the scope we chose. It has not been checked against the current Active Duty pool.

The score is per-second log loss: the average of −log(probability given to the true result), taken over every second of held-out matches. Lower is better. It is a proper scoring rule, so a confident wrong answer costs more than a hedged one. Round-weighted log loss, which counts each round equally, is a second view. The 95% intervals resample whole matches, because seconds in one round are strongly correlated. Treating seconds as independent would make the intervals 6.1 times too narrow.

## 2. Data and split

The data are 2026 PureSkill CSDS matches. The served 13-input model also trains on 2024 matches. In 2026, 14,804 matches have snapshot features. The grid holds 14,814 2026 matches. The 10 extra matches are not in the 2026 snapshot cache: 7 train, 1 validation and 2 test. The cache build skipped them, because every round of each one has a missing player and the roster-defect rule drops all of them (`data/combined/exp_cache_skip_trace.json`). The 13-input model drops them too, so they are in neither model's test set.

Matches are assigned once, at random, to training, validation and test, in `data/combined/split_manifest.json` (sha256 starts `a17f219b`). New matches get their split from a hash rule, so later results stay comparable.

The split is by match, not by second, because seconds from one match are nearly identical. Splitting by second would leak the answer. The split is random rather than chronological, because a chronological split puts 77% of FACEIT-server matches in the test set. As a separate check, a model trained up to 11 October scored 0.00028 worse on 644 later matches (interval [0.00006, 0.00052]).

| | Train | Validation | Test |
|---|---|---|---|
| 59-input model (2026): matches | 10,577 | 1,168 | 3,059 |
| 59-input model: seconds | 13,758,901 | 1,515,228 | 3,956,374 |
| 13-input model (2024 and 2026): matches | 12,509 | 1,383 | 3,601 |
| 13-input model: seconds | 15,933,894 | 1,757,999 | 4,568,043 |

All 17,563 manifest matches have grid rows. The 13-input counts are lower than the manifest's 12,563 / 1,389 / 3,611. The reason is that 70 matches (54 train, 6 validation, 10 test) have no rounds left after the roster-defect drop.

The two models are scored on different match sets overall. On the same 3,059 2026 test matches, the 13-input model scores 0.49950 [0.49688, 0.50204] and the 59-input model scores 0.45641 [0.45333, 0.45914]. The gap is 0.04309. This is not a paired interval. The two intervals do not overlap (`data/combined/closeout_checks.json`).

## 3. Which seconds are used

Each round gives one row per second of live play, at tick = freeze end + 64 × k. Rows stop at the round's effective end, the decisive kill or bomb. Once that has happened, the winner is known.

If no bomb was planted, rows are capped at 116 seconds after freeze end. The cap follows the game's round clock of 1.92 minutes, about 115 seconds of live play. The code uses 116, while its own comment says 115, so the extra second is unexplained. The extra second has no measurable effect. 94 test rows (79 matches) fall between 115 and 116 seconds with no bomb planted, and every one ended with a CT win (the timer). Dropping them changes test log loss from 0.50220 to 0.50221.

Past 116 seconds, 17 unplanted test rows remain. Sixteen of them belong to six rounds planted after 116 seconds. The cap applies only to rounds with no plant recorded, so a round planted late keeps its pre-plant rows (`data/combined/exp_cap_trace.json`). The last row is in a 2024 round whose recorded plant comes after the round's effective end, at 116.1 seconds after freeze. The round has a plant record, so no cap applies, and the row at 116.0 seconds lies before the effective end (`data/combined/exp_cap_row_trace.json`).

The following rounds are dropped: rounds marked unused, rounds with a missing player (3,841 of 329,028 rounds across both eras, about 1.2%), and rounds where the player table and the event log disagree. Training, validation and test use the same rule. Including the missing-player rounds in training makes clean test log loss 0.00004 worse, with interval [+0.00002, +0.00006] (`data/combined/exp_roster_sensitivity.json`), so the drop helps slightly.

Worked example: test match `csds/2026/09/01/10b093be-5095-4def-b6ab-660f09705c5d/csds` has 13 rounds. Round 1 has 46 rows, from tick 1441 to tick 4321, with no bomb planted.

## 4. The inputs

**Served model, 13 inputs.** The inputs are CT alive, T alive, their difference, bomb planted, seconds since the round went live, seconds since the plant, and seven map one-hots. The model uses only the event channels that the DSDK tome carries, so it runs on any tome. Its output matches the same booster applied to the grid rows to 0.0001 percentage points on three test matches.

**Best model, 59 inputs.** It has the 13 inputs above, plus 40 per-side loadout totals and 6 CT-minus-T differences. The loadout totals cover health, armour, helmets, defuse kits, defusing, weapon-class counts, grenades by type, and equipment value. It is not served, because the loadout needs per-tick player status, which the 2026 tome does not carry.

**Monotone constraints** on the alive counts: more CT players alive never lowers CT's chance, and more T players alive never raises it. Section 5 tests this constraint.

**Left out:** positions, facing direction, smokes, fire, blindness and score lead. Positions and score lead were tested (section 7).

**HOLE:** facing, smokes, fire and blindness are untested. They are not in the grid, and no test or reason is recorded for them.

**HOLE:** event-derived features the tome does carry are untested. These are grenades, weapon fire, player hurt and deaths. They are the stated route to narrowing the gap to the 59-input model.

## 5. The model

The method is XGBoost: gradient-boosted decision trees, with the sum passed through a logistic function. Both models use these settings: depth 4, learning rate 0.05, minimum child weight 20, subsample 0.8, early stopping after 100 rounds without validation improvement, and seed 0. The best iteration was 1,051 trees for the 59-input model and 443 for the 13-input model.

The served model has no calibration step. Isotonic calibration was tried on pistol seconds only, fitted on validation pistol seconds. On test it made log loss worse: +0.00039 overall [+0.00020, +0.00063] (`data/combined/exp_pistol.json`, stage V4). No Platt scaling result is recorded, so the earlier Platt claim has been removed.

**Device.** The GPU and CPU runs agree. The 13-input model scores 0.50224 on CPU and 0.50220 on GPU.

**Seed variance** (seeds 0, 1 and 2, same split). The 59-input model scores 0.45359 to 0.45369 on validation and 0.45641 to 0.45651 on test: a spread of 0.00010 on each. The 13-input model scores 0.49960 to 0.49963 on validation and 0.50219 to 0.50221 on test: spreads of 0.00003 and 0.00002.

**Early stopping.** Patience 50 costs the 59-input model 0.00002 on validation (0.45371 against 0.45369). Patience 200 changes nothing. The 13-input model stops at iteration 443 under all three settings.

**Monotone constraints, paired** on the same seconds (`data/combined/exp_mono_paired.json`). Each value is the log loss without the constraints minus the log loss with them. For the 13-input model, validation gives −0.00019 (interval [−0.00036, −0.00002], 1,383 matches) and test gives −0.00044 ([−0.00059, −0.00031], 3,601 matches). A negative value means dropping the constraints is better. For the 59-input model, the difference is under 0.0001 on both splits.

**HOLE, decision open:** keep or drop the monotone constraints in the served model. Dropping them lowers log loss on both splits, but the served model still has them. The 0.0005 bar used for tuning (section 6) was not set for this choice, so it does not decide it. For the winning 59-input model, dropping the constraints changes test log loss by +0.00009 [+0.00001, +0.00018] and validation by −0.00006 [−0.00018, +0.00007]. The two splits disagree and both effects are about 0.0001, so the constraints stay on that model (`data/combined/exp_59_nomono.json`).

## 6. How settings were chosen

The baseline is the standard setting above, fixed before any tuning. An early sweep covered depth (3, 4, 6, 8) and minimum child weight (20, 100). It picked depth 3 on validation by 0.00007, which is noise. On test, depths 3, 4 and 6 were indistinguishable, and depth 8 was about 0.0002 worse. Depth 4 was therefore kept.

The tuning run followed a rule fixed beforehand. It ran 40 random trials plus the baseline, judged on validation only. A tuned setting was adopted only if it beat the baseline by at least 0.0005. The baseline scored 0.45584 on validation and the best tuned setting scored 0.45569, a gain of 0.00015, so the baseline was kept. On test, the top three tuned settings differ from the baseline by −0.00004, −0.00008 and −0.00015. The deltas are tuned minus baseline, so negative means better on test. The first two intervals include zero. The third, [−0.00025, −0.00004], excludes zero, but it was not the validation pick. The run trained on every third second, so its validation numbers are not comparable with the final model's 0.45369.

**HOLE:** the tuning covered the 59-input model only. There is no tuning evidence for the served 13-input model.

The local `grillme` audit skill is a reconstruction. None of the four public versions found has its seven-field table, so the skill is unverified. The per-choice audit is in `notes/GRILLME_AUDIT_2026-10-17.md`.

## 7. What was tried and rejected

**Model families** (same split, every second; `data/combined/exp_full_comparison.json`, table in `notes/FINAL_RESULTS_2026-10-16.md` section 4d). On validation, the boosting families sit within 0.0003 of each other, and none beats XGBoost by 0.0005. CatBoost is 0.00034 worse on test, LightGBM 0.00037 and histogram boosting 0.00049. The forests are 0.0039 to 0.0042 worse, ExtraTrees 0.0053, and the MLPs 0.0061 to 0.0077. Logistic regression is 0.0110 worse.

Some families ran on GPU: CatBoost, a PyTorch MLP, cuML logistic regression and random forest. The others ran on CPU. The GPU versions are different implementations, so their differences are about the algorithm, not the hardware. The CatBoost GPU run had no monotone constraints, because the GPU refused them. It is therefore not a like-for-like candidate. Deep models are planned for next week under the same split.

**Sampling fewer seconds per round.** Validation and test agree (`exp_tick_sampling.json`). One sampled second per round is 0.00195 worse on validation and 0.00232 worse on test than using every second. Sixty-four per round is 0.00010 worse on validation and 0.00029 worse on test. So every second is used.

**Round types.** The groups differ in difficulty on test. The 59-input model scores 0.52038 on pistol rounds (rounds 1 and 13), 0.31462 on after-pistol rounds (rounds 2 and 14), and 0.46450 on the rest. Three approaches were tried.
- Pooled model with round-type flags as inputs: validation +0.00015 [−0.00013, +0.00047], so no gain. Test −0.00018 [−0.00036, −0.000006], a small gain that validation does not confirm. The flags were kept out.
- Model for pistol rounds only: validation +0.00010 overall, and +0.00128 on pistol rounds [−0.00022, +0.00281]. Test +0.00082 [−0.00010, +0.00183] on pistol rounds.
- Model for after-pistol rounds only: validation −0.00005 overall, and −0.00066 on after-pistol rounds [−0.00273, +0.00129]. Test +0.00251 [+0.00105, +0.00390] on that group. The test and validation results disagree, so the group effect is noise at these sizes.

The separate models have far fewer training seconds: 1.02 million for pistol, 1.10 million for after pistol, and 11.6 million for the rest. That is the likely reason they cannot beat one model. One model for all rounds stays.

**Grid resolution.** The 13-input model was trained either on one-second rows or on quarter-second rows (89,657,037 rows). The rule was fixed before the run: adopt quarter-second only if validation improved by at least 0.0005. Validation (1,389 matches): one-second 0.49851, quarter-second 0.49854, difference +0.00003 [0.00000, 0.00006]. These use the manifest's 1,389 validation matches, not the served model's 1,383, so they are not directly comparable with 0.49962. Test (3,611 matches): difference +0.00001 [−0.00001, 0.00003]. One second is kept. An earlier run reported 1.10 for the quarter-second model, because the two models read rows in different orders. The script now sorts the rows and asserts that they are identical.

**Score lead at the freeze end** (`data/combined/exp_scorelead.json`). The grid carries score lead as a diagnostic column. Added as a 14th input, with the same settings and the same test seconds, it scores 0.50126 on test, against 0.50220 for the served model: a gain of 0.00094 [−0.00124, −0.00065]. On validation the gain is 0.00099 [−0.00145, −0.00051]. It has not been adopted yet. The grid's notes say the score fields are mostly side-attached. A check on the tome found that the `round_state` scores give the grid's freeze-end lead in every one of 95 rounds across five 2026 matches, so the tome supports it (`scripts/serving/check_score_lead_tome.py`). On the winning 59-input model, score lead still helps: test −0.00070 [−0.00101, −0.00040], validation −0.00094 [−0.00144, −0.00047] (`data/combined/exp_59_scorelead.json`). The final model with it is `models/winprob_v5_59_scorelead`, saved and not served. The served model stays at 13 inputs. **Decision open:** add score lead to the served model. That means retraining the served model with the extra input, updating the serving function, and rerunning the parity test.

**Positions.** Adding 24 position inputs improved the full test set from 0.45654 to 0.45326, a difference of −0.00328 [−0.00358, −0.00296]. It helped in every round group. Positions were rejected for serving, not on merit, because the tome carries no per-tick positions. The positions run's base model (0.45654) was trained through a different path from the served model: a QuantileDMatrix, stopping at iteration 776, against the standard classifier path and iteration 1,051 for the served model (`data/combined/exp_positions_groups.json`, `scripts/model/exp_positions_np.py`). That likely explains the gap of 0.00013. **HOLE:** the cause is inferred from the two configurations and has not been reproduced by retraining.

## 8. Final numbers

| Model | Inputs | Validation log loss (matches) | Test log loss, 95% interval (matches) | Test AUC | Test calibration error (10 bins) |
|---|---|---|---|---|---|
| `winprob_v4_basic` (served) | 13 | 0.49962 (1,383) | 0.50220 [0.49980, 0.50459] (3,601) | 0.8142 | 0.0043 |
| `winprob_v4` (best, not served) | 59 | 0.45369 (1,168) | 0.45641 [0.45333, 0.45914] (3,059) | 0.8578 | 0.0032 |

The DSDK test suite (commit `e8d0e24`) passes 372 tests. Hashes are in `notes/FINAL_RESULTS_2026-10-16.md` section 8. Every v4 `.meta.json` file records manifest `a17f219b`.

## 9. Limits and open items

The data cover one year, one source and seven maps. Scores on an unseen map are not measured. Of the 3,059 test matches, 2,452 have no match type (FACEIT-style). On the 574 Valve test matches, training on all matches beat Valve-only training by 0.0014 and FACEIT-only training by 0.0007.

**Open items.**
- **HOLE:** monotone constraints in the served model, keep or drop (section 5).
- **Decision open:** add score lead to the served model. It gains 0.00094 on test, and the tome supports it (section 7).
- **HOLE:** facing, smokes, fire and blindness are untested (section 4).
- **HOLE:** event-derived features the tome carries are untested (section 4).
- **HOLE:** tuning evidence for the 13-input model (section 6).
- **HOLE:** the positions base figure, 0.45654, is explained by configuration but not reproduced (section 7).
- Email of the models to the user: not sent. Each send needs explicit approval.
