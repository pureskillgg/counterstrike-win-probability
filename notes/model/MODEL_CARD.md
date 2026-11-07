# Model card: CS2 round win-probability baseline (XGBoost, primary split, seed 0, deduplicated)

Created 2026-10-12. Status: first baseline. Offline analysis only; not validated outside its training window.

## Scope and training window

- The model is trained on 2026-09-18 to 2026-10-08 data only. It has NOT been tested on 2024 or any other period.
- Tome: `tomes/2026-10-01_to_08/tome/csds`. Data files carry the suffix `.2026-09-18,2026-10-08` (from `scripts/model/feature_validation.py`).
- Match dates in use: 2026-09-18 to 2026-10-08.

## Purpose

Estimate the probability that the CT side wins the round, from the game state at the round's live start and at each in-window death. This is the baseline for the round win-probability project, to compare later models against. It is not a production model.

## Label

`ct_win = 1 if winner_team_code == 3 (CT) else 0`. The round label is repeated on every state of the round, so states from one round share a label.

## Features (10, in this order)

Monotone constraints are as set in `scripts/model/bakeoff.py` (MONO_MAP).

| feature | definition | monotone |
|---|---|---|
| alive_ct | CT players alive at the state tick: 5 at live start, minus one per in-window CT death (first death per victim) | +1 |
| alive_t | T players alive at the state tick, same rule | -1 |
| ct_alive_minus_t | alive_ct minus alive_t | +1 |
| bomb_planted | 1 if a bomb plant in the live window has happened at or before the state tick, else 0 | none |
| seconds_since_round_live | (state tick minus live-start tick) / 64 | none |
| seconds_since_plant | (state tick minus plant tick) / 64 if bomb_planted, else -1 | 0 |
| round_number | round index within the match (round field of round_end) | none |
| ct_score_minus_t_score | CT score minus T score at the live start (round_freeze_end row); constant within the round | +1 |
| is_pistol_round | 1 if the round is 1 or max_rounds//2 + 1, else 0 | none |
| is_overtime | 1 if the round is greater than max_rounds, else 0 | none |

States per kept round: one live-start state, plus one state per in-window death. Plants after round_end are excluded (1162 plants, per the bakeoff log).

## Filters

Match level: a match is dropped if a required channel is missing. Dense channels (round_end, round_start, round_state, player_death, player_info, bomb_state, team_change) need rows; zero rows count as missing. Sparse channels (player_disconnect, bot_takeover) only need keyset membership, since zero rows is a normal outcome. Result: 43 matches dropped (490 rounds). Missing combinations: bomb_state only 26; team_change only 13; bomb_state and team_change 3; bomb_state, player_info and team_change 1.

Round level, applied in this order, each dropped round counted once:
- round_end.player_count != 10 (3,663 rounds).
- Not exactly 5 T and 5 CT at start, from player_info (476 rounds).
- A disconnect or bot takeover event anywhere in the round (1,767 rounds).
- No round_start row, no round_freeze_end (live start) row, or null score at live start (0 rounds each).
- A death in the live window with an unresolved victim side (0 rounds).

Duplicate matches, removed: 46 groups, 97 matches share identical round_end, player_death and player_info events under different keys (the same demo processed twice). The lowest key is kept in each group. That removes 51 matches. 47 of them had rounds left after the round filters, and 913 rounds were removed. Header match_date values for copies differ by seconds to over an hour. Before removal, 17 duplicate groups had members on different sides of the primary split, so the removal is needed to stop leakage between train and test.

Flow (matches / rounds): header 1,752 / 35,992; after channel filter 1,709 / 35,502; after round filters 1,568 / 29,596 (141 matches with no round left); after duplicate removal 1,521 / 28,683. States: 234,265 (28,683 live starts, 205,582 death states).

## Split and seed

- Primary split, seed 0. Matches sorted by key, shuffled with `random.Random(0)`. 20% of matches are test; 10% of the remaining matches are validation (used for early stopping and calibration); the rest train.
- Counts (matches / rounds / states / CT-win share of states):
  - train: 1,095 / 20,470 / 167,379 / 0.5214 (match dates 2026-09-25 to 2026-10-08)
  - validation: 122 / 2,364 / 19,170 / 0.5190 (2026-10-01 to 2026-10-08)
  - test: 304 / 5,849 / 47,716 / 0.5084 (2026-09-18 to 2026-10-08)
- Chronological split (secondary, reported for comparison): earliest 80% of matches by match_date form the train pool, latest 20% are test. Its test set is all dated 2026-10-08.

## Model and settings

XGBoost Booster (xgboost 3.4.2), tree_method hist, device cpu, max_depth 4, learning_rate 0.05, min_child_weight 20, subsample 0.8, n_estimators cap 3000, early stopping 100 rounds on validation logloss, random_state 0, monotone constraints as above. best_iteration 261. Predict with `iteration_range=(0, 262)`. No calibration. No hyperparameter search.

Files:
- Model: `models/xgb_primary_seed0.json` (XGBoost Booster JSON).
- Metadata (feature list, best iteration, filter and split counts, metrics): `models/xgb_primary_seed0_meta.json`.
- Source: `scripts/model/xgb_primary_seed0.py` (reuses `scripts/model/bakeoff.py`).

## Test metrics (all models in the results file)

Source: `notes/model/bakeoff_results.md` section 4. Metrics on states. Log loss (LL), Brier, AUC and ECE (10 bins). Lower LL, Brier and ECE are better; higher AUC is better. Calibrated variants are fitted on validation only.

| model | primary test LL | primary Brier | primary AUC | primary ECE | chrono test LL | chrono Brier | chrono AUC | chrono ECE |
|---|---|---|---|---|---|---|---|---|
| A constant | 0.6933 | 0.2501 | 0.5000 | 0.0130 | 0.6920 | 0.2494 | 0.5000 | 0.0080 |
| B LR ct_alive_minus_t | 0.4609 | 0.1532 | 0.8524 | 0.0102 | 0.4618 | 0.1537 | 0.8509 | 0.0095 |
| LR all features | 0.4539 | 0.1510 | 0.8615 | 0.0151 | 0.4547 | 0.1516 | 0.8593 | 0.0160 |
| **XGBoost** | **0.4366** | **0.1464** | 0.8703 | **0.0089** | 0.4390 | 0.1475 | 0.8677 | 0.0080 |
| LightGBM | 0.4376 | 0.1467 | 0.8700 | 0.0091 | 0.4374 | 0.1470 | 0.8688 | 0.0108 |
| HistGB | 0.4369 | 0.1465 | 0.8703 | 0.0102 | 0.4372 | 0.1469 | 0.8693 | 0.0128 |
| XGBoost + sigmoid | 0.4494 | 0.1485 | 0.8703 | 0.0408 | 0.4521 | 0.1498 | 0.8677 | 0.0430 |
| XGBoost + isotonic | 0.4401 | 0.1464 | 0.8702 | 0.0121 | 0.4431 | 0.1476 | 0.8672 | 0.0102 |
| LightGBM + sigmoid | 0.4379 | 0.1469 | 0.8700 | 0.0123 | 0.4376 | 0.1471 | 0.8688 | 0.0125 |
| LightGBM + isotonic | 0.4384 | 0.1468 | 0.8700 | 0.0119 | 0.4417 | 0.1471 | 0.8686 | 0.0114 |
| HistGB + sigmoid | 0.4372 | 0.1466 | 0.8703 | 0.0125 | 0.4371 | 0.1469 | 0.8693 | 0.0121 |
| HistGB + isotonic | 0.4402 | 0.1467 | 0.8703 | 0.0153 | 0.4413 | 0.1468 | 0.8691 | 0.0083 |

XGBoost on the primary split: validation LL 0.4351, validation Brier 0.1455, test LL 0.43660, test Brier 0.14637. Reproduced on 2026-10-12 by `scripts/model/xgb_primary_seed0.py` (test LL and Brier within 0.0002 of the results file; counts match exactly).

Effect of deduplication: the earlier run with duplicate copies kept gave XGBoost primary test LL 0.4335. Removing the duplicates raised it to 0.4366 (+0.0031). The 0.4335 figure is pre-dedupe and is no longer the reference.

## Why XGBoost (judgment call)

The choice of XGBoost over HistGB and LightGBM is a judgment call, not a clear win.
- Primary split: XGBoost has the lowest validation LL (0.4351, against LightGBM 0.4358 and HistGB 0.4365) and the lowest test LL (0.4366, against LightGBM 0.4376 and HistGB 0.4369).
- Chronological split: XGBoost is not best. Validation LL 0.4324 against LightGBM 0.4315 and HistGB 0.4342. Test LL 0.4390 against LightGBM 0.4374 and HistGB 0.4372.
- XGBoost is within 0.0018 of the best model on each split by test LL, and the three tree models are close.
- Literature precedent for this model class: Xenopoulos et al. 2020, arXiv 2011.01324. Cited as precedent only; this card does not re-check that paper's results.

Selection rule used: primary validation LL. It picks raw XGBoost. Calibrated variants were worse out of fold (XGBoost + isotonic 0.4453, + sigmoid 0.4470 against raw 0.4351). Sigmoid calibration in particular raises test ECE to 0.0408.

## Known issues (copied in full from `notes/model/bakeoff_results.md`, section 6)

- Feature validation's claim (`notes/spec/feature_spec.md`, section 7) that 1,391 matches have fewer than 24 rounds and suggests truncated demos is contradicted by the header: 1350 matches have fewer rounds than max_rounds, of which 1324 end when one side reaches the win threshold (max_rounds//2+1). Only 26 look incomplete. All 149 matches in the 16-round format are below 24 rounds by construction, so the 24-round filter was never meaningful for them. A short match is a normal end, not a fault.
- Round count vs header final scores: 1672 matches agree; 80 have fewer round_end rows than the header total (the tome is missing the last rounds, consistent with truncation), and 0 have more. Cause for the extra rounds: UNVERIFIED. Kept rounds are still complete per round, so the filters do not depend on this.
- Matches missing a dense channel: ['bomb_state'] 26; ['team_change'] 13; ['bomb_state', 'team_change'] 3; ['bomb_state', 'player_info', 'team_change'] 1. Plants per round in the whole tome: 0.51. A match with 9 or more rounds and zero bomb_state rows is then unlikely to be a real no-plant match, so these look like missing data (UNVERIFIED: no per-tick source to confirm).
- Rounds with player_count != 10: 3663 of the rounds in kept matches. Disconnect or takeover rounds dropped: 1767. Rounds not 5 vs 5 at start from player_info: 476.
- Duplicate matches: 46 groups (97 matches) have identical round_end, player_death and player_info events under different keys. Copies differ in header match_date by seconds to over an hour. The first run did not remove them, and copies landed on different sides of the split: primary 17 groups, chronological 0 groups. This run keeps the lowest key per group. demo_id and sharecode are 'redacted' in every header row, so the check is on events only. Round-fingerprint check (>=5 rounds, all matches) found 47 groups.
- match_date: 1133 of 1752 matches (65%) are dated 2026-10-08. The match_date day equals the processing day in the key path for 1730 of 1752 matches. The spec calls matchDate the played date. Whether the field is really the played date is UNVERIFIED. The chronological test set is the latest 20% by timestamp, and here it is all on 2026-10-08, so it checks within-day drift, not multi-day drift.
- Sigmoid calibration: sklearn's `_SigmoidCalibration.fit` fits a logistic on the raw predicted probabilities (not their logits), so it cannot represent the identity map. For that reason it can make a model worse than the raw score, even in-sample. Compare XGBoost raw and XGBoost + sigmoid validation log loss in the results table. Isotonic is not limited this way. No other calibrator was tried.
- Event-anchored states: a state exists only at the live start and at each death. A bomb plant with no later death adds no state, and the plant's effect appears only at the next state. This follows the spec, but it changes the time distribution of states.
- HistGB early stopping uses a random 10% of training rows (row-level), not the match-level validation set. Its stopping point may be optimistic. Calibration of HistGB still uses the match-level validation set.
- LightGBM 4.7.0 deprecates eval_set in favour of eval_X/eval_y. The run uses eval_X/eval_y as the task specifies.
- Calibrated models: in-sample validation scores are flattering. Selection uses out-of-fold validation scores. Test scores come from a calibrator fitted on all of validation.
- The validation fraction is read as 10% of the 80% (primary) or the earliest 80% (chrono) train pool, not 10% of all matches. This is an interpretation.
- Metrics are on states, not rounds, so long rounds (more deaths) carry more weight. Round-level metrics were not computed.
- No hyperparameter search was run, so these results are for one fixed setting per model.

## Limitations

- Trained and tested on one window, 2026-09-18 to 2026-10-08. The model is NOT tested on 2024 or any other period, so no claim is made about generalization.
- The unit is the state, not the round, so long rounds with more deaths carry more weight. Round-level metrics were not computed.
- Event-anchored states only: a bomb plant with no later death adds no state.
- No hyperparameter search. One fixed setting per model.
- The data-quality items above are unresolved. Some are marked UNVERIFIED in the source.
