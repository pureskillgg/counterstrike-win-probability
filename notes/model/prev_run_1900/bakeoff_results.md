# Round win-probability model bake-off (2026 Oct 01-08)

Script: `scripts/model/bakeoff.py` (seed 0). Log: `logs/bakeoff.log`. Tome: `tomes/2026-10-01_to_08/tome/csds`. Per-tick channels are not in the tome. All quantities are event-derived (spec: `notes/spec/feature_spec.md`).

**Headline (primary split, test set, 234265 states total):** best by primary validation log loss is `XGBoost`: test log loss 0.43660, Brier 0.14637, AUC 0.8703, ECE 0.0089.

## 1. Data counts and filters

Header matches: 1752. round_end rows (rounds): 35992.

| channel | kind | rows | matches with rows | keyset matches | zero-row matches | not in keyset | rows outside header | counted missing |
|---|---|---|---|---|---|---|---|---|
| round_end | dense | 35992 | 1752 | 1752 | 0 | 0 | 0 | 0 |
| round_start | dense | 35992 | 1752 | 1752 | 0 | 0 | 0 | 0 |
| round_state | dense | 531365 | 1752 | 1752 | 0 | 0 | 0 | 0 |
| player_death | dense | 253480 | 1752 | 1752 | 0 | 0 | 0 | 0 |
| player_info | dense | 347243 | 1751 | 1752 | 1 | 0 | 0 | 1 |
| bomb_state | dense | 25733 | 1722 | 1752 | 30 | 0 | 0 | 30 |
| team_change | dense | 19302 | 1735 | 1752 | 17 | 0 | 0 | 17 |
| player_disconnect | sparse | 11594 | 1732 | 1752 | 20 | 0 | 0 | 0 |
| bot_takeover | sparse | 983 | 112 | 1752 | 1640 | 0 | 0 | 0 |

Keysets: every channel's keyset equals the header's match list, so keyset membership drops nothing. The dense-channel zero-row matches are the real filter. Sparse channels (player_disconnect, bot_takeover) are not filtered on zero rows, since zero events is a normal outcome.

| stage | matches | rounds |
|---|---:|---:|
| header / round_end | 1752 | 35992 |
| dropped: missing a required channel | 43 | 490 |
| rounds in kept matches (before round filters) | 1709 | 35502 |
| dropped round: round_end.player_count != 10 | - | 3663 |
| dropped round: disconnect or bot takeover event in round | - | 1767 |
| dropped round: duplicate copy of a kept match (identical rounds, deaths, players) | - | 913 |
| dropped round: not exactly 5 T and 5 CT at start (player_info) | - | 476 |
| matches with no round left after the round filters | 141 | - |
| **remaining (used)** | **1521** | **28683** |

Missing-channel combinations: ['bomb_state']: 26 matches; ['team_change']: 13 matches; ['bomb_state', 'team_change']: 3 matches; ['bomb_state', 'player_info', 'team_change']: 1 matches

States: 234265 (28683 live starts, 205582 death states). Deaths in window: 205582; before live: 0; after round_end: 2029; victim team null in window: 0 (imputed from player_info: 0); repeat in-window victims skipped: 0. Bomb plants in window: 14157; after round_end: 1162; rounds with more than one plant: 0.

## 2. Score fields and side swaps

- t_score + ct_score == round - 1 at live start: 35953 of 35953 rounds.
- Round-to-round transitions: {'consistent': 34200, 'no_next_round_with_freeze': 1753}.
- Transitions where the side-swap schedule (halftime at max_rounds/2, then after the 3rd round of each overtime period) predicts a swap: 1948; consistent with a swap: 1948.
- Inconsistent transitions (round, max_rounds, schedule swap): none.
- Reading: the score fields follow the team on each side, so ct_score - t_score is the CT-minus-T score for the sides in play. This is the basis for the feature.

## 3. Splits

| split | set | matches | rounds | states | CT-win share (states) | match dates |
|---|---|---|---|---|---|---|
| primary | train | 1095 | 20470 | 167379 | 0.5214 | 2026-09-25 .. 2026-10-08 |
| primary | val | 122 | 2364 | 19170 | 0.5190 | 2026-10-01 .. 2026-10-08 |
| primary | test | 304 | 5849 | 47716 | 0.5084 | 2026-09-18 .. 2026-10-08 |
| chronological | train | 1095 | 20497 | 167884 | 0.5171 | 2026-09-18 .. 2026-10-08 |
| chronological | val | 122 | 2466 | 20017 | 0.5154 | 2026-10-08 .. 2026-10-08 |
| chronological | test | 304 | 5720 | 46364 | 0.5251 | 2026-10-08 .. 2026-10-08 |

Primary: 20% of matches are test; 10% of the remaining matches are validation. Chronological: earliest 80% (train pool) and latest 20% (test); the latest 10% of the train pool are validation. No match appears on two sides of any split (checked by assertion).

## 4. Results

Metrics are on states. Log loss, Brier and ECE are the primary criteria; accuracy is reference only. Calibrated validation log loss is out-of-fold (2 folds by match inside validation). The in-sample value is in the last column and is not for selection.

### primary split

| model | set | states | log loss | Brier | AUC | ECE (10 bins) | accuracy (ref) | val LL in-sample (calibrated only) |
|---|---|---|---|---|---|---|---|---|
| A constant | val | 19170 | 0.6924 | 0.2496 | 0.5000 | 0.0024 | 0.5190 |  |
| A constant | test | 47716 | 0.6933 | 0.2501 | 0.5000 | 0.0130 | 0.5084 |  |
| B LR ct_alive_minus_t | val | 19170 | 0.4567 | 0.1514 | 0.8554 | 0.0084 | 0.7577 |  |
| B LR ct_alive_minus_t | test | 47716 | 0.4609 | 0.1532 | 0.8524 | 0.0102 | 0.7595 |  |
| LR all features | val | 19170 | 0.4521 | 0.1501 | 0.8621 | 0.0158 | 0.7657 |  |
| LR all features | test | 47716 | 0.4539 | 0.1510 | 0.8615 | 0.0151 | 0.7641 |  |
| XGBoost | val | 19170 | 0.4351 | 0.1455 | 0.8716 | 0.0124 | 0.7730 |  |
| XGBoost | test | 47716 | 0.4366 | 0.1464 | 0.8703 | 0.0089 | 0.7697 |  |
| LightGBM | val | 19170 | 0.4358 | 0.1457 | 0.8716 | 0.0071 | 0.7746 |  |
| LightGBM | test | 47716 | 0.4376 | 0.1467 | 0.8700 | 0.0091 | 0.7711 |  |
| HistGB | val | 19170 | 0.4365 | 0.1459 | 0.8715 | 0.0083 | 0.7749 |  |
| HistGB | test | 47716 | 0.4369 | 0.1465 | 0.8703 | 0.0102 | 0.7704 |  |
| XGBoost + sigmoid | val | 19170 | 0.4470 | 0.1474 | 0.8713 | 0.0364 | 0.7739 | 0.4467 |
| XGBoost + sigmoid | test | 47716 | 0.4494 | 0.1485 | 0.8703 | 0.0408 | 0.7701 |  |
| XGBoost + isotonic | val | 19170 | 0.4453 | 0.1455 | 0.8710 | 0.0071 | 0.7735 | 0.4318 |
| XGBoost + isotonic | test | 47716 | 0.4401 | 0.1464 | 0.8702 | 0.0121 | 0.7701 |  |
| LightGBM + sigmoid | val | 19170 | 0.4360 | 0.1458 | 0.8714 | 0.0078 | 0.7746 | 0.4357 |
| LightGBM + sigmoid | test | 47716 | 0.4379 | 0.1469 | 0.8700 | 0.0123 | 0.7713 |  |
| LightGBM + isotonic | val | 19170 | 0.4409 | 0.1457 | 0.8712 | 0.0054 | 0.7749 | 0.4327 |
| LightGBM + isotonic | test | 47716 | 0.4384 | 0.1468 | 0.8700 | 0.0119 | 0.7715 |  |
| HistGB + sigmoid | val | 19170 | 0.4367 | 0.1460 | 0.8712 | 0.0062 | 0.7750 | 0.4363 |
| HistGB + sigmoid | test | 47716 | 0.4372 | 0.1466 | 0.8703 | 0.0125 | 0.7705 |  |
| HistGB + isotonic | val | 19170 | 0.4442 | 0.1462 | 0.8700 | 0.0055 | 0.7753 | 0.4331 |
| HistGB + isotonic | test | 47716 | 0.4402 | 0.1467 | 0.8703 | 0.0153 | 0.7694 |  |

### chronological split

| model | set | states | log loss | Brier | AUC | ECE (10 bins) | accuracy (ref) | val LL in-sample (calibrated only) |
|---|---|---|---|---|---|---|---|---|
| A constant | val | 20017 | 0.6927 | 0.2498 | 0.5000 | 0.0017 | 0.5154 |  |
| A constant | test | 46364 | 0.6920 | 0.2494 | 0.5000 | 0.0080 | 0.5251 |  |
| B LR ct_alive_minus_t | val | 20017 | 0.4478 | 0.1488 | 0.8601 | 0.0120 | 0.7622 |  |
| B LR ct_alive_minus_t | test | 46364 | 0.4618 | 0.1537 | 0.8509 | 0.0095 | 0.7545 |  |
| LR all features | val | 20017 | 0.4464 | 0.1486 | 0.8635 | 0.0148 | 0.7640 |  |
| LR all features | test | 46364 | 0.4547 | 0.1516 | 0.8593 | 0.0160 | 0.7624 |  |
| XGBoost | val | 20017 | 0.4324 | 0.1451 | 0.8712 | 0.0176 | 0.7700 |  |
| XGBoost | test | 46364 | 0.4390 | 0.1475 | 0.8677 | 0.0080 | 0.7690 |  |
| LightGBM | val | 20017 | 0.4315 | 0.1449 | 0.8720 | 0.0184 | 0.7701 |  |
| LightGBM | test | 46364 | 0.4374 | 0.1470 | 0.8688 | 0.0108 | 0.7687 |  |
| HistGB | val | 20017 | 0.4342 | 0.1459 | 0.8711 | 0.0146 | 0.7705 |  |
| HistGB | test | 46364 | 0.4372 | 0.1469 | 0.8693 | 0.0128 | 0.7694 |  |
| XGBoost + sigmoid | val | 20017 | 0.4456 | 0.1474 | 0.8706 | 0.0440 | 0.7677 | 0.4448 |
| XGBoost + sigmoid | test | 46364 | 0.4521 | 0.1498 | 0.8677 | 0.0430 | 0.7671 |  |
| XGBoost + isotonic | val | 20017 | 0.4345 | 0.1451 | 0.8699 | 0.0088 | 0.7677 | 0.4280 |
| XGBoost + isotonic | test | 46364 | 0.4431 | 0.1476 | 0.8672 | 0.0102 | 0.7692 |  |
| LightGBM + sigmoid | val | 20017 | 0.4318 | 0.1451 | 0.8716 | 0.0144 | 0.7694 | 0.4311 |
| LightGBM + sigmoid | test | 46364 | 0.4376 | 0.1471 | 0.8688 | 0.0125 | 0.7690 |  |
| LightGBM + isotonic | val | 20017 | 0.4345 | 0.1450 | 0.8701 | 0.0093 | 0.7661 | 0.4275 |
| LightGBM + isotonic | test | 46364 | 0.4417 | 0.1471 | 0.8686 | 0.0114 | 0.7694 |  |
| HistGB + sigmoid | val | 20017 | 0.4346 | 0.1460 | 0.8708 | 0.0113 | 0.7703 | 0.4340 |
| HistGB + sigmoid | test | 46364 | 0.4371 | 0.1469 | 0.8693 | 0.0121 | 0.7693 |  |
| HistGB + isotonic | val | 20017 | 0.4424 | 0.1461 | 0.8694 | 0.0101 | 0.7655 | 0.4307 |
| HistGB + isotonic | test | 46364 | 0.4413 | 0.1468 | 0.8691 | 0.0083 | 0.7698 |  |

Comparison with the first run (duplicate copies not removed). Primary test log loss, from `logs/bakeoff_run1_no_dedupe.log`. The change mixes duplicate removal with a different split (the match set changed), so it is not a clean measure of leakage:

| model | run 1 test LL (duplicates kept) | this run test LL | change |
|---|---|---|---|
| A constant | 0.6921 | 0.6933 | +0.0012 |
| B LR ct_alive_minus_t | 0.4579 | 0.4609 | +0.0030 |
| LR all features | 0.4516 | 0.4539 | +0.0023 |
| XGBoost | 0.4335 | 0.4366 | +0.0031 |
| LightGBM | 0.4342 | 0.4376 | +0.0034 |
| HistGB | 0.4332 | 0.4369 | +0.0037 |

Model settings (no tuning was done; the same settings were used on both splits): XGBoost hist, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, early stopping 100. LightGBM 15 leaves, lr 0.05, min_child_samples 50, early stopping 100. HistGB 15 leaves, lr 0.05, min_samples_leaf 50, internal early stopping. Logistic regressions are standardised with C=1.0.

Per-model details:

- primary, A constant: train CT-win rate 0.5214
- primary, B LR ct_alive_minus_t: standardised, C=1.0
- primary, LR all features: standardised, C=1.0, 10 features
- primary, XGBoost: hist, depth 4, eta 0.05, best_iteration 261; default predict uses same trees: True
- primary, LightGBM: leaves 15, lr 0.05, best_iteration 173; eval_X/eval_y; default predict uses same trees: True
- primary, HistGB: leaves 15, lr 0.05, internal early stopping (random 10% of training rows), n_iter 564
- primary, XGBoost + sigmoid: CalibratedClassifierCV(FrozenEstimator(XGBoost)), fitted on validation; validation selection score is 2-fold out-of-fold
- primary, XGBoost + isotonic: CalibratedClassifierCV(FrozenEstimator(XGBoost)), fitted on validation; validation selection score is 2-fold out-of-fold
- primary, LightGBM + sigmoid: CalibratedClassifierCV(FrozenEstimator(LightGBM)), fitted on validation; validation selection score is 2-fold out-of-fold
- primary, LightGBM + isotonic: CalibratedClassifierCV(FrozenEstimator(LightGBM)), fitted on validation; validation selection score is 2-fold out-of-fold
- primary, HistGB + sigmoid: CalibratedClassifierCV(FrozenEstimator(HistGB)), fitted on validation; validation selection score is 2-fold out-of-fold
- primary, HistGB + isotonic: CalibratedClassifierCV(FrozenEstimator(HistGB)), fitted on validation; validation selection score is 2-fold out-of-fold
- chronological, A constant: train CT-win rate 0.5171
- chronological, B LR ct_alive_minus_t: standardised, C=1.0
- chronological, LR all features: standardised, C=1.0, 10 features
- chronological, XGBoost: hist, depth 4, eta 0.05, best_iteration 174; default predict uses same trees: True
- chronological, LightGBM: leaves 15, lr 0.05, best_iteration 170; eval_X/eval_y; default predict uses same trees: True
- chronological, HistGB: leaves 15, lr 0.05, internal early stopping (random 10% of training rows), n_iter 786
- chronological, XGBoost + sigmoid: CalibratedClassifierCV(FrozenEstimator(XGBoost)), fitted on validation; validation selection score is 2-fold out-of-fold
- chronological, XGBoost + isotonic: CalibratedClassifierCV(FrozenEstimator(XGBoost)), fitted on validation; validation selection score is 2-fold out-of-fold
- chronological, LightGBM + sigmoid: CalibratedClassifierCV(FrozenEstimator(LightGBM)), fitted on validation; validation selection score is 2-fold out-of-fold
- chronological, LightGBM + isotonic: CalibratedClassifierCV(FrozenEstimator(LightGBM)), fitted on validation; validation selection score is 2-fold out-of-fold
- chronological, HistGB + sigmoid: CalibratedClassifierCV(FrozenEstimator(HistGB)), fitted on validation; validation selection score is 2-fold out-of-fold
- chronological, HistGB + isotonic: CalibratedClassifierCV(FrozenEstimator(HistGB)), fitted on validation; validation selection score is 2-fold out-of-fold

## 5. Selection and calibration

- Best by primary validation log loss: **XGBoost** (test log loss 0.43660).
- Best uncalibrated by primary validation log loss: XGBoost.
- Runner-up among uncalibrated models on primary validation: LightGBM (0.43581), gap 0.00070. Gaps of this size between tree models were not tested for significance.
- Calibrated variants: the best is XGBoost + isotonic at 0.44527 out-of-fold validation log loss, against 0.43511 for XGBoost. The rule as written picks the raw model.
- Best by chronological validation log loss: LightGBM.
- Model file: no. best model is uncalibrated, so no model file is saved.

### Calibration of `XGBoost` (primary split, test set)

- XGBoost: test log loss 0.43660, Brier 0.14637, ECE 0.0089
- XGBoost + sigmoid: test log loss 0.44935, Brier 0.14850, ECE 0.0408
- XGBoost + isotonic: test log loss 0.44010, Brier 0.14642, ECE 0.0121

Reliability: XGBoost, primary test

| bin | states | mean predicted | observed CT-win | |gap| |
|---|---|---|---|---|
| 0.0-0.1 | 7283 | 0.0271 | 0.0264 | 0.0007 |
| 0.1-0.2 | 2846 | 0.1383 | 0.1279 | 0.0104 |
| 0.2-0.3 | 4436 | 0.2573 | 0.2561 | 0.0012 |
| 0.3-0.4 | 2403 | 0.3288 | 0.3059 | 0.0229 |
| 0.4-0.5 | 5173 | 0.4616 | 0.4655 | 0.0039 |
| 0.5-0.6 | 6572 | 0.5494 | 0.5169 | 0.0325 |
| 0.6-0.7 | 2832 | 0.6641 | 0.6480 | 0.0162 |
| 0.7-0.8 | 5053 | 0.7420 | 0.7398 | 0.0022 |
| 0.8-0.9 | 3271 | 0.8621 | 0.8575 | 0.0046 |
| 0.9-1.0 | 7847 | 0.9714 | 0.9746 | 0.0033 |

Reliability: XGBoost + sigmoid, primary test

| bin | states | mean predicted | observed CT-win | |gap| |
|---|---|---|---|---|
| 0.0-0.1 | 8020 | 0.0656 | 0.0329 | 0.0327 |
| 0.1-0.2 | 4136 | 0.1486 | 0.1874 | 0.0388 |
| 0.2-0.3 | 4354 | 0.2399 | 0.2827 | 0.0428 |
| 0.3-0.4 | 880 | 0.3543 | 0.3682 | 0.0138 |
| 0.4-0.5 | 4587 | 0.4553 | 0.4689 | 0.0136 |
| 0.5-0.6 | 4821 | 0.5569 | 0.5111 | 0.0458 |
| 0.6-0.7 | 2470 | 0.6327 | 0.5494 | 0.0833 |
| 0.7-0.8 | 4445 | 0.7621 | 0.6884 | 0.0737 |
| 0.8-0.9 | 4738 | 0.8451 | 0.7881 | 0.0570 |
| 0.9-1.0 | 9265 | 0.9347 | 0.9604 | 0.0257 |

Reliability: XGBoost + isotonic, primary test

| bin | states | mean predicted | observed CT-win | |gap| |
|---|---|---|---|---|
| 0.0-0.1 | 7384 | 0.0281 | 0.0267 | 0.0014 |
| 0.1-0.2 | 2578 | 0.1476 | 0.1296 | 0.0180 |
| 0.2-0.3 | 3096 | 0.2520 | 0.2387 | 0.0133 |
| 0.3-0.4 | 4688 | 0.3257 | 0.3114 | 0.0143 |
| 0.4-0.5 | 2692 | 0.4680 | 0.4625 | 0.0055 |
| 0.5-0.6 | 8874 | 0.5378 | 0.5205 | 0.0173 |
| 0.6-0.7 | 2221 | 0.6627 | 0.6592 | 0.0035 |
| 0.7-0.8 | 4951 | 0.7715 | 0.7390 | 0.0325 |
| 0.8-0.9 | 2358 | 0.8528 | 0.8414 | 0.0114 |
| 0.9-1.0 | 8874 | 0.9696 | 0.9643 | 0.0054 |

Reliability: XGBoost raw, chronological test

| bin | states | mean predicted | observed CT-win | |gap| |
|---|---|---|---|---|
| 0.0-0.1 | 6578 | 0.0290 | 0.0234 | 0.0055 |
| 0.1-0.2 | 2685 | 0.1408 | 0.1378 | 0.0030 |
| 0.2-0.3 | 3968 | 0.2590 | 0.2558 | 0.0032 |
| 0.3-0.4 | 2874 | 0.3264 | 0.3386 | 0.0121 |
| 0.4-0.5 | 4263 | 0.4581 | 0.4731 | 0.0150 |
| 0.5-0.6 | 7264 | 0.5481 | 0.5308 | 0.0173 |
| 0.6-0.7 | 2478 | 0.6592 | 0.6747 | 0.0155 |
| 0.7-0.8 | 5322 | 0.7456 | 0.7460 | 0.0003 |
| 0.8-0.9 | 3033 | 0.8649 | 0.8632 | 0.0017 |
| 0.9-1.0 | 7899 | 0.9693 | 0.9747 | 0.0054 |

## 6. What looked wrong or needs a check

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

Run time: 3285.3s.
