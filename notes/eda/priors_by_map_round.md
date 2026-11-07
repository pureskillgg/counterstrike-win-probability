# Task C: pre-round CT win priors (map, round, halves, pistol, score difference, platform, match_type)

Source: 2026 tomes at G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds (header, round_end, round_state). No E: reads. Script: scripts\eda\priors_by_map_round.py. Run date 2026-10-13.

## 1. Scope
- header matches 1752; wingman 149 (max_rounds 16); non-wingman 1603 (max_rounds 24).
- round_end rows 35992; wingman rounds 2038; non-wingman rounds not 5v5 1913 (of which player_count 9: 1380).
- clean 5v5 rounds 32041 in 1592 matches.
- clean rounds with no round_freeze_end row (score-based tables exclude them): 3.
- Batch: 2026-10-08 has 23086 clean rounds (72.1% of clean rounds); 2026-10-01..10-07 has 8915; other dates 40.

## 2. Verified structure (from scripts\eda\priors_probe.py, priors_score_diag.py, priors_score_diag2.py, priors_playercount_check.py)
- Halves are 12 rounds, not 24. The phase label flips to 'second' at round 12 for 1,566 non-wingman matches (20 at round 13; wingman halves at round 8). The task text's '24-round halves' is contradicted by the data.
- round_end.winner_team_code is side-based (2 = T, 3 = CT), consistent with the win_reason pairing in schema_dictionary.md.
- round_state t_score/ct_score follow the TEAM across side swaps, not the side. Every r -> r+1 transition (34,200 pairs, wingman included) fits at least one of two hypotheses; none fits neither. At the halftime swap after round 12, the 1,329 decisive pairs fit only the swap form and 255 fit both (tied post-round scores, so uninformative); at round 13 and most other rounds the decisive pairs fit only the side-stay form. OT swap-only pairs occur after rounds 27 (163 matches) and 33 (37 matches). The schema_dictionary.md section 7 claim 'scores are cumulative per side' is contradicted here.
- score_update has non-unit deltas (3,044 rows, e.g. +8 or -8) at the same swaps, consistent with the team-based score.
- Round 1 live start is 0-0 in every match. Live start exists for 35,953 of 35,992 round_end rounds (39 missing).
- Overtime rounds exist: 1,165 non-wingman rounds in 163 matches (rounds 25 to 48).
- The 10-player filter drops 1,913 non-wingman rounds (5.6%): 1,380 are 9-player rounds, and the non-clean share is about 3% in rounds 1 to 9, 6% in round 12, 7% in round 15 and 21% in round 24 (matches that end early). Non-clean CT rates differ from clean ones in some rounds (for example R9: 61% versus 55%), but each round has only about 40 to 100 non-clean rounds, so the filter is a weak selection risk (scripts\eda\priors_playercount_check.py).

## 3. CT win rate tables (clean 5v5 rounds)

**By round number (OT pooled)** (overall CT rate 52.14% over 32041 rounds; Bonferroni alpha 0.002 across 25 cells)

| cell | matches | rounds | CT wins | CT win rate | diff vs overall (pp) | binomial p vs overall | sig. after Bonferroni |
|---|---|---|---|---|---|---|---|
| R1 | 1559 | 1559 | 770 | 49.39% | -2.75 | 0.031 |  |
| R2 | 1556 | 1556 | 766 | 49.23% | -2.91 | 0.022 |  |
| R3 | 1557 | 1557 | 804 | 51.64% | -0.50 | 0.7 |  |
| R4 | 1557 | 1557 | 804 | 51.64% | -0.50 | 0.7 |  |
| R5 | 1552 | 1552 | 814 | 52.45% | +0.31 | 0.82 |  |
| R6 | 1542 | 1542 | 828 | 53.70% | +1.56 | 0.22 |  |
| R7 | 1532 | 1532 | 815 | 53.20% | +1.06 | 0.41 |  |
| R8 | 1528 | 1528 | 811 | 53.08% | +0.94 | 0.47 |  |
| R9 | 1532 | 1532 | 838 | 54.70% | +2.56 | 0.046 |  |
| R10 | 1526 | 1526 | 809 | 53.01% | +0.88 | 0.51 |  |
| R11 | 1518 | 1518 | 821 | 54.08% | +1.95 | 0.13 |  |
| R12 | 1490 | 1490 | 808 | 54.23% | +2.09 | 0.11 |  |
| R13 | 1504 | 1504 | 712 | 47.34% | -4.80 | 0.0002 | yes |
| R14 | 1481 | 1481 | 686 | 46.32% | -5.82 | 7.6e-06 | yes |
| R15 | 1426 | 1426 | 715 | 50.14% | -2.00 | 0.14 |  |
| R16 | 1370 | 1370 | 719 | 52.48% | +0.35 | 0.81 |  |
| R17 | 1297 | 1297 | 704 | 54.28% | +2.14 | 0.13 |  |
| R18 | 1181 | 1181 | 649 | 54.95% | +2.82 | 0.055 |  |
| R19 | 1072 | 1072 | 574 | 53.54% | +1.41 | 0.36 |  |
| R20 | 951 | 951 | 499 | 52.47% | +0.33 | 0.85 |  |
| R21 | 810 | 810 | 423 | 52.22% | +0.09 | 0.97 |  |
| R22 | 648 | 648 | 372 | 57.41% | +5.27 | 0.0075 |  |
| R23 | 480 | 480 | 258 | 53.75% | +1.61 | 0.49 |  |
| R24 | 284 | 284 | 134 | 47.18% | -4.95 | 0.097 |  |
| OT R25+ | 162 | 1088 | 572 | 52.57% | +0.44 | 0.78 |  |

Chi-square homogeneity over 25 cells with >= 30 rounds: chi2 76.43, df 24, p = 2.2e-07; min expected count 135.9. Cells left out of the omnibus test (< 30 rounds): none.
Each match contributes at most one round to each regulation round-number cell (R1 to R24), so those cells are independent across matches. The pooled OT cell can hold several rounds from one match, so its test is optimistic.

**First half versus second half versus overtime** (overall CT rate 52.14% over 32041 rounds; Bonferroni alpha 0.01667 across 3 cells)

| cell | matches | rounds | CT wins | CT win rate | diff vs overall (pp) | binomial p vs overall | sig. after Bonferroni |
|---|---|---|---|---|---|---|---|
| 1st half R1-12 | 1591 | 18449 | 9688 | 52.51% | +0.38 | 0.31 |  |
| 2nd half R13-24 | 1525 | 12504 | 6445 | 51.54% | -0.59 | 0.19 |  |
| OT R25+ | 162 | 1088 | 572 | 52.57% | +0.44 | 0.78 |  |

Chi-square homogeneity over 3 cells with >= 30 rounds: chi2 2.89, df 2, p = 0.24; min expected count 520.8. Cells left out of the omnibus test (< 30 rounds): none.
Caveat: a match contributes up to 12 rounds to each half, so the half tests ignore within-match correlation and are optimistic.

**Pistol rounds (R1, R13) versus other rounds; CT rate, T rate = 1 - CT** (overall CT rate 52.14% over 32041 rounds; Bonferroni alpha 0.01667 across 3 cells)

| cell | matches | rounds | CT wins | CT win rate | diff vs overall (pp) | binomial p vs overall | sig. after Bonferroni |
|---|---|---|---|---|---|---|---|
| R1 pistol | 1559 | 1559 | 770 | 49.39% | -2.75 | 0.031 |  |
| R13 pistol | 1504 | 1504 | 712 | 47.34% | -4.80 | 0.0002 | yes |
| non-pistol | 1588 | 28978 | 15223 | 52.53% | +0.40 | 0.18 |  |

Chi-square homogeneity over 3 cells with >= 30 rounds: chi2 20.40, df 2, p = 3.7e-05; min expected count 719.9. Cells left out of the omnibus test (< 30 rounds): none.

**By map_name** (overall CT rate 52.14% over 32041 rounds; Bonferroni alpha 0.004545 across 11 cells)

| cell | matches | rounds | CT wins | CT win rate | diff vs overall (pp) | binomial p vs overall | sig. after Bonferroni |
|---|---|---|---|---|---|---|---|
| cs_office | 4 | 78 | 34 | 43.59% | -8.55 | 0.14 |  |
| de_ancient | 217 | 4477 | 2400 | 53.61% | +1.47 | 0.05 |  |
| de_anubis | 207 | 4221 | 2179 | 51.62% | -0.51 | 0.51 |  |
| de_boulder | 1 | 22 | 9 | 40.91% | -11.23 | 0.39 |  |
| de_cache | 139 | 2726 | 1402 | 51.43% | -0.71 | 0.47 |  |
| de_dust2 | 315 | 6248 | 3181 | 50.91% | -1.22 | 0.054 |  |
| de_inferno | 193 | 3931 | 1986 | 50.52% | -1.61 | 0.044 |  |
| de_mirage | 344 | 6870 | 3689 | 53.70% | +1.56 | 0.0098 |  |
| de_nuke | 135 | 2775 | 1441 | 51.93% | -0.21 | 0.83 |  |
| de_train | 16 | 265 | 144 | 54.34% | +2.20 | 0.5 |  |
| de_vertigo | 21 | 428 | 240 | 56.07% | +3.94 | 0.11 |  |

Chi-square homogeneity over 10 cells with >= 30 rounds: chi2 24.95, df 9, p = 0.003; min expected count 37.3. Cells left out of the omnibus test (< 30 rounds): de_boulder.
Match-level check, map_name: Kruskal-Wallis on per-match CT share over 9 groups (de_ancient 217 matches, de_anubis 207 matches, de_cache 139 matches, de_dust2 315 matches, de_inferno 193 matches, de_mirage 344 matches, de_nuke 135 matches, de_train 16 matches, de_vertigo 21 matches): H 13.04, p = 0.11. Groups with < 5 matches left out: {'de_boulder': 1, 'cs_office': 4}.

**By score difference going into the round (ct_score - t_score at live start, team-based, clipped to +/-4)** (overall CT rate 52.14% over 32038 rounds; Bonferroni alpha 0.005556 across 9 cells)

| cell | matches | rounds | CT wins | CT win rate | diff vs overall (pp) | binomial p vs overall | sig. after Bonferroni |
|---|---|---|---|---|---|---|---|
| <= -4 | 1166 | 4770 | 2207 | 46.27% | -5.87 | 5.1e-16 | yes |
| -3 | 1106 | 2310 | 1089 | 47.14% | -5.00 | 1.6e-06 | yes |
| -2 | 1259 | 3094 | 1526 | 49.32% | -2.82 | 0.0017 | yes |
| -1 | 1316 | 3772 | 1690 | 44.80% | -7.33 | 2e-19 | yes |
| 0 | 1584 | 4736 | 2446 | 51.65% | -0.49 | 0.5 |  |
| +1 | 1345 | 3721 | 2216 | 59.55% | +7.42 | 1e-19 | yes |
| +2 | 1292 | 2965 | 1640 | 55.31% | +3.17 | 0.00055 | yes |
| +3 | 1136 | 2220 | 1291 | 58.15% | +6.02 | 1.4e-08 | yes |
| >= +4 | 1097 | 4450 | 2599 | 58.40% | +6.27 | 5.1e-17 | yes |

Chi-square homogeneity over 9 cells with >= 30 rounds: chi2 376.75, df 8, p = 1.8e-76; min expected count 1062.5. Cells left out of the omnibus test (< 30 rounds): none.
Caveat: rounds within a match share the score history, so the omnibus chi-square is optimistic; the effect sizes are large enough that this should not change the reading.

**By match platform (header)** (overall CT rate 52.14% over 32041 rounds; Bonferroni alpha 0.01667 across 3 cells)

| cell | matches | rounds | CT wins | CT win rate | diff vs overall (pp) | binomial p vs overall | sig. after Bonferroni |
|---|---|---|---|---|---|---|---|
| faceit | 14 | 290 | 149 | 51.38% | -0.76 | 0.81 |  |
| steam | 511 | 9244 | 4811 | 52.04% | -0.09 | 0.86 |  |
| unknown | 1067 | 22507 | 11745 | 52.18% | +0.05 | 0.89 |  |

Chi-square homogeneity over 3 cells with >= 30 rounds: chi2 0.12, df 2, p = 0.94; min expected count 138.8. Cells left out of the omnibus test (< 30 rounds): none.
Match-level check, platform: Kruskal-Wallis on per-match CT share over 3 groups (faceit 14 matches, steam 511 matches, unknown 1067 matches): H 0.21, p = 0.9. Groups with < 5 matches left out: none.

**By match_type (header)** (overall CT rate 52.14% over 32041 rounds; Bonferroni alpha 0.025 across 2 cells)

| cell | matches | rounds | CT wins | CT win rate | diff vs overall (pp) | binomial p vs overall | sig. after Bonferroni |
|---|---|---|---|---|---|---|---|
| competitive | 525 | 9534 | 4960 | 52.02% | -0.11 | 0.83 |  |
| unknown | 1067 | 22507 | 11745 | 52.18% | +0.05 | 0.89 |  |

Chi-square homogeneity over 2 cells with >= 30 rounds: chi2 0.06, df 1, p = 0.8; min expected count 4563.3. Cells left out of the omnibus test (< 30 rounds): none.
Match-level check, match_type: Kruskal-Wallis on per-match CT share over 2 groups (competitive 525 matches, unknown 1067 matches): H 0.02, p = 0.89. Groups with < 5 matches left out: none.

### Platform and match_type are confounded with batch
| batch | platform | matches | rounds | CT wins | CT win rate |
|---|---|---|---|---|---|
| new_1008 | steam | 56 | 1007 | 514 | 51.04% |
| new_1008 | unknown | 1043 | 22079 | 11541 | 52.27% |
| old_1001_1007 | faceit | 14 | 290 | 149 | 51.38% |
| old_1001_1007 | steam | 452 | 8197 | 4280 | 52.21% |
| old_1001_1007 | unknown | 24 | 428 | 204 | 47.66% |
| other_dates | steam | 3 | 40 | 17 | 42.50% |

Within 2026-10-08 only (steam versus unknown platform):

**2026-10-08 only, by platform** (overall CT rate 52.22% over 23086 rounds; Bonferroni alpha 0.025 across 2 cells)

| cell | matches | rounds | CT wins | CT win rate | diff vs overall (pp) | binomial p vs overall | sig. after Bonferroni |
|---|---|---|---|---|---|---|---|
| steam | 56 | 1007 | 514 | 51.04% | -1.18 | 0.47 |  |
| unknown | 1043 | 22079 | 11541 | 52.27% | +0.05 | 0.88 |  |

Chi-square homogeneity over 2 cells with >= 30 rounds: chi2 0.53, df 1, p = 0.46; min expected count 481.2. Cells left out of the omnibus test (< 30 rounds): none.

Within 2026-10-01..10-07 only (steam versus faceit):

**2026-10-01..10-07 only, by platform** (overall CT rate 51.97% over 8915 rounds; Bonferroni alpha 0.01667 across 3 cells)

| cell | matches | rounds | CT wins | CT win rate | diff vs overall (pp) | binomial p vs overall | sig. after Bonferroni |
|---|---|---|---|---|---|---|---|
| faceit | 14 | 290 | 149 | 51.38% | -0.59 | 0.86 |  |
| steam | 452 | 8197 | 4280 | 52.21% | +0.25 | 0.66 |  |
| unknown | 24 | 428 | 204 | 47.66% | -4.31 | 0.081 |  |

Chi-square homogeneity over 3 cells with >= 30 rounds: chi2 3.42, df 2, p = 0.18; min expected count 139.3. Cells left out of the omnibus test (< 30 rounds): none.

## 4. Prior model: map and round number only

Match split (seed 0, test 0.2): train matches 1273, test matches 319; train rounds 25808, test rounds 6233; overlap 0 matches.
Train CT rate 52.03%; test CT rate 52.59%; (map x round) cells seen in train: 267.

**Prior model, test set (match-level 80/20, seed 0)**

| model | test log loss | test Brier | test AUC | log-loss skill vs constant |
|---|---|---|---|---|
| constant guess (train CT rate) | 0.6919 | 0.2494 | 0.500 | +0.00% |
| round number only (smoothed) | 0.6918 | 0.2493 | 0.513 | +0.01% |
| map only (smoothed) | 0.6923 | 0.2496 | 0.501 | -0.07% |
| map x round (smoothed, primary) | 0.6952 | 0.2510 | 0.504 | -0.47% |

Primary model in-sample (train) log loss 0.6873 vs test 0.6952 (gap +0.0078).
Predicted CT probability on test: min 0.362, max 0.653, sd 0.044.
Bootstrap over test matches (1000 resamples): log-loss difference model - constant = +0.0033 (95% CI +0.0008 to +0.0058); positive means the model is worse than the constant.
Bootstrap AUC minus 0.5: +0.0039 (95% CI -0.0119 to +0.0202).

Pseudo-count sensitivity (both levels set to the value shown; 20 is the primary setting):

| pseudo-count | test log loss | test Brier | test AUC |
|---|---|---|---|
| 5 | 0.6963 | 0.2515 | 0.503 |
| 20 | 0.6952 | 0.2510 | 0.504 |
| 100 | 0.6933 | 0.2501 | 0.506 |

Calibration on test, quintiles of predicted CT probability:

| quintile | rounds | mean predicted | observed CT rate |
|---|---|---|---|
| Q1 | 1232 | 0.454 | 0.513 |
| Q2 | 1246 | 0.500 | 0.551 |
| Q3 | 1244 | 0.522 | 0.502 |
| Q4 | 1252 | 0.542 | 0.520 |
| Q5 | 1259 | 0.575 | 0.544 |

Test rounds from 2026-10-01 to 10-07 only: n 1706; primary model log loss 0.6975, AUC 0.497; constant log loss 0.6934.

### Extension (not in the spec): pre-round score difference
Rounds with a live-start score: train 25805, test 6233 (rounds without one are dropped here only). Score difference = CT-team lead at live start, clipped to +/-4; the model is fitted on train only.

**Test set with a live-start score (same match split)**

| model | test log loss | test Brier | test AUC | log-loss skill vs constant |
|---|---|---|---|---|
| constant guess (train CT rate) | 0.6919 | 0.2494 | 0.500 | +0.00% |
| map x round (primary, spec) | 0.6952 | 0.2510 | 0.504 | -0.47% |
| score difference only (smoothed) | 0.6840 | 0.2454 | 0.572 | +1.14% |
| round x score difference (smoothed) | 0.6721 | 0.2401 | 0.593 | +2.86% |
- score difference only minus constant: log-loss difference -0.0079 (95% CI -0.0108 to -0.0052); AUC minus 0.5 +0.0721 (95% CI +0.0573 to +0.0887).
- round x score difference minus constant: log-loss difference -0.0199 (95% CI -0.0252 to -0.0150); AUC minus 0.5 +0.0935 (95% CI +0.0769 to +0.1104).
- score-only model in-sample (train) log loss 0.6869; predicted CT probability on test: min 0.450, max 0.594.
