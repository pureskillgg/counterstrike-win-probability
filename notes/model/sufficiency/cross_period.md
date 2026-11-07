# CHECK C: cross-period transfer (2026 kept set to 2024-02 tome)

Run: 2026-10-13 15:01:54. Script: `scripts/model/sufficiency/cross_period.py` (seed 0, bake-off XGBoost settings: hist, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, early stopping 100, monotone MONO). Device: A=cuda, B=cuda.

**Verdict:** Partial transfer: 2024 log loss 0.50441 vs 2026 in-period 0.43660 (gap +0.06781 exceeds 0.02), better than the constant baseline 0.69363. 2024 round_end winners UNVERIFIED.

## 1. Counts (label VERIFIED)

| Set | Matches | Rounds | States | CT-win share (states) | Label |
|---|---:|---:|---:|---:|---|
| 2026 kept set (bake-off filters; reproduces logs/bakeoff.log) | 1521 | 28683 | 234265 | - | VERIFIED |
| 2026 A train = primary train + test (92%) | 1399 | 26319 | 215095 | 0.5185 | VERIFIED |
| 2026 A early-stop slice = primary val (8%) | 122 | 2364 | 19170 | 0.5190 | VERIFIED |
| 2026 B train = primary train (72%) | 1095 | 20470 | 167379 | 0.5214 | VERIFIED |
| 2026 B val = primary val (8%) | 122 | 2364 | 19170 | 0.5190 | VERIFIED |
| 2026 B test = primary test (20%, in-period) | 304 | 5849 | 47716 | 0.5084 | VERIFIED |
| 2024-02 kept set (relaxed filter; test for A) | 3299 | 53816 | 447555 | 0.5027 | VERIFIED (winner labels UNVERIFIED) |

2024 header matches: 4435; 2026 header matches: 1752.

## 2. Metrics (states, bake-off convention; label VERIFIED)

| Model | Set | Matches | Rounds | States | Log loss | Brier | AUC | ECE (10 bins) | Accuracy (ref) | Label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| XGBoost (A) | 2024-02 relaxed kept set, all states | 3299 | 53816 | 447555 | 0.50441 | 0.16365 | 0.8401 | 0.0270 | 0.7482 | VERIFIED |
| XGBoost (A) | 2024-02 relaxed kept set, live-start states | 3299 | 53816 | 53816 | 0.67834 | 0.24281 | 0.5900 | 0.0189 | 0.5596 | VERIFIED |
| Constant (A train rate) | 2024-02 relaxed kept set, all states | 3299 | 53816 | 447555 | 0.69363 | 0.25024 | 0.5000 | 0.0158 | 0.5027 | VERIFIED |
| LR ct_alive_minus_t (A) | 2024-02 relaxed kept set, all states | 3299 | 53816 | 447555 | 0.51651 | 0.17169 | 0.8190 | 0.0300 | 0.7300 | VERIFIED |
| XGBoost (B) | 2026 primary test (in-period, 20%), all states | 304 | 5849 | 47716 | 0.43660 | 0.14637 | 0.8703 | 0.0089 | 0.7697 | VERIFIED |
| XGBoost (B) | 2026 primary test (in-period, 20%), live-start states | 304 | 5849 | 5849 | 0.68102 | 0.24417 | 0.5696 | 0.0178 | 0.5524 | VERIFIED |
| Constant (B train rate) | 2026 primary test, all states | 304 | 5849 | 47716 | 0.69334 | 0.25010 | 0.5000 | 0.0130 | 0.5084 | VERIFIED |
| LR ct_alive_minus_t (B) | 2026 primary test, all states | 304 | 5849 | 47716 | 0.46093 | 0.15321 | 0.8524 | 0.0102 | 0.7595 | VERIFIED |
| XGBoost (A) | 2026 early-stop slice (8%; used for stopping, not clean) | 122 | 2364 | 19170 | 0.43549 | 0.14565 | 0.8714 | 0.0117 | 0.7714 | VERIFIED |

Reference: the bake-off's XGBoost primary test row (`notes/model/bakeoff_results.md`) is log loss 0.4366 on 47716 states; Model B on the same split gives 0.43660 on 47716 states.

## 3. 2024 filter and data checks (label VERIFIED unless marked)

| Channel | Kind | Rows | Matches with rows | Keyset matches | Zero-row matches | Counted missing |
|---|---|---:|---:|---:|---:|---:|
| round_end | dense | 83527 | 4435 | 4435 | 0 | 0 |
| round_start | dense | 81020 | 4434 | 4435 | 1 | 1 |
| round_state | dense | 1009532 | 4435 | 4435 | 0 | 0 |
| player_death | dense | 579124 | 4435 | 4435 | 0 | 0 |
| player_info | dense | 781897 | 4435 | 4435 | 0 | 0 |
| bomb_state | dense | 56009 | 4222 | 4435 | 213 | 213 |
| player_disconnect | sparse | 20898 | 4249 | 4435 | 186 | 0 |
| bot_takeover | sparse | 0 | 0 | 4435 | 4435 | 0 |

team_change: no folder in the 2024 tome, so not required (relaxed). Relaxed round rules: player_count == 10 not required; player_info must show 5 T and 5 CT by unique player id; disconnect or bot takeover drops the round; then round_start, live start and score checks as in the bake-off.

| Stage | Matches | Rounds |
|---|---:|---:|
| 2024 header | 4435 | 83527 (round_end rows) |
| dropped: missing a required channel | 214 | - |
| rounds in channel-kept matches | 4221 | 80236 |
| dropped round: not exactly 5 T and 5 CT at start (player_info, unique player ids) | - | 14960 |
| dropped round: disconnect or bot takeover event in round | - | 4784 |
| dropped round: duplicate copy of a kept match (identical events) | - | 4559 |
| dropped round: death with unresolved victim side in live window | - | 1976 |
| dropped round: no round_start row | - | 141 |
| **kept (used)** | **3299** | **53816** |

States: 447555 (53816 live starts, 393739 death states). Kept rounds with total unique players not 10: 0. Kept rounds where round_end player_count is not 10 (not filtered): 53816.

- **max_rounds (2024, inferred, not in header):** halftime announcement rule 2 x round + 2. The same rule matches the 2026 header in 1736 of 1736 matches with the event (VERIFIED on 2026). 2024 source for kept matches: {'halftime announcement': 53452, 'default 24 (no halftime announcement)': 364}. max_rounds only drives is_pistol_round and is_overtime.
- **Score transitions (partial evidence only, 2024 winners UNVERIFIED):** {'consistent': 76370, 'no_next_round_with_freeze': 4440, 'inconsistent': 2572} across all 2024 matches; inconsistent rounds by (round, max_rounds, swap): [((9, 16, False), 292), ((13, 24, False), 2051), ((25, 24, False), 124), ((28, 24, False), 105)].
- **Completion (round_end count vs header final score sum):** {'n_rounds_equals_header_final_sum': 1318, 'not_completed_overall': 279, 'n_rounds_below_header_final_sum': 3117, 'matches_below_max_rounds': 3847, 'below_max_rounds_completed_by_threshold': 3655, 'below_max_rounds_not_completed': 192}.
- **Duplicates:** 202 groups, 271 copies removed inside 2024; 0 kept 2024 matches have identical events to a kept 2026 match.

## 4. Caveats and failures

- 2024 round_end winners are UNVERIFIED. Results on 2024 depend on them.
- 2024 has no max_rounds in its header; inferred as above. Matches with no halftime announcement (128 matches, all short) use 24 by default.
- Model A and Model B differ in training size (92% vs 72% of 2026 matches); the 2026 in-period comparison is therefore not an exact control.
- Early-stop slices are match-level; the 2026 early-stop row is optimistic and not a clean test.
- No tuning, no calibration; the same bake-off settings as the 2026 primary run. Metrics are on states (long rounds weigh more); the live-start rows give one state per round.
- The bot_takeover page files in the 2024 tome have 0 rows and no columns; treated as empty.

