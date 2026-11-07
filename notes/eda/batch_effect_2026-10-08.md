# Batch effect: 2026-10-08 vs 2026-10-01 to 10-07 (Task 2)

Scripts: `scripts\eda\batch_effect.py` (all rounds, per-day table, duplicate and metadata checks), `scripts\eda\batch_5v5.py` (5v5-only comparison, per-day 5v5, identifiability), `scripts\eda\dup_by_batch.py` (duplicates against the batch split). Read-only; nothing written to the tome.

## Definitions

- **Batch** = header `match_date` day. 2026-10-08: 1,133 matches. 2026-10-01 to 10-07: 615 matches. Four matches are dated outside both (2026-09-18 x1, 09-25 x1, 09-30 x2). They are excluded from the comparison.
- The `match_date` day equals the key-path day (`csds/YYYY/MM/DD/...`) for 1,730 of 1,752 matches. I used `match_date` as specified.
- **Channels**: `round_end` (rounds, winner, win reason, player count), `round_state` (live start = `round_freeze_end` row; live-start scores), `bomb_state` (plant counted only if it falls between live start and `round_end`, the model's definition), `player_death` (kill = valid attacker that is not the victim, using `*_fixed` ids), `player_info` (players per round, 5 T and 5 CT).
- **Two views**: all rounds in the channel, and a **5v5 subset** (`is_wingman` False and `player_count == 10`). The wingman mix differs strongly between batches (see below), so the 5v5 view is the fairer comparison of game behaviour.
- **Tests**: two-proportion z for shares (round level); Welch t for means (round level, and per-match where noted). Normal approximations, n large throughout. Round-level p-values assume independence across rounds, so they are optimistic. Per-match checks give the same conclusions.

## Summary

- **Winner base rate does not shift.** T share 47.6% vs 47.4% (p = 0.73) on all rounds; 47.8% vs 48.0% (p = 0.69) on 5v5.
- **Metadata differs sharply.** 92.7% of 10-08 matches have `match_type` and `platform` = `unknown`, against 3.9% of the other days (p about 3e-290). Wingman matches are 2.2% of 10-08 against 20% of the other days.
- **Outcome-side differences on 5v5 rounds are small**: plant rate +2.6 points, Target_Bombed share +2.6 points, plant time +3.5 s, kills per round -0.13 (-1.8%). Rounds per match are 15% higher in 10-08 (+2.8 rounds).
- **Batch identifiability from round-start features is weak**: AUC 0.565 (folds 0.547 to 0.578).

## Table 1: all rounds in the channels (before any 5v5 filter)

| metric | 2026-10-08 | 10-01 to 10-07 | difference | test |
|---|---|---|---|---|
| matches | 1,133 | 615 | | |
| rounds (round_end rows) | 24,472 | 11,437 | | |
| rounds per match, mean (median) | 21.60 (21) | 18.60 (19) | +3.00 | Welch on matches, p 9e-38 |
| winner T (2) share | 0.4762 | 0.4743 | +0.0019 | z 0.34, p 0.73 |
| winner CT (3) share | 0.5238 | 0.5257 | -0.0019 | z -0.34, p 0.73 |
| win 1 Target_Bombed | 0.0814 | 0.0498 | +0.0315 | z 10.8, p 3e-27 |
| win 7 Bomb_Defused | 0.1261 | 0.1091 | +0.0170 | z 4.6, p 4e-6 |
| win 8 CTs_Win | 0.3842 | 0.4002 | -0.0160 | z -2.9, p 0.004 |
| win 9 Terrorists_Win | 0.3946 | 0.4236 | -0.0290 | z -5.2, p 2e-7 |
| win 12 Target_Saved | 0.0134 | 0.0146 | -0.0012 | z -0.9, p 0.35 |
| win 17 / 18 (surrender) | 3 / 4 rounds | 20 / 8 rounds | | tiny counts |
| plant rate per round (plant in live..round_end) | 0.4919 (12,037) | 0.4157 (4,754) | +0.0762 | z 13.5, p 2e-41 |
| plant rate, any bomb_planted row (naive) | 0.5313 | 0.4603 | +0.0710 | not used |
| plant time from live start, s: mean (median) | 53.99 (49.72) | 49.98 (45.61) | +4.02 | Welch t 11.2, p 6e-29 |
| kills per round (valid attacker, not self) | 7.087 | 6.503 | +0.585 | Welch t 27, p 1e-166 |
| rounds with round_end player_count = 10 | 0.9434 | 0.7795 | +0.164 | z 46, p about 0 |
| player_info exactly 5 T and 5 CT | 0.9553 | 0.8004 | +0.155 | |
| `match_type` = unknown (match level) | 1,050 / 1,133 = 0.9267 | 24 / 615 = 0.0390 | +0.888 | z 36, p 3e-290 |
| `platform` = unknown (match level) | 1,050 / 1,133 = 0.9267 | 24 / 615 = 0.0390 | +0.888 | same counts |

Other platforms: 10-08 is `steam` (83 matches) only. The other days are `steam` (577) and `faceit` (14).
Wingman (`is_wingman` True): 25 matches in 10-08 (2.2%), 123 in the other days (20.0%).

Player count shares, 10-08: 10 (94.3%), 9 (2.6%), 4 (1.3%), 11 (0.9%), other (0.9%). Other days: 10 (78.0%), 4 (14.3%), 9 (6.4%), other (1.3%). The 4-player rounds are wingman rounds.

## Table 2: 5v5 only (`is_wingman` False, `player_count` = 10)

| metric | 2026-10-08 | 10-01 to 10-07 | difference | test |
|---|---|---|---|---|
| matches (with 5v5 rounds) | 1,099 | 490 | | |
| rounds | 23,086 | 8,915 | | |
| rounds per match | 21.01 | 18.19 | +2.81 | Welch on matches, p 1e-22 |
| winner T share | 0.4778 | 0.4803 | -0.0025 | z -0.40, p 0.69 |
| plant rate (in live..round_end) | 0.4974 | 0.4717 | +0.0257 | z 4.1, p 4e-5 |
| plant time from live start, s | 53.99 | 50.47 | +3.52 | Welch t 9.3, p 1e-20 |
| win 1 Target_Bombed | 0.0839 | 0.0580 | +0.0259 | z 7.8, p 6e-15 |
| win 7 Bomb_Defused | 0.1257 | 0.1226 | +0.0031 | z 0.75, p 0.45 |
| win 8 CTs_Win | 0.3828 | 0.3805 | +0.0023 | |
| win 9 Terrorists_Win | 0.3938 | 0.4222 | -0.0284 | |
| kills per round | 7.159 | 7.293 | -0.134 | Welch t -8.8, p 1e-18 |
| `match_type` unknown (5v5 matches) | 1,050 / 1,108 = 0.948 | 24 / 492 = 0.049 | | |

Per day, 5v5 only (round-level shares):

| day | matches | rounds | T share | plant rate (in window) |
|---|---|---|---|---|
| 2026-10-01 | 56 | 957 | 0.4775 | 0.4608 |
| 2026-10-02 | 53 | 942 | 0.4958 | 0.4936 |
| 2026-10-03 | 90 | 1,627 | 0.4573 | 0.4561 |
| 2026-10-04 | 80 | 1,556 | 0.4826 | 0.4891 |
| 2026-10-05 | 61 | 1,126 | 0.4876 | 0.4663 |
| 2026-10-06 | 83 | 1,502 | 0.4847 | 0.4680 |
| 2026-10-07 | 67 | 1,205 | 0.4863 | 0.4714 |
| **2026-10-08** | **1,099** | **23,086** | **0.4778** | **0.4974** |

Read-across: the T share on 10-08 sits inside the range of the other days (0.457 to 0.496). The plant rate on 10-08 (0.497) is just above the range of the other days (0.456 to 0.494), so the plant gap is real but modest.

The all-rounds plant gap (+7.6 points, Table 1) is mostly the wingman mix. Within 5v5 it is +2.6 points.

## Identifiability: can a model tell the batch from round-start state?

- Features: round number, `max_rounds`, live-start T and CT scores, pistol flag, overtime flag. Labels not used. Death-based alive counts, plant state and player count are not included.
- 5v5 rounds with live-start scores: 31,998 (72% from 10-08). HistGradientBoosting, 100 iterations, 5-fold CV grouped by match. Threads pinned to 1.
- Batch AUC by fold: 0.547, 0.578, 0.563, 0.565, 0.573. Mean 0.565.
- Reading: weak signal. The likely source is the round-number distribution (10-08 matches run longer, so more late rounds). That is not tested by ablation, so it is a hypothesis.

## Duplicates and the batch split

- All-match duplicate groups (47 groups, 99 member matches) by batch: **69 members in 10-08 (6.1% of its matches)** and 30 in the other days (4.9%). The rates differ slightly; the difference is small.
- No duplicate group spans the two batches. Every member of every group shares the same `match_date` and the same key-path day. Duplicates are same-day copies, so the batch split cannot put copies on both sides.
- The dedupe keeps the lowest key string in each group. Within a group that is the lowest uuid among same-day copies. Kept copies: 32 in 10-08, 15 in the other days. Dropped copies: 37 in 10-08, 15 in the other days.

## Verdict

**Does the 2026-10-08 batch look different enough that the model could learn the batch rather than the game?**

No, not from the channels the model uses. The batch differs in metadata and in match mix, but the winner base rate is the same (47.8% vs 48.0% T on 5v5 rounds). The outcome-side differences are statistically detectable but small: plant rate +2.6 points, Target_Bombed +2.6 points, plant time +3.5 s, kills per round -0.13. Round-start state identifies the batch only weakly (AUC 0.57).

Two caveats keep this from being a clean "no":

1. **Metadata is strongly batch-specific.** 93% of 10-08 matches are `unknown` for match type and platform, against 4% of the other days, and 10-08 has almost no wingman matches. The current 10 model features do not use metadata, so the model cannot read the batch directly through it. Adding those fields would make the batch trivially identifiable.
2. **10-08 is most of the training data.** It is 72% of the 5v5 rounds, and the primary split is random across matches. A random validation split therefore mostly tests within-batch error and cannot show batch-specific bias. The check that answers the question directly is to train on 10-01 to 10-07 and test on 10-08, with the death and plant state features included. I did not run that.

## Duplicate count reconciliation

Current dedupe step: `scripts\model\bakeoff.py`, section 2 ("duplicate matches", lines about 636 to 663). Groups are matches with the same sorted event signature (`round_end`, `player_death`, `player_info`), formed over the **channel-kept** matches (1,709 of 1,752). Within each group the lowest `match_key` string is kept and the rest are dropped.

Replicated from the tome with `scripts\eda\dedupe_replicate.py`, using the same channel filter, signature and split code. The replication matches `logs\bakeoff.log` exactly.

- **Duplicate groups (current dedupe step): 46.** Sizes: 41 pairs and 5 triples. 97 member matches.
- **Matches removed by the current dedupe step: 51** (97 minus 46, lowest key kept).
- Of those 51, **47 still had surviving rounds** (913 rounds removed). 4 had no surviving round, so they did not affect the rounds table. The log prints the 47 as "duplicate matches removed (with kept rounds)".
- **Groups crossing the split** (computed before removal, on matches with surviving rounds): **17 for primary, 0 for chronological.**

Where the other numbers come from:

- **47 duplicate groups** is the count over **all 1,752 matches with no channel filter**. The round-fingerprint diagnostic in `bakeoff.py` (`>=5` rounds, all matches) gives the same 47 groups with identical member sets (99 matches, 52 above the group minimum). `notes\model\bakeoff_results.md` reports this as "found 47 groups" next to "46 groups (97 matches)" for the dedupe.
- The channel filter takes it from 47 to 46. One 2-member group has both members missing `team_change`, which is a dense channel in the bakeoff rule, so the group is dropped. That gives 47 to 46 groups, 99 to 97 members, 52 to 51 removable.
- The "47" in the log "with kept rounds" line is a different number from the 47 fingerprint groups. It is a count of removed matches that still have rounds.

Log timing: `logs\bakeoff.log` was written at 6:58:40 PM. `scripts\model\bakeoff.py` was last modified at 7:00:30 PM. The replication under the current code gives the same numbers as the log, so the dedupe counts above hold. I did not identify what the later edit changed.

**Answer:** the current dedupe step forms **46 duplicate groups** over channel-kept matches, **removes 51 matches** (keeping the lowest key in each group), and 47 of those removed matches had rounds in the training table. The 47 groups and 51 removed matches are not the same measure: 47 is the all-match count before the channel filter, and 51 is the match-level removal after it.
