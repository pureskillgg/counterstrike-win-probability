# Event-only feature spec and validation: CS2 round win probability

Date: 2026-10-12. Data: `tomes/2026-10-01_to_08/tome/csds` (basic_* channels and header). No per-tick channels. Script: `scripts/model/feature_validation.py`. Full run log: `logs/feature_validation.log`. All numbers below come from that run. Ticks are used throughout; 64 ticks = 1 second (`second` = tick / 64 in every channel checked).

## 1. Validation numbers (headline)

- Sample: 172 matches, 4,600 rounds. The request was 200 matches, but only 172 are eligible (>= 24 rounds and every round player_count = 10). All 172 were used, so the seed has no effect.
- States: 4,600 live-start states + 32,995 death states = 37,595 states.
- (a) alive count negative: 0 rounds, 0 states.
- (b) side at 0 alive wins the round: 0 unexplained. 36 rounds have T at 0 alive and T winning after a bomb plant (bomb path, see section 4).
- (c) winner has >= 1 survivor at the last state: 36 failures, the same 36 rounds (34 Target_Bombed, 2 Terrorists_Win). 0 failures for CT winners.
- (d) bomb_planted never drops within a round: 0 failures. 0 rounds have more than one plant.
- Start check: 4,581 of 4,600 rounds have exactly 5 T and 5 CT in basic_player_info. All 36 failures are in those 4,581 rounds.

## 2. Round lifecycle (task 1)

`basic_round_state` has 21 distinct event_type values, 531,365 rows:

| event_type | rows | matches |
|---|---:|---:|
| cs_round_start_beep | 108,130 | 1,751 |
| round_start | 69,831 | 1,752 |
| round_end | 35,992 | 1,752 |
| freezetime_ended_inferred | 35,953 | 1,751 |
| round_freeze_end | 35,953 | 1,751 |
| cs_round_final_beep | 35,953 | 1,751 |
| buytime_ended | 35,244 | 1,734 |
| round_poststart | 34,896 | 1,734 |
| cs_pre_restart | 34,240 | 1,751 |
| round_officially_ended | 33,839 | 1,734 |
| round_prestart | 33,839 | 1,734 |
| bomb_planted | 18,306 | 1,722 |
| bomb_defused | 4,337 | 1,478 |
| bomb_exploded | 3,090 | 1,272 |
| round_announce_match_point | 2,545 | 1,581 |
| announce_phase_end | 2,166 | 1,734 |
| round_announce_last_round_half | 1,952 | 1,736 |
| cs_win_panel_match | 1,752 | 1,752 |
| round_announce_match_start | 1,750 | 1,750 |
| begin_new_match | 1,074 | 1,074 |
| round_announce_final | 523 | 402 |

- **Live start = `round_freeze_end`** (one row per round, never repeated). **Round end = `basic_round_end`** (one row per round, 35,992 rounds in 1,752 matches).
- **Rounds with a live start event: 35,953 of 35,992.** The 39 rounds without one are outside the eligible sample.
- In every round with a live start: start_tick <= live_tick < end_tick.
- `freezetime_ended_inferred` is within 1 s of `round_freeze_end` in 35,858 of 35,953 rounds and never has the same tick. It is not used.
- `round_start` appears twice per round in round_state (69,831 rows), so it is not a usable marker. The `basic_round_start` channel has exactly one row per round.
- Round numbers run 1..N with no gaps in all 1,752 matches. Matches end early when one team reaches the win threshold (max_rounds//2 + 1), so a match with fewer than 24 rounds is normal, not truncated. Correction 2026-10-12: the earlier "1,391 of 1,752 truncated" wording was wrong (see the change log).

## 3. Label and side codes (verified, not just assumed)

- **Label: `ct_win = (winner_team_code == 3)`.** Across all 35,992 rounds: code 2 = 17,127, code 3 = 18,865. In the sample: 2,439 of 4,600 (53.0%) are CT wins.
- **2 = T, 3 = CT** is confirmed. 18,278 of 18,306 bomb_planted rows have planter team code 2 (28 are null). bomb_defused planters are code 3.
- **Win reason map** (win_reason_code -> winner): 1 Target_Bombed -> 2 (2,566); 7 Bomb_Defused -> 3 (4,337); 8 CTs_Win -> 3 (14,006); 9 Terrorists_Win -> 2 (14,547); 12 Target_Saved -> 3 (497; this confirms the GUESS); 17 Terrorists_Surrender -> 3 (23); 18 CTs_Surrender -> 2 (12); 11 and 13 are hostage codes (2 rounds each).
- **Team codes are sides.** The score fields swap with the sides at halftime. In every round-to-round transition (34,200 total), the score pair either increments for the winner (29,014) or swaps and increments (1,667). The rest fit both patterns (3,519). Swaps occur after round 8 (122, 16-round matches), round 12 (1,329, 24-round matches), and in overtime after rounds 27, 33, 39 and 45 (163, 37, 13, 3).
- `t_score + ct_score = round - 1` at every live start (35,953 of 35,953). The score fields are a valid pre-round input.

## 4. Sample points and validation (tasks 2 and 3)

**Sample construction**
- Eligible: 1,752 matches -> 361 with >= 24 rounds -> 172 of those with every round player_count = 10.
- Start: 5 T and 5 CT at round_freeze_end (the spec). The player_info split is exact in 4,581 of 4,600 rounds. The other 19 rounds have 9, 11 or 14 player_info rows (5 vs 4, 5 vs 6, 6 vs 5, or 7 vs 7). 0 failures occur in those 19 rounds.
- A state is each live-start point, plus each basic_player_death with tick in [round_freeze_end, round_end]. Victim side = player_team_code, else basic_player_info team for (match, round, player_id).
- Deaths excluded from states: 60 before round_freeze_end (all weapon 'world'; 24 of those victims fire a weapon later in the same round, so they are treated as non-events); 523 after round_end.
- Null victim team in window: 1, imputed from player_info. Across the full tome, all 252,742 non-null death teams agree with player_info.
- Repeat victim in window: 1 (round a96124a5 r12, T victim logged at tick 78896 and again at 80841 with a null team). The spec rule keeps the first death only.
- Bomb plants: 2,458 in the sample. 2,304 are inside the live window and count. 154 are after round_end and are excluded. 0 are before live. 0 rounds have more than one plant.

**Checks** (decrement the victim side on each counted death)

| Check | Failures (rounds) | Notes | Examples (match, round, tick: alive T / CT, winner, reason) |
|---|---:|---|---|
| (a) alive count < 0 | 0 | 0 states | none |
| (b) side at 0 alive is the winner, not via a plant | 0 | all zero-side wins are the bomb path below | none |
| (b) bomb path: T at 0 alive after plant, T wins | 36 | 34 Target_Bombed with bomb_exploded in window; 2 Terrorists_Win (see c) | 6b568fc6 r21 tick 115752: 0/1, Target_Bombed. 3849a14e r21 tick 114674: 0/1, Terrorists_Win. 7660b247 r23 tick 131184: 0/1, Target_Bombed |
| (c) winner has no survivor at last state | 36 | 34 Target_Bombed; 2 Terrorists_Win; 0 CT winners | 6b568fc6 r21 tick 115752: 0/1, Target_Bombed. 3849a14e r21 tick 115761: 0/0, Terrorists_Win (last kill 'inferno', attacker id equals victim). 005a6b2a r30 tick 210719: 0/0, Terrorists_Win (last kill 'inferno') |
| (d) bomb_planted flag drops within a round | 0 | 0 rounds with > 1 plant | none |

- Both Terrorists_Win failures have a molotov (inferno) as the last in-window kill after the last T died. The T molotov probably kept burning after its thrower died. This is an inference from events alone. For 3849a14e r21 the attacker field is self-referencing, which suggests attribution noise.
- The bomb path is a real game rule, not a model defect. With a planted bomb, T can win after all T are dead by exploding the bomb, so "the side at 0 alive loses" (b) and "the winner has a survivor" (c) need that exception in any model that uses them.

**Caveats**
- 29 rounds have an in-window player_disconnect or bot_takeover. A disconnect is not a death, so alive counts can be too high in those rounds.

## 5. Final feature list (task 4)

Label: `ct_win` (round level, repeated on every state of its round).

| feature | definition | source (event fields) |
|---|---|---|
| alive_ct | 5 minus CT deaths in window so far | basic_round_state live start; basic_player_death player_team_code (fallback basic_player_info) |
| alive_t | 5 minus T deaths in window so far | same |
| ct_alive_minus_t | alive_ct - alive_t | derived |
| bomb_planted | 1 if a bomb_planted row has tick <= state tick and tick in the live window | basic_bomb_state |
| seconds_since_round_live | (state tick - round_freeze_end tick) / 64 | basic_round_state, tick |
| seconds_since_plant | (state tick - plant tick) / 64 if bomb_planted, else -1 | basic_bomb_state, tick |
| round_number | round (1-based) | basic_round_end |
| ct_score_minus_t_score | CT score - T score at live start | round_freeze_end row of basic_round_state (side-attached, see section 3) |
| is_pistol_round | round in {1, max_rounds/2 + 1} | round; max_rounds from header |
| is_overtime | round > max_rounds from header | round; header |

- **Leakage rule:** use only rows with tick <= state tick. Do not use bomb_defused, bomb_exploded, anything in basic_round_end (winner, win_reason), or any event after the state. Plants after round_end are never used.
- Quantities the spec does not use (equipment, money) are excluded by design; see section 6.

## 6. Quantities that cannot be derived without per-tick data

- **None of the seven required quantities** (alive_ct, alive_t, ct_alive_minus_t, bomb_planted, seconds_since_round_live, seconds_since_plant, round_number) needs per-tick data. All come from event rows and their tick stamps. Their assumptions are: 5 vs 5 at live start (exact in 4,581 of 4,600 rounds) and death rows being complete (not true for 29 rounds with disconnect or takeover).
- **Cannot be derived as a continuous state:**
  - Player positions, velocities, distances to the bomb site, line of sight. Positions exist only as merged values on event rows (death, hurt, bomb, etc.), not at each state tick.
  - Money, economy, buy decisions. No money channel. buytime_ended only marks the end of buy time.
  - Full per-player equipment and utility at a state. item_* events are partial histories. Excluded here.
  - Per-player health and armor at a state. basic_player_hurt gives the health value at hit events only. Excluded from v1.
  - In-flight grenade and molotov positions, and active smoke or fire extents. grenade_vector and per-tick molotov data are not in these tomes.
  - Bomb world position and carrier at a state. Not continuous.
  - Presence of a disconnected or taken-over player. These are not deaths.

## 7. Open items

- The validation pool is 172 matches, not 200. The 200 target needs more eligible matches from other days or looser eligibility.
- Corrected 2026-10-12: short matches are normal, not truncated. A match ends when one team reaches the win threshold (max_rounds//2 + 1), so a match can have fewer rounds than max_rounds with no data loss. Per `notes/model/bakeoff_results.md` section 6, 1,350 of 1,752 matches end before max_rounds; 1,324 of them end at the win threshold, and only 26 look incomplete. The earlier claim (1,391 matches with fewer than 24 rounds, read as truncation) is withdrawn: 24 is not the right bound, since all 149 matches in the 16-round format are below 24 by construction.
  - Change log 2026-10-12: this bullet replaced the 1,391 / truncation wording. No other section was changed by this edit.
- Consider de-duplicating deaths per victim per round in the parser output. One such case exists here.
- Consider whether to treat the 154 post-round_end plants as data errors. They are excluded from bomb_planted.
