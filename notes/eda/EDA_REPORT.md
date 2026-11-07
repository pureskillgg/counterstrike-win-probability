# EDA report: csds-cs2 tome 2026-10-01 to 10-08

Copied from Meridian note `61bd3a5d-08f8-4cb8-84db-a2fac61885bd` (slug `eda-report-csds-cs2-tome-2026-10-01-to-10-08`), written 2026-10-13 by a consolidation agent. The agent could not write this file itself, so the text was saved here from the note.

Labels as in the source: **VERIFIED** = re-run or read directly in that pass. **UNVERIFIED** = taken from an earlier note in `notes\eda` and not re-run.

Tome: `G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds`

## Reconciliation (added by the lead session, 2026-10-13)

**Score attachment conflicts. Treat the note's section 3.1 as unresolved.**

- Section 3.1 says round scores are team-attached in all 34,200 round transitions. It also says a side-cumulative reading matches only 61% of freeze rows.
- A direct check, `scripts\eda\score_semantics_by_day.py`, tests round 12 to 13 score transitions in non-tied matches. Side-attached fits the majority every day. Team-attached fits a minority.

| Day | Non-tied matches | Side-only | Team-only | Neither |
|---|---|---|---|---|
| 01 | 63 | 43 | 11 | 0 |
| 02 | 61 | 42 | 9 | 0 |
| 03 | 94 | 78 | 3 | 0 |
| 04 | 99 | 71 | 15 | 0 |
| 05 | 76 | 53 | 8 | 0 |
| 06 | 99 | 66 | 14 | 0 |
| 07 | 82 | 60 | 13 | 0 |
| 08 | 1,121 | 916 | 13 | 0 |

- This check is one transition, so it cannot settle the question alone. A per-round check against the round winners is needed before any feature is changed.
- `HANDOFF.md` and `feature_spec.md` currently say side-attached with a team-attached minority. That matches the direct check above.

**Other items to verify:** the note's section 3.9 (rank_update timing) and 3.3 (post-end plants) are marked VERIFIED in the source. Re-check them before relying on them.

---

## 1. Data inventory

- 1,752 header matches, 1,752 keys, 35,992 rounds in `round_end`, no duplicate (match, round). VERIFIED.
- `match_date`: 2026-10-01 to 10-07 = 615 matches; 2026-10-08 = 1,133; 4 outside (09-18 x1, 09-25 x1, 09-30 x2). VERIFIED.
- Wingman 149 (max_rounds 16); non-wingman 1,603 (max_rounds 24). VERIFIED.
- `match_type`: unknown 1,074 (1,050 on 10-08), competitive 678. Platform: unknown 1,074, steam 664, faceit 14. VERIFIED.
- Maps in header: de_mirage 346, de_dust2 316, de_inferno 290, de_ancient 218, de_anubis 207, de_nuke 165, de_cache 141, de_vertigo 32, de_train 16, de_overpass 8, de_debris 4, cs_office 4, de_poseidon 3, de_eldorado 1, de_boulder 1. VERIFIED.
- Per-tick data: none in the tome. Tome channels are event rows stamped with tick, or per-round / per-player-per-round rows. Per-tick player snapshots (`player_status`, `player_vector`) exist only in the E: raw store. VERIFIED (folder listing). The economy note says `player_status` is incomplete at some ticks (buy-end tick incomplete in 681 of 1,045 rounds). UNVERIFIED.
- Per-round: `round_start`, `round_end` (one per round), `round_mvp`, `score_update` (1 to 3 per round), `player_info` and `player_personal` (about 10 per round). Event rows: all death, hurt, shot, grenade, blind, footstep, bomb, team_change, bot_takeover, chat, sound, spawn, connect, disconnect, name, rank_update.
- Row counts VERIFIED this pass: `weapon_fire` 5,483,781; `weapon_action` 534,502; `grenade_state` 531,031; `grenade_bounce` 1,236,044; `molotov_state` 208,908; `molotov_fire` 3,278,394; `rank_update` 5,873; `player_sound` 315,382; `player_chat` 19,586; `player_name` 507; `player_connect` 1,827; `item_equip` 9,585,945; `item_pickup` 2,084,041; `item_dropped` 944,718; `item_refund` 62,193.
- Row counts from earlier notes (UNVERIFIED): `round_state` 531,365; `bomb_state` 25,733; `bomb_action` 163,451; `bomb_defuse` 7,148; `round_mvp` 35,944; `score_update` 39,330; `player_death` 253,480; `player_hurt` 1,007,846; `other_death` 69,629; `bullet_damage` 289,611; `player_blind` 244,469; `player_footstep` 1,333,987; `player_info` 347,243; `player_personal` 347,278; `player_spawn` 331,291; `player_disconnect` 11,594; `team_change` 19,302; `bot_takeover` 983.
- Batch: 10-08 is 66% of matches and 72% of clean 5v5 rounds (23,086 of 32,041). UNVERIFIED.
- Economy and weapons, from JSON only (no markdown report exists for either):
  - `economy.json` covers `rank_update` only: 5,873 rows, 677 matches with rows, 1,075 zero-row matches. Sentinel 9999 in `rank_old` (20 rows) and `rank_new` (24). `rank_change` differs from `rank_new` minus `rank_old` in 260 rows. Position fields are 73.6% null. VERIFIED.
  - `weapons.json` covers six channels (`weapon_fire`, `weapon_action`, `grenade_state`, `grenade_bounce`, `molotov_state`, `molotov_fire`). VERIFIED.
- Channels not profiled by any report: `item_dropped`, `item_equip`, `item_pickup`, `item_refund`. Row counts VERIFIED; no content checked.

## 2. Coverage and gaps

- `bullet_damage`: zero rows in 1,075 matches (61%), 1,050 of them on 10-08. UNVERIFIED.
- `other_death`: zero rows in 370 matches, including all 316 de_dust2 matches (map-level gap). UNVERIFIED.
- `bomb_state` zero in 30 matches (their 379 rounds carry no plant data). `bomb_defuse` zero in 203. UNVERIFIED.
- `player_sound`: 20 matches only, all on 10-08 (315,382 rows). `player_blind` zero in 21 (19 on 10-08). `player_footstep` zero in 18 (17 on 10-08). `bot_takeover` covers 112 matches (106 on 10-08). `team_change` zero in 17. UNVERIFIED.
- One match (`csds/2026/10/07/ab7fe176-...`) has zero rows in all six weapons channels. VERIFIED.
- Wingman matches are 149 of 1,752; 4-player rounds in those. UNVERIFIED.

## 3. Goofy columns, ranked by potential to mislead a model

1. **Score attachment (conflicted, see reconciliation above).** The note says team-attached. Feature spec and schema say side-attached. Halves are 12 rounds, not 24. A side-cumulative rebuild would give wrong scores after round 12. UNVERIFIED.
2. **Wingman and partial rosters inside the 5v5 tome.** `round_end` player_count != 10 in 3,951 rounds (11.0%); 1,380 of them 9-player. VERIFIED. Bake-off drops 3,663 (UNVERIFIED).
3. **`bomb_planted` logged after `round_end`** in 1,479 rows (8.1%): 1,455 in code-9 rounds, 24 in code-12. VERIFIED. Naive plant rate 50.9% vs 46.8% when windowed. Rates UNVERIFIED.
4. **`round_start` tick = 66 sentinel:** 678 rows, all round 1, i.e. 38.7% of matches. VERIFIED. The 249 rounds over 200 s are not explained by it. UNVERIFIED.
5. **Batch 10-08 metadata:** 92.7% unknown `match_type` and platform, against 3.9% on other days; wingman 2.2% vs 20%. 5v5 plant rate +2.6 pts; kills per round flip sign with wingman mix (all rounds +0.59 on 10-08; 5v5 -0.13). UNVERIFIED.
6. **Raw `player_id`** differs from `player_id_fixed` in about 91% of rows in every channel. Cross-match keys are invalid. Disputed: the flow note calls raw 0 a sentinel in deaths, but `player_info` gives raw 0 in 10.0% of rows (other slots about 10% each), so 0 may be an ordinary slot. The 27 rounds with two or more raw-0 deaths are the main evidence for sentinel behaviour. The combat note says fixed ids are "1-10" but its own table shows 1 to 19. UNVERIFIED.
7. **Duplicate matches.** Bake-off: 46 groups, 97 matches (channel-filtered); 47 groups, 99 matches (all matches); `player_info` 47 groups; `team_change` 46 groups. 17 groups cross the primary split before removal; 0 for chronological. All copies are same-day. UNVERIFIED.
8. **Assister encodings:** null 139,891; 65535 33,399 (sentinel); real slot 0 7,865. About 60% of real assists have null position. UNVERIFIED.
9. **`rank_update`:** sentinels, inconsistent `rank_change` (260 rows), 73.6% null positions. All 5,873 rows fall after their round's end tick and in the last round of their match (section 4). VERIFIED for timing.
10. **`molotov_state`:** "no player" encoded as -1 (15,414 rows) and as null (56.8% of rows have null positions). VERIFIED.
11. **Velocity extremes:** |v| up to about 2.6e5 in `bomb_action` and weapon channels; `player_spawn` 5.1% above 10,000 units/s; `team_change` 889 rows above. Exact -1.0 placeholders. VERIFIED for weapons; spawn figures UNVERIFIED.
12. **`weapon_code` vs `weapon_name`:** 2,939 rows (0.054%) disagree. Codes 59 (`knife_t`, 119,693 rows), 45 (`smokegrenade`, 138,452) and 500 (`bayonet`, 13,587) carry other names. VERIFIED.
13. **`player_hurt.health_removed`** is raw damage with overkill: 9.8% of rows above 100 (max 4,999); equals prior health minus current on only 38%. UNVERIFIED.
14. **`weapon_action` reload and zoom rows** come in bursts: 82.3% of gaps are 2 ticks or less, so row counts are not event counts. UNVERIFIED.

**Cleared (not defects):** tick and second sentinels 65535, 32767, 9999, 999 are real times. `entity_id` 999 is a valid raw index. z above 10,000 is de_vertigo geometry. VERIFIED.

## 4. Leakage and exclusions

- **Post-round plants** (1,479 rows): bake-off excludes them (`plants_after_end`). Plant rate must use the live-to-`round_end` window. Timing VERIFIED; exclusion UNVERIFIED.
- **`rank_update`:** all 5,873 rows are after `round_end` and all are in each match's last round. Do not use as a pre-round feature. VERIFIED.
- **Outcome-side channels:** `round_end`, `round_mvp`, `score_update` (3 rows in 1,669 rounds, score swap), `bomb_defused` and `bomb_exploded` in `round_state` and `bomb_state`. These must not feed a pre-round model. Counts UNVERIFIED.
- **Selection on match length:** the economy sample (40 matches) has a score-lead AUC of 0.405 because it keeps only matches that reached round 24, which depends on the future outcome. Early-ending matches give 0.616. Do not estimate the score effect from the reached-24 subset. UNVERIFIED.
- **Within-round dependence:** about 67 snapshots per round. Split and CV by match only (state extraction and ablations do this). UNVERIFIED for split details.
- **Player identity is not recoverable:** name, `clan_tag` and chat are the literal "redacted" in every row; `steam_id` is a single letter. Cross-match player keys are invalid. UNVERIFIED.
- **Metadata** (`match_type`, platform) is a batch marker. Using it would make the batch identifiable. UNVERIFIED.
- Snapshots are restricted to `tick < round_end` (state extraction). VERIFIED.

## 5. Features the data supports (ablation evidence)

**Ablation 1** (HP, equipment, weapons, utility; 30 test matches, 2,673 rounds). Headline test log loss checked against JSON. VERIFIED: BASE 0.51683; BASE+HP 0.50737; BASE+EQUIP 0.49686. Gains and CIs below are from `ablation_hp_equip.md` (UNVERIFIED).

- Base state: `alive_diff`, `bomb_planted`, `seconds_since_plant`, round number, `score_diff`. Permutation importance on test: `alive_diff` +0.076 log loss; `equip_diff` +0.052; `seconds_since_plant` +0.024; `t_health` +0.014.
- **Equipment difference:** gain +0.020 on test (CI -0.0001 to +0.040); +0.047 without `score_diff` (CI +0.023 to +0.071). Strongest addition, borderline with score.
- **Health and armour:** +0.009 on test (CI includes 0); CV +0.015 (CI +0.009 to +0.022).
- **Weapon categories:** +0.007 on test (CI includes 0); CV +0.007 (CI excludes 0).
- **Utility:** no gain on test (-0.0004); about +0.005 in CV. Drop.
- **All groups:** +0.017 on test (CI -0.003 to +0.038); CV +0.015 (CI +0.008 to +0.022).
- `score_diff` alone is worth 0.033 log loss; once groups are in, it adds nothing.

**Ablation 2** (positions and map; 30 test matches). Headline values checked against JSON. VERIFIED: BASE 0.51683; BASE+POS 0.52603; BASE+POS+MAP 0.51942.

- Positions with `score_diff`: worse by 0.009 (CI -0.006 to +0.024), worse on all 5 seeds. Do not add.
- Positions without `score_diff`: better by 0.015, but still worse than BASE with score (0.535 vs 0.517). They recover part of what the score carries.
- Map adds nothing beyond positions (-0.007, CI -0.017 to +0.003).
- The positions gain is mostly post-plant CT proximity to the bomb, which overlaps bomb state.

**Pre-round priors** (6,233 test rounds). Round number alone +0.01% log-loss skill; map alone -0.07%; map x round -0.47%. Score difference at live start +1.14%; round x score +2.86% (log-loss gain 0.020, CI -0.025 to -0.015). UNVERIFIED.

**CT win rate:** R13 47.3% and R14 46.3% are significant after Bonferroni (R13 is the second pistol round). Map omnibus p = 0.003 but Kruskal p = 0.11. Platform and `match_type` show no signal (p 0.94, 0.8) and are confounded with batch. UNVERIFIED.

**Economy** (40 matches, 1,045 rounds): equipment-value AUC 0.621 alone; money 0.533. Economy plus score has fold AUC 0.661, but the score coefficient is not trustworthy from that sample. UNVERIFIED.

**Not supported by any test:** `rank_update` (post-match), item channels (not profiled), grenade and molotov channels (no feature tests).

## 6. Data-regime breaks by date

- **2026-10-08 batch:** 1,133 matches, 72% of clean 5v5 rounds, almost all metadata unknown, almost no wingman. 5v5 rounds per match +2.8 (21.0 vs 18.2). Plant time +3.5 s. Winner base rate unchanged (T 47.8% vs 48.0%). Round-start batch AUC 0.565 (weak). UNVERIFIED.
- **Channel gaps concentrated on 10-08:** `bullet_damage` (1,050 of 1,075 zero matches), `bot_takeover` (106 of 112), `player_sound` (all 20), `other_death` (248 of 370 zero matches). UNVERIFIED.
- 14 faceit matches, all on 10-01 to 10-07. UNVERIFIED.
- Halftime swap at round 12 (wingman at round 8). Overtime from round 25 (1,165 non-wingman rounds in 163 matches). UNVERIFIED.
- Round-1 tick-66 sentinel: date pattern not checked.

## 7. Open questions

1. Confirm score attachment in bake-off code and fix `feature_spec.md` and `schema_dictionary.md`. See reconciliation above.
2. Is raw `player_id` 0 a slot or a sentinel, and which id key is correct per channel?
3. Why are plants logged after `round_end` (demo ordering)?
4. What causes the 249 rounds over 200 s?
5. Are rank and wins in `player_info` pre-match snapshots, or updated after the match? Check before any use.
6. Profile the four item channels (`item_equip` has 9.6M rows) before any economy feature; they may carry buy-phase data.
7. Run the planned check: train 10-01 to 10-07, test 10-08 (not run).
8. Does freeze-end equipment already include pre-freeze purchases? Not checked.
9. The economy score sign needs a sample not selected on match length.
10. Representativeness: 150 of 464 eligible matches (state sample); 40 of 48 (economy).
11. Do the derived A/B bomb centres match real sites? Site codes were not used.
12. Cause of the round 16 alive-count gap in `22e9c3e7-...` (9 snapshots; dropped from the ablations).
13. Weapon codes 8, 11, 19, 23, 24, 26, 32, 38, 39, 63 verified only on sample days.
14. `bakeoff.py` was modified after its log was written. Changes unknown. Dedupe: 46 groups, 51 removed; 47 is the all-match fingerprint count.
15. Two notes conflict on raw `player_id` (flow: sentinel; state_sample: consistent within match). Resolve with a direct check.
16. `weapons.md` and `economy.md` were never written; only the JSON exists.
17. `combat.md` says `player_id_fixed` is a 1-10 slot; its table shows 1-19.
