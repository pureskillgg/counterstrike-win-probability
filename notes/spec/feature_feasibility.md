# CS2 round win probability: feature feasibility (Task B)

Date: 2026-10-13. Scope: the 11 candidate feature groups in the brief, checked against the 2026 tome (1,752 matches).

Evidence labels:
- VERIFIED-S1: `scripts\spec\feature_feasibility_check.py` (run with `pixi run python -I`). Compares every tome channel's parquet schema with `notes\spec\csds_dictionary.csv`, checks the claimed (table, column) pairs, and reads header, round-end, player_info and rank_update values.
- VERIFIED-S2: `scripts\spec\feature_feasibility_check2.py` (same method). Bomb site codes, bomb timers, round length.
- VERIFIED-CSV: read directly in `notes\spec\csds_dictionary.csv`.
- NOTE: taken from an EDA or spec note (`notes\eda\*.md`, `notes\spec\feature_spec.md`, `notes\schema_dictionary.md`) and not re-run, so UNVERIFIED by me.

## 0. Where the data is

- The tomes are not on E:. The 2026 tome is on G: at `tomes\2026-10-01_to_08\tome\csds`, 36 channel folders (header included) (VERIFIED-S1).
- E: holds the raw per-match folders. The match folder `E:\csds_final\csds\2026\10\01\0f044ff9-4360-4738-8eb2-a82b616b2d87` lists 41 table folders plus `csds`, including the per-tick channels `player_status`, `player_vector`, `tick`, `grenade_vector`, `world_item_vector`, `player_inputs` (VERIFIED-S1, directory listing only; no rows read; other matches not listed).
- Tome row counts (VERIFIED-S1): round_end 35,992; round_state 531,365; player_hurt 1,007,846; player_death 253,480; grenade_state 531,031; molotov_state 208,908; molotov_fire 3,278,394; weapon_fire 5,483,781; item_equip 9,585,945; player_footstep 1,333,987; header 1,752.

## 1. Feasibility table

"Knowable" means computable from rows with tick <= the state tick. "Where" is tome (T) or E:-only per-tick (E).

| # | Group | Feature | Channel.column (checked) | Where | Knowable at state? | Known problems | Build cost |
|---|---|---|---|---|---|---|---|
| 1a | Economy | Money and equipment at freeze end | player_status.money, round_start_equipment_cost, freezetime_end_equipment_cost, current_equipment_cost, equipment_value_calc | E (absent from tome, VERIFIED-S1) | Yes if read at or before the state tick | Rows only for living players; no money for dead players; no tome copy | High (USB read; cut to one tick per round) |
| 1b | Economy | Buy proxy from weapon events | item_equip.item, item_pickup.item, item_pickup.def_index, item_refund.item | T | Yes (replay events <= t) | No prices; item_equip.def_index is 0 on every row (NOTE); item text differs by table for one def_index (NOTE) | Medium |
| 1c | Economy | Loss streak and prior results | round_end.winner_team_code, round_end.win_reason_code, player_info.team_code | T | Yes, prior rounds only | winner is side-based; team identity flips at halftime (NOTE players.md: all rows swap at rounds 12 and 27) | Low-medium |
| 2a | Health and armor | Last-known HP and armor per player | player_hurt.health, player_hurt.armor, player_hurt.player_id_fixed, player_hurt.tick | T | Yes (last hurt with tick <= t) | health is post-hit (0 on kills); no tome value at round start, so 100/0 is an assumption (UNVERIFIED for CS2); 1 match has zero hurt rows; 18,536 rows in duplicate key groups (NOTE combat.md) | Low |
| 2b | Health and armor | Exact per-tick HP and armor | player_status.health, player_status.armor | E only | Yes | Rows only for living players | High |
| 3a | Damage dealt | Damage by attacker so far | player_hurt.attacker_id_fixed, player_hurt.effective_health_removed, player_hurt.health, player_hurt.attacker_team_code | T | Yes | health_removed is raw damage with overkill (up to 4,999; NOTE); attacker_id_fixed is double on page 0 and int64 on page 1 (NOTE); raw attacker_id 65535 on 1,256 rows; world damage null; ids are per-match only | Low |
| 3b | Damage dealt | Hit detail | bullet_damage.distance, bullet_damage.num_penetrations, bullet_damage.is_attacker_in_air | T | Yes | No damage amount; NOTE: 1,075 of 1,752 matches have zero bullet_damage rows (1,050 on 2026-10-08); several rows per tick | Low (coverage gap) |
| 4a | Utility | Grenades detonated, by type | grenade_state.event_type, grenade_state.player_id_fixed, grenade_state.tick_throw, grenade_state.entity_id_fixed, grenade_state.x_pos, y_pos, z_pos | T | Yes for detonations <= t; in-flight grenades are not visible | entity_id reused across rounds (join on round + entity_id_fixed); tick_throw null when unmatched (NOTE schema) | Low-medium |
| 4b | Utility | Flashes blinding enemies | player_blind.player_id_fixed, player_blind.attacker_id_fixed, player_blind.blind_duration, player_blind.player_team_code, player_blind.attacker_team_code | T | Yes | 20.7% of rows have player == attacker (own flash? NOTE, unverified); 21 zero-row matches (19 on 2026-10-08, NOTE); enemy test needs side at the tick | Low |
| 4c | Utility | Molotov and incendiary fire | molotov_state.event_type, molotov_state.burn_duration, molotov_state.was_extinguished_by_smoke, molotov_state.tick_throw, molotov_fire.is_burning, molotov_fire.x_pos, molotov_fire.fire_index | T (both channels present, VERIFIED-S1; contradicts feature_spec.md section 6) | Yes for samples <= t | molotov_fire has 3.28M rows; inferno_detonate and inferno_extinguish are declared but absent in the sample match (NOTE schema); -1 encodings in older matches | Medium |
| 4d | Utility | Smoke active window | grenade_state.event_type (smokegrenade_detonate, smokegrenade_expired), grenade_state.x_pos, y_pos | T | Yes | CS2 smokes are volumetric; extent is not in the data; only the detonate-to-expire window and the detonation point | Medium |
| 4e | Utility | Per-tick grenade track | grenade_vector.* | E only (absent from tome, VERIFIED-S1) | Yes | - | High |
| 5a | Weapons held | Inventory by event replay | item_equip.item, item_equip.player_id_fixed, item_equip.weapon_type_code, item_pickup.item, item_dropped.reason, item_refund.item | T | Yes (replay events <= t) | item_equip.def_index is 0 (NOTE); weapon_type_code meaning is a GUESS (NOTE); replay is approximate | Medium |
| 5b | Weapons held | Exact inventory and active weapon | player_status.inv_primary, player_status.inv_secondary, player_vector.weapon_code | E only | Yes | Codes need a name map from weapon_fire | High |
| 5c | Weapons held | Last shot | weapon_fire.weapon_name, weapon_fire.player_weapon_code | T | Last shot only | Code to name not 1:1 (code 40: ssg08 and bayonet; NOTE schema) | Low |
| 6a | Positions and map control | Last-known position from events | player_hurt.player_x_pos, player_death.player_x_pos, player_blind.player_x_pos, weapon_fire.player_x_pos, bomb_state.player_x_pos | T | Yes if the event is <= t; staleness is unbounded, so an age feature is needed | player_footstep has no position columns (VERIFIED-S1); about 60% of real assists have null assister position (NOTE combat.md); "asof merge" claim checked in one E: match only (NOTE schema) | Medium |
| 6b | Positions and map control | Continuous positions, aim, callouts | player_vector.x_pos, y_pos, z_pos, phi_ang, theta_ang; player_status.place_name | E only (absent from tome, VERIFIED-S1) | Yes | place_name blank in 78 rows (NOTE); dead players have no rows | High |
| 6c | Positions and map control | Site identity | bomb_state.site_code, bomb_action.site_code | T | Yes | Not A/B. Each match has 1 or 2 codes, and codes differ across matches on the same map (de_dust2: 265 codes over 316 matches) (VERIFIED-S2). The dictionary's 281 = BombsiteA, 282 = BombsiteB holds only on some maps (de_anubis, de_inferno; 282 on de_mirage) | Medium (cluster planter positions); High (true A/B needs place names from E) |
| 6d | Positions and map control | Distance to bomb | bomb_state.player_x_pos, y_pos, z_pos at bomb_planted (planter position) | T | Yes after plant | Bomb position itself is not continuous; whether a dropped C4 appears in item_dropped is UNVERIFIED | Medium |
| 7a | Bomb state | Plant event and planter | bomb_state.event_type, bomb_state.tick, bomb_state.player_id_fixed, bomb_action.event_type, bomb_defuse.has_kit | T | Yes if tick <= t | 30 matches have no bomb_state rows (NOTE flow.md); 8.1% of plant rows fall after round_end (NOTE flow.md), so filter to the live window; raw player_id 0 is a sentinel in about 10% of bomb_state rows (NOTE flow.md) | Low |
| 7b | Bomb state | Time left | header.freeze_time; round timer; bomb_state.tick at bomb_planted | T | Yes | Of 497 time-out rounds (win reason 12, no plant), 485 non-wingman rounds last exactly 115 s after round_freeze_end; the other 12 are wingman rounds at 90 s (VERIFIED-S2). After plant, bomb_exploded is 41.00 s after bomb_planted in all 3,090 rounds (VERIFIED-S2). header.bomb_time is NULL in all 1,752 matches (VERIFIED-S1); header.round_time_defuse (1.92 or 1.5) is not the round timer | Low |
| 7c | Bomb state | Defuse | bomb_state.tick at bomb_defused; bomb_defuse.tick | T | Outcome event: leakage if used at or after t | Plant-to-defuse 5.0 to 40.0 s, median 25.5 s (VERIFIED-S2) | Low |
| 8a | Trades and kills | Kill sequence, trades, flash and smoke flags | player_death.tick, player_death.player_id_fixed, player_death.attacker_id_fixed, player_death.assister_id_fixed, player_death.weapon_name, player_death.is_headshot, player_death.is_flash_assist, player_death.is_through_smoke, player_death.is_attacker_blind, player_death.is_noscope | T | Yes if tick <= t | assister_id has three encodings: null 139,891, 65535 33,399, raw 0 is a real slot 7,865 (NOTE combat.md); raw ids are per-match relabels (use *_fixed); 1 repeat victim in window (NOTE); other_death holds non-player deaths | Low |
| 8b | Trades and kills | Trade window | none (window length is a modelling choice) | - | - | Window length is not in the data (UNVERIFIED choice) | Low |
| 9a | Player skill | Pre-match rank snapshot | player_info.rank, player_info.rank_type, player_info.wins, player_info.rank_raw, player_info.rank_platform, player_info.radar_color_code | T | Yes: rank is constant within a match for 99.95% of match-player pairs (17,191 pairs, VERIFIED-S1) | rank 0 on 74.7% of rows (placeholder, NOTE); rank 9999 on 344 rows; rank_type -1 on 68%; wins capped at 2501 (NOTE); elo_platform is in the tome but not in the CSV, 99.1% null (NOTE) | Low |
| 9b | Player skill | Match result, if used | rank_update.rank_old, rank_update.rank_new, rank_update.rank_change, rank_update.win_count | T | NO. Leakage. All 5,873 rows (677 matches) sit in each match's last round (VERIFIED-S1) | Exclude | - |
| 9c | Player skill | Past stats across matches | player_personal.name, clan_tag (both "redacted"); player_personal.steam_id (per-match alias A to S) (VERIFIED-CSV lines 534 to 536) | T | Not possible | No cross-match player identity; player_id is a per-match slot | Infeasible |
| 10a | Map and side | Map prior | header.map_name | T | Yes | 15 maps: de_mirage 346, de_dust2 316, de_inferno 290, de_ancient 218, de_anubis 207, de_nuke 165, de_cache 141, de_vertigo 32, de_train 16, de_overpass 8, and 5 maps with 4 or fewer (VERIFIED-S1). NOTE: other_death has zero rows for every de_dust2 match (extraction gap) | Low |
| 10b | Map and side | Side (T or CT) | player_info.team_code (per round, per slot); player_death.player_team_code; round_end.winner_team_code | T | Yes | Side, not identity; swaps at halftime | Low |
| 11a | Round, score, streaks | Round number | round_end.round, round_state.round | T | Yes | - | Low |
| 11b | Round, score, streaks | Live score | round_state.t_score, round_state.ct_score (*_raw holds the original) | T | Yes at round_freeze_end | t_score + ct_score = round - 1 (NOTE feature_validation); score_update has 3 rows in 1,669 rounds (NOTE), so do not use it for score | Low |
| 11c | Round, score, streaks | Pistol and overtime flags | round_end.round; header.max_rounds (24 on 1,603 matches, 16 on 149 wingman, VERIFIED-S1) | T | Yes | round_start tick is 66 for round 1 in 678 matches (NOTE flow.md); time from round_freeze_end, not round_start | Low |
| 11d | Round, score, streaks | Phase and live start | round_state.event_type = round_freeze_end; round_state.phase | T | Yes | freezetime_ended_inferred is not used (NOTE feature_validation) | Low |
| L | Leakage (do not use) | Outcome events | round_end (winner, win_reason); round_mvp (round end, NOTE); bomb_state bomb_defused and bomb_exploded; rank_update; any event after t | T | No | - | - |

## 2. Build next (ranked)

1. Round-state core (groups 7b, 11a to 11d, 8a): round number, live score, alive counts from death events, plant state, and time left using the 115 s and 41 s rules (VERIFIED-S2). Low cost. Reuse `feature_validation.py` from the spec note.
2. Damage state (groups 2a, 3a): last-known HP and armor per alive player from player_hurt, using *_fixed ids within a match. Low to medium cost.
3. Kill sequences (group 8a). Low cost.
4. Loss streak with team-identity mapping (groups 1c, 11). Low to medium cost. Needed before any economy proxy.
5. Utility (group 4a to 4d): detonations, smoke windows, enemy flashes, molotov fire from molotov_state and molotov_fire. Medium cost.
6. Weapons held by event replay (group 5a), validated against weapon_fire (5c). Medium cost.
7. Site identity by clustering planter positions (group 6c), then decide whether true A/B labels are worth E: reads. Medium cost.
8. Economy snapshot from player_status (group 1a), one tick per round at freeze end. High cost, but the only true economy input.
9. Continuous positions and place names from player_vector and player_status (groups 6b, 2b, 5b). High cost; defer until 1 to 8 are validated.

Not worth building: past player stats (9c, infeasible). Rank (9a) is a weak in-match proxy only.

## 3. Conflicts and corrections

- feature_spec.md section 6 says per-tick molotov data is not in the tomes. basic_molotov_fire is in the tome: 3,278,394 rows, 2 pages (VERIFIED-S1). Molotov fire extents should move out of "cannot be derived".
- csds_dictionary.csv does not list some tome columns: header.key, match_id, network_protocol, freeze_time, round_time_defuse, bomb_time, max_rounds; item_equip.def_index; player_info.elo_platform and player_controller_id (VERIFIED-S1). schema_dictionary.md covers the header fields and item_equip.def_index. Build any column check against the parquet schema, not the CSV alone.
- site_code: the dictionary says 281 = BombsiteA and 282 = BombsiteB, checked on one match. Across the tome the codes are per-match identifiers (VERIFIED-S2). The A/B mapping does not hold generally.
- The brief says "24-round halves". The data fits MR12: max_rounds 24 (1,603 matches) and 16 (149 wingman) (VERIFIED-S1). Halftime swap after round 12 and overtime blocks of 6 rounds with a swap after 3 are NOTE (players.md). "24-round halves" does not fit; the brief may mean 24 rounds in total.
- Raw player_id 0: schema_dictionary.md says it is a real slot; flow.md treats it as a sentinel in deaths and bomb rows; combat.md says raw 0 is a real assister slot. Unresolved. Use *_fixed ids.
- players.md shows 47 groups (99 matches) of duplicated player_info content under different match keys (NOTE). Match-level splits must group these.

## 4. CS2 versus CS:GO

- Format: MR12 (max_rounds 24) in the data, not MR15. CS:GO halftime and economy assumptions tied to round 15 do not apply.
- Timing: ticks are integers at 64 per second (`second` = tick / 64 exactly, NOTE), so event times sit on a 15.6 ms grid. Sub-tick timing is not in the data.
- Smokes: CS2 smokes are volumetric. Extent is not in the data, so only detonate and expire ticks and the detonation point are usable. CS:GO sphere radii do not transfer.
- Economy: CS:GO MR15 economy values (round loss bonus, income) do not transfer. Loss effects must be learned from this data. The data has money 800 at the start (player_status sample, NOTE) and equipment costs, but no income rules.
- Timers: 115 s (5v5) and 90 s (wingman) round timers and the 41 s bomb timer are inferred from the data (VERIFIED-S2). They are not checked against game documentation.

## 5. Not verified

- Every count and claim taken from the EDA and spec notes was not re-run: the bullet_damage gap, other_death for de_dust2, the 30 matches without bomb_state, plant rows after round_end, duplicate matches, round_start sentinel, round_mvp rows, score_update rows.
- E: per-tick contents: one match folder listed, no rows read. money, health, armor and place_name values come from the sample in schema_dictionary.md for one match.
- Meaning of rank 0 and 9999, rank_type -1, weapon_type_code, hit_box_code, radar_color_code, freezetime_end_equipment_cost.
- The asof-merge claim for event positions (checked in one E: match only, per NOTE).
- Whether a dropped C4 appears in item_dropped; whether molotov_fire samples cover full fire extents (only columns were checked).
- Notes read by heading or summary only: schema_dictionary.md per-table sections for bullet_damage, player_blind and the middle tables; combat.md sections for bullet_damage, player_blind, player_footstep (lines 354 to 656); players.md team_change and player_chat (lines 573 to 737).
- Trade window length, A/B mapping of site codes, and the 100/0 round-start HP and armor assumption.
