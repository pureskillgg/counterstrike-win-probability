# 1 Hz in-round state sample: extraction

Scripts: `scripts/eda/extract_state_1hz.py` (extractor), `scripts/eda/verify_state_1hz.py` (independent checks on the saved outputs), `scripts/eda/state_1hz_probe.py` (one-match probe that set the design). Logs: `logs/eda/extract_state_1hz.log`, `logs/eda/state_1hz_validation.json`, `logs/eda/state_1hz_e_timing.csv`, `logs/eda/verify_state_1hz.log`.

Outputs (G: only; nothing written to E:):
- `data/state_sample/players_1hz.parquet`: 1,231,817 rows, 27 columns, 15.00 MB (zstd). One row per alive player per snapshot.
- `data/state_sample/rounds.parquet`: 2,673 rows, 12 columns, 30.8 KB. One row per clean round.

## 1. Kept set (bake-off port)

- VERIFIED: the bake-off's channel filter, duplicate removal and round filters are ported from `scripts/model/bakeoff.py` main() (main is not importable). Round states come from `bakeoff.build_states`, imported read-only. The script asserts the counts against `logs/bakeoff.log`: 1,709 matches after the channel filter; 47 duplicate matches removed; 1,521 matches and 28,683 rounds kept; 234,265 states. All six match exactly.
- Bake-off round drops over all matches: `player_count != 10` 3,663; disconnect or bot takeover 1,767; duplicate copy 913; not 5 v 5 at start 476.

## 2. Sample and split

- Kept matches by key day: 01: 50, 02: 48, 03: 85, 04: 78, 05: 58, 06: 77, 07: 68, 08: 1,057. Days 01 to 07 hold 464 kept matches, enough for 150, so no fallback was needed. Sample = `random.Random(0).sample` over the sorted pool.
- Match-level split: sorted sample shuffled with `random.Random(0)`, first 20% (30 matches) test, 120 train.
- Clean rounds: 2,673 (train 2,185; test 488). Snapshots: 179,466 (train 146,582; test 32,884). CT-win share over rounds 0.5339.
- Bake-off round filters applied to the sampled matches. These include the task's three criteria (`player_count == 10`, exact 5 v 5 at start, no disconnect or takeover) plus its other filters.
- VERIFIED: 1,238 of 2,673 rounds have a bomb plant inside the live window (`plant_tick`); the rest are null. No plant tick falls outside the live window.

## 3. Snapshots and sources

- Snapshot ticks: `range(freeze_end, round_end, 64)`, so `tick - freeze_end` is a multiple of 64 and `tick < round_end`. `seconds_since_live` is the integer second count.
- VERIFIED: every snapshot tick has per-tick rows in both raw channels (179,466 distinct snapshots in the output, equal to the grid count from the rounds table).
- E: reads are `player_status` and `player_vector` only, column-projected, for the 150 matches. Path `E:\csds_final\csds\2026\10\<DD>\<id>\<channel>`.
- VERIFIED: `team_code` is in `player_vector` (not `player_status`, as the economy script noted). Team comes from there.
- VERIFIED: the two raw channels join on (round, tick, player_id) with 0 unmatched rows either way, 0 duplicate keys and 0 null `team_code`.
- Key: `player_id`, within the match only. `player_id_fixed` was not used. The task states it is the per-match key, and the note `notes/eda/players.md` says `player_id` and `player_id_fixed` disagree on about 91% of `player_info` rows. Within-match `player_id` is consistent across raw status, raw vector and tome deaths (see check 2). Cross-match keys are not used. The `player_id_fixed` question is UNVERIFIED here.

## 4. Validation

1. VERIFIED: health 1 to 100 and armor 0 to 100 on every output row (0 rows outside 0-100).
2. Alive count per side against player_death, with the victim dead from its death tick on (deaths with tick <= snapshot tick; first death per victim in the live window; side from `player_team_code`, else `player_info`, the bake-off rule). VERIFIED:
   - Agreement: 179,457 of 179,466 snapshots (99.995%), above the 99% bar.
   - Strict rule (victim still counted at its death tick): 179,189 (99.85%). The two rules differ on 268 snapshots, and all 268 have a death at that exact tick. The table already excludes the victim at that tick, so the inclusive rule is correct.
   - The 9 differing snapshots are one round: `csds/2026/10/07/m_7392c5c6d537a57c2e735e98/csds`, round 16 (live start tick 94,774, snapshots k = 0 to 8). The table lists 4 T; the tome says 5 T alive (`player_info`; first T death at tick 96,603). T `player_id` 4 has no `player_status` or `player_vector` row until tick 95,318, then rows through the round. Its `weapon_code` is null in all 65 of its snapshot rows. The team_change table has no event in this round (its only change is in round 12). Cause of the gap: UNVERIFIED. The round passes the clean-round criteria, so it is kept and flagged in `rounds.alive_gap_snapshots` (9).
   - Live-start 5 v 5 in the table: 2,672 of 2,673 rounds (the exception is that round).
   - Probe on one match (`state_1hz_probe.py`, match `csds/2026/10/07/m_partial...`): the inclusive rule holds on 1,686 of 1,686 snapshots; the strict rule fails on 1 (a death at its own tick).
3. VERIFIED: at most 5 players per side in every snapshot (max 5; 0 snapshots over 5).
4. I/O, VERIFIED: E: reads are 300 files (150 `player_status`, 150 `player_vector`), 4.19 GB on disk (`player_status` 0.65 GB, `player_vector` 3.55 GB), 132.6M rows scanned per channel. Read time 30.8 s in the final run (`player_status` 23.1 s, `player_vector` 7.7 s). The first run took 43.6 s on the same files, so read time varies with disk cache. Kept 1,231,817 player-snapshot rows (0.93% of rows read). Script wall time 62.6 s, including tome reads from G:.
- VERIFIED (independent script): no duplicate (match, round, tick, player) rows; no rows off the 64-tick grid or at or after `round_end`; each rounds key unique; each player row has a rounds row; `ct_score + t_score == round - 1` for all 2,673 rounds (bake-off score check); each match has one split; `round_number == round`; `team_code` in {2, 3}.
- VERIFIED: nulls in output: `weapon_code` only, 65 rows (the T player in the round above).

## 5. Columns

- `players_1hz`: match_key, round, tick, seconds_since_live, split, player_id (extra, to key rows), team_code, health, armor, has_helmet, has_defuser, money, equipment_value_calc, inv_primary, inv_secondary, inv_flashbang, inv_smokegrenade, inv_hegrenade, inv_molotov, inv_incgrenade, inv_decoy, inv_c4, weapon_code, x_pos, y_pos, z_pos, is_defusing. Mean 6.86 alive players per snapshot.
- `rounds`: match_key, round, ct_win, map_name, round_number (equals round), ct_score, t_score (both before the round, at freeze end), plant_tick (null if none in the live window), round_end_tick, split, live_start_tick (extra), alive_gap_snapshots (extra; snapshots where the table disagrees with the death count).

## 6. Caveats

- Snapshots within a round are strongly dependent (about 67 per round). Evaluate with match-level splits or grouped CV, as the split here does.
- The sample is 150 of 464 eligible matches (days 01 to 07). How it compares with the other 1,371 kept matches is UNVERIFIED.
- Round 16 of match `m_7392c5c6-...` is the only data anomaly found. Drop it with `alive_gap_snapshots > 0` if a clean sample is needed.

Reproduce: `pixi run python -I -X utf8 scripts\eda\extract_state_1hz.py`, then `pixi run python -I -X utf8 scripts\eda\verify_state_1hz.py`.
