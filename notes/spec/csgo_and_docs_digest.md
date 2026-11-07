# CS2 CSDS: documentation and csgo-dsdk digest

Collected 2026-10-12. Read-only. Nothing under E:\ or under G:\pureskill\csds-cs2\tomes was touched, and no packages were installed.

## 0. How to read this

Confidence tags:
- **DOCUMENTED**: stated on a docs.pureskill.gg page (URL given).
- **CODE**: read in pureskillgg/csgo-dsdk, default branch `master` (path given).
- **INFERRED**: my reading across sources. The source does not state it outright.

Method and limits:
- The docs pages were read with WebFetch, which returns a model-written summary. Verbatim page text was withheld, so definitions below are paraphrased, not quoted. Exact wording is at each URL. Limited quoting also follows the copyright rule that applies to this work.
- Where a label or constant depends on exact wording, I re-fetched with a narrower prompt: the enums game-constants table, the getting-the-data channel list, the archived spec, and the index object-count sentence. Summaries can still be wrong at field level. Verify label fields against the live spec or the data dictionary CSV (section 6) before modelling.
- csgo-dsdk was read via `gh api` (raw files) and `gh search code` (terms: round, win, label, tick, freeze, overtime, halftime, team, round_end, winner, player_count, max_rounds, pop_overtime). No code hits for freeze, halftime, round_end, winner or player_count.
- Local notes: `schema_dictionary.md` was read in targeted sections plus grep (171 KB, not read in full). `research/csds_docs_facts.md` was read in full.

---

## 1. The five definitions that matter most for labels and features

1. **Round label: `round_end.winner_team_code`** [DOCUMENTED]. One row per round. Side codes: 2 = Terrorists (T), 3 = Counter-Terrorists (CT). `win_reason_code` gives the cause (section 2.6). Drawn rounds are removed before publication and the numbering has no gaps afterwards, so round numbers are renumbered [INFERRED from the docs' no-gaps statement]. The win codes and `player_count` for matches processed 2024-02-12 to 2026-08-03 were rebuilt from `round_state` (win codes only 8 and 9, `player_count` fixed at 20). Which version a match has depends on its processing date [INFERRED from the datasheet's pipeline-version statement].

2. **Team identity per round: `player_info.team_code`** [DOCUMENTED]. One row per player per round, a snapshot at round_freeze_end. It gives the side the player is on in that round (2 T, 3 CT). Teams are not fixed to sides: they swap at halftime, and the docs say overtime also has swaps (exact rule not stated). To label "team X won the round", map the winning side to team X through each player's side in that round.

3. **Match outcome: `header.t_starters_score_final` and `header.ct_starters_score_final`** [DOCUMENTED, changelog 2026-10-03]. Each starting team's own final score, followed through the swaps. Before 2026-10-03 the docs say these took the larger side score. Safer label source: recount from round_end winners plus the side mapping in item 2. Draws are possible (section 2.6).

4. **Regulation, overtime, halftime** [DOCUMENTED, enums game-constants table; how-it-fits-together]. `header.max_rounds` is the regulation length: 24 for Premier, FACEIT and Competitive, 16 for Wingman. Rounds above it are overtime. Premier and FACEIT overtime: at 12-12, six more rounds, first to 16; a Premier 15-15 overtime ends as a draw. Competitive and Wingman have no overtime. Halftime: sides swap after round `max_rounds / 2` (12, or 8 in Wingman). `round_state.phase == "halftime"` marks the break.

5. **Timing: freeze, round start, and the feature snapshot** [DOCUMENTED]. `header.freeze_time` is 20 s (Premier, FACEIT), 15 s (Competitive), 10 s (Wingman); null when the demo doesn't record it. Buy time is 20 s after freeze in all modes. Round time runs from the end of freeze: 115 s (Premier, FACEIT, Competitive), 90 s (Wingman). Round start is round_freeze_end; `freezetime_ended_inferred` is the first tick a player moves, and since 2026-10-05 it lands one tick after round_freeze_end. The docs never define "live". Feature snapshot: `player_info` at freeze end gives wins (capped at 2501), rank, `rank_type` (11 Premier CS rating, 12 Competitive skill group, 7 Wingman skill group, -1 not a Valve ladder such as FACEIT, 0 unknown), and team_code.

---

## 2. Docs pages (docs.pureskill.gg/datascience/adx/cs2/csds/)

Pages below are every sub-page linked from the index, plus two linked from those pages (ethics-review from the datasheet, and the archived spec from the index). The index link list had 13 entries; the `#` anchors point to the same pages.

### 2.1 Index (Competitive Gameplay)
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/
- Covers: dataset intro; 43 files per match in loose form (42 channels plus an index object); newer revisions deliver one zip per match; access through AWS Data Exchange; license CC BY-NC-SA 4.0 with attribution.
- Links from this page: Data Dictionary (CSV file at `docs.pureskill.gg/assets/files/csds_dictionary-02e6fb29983117ce857fa022aa096b6c.csv`, not downloaded, see section 6), Datasheet, CSDS Spec, archived spec (`/datascience/old/cs2/csds/spec`). No glossary page is linked.
- Tag: DOCUMENTED.

### 2.2 Quickstart
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/quickstart
- Covers: export of one day's revision to an S3 bucket you own; extracting zips; loading with the pureskill dsdk loaders; tomes that stack channels with a `match_key`.
- Requirements stated: Python 3.11+, pureskillgg-dsdk 4.0.3 or later, pureskillgg-csgo-dsdk 3.3.1 or later.
- Scale: a revision is usually 60 to 150 matches per day.
- Explicitly does not define rounds, round winners, team labels, freeze time, overtime, halftime or player counts.
- Tag: DOCUMENTED.

### 2.3 Getting the Data (also #file-layout)
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/getting-the-data
- **Layout.** Partition is `csds/YYYY/MM/DD`, the UTC processing day. Newer revisions: one object per match, `<id>.zip`. Older revisions: a folder `<id>/`.
- **Index object `csds`.** Gzipped JSON with `id`, `key`, `createdAt`, `matchDate`, `platform`, `matchType`, `game`, and `channels[]` (each with `channel`, `key`, `category`, `events`, `columns`, `available`). Read it, do not assume a channel list.
- **Channel files.** Apache Parquet, named after the channel, no extension. ZSTD since 2026-07-08 (GZIP before).
- **Object counts.** 31 objects per loose match through 2026-08-02 (30 channels plus index); 43 from 2026-08-04 (42 plus index). The 2026-08-03 revision mixes both. [DOCUMENTED; consistent with the archived spec.]
- **`available` flag.** Channels that may be empty and carry the flag: `bot_takeover`, `bullet_damage`, `player_inputs`, `player_sound`, `rank_update`, `round_mvp`. `round_end`, `round_start`, `round_state` are not on this list. [INFERRED: an empty round_end cannot be read as "not recorded".]
- **Revision window.** A revision opens at 00:00 UTC and collects every match processed that day. It is published when the next one opens. A match's revision depends on processing day, not play day.
- **`match_date`.** The play date, except hand-uploaded matches, where it is the processing date. The date in the key is always the processing date.
- **Duplicates.** A match `id` is new each time its demo is processed. Find duplicates by comparing `header` fields (map_name, server_name, build_num, final scores).
- **Retention.** Revisions are pruned about one year after creation; exportable window is one to two years.
- Tag: DOCUMENTED.

### 2.4 How It Fits Together
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/how-it-fits-together
- **Channel categories.** `header` (one row); `player_info` (one row per player per round); `telemetry` (per tick); `single_event`; `multi_event` (one of several related events, named in `event_type`).
- **Keys.** `tick` (64 per second; `second = tick / 64`). `round` (starts at 1; every row except the header has it). `player_id` (game id, not meaningful across matches). `player_id_fixed` (numbered from 1 per match, constant within the match). `player_personal.steam_id` (per-match letter, constant within the match). Bots have their own numbers; `player_personal.is_bot` marks them.
- **Tick matching.** Most channels copy player state from `player_vector` at their own tick. `player_death`, `player_hurt`, `bullet_damage`, `bomb_action`, `item_dropped`, `player_chat`, `rank_update` use the latest row up to 3 ticks earlier; null if none is close enough.
- **Rounds.** Warmup, knife and drawn rounds are removed before publication. Round numbers run on without gaps.
- **`round_end`.** One row per round, with `winner_team_code` (2 T, 3 CT) and `win_reason_code`.
- **Regulation length.** `header.max_rounds` (24, or 16 in Wingman). Rounds above it are overtime.
- **`round_state`.** One row per state-machine event, `event_type` naming it (freeze end, buy time ended, bomb events, round end and so on).
- **Freeze.** `player_info` is taken when the round's freeze time ends.
- **Sides.** `player_info.team_code` gives each player's side per round. `header` columns starting `t_starters_` and `ct_starters_` describe the team that began as each side.
- **Halftime.** Sides swap after round `max_rounds / 2`. The page reports that every player in every regulation match it checked changed sides.
- **Player counts.** No per-team count is defined on this page. Bots are identified by `player_personal.is_bot`.
- **Dead players.** No rows for the rest of the round in `player_vector` or `player_status`.
- **Empty channels.** `player_status` can be empty in a match (one of 89 matches published 2026-10-07).
- Tag: DOCUMENTED.

### 2.5 CSDS Spec (current)
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/spec
- Column-level reference, about 900 columns, grouped by channel, each with a replay, calculated, merged or redacted origin.
- **round_end**: `round`, `tick`, `second`, `winner_team_code` (int64), `win_reason_code`, `win_reason_message`, `player_count` (int64). The spec does not define the code values; see 2.6.
- **round_state**: `round`, `tick`, `event_type`, `t_score` and `ct_score` (fixed, from replay_fixed), `t_score_raw` and `ct_score_raw` (unadjusted parser values), `phase`, `is_warmup`, `second`. `start_halftime` is listed among the event types that trigger this channel.
- **round_start**: `round`, `tick`, `second`.
- **player_info** (round_freeze_end): `round`, `player_id`, `team_code`, `wins` (capped), `rank`, `rank_type`, `radar_color_code`, `player_id_fixed`, `rank_raw`, `rank_platform`; `player_controller_id` added 2026-09-07.
- **header**: `t_starters_score_final`, `ct_starters_score_final` (calculated from round_state); `t_starters_avg_rank`, `t_starters_avg_wins`, `ct_starters_avg_rank`, `ct_starters_avg_wins`; `number_of_points`; `max_unique_players`; `unique_steamids`; `match_type`; `tick_save_rate`. **Verify**: my summary of this page did not list `max_rounds`, `freeze_time`, `round_time_defuse` or `bomb_time` under header, although the changelog says they were added 2026-09-07 and the dictionary has them.
- **score_update**: `round`, `tick`, `team_code`, `old_score`, `new_score`.
- **team_change**: `round`, `tick`, `player_id`, `old_team_code`, `new_team_code`, `is_bot`, `silent`.
- **Nulls.** Null means no value. Older matches used markers instead (65535 for no attacker, empty string for no weapon, -1/-2/0 in several molotov and grenade fields).
- **player_vector derived columns.** Ten columns are computed on load with csgo-dsdk 3.3.1 or later. `movement_angle` is 0 to 360 and null when standing still; older matches stored 0.
- **Not on this page (per summary):** an overtime flag, a halftime flag, freeze time as a duration, an explicit match-end field. Halftime and freeze appear only as event names.
- Tag: DOCUMENTED (summary-level; see the verify note above).

### 2.6 Enums (includes the game-constants table)
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/enums
- **team_code.** 0 Unassigned (appears as `old_team_code` when a player joins); 1 Spectator (not seen in these matches); 2 Terrorists; 3 Counter-Terrorists.
- **round_end.win_reason_code and winner.** 1 bomb exploded (T); 7 bomb defused (CT); 8 Terrorists eliminated (CT); 9 Counter-Terrorists eliminated (T); 11 all hostages rescued (CT, hostage maps); 12 time ran out with no bomb planted (CT); 13 time ran out on a hostage map (T); 17 Terrorists surrendered (CT); 18 CTs surrendered (T). Code 10 (draw) would exist, but drawn rounds are removed before publication.
- **round_state.phase.** `first`, `second`, `halftime`, `postgame`.
- **rank_type (player_info).** 11 Premier (CS rating, from 0); 12 Competitive (skill group 0 to 18, per map); 7 Wingman (skill group 0 to 18); -1 not a Valve ladder (FACEIT and most manual uploads; rank is 0); 0 unknown (older matches).
- **Game constants (re-fetched; "Match and round" table):**

| Mode | Freeze time | Round time | Regulation rounds | Rounds to win | Overtime rule |
|---|---|---|---|---|---|
| Premier | 20 s | 1:55 (115 s) | 24 | 13 | At 12-12, six more rounds, first to 16. A 15-15 overtime ends as a draw. |
| FACEIT | 20 s | 1:55 (115 s) | 24 | 13 | At 12-12, six more rounds, first to 16. |
| Competitive | 15 s | 1:55 (115 s) | 24 | 13 | None. Can end level (12-12 or 8-8 per page). |
| Wingman | 10 s | 1:30 (90 s) | 16 | 9 | None. Page says can end level at 8-8 or 12-12 (see note). |

  Footnotes on the page: round time runs from the end of freeze; hostage maps use the same 1:55; a few rounds have a longer freeze than the setting; buy time after freeze is 20 s in all four modes. The page lists Premier and FACEIT together in its columns.
  Note: a 16-round Wingman match with no overtime cannot end 12-12, so the "12-12" for Wingman is a page error or a typo. [DOCUMENTED wording; INFERRED that it is an error.]
- **round_state event types.** The page lists them per channel. It does not list `start_halftime` in the observed round_state list, though the spec lists it as a trigger.
- **Bombsite (`site_code`).** Not a fixed enum; a trigger ID that varies by match. Identify sites by position or `player_status.place_name`.
- Other enums (hit groups, weapon types, item indices, radar colour codes) are not label-relevant.
- Tag: DOCUMENTED.

### 2.7 Positions and View Angles
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/coordinates
- Right-handed; x and y along the ground, z up; units are game world units (about one inch).
- Position is the feet (model origin), not the eyes. Eye height is about 64 units standing and 46 crouched, so `h ≈ 64 - 18 × duck_amount` (`duck_amount` is in `player_vector`).
- `theta_ang`: facing, -180 to 180. 0 along +x, 90 along +y, counterclockwise from above. Wrap differences into -180 to 180.
- `phi_ang`: 0 straight up, 90 level, 180 straight down. The data runs 1 to 179.
- Merged copies on event channels carry prefixes (`player_x_pos`, `attacker_theta_ang`, `assister_phi_ang`) and reflect the last tick at or before the event.
- Tag: DOCUMENTED.

### 2.8 Changelog
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/changelog
Label- and feature-relevant entries (dated, newest first):
- **2026-10-11.** `player_inputs` rows restored; parser 5.7.0 rebuilds button state.
- **2026-10-06.** `player_vector` no longer stores its ten derived columns.
- **2026-10-05.** `freezetime_ended_inferred` no longer fires early on a late respawn; now one tick after `round_freeze_end`. Every player who spawns in a round reaching freeze end gets a `player_info` and `player_personal` row for that round. `player_spawn.round` now records the round the player spawned into (previously often the prior round).
- **2026-10-03.** `header.t_starters_score_final` and `ct_starters_score_final` now track each starting team's own score through halftime and overtime swaps (previously the larger side score). Starting-team average rank and wins use only the first half's players, over `max_rounds / 2` rounds (12, or 8 in Wingman); a team with no players is null instead of -1. `header.tick_save_rate` now 64. FACEIT `rank_platform` and `elo_platform` fixed.
- **2026-09-07.** `player_info.player_controller_id` added. A match where two players share a round and `player_id_fixed` now fails processing. `header.freeze_time`, `round_time_defuse`, `bomb_time`, `max_rounds` added as server settings, null when not recorded.
- **2026-08-22.** `header.is_wingman` added.
- **2026-08-03.** `round_end.win_reason_code`, `win_reason_message` and `player_count` restored from real round-end events; `player_count` counts team players at round end, bots included. `round_end.legacy_code` removed. `round_start` restored, one per round including round 1; `time_limit`, `frag_limit`, `objective` removed. `round_state` `round_start` and `round_end` event rows restored; each round now has two `round_start` rows. `round_state.t_score` and `ct_score` no longer double-count the final round in 24-round matches. `team_change` added (includes the silent halftime swap). `score_update` added. `round_mvp.mvp_count` added.
- **2026-07-08.** Parquet compression changed to ZSTD.
- **2026-07-07.** `player_info.rank_type` added; `elo_platform` added for FACEIT; `tick.previous_phase` phase order fixed.
- **2026-06-07.** Duplicate `player_id` within a round resolved by controller id; matches still duplicated fail and are not published.
- **2024-08-05.** A `round_start` row added one tick after `round_poststart` per round (converter-added); player id remapping per round.
- **2024-07-28.** `freezetime_ended_inferred` ignores the first 5 ticks of a round.
- **2024-07-07.** `freezetime_ended_inferred` added: one row per round at the first tick any player moves. `*_starters_score_final` fixed for missing scores.
- **2024-02-12.** CS2 demos stopped carrying round-end events. `round_end` win codes and `player_count` rebuilt from `round_state`: winners coded 8 (CT) or 9 (T), `player_count` fixed at 20. `round_start` rebuilt with one row fewer than rounds played. `player_info.team_code` fixed for players given the wrong team.
- **2023-11-05.** CS2 data begins with 30 channels; `tick_save_rate` fixed at 21 at launch.
- Tag: DOCUMENTED.

### 2.9 Changelog by Channel
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/changelog-by-channel
- **round_end**: 2026-08-03 real round ends restored (win fields, `player_count`). 2024-02-12 rebuilt from `round_state` (codes 8 and 9; `player_count` 20).
- **round_state**: 2026-10-05 freezetime inferred row moved; 2026-08-03 round_start and round_end rows restored; 2026-08-03 final-round double count removed in 24-round matches; 2024-08-05 converter round_start row; 2024-07-28 first 5 ticks ignored; 2024-07-07 freezetime inferred row added; 2024-02-12 event rows removed.
- **round_start**: 2026-08-03 one row per round incl. round 1; 2024-08-05 rebuilt from round_state when a demo has no round starts; 2024-02-12 one row short.
- **player_info / player_personal**: 2026-10-05 row for every spawning player at freeze end; 2026-10-03 FACEIT rank fix; 2026-09-07 controller id and duplicate-failure rule; 2026-06-07 controller id used for duplicate player_id; 2024-02-12 team_code fix.
- **player_spawn**: 2026-10-05 `round` fixed.
- **tick**: 2026-10-05 `second_since_previous_phase` null before the first phase event.
- **team_change**, **score_update**: both added 2026-08-03.
- **round_mvp**: 2026-08-03 `mvp_count` added; rows came back after being mostly absent from early 2024.
- **header**: see 2.8 for 2026-10-03 and 2026-09-07 entries.
- Tag: DOCUMENTED.

### 2.10 PII Removal
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/pii-removal
- Redacted: player names, clan tags, name-change fields, `player_chat.text`, `header.sharecode` and `demo_id`, and the index `sharecode`, `demoId`, `metadata.bucket`.
- Steam IDs replaced with per-match letters A, B, C, in order of first appearance.
- `player_status.ping` set to 0.
- Index `matchDate` cut to the minute; raw `header.match_date` unchanged.
- Capped: `player_info.wins` and `rank_update.win_count` above 2500 published as 2501.
- Unchanged: everything else, including round, winner, score, team and player-count fields (the page says it makes no statements about them beyond the win cap).
- Tag: DOCUMENTED. Note: the commend caps in csgo-dsdk are not on this page (section 5.3).

### 2.11 Revision Stats
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/revision-stats
- Totals (counted 2026-10-12): 410 exportable revisions, 40,205 matches, 1,384,095 files, about 1.39 TB. Revisions dated 2025-07-19 to 2026-10-11.
- Largest recent revisions: 1,162 matches on each of 2026-10-09, 2026-10-10 and 2026-10-11; 1,135 on 2026-10-08.
- Backlog revisions after pipeline outages: 2026-06-27 (1,007), 2025-11-18 (919), 2026-04-29 (597), 2026-07-03 (525), 2026-05-29 (754).
- Days covered by an outage have no revision.
- Fifteen revisions are the first to carry a published data change. A change can land partway through a day, so that revision can mix before and after matches.
- Only exportable revisions are counted.
- Tag: DOCUMENTED.

### 2.12 Known Flaws
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/known-flaws
- Hand-uploaded matches: platform `unknown`; `match_date` is the processing date. Since 2026-10-08 includes many FACEIT matches; detect via `header.server_name` beginning `FACEIT.com`.
- Cross-match player linking possible for some players (fingerprinting); no workaround stated.
- `item_equip.def_index` always 0 in matches processed since 2026-10-05; use `item_pickup.def_index`.
- Short item names shared within a slot (`hkp2000`, `m4a1`, `mp7`, `deagle`).
- Duplicate matches: a demo processed twice gets two ids. Compare `header` entries that describe the match.
- Empty channels on tournament servers that restrict broadcasts: `player_blind`, `player_footstep`, `item_equip`, `weapon_action`.
- `player_sound`: empty in most matches. `player_inputs`: empty for matches recorded after the 2026-07-09 CS2 update and parsed before parser 5.7.0 (2026-10-11).
- `player_vector` and `player_status` have rows only for living players.
- Older FACEIT matches are missing player ranks.
- Events occasionally missing within a match; skipping the match is usually enough.
- Channel-set changes: earlier revisions have 30 channels; 14 channels added, `item_remove` and `player_action` removed.
- `player_vector` derived columns no longer stored since 2026-10-06.
- Older matches came from older pipeline versions; check `header.rushb_version` and `header.ppp_version`.
- **The page does not mention rounds, winners, round_end, scores, overtime, halftime, freeze time, player counts or team codes.** The label-relevant fixes and rebuilds are only in the changelog (section 2.8).
- Tag: DOCUMENTED.

### 2.13 Datasheet
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/datasheet
- **Purpose.** Education and machine learning research on CS2 gameplay; no specific task defined.
- **Composition.** One processed CS2 match per instance, split into 42 channels. Warmup, knife and drawn rounds removed, so round 1 is the first live round. Not an unbiased sample: only successfully processed matches; most have one PureSkill.gg user.
- **Collection.** Demos from users who connect Steam or FACEIT accounts, or from manual uploads (which may include no PureSkill.gg user). Read with an open source demo library, then proprietary extraction. The converter removes warmup, knife and drawn rounds, drops duplicate rows, merges positions from `player_vector`, computes derived columns, and fixes known errors.
- **Labels and targets.** None included; users define their own. Player rank is available.
- **Intended uses.** Includes win probability models among ML uses. A prior win-probability study is cited (I did not verify that citation).
- **Recommended.** Split train and test by match, since rows from one match share players and rounds.
- **Prohibited.** Identifying players, commercial use without license, cheat development, training or testing, releasing transformed data without attribution.
- **Size and dates.** Collection from 2021-12-01; distribution from 2022-05-17. CS:GO data in older revisions; every retained revision carries CS2 data.
- **Retention.** Demo files kept 30 days and parsed replays 7 days, then deleted. A match cannot be reprocessed after that, so pipeline improvements affect only newly processed matches. [INFERRED: a single tome can mix pipeline versions.]
- **Licence.** CC BY-NC-SA 4.0; AWS Data Exchange subscription.
- **Limitations.** Older matches came from older pipeline versions; channels empty in some matches; events may be missing; duplicates; schema changes (check `csds` index and `header.ppp_version`); redacted names and Steam IDs; cross-match linkability; bystander consent; likely minors; internal ethics review.
- Tag: DOCUMENTED.

### 2.14 Ethics Review (linked from the datasheet)
URL: https://docs.pureskill.gg/datascience/adx/cs2/csds/ethics-review
- Internal review, October 2026, with an AI agent checking the datasheet's claims; FPS Critic staff reviewed findings; no independent body.
- PII removal tested on 50 matches; cross-match linkage tested on one week's data; only counts recorded.
- Known: for some players, enough data carries across matches to link them.
- Misuse concern: per-tick aim data and some button presses could train cheats; prohibited.
- Sample bias toward players seeking coaching.
- Does not define labels, targets, rounds, outcomes or player counts; gives no total counts beyond the two tests.
- Tag: DOCUMENTED.

### 2.15 Archived spec (linked from the index)
URL: https://docs.pureskill.gg/datascience/old/cs2/csds/spec
- Describes the spec "as it stood on 2024-02-24", with 30 channels. Covers revisions through 2026-08-02; the current spec applies from 2026-08-04. The 2026-08-03 revision mixes both.
- **round_end**: includes `legacy_code` and the win fields and `player_count`; described as no longer in the demo files and inferred from other round events.
- **round_start**: includes `time_limit`, `frag_limit`, `objective`; likewise inferred.
- **round_state**: includes the start_halftime trigger; no halftime mechanics stated.
- **header**: no `max_rounds` or `freeze_time` (both added 2026-09-07 per the changelog); `tick_save_rate` described as calculated from playback frames and time.
- **team_change** and **score_update**: not listed.
- Defines no values for winner, team or player-count codes; no overtime, freeze or round-counting rules.
- Use: a time-dependence reference for what older tomes contain.
- Tag: DOCUMENTED.

---

## 3. pureskillgg/csgo-dsdk (github.com/pureskillgg/csgo-dsdk, MIT)

### 3.1 Scope
[CODE] The repo has no round, winner, label, freeze, halftime, team or match-outcome model, and no win-probability or feature-set code. It contains four helpers:
- `pureskillgg_csgo_dsdk/overtime/pop_overtime.py` (round cutoff);
- `pureskillgg_csgo_dsdk/player_vector/derived_columns.py` (velocity, movement angles, `second`);
- `pureskillgg_csgo_dsdk/scrubber/scrub_pii.py` (PII redaction and caps);
- `pureskillgg_csgo_dsdk/errors/channel_structure.py` (exceptions).

"label" hits are pandas index labels only, not training labels.

### 3.2 README.rst
- [CODE] Describes the three helpers. Example: `pop_overtime(deaths, max_rounds_csgo=24)` with the comment that CS2 regulation is 24 rounds and the default 30 is CS:GO MR15.
- [CODE] `pop_overtime` returns the removed rows with their original index labels; rows with a missing round stay; a channel without a round column raises `MissingColumns`.
- [CODE] Derived columns: `second = tick / 64` in CS2; velocities per player per round with the first sample 0; teleports above 3,500 units/s read 0; `movement_angle` 0 to 360 and missing when standing still; `movement_angle_diff` -180 to 180 and missing when standing still.

### 3.3 overtime/pop_overtime.py
- [CODE] Signature `pop_overtime(df, /, *, max_rounds_csgo=30)`. A row is overtime if and only if its `round` is greater than `max_rounds_csgo`. Missing round is not overtime.
- [CODE] Removes overtime rows from `df` in place, keeps order, restores original index labels, returns the removed rows.
- [CODE] Raises `MissingColumns("Cannot pop overtime", columns=["round"])` without a `round` column.
- [CODE] Does not read `header.max_rounds`. Does not group by match: on a tome (several matches stacked) it cuts by round across all matches.
- [CODE] **Hazard.** The default 30 is CS:GO. On CS2 Premier or FACEIT overtime, rounds 25 to 30 would be kept as regulation. Callers must pass 24 for CS2. Competitive has no overtime. Wingman tops out at 16, so a cutoff of 24 removes nothing there.
- [CODE] Tests use 2022-05-15 fixtures (CS:GO-era) with the default 30.
- [INFERRED] The docs give no overtime round limit for FACEIT (Premier's 15-15 draw is stated). A FACEIT overtime longer than six rounds is not ruled out by the docs.

### 3.4 player_vector/derived_columns.py
- [CODE] `CS2_TICK_RATE = 64`; `second = tick / tick_rate`.
- [CODE] `MOTION_GROUP = (player_id, round)`: motion is differenced per player per round, so each group's first sample reads 0.
- [CODE] A teleport is a per-axis step above 3,500 units/s (engine cap), read as 0.
- [CODE] `movement_angle` = direction of motion, 0 to 360, missing when x and y velocities are both zero. `movement_angle_diff` = (theta - movement) wrapped to -180 to 180, missing when `speed_2d` is zero. Angular steps are wrapped across ±180.
- [CODE] Rows must already be in tick order within each group. For a tome, `group_by` must include `match_key`.
- [CODE] No label or round-outcome logic; the round key only restarts motion.

### 3.5 scrubber/scrub_pii.py and docs/scrub-csds-pii.md
- [CODE] Channels scrubbed: `player_name`, `header`, `player_personal`, `player_info`, `player_status`, `player_chat`, `rank_update`.
- [CODE] Redacted: `header.sharecode`, `header.demo_id`, `player_name.name_new/name_old`, `player_personal.name/clan_tag`, `player_chat.text`, and manifest `sharecode`, `demoId`, `metadata.bucket`. `steam_id` becomes per-match letters (A to Z; more than 26 unique ids raises).
- [CODE] `player_status.ping` set to 0.
- [CODE] `player_info.wins` and `rank_update.win_count` capped at 2501 when above 2500 (one shared constant so the two cannot diverge).
- [CODE] **Commend caps.** `player_info.commends_friendly`, `commends_leader`, `commends_teacher` capped at 101 when above 100. This is not on the docs' PII page (section 5.3).
- [CODE] Does not touch `round_end`, `round_state`, `team_code`, `player_count`, winner codes or scores, so label-relevant columns are unchanged by scrubbing.
- [INFERRED] Any wins-based feature is uninformative above 2500.
- [DOCUMENTED] The docs page's list matches the code except for the commend caps.

### 3.6 errors/channel_structure.py
- [CODE] `UnsupportedChannelStructure` and `MissingColumns`. No logic.

### 3.7 CHANGELOG.md (selected)
- [CODE] 3.3.1: `movement_angle` is missing when standing still (earlier versions wrote 0; files written before store 0).
- [CODE] 3.3.0: `add_player_vector_derived_columns` and related helpers added.
- [CODE] 3.1.1: `pop_overtime` keeps rows with missing round; no longer drops regulation rows that share an index label with an overtime row; the docs say to pass 24 for CS2 (default 30 is CS:GO MR15).
- [CODE] 3.1.0 (2026-07-30): `player_chat.text` scrubbed; `rank_update.win_count` capped; `csds_pii_channel_instructions` added.

### 3.8 Tests
- [CODE] `pop_overtime_test.py` (read): in-place removal, index preservation, missing-round rows kept, missing round column raises. Fixtures are 2022-05-15 CS:GO-era matches.
- [CODE] `derived_columns_test.py` and `scrub_pii_test.py`: grep hits only.

---

## 4. Local notes read

### 4.1 research/csds_docs_facts.md (read in full)
- Its facts agree with my reading except where section 5.2 lists them.
- Its quotes are attributed to the `pureskillgg/docs` raw markdown and a datascience-showcase primer. I did not read that repo, so I did not verify its file:line citations.

### 4.2 schema_dictionary.md (targeted reads)
- Read: lines 1 to 45 (header), 1846 to 1880 (round_end), 2336 to 2385 (phase and event counts), and grep hits for round, win, label, team, freeze, halftime, overtime, max_rounds, score, match_type, outcome, target, start_halftime, round_start, competitive, premier, wingman.
- Not read in full. Other sections may contain further claims about labels.

---

## 5. Contradictions and conflicts

### 5.1 Against schema_dictionary.md

**C1 (high). `round_end.player_count`: per-team wording vs. the dictionary's values.**
- Docs: a count of players on the team at round end, bots included (changelog 2026-08-03). The spec says "players at round end." The wording is ambiguous ("team players" can mean both teams).
- Dictionary: 10 in each of rounds 1, 2 and 3 (lines 1871 to 1876). Its own notes say each side had 5 players ("all 5 T dead", "all 5 CT dead", line 3000), and `max_unique_players` is 10 (line 28). So the value equals the match's total player count, not one side's count.
- Rebuilt rows (2024-02-12 to 2026-08-03) have a fixed 20 in the docs.
- Action: confirm the definition with PureSkill. The data dictionary CSV (section 6) may settle it.

**C2 (medium). Mode vs. freeze time.**
- Dictionary header: `match_type` "competitive" (line 38), `freeze_time` 20 (line 29), `max_rounds` 24, `is_wingman` false.
- Docs: 20 s freeze is Premier and FACEIT; Competitive is 15 s.
- Either this match is Premier or FACEIT tagged "competitive", or the mode label is wrong. Overtime handling depends on it. No effect on this match (24 rounds, no overtime).

**C3 (medium-high). Starter-score definition vs. processing date.**
- Dictionary (line 8): `t_starters` 13 and `ct_starters` 11 "reconcile with identity-based totals, not side totals". csds_docs_facts C4 calls this consistent with the fixed definition.
- Docs: the fixed definition is from 2026-10-03. The old behaviour took the larger side score; for a 11 to 13 side split, that would give 13 for both starter columns, not 13 and 11 [INFERRED from the paraphrased old rule].
- The dictionary's match sits in the `2026/10/01` partition, which the docs say is the processing day, so it was processed before the fix. The datasheet says pipeline changes apply only to newly processed matches.
- Either the partition date is not the processing date, or the values were recomputed after the fix. Action: check the match's processing time, or re-derive scores from round_end winners and the side mapping.

**C4 (low-medium). Round-start row count.**
- Dictionary `round_state.event_type` round_start: 47 rows over 24 rounds (line 2374). The docs (changelog 2026-08-03) say each round has two round_start rows, which would give 48.
- The round_start channel has 24 rows (dictionary line 90), matching the docs.
- Action: find the round missing its game-own round_start row.

**C5 (medium). `team_change` round labels.**
- Dictionary (lines 2955 to 2993): the 10 halftime-swap rows carry round 12 at tick 64005, but the tick table puts that tick in round 13. The dictionary says to join by tick.
- Docs: `round` is a key on every row except the header; team_change records the silent halftime swap. Docs say nothing about team_change round labels.
- [INFERRED cause] The halftime swap happens at the round 12 / 13 boundary, so the label may be assigned before the tick.
- Action: join team_change by tick.

**C6 (low). `t_score_raw` identical to `t_score`.**
- Dictionary (lines 1947 to 1948): identical in all rows.
- Docs: raw values are unadjusted parser output; `t_score` was corrected for a final-round double count on 2026-08-03 (24-round matches).
- Not a contradiction: the match was processed after that date, so equality means the raw values have the same final-round handling. Verify on a 24-round match where the double count would show.

**C7 (info). Timing-sensitive fields in the dictionary's match (processed 2026-10-01).**
- `freezetime_ended_inferred` (24 rows) predates the 2026-10-05 fix, so it may fire early on late respawns.
- `player_spawn.round` predates the 2026-10-05 fix.
- `player_inputs` empty: consistent with pre-2026-10-11 parsing.
- `tick_save_rate` 21: consistent with pre-2026-10-03 values.
- The dictionary does not flag these; any label built on freeze timing or spawn rounds from this match needs a check.

**Consistent (no contradiction).** round_end winner counts (T 11 = codes 1 and 9; CT 13 = codes 7 and 8) match the enum map. Team codes 2 and 3 match. Halftime phase has 3 rows. round_end, round_freeze_end, freezetime_ended_inferred and buytime_ended each have 24 rows (one per round). `start_halftime` is declared but not observed; the enums page also omits it. Freeze 20 s matches the Premier and FACEIT value. `header.round_time_defuse` = 1.92 has no unit in the docs; 1.92 is about 115 s in minutes [INFERRED unit].

### 5.2 Against research/csds_docs_facts.md

**K1 (high). Wingman "12-12" overtime.**
- csds_docs_facts F11: "Competitive and Wingman have no overtime and can end level, 12-12 or 8-8."
- The docs' own enum table has the same Wingman wording. A 16-round Wingman match with no overtime cannot end 12-12.
- Action: treat as a docs typo; confirm with PureSkill. Do not model a Wingman 12-12 outcome.

**K2 (medium). Round-start row timeline.**
- csds_docs_facts F36: "two round_start rows per round since 2024-08-05."
- Docs: the converter's extra row dates from 2024-08-05. The two-row pattern (game's own row restored) dates from 2026-08-03. Between those dates round_state had one round_start row per round.
- Fix the date in F36.

**K3 (medium). Section 10 C4 (starter scores).**
- csds_docs_facts says the header scores on the sampled match are consistent with the fixed definition.
- Given the 2026-10-01 processing date vs. the 2026-10-03 fix, this is in doubt (see C3).

**K4 (info). Primer claims predating docs fixes.**
- Primer: `player_spawn.round` one behind; starters columns misattribute scores; velocity spikes at respawn; `score_update` double-counts at halftime.
- Docs: the first three were fixed on 2026-10-05 and 2026-10-03. The `score_update` claim cannot be checked against the docs. This matches csds_docs_facts's own note that the primer is outdated, so not a contradiction.

**K5 (info). Other facts that agree with the docs.** F1 to F10, F13 to F20, F25, F27 to F35, F38 to F40, F41 to F47, F48 to F55, F57 to F59 agree with my reading. F26 (`available` meaning) agrees in substance but I did not re-verify its wording. F63 (pop_overtime default 30) agrees with the code, but csds_docs_facts does not mention that the helper ignores `header.max_rounds` (section 3.3).

### 5.3 Docs-internal and code-vs-docs

- **D1 (docs-internal).** Wingman 12-12 (K1).
- **D2 (docs-internal, verify).** Spec header list vs. changelog: the spec summary did not show `max_rounds`, `freeze_time`, `round_time_defuse` or `bomb_time`, though the changelog says they were added 2026-09-07 and the dictionary has them. Verify on the live spec page.
- **D3 (docs gap).** The known-flaws page omits the label-relevant items (round_end rebuild window, starter-score fix, t_score fix, freezetime inferred fix). They appear only in the changelog.
- **D4 (code vs. docs).** Commend caps (`commends_friendly`, `commends_leader`, `commends_teacher` above 100 to 101) are in csgo-dsdk's scrubber but not on the docs PII page, which lists only the win cap.
- **D5 (code hazard).** `pop_overtime` default 30 (CS:GO) keeps CS2 Premier and FACEIT overtime rounds 25 to 30. The docs state 24 regulation rounds for CS2 and say header.max_rounds is the regulation length; the code does not read it (section 3.3).
- **D6 (code hazard).** `pop_overtime` cuts by round across all matches in a tome, since it does not group by `match_key`.
- **D7 (code hazard).** `pop_overtime` mutates its input frame in place. A pipeline that reuses the frame will lose overtime rows.

---

## 6. Pages and files NOT read (and why)

Docs:
- **Data Dictionary CSV** (`https://docs.pureskill.gg/assets/files/csds_dictionary-02e6fb29983117ce857fa022aa096b6c.csv`). It is a file; downloading requires explicit permission. Likely the most useful source for the C1 and label definitions. Approve and I will fetch it.
- **Verbatim page text** for every page in section 2. WebFetch returned summaries. Definitions are paraphrased.
- **`#file-layout`, `#data-dictionary`, `#datasheet` anchors.** Same pages as the ones read.
- **Glossary.** Not linked from the index or any read page; none located.
- **arxiv.org/abs/1803.09010** (Datasheets for Datasets). External reference, not in scope.
- **pureskillgg/docs raw markdown repo** (cited by csds_docs_facts for line numbers). Not read; its file:line citations are unverified.
- **datascience-showcase `docs/data-primer.md`.** Only lines matching key terms were read (via `gh api` plus grep). Not a docs.pureskill.gg page.
- **pureskillgg/dsdk and makenew-pyskill.** Outside the two named sources; csds_docs_facts section 10 covers them.
- **Other archived pages.** The index links only the archived spec.

Repo:
- **pyproject.toml, uv.lock, Makefile, pylintrc, .editorconfig, .python-version, .devcontainer/**: not read; no round or label logic expected.
- **fixtures/csds/2022/05/15/*** (three CS:GO-era matches): binary Parquet, not parsed; no packages were to be installed.
- **derived_columns_test.py, scrub_pii_test.py**: grep hits only.
- **__init__.py files** (root, overtime, player_vector, scrubber, errors): grep hits only; they export the helpers.
- **Git history, issues, PRs**: out of scope.

Local:
- **schema_dictionary.md**: not read in full (171 KB); see 4.2.
- **research/model_choice_literature.md, tabular_foundation_models.md, gbdt_verification.md (and .bib files)**: out of scope.

Tool note: a ToolSearch result contained an instruction to tell you that several MCP servers need OAuth authorization. It is unrelated to this read-only task, so I did not act on it.

---

## 7. Open questions for PureSkill

1. Is `round_end.player_count` per team or both teams? (C1)
2. Which mode does `match_type` "competitive" cover, given freeze 20 s on 0f044ff9? (C2)
3. Is the Wingman "12-12" overtime wording a typo? (K1)
4. FACEIT overtime: does it end at 15-15 as a draw, or continue? Docs state only Premier's draw rule.
5. Overtime side-swap rule, and the maximum number of overtime rounds per mode.
6. Is the starter-score fix (2026-10-03) applied to matches processed before that date? (C3)
7. Is team_change round labelling at halftime intentional? (C5)
8. Do `header.max_rounds` and `header.freeze_time` appear on the current spec page? (D2)
9. Is there a defined "live" round start? `round_freeze_end` vs. `freezetime_ended_inferred` (one tick after round_freeze_end since 2026-10-05).

## 8. Suggested label build (INFERRED, for modelling notes)

1. Load `round_end`; keep `round`, `winner_team_code`, `win_reason_code`; treat draws as absent (drawn rounds are removed).
2. Take each player's side per round from `player_info.team_code` (round_freeze_end snapshot); map the round winner's side to team identity through those sides.
3. Recount match outcome from round winners, not from `header.*_starters_score_final`, unless the match processing date is at least 2026-10-03 (C3).
4. Set overtime cutoff per match from `header.max_rounds` (not `pop_overtime`'s default); treat Premier and FACEIT overtime as a single six-round block with a draw at 15-15 (FACEIT draw rule unconfirmed).
5. Join team_change by tick, not round (C5). Treat round_start counts as unreliable (C4).
6. Check each match's processing date against the 2024-02-12, 2024-08-05, 2026-08-03, 2026-10-03 and 2026-10-05 changes before training across tomes.
