# Data regimes in the CSDS revision list (2025-07 to 2026-10)

Collected 2026-10-13. Research only. Live docs were read with GET requests (curl, and WebFetch for one page). No datasets were downloaded. This is the only file written.

## Tags and sources

- **VERIFIED**: stated on a docs page I opened in this session, or a number I computed from one. The page is named.
- **UNVERIFIED**: from a local file I did not re-check, an inference, or not found on any page I opened.
- Changelog text was read from the raw page HTML. WebFetch returned a paraphrase and said it could not reproduce the full changelog, so I did not use it for any claim.

Short names (all under https://docs.pureskill.gg/datascience/adx/cs2/csds/ unless shown):

| Short name | Page |
|---|---|
| REV | revision-stats |
| CL | changelog |
| CLC | changelog-by-channel (I checked its dates only) |
| GTD | getting-the-data |
| FLAW | known-flaws |
| DS | datasheet |
| SPEC | spec (current) |
| ARCH | https://docs.pureskill.gg/datascience/old/cs2/csds/spec (archived) |
| ENUM | enums |

## Bottom line

- **VERIFIED.** Four breaks matter most for round labels. On 2024-02-12, round_end was rebuilt from round_state. On 2026-08-03, real round_end and round_state rows came back. On 2026-10-03 and 2026-10-05, starter scores, spawn round, velocities and the freeze-inferred row were fixed. On 2026-10-06, the ten derived player_vector columns stopped being stored.
- **VERIFIED for the page numbers, UNVERIFIED for the tome contents.** Our 2026-10 tome (tomes\2026-10-01_to_08) spans at least two schema states, and 65% of its matches come from one revision (2026-10-08).
- **Judgment (mine).** Most comparable month: 2026-09. Least comparable: 2025-07 to 2026-07, all of which predate the 2026-08-03 restore.
- **Suggested additions:** 2026-09, 2026-06 and 2025-12 (section 3).

## 1. Timeline of data regimes

### 1a. History before the revision list

The revision list starts on 2025-07-19, so these rows are context.

| Date (processing day, UTC) | What changed | Source | Tag |
|---|---|---|---|
| 2023-11-05 | CS2 data begins with 30 channels. header.tick_save_rate is a fixed 21. | CL | VERIFIED |
| 2023-11-23 | Some player ids about 65,000 too high are corrected. | CL | VERIFIED |
| 2023-12-09 | player_controller_id added to player_status and player_personal. player_blind keeps only living players. | CL | VERIFIED |
| 2024-02-06 to 02-11 | CS2 update of 2024-02-06 broke parsing. No matches are dated 2024-02-08 to 02-11. | CL | VERIFIED |
| 2024-02-12 | CS2 demos stop carrying round start and end events. round_end win codes are rebuilt from round_state (8 for CT, 9 for T). player_count is fixed at 20. round_start is one row short. round_state loses its round_start and round_end rows. | CL | VERIFIED |
| about 2024-02-24 | Publisher fix after AWS changed Data Exchange permissions. Some matches were published late. | CL | VERIFIED |
| about 2024-03-01 | round_mvp goes empty for most matches. Rows return on 2026-08-03. | CL | VERIFIED |
| 2024-04-28 and 2024-06-06 | Demos recorded after the CS2 updates of 2024-04-25 and 2024-05-29 parse again. | CL | VERIFIED |
| 2024-07-07 | freezetime_ended_inferred added: one row per round, at the first tick any player moves. header.unique_steamids added. | CL | VERIFIED |
| 2024-07-28 | freezetime_ended_inferred ignores the first 5 ticks of a round. | CL | VERIFIED |
| 2024-08-05 | The converter adds a round_start row one tick after round_poststart. Player ids are remapped round by round. | CL | VERIFIED |

### 1b. Revision window, 2025-07-19 to 2026-10-11

| Date | What changed | Source | Tag |
|---|---|---|---|
| 2025-08-21 | player_status.reload_visually_complete is null on every row from this date. The first revision to carry it is 2025-08-22. | CL, REV | VERIFIED |
| about 2026-01-22 | CS2 demos stop carrying player_action events. The channel is removed on 2026-08-03. | CL | VERIFIED |
| 2026-01-25 | header.network_protocol is always 1234 (parser 2.7.1). | CL, REV | VERIFIED |
| 2026-04-29 | Five aim-punch columns are no longer written (parser 2.8.0 and 2.9.0). The first revision to carry it is a 597-match backlog. | CL, REV | VERIFIED |
| 2026-06-05 | Matches without platform metadata are published, without header.providence, rank_raw and rank_platform. Bot-takeover matches are published (they used to fail). This changes which matches appear. | CL, REV | VERIFIED |
| 2026-06-07 | A duplicate player_id within a round is resolved by controller id. Matches that still duplicate fail. | CL, REV | VERIFIED |
| 2026-06-27 | Backlog revision of 1,007 matches. | REV | VERIFIED (size). Cause not stated. |
| 2026-07-03 | Backlog revision of 525 matches. | REV | VERIFIED (size). Cause not stated. |
| 2026-07-07 | player_info.rank_type added (11 Premier, 12 Competitive, 7 Wingman, 0 or -1 unknown). player_info.elo_platform added for FACEIT. tick.previous_phase order fixed. | CL | VERIFIED |
| 2026-07-08 | Parquet compression changed from GZIP to ZSTD. Readers need ZSTD support. | CL, GTD | VERIFIED |
| 2026-07-09 | CS2 update removed the player property the parser read player_inputs from. The channel was empty on every match recorded after that update, until parser 5.7.0. | CL (2026-10-11 entry), FLAW | VERIFIED (parsing effect). Gameplay content not described. |
| 2026-08-03 | **Largest break in the window.** round_end win fields and player_count (bots included) are real again. round_start is real again, one per round including round 1. round_state round_start and round_end rows return, so each round has two round_start rows. The final-round double count in t_score and ct_score is fixed for 24-round matches. 14 channels are added, including team_change, score_update, player_inputs, player_sound, rank_update, molotov_fire and grenade_vector. item_remove and player_action are removed. round_mvp and bot_takeover get rows again. The revision dated 2026-08-03 holds both layouts (31 objects and 43 objects per match). | CL, GTD, REV | VERIFIED |
| 2026-08-22 | header.is_wingman, is_gotv_recording and max_unique_players added. Molotov fields changed. Bool types. 62 columns that were never written dropped from the index. | CL | VERIFIED |
| 2026-09-07 | player_info.player_controller_id added. header.freeze_time, round_time_defuse (minutes), bomb_time and max_rounds added, null when the demo does not record them. bomb_time is null in every match checked. Duplicate player ids within a round now fail processing. Before this date, a failed close pushed a day's matches into the next revision. | CL, GTD | VERIFIED |
| 2026-10-03 | header.t_starters_score_final and ct_starters_score_final now give each starting team's own score through swaps. Before, they gave the larger side score. Starter avg rank and wins are averaged over the first half. Empty team is null, not -1. tick_save_rate is 64 (was 21). FACEIT rank and elo fixed. | CL | VERIFIED |
| 2026-10-05 | freezetime_ended_inferred no longer fires early on a late respawn. In 8 of 927 rounds in 50 test matches it had fired 43 to 3,520 ticks early. player_spawn.round fixed (7,913 rows in the same test matches carried the prior round). player_info and player_personal get a row for every player who spawns in a round that reaches freeze end. Velocities are computed per player per round. header.rank_type added. header.network_protocol and item_equip.def_index removed. Parser 5.5.0 drew grenade ids at random for part of the day; 5.6.0 numbers them in first-seen order. Many type changes. | CL, FLAW | VERIFIED |
| 2026-10-06 | Ten derived player_vector columns are no longer stored. Compute them on load with csgo-dsdk 3.3.1 or later. The changelog says the file is now under a third of its former size. | CL, FLAW | VERIFIED |
| 2026-10-08 | Hand-uploaded matches (platform unknown; match_date is the processing date) include many FACEIT matches from this date. Daily volume steps up to 1,135 matches on 10-08 and 1,162 on each of 10-09 to 10-11. | FLAW, REV | VERIFIED (both facts). Link between them not stated. |
| 2026-10-11 | Parser 5.7.0. player_inputs has rows again on every kind of demo (6,600 to 82,000 rows per match in a 50-match sample). world_item_vector.prev_owner_id is stable on reparse. | CL, FLAW | VERIFIED |

### 1c. Freeze time, bomb timer, mode and game patches

- **VERIFIED.** ENUM gives freeze time as 20 s (Premier, FACEIT), 15 s (Competitive) and 10 s (Wingman). A few rounds run longer than the setting.
- **VERIFIED.** No changelog entry changes freeze-time values. The freeze entries are column additions (2026-09-07) and inferred-event fixes (2024-07-07, 2024-07-28, 2026-10-05).
- **VERIFIED.** header.freeze_time, header.max_rounds and header.round_time_defuse first appear on 2026-09-07. header.bomb_time appears the same day and is null in every match checked. Whether older matches have the column as null or lack it is **UNVERIFIED**.
- **VERIFIED.** header.is_wingman appears on 2026-08-22. Before 2026-09-07 there is no documented header.max_rounds. A label that needs the regulation length cannot read it for matches processed earlier. The fallback choice is **UNVERIFIED**.
- **VERIFIED.** The CS2 updates named in CL are 2024-02-06 (parsing broke), 2024-04-25 and 2024-05-29 (reparse), and 2026-07-09 (parser property removed). None is described as changing round rules, economy, win conditions or bomb timing.
- **UNVERIFIED.** Whether any balance or bomb-timer patch changed round outcomes in 2025-07 to 2026-10. The docs do not say, and Valve's patch notes were outside this docs-only scope.

## 2. Comparability to our 2026-10 data

### 2a. Our data

- **UNVERIFIED.** tomes\2026-10-01_to_08 holds revisions 2026-10-01 to 2026-10-08 (per HANDOFF.md). I did not open the tome.
- **VERIFIED.** Those eight revisions hold 1,752 matches and 26.57 GB (REV, computed). The local EDA note gives the same 1,752 matches (**UNVERIFIED**, not re-counted).
- **VERIFIED.** 2026-10-08 is 1,135 of the 1,752 matches, or 65% (REV). The local note says it is 72% of clean rounds (**UNVERIFIED**).
- **VERIFIED.** REV size per match, in MB: 31.3, 33.2, 36.7 and 31.9 for 10-01 to 10-04; 25.8 for 10-05; 10.0, 6.8 and 10.0 for 10-06 to 10-08.
- **VERIFIED.** The 10-06 size drop matches the removal of the derived columns (CL). The causal link is my inference (**UNVERIFIED**).
- **UNVERIFIED.** The tome therefore probably mixes at least two schema states. Before pooling, check header.ppp_version on each match and whether player_vector still has the ten derived columns.

### 2b. Month by month

Monthly figures are sums of REV's daily rows (computed, **VERIFIED**). Notes in the fourth column are from REV and CL unless tagged.

| Month | Revs | Matches | GB | Notes | Closeness to 2026-10 |
|---|---|---|---|---|---|
| 2025-07 | 12 | 907 | 32.58 | 31 objects per match. No data-change marker. Gap on 07-31. | Least |
| 2025-08 | 21 | 1,347 | 52.63 | Marker on the 08-22 row for the 08-21 change (reload column). Gaps 08-07 to 08-12, 08-17 to 08-19 and 08-21. | Least |
| 2025-09 | 30 | 1,974 | 76.12 | No marker, no backlog. | Least |
| 2025-10 | 26 | 630 | 26.82 | 2025-10-20 has 1,240 files for 27 matches. At 31 objects per match, that is 40 match-equivalents. Cause **UNVERIFIED**. | Least |
| 2025-11 | 22 | 2,158 | 74.50 | Backlog of 919 on 11-18 (43% of the month). Gap 11-14 to 11-17. | Least (composition) |
| 2025-12 | 31 | 2,132 | 82.34 | No marker, backlog or gap. | Least (clean baseline) |
| 2026-01 | 31 | 2,738 | 106.61 | Markers on 01-22 (about) and 01-25. | Least |
| 2026-02 | 28 | 2,907 | 117.51 | 2026-02-25 has 4,557 files for 146 matches, one match-equivalent more than listed. Cause **UNVERIFIED**. | Least |
| 2026-03 | 31 | 3,255 | 116.50 | No marker, backlog or gap. | Least |
| 2026-04 | 30 | 2,814 | 116.74 | Backlog of 597 on 04-29 (21%). Aim-punch marker. | Least |
| 2026-05 | 16 | 1,871 | 77.67 | Backlog of 754 on 05-29 (40%). No revision from 05-17 to 05-28 (12 days). | Least |
| 2026-06 | 29 | 2,292 | 95.90 | Backlog of 1,007 on 06-27 (44%). Bot-takeover and no-metadata matches published from 06-05. | Least (composition) |
| 2026-07 | 31 | 3,469 | 133.09 | Backlog of 525 on 07-03 (15%). CS2 update on 07-09 emptied player_inputs until 10-11. | Low |
| 2026-08 | 31 | 3,450 | 122.38 | 08-01 and 08-02 on the old layout. 08-03 mixed. 08-04 onward on the new layout. | Partial (08-04 to 08-31) |
| 2026-09 | 30 | 3,023 | 101.37 | One marker (09-07). 43 objects per match. Median 33.3 MB per match. 09-23 and 09-27 each have one match-equivalent more than listed. Cause **UNVERIFIED**. | Most |
| 2026-10 (01 to 11) | 11 | 5,238 | 61.92 | 10-08 to 10-11 at 1,135 to 1,162 matches per day. From 10-06, about 10 MB per match. | Our data is 01 to 08 |

Object counts. **VERIFIED.** REV's file counts divided by match counts give 31 objects per match through 2026-08-02 and 43 from 2026-08-04, which GTD confirms. The 2026-08-03 revision has 177 matches and 6,915 files. If it holds only 31-object and 43-object matches, that is 58 and 119. That split is my arithmetic (**UNVERIFIED**).

Gap days. **VERIFIED.** Of 450 days from 2025-07-19 to 2026-10-11, 40 have no revision. REV says days covered by an outage have none.

### 2c. Least comparable: 2025-07-19 to 2026-08-02

This window has 340 revisions and 28,712 matches (71% of the page's 40,205). Against the 2026-10 data:

- **VERIFIED.** round_end win codes are only 8 and 9, and player_count is fixed at 20 (CL, 2024-02-12). Whether winner_team_code is correct in those rows is not stated (**UNVERIFIED**).
- **VERIFIED.** round_state has no round_start or round_end rows, and round_start is one row short (CL).
- **VERIFIED.** team_change and score_update are absent, so there are no halftime swap rows (added 2026-08-03, CL).
- **VERIFIED.** 31 objects per match (GTD).
- **VERIFIED.** No header.max_rounds, freeze_time, is_wingman or rank_type. These arrived between 2026-07-07 and 2026-09-07 (CL).
- **VERIFIED.** The t_score double count in 24-round matches was fixed on 2026-08-03 (CL). Starter scores were the larger side score until 2026-10-03 (CL).
- **VERIFIED.** Spawn round, velocities, the freeze-inferred row and missing late-spawn player rows were wrong until 2026-10-05 (CL).
- **VERIFIED.** DS says demo files are kept 30 days and replays 7, and a match cannot be reprocessed after that. Matches from this window therefore keep rebuilt rows. That they were never backfilled is **UNVERIFIED**.

Within this window, 2025-11 (43% backlog), 2026-04 (21%), 2026-05 (40%) and 2026-06 (44%) are least comparable on composition, because one processing day dominates the month.

### 2d. Why 2026-09 is the closest

- **VERIFIED.** 43 objects per match, and the post-2026-08-03 schema (CL, GTD).
- **VERIFIED.** Real round_end, two round_start rows per round, round_state start and end rows, team_change and score_update.
- **VERIFIED.** 30 revisions, no backlog, no gap days, and one marker (09-07).
- **VERIFIED.** Size per match (median 33.3 MB) is close to 10-01 to 10-04 and far from 10-06 onward (about 10 MB).
- **VERIFIED.** What 2026-09 lacks, compared with 10-03 onward: the starter-score fix, the spawn-round, velocity and freeze-inferred fixes (10-05), the derived-column removal (10-06), and the player_inputs restore (10-11).
- **UNVERIFIED.** Whether the missing fields can be recomputed for 2026-09 matches, for example velocities from positions, or starter scores from round winners and side maps. Both are plausible, but I did not test them.

## 3. Recommended months to add for variety

Matches and GB are sums of REV's daily rows (computed, **VERIFIED**).

| Month | Matches | GB | Why it was chosen |
|---|---|---|---|
| 2026-09 | 3,023 | 101.37 | Same label-relevant schema as the early 2026-10 days, with normal daily volume and no 10-08 hand-upload block. It tests whether the 10-08 batch behaves differently from ordinary matches. |
| 2026-06 | 2,292 | 95.90 | A different match mix: bot-takeover and no-metadata matches are published from 06-05, and a 1,007-match backlog lands on 06-27. It sits on the pre-08-03 label rebuild, so use it for feature tests until winner correctness is checked (**UNVERIFIED**). |
| 2025-12 | 2,132 | 82.34 | A quiet month: no backlog, no data-change marker, no gap days, 31 objects per match. It is a clean baseline for the rebuilt round_end regime, a year from 2026-10. |

Alternative not chosen: 2026-07 (3,469 matches, 133.09 GB). It spans the 2026-07-09 CS2 update, but it also has a backlog and the rebuilt labels.

## 4. Checks against the live docs, and corrections to our notes

- **VERIFIED.** SPEC (live). The header table has no rows for freeze_time, round_time_defuse, bomb_time or max_rounds. max_rounds appears only as a dependency of the starter columns. CL says these four were added 2026-09-07, and ENUM points users to header.max_rounds, header.freeze_time and header.round_time_defuse. This confirms item D2 in csgo_and_docs_digest.md.
- **VERIFIED.** ENUM says Competitive and Wingman "can end level, 12–12 or 8–8". Wingman has 16 regulation rounds and no overtime, so a 12-12 Wingman result is impossible. Calling this a docs error is my inference (**UNVERIFIED**). This confirms K1 in csgo_and_docs_digest.md.
- **VERIFIED.** ARCH has no max_rounds, freeze_time, rank_type or is_wingman in its header, and no team_change or score_update. This was checked by text search.
- **VERIFIED.** GTD says the changelog records the first revision with zips. The changelog has no zip entry (its only "zip" match is inside "GZIP"). The zip start date is therefore not on any page I opened (**UNVERIFIED** as a missing date).
- **VERIFIED.** csds_docs_facts.md F36 says two round_start rows per round since 2024-08-05. CL dates the converter's extra row to 2024-08-05 and the game's own row to 2026-08-03. F36 needs its date corrected (this matches K2 in csgo_and_docs_digest.md).
- **VERIFIED.** REV: 410 revisions, 40,205 matches, 1,384,095 files, 15 data-change markers. Summing the GB column gives 1,394.68 GB, which matches the page's 1.39 TB.
- **VERIFIED.** CLC has the same 32 dates as CL. I did not read its text.

## 5. Open questions

All of these are **UNVERIFIED**.

1. Why the daily volume steps up from 10-08 (1,135 to 1,162 matches a day). Whether this is the same change as the FACEIT hand-upload note on the known-flaws page.
2. Whether winner_team_code is correct in rebuilt round_end rows (2024-02-12 to 2026-08-03).
3. Whether any game balance or bomb-timer patch fell in the window. The docs record none.
4. The cause of the file-count anomalies: 2025-10-20 (about 13 extra match-equivalents) and one extra match-equivalent each on 2026-02-25, 2026-09-23 and 2026-09-27. Duplicates are one possibility, since the page says a demo processed twice appears twice.
5. Whether our tome has the derived player_vector columns, and which ppp_version each match has.
6. Whether pre-2026-09-07 matches have header.freeze_time and max_rounds as null or as absent columns.
7. Local, not re-checked: notes\eda\batch_effect_2026-10-08.md reports 92.7% of 10-08 matches with unknown match type and platform, against 3.9% on the other days. It also reports 20% wingman on the other days against 2.2% on 10-08.
8. Local, not re-checked: HANDOFF.md says the local folders for 2026-10-09 and 10-10 hold 2,301 and 2,758 match folders. The revision list has 1,162 per day. Duplicates are a likely reason.

## 6. Method

- Live reads: revision-stats (HTML table parsed in memory), changelog, changelog-by-channel (dates only), getting-the-data, known-flaws, datasheet, spec (current and archived) and enums. Changelog and spec text were extracted from the raw HTML.
- Computations: monthly and window sums, files per match, size per match, and gap days, all from REV's 410 rows.
- Local files read for context only: HANDOFF.md, notes\research\csds_docs_facts.md, notes\spec\csgo_and_docs_digest.md, and a directory listing of tomes\.
- No datasets, downloads or writes outside this file. Temporary copies of fetched pages were not kept.
