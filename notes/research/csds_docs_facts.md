# PureSkill CS2 CSDS: documented facts for round win-probability labels and features

Collected 2026-10-12. Scope: the docs.pureskill.gg CSDS pages and the pureskillgg GitHub repos. Nothing under E:\ or in the tomes was touched, and no packages were installed.

## How to read this file

- **Quotes** are verbatim from the raw markdown that the docs site is built from. The source is the `pureskillgg/docs` repo, path `docs/datascience/adx/cs2/csds/<page>.md`, fetched with `gh api` on 2026-10-12. I also read the live pages with WebFetch. They agree with the raw text on every fact below.
- **Line numbers** such as `enums.md:197-199` refer to those raw files.
- **Confidence levels**
  - **documented**: stated in a cited source.
  - **inferred**: my reading across sources.
  - **unverified**: not stated in the sources, or not checked against data.

Page short names used below:

| Tag | Page | Raw file (pureskillgg/docs) |
|---|---|---|
| IDX | https://docs.pureskill.gg/datascience/adx/cs2/csds/ | `csds/README.md` |
| SPEC | https://docs.pureskill.gg/datascience/adx/cs2/csds/spec | `csds/spec.md` |
| ENUM | https://docs.pureskill.gg/datascience/adx/cs2/csds/enums | `csds/enums.md` |
| HIW | https://docs.pureskill.gg/datascience/adx/cs2/csds/how-it-fits-together | `csds/how-it-fits-together.md` |
| FLAW | https://docs.pureskill.gg/datascience/adx/cs2/csds/known-flaws | `csds/known-flaws.md` |
| CL | https://docs.pureskill.gg/datascience/adx/cs2/csds/changelog | `csds/changelog.md` |
| CLC | https://docs.pureskill.gg/datascience/adx/cs2/csds/changelog-by-channel | `csds/changelog-by-channel.md` |
| GTD | https://docs.pureskill.gg/datascience/adx/cs2/csds/getting-the-data | `csds/getting-the-data.md` |
| REV | https://docs.pureskill.gg/datascience/adx/cs2/csds/revision-stats | `csds/revision-stats.md` |
| DS | https://docs.pureskill.gg/datascience/adx/cs2/csds/datasheet | `csds/datasheet.md` |
| PRIMER | https://github.com/pureskillgg/datascience-showcase/blob/HEAD/docs/data-primer.md | not a docs.pureskill.gg page; undated |

---

## 1. Round end, winners and reasons (Q1)

**F1. documented.** One row per round; winner and reason are separate columns.
- Quote: "`round_end` has one row per round: who won in `winner_team_code` (2 Terrorists, 3 Counter-Terrorists) and why in `win_reason_code`."
- Source: HIW, `how-it-fits-together.md:41-42`.

**F2. documented.** Column list for `round_end`.
- Quote: "Event that triggers this channel: round_end". Columns are `round`, `tick`, `winner_team_code` (int64, not nullable), `win_reason_code` (int64), `win_reason_message` (string), `player_count` (int64), `second` (calculated).
- Source: SPEC, `spec.md:1078-1090`.

**F3. documented.** Win-reason codes and their winning side.
- Quote: "No other code appears. A drawn round would be code 10, but drawn rounds are removed before publication."
- Source: ENUM, `enums.md:30-43`.
- Mapping: 1 `#SFUI_Notice_Target_Bombed` (T); 7 `#SFUI_Notice_Bomb_Defused` (CT); 8 `#SFUI_Notice_CTs_Win` (CT); 9 `#SFUI_Notice_Terrorists_Win` (T); 11 `#SFUI_Notice_All_Hostages_Rescued` (CT); 12 `#SFUI_Notice_Target_Saved` (CT, "Time ran out with no bomb planted"); 13 `#SFUI_Notice_Hostages_Not_Rescued` (T); 17 `#SFUI_Notice_Terrorists_Surrender` (CT); 18 `#SFUI_Notice_CTs_Surrender` (T).

**F4. documented.** Team codes.
- Quote: "| 2 | Terrorists |" and "| 3 | Counter-Terrorists |". Code 0 is "Unassigned" and code 1 is "Spectator (the game's code; not seen in these matches)".
- Source: ENUM, `enums.md:18-23`.

**F5. documented.** Provenance of the 2026-08-03 round_end values.
- Quote: "`win_reason_code` and `win_reason_message` come from the game, and `player_count` is the number of players on the team at the end of the round, bots included."
- Source: CL, `changelog.md:214` (2026-08-03 entry).

**F6. documented, with a caveat for labels.** Rows built during the rebuild window are different.
- Quote: "CS2 demos stopped carrying round-end events. The channel is rebuilt from `round_state`: `win_reason_code` is 8 (CT win) or 9 (T win), `win_reason_message` is the generic team-win message, `legacy_code` is 0 and `player_count` is 20."
- Source: CL, `changelog.md:390` (2024-02-12 entry).
- Inferred: rows from matches processed between 2024-02-12 and 2026-08-03 have only win codes 8 and 9 and `player_count` of 20. Processing date determines this (see F47 and DS:240-243).
- Unverified: whether `winner_team_code` is correct in those rebuilt rows. The changelog does not say.

**F7. documented.** Round score fix.
- Quote: "`round_state.t_score`, `round_state.ct_score`: On `round_end` rows, the final round is no longer counted twice in 24-round matches (a CS:GO 30-round rule had been applied)."
- Source: CL, `changelog.md:221` (2026-08-03).
- Inferred: any score-based label built from `t_score`/`ct_score` before this fix double-counts the last round of 24-round matches.

**F8. documented.** Round numbering and removed rounds.
- Quote: "Rounds count from 1." and "Warmup, knife rounds and drawn rounds are removed before publication, so round 1 is the first live round and the numbers run on without gaps."
- Source: HIW, `how-it-fits-together.md:38-40`.

**F9. documented.** The converter's cleaning steps.
- Quote: "it removes warmup, knife and drawn rounds, drops duplicate rows, merges each event's player positions in from `player_vector`, computes derived columns, and fixes known errors."
- Source: DS, `datasheet.md:235-237`.

## 2. Overtime, halftime, freeze time and round-live definition (Q1)

**F10. documented.** Regulation length and overtime.
- Quote: "`header.max_rounds` is the regulation length: 24, or 16 in Wingman. Rounds above it are overtime."
- Source: HIW, `how-it-fits-together.md:43-44`.

**F11. documented.** Overtime rules per mode.
- Quote: "At 12–12, Premier and FACEIT play overtime: six more rounds, first to 16. Premier ends a 15–15 overtime as a draw; two matches here did. Competitive and Wingman have no overtime and can end level, 12–12 or 8–8."
- Source: ENUM, `enums.md:197-199`.
- Inferred: "first to 16" reads as a total-score target (12 + 4), with six-round blocks. The overtime win rule is not stated precisely. Draws are possible, so a binary label needs a rule for them.

**F12. documented (undated, third-party primer).** Overtime and regulation as described by the showcase primer.
- Quote: "Regulation is 24 rounds (MR12): pistol rounds are 1 and 13, and overtime is rounds 25 and up, in sets of six. Old CS:GO rules (30 rounds) don't apply."
- Source: PRIMER, section "Sides, teams and rounds".
- Consistent with F10 and F11 for Premier and FACEIT. The primer does not mention that Competitive and Wingman have no overtime.

**F13. documented.** Halftime phase values.
- Quote: "`round_state.phase`: `first` and `second` for the halves, `halftime` between them, and `postgame` after the last round."
- Source: ENUM, `enums.md:139-140`.

**F14. documented.** Side swap at halftime.
- Quote: "**Teams switch sides at the half,** after round `max_rounds / 2`: every player in every regulation match we checked was on the other side in the next round."
- Source: HIW, `how-it-fits-together.md:65-67`.

**F15. documented.** Overtime side swaps exist.
- Quote: "Each starting team's own final score, followed through the half-time and overtime side swaps."
- Source: CL, `changelog.md:111` (2026-10-03 entry).
- Unverified: the exact overtime swap rule. The docs say only that overtime swaps happen.

**F16. documented.** `team_change` records the silent halftime swap.
- Quote: "`team_change`: Players switching teams, including the silent half-time swap."
- Source: CL, `changelog.md:193` (2026-08-03 entry).

**F17. documented, with a gap.** The halftime marker in `round_state`.
- The spec lists `start_halftime` among the round_state triggers (SPEC, `spec.md:1117`). The enums page lists the event types it observed, and it does not include `start_halftime` (ENUM, `enums.md:154`).
- Inferred: use `round_state.phase == "halftime"` (ENUM:139-140) as the halftime marker, and treat `start_halftime` as unverified.

**F18. documented.** Freeze time per mode, and round-time and buy-time values.
- Quote (table row): "| Freeze time | 20 s | 15 s | 10 s | `header.freeze_time` |". The columns are Premier and FACEIT, Competitive, Wingman.
- Source: ENUM, `enums.md:185-199`.
- Other quotes: "Round time runs from the end of freeze time." and "A few rounds have a longer freeze time than the setting." Buy time runs from `round_freeze_end` to `buytime_ended` (20 s in all modes).
- Null handling: "Null when the demo does not record the setting; `bomb_time` has been null in every match checked." (CL, `changelog.md:126`, 2026-09-07 entry, on `header.freeze_time`, `header.max_rounds` and related.)

**F19. documented.** player_info row timing.
- Quote: "One player in one round, taken when the round's freeze time ends."
- Source: HIW, `how-it-fits-together.md:22`. The spec lists `round_freeze_end` as the trigger for `player_info` (SPEC, `spec.md:836` and `908`).

**F20. documented gap: "live" is not defined.**
- The docs use "live" only in the phrase "round 1 is the first live round" (HIW:38-40; DS:38-40). No page defines a live round or its start and end.
- Closest documented events:
  - `round_freeze_end` and `buytime_ended` (ENUM:154; HIW:45-46).
  - `freezetime_ended_inferred`: "New event `freezetime_ended_inferred`: one row per round, at the first tick any player moves." (CL, 2024-07-07 entry, `changelog.md:346`).
  - Quote from the 2026-10-05 fix: "The `freezetime_ended_inferred` row, the first tick a player moves in a round, no longer fires early on a late player's respawn." (CL, `changelog.md:94`.)
- Inferred: a round goes live at freeze end (`round_freeze_end`, or `freezetime_ended_inferred` when present) and ends at its `round_end` tick.

## 3. Player counts (Q1)

**F21. documented.** `round_end.player_count` definition.
- Quote: "`player_count` is the number of players on the team at the end of the round, bots included."
- Source: CL, `changelog.md:214` (2026-08-03). Also CLC `changelog-by-channel.md:297`.
- Caveat: "on the team" conflicts with the local dictionary's sample value of 10. See section 10, item C1.

**F22. documented.** Value of `player_count` in rebuilt rows.
- Quote: "`player_count` is 20." (CL, `changelog.md:390`, 2024-02-12.) See F6.

**F23. documented.** Bots are players in the data.
- Quote: "Bots get their own number, and `player_personal.is_bot` says which they are."
- Source: HIW, `how-it-fits-together.md:51-53`.

**F24. documented.** Dead players have no rows.
- Quote: "A dead player has no rows for the rest of the round."
- Source: HIW, `how-it-fits-together.md:76`.

## 4. The `available` flag (Q1)

**F25. documented.** Which channels carry `available`.
- Quote: "`available`, on six channels whose events a demo may not carry (`bot_takeover`, `bullet_damage`, `player_inputs`, `player_sound`, `rank_update` and `round_mvp`)."
- Source: GTD, `getting-the-data.md:150-152`.

**F26. documented.** Meaning of true and false.
- Quote: "`false` means the demo didn't carry the event, so the channel is empty for that reason; `true` means it did, though the channel can still be empty when nothing happened."
- Source: GTD, `getting-the-data.md:152-154`.
- Also CL, `changelog.md:225`: "False means the demo did not carry that data, so an empty channel means "not recorded" rather than "nothing happened"."

**F27. documented.** Restatement in the known-flaws page.
- Quote: "The index object marks a channel whose events the demo didn't carry with `available: false`."
- Source: FLAW, `known-flaws.md:56-57`.

**F28. inferred.** The flag does not cover round_end.
- `round_end`, `round_start` and `round_state` have no `available` flag (the dictionary shows "available=(not set)"). `round_end` is not in the six-channel list (F25). So an empty round_end cannot be read as "not recorded".

## 5. Team and side identity (Q1)

**F29. documented.** Side per round.
- Quote: "`player_info.team_code` gives each player's side in each round: 2 Terrorists, 3 Counter-Terrorists."
- Source: HIW, `how-it-fits-together.md:63-64`.

**F30. documented.** Starter columns follow the side swaps.
- Quote: "The `header` columns starting `t_starters_` and `ct_starters_` describe the team that started as Terrorists and the team that started as Counter-Terrorists, through the side swaps."
- Source: HIW, `how-it-fits-together.md:68-70`.

**F31. unverified.** Team names or other team IDs. None found in the raw spec pages I searched. Only `team_code` side values appear.

## 6. Final-score and starter columns (Q1, Q3)

**F32. documented (fixed 2026-10-03).** Final score definition.
- Quote: "Each starting team's own final score, followed through the half-time and overtime side swaps. Before, it was the larger of the side scores, which could give a team the other team's score."
- Source: CL, `changelog.md:111` (2026-10-03).

**F33. documented (fixed 2026-10-03).** Average rank and wins denominators.
- Quote: "Averaged over the first half's players only, rounds up to `max_rounds / 2` (12, or 8 in Wingman)."
- Source: CL, `changelog.md:110`.

**F34. documented (contradicts the primer).** Primer warning, undated.
- Quote: "Don't trust the header's "starters" columns (`*_starters_score_final`, `*_starters_avg_rank`, `*_starters_avg_wins`). They're computed on the old 30-round boundary and misattribute scores ... Count rounds and wins from `round_end`."
- Source: PRIMER.
- Inferred: this describes the pre-2026-10-03 behavior. The current docs say the columns are fixed. The local dictionary's evidence supports the fixed version (see C4).

**F35. documented.** `score_update` and `team_change` are separate channels.
- Quote: "`score_update`: Each change in a team's score."
- Source: CL, `changelog.md:192` (2026-08-03).
- Primer claim, unverified: "`score_update` double-counts at halftime ... Take round winners from `round_end.winner_team_code`."

## 7. Round-state, round_start and player_spawn caveats (Q1, Q3)

**F36. documented.** Duplicate round_start rows.
- Quote: "Each round now has two `round_start` rows: the game's own, and the one the converter adds after `round_poststart` (since 2024-08-05)."
- Source: CLC, `changelog-by-channel.md:318` (2026-08-03). Also CL, `changelog.md:216`.
- The dictionary's observed counts agree: `round_start` 47 rows over 24 rounds (dictionary:2374).

**F37. documented.** round_start channel rebuild.
- Quote: "The rebuilt channel used since 2024-02-12 was one row short."
- Source: CL, `changelog.md:215` (2026-08-03).

**F38. documented (fixed 2026-10-05).** player_spawn round.
- Quote: "`player_spawn.round`: The round the player spawned into. Before, many rows carried the round before it (7,913 in our 50 test matches)."
- Source: CL, `changelog.md:97` (2026-10-05).
- Conflicts with the primer's "player_spawn.round is one behind" (PRIMER). See C5.

**F39. documented (fixed 2026-10-05).** Velocities.
- Quote: "Computed per player per round, so a player's first sample in each round is 0."
- Source: CL, `changelog.md:80` (2026-10-05).
- The primer's respawn velocity-spike advice predates this change.

**F40. documented (fixed 2026-10-06).** `player_vector` derived columns.
- Quote: "`player_vector` stores fewer columns. Since 2026-10-06 it no longer stores its ten derived columns; compute them on load, as the spec describes."
- Source: FLAW, `known-flaws.md:82-84`.

## 8. Revisions, duplicates and missing channels (Q3)

**F41. documented.** Revision window.
- Quote: "A revision opens at 00:00 UTC and collects every match processed that UTC day."
- Source: GTD, `getting-the-data.md:26-27`.

**F42. documented.** Processed date, not played date.
- Quote: "A match sits in the revision for the day it was processed, which can differ from the day it was played."
- Source: GTD, `getting-the-data.md:28-30`.

**F43. documented.** Outage gaps and large revisions.
- Quote: "After a pipeline outage, the backlog was published in a single revision, so a few revisions are much larger than the rest, and the days of the outage have no revision."
- Source: REV, `revision-stats.md:15-18`.

**F44. documented.** A change can split a day across revisions.
- Quote: "A change reaches production partway through a day, so that revision holds matches from both before and after it."
- Source: REV, `revision-stats.md:21-22`.

**F45. documented.** Pruning window.
- Quote: "Revisions more than about a year old are revoked and emptied in a batch about once a year, so the window runs from one to two years."
- Source: GTD, `getting-the-data.md:33-35`.

**F46. documented.** Retention and reprocessing.
- Quote: "Demo files are kept for 30 days and parsed replays for 7, then deleted. After that a match can't be processed again, so an improvement to the pipeline reaches newly processed matches only."
- Source: DS, `datasheet.md:240-243`.
- Inferred: older matches are never backfilled, so one tome can mix pipeline versions.

**F47. documented.** Object layout.
- Quote: "A loose match is 43 objects since 2026-08-04 and 31 before; the 2026-08-03 revision holds both, and so does the revision in which zips start. Read the index rather than assuming a channel list."
- Source: GTD, `getting-the-data.md:120-123`.
- Also: "Revisions through 2026-08-02 carry 30 channels rather than 42, and the 2026-08-03 revision holds a mix." (FLAW, `known-flaws.md:74-75`.)

**F48. documented.** Channel-set changes.
- Quote: "Fourteen channels were added: `bullet_damage`, `grenade_bounce`, `grenade_vector`, `item_dropped`, `item_refund`, `molotov_fire`, `player_chat`, `player_connect`, `player_inputs`, `player_sound`, `rank_update`, `score_update`, `team_change` and `world_item_vector`. Two were removed: `item_remove` and `player_action`."
- Source: FLAW, `known-flaws.md:76-80`.

**F49. documented.** Duplicate matches.
- Quote: "A match's `id` is new each time a demo is processed. A demo processed twice, for example one uploaded by two users, appears twice with different ids. Find duplicates by comparing `header` columns such as `map_name`, `server_name`, `build_num` and the final scores."
- Source: GTD, "Match identity and dates" section, `getting-the-data.md:158-164`.
- Also: "There may be duplicate matches." (DS:91.) The converter "drops duplicate rows" (DS:235-236).
- Also: "A match that would still have duplicate `tick` and `player_id_fixed` rows fails processing and is not published." (CL, `changelog.md:262`, 2026-06-07.)

**F50. documented.** Hand-uploaded matches.
- Quote: "A match a user uploads by hand has `platform` `unknown`, and its `match_date` is when it was processed, not when it was played."
- Source: FLAW, `known-flaws.md:15-16`.
- Quote: "Since 2026-10-08 that includes many FACEIT matches. Tell them apart by `header.server_name`, which begins `FACEIT.com` for a match played on a FACEIT server."
- Source: FLAW, `known-flaws.md:18-20`.

**F51. documented.** Empty channels.
- Quote: "Servers that restrict what their recording broadcasts, as tournament servers do, leave `player_blind`, `player_footstep`, `item_equip` and `weapon_action` empty; the rest of the match is complete."
- Source: FLAW, `known-flaws.md:52-54`.
- Also: "`player_sound` is empty in most matches, because most demos don't carry that event." (FLAW:55-56.)
- Also: "`header.is_gotv_recording`: ... Such matches lack the per-player channels GOTV restricts, such as `player_footstep` and `item_equip`." (CL, `changelog.md:147`, 2026-08-22.)

**F52. documented.** player_inputs timing.
- Quote: "`player_inputs` is empty in every match recorded after the CS2 update of 2026-07-09 and parsed before parser 5.7.0 (2026-10-11), whatever the demo's source."
- Source: FLAW, `known-flaws.md:58-61`.
- Later fix: "Has rows again, on every kind of demo" (CL, 2026-10-11 entry).

**F53. documented.** Missing events and ranks.
- Quote: "Within a match, events can be missing. This is rare, but can upset some calculations; skipping the match is usually enough."
- Source: FLAW, `known-flaws.md:69-70`.
- Also: "Older matches from FACEIT are missing player ranks." (FLAW:68.)

**F54. documented.** player_status can be empty.
- Quote: "`player_status` can be empty in a match; one of the 89 matches of 2026-10-07 had no rows in it."
- Source: HIW, `how-it-fits-together.md:83-85`.

**F55. documented.** Composition changes over time.
- Quote: "The data changes over time: channels and columns are added and removed, types change, and values are fixed, as the changelog lists."
- Source: DS, `datasheet.md:265-267`.

**F56. unverified.** Whether one match `id` can appear in more than one revision. Only same-day processing is documented (F41).

## 9. Label and target definitions in the docs (Q1, Q2 context)

**F57. documented.** No label is defined.
- Quote: "No. There are many possible machine learning projects with the data, and a target is whatever the user defines for theirs. We note that player rank is available and often useful, including for unsupervised tasks."
- Source: DS, `datasheet.md:60-63`.

**F58. documented.** Win-probability modeling is listed as an intended use.
- Quote: "building win probability models"
- Source: DS, `datasheet.md:18-20`.

**F59. documented.** Prior win-probability work on earlier versions of these data.
- Quote: "An older, unavailable version of these data was used in _Analyzing the Differences between Professional and Amateur Esports through Win Probability_ by authors Peter Xenopoulos, William Robert Freeman, and Claudio Silva."
- Source: DS, `datasheet.md:254-255`.
- Not fetched. The citation is copied from the datasheet only.

## 10. GitHub repos (Q2) and contradictions with notes\schema_dictionary.md

**Q2 result: no round win-probability or match-outcome model, feature list, or label definition found.**

- **F60. documented (absence).** Org-wide code search for `win_probability`, `winprob`, `outcome`, `feature_names` and `logistic` returned no code hits. The "win probability" hits are only docs pages (`docs` repo, `datasheet.md` and old CS:GO/CS2 READMEs).
- **F61. documented.** The `xgboost` hits in `pureskillgg/dsdk` are a generic model-serving wrapper, not a CS model. Path: `pureskillgg_dsdk/ds_models/s3_xgboost.py`. The `predict` hits are in the same generic `ds_models` wrappers (`s3_scikit.py`, `s3_scikit_set.py`, `s3_xgboost.py`) and in old CS:GO/CS2 READMEs. None define features or labels for a CS model.
- **F62. documented.** `pureskillgg/makenew-pyskill` `notebooks/usual_suspects.py` is a 16-line stub with imports and setup only.
- **F63. documented.** `pureskillgg/csgo-dsdk` has no model. Its overtime helper defaults to CS:GO rules. Quote: "max_rounds_csgo (int): Default value: 30 ... CS2 regulation is MR12, 24 rounds, so CS2 callers should pass 24." Path: `pureskillgg_csgo_dsdk/overtime/pop_overtime.py` (overtime = `round > max_rounds_csgo`).
- **F64. documented.** `pureskillgg/csgo-datascience` does not exist. The GitHub API returned 404, and it is not in the org repo list.
- Caveat: GitHub code search is index-based and may miss files. I checked the top-level tree of each repo and the matching code hits, not a full clone.

**Contradictions and gaps against `notes\schema_dictionary.md`**

- **C1. Unresolved.** `round_end.player_count`. The docs say "the number of players on the team at the end of the round, bots included" (CL:214). The dictionary gives no definition, and its sample value for match `m_a661aadb` is 10 (dictionary:1864, 1876). That match has `unique_steamids` = 10 (dictionary:7). In a 5v5 that is the total for both teams, not one team. Either the docs wording is loose or the dictionary value is the total. Check by counting `player_info` rows per round for that match.
- **C2. Not a contradiction, but incomplete.** The dictionary's `round_state` declared events include `start_halftime` (dictionary:1938), but the observed event counts do not (dictionary:2934). The observed halftime marker is `phase == "halftime"` (dictionary:2366, 3 rows). This matches F13 and F17.
- **C3. Not a contradiction, but incomplete.** The dictionary says `t_score_raw` equals `t_score` "in all rows of this match" (dictionary:1947-1948). That holds for one match only. Given F7, scores can differ across revisions processed before and after 2026-08-03, so the dictionary should not generalize this.
- **C4. Consistent with the docs; the primer is the outlier.** The dictionary reports that the header final scores (13 and 11) "reconcile with identity-based totals, not side totals" (dictionary:8). That matches the fixed definition in F32. The primer's warning (F34) describes the old behavior.
- **C5. Consistent with the docs; the primer is the outlier.** The primer says `player_spawn.round` is "one behind" (PRIMER). The 2026-10-05 fix corrected it (F38). The dictionary does not check this field.
- **C6. Gap.** The dictionary does not mention the rebuilt rows of 2024-02-12 to 2026-08-03 (F6). Any tome that spans that window will have round_end win codes of only 8 and 9, and `player_count` of 20. The sampled match `m_a661aadb` (2026/10/01, dictionary:7) is after the 2026-08-03 change, so its codes are real.
- **C7. Time-dependent.** `player_inputs` has 0 rows and `available=False` in the sample (dictionary:1436). That sample is from 2026/10/01, before the 2026-10-11 fix (F52), so this is expected, not a contradiction.
- **C8. Consistent.** Win-code and winner counts agree with the enum mapping (F3, F4). Sample: winner 2 (T) = 11 rounds; codes 1 (2 rows) and 9 (9 rows) sum to 11. Winner 3 (CT) = 13 rounds; codes 7 (3 rows) and 8 (10 rows) sum to 13 (dictionary:2224-2238).

## 11. Open questions (unverified)

- Whether `winner_team_code` is correct in the rebuilt round_end rows (F6).
- The exact overtime win rule and the overtime side-swap rule (F11, F15).
- Whether `player_count` is per team or total (C1).
- Whether a live round ends at `round_end` or at another event. The docs do not define "live" (F20).
- Whether one match `id` can appear in more than one revision (F56).
- Whether `team_change` records overtime swaps (F16 covers the halftime swap only).
- `header.max_rounds` and `header.freeze_time` can be null (F18). Labels that depend on overtime cutoffs need a fallback.

## 12. Summary of the five facts most relevant to labels and features

1. **Round label.** `round_end.winner_team_code` (2 = T, 3 = CT), one row per round (F1, F4). Caveat: the 2024-02-12 to 2026-08-03 rebuild has only win codes 8 and 9 and `player_count` of 20 (F6).
2. **Match outcome and draws.** Regulation is `header.max_rounds` (24, or 16 in Wingman). Premier and FACEIT overtime starts at 12-12 in six-round blocks. Premier can end 15-15 as a draw. Competitive and Wingman can end level with no overtime (F10, F11).
3. **Halftime and side swaps.** Sides swap after round `max_rounds / 2`, with overtime swaps also described (F14, F15). Use `player_info.team_code` per round, and `round_state.phase == "halftime"` as the marker (F13, F17).
4. **Final scores.** `header.t_starters_score_final` and `ct_starters_score_final` were fixed on 2026-10-03 and follow side swaps (F32). The primer says not to trust them (F34). Counting `round_end` winners is the safest label source. The t_score double-count was fixed on 2026-08-03 (F7).
5. **Round start and freeze.** Freeze time is 20 s (Premier and FACEIT), 15 s (Competitive) or 10 s (Wingman) (F18). Round 1 is the first live round (F8), but "live" is not defined in the docs (F20). `round_start` has two rows per round since 2024-08-05 (F36).

No win-probability or outcome model, feature set, or label definition exists in the pureskillgg repos checked (F60-F64). The datasheet says the data has no label or target (F57).
