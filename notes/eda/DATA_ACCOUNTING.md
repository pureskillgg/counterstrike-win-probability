# Data accounting: counts quoted for the 2026-10 data

Checked 2026-10-13. Scope: logs\bakeoff.log; the dedupe rule in scripts\model\bakeoff.py; subfolder counts for E:\csds_final\csds\2026\10\01 to 11 (directories only); the S3 object counts for 10-09 and 10-10 given in the brief; the revision counts quoted in notes\research\data_regimes_timeline.md. Nothing was written under E:. No file on E: was opened.

Tags: **VERIFIED** = I ran code, counted, or opened the file. **UNVERIFIED** = copied from another note, derived, or not checked.

## Status

- **Not resolved.** 2026-10-09 has 2,301 folders on E:, against 1,162 matches in the revision list. 2026-10-10 has 2,758, against 1,162. 2026-10-11 has 72 folders on E:, against 1,162. Nothing here explains these gaps. (VERIFIED counts; causes UNVERIFIED.)
- **Reconciled for 10-01 to 10-08.** E: folders for 01 to 07 sum to 617, and 08 has 1,135. Together they give 1,752, the bakeoff header count. (VERIFIED.)
- **Duplicates.** The code removes 51 matches (46 groups, lowest key kept). The log's "47 removed" counts only removed copies that still had rounds. (VERIFIED, see section 2.)

## 1. Reconciliation table

| # | Count | Value | Unit | Source | Tag |
|---|---|---|---|---|---|
| 1 | Revisions in 2026-10 (01 to 11) | 11 | revisions | timeline note (cites REV) | UNVERIFIED (not re-fetched) |
| 2 | Matches 10-01 to 10-07 | 617 | matches | 1,752 minus 1,135 | UNVERIFIED as a revision count. Equals the E: sum, VERIFIED (row 6) |
| 3 | Matches 10-08 | 1,135 | matches | timeline note | UNVERIFIED as a revision count. E: folders = 1,135, VERIFIED (row 7) |
| 4 | Matches each of 10-09, 10-10, 10-11 | 1,162 | matches | timeline note | UNVERIFIED |
| 5 | Matches 10-01 to 10-11 | 5,238 | matches | timeline note | UNVERIFIED. Identity 617 + 1,135 + 3 x 1,162 = 5,238, VERIFIED |
| 6 | E: folders 01 to 07 | 617 (67, 65, 100, 109, 83, 104, 89) | folders | counted | VERIFIED |
| 7 | E: folders 08 | 1,135 | folders | counted | VERIFIED |
| 8 | E: folders 09 | 2,301 | folders | counted. HANDOFF.md line 31 gives the same figure | VERIFIED. Differs from 1,162 by +1,139. NOT RECONCILED |
| 9 | E: folders 10 | 2,758 | folders | counted. HANDOFF.md line 31 gives the same figure | VERIFIED. Differs from 1,162 by +1,596. NOT RECONCILED |
| 10 | E: folders 11 | 72 | folders | counted | VERIFIED. Differs from 1,162 by -1,090. NOT RECONCILED. Cause not checked |
| 11 | S3 objects, 10-09 | 98,943 | objects | the brief | Not found in notes\ (text and JSON) or other project text files, VERIFIED. 98,943 = 43 x 2,301, VERIFIED. S3 name and size check NOT RUN |
| 12 | S3 objects, 10-10 | 118,594 | objects | the brief | Same as row 11. 118,594 = 43 x 2,758, VERIFIED. S3 check NOT RUN |
| 13 | Objects per match after 2026-08-03 | 43 | objects | timeline note (cites GTD) | UNVERIFIED here |
| 14 | Tome header matches, 10-01 to 10-08 | 1,752 | matches | bakeoff.log line 8 | VERIFIED |
| 15 | round_end rows (unique match, round) | 35,992 | rounds | bakeoff.log line 9 | VERIFIED |
| 16 | Dropped for a missing channel | 43 matches (490 rounds) | matches | bakeoff.log line 21 | VERIFIED |
| 17 | Kept at channel stage | 1,709 | matches | bakeoff.log line 26 | VERIFIED. 1,752 - 43 = 1,709 |
| 18 | Duplicate groups, channel-kept | 46 groups, 97 members | groups | bakeoff.log line 30 | VERIFIED |
| 19 | Copies removed by the code | 51 | matches | bakeoff.py lines 643 to 645 (97 - 46) | VERIFIED. Not printed by the code |
| 20 | Removed copies that still had rounds | 47 (913 rounds) | matches | bakeoff.log line 30; bakeoff.py lines 656 to 657 | VERIFIED |
| 21 | Removed copies with no kept round | 4 | matches | 51 - 47 | VERIFIED (arithmetic) |
| 22 | Kept matches with no kept round | 141 | matches | bakeoff.log line 31 | VERIFIED. Includes the 4 in row 21 |
| 23 | Matches with rounds before dedupe | 1,568 | matches | 1,709 - 141. bakeoff_run1_no_dedupe.log: "in 1568 matches" | VERIFIED |
| 24 | Matches with rounds after dedupe | 1,521 | matches | 1,568 - 47. bakeoff.log line 38 | VERIFIED |
| 25 | Matches after dedupe, all | 1,658 | matches | 1,709 - 51 = 1,521 with rounds + 137 without | VERIFIED |
| 26 | Rounds in kept matches | 35,502 | rounds | 35,992 - 490. bakeoff.log line 33 | VERIFIED |
| 27 | Dropped: player_count != 10 | 3,663 | rounds | bakeoff.log line 34 | VERIFIED |
| 28 | Dropped: disconnect or takeover | 1,767 | rounds | bakeoff.log line 35 | VERIFIED |
| 29 | Dropped: not 5 vs 5 at start | 476 | rounds | bakeoff.log line 37 | VERIFIED |
| 30 | Dropped: duplicate copy rounds | 913 | rounds | bakeoff.log line 36 | VERIFIED |
| 31 | Rounds kept | 28,683 in 1,521 matches | rounds | bakeoff.log line 38. 35,502 - 3,663 - 1,767 - 476 - 913 = 28,683 | VERIFIED |
| 32 | Run 1, no dedupe | 29,596 rounds in 1,568 matches | rounds | bakeoff_run1_no_dedupe.log line 34. 29,596 - 913 = 28,683 | VERIFIED |
| 33 | Primary split, train / val / test | 1,095 / 122 / 304 matches; 20,470 / 2,364 / 5,849 rounds; 167,379 / 19,170 / 47,716 states | | bakeoff.log lines 63 to 65 | VERIFIED. Sums: 1,521 matches, 28,683 rounds, 234,265 states |
| 34 | Chronological split, train / val / test | 1,095 / 122 / 304 matches; 20,497 / 2,466 / 5,720 rounds; 167,884 / 20,017 / 46,364 states | | bakeoff.log lines 66 to 68 | VERIFIED. Same sums |
| 35 | Round-fingerprint diagnostic (not a removal) | 47 groups, 99 members, matches with at least 5 rounds | groups | bakeoff.log line 58; bakeoff.py lines 714 to 725 | VERIFIED as printed. Not used for removal |

## 2. The three duplicate counts

| Number | What it counts | Where it is quoted | Does the current code produce it? |
|---|---|---|---|
| **46 groups** (97 matches) | Channel-kept matches with the same sorted round_end, player_death and player_info events | bakeoff.log line 30; bakeoff.py lines 639 to 642; notes\model\MODEL_CARD.md line 49 | Yes. VERIFIED |
| **51 removed** | Every group member except the lowest key (97 - 46) | bakeoff.py lines 643 to 645 (not printed); MODEL_CARD.md line 49 ("removes 51 matches") | Yes. This is the number the code removes. VERIFIED by reading the code and by 97 - 46 |
| **47 removed** | The subset of the 51 that still had rounds after the round filters | bakeoff.log line 30 ("removed (with kept rounds): 47"); bakeoff.py lines 656 to 657 | Printed by the code, but it is a subset of the 51. 4 of the 51 had no rounds |
| **47 groups** (99 matches) | Round-fingerprint diagnostic over all matches with at least 5 rounds. Different signature, no channel filter | bakeoff.log line 58; bakeoff.py lines 714 to 725; notes\model\bakeoff_results.md line 245 | No. It is a check, not the dedupe |

**Answer.** The current code produces 46 groups and removes 51 matches, keeping the lowest key in each group. The log's 47 counts the removed copies that still had rounds in the table. The 47-group figure comes from the separate fingerprint check.

**Log and code timing.** bakeoff.py was last modified 2026-10-12 19:00:30. bakeoff.log and bakeoff_rerun_reference.log were written 2026-10-12 20:44:53. So the log is newer than the code. Its numbers agree with the code's logic. (VERIFIED timestamps; logic VERIFIED by reading. A fresh run was not done.)

**Risk left open.** The code keeps the lowest key even if that copy has no rounds left. If a group's lowest key had no rounds while a removed copy did, the group's data is lost. This was not checked (NOT RUN).

## 3. Day-level discrepancy (09 and 10)

- The S3 figures in the brief equal 43 objects times the E: folder counts exactly (2,301 and 2,758). So S3 and E: hold the same number of match folders. (VERIFIED arithmetic.)
- Both figures are about 2.0 times the revision count of 1,162 per day, and the gap does not match any day's count. The gap is +1,139 on 09 and +1,596 on 10. Possible causes, not tested: duplicate copies (the dedupe rule shows copies exist), re-processed folders, or partial sync. (UNVERIFIED.)
- HANDOFF.md line 31 says the 09 and 10 folders "may contain duplicates or be incomplete." This note does not settle that.

## 4. Checks

**Passed (VERIFIED, pixi Python script).** 26 of 29 checks passed. The other 3 are the "not reconciled" difference rows, which expect zero and show the gap.

- Match funnel: 1,752 - 43 = 1,709; 1,709 - 141 = 1,568; 1,568 - 47 = 1,521; 1,709 - 51 = 1,658; 51 - 47 = 4.
- Round funnel: 35,992 - 490 = 35,502; 35,502 - 3,663 - 1,767 - 476 - 913 = 28,683; 29,596 - 913 = 28,683.
- Splits: matches sum to 1,521, rounds to 28,683 and states to 234,265, for both the primary and chronological splits.
- Revision identity: 617 + 1,135 + 3 x 1,162 = 5,238. Also 617 + 1,135 = 1,752.
- E: folders 01 to 07 sum to 617; 01 to 08 sum to 1,752; folder 08 = 1,135.
- S3 arithmetic: 98,943 = 43 x 2,301; 118,594 = 43 x 2,758.
- Fingerprint arithmetic: 99 - 47 = 52.

**Not reconciled (VERIFIED).** 09 (+1,139), 10 (+1,596), 11 (-1,090).

**Not run or not checked.**
1. The S3 name and size comparison against the listing. No S3 access was used, and the figures are not in notes\eda.
2. Whether the E: folders for 09, 10 and 11 are duplicates or incomplete. The brief asked for folder counts only.
3. Per-day counts for 10-01 to 10-07. Only the 617 total is quoted.
4. Re-running bakeoff.py or scripts\eda\dedupe_replicate.py. The log was taken as written.
5. Whether any group's lowest key had no rounds while a removed copy had rounds (see section 2).
6. Whether fingerprint and signature groups have identical members, as batch_effect_2026-10-08.md line 120 claims.
7. The live revision-stats page. Its figures are quoted from the timeline note and are UNVERIFIED here.
8. Match-key overlap between the tome for 10-01 to 10-08 and E: 01 to 08. Only counts were compared.

## 5. Notes that disagree with the code or the log

- **notes\eda\batch_effect_2026-10-08.md line 126** says 47 is "the all-match count before the channel filter" and 51 is "the match-level removal after it." The code says 47 is the removed copies with rounds, a subset of the 51. The all-match 47 is the fingerprint check (row 35). VERIFIED.
- **batch_effect_2026-10-08.md line 94** gives 52 dropped copies (37 on 10-08, 15 on other days) and 47 kept. That is the 47-group, 99-member measure (99 - 47 = 52), not the code's 51. The attribution is my inference. (UNVERIFIED.)
- **batch_effect_2026-10-08.md line 124** says the log was written at 6:58:40 PM, before the code's 7:00:30 PM edit. The current timestamps are reversed (section 2). The note is stale. VERIFIED.
- **notes\eda\players.md lines 7 and 861, and players.json line 5** quote 47 groups (99 matches) for player_info and 46 groups (97 matches) for team_change. Those are different signatures from the bakeoff rule. Not checked here. (UNVERIFIED.)

## 6. Method

- Subfolder counts: a .NET directory call on each day folder under E:\csds_final\csds\2026\10. Directories only. An earlier listing printed some folder names under 01 to 04. No file contents were read.
- Logs and code were read with grep and read. The checks were run with `pixi run python -I -X utf8` from G:\pureskill\csds-cs2. The script lives in the session scratchpad on C:. Nothing was installed.
