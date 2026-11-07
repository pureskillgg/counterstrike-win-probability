# Swarm E2: map profile (CS2 CSDS tomes)

Written 2026-10-13. Two tomes on G: only: 2026-10-01_to_08 and 2024-02-g. No E: reads. Nothing under models\ or scripts\model\ was edited. All numbers below were produced by code run in this session unless marked UNVERIFIED.

## Sources and scope
- 2026: `tomes\2026-10-01_to_08\tome\csds` (header + `basic_*.2026-09-18,2026-10-08`). Header rows 1,752. After dedupe 1,700 matches; 1,544 contain clean 5v5 rounds.
- 2024: `tomes\2024-02-g\tome\csds` (header + `basic_*.2023-12-28,2024-02-25`). Header rows 4,435. After dedupe 4,138 matches; 3,924 contain clean 5v5 rounds.
- Map key: `map_name` from the header tome. Eight 2026 maps have 18 or more matches; cs_office and de_boulder are small (see below).

## Definitions (from `scripts\swarm\e2_map_profile.py`)
- **Dedupe:** matches with an identical event signature (round_end, player_death, player_info) are collapsed to the lowest match_key. 2026 removed 52 matches; 2024 removed 296.
- **Clean round:** round_end.player_count = 10 (2026) or 20 (2024). Wingman rounds are excluded (147 wingman matches in 2026 after dedupe, 0 clean rounds from them).
- **CT win:** round_end.winner_team_code = 3. Side codes are 2 = T, 3 = CT (`notes\schema_dictionary.md`, lines 132 and 185).
- **Average round length:** (round_end tick − first round_freeze_end tick) / tick_rate. tick_rate = 64 for every round in both eras. This is the live phase, not the full round including freeze time.
- **Plant rate:** rounds with a bomb_planted row inside the live window / live rounds in matches that have bomb_state rows. Denominators exclude 12 matches (152 rounds) in 2026 and 163 matches (2,447 rounds) in 2024 with no bomb rows.
- **Plant-to-end:** (round_end tick − plant tick) / tick_rate, in seconds, for planted rounds.
- **Kills per round:** player_death inside the live window, attacker side known (2 or 3), attacker not the victim, and not a team kill. Split into CT and T by attacker side.
- **CI:** Wilson 95% on rounds. Sensitivity: match-cluster bootstrap (2,000 resamples, seed 0), because rounds in one match share teams.

## Results: 2026 tome (1,544 matches, 31,065 clean rounds). VERIFIED (run log `scripts\swarm\out\e2_map_profile_run_1545.log`)

| Map | Matches | Rounds | CT win | 95% CI Wilson | 95% CI match cluster | Avg live round (s) | Plant rate | Plant-to-end mean / median (s) | Kills per round, all (CT / T) |
|---|---|---|---|---|---|---|---|---|---|
| de_mirage | 337 | 6745 | 53.7% | 52.5 to 54.9 | 52.3 to 55.1 | 67.0 | 45.6% | 23.1 / 22.8 | 7.15 (3.77 / 3.37) |
| de_dust2 | 307 | 6061 | 50.9% | 49.7 to 52.2 | 49.2 to 52.7 | 66.4 | 46.9% | 22.5 / 21.9 | 7.13 (3.68 / 3.45) |
| de_ancient | 210 | 4337 | 53.3% | 51.8 to 54.7 | 51.4 to 55.3 | 64.0 | 53.4% | 23.9 / 23.7 | 7.18 (3.79 / 3.39) |
| de_anubis | 195 | 3995 | 51.4% | 49.8 to 52.9 | 49.4 to 53.3 | 69.4 | 52.8% | 23.4 / 22.6 | 7.16 (3.70 / 3.46) |
| de_inferno | 188 | 3837 | 50.6% | 49.0 to 52.2 | 48.5 to 52.8 | 74.4 | 52.5% | 24.9 / 24.7 | 7.05 (3.67 / 3.38) |
| de_cache | 137 | 2696 | 51.6% | 49.7 to 53.5 | 49.2 to 54.1 | 68.2 | 50.1% | 22.7 / 22.1 | 7.14 (3.70 / 3.44) |
| de_nuke | 131 | 2670 | 51.8% | 49.9 to 53.7 | 49.5 to 54.1 | 68.3 | 47.9% | 23.4 / 22.8 | 7.18 (3.74 / 3.44) |
| de_vertigo | 18 | 359 | 53.8% | 48.6 to 58.9 | 47.7 to 59.4 | 64.9 | 43.7% | 22.1 / 21.9 | 7.26 (3.81 / 3.45) |
| de_train | 16 | 265 | 54.3% | 48.3 to 60.2 | 47.8 to 60.1 | 63.3 | 44.9% | 22.8 / 23.4 | 7.19 (3.74 / 3.46) |
| cs_office | 4 | 78 | 43.6% | 33.1 to 54.6 | 40.5 to 46.3 | 64.5 | n/a | n/a | 7.32 (3.60 / 3.72). **n<100** |
| de_boulder | 1 | 22 | 40.9% | 23.3 to 61.3 | not estimable (one match) | 71.8 | 54.5% | 21.6 / 21.3 | 7.41 (3.73 / 3.68). **n<100** |

All maps, 2026: CT win 52.04% (Wilson 51.49 to 52.60); plant rate 49.29%; average live round 67.84 s; kills per round 7.14.

## Results: 2024 tome (3,924 matches, 72,782 clean rounds). VERIFIED (same run)

| Map | Matches | Rounds | CT win | 95% CI Wilson | 95% CI match cluster | Avg live round (s) | Plant rate | Plant-to-end mean / median (s) | Kills per round, all (CT / T) |
|---|---|---|---|---|---|---|---|---|---|
| de_mirage | 930 | 17190 | 52.1% | 51.4 to 52.9 | 51.2 to 53.0 | 73.0 | 47.3% | 25.9 / 25.9 | 6.75 (3.49 / 3.27) |
| de_inferno | 641 | 11685 | 49.8% | 48.9 to 50.7 | 48.6 to 50.9 | 74.2 | 47.3% | 26.6 / 27.0 | 6.36 (3.23 / 3.13) |
| de_vertigo | 605 | 11007 | 51.7% | 50.7 to 52.6 | 50.6 to 52.8 | 64.6 | 44.8% | 23.4 / 23.0 | 6.22 (3.16 / 3.07) |
| de_nuke | 487 | 9024 | 51.7% | 50.7 to 52.7 | 50.4 to 53.1 | 70.0 | 47.7% | 25.9 / 26.1 | 6.72 (3.45 / 3.27) |
| de_ancient | 325 | 6323 | 51.1% | 49.8 to 52.3 | 49.5 to 52.6 | 68.6 | 55.5% | 25.9 / 25.8 | 6.87 (3.50 / 3.37) |
| de_overpass | 330 | 6183 | 50.7% | 49.4 to 51.9 | 49.1 to 52.1 | 74.5 | 49.4% | 25.1 / 25.4 | 6.54 (3.32 / 3.23) |
| de_anubis | 270 | 5331 | 48.1% | 46.8 to 49.5 | 46.3 to 49.9 | 74.0 | 56.6% | 26.3 / 26.2 | 6.78 (3.39 / 3.39) |
| de_dust2 | 246 | 4356 | 49.4% | 47.9 to 50.9 | 47.5 to 51.5 | 69.7 | 49.0% | 25.4 / 25.7 | 6.87 (3.46 / 3.41) |
| cs_office | 90 | 1683 | 42.0% | 39.7 to 44.4 | 39.0 to 44.9 | 72.3 | n/a | n/a | 7.00 (3.28 / 3.72) |

All maps, 2024: CT win 50.72% (Wilson 50.35 to 51.08); plant rate 48.69%; average live round 71.15 s; kills per round 6.61. No 2024 map is below 100 rounds.

**Caution on 2024 CT rates:** round 13 has a side-code defect (see below). Excluding round 13 changes the all-map CT rate to 50.64%, and no map moves by more than 0.53 points (VERIFIED, `scripts\swarm\out\e2_r13_sensitivity.log`):

| Map (2024) | CT all rounds | CT excl. round 13 | Round-13 rounds | CT in round 13 |
|---|---|---|---|---|
| de_mirage | 52.11% | 52.24% | 852 | 49.5% |
| de_inferno | 49.78% | 49.56% | 572 | 54.0% |
| de_vertigo | 51.67% | 51.65% | 527 | 52.0% |
| de_nuke | 51.70% | 51.66% | 446 | 52.5% |
| de_ancient | 51.05% | 50.94% | 308 | 53.2% |
| de_overpass | 50.67% | 50.50% | 298 | 54.0% |
| de_anubis | 48.13% | 47.87% | 261 | 53.3% |
| de_dust2 | 49.40% | 49.27% | 222 | 51.8% |
| cs_office | 42.01% | 41.47% | 82 | 52.4% |

## Maps with fewer than 100 rounds
- **2026 cs_office:** 4 matches, 78 rounds. CT 43.6%, Wilson 33.1 to 54.6. Not reliable at map level.
- **2026 de_boulder:** 1 match, 22 rounds. CT 40.9%. The match-cluster interval collapses because there is one match. Treat as anecdote.
- Not below 100 rounds but few matches: 2026 de_vertigo (18 matches, 359 rounds) and de_train (16 matches, 265 rounds). Their match-cluster intervals are about 12 points wide.

## Checks
- **Independent recount, no dedupe, no kill logic** (`scripts\swarm\e2_crosscheck.py`, log `scripts\swarm\out\e2_crosscheck_1545.log`): per-map CT rates agree with the main run within 0.4 points for every map in both eras. Example: 2026 de_mirage 53.7% both ways; 2024 de_mirage 52.0% raw vs 52.1% deduped. Raw rounds are higher than deduped (2026: 32,041 vs 31,065), as expected from removing replicated matches. VERIFIED.
- **Side codes against bomb outcomes** (`scripts\swarm\e2_side_check.py`, log `scripts\swarm\out\e2_side_check_1600.log`): in 2026, all 7,427 rounds with a defuse or explode row agree with the round_end winner (exploded -> 2, defused -> 3). VERIFIED.
- The same check on 2024 finds 499 disagreements out of 15,436 clean rounds with a bomb outcome. Round 13 accounts for 434 of them (details below). VERIFIED.

## Open issue: 2024 round 13 (side code disagrees with bomb outcome)
- **Where:** round 13 in 2024 matches dated 12 to 25 Feb (day taken from match_key `csds/2024/02/DD/`). Round 13 is the first round of the second half under MR12 rules, which is an assumption (UNVERIFIED). For Feb 1 to 7 the round-13 mismatch is 0 of 180. For Feb 12 to 25 it is 434 of 605 (71.7%). Outside round 13 the mismatch is 41 of 14,602 (0.28%). Round 25 shows 24 of 49; its meaning (likely first overtime round) is UNVERIFIED. VERIFIED counts (`scripts\swarm\out\e2_side_check_r13.log`).
- **What it is not:** round_end is internally consistent. Its `win_reason_message` matches `winner_team_code` in every round-13 row inspected (`scripts\swarm\out\e2_r13_sensitivity.log`). The disagreement is not a one-round index shift: the bomb outcome matches the round-12 winner 48% of the time and the round-14 winner 65%. VERIFIED (`scripts\swarm\out\e2_side_check_shift.log`).
- **Planter side:** of the 434 round-13 mismatches, 421 have a bomb_planted row with planter side 2 (T), as expected; 13 have no planter row. VERIFIED.
- **Cause:** UNVERIFIED. Either the 2024 round-13 round_end winner or the 2024 round-13 bomb rows are wrong for these matches. Not resolved here. The 2024 `score_update` channel is not in the tome, so it could not be used as a third source.
- **Impact:** small on map-level CT rates (see sensitivity table). The 2024 headline numbers above keep round 13.

## Other data-quality notes (VERIFIED from run diagnostics)
- **Unknown-attacker deaths:** 2024 has 24,608 deaths in live windows with no attacker side (4.8% of 509,298). 2026 has 450 (0.2% of 222,830). Those deaths are not counted as kills. The 2024 kills per round (6.61) is therefore probably low relative to 2026 (7.14). Cause UNVERIFIED.
- **Team kills and self kills:** 2026 416 and 155. 2024 2,652 and 1,252. Both are excluded from kills.
- **Planted rows after round end:** 1,210 in 2026 and 1,317 in 2024. Excluded from plant-to-end.
- **Multiple planted rows in one round:** 6 rounds in 2024, none in 2026.
- **Rounds without live start:** 3 in 2026, 78 in 2024. Excluded from length, plant, and kill averages, but kept in the round count.

## 2026 batch split (2026-10-08 vs other days). VERIFIED (run log, batch split section)
The 2026 CT rate differs by batch on several maps. de_cache: 54.2% on 2026-10-08 (1,718 rounds) vs 47.1% on other days (978 rounds). de_dust2: 49.6% (4,249 rounds) vs 53.9% (1,812 rounds). de_nuke: 50.8% vs 53.5%. de_mirage and de_ancient differ by 1 to 2 points. de_anubis and de_inferno match. This is the same question as the existing `notes\eda\batch_effect_2026-10-08.md`, which was not re-checked here. UNVERIFIED relation to that report.

## Files
- Script (existing, re-run this session with exit 0): `scripts\swarm\e2_map_profile.py`. Outputs `scripts\swarm\out\e2_map_profile.json` and `scripts\swarm\out\e2_map_profile_run_1545.log`.
- Cross-check: `scripts\swarm\e2_crosscheck.py`.
- Side-code checks: `scripts\swarm\e2_side_check.py`, `scripts\swarm\e2_side_check_2024_detail.py`, `scripts\swarm\e2_side_check_r13.py`, `scripts\swarm\e2_side_check_shift.py`.
- Round-13 sensitivity: `scripts\swarm\e2_r13_sensitivity.py`.
- Logs: `scripts\swarm\out\e2_*.log`.
