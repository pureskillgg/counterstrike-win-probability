# Decision window and movement start (feasibility, Task A)

Read-only feasibility check. Nothing in the sample, the tome, or any existing output was edited.

- Sample: `data\state_sample` (2,673 clean rounds, 150 matches, 1,231,817 player-snapshot rows, 179,466 one-second slices).
- Tome: `tomes\2026-10-01_to_08\tome\csds` (read-only).
- Script: `scripts\eda\feas_decision_window.py`. Log: `logs\feas\feas_decision_window.log`. Per-round table: `logs\feas\feas_decision_window_rounds.csv`.
- Supporting probes: `scripts\eda\feas_probe_schema.py`, `feas_probe_values.py`, `feas_probe_decision_detail.py`, `feas_probe_speed.py` (logs in `logs\feas\`).

## Bottom line

- In this sample the decision tick equals `round_end.tick` in all 2,673 rounds. The decision window therefore removes **0.00 of 178,194.81 live seconds** (0.0000). The bake-off live window (freeze to round end) already ends at the outcome-determining event.
- Post-decision 1 Hz snapshots: **0**. This is by construction (decision = round end, and the extraction keeps only ticks before round end), not an empirical finding about the game.
- Movement start cannot be resolved at 1 Hz. In all 2,673 rounds some player is more than 50 units from the freeze position at the first 1 s snapshot, so the first detection is 1 s in every round.

## Freeze and end (item 1, 2)

| Check | Result |
|---|---|
| `round_freeze_end` rows per round | exactly 1 in all 2,673 rounds |
| `rounds.live_start_tick` = `round_freeze_end` tick | 2,673 / 2,673 |
| `round_end.tick` = `rounds.round_end_tick` | 2,673 / 2,673 |
| `round_end` null winner or reason | 0 |
| `freezetime_ended_inferred` (a `round_state` event_type, present in all rounds) minus freeze, ticks | median +1; +1 in 2,657 rounds; negative in 16 rounds (-1280 x11, -960 x2, -857 x1, -764 x1, -270 x1); 0 in none |

## Decision tick (item 3)

Reason names come from `WIN_REASON` in `scripts\eda\batch_effect.py`, parsed from the file, not re-typed.

| win_reason_code | Name | Decision rule | Rounds | Share | Decision tick equals round end |
|---|---|---|---|---|---|
| 8 | CTs_Win | elimination (last losing-side death) | 1,053 | 39.4% | yes |
| 9 | Terrorists_Win | elimination (last losing-side death) | 1,083 | 40.5% | yes |
| 7 | Bomb_Defused | `bomb_state.bomb_defused` | 314 | 11.7% | 314 / 314 |
| 1 | Target_Bombed | `bomb_state.bomb_exploded` | 163 | 6.1% | 163 / 163 |
| 12 | Target_Saved | decision = round end (time-out) | 60 | 2.2% | by definition |

Totals by decision type: elimination 2,136 (79.9%), bomb_defused 314 (11.7%), bomb_exploded 163 (6.1%), time-out or other 60 (2.2%). Codes 11, 13, 17 and 18 do not occur in the clean rounds.

Decision minus freeze, in seconds (median, p10, p90):

- elimination: 58.5, 36.3, 92.2
- bomb_defused: 75.0, 54.2, 111.0
- bomb_exploded: 85.8, 70.8, 116.4
- time-out: 115.0 for all 60 rounds

Elimination checks:

- The winner matches the reason code in 2,673 / 2,673 rounds (8 means CT win, 9 means T win).
- The losing side's last death in the window equals round end in 2,136 / 2,136 elimination rounds, and no losing-side death falls after round end.
- Distinct losing-side victims in the window: 5 in 2,135 rounds, 4 in one round (match `csds/2026/10/07/m_7392c5c6d537a57c2e735e98/csds`, round 16, code 8). This round is an unexplained anomaly; I did not investigate it.

Null victim team (`player_death.player_team_code`):

- Within the 2,673 clean rounds there are 19,554 deaths, and none has a null team (0 nulls at any tick, 0 inside the window).
- The 51 null-team rows in the sample matches all fall in rounds that are not in the clean sample.
- Because there were no nulls, the `player_info` fallback was never used. There were 0 `player_info` team conflicts.

Bomb events:

- `bomb_state.bomb_defused`: 314 clean code-7 rounds, each at round end.
- `bomb_state.bomb_exploded`: 163 clean code-1 rounds, each at round end. A further 29 code-9 rounds (T win by elimination) have an explosion after round end, by up to 444 ticks (6.9 s). Those explosions fall outside the live window, and the decision for those rounds is still the last losing-side death.
- The `bomb_defuse` channel holds only `bomb_begin_defuse` rows (610 in the sample matches). Completion is therefore taken from `bomb_state.bomb_defused`. Cross-check: the last begin-defuse row before completion lies 5.0 s before (p10 5.0, p90 10.0 s), which matches the kit and no-kit defuse durations.

## Snapshots after decision (item 4)

| Quantity | Value |
|---|---|
| `players_1hz` rows with tick > decision | 0 of 1,231,817 (0.0000) |
| One-second slices with tick > decision | 0 of 179,466 (0.0000) |
| Rounds with at least one post-decision snapshot | 0 of 2,673 (0.0000) |
| Median seconds after decision per round | undefined (no such rows) |

Reason: decision equals round end in every round, and the extraction keeps only ticks before round end. Post-round time is therefore absent from the sample. In the tome, `round_officially_ended` falls a median 448 ticks (7.0 s) after `round_end` (min 447, max 545, n = 2,672; one round lacks the event).

## Movement start (item 5)

| Quantity | Value |
|---|---|
| Rounds with a freeze snapshot | 2,673 / 2,673 |
| Rounds with any player moved more than 50 units at some snapshot | 2,673 / 2,673 |
| Rounds whose first detection is the 1 s snapshot | 2,673 / 2,673 (100%) |
| Seconds after freeze of first detection (median, p10, p90) | 1.00, 1.00, 1.00 |

Resolution: a first detection at k means movement occurred in (k-1, k] seconds, so the 1 Hz data can only say movement began within the first second.

Diagnostics at the 1 s snapshot:

- 26,728 player pairs (alive at freeze and at 1 s). 81.6% moved more than 50 units (T 78.9%, CT 84.4%).
- Displacement median 192.9, p10 0.0 (3,296 pairs, 12.3%, exactly zero), p90 209.5 units.
- Players moved per round: median 8 of 10; 461 rounds with all 10 moving; 0 rounds with none.

Continuity check: over 1,205,087 consecutive 1 s steps per player, displacement has median 111.4, p99 250.0 and max 425.5 units. Only 205 steps (0.017%) exceed 260 units and 1 exceeds 300. The positions are continuous, not teleporting, so the 1 s result is not a tracking artifact.

Supplementary, not part of the spec and not used: the first `player_footstep` after freeze has median 1.92 s (p10 0.66, p90 4.08), 20.5% within 1 s. Footsteps do not corroborate movement: only 2.9% of pairs that moved more than 50 units in the first second had a footstep in that second (637 of 21,823), against 0.02% of unmoved pairs (1 of 4,905). The channel is sparse (1.33M rows for the whole tome), so it is not a usable onset signal.

## Live window versus decision window (item 6)

| Quantity | Value |
|---|---|
| Live seconds, freeze to round end (bake-off window) | 178,194.81 (mean 66.7 s per round) |
| Decision-window seconds, freeze to decision | 178,194.81 |
| Seconds removed | 0.00 (0.0000 of live) |
| Per-round share removed (median, p10, p90) | 0, 0, 0; rounds with any removal: 0 |
| One-second slices, live window | 179,466 (equal to the expected grid) |
| One-second slices, decision window | 179,466 (removed 0) |

## Assumptions and deviations

- ASSUMPTION: the radius of 50 units is taken from the task. Distance is 2D Euclidean on `(x_pos, y_pos)` in CS2 world units, with z ignored.
- ASSUMPTION: the 1 Hz grid (freeze + 64k, ticks before round end) is as extracted by `scripts\eda\extract_state_1hz.py`. I did not re-derive it.
- Tick rate 64 is verified, not assumed: the header `tick_rate` is 64 for all 150 matches.
- Codes 8 and 9 are read as elimination wins, following the reason names. The winner check and the 5-victim check (2,135 of 2,136) support this.
- Deviation: the literal instruction for bomb_defused points to the `bomb_defuse` channel, which contains only begin events. Completion is taken from `bomb_state.bomb_defused` instead.
- `freezetime_ended_inferred` is present but sits one tick after `round_freeze_end`, so freeze is taken from `round_freeze_end`.
- `alive_gap_snapshots` (9 snapshots in 1 round) is the extraction's own count of alive-count mismatches. It is not missing slices. All 179,466 grid slices are observed.

## Implications for the win-probability question

- In this sample there is nothing to remove. The live window already ends at the outcome-determining event, so a decision cut-off changes no rows.
- The 1 Hz sample cannot resolve movement start. Resolving it needs tick-level positions. The tome carries positions on some event channels (for example `basic_weapon_fire`, `basic_item_pickup`, `basic_player_hurt` have `player_x_pos`), but not every tick. That would be a separate feasibility check.
- Any post-round signal (about 7 s after round end) is outside both windows and is not in the sample.
