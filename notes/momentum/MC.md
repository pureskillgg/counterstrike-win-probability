# MC: streak mechanism, economy channel for momentum

Economy channel for momentum. Association only. Built 2026-10-13 from the 1 Hz sample (150 matches, days 01 to 07). Outputs: `data/viz/MC.json`, `data/viz/MC.svg`. Scripts: `scripts/momentum/MC_01_team_rounds.py`, `scripts/momentum/MC_02_streak_economy.py`.

## Readout

In the 1 Hz sample, a team's round win rate tracks its streak mostly at length 1. After one loss the team wins 38.2% of non-pistol rounds (95% CI 35.2 to 41.2), against 61.8% after one win (58.8 to 64.8). Longer streaks add little: the win rate sits between 34.7% and 38.7% for 1 to 5+ losses and between 61.3% and 65.3% for 1 to 5+ wins. Economy moves with the streak. Team equipment at freeze end is 15.6k after one loss and 23.8k after a round won, and rises again to 18.2k after 5+ losses, where the win rate stays flat. Equipment difference against the opponent accounts for most of the raw gap: conditioning on equipment-difference quintile leaves residuals of -1.5 pp after one loss and +1.6 pp after one win, each with a 95% interval about 6 points wide. The loss-bonus ladder is not visible as a monotone rise in freeze-end money. Money falls from 1,812 per player after one loss to 1,190 after four, and equipment rises at four or more losses (3,509 and 3,647). Freeze-end snapshots cannot separate bonus received from spending. Equipment difference is itself downstream of the streak, so the residual check does not show mechanism.

## Exact numbers (primary: rounds 2 to max, excluding pistol round 13; 4,794 team-rounds)

Per-player figures are team totals divided by 5. Win rates are pooled. Intervals are 95% match-cluster bootstrap, 1000 resamples, seed 0.

After a loss streak (k = consecutive losses immediately before the round; k = 0 means the previous round was won):

| k | team-rounds (matches) | win % [95% CI] | team equipment $ | team money $ | per-player money $ | per-player equipment $ |
|---|---|---|---|---|---|---|
| 0 | 2329 (149) | 63.1 [61.1, 65.3] | 23,844 | 17,186 | 3,437 | 4,769 |
| 1 | 931 (149) | 38.2 [35.2, 41.2] | 15,626 | 9,058 | 1,812 | 3,125 |
| 2 | 539 (147) | 34.7 [31.2, 38.4] | 14,650 | 6,641 | 1,328 | 2,930 |
| 3 | 314 (142) | 36.0 [30.8, 41.3] | 15,064 | 6,275 | 1,255 | 3,013 |
| 4 | 186 (126) | 38.7 [31.7, 45.9] | 17,546 | 5,950 | 1,190 | 3,509 |
| 5+ | 250 (91) | 36.4 [31.0, 42.5] | 18,237 | 6,347 | 1,269 | 3,647 |

After a win streak (k = consecutive wins immediately before the round; k = 0 means the previous round was lost):

| k | team-rounds (matches) | win % [95% CI] | team equipment $ | team money $ | per-player money $ | per-player equipment $ |
|---|---|---|---|---|---|---|
| 0 | 2329 (149) | 36.9 [34.7, 38.9] | 15,777 | 7,496 | 1,499 | 3,155 |
| 1 | 931 (149) | 61.8 [58.8, 64.8] | 22,057 | 11,046 | 2,209 | 4,411 |
| 2 | 539 (147) | 65.3 [61.6, 68.8] | 23,932 | 16,032 | 3,206 | 4,786 |
| 3 | 314 (142) | 64.0 [58.7, 69.2] | 25,437 | 20,409 | 4,082 | 5,087 |
| 4 | 186 (126) | 61.3 [54.1, 68.3] | 25,764 | 24,441 | 4,888 | 5,153 |
| 5+ | 250 (91) | 63.6 [57.5, 69.0] | 26,598 | 32,389 | 6,478 | 5,320 |

Win-rate residual after conditioning on equipment-difference quintile (percentage points, 95% CI). Loss series: k = 1 -1.54 [-4.80, 1.64]; k = 2 -1.28 [-5.23, 2.66]; k = 3 +1.90 [-3.33, 7.07]; k = 4 +1.81 [-4.98, 9.05]; k = 5+ -0.05 [-5.02, 5.85]. Win series: k = 1 +1.56 [-1.61, 4.84]; k = 2 +1.21 [-2.73, 5.16]; k = 3 -1.92 [-7.07, 3.31]; k = 4 -1.76 [-9.00, 5.02]; k = 5+ +0.08 [-5.81, 5.05].

Loss-bonus ladder (losing side after k consecutive losses): per-player money 1,812 (1,728 to 1,903), 1,328, 1,255, 1,190, 1,269 for k = 1, 2, 3, 4, 5+. Per-player equipment 3,125, 2,930, 3,013, 3,509, 3,647. The reference loss bonus of 1,400 to 3,400 is an external CS2 rule and is not checked here.

Sensitivity including pistol round 13: win rate after one loss 39.0%, after one win 61.0%. The other buckets move by under 2 points.

## Checks

- Team-rounds: 5,334 from 2,667 kept rounds across 150 matches. Excluded: the flagged anomaly (22e9c3e7, round 16; 9 players at freeze end) and 5 rounds of match 52feeb79 whose player-ID partition differs from the match's dominant partition.
- Censored streaks: 245 primary team-rounds have a run that reaches an unobserved round before the run ends, so they are excluded from the streak tables, not imputed. 46 of 150 matches have gaps in round numbering.
- Team identity: per match, the 5-player group holding the smallest player_id is team X. The two-team partition is stable across rounds in 149 of 150 matches (the exception is 52feeb79).
- Team-code mapping (3 = CT, 2 = T): team equipment and team money match `round_buy_types` on 5,334 of 5,334 team-rounds.
- Win identity: win rate across all rows is exactly 0.5.
- Overtime sides: observed flip at round 13, no flip at round 25, flips every 3 rounds from round 28. 0 violations across 2,588 rounds with round 1 observed.
- Money is post-buy: 94.8% of round-1 armor buyers show money exactly 150 (800 minus the 650 Kevlar price), and 99.1% show money plus equipment exactly 1,000 (800 start money plus the 200 default pistol).
- Known-fact cross-check: after one win, non-pistol win rate is 61.8%, close to the 61.4% in the brief. Definitions differ, so this is a loose check.

## Caveats

- Association only. Nothing here identifies a cause. Equipment difference is a possible mediator, so the residual check tests whether equipment explains the streak gap, not whether it is the mechanism.
- Freeze-end money is the balance after the buy. Equipment includes items carried over from earlier rounds, so money plus equipment is not a pre-buy bankroll outside the pistol rounds. For this reason no bankroll ladder is reported.
- Streak lengths 3 and up rest on 186 to 314 team-rounds each (250 for 5+), so their intervals are wide.
- The 1 Hz sample is 150 matches (days 01 to 07). Results may not generalise to the full 2026 data (1,752 matches).
- Not checked: round winners against the round_state score fields, and the external loss-bonus schedule. The claim that winning the pistol round predicts the game (the "3x" figure in the request) is outside this task and is not tested here.
