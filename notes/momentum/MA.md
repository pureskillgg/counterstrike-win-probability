# MA: pistol round to match outcome

Script: `scripts/momentum/MA_pistol_match.py` (project interpreter, `-I -X utf8`, about 5 s). Outputs: `data/viz/MA.json`, `data/viz/MA.svg`. Data: `basic_round_end` and `header` channels in `tomes/2026-10-01_to_08`. Association only, no causal claims. Intervals: match-cluster bootstrap (matches resampled, both team rows together), 1,000 resamples, seed 0, percentile 95%.

## Readout

Among 1,453 decided 24-round 5v5 matches with both pistols played, the team that won round 1 won the match 57.9% of the time (95% CI 55.3 to 60.4), and the round 13 pistol winner won 60.3% (57.9 to 62.6). Teams that won both pistols won 67.9% (64.4 to 71.0) and teams that lost both won 32.1% (29.0 to 35.6); single-pistol teams sat near a coin flip (round 1 only 47.6%, round 13 only 52.5%). The pistols are not linked: the same team won both 50.8% of the time (48.0 to 53.3), against 50% under independence. The round 1 odds ratio, winner vs loser over team rows, is 1.89 (1.53 to 2.33). On a per-match basis the pistol winner's odds are 1.37 (1.24 to 1.53). The snippet's 2.95 is not reproduced: its "about three times" is outside our interval under either reading (it would imply 63% or 75% pistol-winner match wins; we see 58%). The more important point for momentum is that these rates are what counting one round predicts. If rounds were independent fair coins and a team needed 13 of 24 rounds, a team that won a given round would win the match 58.0% of the time (2 million simulated matches), and the observed 57.9% sits on that line. Round 1 is not special: rounds 2 to 12 each show 61 to 64%, and round 1 sits 4.4 points below their per-match average (matched difference -4.4, 95% CI -7.0 to -1.7). Round 13 shows no detectable excess over the counting benchmark (60.3% vs 58.0%, interval covers the benchmark). Mapping does not move the round 1 odds ratio (1.89 unadjusted, 1.88 map-stratified). Round-to-round persistence (round 2 winner = round 1 winner) is 83.0% (81.1 to 84.9), a separate measure not decomposed here.

## Exact numbers

### Population funnel (main)

| Step | Remaining |
|---|---:|
| Header rows (one per match key) | 1,752 |
| Duplicate copies removed (identical round_end sequence, 47 groups) | 1,700 |
| max_rounds = 24 (5v5) | 1,553 |
| round_end rounds = header final total (complete) | 1,487 |
| Header finals differ (decided winner; 18 ties dropped) | 1,469 |
| Team identity: swap rule reproduces header totals | 1,469 (0 excluded) |
| Round 1 and round 13 present (16 decided matches end before round 13) | 1,453 |

Wingman sensitivity (max_rounds 16, pistol 2 = round 9): 147, 135 complete, 114 decided, 104 team-consistent (10 failed the team check), 103 with round 9 present.

### Pistol patterns (team rows; each match gives one row per team)

| Pattern | Matches | P(team wins match) | 95% CI | Fair-coin benchmark |
|---|---:|---:|---|---:|
| Won both pistols | 738 | 67.9% | 64.4 to 71.0 | 66.1% |
| Won round 1 only | 715 | 47.6% | 43.9 to 51.3 | 49.9% |
| Won round 13 only | 715 | 52.5% | 48.7 to 56.1 | 50.0% |
| Lost both pistols | 738 | 32.1% | 29.0 to 35.6 | 33.8% |

### Single pistols and linkage

| Quantity | Value | 95% CI | Fair-coin benchmark |
|---|---:|---|---:|
| P(team wins match / won round 1) | 57.9% | 55.3 to 60.4 | 58.0% |
| P(team wins match / won round 13) | 60.3% | 57.9 to 62.6 | 58.0% |
| P(won round 13 / won round 1) = P(same team won both) | 50.8% | 48.0 to 53.3 | 50.0% |
| P(won round 13 / lost round 1) | 49.2% | 46.7 to 52.0 | 50.0% |
| Difference, P(round 13 / won round 1) minus P(round 13 / lost round 1) | +1.6 pts | -3.9 to +6.7 | 0 |

### Odds ratios

| Comparison | Team-row OR (brief definition) | Match-level odds |
|---|---:|---:|
| Round 1 winner vs loser | 1.89 (1.53 to 2.33) | 1.37 (1.24 to 1.53) |
| Round 13 winner vs loser | 2.30 (1.89 to 2.81) | 1.52 (1.37 to 1.68) |
| Both pistols vs neither | 4.47 (3.28 to 6.00) | 2.11 (1.81 to 2.45) |
| Round 1 and round 13 link (same team) | 1.07 (0.85 to 1.31) | 1.03 (0.92 to 1.14) |

The team-row OR equals the square of the match-level odds, because each match contributes one winner row and one loser row with complementary outcomes. Both are reported. The snippet's 2.95 reads as 63.2% pistol-winner wins under the team-row reading and 74.7% under the match-level reading; neither interval contains that rate (ours is 57.9%, 55.3 to 60.4).

### Round-by-round profile (P(team that won round k wins match), main population)

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| % | 57.9 | 63.2 | 62.0 | 62.9 | 62.3 | 62.2 | 61.3 | 63.6 | 62.5 | 62.5 | 61.3 | 61.1 | 60.3 |
| 95% CI low | 55.3 | 60.7 | 59.3 | 60.4 | 59.9 | 59.5 | 58.7 | 61.2 | 59.9 | 60.2 | 58.6 | 58.5 | 57.9 |
| 95% CI high | 60.4 | 65.5 | 64.4 | 65.3 | 64.8 | 64.8 | 63.7 | 66.1 | 65.0 | 65.2 | 63.8 | 63.7 | 62.6 |

Matched difference, round k minus the per-match average over rounds 2 to 12: round 1 -4.4 pts (-7.0 to -1.7); round 13 -2.0 pts (-4.6 to +0.6).

### Fair-coin benchmark (simulated, 2,000,000 matches, Monte Carlo error about 0.05 pts)

24 independent 50/50 rounds; a team wins with at least 13 round wins; 12-12 decided by a coin (stands in for overtime).

| Pattern | Observed | Benchmark |
|---|---:|---:|
| Won round 1 | 57.9% | 58.0% |
| Won round 13 | 60.3% | 58.0% |
| Won both | 67.9% | 66.1% |
| Won round 1 only | 47.6% | 49.9% |
| Won round 13 only | 52.5% | 50.0% |
| Lost both | 32.1% | 33.8% |

### Control: map

- Team-row logistic, won round 1 only: 1.887 (1.534 to 2.329). With map one-hot (reference de_mirage; maps under 20 matches pooled): 1.887, identical, because the map terms are exactly zero in this design (see caveats).
- Mantel-Haenszel team-row OR stratified by map: 1.88 (1.53 to 2.31).
- Team-row logistic with both pistols and map: round 1 OR 1.91 (1.53 to 2.37); round 13 OR 2.33 (1.91 to 2.85).
- Per-map P(round 1 winner wins): de_ancient 61.1% (n = 203), de_anubis 58.1% (191), de_cache 63.2% (133), de_dust2 57.1% (287), de_inferno 58.8% (170), de_mirage 56.1% (303), de_nuke 55.0% (129), other pooled 46.0% (37).

### Round-to-round check

Round 2 winner = round 1 winner: 83.0% (81.1 to 84.9) on the 1,453 main matches. On all 1,660 decided, deduplicated matches (any format) it is 81.6%, against the 81.3% (79.6 to 83.1) given in the brief. Close but not identical, so the population differs.

### Sensitivity

| Set | Matches | P(R1 winner wins) | Team-row OR, R1 | P(R13 winner wins) | P(same team won both) |
|---|---:|---|---|---|---|
| Main | 1,453 | 57.9 (55.3 to 60.4) | 1.89 (1.53 to 2.33) | 60.3 (57.9 to 62.6) | 50.8 (48.0 to 53.3) |
| Both pistols with player_count 10 | 1,356 | 56.8 (54.1 to 59.1) | 1.73 (1.38 to 2.10) | 59.6 (56.9 to 62.1) | 49.7 (47.2 to 52.4) |
| Regulation only (no overtime) | 1,310 | 59.3 (56.7 to 62.1) | 2.13 (1.72 to 2.68) | 61.8 (59.1 to 64.4) | 51.1 (48.3 to 53.5) |
| Wingman (pistol 2 = round 9, team-consistent) | 103 | 64.1 (55.3 to 73.8) | 3.18 (1.54 to 7.92) | 68.0 (58.3 to 75.7) | 49.5 (40.8 to 59.2) |
| Main plus wingman | 1,556 | 58.3 (55.9 to 60.6) | 1.95 (1.61 to 2.37) | 60.8 (58.4 to 63.3) | 50.7 (48.2 to 53.1) |

The wingman subset is the only one near the snippet's 2.95 (OR 3.18), but it has 103 matches, a different format, and an interval from 1.54 to 7.92. It cannot be read as the same effect.

## Caveats

1. Association only. Pistol winners may be stronger teams, and strength raises both round wins and match wins. The round 2 to 12 excess (61 to 64%) may reflect strength rather than momentum; nothing here separates the two.
2. The fair-coin benchmark is a reference point, not a test of momentum. Unequal teams change the counting effect. The data show fewer overtime matches than the benchmark predicts: 143 of 1,453 (9.8%) went past round 24, against about 16% under fair 50/50 rounds, so real rounds are less even than the benchmark assumes.
3. Round 1 and round 13 are counted in the match total, so part of any pistol-to-match association is mechanical. The benchmark quantifies this for fair rounds only.
4. Team-row ORs are the square of match-level odds (see the odds ratio section). The snippet's 2.95 could be either reading; neither reproduces it.
5. Map control: the team-row logistic with map one-hot cannot move the odds ratio here. Each match has one winner row and one loser row, so per-map win totals are fixed and the map terms come out exactly zero. The informative check is the map-stratified Mantel-Haenszel estimate (1.88), which does not move. Per-map pistol rates vary (46 to 63%), but each map has 37 to 303 matches, so the per-map rates are noisy.
6. Duplicates: 52 copies removed using identical round_end sequences (round, tick, winner), keeping the lowest key. The bake-off's duplicate rule also compared deaths and player info; that rule was not reproduced.
7. Team identity: the swap rule reproduces the header totals for all 1,469 decided 24-round matches. The same check fails for 10 wingman matches, which are excluded from the wingman sensitivity; the cause was not investigated.
8. The CS:GO Majors snippet (OR 2.95, 334 matches) was not located or verified. It is compared numerically only.
9. Not checked: the bake-off kept set (1,521 matches) and the 1 Hz equipment sample were not used. Roster changes and disconnects were checked only through the player_count sensitivity. The regulation-only subset (2.13) excludes coin-decided overtime ties, which raises the counting benchmark too; this direction was not simulated in the script.
10. The fair-coin benchmark assumes a coin for 12-12 and 50/50 rounds, and the simulation uses 2 million matches. It is not a fitted model.
