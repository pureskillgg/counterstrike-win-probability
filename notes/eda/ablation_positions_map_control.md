# Ablation: bomb-site positions and map control (round win probability)

Script: `scripts/eda/ablation_positions.py`. Log: `logs/eda/ablation_positions.log`. Results: `logs/eda/ablation_positions_results.json`. Snapshot features: `data/state_sample/positions_snapshots.parquet` (179,392 rows).
Inputs: `data/state_sample/players_1hz.parquet`, `rounds.parquet`, and the validation log (preflight). No tome or E: reads.

Question: do in-round positions add information about the round winner beyond BASE (alive counts, bomb state, time, score)? Does map name add more?

## Answer

- VERIFIED: with `score_diff` in BASE (the specified BASE), positions do not help. Log loss +0.0093 (95% bootstrap [-0.0057, +0.0244]); 90% of draws are worse. Brier and AUC gains also cross zero. The positions model is worse on all 5 XGBoost seeds (mean +0.0080).
- VERIFIED: without `score_diff` (diagnostic), positions cut log loss by 0.0151 (95% [-0.0292, -0.0013]; 98.5% of draws better). Brier (-0.0035) and AUC (+0.0085) cross zero. The no-score positions model (0.5350) is still worse than BASE with score (0.5168). Positions appear to recover part of what the score carries, not add to it. UNVERIFIED mechanism; not tested directly.
- VERIFIED: map control adds nothing beyond positions. BASE+POS+MAP minus BASE+POS: log loss -0.0066 [-0.0167, +0.0028] with score; +0.0005 without.
- VERIFIED: gain importance overstates map. Map takes 0.198 of gain in BASE+POS+MAP but no log-loss gain.

## Sample and split

- VERIFIED: 150 matches (train 120, test 30). Rounds 2,673. Dropped the one round with `alive_gap_snapshots > 0` (match `22e9c3e7-a333-4ebc-a51c-19ebbcc13ef1`, round 16, de_train, train). It has 74 snapshots in total; 9 are the flagged ones. Main sample: 2,672 rounds (train 2,184, test 488), 179,392 snapshots.
- VERIFIED: inner validation = 18 of 120 train matches (15%, `random.Random(0)` shuffle of the sorted match list). Fit 102 matches (123,189 rows); validation 23,319 rows (CT-win share 0.602, against 0.509 in fit); test 32,884 rows (30 matches).
- VERIFIED: no match on both sides (asserted in the script).

## Planter positions and bomb sites

- VERIFIED: planter = the T player holding `inv_c4` at the latest snapshot at or before `plant_tick`. Position = that player's row at `plant_tick` if present, otherwise their nearest snapshot. All 1,238 planted rounds were tier 1 (carrier at or before the plant). Exact row at `plant_tick`: 18. Median |snapshot - plant| = 16 ticks (0.25 s), max 48.
- VERIFIED: k-means k = 2 (3D, n_init 10, seed 0) per map, fit on TRAIN planted rounds only. A = larger cluster. Test plants used only for the out-of-sample column.

| map | train plants | A / B | centre distance | separation ratio | silhouette | test plants | test within 1000 | test median dist |
|---|---|---|---|---|---|---|---|---|
| de_ancient | 101 | 63 / 38 | 2302 | 20.5 | 0.947 | 31 | 100% | 97 |
| de_anubis | 97 | 55 / 42 | 2653 | 25.0 | 0.953 | 35 | 100% | 87 |
| de_cache | 126 | 73 / 53 | 3100 | 20.5 | 0.945 | 8 | 100% | 92 |
| de_dust2 | 198 | 109 / 89 | 2679 | 21.3 | 0.943 | 41 | 100% | 97 |
| de_inferno | 203 | 102 / 101 | 2980 | 15.2 | 0.921 | 24 | 100% | 158 |
| de_mirage | 158 | 89 / 69 | 2898 | 19.7 | 0.939 | 68 | 100% | 130 |
| de_nuke | 105 | 74 / 31 | 544 | 3.1 | 0.543 | 12 | 100% | 213 |
| de_train | 31 | 18 / 13 | 1405 | 8.6 | 0.867 | 0 | n/a | n/a |

Separation ratio = centre distance / mean within-cluster RMS distance. de_nuke is weakly separated (ratio 3.1, silhouette 0.54), so its A/B split is the least reliable.

## Features (3D game units; NaN where undefined)

- Per team: mean and min distance of alive players to A and to B; alive players within 1000 of A and of B; team spread (mean pairwise distance).
- T carrier distance to nearest site (NaN when no T carrier alive; 64,772 snapshots).
- After plant: alive CT within 1500 of the planted bomb, and nearest CT distance (NaN before plant, and for 0 alive CT).
- BASE: alive_ct, alive_t, alive_diff, bomb_planted, seconds_since_live, seconds_since_plant (-1 if none), round_number, score_diff. Map: one-hot (7 maps in the test set; de_train has no test rounds).

## Test results (30 matches)

| model | features | test log loss | Brier | AUC | best iter |
|---|---|---|---|---|---|
| CONST (fit CT share) | 0 | 0.69132 | 0.24909 | 0.500 | n/a |
| BASE | 8 | 0.51683 | 0.17226 | 0.81900 | 343 |
| BASE+POS | 25 | 0.52603 | 0.17695 | 0.81180 | 476 |
| BASE+POS+MAP | 33 | 0.51942 | 0.17485 | 0.81324 | 194 |
| BASE (no score_diff) | 7 | 0.55000 | 0.18518 | 0.78710 | 529 |
| BASE+POS (no score_diff) | 24 | 0.53504 | 0.18170 | 0.79549 | 314 |
| BASE+POS+MAP (no score_diff) | 32 | 0.53548 | 0.18100 | 0.79862 | 292 |

Gain importance (share of total gain): BASE+POS POS 0.304. Top positions features: ct_n_near_bomb 0.060, ct_nearest_bomb_dist 0.035, t_mean_dist_A 0.019, t_spread 0.018, ct_spread 0.018, t_carrier_site_dist 0.016. alive_diff is 0.535. The POS gain is mostly post-plant bomb proximity, which is close to bomb state.

## Bootstrap (1,000 match-level draws over 30 test matches, seed 0)

Difference = first minus second. Interval = 2.5 to 97.5 percentile. "Better" = share of draws where the first model is better.

| comparison | log loss mean [95%] | better | Brier mean [95%] | AUC mean [95%] |
|---|---|---|---|---|
| with score: POS - BASE | +0.00925 [-0.00566, +0.02437] | 0.10 | +0.00475 [-0.00091, +0.01037] | -0.00738 [-0.01950, +0.00427] |
| with score: MAP - BASE | +0.00264 [-0.01171, +0.01599] | 0.35 | +0.00264 [-0.00267, +0.00750] | -0.00578 [-0.01602, +0.00586] |
| with score: MAP - POS | -0.00661 [-0.01674, +0.00280] | 0.92 | -0.00211 [-0.00568, +0.00153] | +0.00160 [-0.00531, +0.00847] |
| no score: POS - BASE | -0.01507 [-0.02916, -0.00131] | 0.99 | -0.00348 [-0.00781, +0.00081] | +0.00853 [-0.00290, +0.01963] |
| no score: MAP - BASE | -0.01461 [-0.02767, -0.00223] | 0.99 | -0.00415 [-0.00860, +0.00057] | +0.01162 [+0.00002, +0.02424] |
| no score: MAP - POS | +0.00046 [-0.00900, +0.01089] | 0.49 | -0.00067 [-0.00406, +0.00318] | +0.00309 [-0.00493, +0.00979] |

Seed noise (5 XGBoost seeds, same split, with score): BASE log loss sd 0.00055; BASE+POS sd 0.00229; POS - BASE mean +0.00795 (sd 0.00231), positive on all 5 seeds.

## Per-map differences (with score; BASE+POS minus BASE; test rows)

| map | test matches | log loss diff | 95% per-map bootstrap | Brier diff |
|---|---|---|---|---|
| de_anubis | 5 | +0.03248 | [+0.00593, +0.04651] | +0.01397 |
| de_nuke | 2 | +0.04659 | [-0.01517, +0.16650] | +0.01759 |
| de_cache | 1 | -0.03179 | degenerate (1 match) | -0.01239 |
| de_inferno | 4 | -0.01734 | [-0.04344, +0.06020] | -0.00665 |
| de_mirage | 9 | +0.00623 | [-0.00906, +0.02031] | +0.00416 |
| de_dust2 | 6 | +0.00739 | [-0.00751, +0.02263] | +0.00268 |
| de_ancient | 3 | -0.00045 | [-0.06436, +0.07409] | +0.00400 |

Per-map samples are 1 to 9 matches and there is no multiplicity correction across 7 maps. Only de_anubis has an interval excluding zero, and it is in the direction of POS hurting.

## Caveats

- VERIFIED: this replaces an earlier version of this script and its outputs. That version took bomb sites from all 2026 tome matches (test included) via the bomb_state tome, used a 20% validation split, kept the flagged round, and had no no-score variant. Its log showed BASE test log loss 0.52151 and BASE+POS 0.52260. Those numbers are superseded.
- VERIFIED: cluster labels A/B are by cluster size per map. The game's site codes were not used, so the mapping of A and B to real sites is UNVERIFIED.
- UNVERIFIED: whether the derived centres match the game's bomb sites. Test plants land within 1000 units of the train centres on all maps (100%), which supports the clustering but does not confirm the site identity.
- The planter position is taken from players_1hz, about 16 ticks from the plant on median. Some error is expected.
- Snapshots are dependent within a round (about 67 per round). All intervals are match-level, but with 30 test matches they are wide.
- The score-diff verdict is the one that matters for the spec. The no-score gain is a diagnostic only.

Reproduce: `pixi run python -I -X utf8 scripts\eda\ablation_positions.py` from `G:\pureskill\csds-cs2` (about 8 minutes).
