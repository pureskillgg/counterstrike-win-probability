# SageMaker tome build: current inputs, coverage and status

Written 2026-11-03 from work on the adam-gpu space. Scripts are in `pipeline/sagemaker/`. Nothing here is committed to the other branch, and no match data is in the repo.

## Tome inputs (what the bake-off reads today)

- **Header tome**, one row per match: `key` (`csds/YYYY/MM/DD/<match_id>/csds`), `map_name`, `match_date`, `tick_rate` (64), `max_rounds`, `t_starters_score_final`, `ct_starters_score_final`. The rebuilt header fills `max_rounds` with 16 or 24. The admin's header leaves it null for every match.
- **DENSE channels** (`scripts/model/bakeoff.py`, line 87): `round_end`, `round_start`, `round_state`, `player_death`, `player_info`, `bomb_state`, `team_change`. A match with no keyset entry for one of these is dropped.
- **SPARSE channels** (line 89): `player_disconnect`, `bot_takeover`. Zero rows is normal.
- **Label**: `ct_win = (winner_team_code == 3)` from `round_end`. Code 2 is T and code 3 is CT.
- **Event features** (`FEATS`, line 90, 10 features): `alive_ct`, `alive_t`, `ct_alive_minus_t`, `bomb_planted`, `seconds_since_round_live`, `seconds_since_plant`, `round_number`, `ct_score_minus_t_score`, `is_pistol_round`, `is_overtime`. Built per state (each live round start and each death in the live window). No per-tick data.
- **Not in the basic tomes**: positions, equipment, health and money. The 59-input base and the transformer need per-tick channels from a separate pass. Its spec is in `scripts/data/build_random_tick_pool59.py` and `scripts/model/exp_59_nomono.py`, which this note hasn't reviewed.

## Coverage

- Admin key list: 67,224 matches. Two ranges with no overlap:
  - 2023-11-23 to 2026-07-10: 46,627 matches on 365 calendar days.
  - 2026-10-09 to 2026-10-25: 20,597 matches on 17 days, including 2,598 on 10-09 and 10-10, which the handoff marks unreconciled.
- Missing from both buckets: 2026-07-11 to 2026-10-08 (July 11–31, August, September, October 1–8). The source bucket `csds/2026/` holds only October 9–31 and November.
- Holdout candidate: 2026-10-26 to 10-31, 3,463 matches in the source zips. Not in the admin's tomes.

## What was tested

- 100 matches from 2026-10-14, built from loose files: the header and five channels (`round_end`, `round_state`, `player_death`, `player_info`, `bomb_state`) match the admin's tomes row for row per match.
- `team_change` is present for 99 of those 100 matches. The admin's tomes have it only for the newer range.
- The DSDK builder needs pyarrow 25. pyarrow 21 fails with an unhashable schema error.
- `build_basic_tomes_from_fs` needs at least one channel, and the default tome name is `basic_{channel}.{dates}`.
- Throughput on the c6id.2xlarge: 100 matches fetched in about 25 seconds (about 170 files per second, the bottleneck). The build itself took about 10 seconds.

## Status

- Whole-archive header build: running on the space, roughly two-thirds of the matches fetched at the time of writing. `pipeline/sagemaker/header_full.py` verifies against the admin's keys before uploading to `adam/tomes/header_build/`.
- Holdout tomes for 2026-10-26 to 10-31: running. `pipeline/sagemaker/holdout_build.py` uploads to `adam/tomes/holdout_2026-10-26_31/`.
- Basic channels for the whole archive: not started. The plan is chunked fetch, build, upload and cleanup, using `pipeline/sagemaker/pipe.py` as the pattern.

## Open items

- Ask the admin about August, September and July 11–31, and about October 1–8, which may exist on the laptop's E: drive per the handoff.
- Decide whether days 2026-10-09 and 10-10 stay in the build.
- The 59-input base needs a per-tick pass that hasn't been built.
