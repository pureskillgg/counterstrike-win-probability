# Flow channel profile (Task 1)

Tome: `G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds`. 1,752 header matches. Each channel folder has one page (`dataframe_00000`, one schema, no schema drift) and a `keyset_00000`. Read-only. Scripts: `scripts\eda\schema_peek.py`, `flow_profile.py` (tables), `flow_checks.py` and `flow_checks2.py` (follow-ups), `post_end_plants.py`, `wingman_and_sentinels.py`.

Ticks are 64 per second. "Live start" = the `round_freeze_end` row in `round_state`.

## Goofy list (ranked by how much each could mislead the round model)

1. **Wingman and partial-roster rounds sit inside the 5v5 tome.** 3,951 of 35,992 `round_end` rows (11.0%) have `player_count != 10`, across 959 matches. 2,038 rounds belong to the 149 `is_wingman` matches (max_rounds 16, 7 to 16 rounds each); 1,971 of those rounds show 4 players. The other 1,913 rows are in 810 non-wingman matches with at least one non-10 round: 799 are mixed (some rounds at 10) and 11 have no 10-player round at all. Unfiltered, these break the 5-alive-per-side assumption. The bakeoff drops them with its `player_count != 10` rule. `header.is_wingman` is the clean flag.

2. **Bomb plants logged after `round_end`.** 1,479 of 18,306 `bomb_planted` rows (8.1%) have a tick after the round's `round_end` tick. None are before `round_start`. 1,455 of them are in T-elimination rounds (`win_reason_code` 9, about 10% of those rounds) and 24 in time-outs (code 12). Gap to `round_end`: median 5.0 s, max 7.0 s. Counting every `bomb_planted` row gives a plant rate of 50.9% per round. Counting only plants inside live start to `round_end` gives 46.8% (16,827 rounds). Feature code that takes the first plant row without a window check overstates plants by about 4 points. The bakeoff excludes these (`plants_after_end`). Why the plant is logged after the round is not verified against the demo.

3. **Bomb channel missing for 30 matches.** 30 matches (1.7%) have zero `bomb_state` rows. Their 379 rounds have no plant events in `bomb_state` or `round_state`, and no bomb-ending win (`win_reason_code` 1 or 7). At the tome-wide rates (plant about 51% per round, bomb-ending about 19%), about 190 plants and about 70 bomb-ending rounds would be expected. So this is missing data, not real no-plant matches. Kept as-is, these rounds would carry `bomb_planted = 0` in every state. The bakeoff drops all 30 under its missing-channel rule.

4. **`round_start` tick = 66 sentinel for round 1.** In 678 of 1,752 matches (38.7%), the round-1 `round_start` row (and the matching `round_state` `round_start` row) has tick 66 and `second` 1.03125, which is the first tick of the demo. The other 1,074 matches have real round-1 start ticks. Any feature that measures time from `round_start` is wrong for those rounds. The bakeoff measures from `round_freeze_end` instead, and 676 of the 678 sentinel rounds have one. Separately, 249 rounds run more than 200 s from `round_start` to `round_end` (max 529.6 s). Only one of them is a tick-66 round, so the sentinel does not explain the long rounds. The cause is not verified (pauses or timeouts are a guess).

5. **Raw `player_id = 0` sentinel.** The raw id is 0 in 9.7% of in-window deaths (24,084 of 247,717), 10% of `bomb_state` rows (2,577), 9% of `bomb_action` rows (14,798) and 11.7% of `round_mvp` rows (4,211). The `*_fixed` columns hold the real slot (1 to 18) for these rows. The bakeoff keys first-death-per-victim on the raw id. 27 rounds in the whole tome have two or more in-window raw-0 deaths, but none of them survive the round filters (checked: 0), so the current results are not affected. Any new code that keys on the raw id would collapse these deaths.

## Other things noticed (below the top 5)

- `header.match_type` is `unknown` for 1,074 of 1,752 matches (61%); `competitive` for 678. See the batch report.
- Rounds per match fall outside 16 to 30 in 275 matches (4,536 rounds). These are the 149 wingman matches, 1 to 15 round partial matches, and 37 matches with 34 to 48 rounds (overtime).
- `score_update`: 1,669 rounds have 3 rows instead of 1. The two extra rows are a score swap at a halftime-type point (team codes 2 and 3 exchange scores), so the deltas are ±2 to ±12 in those rounds. The bakeoff does not use `score_update`; it reads scores from `round_state`.
- `bomb_action`: 104 rows have |x_vel| > 1000 (max about 2.6e5), almost all `bomb_pickup`. `bomb_state.site_code` takes 751 values (66 to 1023), so it is not an A/B site code. Not used by the model.
- `round_mvp`: `mvp_count` runs 1 to 12, so it is not a per-round count. 5 (match, round) pairs have two MVP rows for different players. 53 rounds have no MVP row.
- 524 rounds with a `bomb_exploded` row end with `win_reason_code` 9 (Terrorists_Win), not 1. Both are T wins, so the label is unaffected.
- `round_state` has a `freezetime_ended_inferred` event (35,953 rows) alongside `round_freeze_end` (35,953 rows). The "inferred" name suggests a derived marker rather than a game event. Not verified.
- `bomb_defuse`: `player_x_vel`, `player_y_vel` are constant 0 (1 distinct value) and `player_z_vel` is almost all 0.

## Checked and clean

- `winner_team_code` agrees with `win_reason_code` in all 35,992 rounds (T with codes 1, 9, 13, 18; CT with 7, 8, 11, 12, 17).
- No duplicate (match, round) rows in `round_end` or `round_start`. Every match numbers rounds 1..N with no gaps. Round order agrees with tick order everywhere.
- `round_end.second` = `tick / 64` exactly. No negative ticks or seconds. `round_state` round_end tick equals `round_end` tick in every round.
- `round_freeze_end` appears once per round in 35,953 rounds; 39 rounds have none (no live start).
- Exactly one `bomb_planted` per round where present. `round_state` scores are 0 to 24 (t) and 0 to 25 (ct), never negative.
- `bomb_defused` (`round_state`) = 4,337 = `win_reason_code` 7 rounds. `bomb_exploded` (`bomb_state`) = 3,090 = 2,566 code-1 rounds + 524 code-9 rounds.

## Channel coverage

| channel | rows | matches with rows | header matches with 0 rows | median rows per match |
|---|---|---|---|---|
| round_end | 35,992 | 1,752 | 0 | 21 |
| round_start | 35,992 | 1,752 | 0 | 21 |
| round_state | 531,365 | 1,752 | 0 | 305 |
| bomb_state | 25,733 | 1,722 | 30 | 15 |
| bomb_action | 163,451 | 1,748 | 4 | 93 |
| bomb_defuse | 7,148 | 1,549 | 203 | 4 |
| round_mvp | 35,944 | 1,751 | 1 | 21 |
| score_update | 39,330 | 1,752 | 0 | 22 |

No extra match keys outside the header in any channel.

## Round-level checks

- Rounds per match (round_end): {1: 1, 3: 1, 6: 2, 7: 3, 8: 3, 9: 10, 10: 12, 11: 9, 12: 14, 13: 46, 14: 54, 15: 83, 16: 104, 17: 115, 18: 118, 19: 130, 20: 155, 21: 167, 22: 166, 23: 198, 24: 198, 28: 26, 29: 50, 30: 50, 34: 5, 35: 6, 36: 13, 40: 4, 41: 3, 42: 3, 46: 1, 48: 2}.
- Matches outside 16 to 30: 275 (238 below 16, 37 above 30).
- `round_end` duplicates: 0. `round_start` duplicates: 0. Every match has the same round count in `round_start` and `round_end`.
- Round duration (round_start to round_end): min 2.50 s, median 84.28 s, max 529.59 s; 0 negative; 249 above 200 s.
- `round_end.player_count`: 10 (32,041), 9 (1,380), 4 (1,973), 11 (217), 8 (199), 7 (68), 6 (30), 5 (15), 3 (51), 2 (17), 12 (1).
- `round_end.win_reason_code`: 1 (2,566), 7 (4,337), 8 (14,006), 9 (14,547), 11 (2), 12 (497), 13 (2), 17 (23), 18 (12). Map from `feature_validation.py`.
- `round_end.winner_team_code`: 2 = T (17,127), 3 = CT (18,865).
- `round_state` event_type counts: cs_round_start_beep 108,130; round_start 69,831; round_end 35,992; cs_round_final_beep 35,953; round_freeze_end 35,953; freezetime_ended_inferred 35,953; buytime_ended 35,244; round_poststart 34,896; cs_pre_restart 34,240; round_officially_ended 33,839; round_prestart 33,839; bomb_planted 18,306; bomb_defused 4,337; bomb_exploded 3,090; round_announce_match_point 2,545; announce_phase_end 2,166; round_announce_last_round_half 1,952; cs_win_panel_match 1,752; round_announce_match_start 1,750; begin_new_match 1,074; round_announce_final 523.
- `round_state` phase: first 302,486; halftime 6,831; postgame 3,676; second 218,372. `is_warmup` is always False.
- `bomb_state` event_type: bomb_planted 18,306; bomb_defused 4,337; bomb_exploded 3,090.
- `bomb_action` event_type: bomb_pickup 79,770; bomb_dropped 62,733; bomb_begin_plant 20,948.
- `bomb_defuse` event_type: bomb_begin_defuse 7,148 (has_kit True 4,072, False 3,076).
- `round_mvp`: rows per (match, round) 1 in 35,934 pairs, 2 in 5 pairs.
- `score_update`: rows per (match, round) 1 in 34,323 rounds, 3 in 1,669 rounds. Deltas in 1-row rounds are all +1.

## Column profiles

Columns: dtype, null %, NaN count, distinct, min, max, then the full value list when there are 12 or fewer distinct values, else the top 3 (the `values` column is cut at 160 characters). Generated by `scripts\eda\flow_profile.py`.

### round_end (rows 35,992, 8 columns)

| column | dtype | null % | NaN | distinct | min | max | values or top3 |
|---|---|---|---|---|---|---|---|
| round | Int64 | 0.000% | 0 | 48 | 1 | 48 | top3: 1=1752; 2=1751; 3=1751 |
| tick | Int64 | 0.000% | 0 | 31,195 | 1466 | 334221 | top3: 45302=6; 11427=5; 27059=5 |
| winner_team_code | Int64 | 0.000% | 0 | 2 | 2 | 3 | 2=17127; 3=18865 |
| win_reason_code | Int64 | 0.000% | 0 | 9 | 1 | 18 | 1=2566; 7=4337; 8=14006; 9=14547; 11=2; 12=497; 13=2; 17=23; 18=12 |
| win_reason_message | String | 0.000% | 0 | 9 | len 20 | len 33 | `#SFUI_Notice_*` strings (cut in output; mapped by code above) |
| player_count | Int64 | 0.000% | 0 | 11 | 2 | 12 | 2=17; 3=51; 4=1973; 5=15; 6=30; 7=68; 8=199; 9=1380; 10=32041; 11=217; 12=1 |
| second | Float64 | 0.000% | 0 | 31,195 | 22.9062 | 5222.2 | top3: 707.844=6; 178.547=5; 422.797=5 |
| match_key | String | 0.000% | 0 | 1,752 | len 57 | len 57 | - |

### round_start (rows 35,992, 4 columns)

| column | dtype | null % | NaN | distinct | min | max | values or top3 |
|---|---|---|---|---|---|---|---|
| round | Int64 | 0.000% | 0 | 48 | 1 | 48 | top3: 1=1752; 2=1751; 3=1751 |
| tick | Int64 | 0.000% | 0 | 30,487 | 66 | 325581 | top3: 66=678; 11875=5; 27507=5 |
| second | Float64 | 0.000% | 0 | 30,487 | 1.03125 | 5087.2 | top3: 1.03125=678; 185.547=5; 429.797=5 |
| match_key | String | 0.000% | 0 | 1,752 | len 57 | len 57 | - |

### round_state (rows 531,365, 11 columns)

| column | dtype | null % | NaN | distinct | min | max | values or top3 |
|---|---|---|---|---|---|---|---|
| round | Int64 | 0.000% | 0 | 48 | 1 | 48 | top3: 11=26743; 12=26495; 1=26242 |
| tick | Int64 | 0.000% | 0 | 143,607 | 66 | 334415 | top3: 1761=1023; 66=678; 1441=507 |
| event_type | String | 0.000% | 0 | 21 | len 9 | len 30 | top3: cs_round_start_beep=108130; round_start=69831; round_end=35992 |
| t_score | Int64 | 0.000% | 0 | 25 | 0 | 24 | top3: 0=59831; 2=53531; 3=46318 |
| ct_score | Int64 | 0.000% | 0 | 26 | 0 | 25 | top3: 0=57082; 2=50750; 3=48238 |
| t_score_raw | Int64 | 0.000% | 0 | 25 | 0 | 24 | top3: 0=59831; 2=53531; 3=46318 |
| ct_score_raw | Int64 | 0.000% | 0 | 26 | 0 | 25 | top3: 0=57082; 2=50750; 3=48238 |
| phase | String | 0.000% | 0 | 4 | len 5 | len 8 | first=302486; halftime=6831; postgame=3676; second=218372 |
| is_warmup | Boolean | 0.000% | 0 | 1 | - | - | False=531365 |
| second | Float64 | 0.000% | 0 | 143,607 | 1.03125 | 5225.23 | top3: 27.5156=1023; 1.03125=678; 22.5156=507 |
| match_key | String | 0.000% | 0 | 1,752 | len 57 | len 57 | - |

### bomb_state (rows 25,733, 18 columns)

| column | dtype | null % | NaN | distinct | min | max | values or top3 |
|---|---|---|---|---|---|---|---|
| round | Int64 | 0.000% | 0 | 48 | 1 | 48 | top3: 1=1397; 13=1311; 4=1258 |
| tick | Int64 | 0.000% | 0 | 22,953 | 2165 | 333516 | top3: 54399=5; 120092=5; 79015=5 |
| event_type | String | 0.000% | 0 | 3 | len 12 | len 13 | bomb_defused=4337; bomb_exploded=3090; bomb_planted=18306 |
| player_id | Int64 | 0.000% | 0 | 13 | 0 | 12 | top3: 6=2643; 2=2594; 3=2578 |
| site_code | Int64 | 0.000% | 0 | 751 | 66 | 1023 | top3: 168=759; 181=494; 182=475 |
| second | Float64 | 0.000% | 0 | 22,953 | 33.8281 | 5211.19 | top3: 849.984=5; 1876.44=5; 1234.61=5 |
| player_id_fixed | Int64 | 0.000% | 0 | 18 | 1 | 18 | top3: 4=2683; 3=2581; 2=2539 |
| player_x_pos | Float64 | 3.832% | 0 | 22,448 | -2615.24 | 3435.54 | top3: -1136.03=102; -86.0312=99; -1969.03=75 |
| player_y_pos | Float64 | 3.832% | 0 | 22,588 | -2542.93 | 3499.3 | top3: 699.969=167; 1804.03=94; 450.031=90 |
| player_z_pos | Float64 | 3.832% | 0 | 6,514 | -775.969 | 11788.2 | top3: -159.969=1890; 130.901=1338; -191.969=1285 |
| player_x_vel | Float64 | 3.832% | 0 | 1,497 | -275.844 | 264.484 | top3: 0=23180; -0.015625=4; 3.59375=3 |
| player_y_vel | Float64 | 3.832% | 0 | 1,500 | -278.934 | 263.688 | top3: 0=23175; 0.015625=5; -214.992=3 |
| player_z_vel | Float64 | 3.832% | 0 | 889 | -565 | 402.797 | top3: 0=23730; 237=5; 213=4 |
| player_phi_ang | Float64 | 3.832% | 0 | 17,159 | 1.10996 | 179 | top3: 179=35; 91.9799=24; 90=23 |
| player_theta_ang | Float64 | 3.832% | 0 | 23,698 | -179.984 | 179.986 | top3: -26.6944=4; -163.593=4; -116.487=4 |
| player_weapon_code | Int64 | 3.839% | 0 | 62 | 1 | 526 | top3: 7=11329; 4=1918; 9=1313 |
| player_team_code | Int64 | 3.836% | 0 | 2 | 2 | 3 | 2=20418; 3=4328 |
| match_key | String | 0.000% | 0 | 1,722 | len 57 | len 57 | - |

### bomb_action (rows 163,451, 20 columns)

| column | dtype | null % | NaN | distinct | min | max | values or top3 |
|---|---|---|---|---|---|---|---|
| round | Int64 | 0.000% | 0 | 48 | 1 | 48 | top3: 3=8328; 4=8304; 5=8172 |
| tick | Int64 | 0.000% | 0 | 100,254 | 564 | 334609 | top3: 2722=11; 29049=9; 9312=8 |
| event_type | String | 0.000% | 0 | 3 | len 11 | len 16 | bomb_begin_plant=20948; bomb_dropped=62733; bomb_pickup=79770 |
| player_id | Int64 | 7.971% | 0 | 19 | 0 | 19 | top3: 3=15354; 2=15303; 6=15105 |
| player_id_pawn | Int64 | 51.196% | 0 | 955 | 66 | 1023 | top3: 529=734; 537=721; 533=707 |
| site_code | Int64 | 87.184% | 0 | 747 | 66 | 1023 | top3: 168=638; 95=437; 181=427 |
| entity_id | Int64 | 61.620% | 0 | 958 | 66 | 1023 | top3: 328=184; 451=178; 462=177 |
| second | Float64 | 0.000% | 0 | 100,254 | 8.8125 | 5228.27 | top3: 42.5312=11; 453.891=9; 145.5=8 |
| player_id_fixed | Float64 | 8.009% | 0 | 19 | 1 | 19 | top3: 4=15516; 3=14969; 6=14847 |
| player_x_pos | Float64 | 8.056% | 0 | 83,626 | -3002 | 3220.46 | top3: 1216=5871; 1136=5176; -520=2406 |
| player_y_pos | Float64 | 8.056% | 0 | 83,787 | -2614.49 | 3440.26 | top3: -2288=4639; -2224=4626; -808=3047 |
| player_z_pos | Float64 | 8.056% | 0 | 39,866 | -775.969 | 11907 | top3: -415.969=8551; -63.9688=7234; -163.256=7166 |
| player_x_vel | Float64 | 8.056% | 0 | 53,726 | -194031 | 264086 | top3: 0=73462; 0.015625=92; -0.015625=85 |
| player_y_vel | Float64 | 8.056% | 0 | 53,956 | -274577 | 128157 | top3: 0=73359; -0.015625=66; 0.015625=60 |
| player_z_vel | Float64 | 8.056% | 0 | 24,510 | -22275.6 | 23046.4 | top3: 0=111465; -0.0078125=96; 0.0078125=78 |
| player_phi_ang | Float64 | 8.056% | 0 | 50,287 | 1.0001 | 179 | top3: 90=21459; 179=247; 95.28=85 |
| player_theta_ang | Float64 | 8.056% | 0 | 116,662 | -179.999 | 179.997 | top3: 90=5624; -90=1783; -178=1669 |
| player_weapon_code | Int64 | 8.057% | 0 | 63 | 1 | 526 | top3: 49=49754; 7=35475; 4=15475 |
| player_team_code | Int64 | 8.061% | 0 | 2 | 2 | 3 | 2=150275; 3=1 |
| match_key | String | 0.000% | 0 | 1,748 | len 57 | len 57 | - |

### bomb_defuse (rows 7,148, 18 columns)

| column | dtype | null % | NaN | distinct | min | max | values or top3 |
|---|---|---|---|---|---|---|---|
| round | Int64 | 0.000% | 0 | 43 | 1 | 48 | top3: 1=439; 13=434; 12=385 |
| tick | Int64 | 0.000% | 0 | 6,746 | 2996 | 333196 | top3: 93616=3; 122068=3; 8494=3 |
| event_type | String | 0.000% | 0 | 1 | len 17 | len 17 | bomb_begin_defuse=7148 |
| player_id | Int64 | 0.000% | 0 | 13 | 0 | 12 | top3: 0=782; 1=764; 3=745 |
| has_kit | Boolean | 0.000% | 0 | 2 | - | - | False=3076; True=4072 |
| second | Float64 | 0.000% | 0 | 6,746 | 46.8125 | 5206.19 | top3: 1462.75=3; 1907.31=3; 132.719=3 |
| player_id_fixed | Int64 | 0.000% | 0 | 18 | 1 | 18 | top3: 7=764; 4=733; 3=710 |
| player_x_pos | Float64 | 0.266% | 0 | 6,618 | -2386.34 | 2197.33 | top3: 811.284=12; -2027.4=11; 539.167=10 |
| player_y_pos | Float64 | 0.266% | 0 | 6,631 | -2357.46 | 3042.27 | top3: -32.8522=12; 428.322=11; 2896.84=10 |
| player_z_pos | Float64 | 0.266% | 0 | 1,780 | -771.969 | 11826 | top3: -159.969=563; -191.969=406; -179.969=393 |
| player_x_vel | Float64 | 0.266% | 0 | 1 | 0 | 0 | 0=7129 |
| player_y_vel | Float64 | 0.266% | 0 | 1 | 0 | 0 | 0=7129 |
| player_z_vel | Float64 | 0.266% | 0 | 143 | -127.723 | 0 | top3: 0=6892; -2=59; -2.00098=10 |
| player_phi_ang | Float64 | 0.266% | 0 | 6,300 | 67.0688 | 179 | top3: 179=19; 124.98=13; 114.64=10 |
| player_theta_ang | Float64 | 0.266% | 0 | 6,795 | -179.988 | 179.978 | top3: 145.346=11; 154.032=9; 95.0791=9 |
| player_weapon_code | Int64 | 0.280% | 0 | 58 | 1 | 526 | top3: 42=1515; 7=900; 60=789 |
| player_team_code | Int64 | 0.266% | 0 | 1 | 3 | 3 | 3=7129 |
| match_key | String | 0.000% | 0 | 1,549 | len 57 | len 57 | - |

### round_mvp (rows 35,944, 7 columns)

| column | dtype | null % | NaN | distinct | min | max | values or top3 |
|---|---|---|---|---|---|---|---|
| round | Int64 | 0.000% | 0 | 48 | 1 | 48 | top3: 1=1750; 2=1750; 3=1750 |
| tick | Int64 | 0.000% | 0 | 31,157 | 1466 | 334221 | top3: 45302=6; 11427=5; 27059=5 |
| player_id | Int64 | 0.000% | 0 | 14 | 0 | 19 | top3: 0=4211; 2=4101; 1=3840 |
| mvp_count | Int64 | 0.000% | 0 | 12 | 1 | 12 | 1=13941; 2=10049; 3=6105; 4=3272; 5=1567; 6=672; 7=230; 8=81; 9=18; 10=6; 11=2; 12=1 |
| second | Float64 | 0.000% | 0 | 31,157 | 22.9062 | 5222.2 | top3: 707.844=6; 178.547=5; 422.797=5 |
| player_id_fixed | Float64 | 0.003% | 0 | 18 | 1 | 18 | top3: 3=3815; 4=3753; 2=3597 |
| match_key | String | 0.000% | 0 | 1,751 | len 57 | len 57 | - |

### score_update (rows 39,330, 7 columns)

| column | dtype | null % | NaN | distinct | min | max | values or top3 |
|---|---|---|---|---|---|---|---|
| round | Int64 | 0.000% | 0 | 48 | 1 | 48 | top3: 12=4373; 8=1989; 1=1752 |
| tick | Int64 | 0.000% | 0 | 32,468 | 1466 | 334221 | top3: 74707=7; 75441=7; 64162=7 |
| team_code | Int64 | 0.000% | 0 | 2 | 2 | 3 | 2=18796; 3=20534 |
| old_score | Int64 | 0.000% | 0 | 25 | 0 | 24 | top3: 3=3588; 2=3529; 5=3523 |
| new_score | Int64 | 0.000% | 0 | 26 | 0 | 25 | top3: 5=3670; 3=3668; 4=3656 |
| second | Float64 | 0.000% | 0 | 32,468 | 22.9062 | 5222.2 | top3: 1167.3=7; 1178.77=7; 1002.53=7 |
| match_key | String | 0.000% | 0 | 1,752 | len 57 | len 57 | - |

Note on the `player_id` columns: the raw `player_id` min is 0 in the bomb and round_mvp channels, which is the sentinel in item 5. The `*_fixed` columns start at 1.
