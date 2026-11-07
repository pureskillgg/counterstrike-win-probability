# Combat channel group EDA

Generated 2026-10-12T19:03:58 from `G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds`. Channels: player_death, player_hurt, other_death, bullet_damage, player_blind, player_footstep.

Universe: 1752 matches in the header tome (`tome\csds\header`); tick_rate counts {'64': 1752}. A match counts as zero-row for a channel when its header key has no row in that channel.

## Goofy list (5 most important, ranked by how much they could mislead a model)
1. **bullet_damage is missing for 61% of matches.** 1,075 of 1,752 header matches have zero bullet_damage rows. 1,050 of them are dated 2026-10-08, the batch where only 83 of 1,133 matches have rows, and 1,060 have match_type 'unknown'. This looks like a batch extraction gap. A model reads these rounds as no shots fired unless they are filtered out.
2. **player_hurt.health_removed is raw damage with overkill, not HP lost.** 9.8% of rows are above 100, every one of them on a kill (health = 0), with a maximum of 4,999. health_removed equals previous health minus current health on only 38% of rows (72% within 1). effective_health_removed is above 100 in 174 rows (max 446) on a 0-100 HP scale. Derive HP lost from health before using it as a damage target.
3. **player_id and player_id_fixed are per-match relabels, not player identities.** Raw player_id is a 0-9 slot and player_id_fixed is a 1-10 slot, reshuffled per match. Within a match the mapping is one-to-one (99.99% of pairs), but raw 0 maps to 15 different fixed values across matches. The two disagree on 91% of player_death rows. 391 matches also carry raw ids 10-19. Any cross-match player feature built on either column is wrong.
4. **other_death has zero rows for every de_dust2 match.** 316 of the 370 zero-row other_death matches are de_dust2, and all 316 Dust2 matches are empty. The other large maps are 97%+ covered. This looks like a map-level extraction gap, so a model sees Dust2 rounds with no prop or chicken deaths.
5. **Assister fields have three encodings, and positions are null for 60% of real assists.** No assister is encoded as null (139,891 rows) or 65535 (33,399 rows), while raw 0 is a real player slot (7,865 rows). Of 80,190 rows with a real assister, 47,827 (60%) have null assister position, velocity and angle. A raw 65535 reads as a huge player id.

## Follow-up checks (scripts/eda/combat_investigate.py and combat_investigate2.py)
- Sentinel hits on tick and second (65535, 32767, 9999, 999) are real times, not sentinels. second equals tick/64 on those rows (the residual check passes in every channel), so do not drop them.
- entity_id = 999 in player_blind (80 rows) is not special: 998 has 120 rows and 1000 has 146.
- player_death z above 10,000 (3,782 rows, 32 matches) is all de_vertigo, a tall map, so it is real.
- Velocities above 3,000 units/s occur in a handful of rows per channel (about 3 to 6), with values near 200,000. These look like teleport or parse glitches. Exact -1.0 velocity values (up to 145 rows in one column) look like placeholders; unverified.
- player_hurt attacker_id_fixed is double on page 0 and int64 on page 1; concatenating the pages coerces it to float.
- player_blind player_id == attacker_id on 50,507 rows (20.7%). These could be own flashes; unverified.
- Angle rule corrected: phi_ang runs 1 to 179 (polar, inside [0,180]) and theta_ang spans -180 to 180 (azimuth). The earlier theta [-90,90] rule was wrong, so its 100k+ hits per channel are not defects.
- Zero-row matches outside other_death: player_blind 21 (19 on 2026-10-08), player_footstep 18 (17 on 2026-10-08), player_hurt 1 (de_dust2, 2026-10-07, competitive).

## Method
- Pages: each channel folder's `tome` manifest lists its pages. Page files are extensionless parquet named `dataframe_NNNNN`. Every page is read with pyarrow.parquet and concatenated, with a `_page` index added.
- Null rate, distinct count (non-null), min/max/median for numeric columns, and top-5 values for columns with 50 or fewer distinct values. Share is over all rows in the channel.
- Constant: exactly one non-null value. >50% null: null rate over 0.5. Sentinels: exact match to 4294967295, 65535, 32767, 2147483647, -1, 999, 9999 in numeric columns. Spike: one non-null value holds more than 20% of all rows (constant columns excluded).
- Impossible: `*health`/`*armor` negative or above 100; `*money` negative or above 16000; `*ammo`/`*ping` negative (matched on column name). Extra non-negative-by-name check on removed/distance/duration/tick/round/second columns.
- Identical: two columns of the same kind, null-aligned, equal in every row (all-null pairs skipped).
- Duplicates: full-row duplicates over all data columns (the within-match-and-round case is included because match_key and round are in each row). Keyed duplicates on match_key, round, tick, player_id, attacker_id, other_type, reported with all copies, since multi-hit ticks can be legitimate.
- Match outliers: row count per header match (zero rows included). Median is over all 1752 header matches, and separately over non-zero matches. Top 3 by absolute deviation from the median.
- Extra: tick order inside a match in stored order; second vs tick/tick_rate residual (tick_rate from header); angle and position/velocity ranges; empty strings; `*_fixed` vs base column disagreement; id zeros; bullet damage_dir unit norm; blind duration over 10 s; effective vs raw health removed; page-spanning matches; schema drift across pages.

## player_death

- Table: `basic_player_death.2026-09-18,2026-10-08`. Pages (manifest): 1 ['dataframe_00000']. On disk: ['dataframe_00000']. Page rows: [253480]. Sum matches total: True.
- Rows: 253,480. Distinct match_key: 1752 of 1752 header matches. Zero-row matches: 0. Match keys not in header: 0.
- Matches with rows on more than one page: 0.
- Schema drift across pages: none (single page or identical schemas).

### Columns

| column | arrow type | null % | distinct | min | max | median | top 5 (value x count, share) |
|---|---|---|---|---|---|---|---|
| round | int64 | 0.00% | 48 | 1 | 48 | 11.0 | 1 x12159 (4.8%); 12 x12104 (4.8%); 3 x12056 (4.8%); 8 x12053 (4.8%); 7 x12025 (4.7%) |
| tick | int64 | 0.00% | 125708 | 80 | 334609 | 68954.0 |  |
| player_id | int64 | 0.00% | 20 | 0 | 19 | 5.0 | 3 x25953 (10.2%); 2 x25828 (10.2%); 1 x25498 (10.1%); 7 x24986 (9.9%); 6 x24820 (9.8%) |
| attacker_id | int64 | 0.40% | 21 | 0 | 65535 | 4.0 | 2 x26258 (10.4%); 0 x26050 (10.3%); 3 x25983 (10.3%); 1 x25582 (10.1%); 4 x24700 (9.7%) |
| assister_id | int64 | 55.19% | 18 | 0 | 65535 | 7.0 | 65535 x33399 (13.2%); 2 x8322 (3.3%); 1 x8199 (3.2%); 3 x8172 (3.2%); 4 x8030 (3.2%) |
| weapon_name | string | 0.00% | 68 |  |  |  |  |
| is_headshot | bool | 0.00% | 2 |  |  |  | False x137506 (54.2%); True x115974 (45.8%) |
| penetration_amount | int64 | 0.00% | 5 | 0 | 4 | 0.0 | 0 x245369 (96.8%); 1 x7084 (2.8%); 2 x850 (0.3%); 3 x163 (0.1%); 4 x14 (0.0%) |
| has_replay | bool | 0.00% | 2 |  |  |  | False x238467 (94.1%); True x15013 (5.9%) |
| is_flash_assist | bool | 0.00% | 2 |  |  |  | False x248383 (98.0%); True x5097 (2.0%) |
| is_through_smoke | bool | 0.00% | 2 |  |  |  | False x237655 (93.8%); True x15825 (6.2%) |
| is_attacker_blind | bool | 0.00% | 2 |  |  |  | False x252031 (99.4%); True x1449 (0.6%) |
| is_noscope | bool | 0.00% | 2 |  |  |  | False x252830 (99.7%); True x650 (0.3%) |
| second | double | 0.00% | 125708 | 1.25 | 5228.265625 | 1077.40625 |  |
| player_id_fixed | double | 0.13% | 19 | 1.0 | 19.0 | 6.0 | 4.0 x25934 (10.2%); 3.0 x25302 (10.0%); 10.0 x24762 (9.8%); 9.0 x24699 (9.7%); 6.0 x24645 (9.7%) |
| attacker_id_fixed | double | 0.65% | 19 | 1.0 | 19.0 | 6.0 | 4.0 x25276 (10.0%); 3.0 x25096 (9.9%); 10.0 x24947 (9.8%); 7.0 x24916 (9.8%); 9.0 x24701 (9.7%) |
| assister_id_fixed | double | 68.36% | 19 | 1.0 | 19.0 | 6.0 | 4.0 x8120 (3.2%); 3.0 x7946 (3.1%); 7.0 x7918 (3.1%); 5.0 x7896 (3.1%); 8.0 x7835 (3.1%) |
| player_x_pos | double | 0.28% | 242534 | -2975.521728515625 | 3295.82470703125 | 73.50602722167969 |  |
| player_y_pos | double | 0.28% | 242569 | -2603.06640625 | 3513.952392578125 | 273.6955871582031 |  |
| player_z_pos | double | 0.28% | 113305 | -775.96875 | 11912.0625 | 0.36146509647369385 |  |
| player_x_vel | double | 0.28% | 82503 | -70224.123046875 | 206064.890625 | 0.0 |  |
| player_y_vel | double | 0.28% | 79687 | -205799.359375 | 32612.30322265625 | 0.0 |  |
| player_z_vel | double | 0.28% | 46032 | -4857.5859375 | 5734.4627685546875 | 0.0 |  |
| player_phi_ang | double | 0.28% | 65725 | 1.0000991821289062 | 178.99990844726562 | 92.15606689453125 |  |
| player_theta_ang | double | 0.28% | 216472 | -180.0 | 179.99966430664062 | 2.1691131591796875 |  |
| player_weapon_code | int64 | 0.28% | 64 | 1 | 526 | 9.0 |  |
| player_team_code | int64 | 0.29% | 2 | 2 | 3 | 2.0 | 2 x132137 (52.1%); 3 x120605 (47.6%) |
| attacker_x_pos | double | 0.85% | 240483 | -2991.857177734375 | 3290.67626953125 | 53.6805534362793 |  |
| attacker_y_pos | double | 0.85% | 240411 | -2603.08203125 | 3513.9658203125 | 277.999267578125 |  |
| attacker_z_pos | double | 0.85% | 109326 | -775.96875 | 11929.78125 | 0.3887767791748047 |  |
| attacker_x_vel | double | 0.85% | 61385 | -84962.234375 | 73021.71875 | 0.0 |  |
| attacker_y_vel | double | 0.85% | 59576 | -205799.359375 | 34451.2783203125 | 0.0 |  |
| attacker_z_vel | double | 0.85% | 34522 | -4729.6015625 | 5734.4627685546875 | 0.0 |  |
| attacker_phi_ang | double | 0.85% | 64734 | 1.0000991821289062 | 178.99990844726562 | 92.18215942382812 |  |
| attacker_theta_ang | double | 0.85% | 215309 | -180.0 | 179.99966430664062 | -1.38531494140625 |  |
| attacker_weapon_code | int64 | 0.85% | 64 | 1 | 526 | 9.0 |  |
| attacker_team_code | int64 | 0.86% | 3 | 0 | 3 | 3.0 | 3 x130979 (51.7%); 2 x120322 (47.5%); 0 x1 (0.0%) |
| assister_x_pos | double | 87.23% | 31190 | -2648.70263671875 | 2663.969482421875 | 93.22052764892578 |  |
| assister_y_pos | double | 87.23% | 31158 | -2597.96240234375 | 3513.84765625 | 322.6628723144531 |  |
| assister_z_pos | double | 87.23% | 16786 | -775.96875 | 11907.03125 | 1.03125 |  |
| assister_x_vel | double | 87.23% | 22085 | -295.0703125 | 277.8125 | 0.0 |  |
| assister_y_vel | double | 87.23% | 21708 | -280.578125 | 276.921875 | 0.0 |  |
| assister_z_vel | double | 87.23% | 10619 | -595.0 | 973.22265625 | 0.0 |  |
| assister_phi_ang | double | 87.23% | 18388 | 12.125473022460938 | 178.99990844726562 | 91.89755249023438 |  |
| assister_theta_ang | double | 87.23% | 30700 | -180.0 | 179.99517822265625 | 1.3959503173828125 |  |
| assister_weapon_code | int64 | 87.23% | 64 | 1 | 526 | 16.0 |  |
| assister_team_code | int64 | 87.23% | 2 | 2 | 3 | 2.0 | 2 x16720 (6.6%); 3 x15641 (6.2%) |
| match_key | string | 0.00% | 1752 |  |  |  |  |

### Flags

- **Constant column (one value):** 0. none.
- **More than 50% null:** 12. e.g. `assister_id` (55.2%); `assister_id_fixed` (68.4%)
  - `assister_id` null 55.19%
  - `assister_id_fixed` null 68.36%
  - `assister_x_pos` null 87.23%
  - `assister_y_pos` null 87.23%
  - `assister_z_pos` null 87.23%
  - `assister_x_vel` null 87.23%
  - `assister_y_vel` null 87.23%
  - `assister_z_vel` null 87.23%
  - `assister_phi_ang` null 87.23%
  - `assister_theta_ang` null 87.23%
  - `assister_weapon_code` null 87.23%
  - `assister_team_code` null 87.23%
- **Sentinel-like values:** 15 (col, value) hits. e.g. `tick`=65535 x3; `tick`=32767 x2
  - `tick` = 65535: 3 rows (0.00%); examples {page=0, match_id=m_05f46e63496cbeab6034e297, round=8, tick=65535, value=65535}; {page=0, match_id=m_0bb2ccfd0ca3ca6726ca7275, round=9, tick=65535, value=65535}
  - `tick` = 32767: 2 rows (0.00%); examples {page=0, match_id=m_4ceed6a37dd561e2028e114d, round=5, tick=32767, value=32767}; {page=0, match_id=m_131ad45b040adaeac70813ea, round=5, tick=32767, value=32767}
  - `attacker_id` = 65535: 308 rows (0.12%); examples {page=0, match_id=m_5ca9c86175a823d178dbca54, round=14, tick=99767, value=65535}; {page=0, match_id=m_4274bdbd36bd0775fbf9340c, round=19, tick=123375, value=65535}
  - `assister_id` = 65535: 33,399 rows (13.18%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=65535}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2823, value=65535}
  - `second` = 999: 3 rows (0.00%); examples {page=0, match_id=m_2e56720ee5002c7e653c36cf, round=10, tick=63936, value=999.0}; {page=0, match_id=m_a0f26ef2203f80208156e67c, round=10, tick=63936, value=999.0}
  - `player_x_vel` = -1: 18 rows (0.01%); examples {page=0, match_id=m_568e4ec420cdb80f622fa0c2, round=7, tick=39916, value=-1.0}; {page=0, match_id=m_f624e1aa1eea9da8217b632a, round=16, tick=54933, value=-1.0}
  - `player_y_vel` = -1: 28 rows (0.01%); examples {page=0, match_id=m_ac4e1d9b17bcebdd70e99a46, round=18, tick=101284, value=-1.0}; {page=0, match_id=m_67caf6ea944f5b1210e9a268, round=10, tick=57083, value=-1.0}
  - `player_z_vel` = -1: 32 rows (0.01%); examples {page=0, match_id=m_5b6b262ccc0be51c2fea88f6, round=12, tick=62665, value=-1.0}; {page=0, match_id=m_fa9c9f9254155ff7492ff199, round=4, tick=24844, value=-1.0}
  - `attacker_x_vel` = -1: 20 rows (0.01%); examples {page=0, match_id=m_e06309911bbcc6350b8f2e2a, round=11, tick=60905, value=-1.0}; {page=0, match_id=m_9f15f12fd8cecb9b99dbde88, round=3, tick=10568, value=-1.0}
  - `attacker_y_vel` = -1: 23 rows (0.01%); examples {page=0, match_id=m_5bdb2618ebdc2c58435b896c, round=9, tick=46212, value=-1.0}; {page=0, match_id=m_77067c8b36159ea06e4e521d, round=5, tick=24785, value=-1.0}
  - `attacker_z_vel` = -1: 24 rows (0.01%); examples {page=0, match_id=m_eabccb8bb8ff66d72681cc9d, round=1, tick=2599, value=-1.0}; {page=0, match_id=m_689aac85c8fdf5122d383e13, round=8, tick=42208, value=-1.0}
  - `assister_z_pos` = -1: 1 rows (0.00%); examples {page=0, match_id=m_6770a62aa0c3bfae01c8c5ec, round=9, tick=56686, value=-1.0}
  - `assister_x_vel` = -1: 2 rows (0.00%); examples {page=0, match_id=m_34b161b34c7e8ed904193baf, round=18, tick=105826, value=-1.0}; {page=0, match_id=m_b20638653d879216ced36717, round=2, tick=15780, value=-1.0}
  - `assister_y_vel` = -1: 3 rows (0.00%); examples {page=0, match_id=m_c73cc185a85ed60538bf3f69, round=12, tick=73270, value=-1.0}; {page=0, match_id=m_5eca68e615ccaf004ded8211, round=22, tick=140791, value=-1.0}
  - `assister_z_vel` = -1: 2 rows (0.00%); examples {page=0, match_id=m_60eb145913e0c6b92291b747, round=7, tick=50869, value=-1.0}; {page=0, match_id=m_3e2050f6d48aff47dc12e6d3, round=16, tick=112622, value=-1.0}
- **Spike (one value above 20% of rows):** 16. e.g. `weapon_name`=ak47 (33.6%); `is_headshot`=False (54.2%)
  - `weapon_name` = ak47: 85,059 rows (33.56%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=16155, value=ak47}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=4, tick=18843, value=ak47}
  - `is_headshot` = False: 137,506 rows (54.25%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=4215, value=False}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=6974, value=False}
  - `penetration_amount` = 0: 245,369 rows (96.80%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2766, value=0}
  - `has_replay` = False: 238,467 rows (94.08%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=False}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2766, value=False}
  - `is_flash_assist` = False: 248,383 rows (97.99%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=False}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2823, value=False}
  - `is_through_smoke` = False: 237,655 rows (93.76%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=False}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2766, value=False}
  - `is_attacker_blind` = False: 252,031 rows (99.43%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=False}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2766, value=False}
  - `is_noscope` = False: 252,830 rows (99.74%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=False}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2766, value=False}
  - `player_z_vel` = 0.0: 150,669 rows (59.44%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2766, value=0.0}
  - `player_weapon_code` = 7: 77,538 rows (30.59%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=4, tick=18997, value=7}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=4, tick=19436, value=7}
  - `player_team_code` = 2: 132,137 rows (52.13%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=2}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2823, value=2}
  - `attacker_x_vel` = 0.0: 88,411 rows (34.88%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=13150, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=13904, value=0.0}
  - `attacker_y_vel` = 0.0: 88,436 rows (34.89%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=13150, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=13904, value=0.0}
  - `attacker_z_vel` = 0.0: 176,887 rows (69.78%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2766, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2823, value=0.0}
  - `attacker_weapon_code` = 7: 86,232 rows (34.02%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=16155, value=7}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=4, tick=18843, value=7}
  - `attacker_team_code` = 3: 130,979 rows (51.67%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=3}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2823, value=3}
- **Impossible for name:** 0 rule hits. none.
- **Identical columns:** 0 pairs. none.
- **Duplicate rows within match and round (full-row):** 0 rows (0.0000%) in 0 match-round groups across 0 matches. Examples none
  - Keyed (match_key, round, tick, player_id, attacker_id, other_type where present): 0 rows sit in duplicate key groups. Examples none
- **Matches far from median row count:** median over all 1752 matches = 151, zero-row matches = 0.
  - all-matches top: `m_2ef54666039eb9f17bea34e0` rows 366 (ratio 2.4238)
  - all-matches top: `m_cfe5c030c3963b8a4e858ab0` rows 353 (ratio 2.3377)
  - all-matches top: `m_57108affc95d3927cd09f0ef` rows 338 (ratio 2.2384)
  - non-zero median = 151
  - non-zero top: `m_2ef54666039eb9f17bea34e0` rows 366 (ratio 2.4238)
  - non-zero top: `m_cfe5c030c3963b8a4e858ab0` rows 353 (ratio 2.3377)
  - non-zero top: `m_57108affc95d3927cd09f0ef` rows 338 (ratio 2.2384)

### Other checks

- Null counts: {'match_key': 0, 'round': 0, 'tick': 0, 'player_id': 0, 'attacker_id': 1012}. Round range 1..48 (48 distinct). Tick range 80..334609.
- Tick decreases inside a match (stored order): 0 rows in 0 matches. Examples none
- second vs tick/tick_rate: median residual 0.000s; rows off by >1s: 0; matches with median off by >1s: 0. Examples none
- Negative in name-nonnegative columns: none.
- Range checks run: 24. Failing: `player_z_pos` abs > 10000 3,782; `player_x_vel` abs > 3000 3; `player_y_vel` abs > 3000 3; `player_z_vel` abs > 3000 2; `attacker_z_pos` abs > 10000 3,763; `attacker_x_vel` abs > 3000 3; `attacker_y_vel` abs > 3000 3; `attacker_z_vel` abs > 3000 2; `assister_z_pos` abs > 10000 495.
- Empty strings: none.
- `_fixed` vs base: `player_id_fixed` differs from `player_id` in 230,736 rows (base null 0, fixed null 329); `attacker_id_fixed` differs from `attacker_id` in 229,847 rows (base null 1,012, fixed null 1,647); `assister_id_fixed` differs from `assister_id` in 72,949 rows (base null 139,891, fixed null 173,290).
- player_id == attacker_id rows: 3,852.
- Id zero counts: `player_id` 24,653, `attacker_id` 26,050, `assister_id` 7,865.
- assister_id encodings: null 139,891; 65535 33,399; 0 (real slot) 7,865. Valid assister rows 80,190; of these 47,827 (59.6%) have null assister position. Invalid rows with a position: 0.
- Raw player_id: matches with more than 10 distinct ids 391; rows with id outside 0-9 2,992. Raw to fixed is one-to-one within match on 99.99% of pairs; raw 0 maps to 15 distinct fixed values across matches; fixed disagrees with raw on 230,736 rows.

### Checks run

- Read all 1 page(s) listed in the channel manifest with pyarrow; on-disk dataframe files: ['dataframe_00000'].
- Matched match_key to header tome key (header has 1752 keys); zero-row matches = header keys with no channel row.
- Per-column dtype (arrow and pandas), null rate, distinct, min/max/median (numeric), top 5 (<=50 distinct).
- Constant, >50% null, sentinel hits (7 values), spike >20% of rows, impossible-for-name, identical columns, full-row and keyed duplicates, match row-count outliers vs median.
- Extra: tick order in stored order, second vs tick/tick_rate, angle/position/velocity ranges, empty strings, _fixed vs base disagreement, id zeros, damage_dir unit norm, blind duration, effective vs raw health removed, page-spanning matches, schema drift.

## player_hurt

- Table: `basic_player_hurt.2026-09-18,2026-10-08`. Pages (manifest): 2 ['dataframe_00000', 'dataframe_00001']. On disk: ['dataframe_00000', 'dataframe_00001']. Page rows: [797857, 209989]. Sum matches total: True.
- Rows: 1,007,846. Distinct match_key: 1751 of 1752 header matches. Zero-row matches: 1. Match keys not in header: 0.
- Matches with rows on more than one page: 0.
- Zero-row matches (zero/total) by group, top 5 each: match_type: competitive 1/678 | day: 2026-10-07 1/87 | map_name: de_dust2 1/316.
- Schema drift across pages: [{"page": 1, "missing_vs_page0": [], "extra_vs_page0": [], "type_changes": [{"col": "attacker_id_fixed", "page0": "double", "page": "int64"}], "same_column_order": true}]

### Columns

| column | arrow type | null % | distinct | min | max | median | top 5 (value x count, share) |
|---|---|---|---|---|---|---|---|
| round | int64 | 0.00% | 48 | 1 | 48 | 11.0 | 3 x52467 (5.2%); 15 x49101 (4.9%); 12 x48443 (4.8%); 4 x47995 (4.8%); 6 x47859 (4.7%) |
| tick | int64 | 0.00% | 180593 | 1192 | 333017 | 68885.0 |  |
| player_id | int64 | 0.00% | 18 | 0 | 19 | 4.0 | 3 x103717 (10.3%); 2 x103128 (10.2%); 1 x101523 (10.1%); 0 x99344 (9.9%); 7 x99286 (9.9%) |
| attacker_id | int64 | 0.50% | 18 | 0 | 65535 | 4.0 | 2 x104785 (10.4%); 3 x103805 (10.3%); 1 x102765 (10.2%); 0 x102195 (10.1%); 5 x97793 (9.7%) |
| health | int64 | 0.00% | 101 | 0 | 100 | 41.0 |  |
| armor | int64 | 0.00% | 101 | 0 | 100 | 89.0 |  |
| weapon_name | string | 0.27% | 40 |  |  |  | ak47 x271752 (27.0%); m4a1 x209755 (20.8%); inferno x85453 (8.5%); hegrenade x69205 (6.9%); glock x43991 (4.4%) |
| health_removed | int64 | 0.00% | 413 | 0 | 4999 | 25.0 |  |
| armor_removed | int64 | 0.00% | 40 | 0 | 39 | 3.0 | 0 x308738 (30.6%); 4 x155159 (15.4%); 3 x144719 (14.4%); 5 x111525 (11.1%); 1 x74283 (7.4%) |
| hit_box_code | int64 | 0.00% | 9 | 0 | 8 | 2.0 | 2 x379449 (37.6%); 0 x168092 (16.7%); 3 x155243 (15.4%); 1 x150253 (14.9%); 5 x71424 (7.1%) |
| second | double | 0.00% | 180593 | 18.625 | 5203.390625 | 1076.328125 |  |
| player_id_fixed | int64 | 0.00% | 19 | 1 | 19 | 6.0 | 4 x104180 (10.3%); 3 x101334 (10.1%); 10 x99329 (9.9%); 7 x98453 (9.8%); 9 x98338 (9.8%) |
| attacker_id_fixed | double | 0.62% | 19 | 1.0 | 19.0 | 6.0 | 4.0 x101850 (10.1%); 3.0 x101413 (10.1%); 7.0 x97981 (9.7%); 8.0 x97266 (9.7%); 10.0 x97103 (9.6%) |
| player_x_pos | double | 0.12% | 903538 | -2991.452392578125 | 3295.82470703125 | 114.74938201904297 |  |
| player_y_pos | double | 0.12% | 904358 | -2603.080322265625 | 3513.96875 | 266.58843994140625 |  |
| player_z_pos | double | 0.12% | 408960 | -775.96875 | 11960.5 | 1.03125 |  |
| player_x_vel | double | 0.12% | 191215 | -154283.0 | 206064.890625 | 0.0 |  |
| player_y_vel | double | 0.12% | 177823 | -132859.75390625 | 49914.125 | 0.0 |  |
| player_z_vel | double | 0.12% | 138211 | -4900.26318359375 | 14342.3984375 | 0.0 |  |
| player_phi_ang | double | 0.12% | 114165 | 1.0000991821289062 | 178.99990844726562 | 92.11212158203125 |  |
| player_theta_ang | double | 0.12% | 604772 | -180.0 | 179.99966430664062 | 4.259613037109375 |  |
| player_weapon_code | int64 | 0.14% | 64 | 1 | 526 | 10.0 |  |
| player_team_code | int64 | 0.12% | 2 | 2 | 3 | 2.0 | 2 x540076 (53.6%); 3 x466558 (46.3%) |
| attacker_x_pos | double | 1.56% | 819850 | -2991.91748046875 | 3295.82470703125 | 89.9520263671875 |  |
| attacker_y_pos | double | 1.56% | 820413 | -2603.08203125 | 3513.97119140625 | 273.960693359375 |  |
| attacker_z_pos | double | 1.56% | 371266 | -775.96875 | 11929.78125 | 0.9771041870117188 |  |
| attacker_x_vel | double | 1.56% | 158957 | -154283.0 | 186339.55126953125 | 0.0 |  |
| attacker_y_vel | double | 1.56% | 154373 | -64815.5546875 | 48723.359375 | 0.0 |  |
| attacker_z_vel | double | 1.56% | 109068 | -10745.6005859375 | 14342.3984375 | 0.0 |  |
| attacker_phi_ang | double | 1.56% | 107701 | 1.0000991821289062 | 178.99990844726562 | 92.021484375 |  |
| attacker_theta_ang | double | 1.56% | 589907 | -180.0 | 179.99966430664062 | -2.2267913818359375 |  |
| attacker_weapon_code | int64 | 1.58% | 64 | 1 | 526 | 16.0 |  |
| attacker_team_code | int64 | 1.56% | 3 | 0 | 3 | 3.0 | 3 x524967 (52.1%); 2 x467114 (46.3%); 0 x1 (0.0%) |
| effective_health_removed | int64 | 0.00% | 144 | 0 | 446 | 23.0 |  |
| match_key | string | 0.00% | 1751 |  |  |  |  |

### Flags

- **Constant column (one value):** 0. none.
- **More than 50% null:** 0. none.
- **Sentinel-like values:** 13 (col, value) hits. e.g. `tick`=65535 x7; `tick`=32767 x7
  - `tick` = 65535: 7 rows (0.00%); examples {page=0, match_id=m_c6b4c49f1bfe000d281e6010, round=9, tick=65535, value=65535}; {page=0, match_id=m_05f46e63496cbeab6034e297, round=8, tick=65535, value=65535}
  - `tick` = 32767: 7 rows (0.00%); examples {page=0, match_id=m_4ceed6a37dd561e2028e114d, round=5, tick=32767, value=32767}; {page=0, match_id=m_67caf6ea944f5b1210e9a268, round=6, tick=32767, value=32767}
  - `tick` = 9999: 8 rows (0.00%); examples {page=0, match_id=m_e17c59496fe60561e5d24211, round=2, tick=9999, value=9999}; {page=0, match_id=m_1fdfb860e9d841f3526ad740, round=4, tick=9999, value=9999}
  - `attacker_id` = 65535: 1,256 rows (0.12%); examples {page=0, match_id=m_5ca9c86175a823d178dbca54, round=9, tick=64422, value=65535}; {page=0, match_id=m_5ca9c86175a823d178dbca54, round=10, tick=69156, value=65535}
  - `second` = 999: 6 rows (0.00%); examples {page=0, match_id=m_8e4d3edd5cbe768dab81ec25, round=14, tick=63936, value=999.0}; {page=0, match_id=m_2e56720ee5002c7e653c36cf, round=10, tick=63936, value=999.0}
  - `player_z_pos` = -1: 1 rows (0.00%); examples {page=0, match_id=m_06d4b33e3c13b84ce29a3df0, round=14, tick=83014, value=-1.0}
  - `player_x_vel` = -1: 57 rows (0.01%); examples {page=0, match_id=m_3c82884428a378b3a1514cc0, round=23, tick=170369, value=-1.0}; {page=0, match_id=m_9be67b69ae156f7acd647b83, round=15, tick=80338, value=-1.0}
  - `player_y_vel` = -1: 84 rows (0.01%); examples {page=0, match_id=m_ac4e1d9b17bcebdd70e99a46, round=18, tick=101284, value=-1.0}; {page=0, match_id=m_9e2c30cb3ae79fdd487040b2, round=9, tick=49555, value=-1.0}
  - `player_z_vel` = -1: 145 rows (0.01%); examples {page=0, match_id=m_5b6b262ccc0be51c2fea88f6, round=12, tick=62665, value=-1.0}; {page=0, match_id=m_87e97dbaa73f21401e819450, round=5, tick=27731, value=-1.0}
  - `attacker_z_pos` = -1: 1 rows (0.00%); examples {page=1, match_id=m_23775e5ae577255e386e0c4a, round=11, tick=71465, value=-1.0}
  - `attacker_x_vel` = -1: 77 rows (0.01%); examples {page=0, match_id=m_4274bdbd36bd0775fbf9340c, round=4, tick=22423, value=-1.0}; {page=0, match_id=m_6c3804f646565034c528c1fd, round=1, tick=1960, value=-1.0}
  - `attacker_y_vel` = -1: 86 rows (0.01%); examples {page=0, match_id=m_6c3804f646565034c528c1fd, round=14, tick=52715, value=-1.0}; {page=0, match_id=m_41555ea221cfbeab839e6fce, round=12, tick=72983, value=-1.0}
  - `attacker_z_vel` = -1: 96 rows (0.01%); examples {page=0, match_id=m_eabccb8bb8ff66d72681cc9d, round=1, tick=2599, value=-1.0}; {page=0, match_id=m_92092559576e54d32c4b26f6, round=3, tick=17531, value=-1.0}
- **Spike (one value above 20% of rows):** 12. e.g. `health`=0 (24.8%); `weapon_name`=ak47 (27.0%)
  - `health` = 0: 249,856 rows (24.79%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2766, value=0}
  - `weapon_name` = ak47: 271,752 rows (26.96%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=8445, value=ak47}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=13117, value=ak47}
  - `armor_removed` = 0: 308,738 rows (30.63%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2622, value=0}
  - `hit_box_code` = 2: 379,449 rows (37.65%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=2}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2716, value=2}
  - `player_z_vel` = 0.0: 574,487 rows (57.00%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}
  - `player_weapon_code` = 7: 310,371 rows (30.80%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=4, tick=18959, value=7}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=4, tick=18965, value=7}
  - `player_team_code` = 2: 540,076 rows (53.59%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=2}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=2}
  - `attacker_x_vel` = 0.0: 259,002 rows (25.70%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}
  - `attacker_y_vel` = 0.0: 258,572 rows (25.66%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}
  - `attacker_z_vel` = 0.0: 643,527 rows (63.85%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}
  - `attacker_weapon_code` = 7: 318,317 rows (31.58%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=8445, value=7}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=13117, value=7}
  - `attacker_team_code` = 3: 524,967 rows (52.09%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=3}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=3}
- **Impossible for name:** 0 rule hits. none.
- **Identical columns:** 0 pairs. none.
- **Duplicate rows within match and round (full-row):** 0 rows (0.0000%) in 0 match-round groups across 0 matches. Examples none
  - Keyed (match_key, round, tick, player_id, attacker_id, other_type where present): 18,536 rows sit in duplicate key groups. Examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=14, tick=73657}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=14, tick=73657}
- **Matches far from median row count:** median over all 1752 matches = 591, zero-row matches = 1.
  - all-matches top: `m_2ef54666039eb9f17bea34e0` rows 1,466 (ratio 2.4805)
  - all-matches top: `m_cfe5c030c3963b8a4e858ab0` rows 1,374 (ratio 2.3249)
  - all-matches top: `m_6ba4651d6a05090572561463` rows 1,342 (ratio 2.2707)
  - non-zero median = 591
  - non-zero top: `m_2ef54666039eb9f17bea34e0` rows 1,466 (ratio 2.4805)
  - non-zero top: `m_cfe5c030c3963b8a4e858ab0` rows 1,374 (ratio 2.3249)
  - non-zero top: `m_6ba4651d6a05090572561463` rows 1,342 (ratio 2.2707)

### Other checks

- Null counts: {'match_key': 0, 'round': 0, 'tick': 0, 'player_id': 0, 'attacker_id': 5007}. Round range 1..48 (48 distinct). Tick range 1192..333017.
- Tick decreases inside a match (stored order): 0 rows in 0 matches. Examples none
- second vs tick/tick_rate: median residual 0.000s; rows off by >1s: 0; matches with median off by >1s: 0. Examples none
- Negative in name-nonnegative columns: none.
- Range checks run: 16. Failing: `player_z_pos` abs > 10000 15,829; `player_x_vel` abs > 3000 6; `player_y_vel` abs > 3000 6; `player_z_vel` abs > 3000 3; `attacker_z_pos` abs > 10000 15,641; `attacker_x_vel` abs > 3000 6; `attacker_y_vel` abs > 3000 6; `attacker_z_vel` abs > 3000 3.
- Empty strings: `weapon_name` 3,472.
- `_fixed` vs base: `player_id_fixed` differs from `player_id` in 918,174 rows (base null 0, fixed null 8); `attacker_id_fixed` differs from `attacker_id` in 912,606 rows (base null 5,007, fixed null 6,267).
- player_id == attacker_id rows: 7,499.
- Id zero counts: `player_id` 99,344, `attacker_id` 102,195.
- effective_health_removed > health_removed: 2,478. Examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=14, tick=73679, value=12}; {page=0, match_id=m_4ceed6a37dd561e2028e114d, round=2, tick=10780, value=15}
- health_removed vs HP lost: 99,045 rows above 100 (share 9.83%), 99,045 of them with health 0; max 4999. Equals previous health minus current health on 38.18% of 703,833 rows with a previous row (72.49% within 1). effective_health_removed above 100 in 174 rows (max 446). Examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2823, value=133}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=3798, value=127}
- Raw player_id: matches with more than 10 distinct ids 83; rows with id outside 0-9 10,121. Raw to fixed is one-to-one within match on 99.99% of pairs; raw 0 maps to 15 distinct fixed values across matches; fixed disagrees with raw on 918,174 rows.

### Checks run

- Read all 2 page(s) listed in the channel manifest with pyarrow; on-disk dataframe files: ['dataframe_00000', 'dataframe_00001'].
- Matched match_key to header tome key (header has 1752 keys); zero-row matches = header keys with no channel row.
- Per-column dtype (arrow and pandas), null rate, distinct, min/max/median (numeric), top 5 (<=50 distinct).
- Constant, >50% null, sentinel hits (7 values), spike >20% of rows, impossible-for-name, identical columns, full-row and keyed duplicates, match row-count outliers vs median.
- Extra: tick order in stored order, second vs tick/tick_rate, angle/position/velocity ranges, empty strings, _fixed vs base disagreement, id zeros, damage_dir unit norm, blind duration, effective vs raw health removed, page-spanning matches, schema drift.

## other_death

- Table: `basic_other_death.2026-09-18,2026-10-08`. Pages (manifest): 1 ['dataframe_00000']. On disk: ['dataframe_00000']. Page rows: [69629]. Sum matches total: True.
- Rows: 69,629. Distinct match_key: 1382 of 1752 header matches. Zero-row matches: 370. Match keys not in header: 0.
- Matches with rows on more than one page: 0.
- Zero-row matches (zero/total) by group, top 5 each: match_type: unknown 236/1074, competitive 134/678 | day: 2026-10-08 248/1133, 2026-10-06 22/104, 2026-10-04 20/108, 2026-10-05 19/81, 2026-10-07 18/87 | map_name: de_dust2 316/316, de_vertigo 31/32, de_overpass 6/8, de_nuke 4/165, de_ancient 4/218.
- Schema drift across pages: none (single page or identical schemas).

### Columns

| column | arrow type | null % | distinct | min | max | median | top 5 (value x count, share) |
|---|---|---|---|---|---|---|---|
| round | int64 | 0.00% | 48 | 1 | 48 | 11.0 | 8 x3524 (5.1%); 6 x3511 (5.0%); 4 x3508 (5.0%); 7 x3493 (5.0%); 11 x3469 (5.0%) |
| tick | int64 | 0.00% | 53601 | 1145 | 333958 | 69088.0 |  |
| other_type | string | 0.00% | 6 |  |  |  | prop_dynamic x42186 (60.6%); prop_physics_multiplayer x21963 (31.5%); chicken x3299 (4.7%); prop_door_rotating x1454 (2.1%); func_breakable x726 (1.0%) |
| attacker_id | int64 | 0.00% | 13 | 0 | 12 | 4.0 | 3 x7259 (10.4%); 0 x7212 (10.4%); 1 x7208 (10.4%); 2 x7030 (10.1%); 7 x6963 (10.0%) |
| weapon_name | string | 0.00% | 66 |  |  |  |  |
| is_headshot | bool | 0.00% | 1 |  |  |  | False x69629 (100.0%) |
| penetration_amount | int64 | 0.00% | 5 | 0 | 4 | 0.0 | 0 x67356 (96.7%); 1 x2004 (2.9%); 2 x230 (0.3%); 3 x30 (0.0%); 4 x9 (0.0%) |
| is_through_smoke | bool | 0.00% | 1 |  |  |  | False x69629 (100.0%) |
| is_attacker_blind | bool | 0.00% | 2 |  |  |  | False x68945 (99.0%); True x684 (1.0%) |
| is_noscope | bool | 0.00% | 2 |  |  |  | False x69574 (99.9%); True x55 (0.1%) |
| second | double | 0.00% | 53601 | 17.890625 | 5218.09375 | 1079.5 |  |
| attacker_id_fixed | int64 | 0.00% | 19 | 1 | 19 | 6.0 | 4 x7069 (10.2%); 7 x6871 (9.9%); 3 x6867 (9.9%); 5 x6816 (9.8%); 6 x6812 (9.8%) |
| attacker_x_pos | double | 0.83% | 63650 | -2655.93994140625 | 2626.086181640625 | -175.3248291015625 |  |
| attacker_y_pos | double | 0.83% | 63912 | -2570.503662109375 | 3513.90966796875 | -288.89990234375 |  |
| attacker_z_pos | double | 0.83% | 29732 | -775.96875 | 11776.03125 | -113.640625 |  |
| attacker_x_vel | double | 0.83% | 41966 | -222310.28125 | 305.85968017578125 | 2.5546875 |  |
| attacker_y_vel | double | 0.83% | 40415 | -171103.8515625 | 269.546875 | 16.69921875 |  |
| attacker_z_vel | double | 0.83% | 17205 | -17128.6015625 | 1025.03125 | 0.0 |  |
| attacker_phi_ang | double | 0.83% | 33456 | 1.0000991821289062 | 178.99990844726562 | 90.61764526367188 |  |
| attacker_theta_ang | double | 0.83% | 61273 | -179.99966430664062 | 179.98110961914062 | 23.64532470703125 |  |
| attacker_weapon_code | int64 | 0.83% | 64 | 1 | 526 | 42.0 |  |
| attacker_team_code | int64 | 0.83% | 2 | 2 | 3 | 3.0 | 3 x39047 (56.1%); 2 x30001 (43.1%) |
| match_key | string | 0.00% | 1382 |  |  |  |  |

### Flags

- **Constant column (one value):** 2. e.g. `is_headshot` = False; `is_through_smoke` = False
  - `is_headshot` = False (null rate 0.00%)
  - `is_through_smoke` = False (null rate 0.00%)
- **More than 50% null:** 0. none.
- **Sentinel-like values:** 4 (col, value) hits. e.g. `tick`=32767 x2; `tick`=9999 x1
  - `tick` = 32767: 2 rows (0.00%); examples {page=0, match_id=m_b6bb8dd43ecea83624b0caae, round=6, tick=32767, value=32767}; {page=0, match_id=m_b6bb8dd43ecea83624b0caae, round=6, tick=32767, value=32767}
  - `tick` = 9999: 1 rows (0.00%); examples {page=0, match_id=m_8463b9e1ace2ede83eb98037, round=2, tick=9999, value=9999}
  - `attacker_y_vel` = -1: 3 rows (0.00%); examples {page=0, match_id=m_de3cb89881b9df3b508b1fa5, round=3, tick=14095, value=-1.0}; {page=0, match_id=m_624944b486150a95b0178724, round=15, tick=93241, value=-1.0}
  - `attacker_z_vel` = -1: 28 rows (0.04%); examples {page=0, match_id=m_e69056ef4d0f78cf30f8d2eb, round=10, tick=63052, value=-1.0}; {page=0, match_id=m_5fdc11b6e7c0e487f55e3ab6, round=20, tick=129802, value=-1.0}
- **Spike (one value above 20% of rows):** 8. e.g. `other_type`=prop_dynamic (60.6%); `weapon_name`=hegrenade (28.4%)
  - `other_type` = prop_dynamic: 42,186 rows (60.59%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=4171, value=prop_dynamic}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=7431, value=prop_dynamic}
  - `weapon_name` = hegrenade: 19,788 rows (28.42%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=7611, value=hegrenade}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=10836, value=hegrenade}
  - `penetration_amount` = 0: 67,356 rows (96.74%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=4171, value=0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=7431, value=0}
  - `is_attacker_blind` = False: 68,945 rows (99.02%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=4171, value=False}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=7431, value=False}
  - `is_noscope` = False: 69,574 rows (99.92%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=4171, value=False}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=7431, value=False}
  - `attacker_z_vel` = 0.0: 34,042 rows (48.89%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=4171, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=7611, value=0.0}
  - `attacker_weapon_code` = 7: 14,643 rows (21.03%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=14542, value=7}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=5, tick=23234, value=7}
  - `attacker_team_code` = 3: 39,047 rows (56.08%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=8438, value=3}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=10412, value=3}
- **Impossible for name:** 0 rule hits. none.
- **Identical columns:** 1 pairs. e.g. `is_headshot` = `is_through_smoke`
  - `is_headshot` == `is_through_smoke` (bool, both constant)
- **Duplicate rows within match and round (full-row):** 0 rows (0.0000%) in 0 match-round groups across 0 matches. Examples none
  - Keyed (match_key, round, tick, player_id, attacker_id, other_type where present): 169 rows sit in duplicate key groups. Examples {page=0, match_id=m_e648fa292dad6dc16d1e7461, round=8, tick=50658}; {page=0, match_id=m_e648fa292dad6dc16d1e7461, round=8, tick=50658}
- **Matches far from median row count:** median over all 1752 matches = 22, zero-row matches = 370.
  - all-matches top: `m_046a770d794b8c6a8bd42261` rows 291 (ratio 13.2273)
  - all-matches top: `m_6ba4651d6a05090572561463` rows 235 (ratio 10.6818)
  - all-matches top: `m_e51f60d0cd8e7d65cb51d7b2` rows 231 (ratio 10.5)
  - non-zero median = 34
  - non-zero top: `m_046a770d794b8c6a8bd42261` rows 291 (ratio 8.5588)
  - non-zero top: `m_6ba4651d6a05090572561463` rows 235 (ratio 6.9118)
  - non-zero top: `m_e51f60d0cd8e7d65cb51d7b2` rows 231 (ratio 6.7941)

### Other checks

- Null counts: {'match_key': 0, 'round': 0, 'tick': 0, 'attacker_id': 0}. Round range 1..48 (48 distinct). Tick range 1145..333958.
- Tick decreases inside a match (stored order): 0 rows in 0 matches. Examples none
- second vs tick/tick_rate: median residual 0.000s; rows off by >1s: 0; matches with median off by >1s: 0. Examples none
- Negative in name-nonnegative columns: none.
- Range checks run: 8. Failing: `attacker_z_pos` abs > 10000 1; `attacker_x_vel` abs > 3000 2; `attacker_y_vel` abs > 3000 2; `attacker_z_vel` abs > 3000 1.
- Empty strings: none.
- `_fixed` vs base: `attacker_id_fixed` differs from `attacker_id` in 63,260 rows (base null 0, fixed null 0).
- Id zero counts: `attacker_id` 7,212.

### Checks run

- Read all 1 page(s) listed in the channel manifest with pyarrow; on-disk dataframe files: ['dataframe_00000'].
- Matched match_key to header tome key (header has 1752 keys); zero-row matches = header keys with no channel row.
- Per-column dtype (arrow and pandas), null rate, distinct, min/max/median (numeric), top 5 (<=50 distinct).
- Constant, >50% null, sentinel hits (7 values), spike >20% of rows, impossible-for-name, identical columns, full-row and keyed duplicates, match row-count outliers vs median.
- Extra: tick order in stored order, second vs tick/tick_rate, angle/position/velocity ranges, empty strings, _fixed vs base disagreement, id zeros, damage_dir unit norm, blind duration, effective vs raw health removed, page-spanning matches, schema drift.

## bullet_damage

- Table: `basic_bullet_damage.2026-09-18,2026-10-08`. Pages (manifest): 1 ['dataframe_00000']. On disk: ['dataframe_00000']. Page rows: [289611]. Sum matches total: True.
- Rows: 289,611. Distinct match_key: 677 of 1752 header matches. Zero-row matches: 1075. Match keys not in header: 0.
- Matches with rows on more than one page: 0.
- Zero-row matches (zero/total) by group, top 5 each: match_type: unknown 1060/1074, competitive 15/678 | day: 2026-10-08 1050/1133, 2026-10-03 10/97, 2026-10-07 6/87, 2026-10-02 4/68, 2026-10-01 2/70 | map_name: de_mirage 237/346, de_dust2 201/316, de_ancient 171/218, de_anubis 148/207, de_inferno 136/290.
- Schema drift across pages: none (single page or identical schemas).

### Columns

| column | arrow type | null % | distinct | min | max | median | top 5 (value x count, share) |
|---|---|---|---|---|---|---|---|
| round | int64 | 0.00% | 30 | 1 | 30 | 10.0 | 3 x16741 (5.8%); 2 x16537 (5.7%); 4 x15471 (5.3%); 14 x15464 (5.3%); 5 x14912 (5.1%) |
| tick | int64 | 0.00% | 118236 | 1192 | 211523 | 55553.0 |  |
| player_id | int64 | 0.00% | 10 | 0 | 9 | 4.0 | 3 x32024 (11.1%); 2 x31582 (10.9%); 1 x31471 (10.9%); 0 x31448 (10.9%); 9 x27442 (9.5%) |
| attacker_id | int64 | 0.00% | 10 | 0 | 9 | 4.0 | 0 x34050 (11.8%); 2 x32513 (11.2%); 1 x32436 (11.2%); 3 x31856 (11.0%); 6 x27452 (9.5%) |
| distance | double | 0.00% | 281616 | 0.22430428862571716 | 3089.86865234375 | 602.9320068359375 |  |
| damage_dir_x | double | 0.00% | 280342 | -0.999999463558197 | 0.9999996423721313 | -0.0032725029159337282 |  |
| damage_dir_y | double | 0.00% | 280384 | -0.9999997615814209 | 0.9999997615814209 | -0.009228970855474472 |  |
| damage_dir_z | double | 0.00% | 282616 | -0.9998210668563843 | 0.9984552264213562 | -0.016853639855980873 |  |
| num_penetrations | int64 | 0.00% | 5 | 0 | 4 | 0.0 | 0 x281220 (97.1%); 1 x7296 (2.5%); 2 x890 (0.3%); 3 x173 (0.1%); 4 x32 (0.0%) |
| is_no_scope | bool | 0.00% | 2 |  |  |  | False x289116 (99.8%); True x495 (0.2%) |
| is_attacker_in_air | bool | 0.00% | 2 |  |  |  | False x288190 (99.5%); True x1421 (0.5%) |
| second | double | 0.00% | 118236 | 18.625 | 3305.046875 | 868.015625 |  |
| player_id_fixed | int64 | 0.00% | 12 | 1 | 13 | 5.0 | 3 x32582 (11.3%); 4 x32261 (11.1%); 1 x31864 (11.0%); 2 x31703 (10.9%); 7 x26956 (9.3%) |
| attacker_id_fixed | int64 | 0.00% | 12 | 1 | 12 | 5.0 | 1 x34166 (11.8%); 3 x32275 (11.1%); 2 x32136 (11.1%); 4 x31786 (11.0%); 8 x26946 (9.3%) |
| player_x_pos | double | 0.00% | 254674 | -2991.452392578125 | 3220.4580078125 | 168.53079223632812 |  |
| player_y_pos | double | 0.00% | 254918 | -2598.3779296875 | 3513.967529296875 | 217.52452087402344 |  |
| player_z_pos | double | 0.00% | 115151 | -775.96875 | 11960.5 | 0.24834203720092773 |  |
| player_x_vel | double | 0.00% | 91559 | -112790.625 | 206064.890625 | 0.0 |  |
| player_y_vel | double | 0.00% | 87554 | -132859.75390625 | 49914.125 | 0.0 |  |
| player_z_vel | double | 0.00% | 49206 | -4900.26318359375 | 2340.15234375 | 0.0 |  |
| player_phi_ang | double | 0.00% | 69722 | 1.0000991821289062 | 178.99990844726562 | 92.40119934082031 |  |
| player_theta_ang | double | 0.00% | 232386 | -180.0 | 179.99966430664062 | 1.996429443359375 |  |
| player_weapon_code | int64 | 0.00% | 64 | 1 | 526 | 13.0 |  |
| player_team_code | int64 | 0.00% | 2 | 2 | 3 | 2.0 | 2 x151248 (52.2%); 3 x138363 (47.8%) |
| attacker_x_pos | double | 0.00% | 234550 | -2991.91748046875 | 3169.248291015625 | 159.01821899414062 |  |
| attacker_y_pos | double | 0.00% | 234641 | -2597.493408203125 | 3513.97119140625 | 224.335693359375 |  |
| attacker_z_pos | double | 0.00% | 104477 | -775.96875 | 11928.78125 | 0.17299270629882812 |  |
| attacker_x_vel | double | 0.00% | 74622 | -98614.84375 | 186339.55126953125 | 0.0 |  |
| attacker_y_vel | double | 0.00% | 71784 | -63315.7890625 | 48723.359375 | 0.0 |  |
| attacker_z_vel | double | 0.00% | 40504 | -10745.6005859375 | 1016.0 | 0.0 |  |
| attacker_phi_ang | double | 0.00% | 67437 | 3.2924652099609375 | 178.99990844726562 | 92.21890258789062 |  |
| attacker_theta_ang | double | 0.00% | 231142 | -180.0 | 179.99966430664062 | -1.061553955078125 |  |
| attacker_weapon_code | int64 | 0.00% | 44 | 1 | 526 | 16.0 | 7 x78512 (27.1%); 60 x44666 (15.4%); 16 x20188 (7.0%); 4 x16895 (5.8%); 61 x15532 (5.4%) |
| attacker_team_code | int64 | 0.00% | 2 | 2 | 3 | 3.0 | 3 x150107 (51.8%); 2 x139504 (48.2%) |
| match_key | string | 0.00% | 677 |  |  |  |  |

### Flags

- **Constant column (one value):** 0. none.
- **More than 50% null:** 0. none.
- **Sentinel-like values:** 11 (col, value) hits. e.g. `tick`=65535 x1; `tick`=32767 x3
  - `tick` = 65535: 1 rows (0.00%); examples {page=0, match_id=m_62bce4c1193e204227f2c837, round=12, tick=65535, value=65535}
  - `tick` = 32767: 3 rows (0.00%); examples {page=0, match_id=m_4ceed6a37dd561e2028e114d, round=5, tick=32767, value=32767}; {page=0, match_id=m_67caf6ea944f5b1210e9a268, round=6, tick=32767, value=32767}
  - `tick` = 9999: 4 rows (0.00%); examples {page=0, match_id=m_e17c59496fe60561e5d24211, round=2, tick=9999, value=9999}; {page=0, match_id=m_1fdfb860e9d841f3526ad740, round=4, tick=9999, value=9999}
  - `second` = 999: 4 rows (0.00%); examples {page=0, match_id=m_8e4d3edd5cbe768dab81ec25, round=14, tick=63936, value=999.0}; {page=0, match_id=m_2e56720ee5002c7e653c36cf, round=10, tick=63936, value=999.0}
  - `player_z_pos` = -1: 1 rows (0.00%); examples {page=0, match_id=m_06d4b33e3c13b84ce29a3df0, round=14, tick=83014, value=-1.0}
  - `player_x_vel` = -1: 13 rows (0.00%); examples {page=0, match_id=m_3c82884428a378b3a1514cc0, round=23, tick=170369, value=-1.0}; {page=0, match_id=m_9be67b69ae156f7acd647b83, round=15, tick=80338, value=-1.0}
  - `player_y_vel` = -1: 21 rows (0.01%); examples {page=0, match_id=m_ac4e1d9b17bcebdd70e99a46, round=18, tick=101284, value=-1.0}; {page=0, match_id=m_9e2c30cb3ae79fdd487040b2, round=9, tick=49555, value=-1.0}
  - `player_z_vel` = -1: 33 rows (0.01%); examples {page=0, match_id=m_5b6b262ccc0be51c2fea88f6, round=12, tick=62665, value=-1.0}; {page=0, match_id=m_ca5b6b4fbc3b9d7a89af8665, round=12, tick=55336, value=-1.0}
  - `attacker_x_vel` = -1: 21 rows (0.01%); examples {page=0, match_id=m_4274bdbd36bd0775fbf9340c, round=4, tick=22423, value=-1.0}; {page=0, match_id=m_6c3804f646565034c528c1fd, round=1, tick=1960, value=-1.0}
  - `attacker_y_vel` = -1: 15 rows (0.01%); examples {page=0, match_id=m_6c3804f646565034c528c1fd, round=14, tick=52715, value=-1.0}; {page=0, match_id=m_41555ea221cfbeab839e6fce, round=12, tick=72983, value=-1.0}
  - `attacker_z_vel` = -1: 21 rows (0.01%); examples {page=0, match_id=m_eabccb8bb8ff66d72681cc9d, round=1, tick=2599, value=-1.0}; {page=0, match_id=m_92092559576e54d32c4b26f6, round=3, tick=17531, value=-1.0}
- **Spike (one value above 20% of rows):** 11. e.g. `num_penetrations`=0 (97.1%); `is_no_scope`=False (99.8%)
  - `num_penetrations` = 0: 281,220 rows (97.10%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0}
  - `is_no_scope` = False: 289,116 rows (99.83%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=False}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=False}
  - `is_attacker_in_air` = False: 288,190 rows (99.51%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=False}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=False}
  - `player_z_vel` = 0.0: 179,512 rows (61.98%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}
  - `player_weapon_code` = 7: 77,239 rows (26.67%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=4, tick=18959, value=7}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=4, tick=18965, value=7}
  - `player_team_code` = 2: 151,248 rows (52.22%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=2}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=2}
  - `attacker_x_vel` = 0.0: 96,320 rows (33.26%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}
  - `attacker_y_vel` = 0.0: 96,242 rows (33.23%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}
  - `attacker_z_vel` = 0.0: 203,235 rows (70.18%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=0.0}
  - `attacker_weapon_code` = 7: 78,512 rows (27.11%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=8445, value=7}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=13117, value=7}
  - `attacker_team_code` = 3: 150,107 rows (51.83%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=3}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2504, value=3}
- **Impossible for name:** 0 rule hits. none.
- **Identical columns:** 0 pairs. none.
- **Duplicate rows within match and round (full-row):** 0 rows (0.0000%) in 0 match-round groups across 0 matches. Examples none
  - Keyed (match_key, round, tick, player_id, attacker_id, other_type where present): 12,214 rows sit in duplicate key groups. Examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=14, tick=73657}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=14, tick=73657}
- **Matches far from median row count:** median over all 1752 matches = 0, zero-row matches = 1075.
  - all-matches top: `m_bb51389530ef935a6000b012` rows 907 (ratio None)
  - all-matches top: `m_8547d22529cf1353f6eb94b8` rows 878 (ratio None)
  - all-matches top: `m_bcd1a068ee5cc1dc08c3d736` rows 874 (ratio None)
  - non-zero median = 468
  - non-zero top: `m_bb51389530ef935a6000b012` rows 907 (ratio 1.938)
  - non-zero top: `m_cc3cfd5a9bdf9133673c83d7` rows 37 (ratio 0.0791)
  - non-zero top: `m_97f223f6628fecc8c0c70974` rows 45 (ratio 0.0962)

### Other checks

- Null counts: {'match_key': 0, 'round': 0, 'tick': 0, 'player_id': 0, 'attacker_id': 0}. Round range 1..30 (30 distinct). Tick range 1192..211523.
- Tick decreases inside a match (stored order): 0 rows in 0 matches. Examples none
- second vs tick/tick_rate: median residual 0.000s; rows off by >1s: 0; matches with median off by >1s: 0. Examples none
- Negative in name-nonnegative columns: none.
- Range checks run: 16. Failing: `player_z_pos` abs > 10000 3,999; `player_x_vel` abs > 3000 5; `player_y_vel` abs > 3000 5; `player_z_vel` abs > 3000 2; `attacker_z_pos` abs > 10000 3,999; `attacker_x_vel` abs > 3000 5; `attacker_y_vel` abs > 3000 5; `attacker_z_vel` abs > 3000 2.
- Empty strings: none.
- `_fixed` vs base: `player_id_fixed` differs from `player_id` in 261,555 rows (base null 0, fixed null 0); `attacker_id_fixed` differs from `attacker_id` in 262,660 rows (base null 0, fixed null 0).
- player_id == attacker_id rows: 0.
- Id zero counts: `player_id` 31,448, `attacker_id` 34,050.
- damage_dir norm: median 1.0000; zero vectors 0; not unit (>0.01 off) 0. Examples none
- Raw player_id: matches with more than 10 distinct ids 0; rows with id outside 0-9 0. Raw to fixed is one-to-one within match on 99.97% of pairs; raw 0 maps to 11 distinct fixed values across matches; fixed disagrees with raw on 261,555 rows.

### Checks run

- Read all 1 page(s) listed in the channel manifest with pyarrow; on-disk dataframe files: ['dataframe_00000'].
- Matched match_key to header tome key (header has 1752 keys); zero-row matches = header keys with no channel row.
- Per-column dtype (arrow and pandas), null rate, distinct, min/max/median (numeric), top 5 (<=50 distinct).
- Constant, >50% null, sentinel hits (7 values), spike >20% of rows, impossible-for-name, identical columns, full-row and keyed duplicates, match row-count outliers vs median.
- Extra: tick order in stored order, second vs tick/tick_rate, angle/position/velocity ranges, empty strings, _fixed vs base disagreement, id zeros, damage_dir unit norm, blind duration, effective vs raw health removed, page-spanning matches, schema drift.

## player_blind

- Table: `basic_player_blind.2026-09-18,2026-10-08`. Pages (manifest): 1 ['dataframe_00000']. On disk: ['dataframe_00000']. Page rows: [244469]. Sum matches total: True.
- Rows: 244,469. Distinct match_key: 1731 of 1752 header matches. Zero-row matches: 21. Match keys not in header: 0.
- Matches with rows on more than one page: 0.
- Zero-row matches (zero/total) by group, top 5 each: match_type: unknown 17/1074, competitive 4/678 | day: 2026-10-08 19/1133, 2026-10-01 1/70, 2026-10-07 1/87 | map_name: de_nuke 5/165, de_inferno 3/290, de_dust2 3/316, de_mirage 3/346, de_anubis 3/207.
- Schema drift across pages: none (single page or identical schemas).

### Columns

| column | arrow type | null % | distinct | min | max | median | top 5 (value x count, share) |
|---|---|---|---|---|---|---|---|
| round | int64 | 0.00% | 48 | 1 | 48 | 11.0 | 12 x14156 (5.8%); 8 x12982 (5.3%); 10 x12767 (5.2%); 9 x12659 (5.2%); 6 x12571 (5.1%) |
| tick | int64 | 0.00% | 79716 | 1310 | 331597 | 70342.0 |  |
| player_id | int64 | 0.00% | 17 | 0 | 19 | 5.0 | 3 x24741 (10.1%); 1 x24675 (10.1%); 2 x24567 (10.0%); 9 x24161 (9.9%); 7 x24112 (9.9%) |
| entity_id | int64 | 0.00% | 958 | 66 | 1023 | 367.0 |  |
| attacker_id | int64 | 0.00% | 14 | 0 | 14 | 5.0 | 2 x24639 (10.1%); 1 x24636 (10.1%); 0 x24512 (10.0%); 9 x24486 (10.0%); 8 x24437 (10.0%) |
| blind_duration | double | 0.00% | 233860 | 0.002139551332220435 | 5.3316121101379395 | 1.66877019405365 |  |
| second | double | 0.00% | 79716 | 20.46875 | 5181.203125 | 1099.09375 |  |
| player_id_fixed | double | 0.00% | 19 | 1.0 | 19.0 | 6.0 | 10.0 x24602 (10.1%); 7.0 x24551 (10.0%); 4.0 x24542 (10.0%); 9.0 x24305 (9.9%); 5.0 x24209 (9.9%) |
| attacker_id_fixed | int64 | 0.00% | 19 | 1 | 19 | 6.0 | 6 x24837 (10.2%); 4 x24592 (10.1%); 8 x24138 (9.9%); 7 x24054 (9.8%); 3 x23958 (9.8%) |
| player_x_pos | double | 0.00% | 231285 | -2991.081298828125 | 3295.82470703125 | 179.8576202392578 |  |
| player_y_pos | double | 0.00% | 231527 | -2603.044921875 | 3513.937255859375 | 230.98876953125 |  |
| player_z_pos | double | 0.00% | 132527 | -775.96875 | 11962.0625 | 1.03125 |  |
| player_x_vel | double | 0.00% | 109725 | -385.984375 | 349.20703125 | 0.0 |  |
| player_y_vel | double | 0.00% | 104171 | -315.984375 | 294.7265625 | 0.0 |  |
| player_z_vel | double | 0.00% | 67561 | -1138.9375 | 1146.40625 | 0.0 |  |
| player_phi_ang | double | 0.00% | 63051 | 1.0000991821289062 | 178.99990844726562 | 91.15939331054688 |  |
| player_theta_ang | double | 0.00% | 207457 | -180.0 | 179.99896240234375 | 0.426055908203125 |  |
| player_weapon_code | int64 | 0.01% | 64 | 1 | 526 | 16.0 |  |
| player_team_code | int64 | 0.00% | 2 | 2 | 3 | 2.0 | 2 x130797 (53.5%); 3 x113672 (46.5%) |
| attacker_x_pos | double | 0.94% | 108029 | -2650.37255859375 | 3288.689208984375 | 203.3920135498047 |  |
| attacker_y_pos | double | 0.94% | 107873 | -2599.52734375 | 3501.67041015625 | 120.5883674621582 |  |
| attacker_z_pos | double | 0.94% | 64217 | -775.96875 | 11929.6875 | 0.6164757013320923 |  |
| attacker_x_vel | double | 0.94% | 69175 | -282.640625 | 349.20703125 | 0.0 |  |
| attacker_y_vel | double | 0.94% | 66641 | -285.421875 | 280.140625 | 0.0 |  |
| attacker_z_vel | double | 0.94% | 37085 | -1138.9375 | 1097.908203125 | 0.0 |  |
| attacker_phi_ang | double | 0.94% | 44778 | 1.0000991821289062 | 178.99990844726562 | 91.04507446289062 |  |
| attacker_theta_ang | double | 0.94% | 103101 | -179.9993133544922 | 179.99896240234375 | 17.817420959472656 |  |
| attacker_weapon_code | int64 | 0.94% | 64 | 1 | 526 | 34.0 |  |
| attacker_team_code | int64 | 0.94% | 2 | 2 | 3 | 2.0 | 2 x138636 (56.7%); 3 x103532 (42.3%) |
| match_key | string | 0.00% | 1731 |  |  |  |  |

### Flags

- **Constant column (one value):** 0. none.
- **More than 50% null:** 0. none.
- **Sentinel-like values:** 11 (col, value) hits. e.g. `tick`=65535 x2; `tick`=32767 x4
  - `tick` = 65535: 2 rows (0.00%); examples {page=0, match_id=m_ac3b608a8c4a9b232e7c4d3f, round=10, tick=65535, value=65535}; {page=0, match_id=m_ac3b608a8c4a9b232e7c4d3f, round=10, tick=65535, value=65535}
  - `tick` = 32767: 4 rows (0.00%); examples {page=0, match_id=m_b7ca8891ae4a8eb198e9c99e, round=6, tick=32767, value=32767}; {page=0, match_id=m_b7ca8891ae4a8eb198e9c99e, round=6, tick=32767, value=32767}
  - `tick` = 9999: 5 rows (0.00%); examples {page=0, match_id=m_12c1a0f503f07e7cadc4822a, round=2, tick=9999, value=9999}; {page=0, match_id=m_12c1a0f503f07e7cadc4822a, round=2, tick=9999, value=9999}
  - `entity_id` = 999: 80 rows (0.03%); examples {page=0, match_id=m_cc66b3bc82e7bd795fafd34e, round=12, tick=66477, value=999}; {page=0, match_id=m_110a923942858dbf84fad6e7, round=22, tick=117039, value=999}
  - `second` = 999: 4 rows (0.00%); examples {page=0, match_id=m_dbd9097c2d870656376e2405, round=9, tick=63936, value=999.0}; {page=0, match_id=m_dbd9097c2d870656376e2405, round=9, tick=63936, value=999.0}
  - `player_x_vel` = -1: 6 rows (0.00%); examples {page=0, match_id=m_689aac85c8fdf5122d383e13, round=1, tick=2403, value=-1.0}; {page=0, match_id=m_8463b9e1ace2ede83eb98037, round=12, tick=65915, value=-1.0}
  - `player_y_vel` = -1: 15 rows (0.01%); examples {page=0, match_id=m_568e4ec420cdb80f622fa0c2, round=21, tick=140707, value=-1.0}; {page=0, match_id=m_9c911577747adecd0a359e14, round=3, tick=21268, value=-1.0}
  - `player_z_vel` = -1: 48 rows (0.02%); examples {page=0, match_id=m_6d1c472562c712004cba52fe, round=17, tick=101552, value=-1.0}; {page=0, match_id=m_003cfaa70ccee322f2208464, round=6, tick=26152, value=-1.0}
  - `attacker_z_pos` = -1: 3 rows (0.00%); examples {page=0, match_id=m_b8d661bc3e339a633c34d483, round=16, tick=90804, value=-1.0}; {page=0, match_id=m_b8d661bc3e339a633c34d483, round=16, tick=90804, value=-1.0}
  - `attacker_y_vel` = -1: 18 rows (0.01%); examples {page=0, match_id=m_9e20d587a0c0422790b84bb5, round=22, tick=133251, value=-1.0}; {page=0, match_id=m_9e20d587a0c0422790b84bb5, round=22, tick=133251, value=-1.0}
  - `attacker_z_vel` = -1: 57 rows (0.02%); examples {page=0, match_id=m_fc6caa282e61e4502d8fe521, round=16, tick=98764, value=-1.0}; {page=0, match_id=m_fc6caa282e61e4502d8fe521, round=16, tick=98764, value=-1.0}
- **Spike (one value above 20% of rows):** 6. e.g. `player_z_vel`=0.0 (45.3%); `player_weapon_code`=7 (27.5%)
  - `player_z_vel` = 0.0: 110,795 rows (45.32%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2451, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2451, value=0.0}
  - `player_weapon_code` = 7: 67,268 rows (27.52%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=2, tick=7166, value=7}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=10958, value=7}
  - `player_team_code` = 2: 130,797 rows (53.50%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2569, value=2}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2569, value=2}
  - `attacker_z_vel` = 0.0: 102,129 rows (41.78%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2451, value=0.0}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2451, value=0.0}
  - `attacker_weapon_code` = 7: 61,404 rows (25.12%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=3, tick=10958, value=7}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=4, tick=18666, value=7}
  - `attacker_team_code` = 2: 138,636 rows (56.71%); examples {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2451, value=2}; {page=0, match_id=m_a661aadb0b4b87e35cdec94c, round=1, tick=2451, value=2}
- **Impossible for name:** 0 rule hits. none.
- **Identical columns:** 0 pairs. none.
- **Duplicate rows within match and round (full-row):** 0 rows (0.0000%) in 0 match-round groups across 0 matches. Examples none
  - Keyed (match_key, round, tick, player_id, attacker_id, other_type where present): 2 rows sit in duplicate key groups. Examples {page=0, match_id=m_ff5c5ba15dea3b18d2fe8299, round=15, tick=114830}; {page=0, match_id=m_ff5c5ba15dea3b18d2fe8299, round=15, tick=114830}
- **Matches far from median row count:** median over all 1752 matches = 133, zero-row matches = 21.
  - all-matches top: `m_2ef54666039eb9f17bea34e0` rows 572 (ratio 4.3008)
  - all-matches top: `m_1938467dc7f4b25024f71bca` rows 515 (ratio 3.8722)
  - all-matches top: `m_2a1827894f6e922da3b8d93f` rows 505 (ratio 3.797)
  - non-zero median = 134
  - non-zero top: `m_2ef54666039eb9f17bea34e0` rows 572 (ratio 4.2687)
  - non-zero top: `m_1938467dc7f4b25024f71bca` rows 515 (ratio 3.8433)
  - non-zero top: `m_2a1827894f6e922da3b8d93f` rows 505 (ratio 3.7687)

### Other checks

- Null counts: {'match_key': 0, 'round': 0, 'tick': 0, 'player_id': 0, 'attacker_id': 0}. Round range 1..48 (48 distinct). Tick range 1310..331597.
- Tick decreases inside a match (stored order): 0 rows in 0 matches. Examples none
- second vs tick/tick_rate: median residual 0.000s; rows off by >1s: 0; matches with median off by >1s: 0. Examples none
- Negative in name-nonnegative columns: none.
- Range checks run: 16. Failing: `player_z_pos` abs > 10000 3,310; `attacker_z_pos` abs > 10000 3,242.
- Empty strings: none.
- `_fixed` vs base: `player_id_fixed` differs from `player_id` in 223,608 rows (base null 0, fixed null 0); `attacker_id_fixed` differs from `attacker_id` in 222,768 rows (base null 0, fixed null 0).
- player_id == attacker_id rows: 50,507.
- Id zero counts: `player_id` 24,043, `attacker_id` 24,512.
- blind_duration > 10s: 0. Examples none
- Raw player_id: matches with more than 10 distinct ids 43; rows with id outside 0-9 2,705. Raw to fixed is one-to-one within match on 100.00% of pairs; raw 0 maps to 15 distinct fixed values across matches; fixed disagrees with raw on 223,608 rows.

### Checks run

- Read all 1 page(s) listed in the channel manifest with pyarrow; on-disk dataframe files: ['dataframe_00000'].
- Matched match_key to header tome key (header has 1752 keys); zero-row matches = header keys with no channel row.
- Per-column dtype (arrow and pandas), null rate, distinct, min/max/median (numeric), top 5 (<=50 distinct).
- Constant, >50% null, sentinel hits (7 values), spike >20% of rows, impossible-for-name, identical columns, full-row and keyed duplicates, match row-count outliers vs median.
- Extra: tick order in stored order, second vs tick/tick_rate, angle/position/velocity ranges, empty strings, _fixed vs base disagreement, id zeros, damage_dir unit norm, blind duration, effective vs raw health removed, page-spanning matches, schema drift.

## player_footstep

- Table: `basic_player_footstep.2026-09-18,2026-10-08`. Pages (manifest): 1 ['dataframe_00000']. On disk: ['dataframe_00000']. Page rows: [1333987]. Sum matches total: True.
- Rows: 1,333,987. Distinct match_key: 1734 of 1752 header matches. Zero-row matches: 18. Match keys not in header: 0.
- Matches with rows on more than one page: 0.
- Zero-row matches (zero/total) by group, top 5 each: match_type: unknown 17/1074, competitive 1/678 | day: 2026-10-08 17/1133, 2026-10-07 1/87 | map_name: de_nuke 4/165, de_dust2 3/316, de_mirage 3/346, de_anubis 3/207, de_ancient 3/218.
- Schema drift across pages: none (single page or identical schemas).

### Columns

| column | arrow type | null % | distinct | min | max | median | top 5 (value x count, share) |
|---|---|---|---|---|---|---|---|
| round | int64 | 0.00% | 48 | 1 | 48 | 11.0 | 6 x65944 (4.9%); 2 x65904 (4.9%); 10 x65335 (4.9%); 4 x65301 (4.9%); 11 x65181 (4.9%) |
| tick | int64 | 0.00% | 190438 | 1142 | 334196 | 67331.0 |  |
| player_id | int64 | 0.00% | 20 | 0 | 19 | 4.0 | 2 x136502 (10.2%); 0 x135933 (10.2%); 3 x134875 (10.1%); 1 x132842 (10.0%); 4 x132115 (9.9%) |
| second | double | 0.00% | 190438 | 17.84375 | 5221.8125 | 1052.046875 |  |
| player_id_fixed | int64 | 0.00% | 19 | 1 | 19 | 6.0 | 9 x141555 (10.6%); 10 x138967 (10.4%); 8 x135816 (10.2%); 7 x135507 (10.2%); 4 x132413 (9.9%) |
| match_key | string | 0.00% | 1734 |  |  |  |  |

### Flags

- **Constant column (one value):** 0. none.
- **More than 50% null:** 0. none.
- **Sentinel-like values:** 4 (col, value) hits. e.g. `tick`=65535 x14; `tick`=32767 x9
  - `tick` = 65535: 14 rows (0.00%); examples {page=0, match_id=m_37893c3cb97ee0809743c442, round=12, tick=65535, value=65535}; {page=0, match_id=m_d5dd840d91d4cc157ba2a037, round=12, tick=65535, value=65535}
  - `tick` = 32767: 9 rows (0.00%); examples {page=0, match_id=m_710c8fea0961ddf263a68fb0, round=6, tick=32767, value=32767}; {page=0, match_id=m_ed804f52028188eabfb78e35, round=6, tick=32767, value=32767}
  - `tick` = 9999: 19 rows (0.00%); examples {page=0, match_id=m_ff06fa88583ee8b6377e2c3a, round=2, tick=9999, value=9999}; {page=0, match_id=m_7d286d6d3b22c632051a50cf, round=4, tick=9999, value=9999}
  - `second` = 999: 5 rows (0.00%); examples {page=0, match_id=m_5a369e1d4d4a24ff81923e72, round=10, tick=63936, value=999.0}; {page=0, match_id=m_3d5d653073ee43d3a4b85ee0, round=10, tick=63936, value=999.0}
- **Spike (one value above 20% of rows):** 0. none.
- **Impossible for name:** 0 rule hits. none.
- **Identical columns:** 0 pairs. none.
- **Duplicate rows within match and round (full-row):** 0 rows (0.0000%) in 0 match-round groups across 0 matches. Examples none
  - Keyed (match_key, round, tick, player_id, attacker_id, other_type where present): 0 rows sit in duplicate key groups. Examples none
- **Matches far from median row count:** median over all 1752 matches = 759, zero-row matches = 18.
  - all-matches top: `m_2ef54666039eb9f17bea34e0` rows 2,710 (ratio 3.5705)
  - all-matches top: `m_2a1827894f6e922da3b8d93f` rows 2,611 (ratio 3.4401)
  - all-matches top: `m_6ba4651d6a05090572561463` rows 2,223 (ratio 2.9289)
  - non-zero median = 765
  - non-zero top: `m_2ef54666039eb9f17bea34e0` rows 2,710 (ratio 3.5425)
  - non-zero top: `m_2a1827894f6e922da3b8d93f` rows 2,611 (ratio 3.4131)
  - non-zero top: `m_6ba4651d6a05090572561463` rows 2,223 (ratio 2.9059)

### Other checks

- Null counts: {'match_key': 0, 'round': 0, 'tick': 0, 'player_id': 0}. Round range 1..48 (48 distinct). Tick range 1142..334196.
- Tick decreases inside a match (stored order): 0 rows in 0 matches. Examples none
- second vs tick/tick_rate: median residual 0.000s; rows off by >1s: 0; matches with median off by >1s: 0. Examples none
- Negative in name-nonnegative columns: none.
- Range checks run: 0. Failing: none.
- Empty strings: none.
- `_fixed` vs base: `player_id_fixed` differs from `player_id` in 1,219,953 rows (base null 0, fixed null 3).
- Id zero counts: `player_id` 135,933.
- Raw player_id: matches with more than 10 distinct ids 74; rows with id outside 0-9 12,186. Raw to fixed is one-to-one within match on 100.00% of pairs; raw 0 maps to 15 distinct fixed values across matches; fixed disagrees with raw on 1,219,953 rows.

### Checks run

- Read all 1 page(s) listed in the channel manifest with pyarrow; on-disk dataframe files: ['dataframe_00000'].
- Matched match_key to header tome key (header has 1752 keys); zero-row matches = header keys with no channel row.
- Per-column dtype (arrow and pandas), null rate, distinct, min/max/median (numeric), top 5 (<=50 distinct).
- Constant, >50% null, sentinel hits (7 values), spike >20% of rows, impossible-for-name, identical columns, full-row and keyed duplicates, match row-count outliers vs median.
- Extra: tick order in stored order, second vs tick/tick_rate, angle/position/velocity ranges, empty strings, _fixed vs base disagreement, id zeros, damage_dir unit norm, blind duration, effective vs raw health removed, page-spanning matches, schema drift.

