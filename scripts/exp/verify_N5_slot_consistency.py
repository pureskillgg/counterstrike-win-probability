import json, os
import polars as pl
P = 'G:/pureskill/csds-cs2/data/state_sample/players_1hz.parquet'
OUT = 'G:/pureskill/csds-cs2/data/viz/verify_N5_slot_consistency.json'
if os.path.exists(OUT):
    raise SystemExit('refusing to overwrite ' + OUT)
RIFLE_SMG = {7, 8, 10, 13, 16, 39, 60, 17, 19, 23, 24, 26, 33, 34, 9, 11, 38, 40, 14, 28, 25, 27, 29, 35}
PISTOL = {1, 2, 3, 4, 30, 32, 36, 61, 63, 64}
s0 = (pl.scan_parquet(P).filter(pl.col('seconds_since_live') == 0)
      .select(['inv_primary', 'weapon_code', 'inv_secondary']).collect())
none_rows = s0.filter(pl.col('inv_primary') == 0)
bad = none_rows.filter(pl.col('weapon_code').is_in(list(RIFLE_SMG)))
prim_rows = s0.filter(pl.col('inv_primary') > 0)
prim_bad = prim_rows.filter(pl.col('weapon_code').is_in(list(PISTOL)) == False)
out = {
    'none_rows': none_rows.height,
    'none_rows_active_weapon_is_rifle_or_smg_or_sniper_or_shotgun': bad.height,
    'primary_rows': prim_rows.height,
    'primary_rows_active_weapon_knife_or_grenade_or_other_non_gun': int(prim_rows.filter(
        ~pl.col('weapon_code').is_in(list(RIFLE_SMG) + list(PISTOL))).height),
    'none_rows_secondary_codes_top': [dict(r) for r in none_rows.group_by('inv_secondary').agg(pl.len().alias('n')).sort('n', descending=True).head(6).iter_rows(named=True)],
}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
