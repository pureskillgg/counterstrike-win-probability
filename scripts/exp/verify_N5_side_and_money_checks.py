import json, os
import numpy as np
import polars as pl

P = 'G:/pureskill/csds-cs2/data/state_sample/players_1hz.parquet'
R = 'G:/pureskill/csds-cs2/data/state_sample/rounds.parquet'
OUT = 'G:/pureskill/csds-cs2/data/viz/verify_N5_side_and_money_checks.json'
if os.path.exists(OUT):
    raise SystemExit('refusing to overwrite ' + OUT)

s0 = (pl.scan_parquet(P).filter(pl.col('seconds_since_live') == 0)
      .select(['match_key', 'round', 'player_id', 'team_code', 'has_defuser', 'inv_c4', 'weapon_code', 'armor', 'money', 'inv_primary'])
      .collect())
out = {'side_identity': {}, 'round_number_vs_round': {}, 'money_stratified': {}}

# side identity: defuser only for CT, C4 only for T, knife codes 42 (CT) / 59 (T)
out['side_identity']['has_defuser_by_team'] = [dict(r) for r in
    s0.group_by(['team_code', 'has_defuser']).agg(pl.len().alias('n')).sort(['team_code', 'has_defuser']).iter_rows(named=True)]
out['side_identity']['inv_c4_gt0_by_team'] = [dict(r) for r in
    s0.with_columns((pl.col('inv_c4') > 0).alias('has_c4')).group_by(['team_code', 'has_c4']).agg(pl.len().alias('n'))
      .sort(['team_code', 'has_c4']).iter_rows(named=True)]
kn = s0.filter(pl.col('weapon_code').is_in([42, 59])).group_by(['team_code', 'weapon_code']).agg(pl.len().alias('n')).sort(['team_code', 'weapon_code'])
out['side_identity']['knife_code_by_team'] = [dict(r) for r in kn.iter_rows(named=True)]

rd = pl.read_parquet(R, columns=['match_key', 'round', 'round_number', 'ct_score', 't_score', 'ct_win'])
out['round_number_vs_round'] = {
    'rows_round_ne_round_number': int((rd['round'] != rd['round_number']).sum()),
    'round_number_range': [int(rd['round_number'].min()), int(rd['round_number'].max())],
}

# money stratification of SMG vs rifle at sec0 (clean 5v5 rounds only)
pr = s0.group_by(['match_key', 'round']).agg([pl.len().alias('n'), (pl.col('team_code') == 2).sum().alias('nT')])
clean = pr.filter((pl.col('n') == 10) & (pl.col('nT') == 5)).select(['match_key', 'round'])
d = s0.join(clean, on=['match_key', 'round'], how='inner')
W = 'G:/pureskill/csds-cs2/tomes/combined_test/tome/csds/basic_weapon_fire.COMBINED/dataframe_*'
PRIMARY_CLASS = {7: 'rifle', 8: 'rifle', 10: 'rifle', 13: 'rifle', 16: 'rifle', 39: 'rifle', 60: 'rifle',
                 17: 'smg', 19: 'smg', 23: 'smg', 24: 'smg', 26: 'smg', 33: 'smg', 34: 'smg',
                 9: 'sniper', 11: 'sniper', 38: 'sniper', 40: 'sniper',
                 14: 'shotgun_mg', 28: 'shotgun_mg', 25: 'shotgun_mg', 27: 'shotgun_mg', 29: 'shotgun_mg', 35: 'shotgun_mg', 0: 'none'}
d = d.with_columns(pl.col('inv_primary').replace_strict(PRIMARY_CLASS, default=None).alias('cls'))
assert d['cls'].null_count() == 0
d = d.with_columns([(pl.col('armor') >= 1).cast(pl.Float64).alias('a1'),
                    pl.when(pl.col('money') < 1000).then(pl.lit('a_<1000'))
                      .when(pl.col('money') < 2000).then(pl.lit('b_1000-1999'))
                      .when(pl.col('money') < 3000).then(pl.lit('c_2000-2999'))
                      .otherwise(pl.lit('d_3000+')).alias('mbin')])
g = d.filter(pl.col('cls').is_in(['smg', 'rifle'])).group_by(['mbin', 'cls']).agg(
    [pl.len().alias('n'), pl.col('a1').mean().alias('armor_ge1'), pl.col('money').mean().alias('mean_money')]).sort(['mbin', 'cls'])
out['money_stratified']['rows'] = [dict(r) for r in g.iter_rows(named=True)]
mm = d.filter(pl.col('cls').is_in(['smg', 'rifle'])).group_by('cls').agg(
    [pl.len().alias('n'), pl.col('a1').mean().alias('armor_ge1'), pl.col('money').mean().alias('mean_money'),
     (pl.col('money') < 2000).mean().alias('share_money_lt_2000')])
out['money_stratified']['unstratified'] = [dict(r) for r in mm.iter_rows(named=True)]

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
print('WROTE', OUT)
