import json, os
import numpy as np
import polars as pl

P = 'G:/pureskill/csds-cs2/data/state_sample/players_1hz.parquet'
OUT = 'G:/pureskill/csds-cs2/data/viz/verify_N5_money_bin_bootstrap.json'
if os.path.exists(OUT):
    raise SystemExit('refusing to overwrite ' + OUT)
B, SEED = 500, 0
PRIMARY_CLASS = {7: 'rifle', 8: 'rifle', 10: 'rifle', 13: 'rifle', 16: 'rifle', 39: 'rifle', 60: 'rifle',
                 17: 'smg', 19: 'smg', 23: 'smg', 24: 'smg', 26: 'smg', 33: 'smg', 34: 'smg',
                 9: 'sniper', 11: 'sniper', 38: 'sniper', 40: 'sniper',
                 14: 'shotgun_mg', 28: 'shotgun_mg', 25: 'shotgun_mg', 27: 'shotgun_mg', 29: 'shotgun_mg', 35: 'shotgun_mg', 0: 'none'}
s0 = (pl.scan_parquet(P).filter(pl.col('seconds_since_live') == 0)
      .select(['match_key', 'round', 'team_code', 'armor', 'money', 'inv_primary']).collect())
pr = s0.group_by(['match_key', 'round']).agg([pl.len().alias('n'), (pl.col('team_code') == 2).sum().alias('nT')])
clean = pr.filter((pl.col('n') == 10) & (pl.col('nT') == 5)).select(['match_key', 'round'])
d = s0.join(clean, on=['match_key', 'round'], how='inner')
d = d.with_columns([pl.col('inv_primary').replace_strict(PRIMARY_CLASS, default=None).alias('cls'),
                    (pl.col('armor') >= 1).cast(pl.Float64).alias('a1'),
                    pl.when(pl.col('money') < 1000).then(pl.lit('lt1000'))
                      .when(pl.col('money') < 2000).then(pl.lit('1000_1999'))
                      .when(pl.col('money') < 3000).then(pl.lit('2000_2999'))
                      .otherwise(pl.lit('3000plus')).alias('mbin')])
matches = np.unique(d['match_key'].to_numpy())
midx = np.searchsorted(matches, d['match_key'].to_numpy())
M = len(matches)
rng = np.random.default_rng(SEED)
boot = rng.integers(0, M, size=(B, M))
cls = d['cls'].to_numpy(); mb = d['mbin'].to_numpy(); a1 = d['a1'].to_numpy()

def sums(mask):
    num = np.bincount(midx[mask], weights=a1[mask], minlength=M)
    den = np.bincount(midx[mask], minlength=M).astype(float)
    return num, den

def share_boot(mask):
    num, den = sums(mask)
    nb = num[boot].sum(1); db = den[boot].sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(db > 0, nb / np.where(db > 0, db, 1), np.nan)

res = {}
for b in ['lt1000', '1000_1999', '2000_2999', '3000plus', 'ALL']:
    bm = np.ones(len(d), bool) if b == 'ALL' else (mb == b)
    ms, mr = bm & (cls == 'smg'), bm & (cls == 'rifle')
    diff = share_boot(ms) - share_boot(mr)
    pt = a1[ms].mean() - a1[mr].mean()
    fin = diff[np.isfinite(diff)]
    res[b] = {'smg_n': int(ms.sum()), 'rifle_n': int(mr.sum()),
              'smg_share': float(a1[ms].mean()), 'rifle_share': float(a1[mr].mean()),
              'diff_point': float(pt), 'diff_ci95_match_cluster': [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))],
              'nan_resamples': int((~np.isfinite(diff)).sum())}
out = {'B': B, 'seed': SEED, 'matches_in_universe': int(M), 'bins': res,
       'ASSUMPTION': 'money bins from money at sec0; same match-cluster draws across bins; stratum-level, not reweighted'}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
for k, v in res.items():
    print(k, v)
print('WROTE', OUT)
