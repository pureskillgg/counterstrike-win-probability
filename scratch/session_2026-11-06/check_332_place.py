import json, collections
import polars as pl

B = '/home/sagemaker-user/adam/builder/'
RAW = B + 'raw_cache_bucket/'
OUTD = B + 'fixes/fix332/'

sc = pl.read_parquet(B + 'fixes/freeze_scope/scope_rounds.parquet')
aff = sc.filter(pl.col('affected'))
fx = (pl.read_csv(B + 'fixes/freeze_fix/override_rounds.csv', infer_schema_length=0)
      .with_columns(pl.col('round').cast(pl.Int64))
      .filter(pl.col('status') == 'fixed').select('match_key', 'round'))
rem = aff.join(fx, on=['match_key', 'round'], how='anti')
print('rem rounds', rem.height, flush=True)

rows = []
schema_done = False
place_col = None
for mk in rem['match_key'].unique().sort().to_list():
    folder = RAW + mk.rsplit('/', 1)[0] + '/'
    ps = pl.read_parquet(folder + 'player_status')
    if not schema_done:
        print('player_status columns', ps.columns, flush=True)
        place_col = [c for c in ps.columns if 'place' in c.lower()]
        print('place columns', place_col, flush=True)
        schema_done = True
    if not place_col:
        raise SystemExit('no place column in player_status')
    pc = place_col[0]
    ps = ps.with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64), pl.col('player_id').cast(pl.Int64))
    rr = rem.filter(pl.col('match_key') == mk)
    for r in rr.iter_rows(named=True):
        s = ps.filter((pl.col('round') == r['round']) & (pl.col('tick') == r['freeze']) & (pl.col('health') > 0))
        places = collections.Counter(s[pc].to_list())
        ct = int(places.get('CTSpawn', 0))
        t = int(places.get('TSpawn', 0))
        other = s.height - ct - t
        rows.append({
            'match_key': mk, 'round': r['round'], 'freeze': r['freeze'], 'alive_n_scope': r['alive_n'],
            'dup_alive': r['dup_alive'], 'unro_alive': r['unro_alive'],
            'alive_n_status': s.height, 'ct_place': ct, 't_place': t, 'other_place': other,
            'gate_5v5': ct == 5 and t == 5,
            'places_raw': json.dumps({str(k): v for k, v in places.items()}),
        })

out = pl.DataFrame(rows)
out.write_csv(OUTD + 'side_rule_332_check.csv')
summ = {
    'rounds': out.height,
    'gate_5v5_pass': int(out['gate_5v5'].sum()),
    'alive_status_eq_scope': int((out['alive_n_status'] == out['alive_n_scope']).sum()),
    'by_alive_n_scope': {},
}
for k, g in out.group_by('alive_n_scope'):
    key = str(k[0]) if isinstance(k, tuple) else str(k)
    summ['by_alive_n_scope'][key] = {
        'rounds': g.height,
        'gate_5v5_pass': int(g['gate_5v5'].sum()),
        'ct_place_hist': {str(a): b for a, b in collections.Counter(g['ct_place'].to_list()).items()},
        't_place_hist': {str(a): b for a, b in collections.Counter(g['t_place'].to_list()).items()},
    }
json.dump(summ, open(OUTD + 'side_rule_332_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1)[:4000])
