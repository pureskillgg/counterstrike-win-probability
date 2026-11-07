import json, subprocess
from collections import Counter
import pyarrow.parquet as pq

D = '/home/sagemaker-user/adam/builder/data'
F = '/home/sagemaker-user/adam/builder/final'
E = F + '/export_clean_v2'
keys = json.load(open(E + '/match_keys.json'))
ks = set(keys)

rows = {}
for c in range(68):
    t = pq.read_table(f'{D}/c{c:03d}/matches.parquet', columns=['match_key', 'map_name', 'match_date', 'in_scope', 'scope_reason', 'is_wingman', 'is_hostage_map', 'era']).to_pydict()
    for i, k in enumerate(t['match_key']):
        rows[k] = {f: t[f][i] for f in t}
universe = set(rows)
print('universe (matches.parquet, c000-c067)', len(universe))
inter = universe & ks
print('universe x export', len(inter))
not_exp = universe - ks
print('universe not in export', len(not_exp))

tome = [l.strip() for l in subprocess.run(['aws', 's3', 'cp', 's3://pureskillgg-csgo-production-sagemaker-0/adam/tomes/holdout_2026-10-26_31/match_keys.txt', '-'], capture_output=True, text=True).stdout.splitlines() if l.strip()]
hb = [l.strip() for l in open(F + '/holdout_b_keys.txt', encoding='utf-8') if l.strip()]
ts, hbs = set(tome), set(hb)
print('universe x tome', len(universe & ts), '| universe x holdout_b', len(universe & hbs))
print('not-in-export x tome', len(not_exp & ts), '| not-in-export x holdout_b', len(not_exp & hbs))
rest = not_exp - ts - hbs
print('not-in-export excluding tome and holdout_b', len(rest))
print('scope_reason of not-in-export (all)', Counter(rows[k]['scope_reason'] for k in not_exp).most_common(12))
print('scope_reason of rest', Counter(rows[k]['scope_reason'] for k in rest).most_common(12))
print('in_scope of rest', Counter(rows[k]['in_scope'] for k in rest))
print('era of rest', Counter(rows[k]['era'] for k in rest))
print('map_name of rest', Counter(rows[k]['map_name'] for k in rest).most_common(12))

cm = pq.read_table(F + '/clean_matches.parquet', columns=['match_key', 'src', 'dup_keeper', 'in_scope', 'n_ok_rounds']).to_pydict()
cm_rows = {}
for i, k in enumerate(cm['match_key']):
    cm_rows.setdefault(k, []).append((cm['src'][i], cm['dup_keeper'][i], cm['in_scope'][i], cm['n_ok_rounds'][i]))
print('rest keys present in clean_matches', sum(1 for k in rest if k in cm_rows))
print('rest keys dup_keeper flags', Counter(tuple(sorted(set(x[1] for x in cm_rows[k]))) for k in rest if k in cm_rows).most_common(5))
absent47 = set(k for k in universe if k in set())  # placeholder
