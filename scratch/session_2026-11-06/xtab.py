import json, subprocess
from collections import Counter, defaultdict
import pyarrow.parquet as pq

D = '/home/sagemaker-user/adam/builder/data'
F = '/home/sagemaker-user/adam/builder/final'
E = F + '/export_clean_v2'
keys = json.load(open(E + '/match_keys.json'))
ks = set(keys)

chunk = {}
for c in range(68):
    t = pq.read_table(f'{D}/c{c:03d}/matches.parquet', columns=['match_key', 'scope_reason', 'in_scope', 'map_name', 'era']).to_pydict()
    for i, k in enumerate(t['match_key']):
        chunk[k] = (t['scope_reason'][i], t['in_scope'][i], t['map_name'][i], t['era'][i])
universe = set(chunk)

cm = pq.read_table(F + '/clean_matches.parquet').to_pydict()
print('clean_matches columns', list(cm.keys()))
cmr = defaultdict(list)
for i, k in enumerate(cm['match_key']):
    cmr[k].append({f: cm[f][i] for f in cm if f != 'match_key'})
print('clean_matches rows', len(cm['match_key']), 'distinct keys', len(cmr))
usable_guess = [k for k, rs in cmr.items() if any(r['dup_keeper'] and r['in_scope'] and (r['n_ok_rounds'] or 0) > 0 for r in rs)]
print('guess usable (dup_keeper & in_scope & n_ok_rounds>0) keys', len(usable_guess), 'of which in export', len(set(usable_guess) & ks), '| export not in guess', len(ks - set(usable_guess)))

not_exp = universe - ks
cat = Counter()
examples = defaultdict(list)
for k in not_exp:
    sr, ins, mp, era = chunk[k]
    rs = cmr.get(k, [])
    if not rs:
        key = ('not_in_clean_matches', sr)
    else:
        r = rs[0]
        key = ('in_clean', 'dup_keeper=%s' % r['dup_keeper'], 'in_scope=%s' % r['in_scope'], 'n_ok>0=%s' % ((r['n_ok_rounds'] or 0) > 0), 'src=%s' % r['src'], 'chunk_scope=%s' % sr)
    cat[key] += 1
    if len(examples[key]) < 1:
        examples[key].append(k)
for key, n in cat.most_common(40):
    print(n, key)

# the 47 pos_parts-only keys and the 653 holdout B keys
hb = set(l.strip() for l in open(F + '/holdout_b_keys.txt', encoding='utf-8') if l.strip())
hb_in = hb & universe
print('holdout_b in universe', len(hb_in), '| their clean_matches presence', Counter(('in_clean' if k in cmr else 'absent') for k in hb_in))
print('holdout_b in universe chunk scope reasons', Counter(chunk[k][0] for k in hb_in).most_common(5))
print('holdout_b in clean_matches dup_keeper', Counter(tuple(sorted(set(str(r['dup_keeper']) for r in cmr[k]))) for k in hb_in if k in cmr).most_common(3))
