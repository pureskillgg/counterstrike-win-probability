import numpy as np, json, glob
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from collections import Counter

E = '/home/sagemaker-user/adam/builder/final/export_clean_v2'
F = '/home/sagemaker-user/adam/builder/final'
keys = json.load(open(E + '/match_keys.json'))
rm = np.load(E + '/round_match.npy').astype(np.int64)
rnum = np.load(E + '/round_num.npy').astype(np.int64)
exp_pairs = set(zip([keys[i] for i in rm.tolist()], rnum.tolist()))
print('export (match,round) pairs', len(exp_pairs), 'rounds', len(rm))

cr = pq.read_table(F + '/clean_rounds.parquet', columns=['match_key', 'round', 'use', 'dup_keeper', 'dup_group_size', 'src', 'round_status', 'in_scope', 'channels_complete']).to_pydict()
n = len(cr['match_key'])
print('clean_rounds rows', n, '| use==True', sum(1 for u in cr['use'] if u))
use_idx = [i for i in range(n) if cr['use'][i]]
use_pairs = set((cr['match_key'][i], cr['round'][i]) for i in use_idx)
print('use==True distinct (match,round)', len(use_pairs))
print('export pairs subset of use pairs', exp_pairs <= use_pairs, '| export not in use', len(exp_pairs - use_pairs))
dropped = use_pairs - exp_pairs
print('use rounds not in export (should be 934)', len(dropped))
dropped_rows = [i for i in use_idx if (cr['match_key'][i], cr['round'][i]) in dropped]
print('dropped: dup_keeper', Counter(cr['dup_keeper'][i] for i in dropped_rows))
print('dropped: dup_group_size', Counter(cr['dup_group_size'][i] for i in dropped_rows))
print('dropped: src', Counter(cr['src'][i] for i in dropped_rows))
print('dropped: round_status', Counter(cr['round_status'][i] for i in dropped_rows).most_common(5))
print('dropped: matches involved', len(set(cr['match_key'][i] for i in dropped_rows)))
print('use rounds with dup_keeper False', sum(1 for i in use_idx if cr['dup_keeper'][i] is False))

# rounds in use and in export, by src
exp_src = Counter()
for i in use_idx:
    if (cr['match_key'][i], cr['round'][i]) in exp_pairs:
        exp_src[cr['src'][i]] += 1
print('export rounds by src (from clean_rounds)', dict(exp_src))

# pos_parts (match, round) pairs not in export
parts = []
for p in sorted(glob.glob(F + '/pos_parts/bucket_c*.parquet')):
    parts.append(pq.read_table(p, columns=['match_key', 'round']))
pt = pa.concat_tables(parts)
g = pt.group_by(['match_key', 'round']).aggregate([])
pp_pairs = set(zip(g.column('match_key').to_pylist(), g.column('round').to_pylist()))
print('pos_parts distinct (match,round) pairs', len(pp_pairs))
pp_not_exp = pp_pairs - exp_pairs
print('pos_parts pairs not in export', len(pp_not_exp))
print('  of which in dropped set', len(pp_not_exp & dropped), '| not in use set at all', len(pp_not_exp - use_pairs))

# 47 empty matches
empt = set(keys) and None
pp_keys_not_exp = set(k for k, r in pp_not_exp)
print('distinct keys in pos_parts-not-export', len(pp_keys_not_exp))
ks = set(keys)
use_rounds_for_empty = Counter()
for i in range(n):
    if cr['match_key'][i] in pp_keys_not_exp and cr['use'][i]:
        use_rounds_for_empty[cr['match_key'][i]] += 1
print('empty matches (pos_parts-only) with any use round in clean_rounds', len(use_rounds_for_empty))
print('empty matches in export keys', len(pp_keys_not_exp & ks))
