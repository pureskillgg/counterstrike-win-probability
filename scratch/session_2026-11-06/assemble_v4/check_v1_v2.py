import json, glob, collections
import numpy as np
import polars as pl

V1 = '/home/sagemaker-user/adam/builder/final/export_clean/'
V2 = '/home/sagemaker-user/adam/builder/final/export_clean_v2/'
FIN = '/home/sagemaker-user/adam/builder/final/'

print('v1 files', sorted(__import__('os').listdir(V1))[:40])
k1 = json.load(open(V1 + 'match_keys.json'))
k2 = json.load(open(V2 + 'match_keys.json'))
print('v1 keys', len(k1), 'v2 keys', len(k2))
s1, s2 = set(k1), set(k2)
print('v1 keys not in v2', len(s1 - s2), 'v2 keys not in v1', len(s2 - s1))
gone = s1 - s2
# order preserved?
kept_in_order = [k for k in k1 if k in s2]
print('v2 order equals v1 order restricted to kept keys', kept_in_order == k2)

o1 = np.load(V1 + 'offsets.npy'); o2 = np.load(V2 + 'offsets.npy')
rm1 = np.load(V1 + 'round_match.npy'); rm2 = np.load(V2 + 'round_match.npy')
rn1 = np.load(V1 + 'round_num.npy').astype(np.int64); rn2 = np.load(V2 + 'round_num.npy').astype(np.int64)
fz1 = np.load(V1 + 'round_freeze.npy').astype(np.int64); fz2 = np.load(V2 + 'round_freeze.npy').astype(np.int64)
ns1 = np.diff(o1); ns2 = np.diff(o2)
print('v1 rounds', len(rm1), 'v1 total steps', int(o1[-1]), 'v2 rounds', len(rm2), 'v2 total steps', int(o2[-1]))

# map each v1 round to (key, round_num)
key1 = np.array(k1, dtype=object)
rows1 = [(k1[m], int(r)) for m, r in zip(rm1, rn1)]
rows2 = [(k2[m], int(r)) for m, r in zip(rm2, rn2)]
idx2 = {p: i for i, p in enumerate(rows2)}
print('v2 (key, round) unique', len(idx2) == len(rows2))
present = [idx2.get(p) for p in rows1]
n_present = sum(1 for x in present if x is not None)
print('v1 rounds present in v2 (key,round)', n_present, 'absent', len(rows1) - n_present)
absent_by_gone_key = sum(1 for p in rows1 if p[0] in gone)
print('v1 rounds absent because key removed', absent_by_gone_key)
dropped_rounds_present_keys = len(rows1) - n_present - absent_by_gone_key
print('v1 rounds dropped inside kept keys', dropped_rounds_present_keys)
# steps and freeze equality on the common rounds
same_steps = same_frz = 0
diff_steps_examples = []
for i, j in enumerate(present):
    if j is None:
        continue
    if ns1[i] == ns2[j]:
        same_steps += 1
    elif len(diff_steps_examples) < 3:
        diff_steps_examples.append((rows1[i], int(ns1[i]), int(ns2[j])))
    if fz1[i] == fz2[j]:
        same_frz += 1
print('common rounds with equal step count', same_steps, 'equal freeze', same_frz, 'examples differing steps', diff_steps_examples)
gone_steps = int(sum(ns1[i] for i, p in enumerate(rows1) if p[0] in gone))
print('v1 steps on removed keys', gone_steps)
print('v1 steps on dropped rounds in kept keys', int(sum(ns1[i] for i, j in enumerate(present) if j is None and rows1[i][0] not in gone)))
# bucket chunks: map rows to v2 rounds
tot = mapped = unmapped_gone = unmapped_round = 0
ok_tick = bad_tick = 0
bad_second = 0
cover = collections.Counter()
ex_unmapped = []
for f in sorted(glob.glob(FIN + 'pos_all_seconds/bucket_c*.parquet')):
    c = f.split('bucket_')[1][:4]
    df = pl.read_parquet(f, columns=['match_key', 'round', 'tick', 'second'])
    tot += df.height
    grp = df.group_by(['match_key', 'round']).agg(pl.len().alias('n'), pl.col('second').max().alias('smax'), pl.col('second').min().alias('smin'), pl.col('tick').first().alias('tick0'), pl.col('second').first().alias('s0'))
    for row in grp.iter_rows(named=True):
        p = (row['match_key'], int(row['round']))
        j = idx2.get(p)
        if j is None:
            if row['match_key'] in gone:
                unmapped_gone += row['n']
            else:
                unmapped_round += row['n']
                if len(ex_unmapped) < 3:
                    ex_unmapped.append((c, p, row['n']))
            continue
        mapped += row['n']
        cover[(c, 'rounds_mapped')] += 1
        # v2 round j: steps ns2[j], freeze fz2[j]
        if row['smax'] >= ns2[j] or row['smin'] < 0:
            bad_second += 1
        exp_tick = fz2[j] + 64 * row['s0']
        if row['tick0'] == exp_tick:
            ok_tick += 1
        else:
            bad_tick += 1
        if row['n'] != ns2[j]:
            cover[(c, 'round_rows_ne_steps')] += 1
print('bucket rows total', tot, 'mapped to v2 rounds', mapped, 'unmapped (removed key)', unmapped_gone, 'unmapped (dropped round in kept key)', unmapped_round)
print('mapped rounds with tick==freeze+64*second on first row ok', ok_tick, 'bad', bad_tick, 'second out of range', bad_second)
print('mapped rounds whose row count != v2 steps', sum(v for k, v in cover.items() if k[1] == 'round_rows_ne_steps'))
print('example unmapped rounds', ex_unmapped)
