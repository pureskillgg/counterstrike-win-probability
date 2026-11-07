import json, csv, collections
import numpy as np
V2 = '/home/sagemaker-user/adam/builder/final/export_clean_v2/'
V3 = '/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/'
V4 = '/home/sagemaker-user/adam/builder/final/export_clean_v4/'
keys = json.load(open(V4 + 'match_keys.json'))
off = np.load(V4 + 'offsets.npy').astype(np.int64)
nst = np.diff(off)
rm = np.load(V4 + 'round_match.npy').astype(np.int64)
N = len(rm)
rows3 = list(csv.DictReader(open(V3 + 'match_loadout_status.csv', encoding='utf-8')))
rows4 = list(csv.DictReader(open(V4 + 'match_loadout_status.csv', encoding='utf-8')))
assert [r['match_key'] for r in rows4] == keys and [r['match_key'] for r in rows3] == keys
built3 = np.array([r['build_status'] != 'not_built' for r in rows3])
built4 = np.array([r['build_status'] in ('s3_channel', 'local_1hz') for r in rows4])
same = built3 & built4
print(json.dumps({'keys': len(keys), 'v4_keys_with_status_in_allowed_set': int(built4.sum()),
                  'v3_built_keys': int(built3.sum()), 'same_keys_built_both': int(same.sum()),
                  'v4_only_keys_coverage_new': int((built4 & ~built3).sum())}))
# export membership counts (key strings only): dates and holdout B list
import re
dates = [re.search(r'csds/(\d{4}/\d{2}/\d{2})', k).group(1) for k in keys]
print(json.dumps({'export_keys_dated_2026_10_26_to_31': sum(1 for d in dates if d in {f'2026/10/{x}' for x in range(26, 32)}),
                  'export_keys_dated_2026_10_21_to_25': sum(1 for d in dates if d in {f'2026/10/{x}' for x in range(21, 26)}),
                  'export_day_min': min(dates), 'export_day_max': max(dates)}))
hb = set(l.strip() for l in open('/home/sagemaker-user/adam/builder/final/holdout_b_keys.txt', encoding='utf-8') if l.strip())
print(json.dumps({'export_keys_in_holdout_b_list': len(set(keys) & hb)}))

rsel = same[rm]
R = np.nonzero(rsel)[0]
F = off[R]
print(json.dumps({'rounds_total': N, 'rounds_in_same_keys': int(len(R)), 'rounds_with_zero_steps': int((nst[R] == 0).sum())}))
A3 = np.load(V3 + 'loadout_step_i16.npy', mmap_mode='r')
A4 = np.load(V4 + 'loadout_step_i16.npy', mmap_mode='r')
E3 = np.load(V3 + 'loadout_equip_i32.npy', mmap_mode='r')
E4 = np.load(V4 + 'loadout_equip_i32.npy', mmap_mode='r')
s3 = np.asarray(A3[F]); s4 = np.asarray(A4[F])
dc = s3 != s4                       # (R, 2, 19) cells
side = dc.any(axis=2)               # (R, 2) side-rows
rnd = side.any(axis=1)
e3 = np.asarray(E3[F]); e4 = np.asarray(E4[F])
de = e3 != e4
res = {'freeze_cells_compared': int(dc.size), 'freeze_cells_differing': int(dc.sum()),
       'freeze_side_rows_differing_steps_array': int(side.sum()),
       'freeze_rounds_with_any_diff': int(rnd.sum()),
       'freeze_matches_with_any_diff': int(len(set(rm[R[rnd]].tolist()))),
       'freeze_side_rows_differing_equip': int(de.sum()),
       'freeze_rounds_differing_equip_any_side': int(de.any(axis=1).sum()),
       'diff_by_feature_index': {str(i): int(c) for i, c in enumerate(dc.sum(axis=(0, 1))) if c},
       'diff_by_side': {'0': int(side[:, 0].sum()), '1': int(side[:, 1].sum())}}
print(json.dumps(res))
# any round-level side-row with diff but equip unchanged, etc.
both = side | de
print(json.dumps({'side_rows_differing_in_steps_or_equip': int(both.sum()),
                  'union_rounds': int((side.any(1) | de.any(1)).sum())}))
# README mentions of 300
t = open(V4 + 'README.json', encoding='utf-8').read()
hits = [t[max(0, i - 200):i + 100].replace('\n', ' ') for i in range(len(t)) if t.startswith('300', i)]
print('v4 README "300" hits', len(hits))
for h in hits[:8]:
    print('   ...', h)
