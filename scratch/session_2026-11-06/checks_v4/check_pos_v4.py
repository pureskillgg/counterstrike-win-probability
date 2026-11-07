import json, csv, glob, time
import numpy as np
import polars as pl

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


V2 = '/home/sagemaker-user/adam/builder/final/export_clean_v2/'
V4 = '/home/sagemaker-user/adam/builder/final/export_clean_v4/pos_v4/'
CH = '/home/sagemaker-user/adam/builder/final/pos_all_seconds/'
CEXP = '/home/sagemaker-user/adam/builder/fixes/cexp_f64/out/pos_v4_changes.csv'

INT8 = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_in_contact", "t_in_contact", "c_sees_enemy",
        "t_sees_enemy", "c_exposed", "t_exposed", "c_n_bomb_500", "c_blind_n", "t_blind_n"]
U16 = ["c_nearest_site_mean", "t_nearest_site_mean", "c_spread", "t_spread", "ct_t_centroid_dist",
       "min_enemy_dist", "c_min_bomb_dist", "t_min_bomb_dist"]
U8 = ["c_blind_sum", "t_blind_sum"]
F16 = ["defuse_slack"]
assert len(INT8 + U16 + U8 + F16) == 24

keys = json.load(open(V2 + 'match_keys.json'))
M = len(keys)
kid = {k: i for i, k in enumerate(keys)}
off = np.load(V2 + 'offsets.npy').astype(np.int64)
rm = np.load(V2 + 'round_match.npy').astype(np.int64)
rn = np.load(V2 + 'round_num.npy').astype(np.int64)
frz = np.load(V2 + 'round_freeze.npy').astype(np.int64)
N = len(rm)
S = int(off[-1])
nst = np.diff(off)
cnt = (nst + 1) // 2
pos_off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
E = int(pos_off[-1])
out = {}

# ---------------- check 2: step count, offsets, shapes, source map ----------------
c2 = {'matches': M, 'rounds': N, 'S_from_v2_offsets': S, 'S_is_72567700': S == 72567700,
      'sum_nst_equals_S': int(nst.sum()) == S, 'v2_even_steps_E': E}
v4off = np.load(V4 + 'pos_offsets.npy').astype(np.int64)
c2['v4_pos_offsets'] = {'len': int(len(v4off)), 'len_is_N_plus_1': len(v4off) == N + 1,
                        'first': int(v4off[0]), 'last': int(v4off[-1]),
                        'nondecreasing': bool(np.all(np.diff(v4off) >= 0)),
                        'equals_v2_offsets_npy': bool(np.array_equal(v4off, off))}
spec = {'pos_int8.npy': ((S, 13), np.int8), 'pos_uint16.npy': ((S, 8), np.uint16),
        'pos_uint8.npy': ((S, 2), np.uint8), 'pos_float16.npy': ((S,), np.float16)}
for nm, (shape, dt) in spec.items():
    a = np.load(V4 + nm, mmap_mode='r')
    c2[nm] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'shape_ok': tuple(a.shape) == shape,
              'dtype_ok': a.dtype == dt}
    del a
msrc = np.load(V4 + 'match_pos_source.npy')
bucket_mid = msrc == 1
c2['match_pos_source'] = {'len_ok': len(msrc) == M,
                          'counts_by_value': {int(v): int((msrc == v).sum()) for v in np.unique(msrc)}}
ek = json.load(open(V4 + 'every_second_match_keys.json'))
c2['every_second_keys'] = {'count': len(ek),
                           'equals_source1_keys_in_v2_order': ek == [keys[i] for i in np.nonzero(bucket_mid)[0]]}
out['check2'] = c2
log('check2', json.dumps(c2))

# ---------------- check 1: even steps vs v2 ----------------
bucket_round = bucket_mid[rm]
rid_e = np.repeat(np.arange(N, dtype=np.int64), cnt)
j_e = np.arange(E, dtype=np.int64) - pos_off[rid_e]
v4_e = off[rid_e] + 2 * j_e
del j_e
e_bucket = bucket_round[rid_e]

per_feat = {}
bad_cells = {}
for nm, cols in [('pos_int8.npy', INT8), ('pos_uint16.npy', U16), ('pos_uint8.npy', U8)]:
    A4 = np.asarray(np.load(V4 + nm, mmap_mode='r')[v4_e])
    B2 = np.asarray(np.load(V2 + nm, mmap_mode='r'))
    for i, x in enumerate(cols):
        bad = np.nonzero(A4[:, i] != B2[:, i])[0]
        per_feat[x] = int(len(bad))
        bad_cells[x] = bad
    del A4, B2
A = np.asarray(np.load(V4 + 'pos_float16.npy', mmap_mode='r')[v4_e])
B = np.asarray(np.load(V2 + 'pos_float16.npy', mmap_mode='r'))
bad = np.nonzero(~((A == B) | (np.isnan(A) & np.isnan(B))))[0]
per_feat['defuse_slack'] = int(len(bad))
bad_cells['defuse_slack'] = bad
del A, B

total_bad = sum(per_feat.values())
bucket_bad = sum(int(e_bucket[bad_cells[x]].sum()) for x in bad_cells)
mism_set = {(x, int(e)) for x in bad_cells for e in bad_cells[x]}

# CSV of cone-patch cells: check each row against v2 (old) and v4 (new)
rt_map = {(m, r): i for i, (m, r) in enumerate(zip(rm.tolist(), rn.tolist()))}
fcol = {'c_sees_enemy': 6, 't_sees_enemy': 7, 'c_exposed': 8, 't_exposed': 9}
v4i8 = np.load(V4 + 'pos_int8.npy', mmap_mode='r')
v2i8 = np.load(V2 + 'pos_int8.npy', mmap_mode='r')
rows = list(csv.DictReader(open(CEXP)))
csv_set = set()
ccount = {'rows': len(rows), 'odd_k_rows': 0, 'unmapped': 0, 'new_matches_v4': 0, 'old_matches_v2': 0,
          'by_feature': {}}
for row in rows:
    mid = kid[row['match_key']]
    r = rt_map.get((mid, int(row['round'])))
    if r is None:
        ccount['unmapped'] += 1
        continue
    dt = int(row['tick']) - int(frz[r])
    k = dt // 64
    if dt % 64 != 0 or k % 2 == 1:
        ccount['odd_k_rows'] += 1
        continue
    e = int(pos_off[r] + k // 2)
    v4i = int(off[r] + k)
    col = fcol[row['feature']]
    ccount['new_matches_v4'] += int(int(v4i8[v4i, col]) == int(row['new']))
    ccount['old_matches_v2'] += int(int(v2i8[e, col]) == int(row['old']))
    ccount['by_feature'][row['feature']] = ccount['by_feature'].get(row['feature'], 0) + 1
    csv_set.add((row['feature'], e))

c1 = {'even_steps_compared': E, 'per_feature_mismatch_cells': per_feat,
      'mismatch_cells_total': total_bad, 'mismatch_cells_in_bucket_rounds': bucket_bad,
      'mismatch_cells_in_laptop_rounds': total_bad - bucket_bad,
      'mismatch_set_equals_cone_csv_cells': mism_set == csv_set,
      'csv_cells': len(csv_set), 'csv_check': ccount}
out['check1'] = c1
log('check1', json.dumps(c1))
del rid_e, v4_e, e_bucket

# ---------------- check 3: odd steps ----------------
r_of_step = np.repeat(np.arange(N, dtype=np.int64), nst)
bstep = bucket_round[r_of_step]
k_of_step = np.arange(S, dtype=np.int64) - off[r_of_step]
odd = (k_of_step & 1).astype(bool)
del k_of_step, r_of_step

ns = np.zeros(S, bool)
a = np.asarray(np.load(V4 + 'pos_int8.npy', mmap_mode='r')); ns |= (a != -1).any(1); del a
a = np.asarray(np.load(V4 + 'pos_uint16.npy', mmap_mode='r')); ns |= (a != 65535).any(1); del a
a = np.asarray(np.load(V4 + 'pos_uint8.npy', mmap_mode='r')); ns |= (a != 255).any(1); del a
a = np.asarray(np.load(V4 + 'pos_float16.npy', mmap_mode='r')); ns |= ~np.isnan(a); del a
log('nonsentinel computed; v4 nonsentinel steps total', int(ns.sum()))

rt = pl.DataFrame({'mid': rm, 'round': rn, 'r': np.arange(N, dtype=np.int64), 'nst': nst,
                   'off': off[:-1], 'frz': frz})
tot = {'rows': 0, 'no_key': 0, 'no_round': 0, 'bad_second': 0, 'bad_tick': 0}
chunk_steps = []
for f in sorted(glob.glob(CH + 'bucket_c*.parquet')):
    df = pl.read_parquet(f, columns=['match_key', 'round', 'tick', 'second'])
    tot['rows'] += df.height
    df = df.with_columns(pl.col('match_key').replace_strict(kid, default=-1, return_dtype=pl.Int64).alias('mid'),
                         pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64),
                         pl.col('second').cast(pl.Int64))
    tot['no_key'] += int((df['mid'] < 0).sum())
    df = df.filter(pl.col('mid') >= 0)
    J = df.join(rt, on=['mid', 'round'], how='left')
    tot['no_round'] += int(J['r'].is_null().sum())
    J = J.filter(pl.col('r').is_not_null())
    tot['bad_second'] += int(((J['second'] < 0) | (J['second'] >= J['nst'])).sum())
    J = J.filter((pl.col('second') >= 0) & (pl.col('second') < pl.col('nst')))
    tot['bad_tick'] += int((J['tick'] != J['frz'] + 64 * J['second']).sum())
    J = J.unique(['r', 'second'], keep='first', maintain_order=True)
    chunk_steps.append((J['off'] + J['second']).to_numpy().astype(np.int64))
    del df, J
CS = np.concatenate(chunk_steps)
written = np.zeros(S, bool)
written[CS] = True
c3 = {'chunk_totals': tot, 'chunk_mapped_steps_total': int(len(CS)),
      'chunk_mapped_steps_unique': int(written.sum()),
      'chunk_steps_all_inside_bucket_rounds': bool(not (written & ~bstep).any())}
bo = int((bstep & odd).sum())
be = int((bstep & ~odd).sum())
lo = int((~bstep & odd).sum())
le = int((~bstep & ~odd).sum())
c3['step_counts'] = {'bucket_total': int(bstep.sum()), 'bucket_even': be, 'bucket_odd': bo,
                     'laptop_total': int((~bstep).sum()), 'laptop_even': le, 'laptop_odd': lo, 'S': S}
c3['bucket_steps_without_chunk_row'] = int((bstep & ~written).sum())
c3['bucket_odd_with_chunk_row'] = int((written & odd).sum())
c3['bucket_odd_share_with_chunk_row'] = float((written & odd).sum()) / bo
c3['bucket_odd_with_chunk_row_nonsentinel'] = int((written & odd & ns).sum())
c3['bucket_odd_with_chunk_row_all_sentinel_looking'] = int((written & odd & ~ns).sum())
c3['bucket_odd_nonsentinel_share_of_bucket_odd'] = float((written & odd & ns).sum()) / bo
c3['bucket_odd_without_chunk_row_nonsentinel'] = int((bstep & odd & ~written & ns).sum())
c3['bucket_even_without_chunk_row_nonsentinel'] = int((bstep & ~odd & ~written & ns).sum())
c3['bucket_even_nonsentinel'] = int((bstep & ~odd & ns).sum())
c3['laptop_odd_nonsentinel'] = int((~bstep & odd & ns).sum())
c3['laptop_even_nonsentinel'] = int((~bstep & ~odd & ns).sum())
out['check3'] = c3
log('check3', json.dumps(c3))

# v2 nonsentinel rows for cross-check of copied even steps
v2ns = np.zeros(E, bool)
a = np.asarray(np.load(V2 + 'pos_int8.npy', mmap_mode='r')); v2ns |= (a != -1).any(1); del a
a = np.asarray(np.load(V2 + 'pos_uint16.npy', mmap_mode='r')); v2ns |= (a != 65535).any(1); del a
a = np.asarray(np.load(V2 + 'pos_uint8.npy', mmap_mode='r')); v2ns |= (a != 255).any(1); del a
a = np.asarray(np.load(V2 + 'pos_float16.npy', mmap_mode='r')); v2ns |= ~np.isnan(a); del a
rid_e = np.repeat(np.arange(N, dtype=np.int64), cnt)
e_b2 = bucket_round[rid_e]
out['check3_v2_crosscheck'] = {'v2_nonsentinel_even_rows_in_laptop_rounds': int(v2ns[~e_b2].sum()),
                               'v4_laptop_even_nonsentinel': c3['laptop_even_nonsentinel'],
                               'v2_nonsentinel_even_rows_in_bucket_rounds': int(v2ns[e_b2].sum()),
                               'v4_bucket_even_nonsentinel': c3['bucket_even_nonsentinel']}
log('check3 v2 crosscheck', json.dumps(out['check3_v2_crosscheck']))

print(json.dumps(out, indent=1, default=str))
