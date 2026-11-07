import json, os, time, hashlib, csv, glob
import numpy as np
import polars as pl

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


V2 = '/home/sagemaker-user/adam/builder/final/export_clean_v2/'
TOP = '/home/sagemaker-user/adam/builder/final/export_clean_v4/'
CH = '/home/sagemaker-user/adam/builder/final/pos_all_seconds/'
CEXP = '/home/sagemaker-user/adam/builder/fixes/cexp_f64/out/pos_v4_changes.csv'
OUT = '/home/sagemaker-user/adam/builder/final/export_clean_v4/pos_v4/'
assert 'export_clean_v4/pos_v4' in OUT and 'export_clean_v2' not in OUT and 'export_clean_v3' not in OUT
assert 'pos_parts' not in OUT and 'pos_all_seconds' not in OUT
os.makedirs(OUT, exist_ok=True)

INT8 = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_in_contact", "t_in_contact", "c_sees_enemy",
        "t_sees_enemy", "c_exposed", "t_exposed", "c_n_bomb_500", "c_blind_n", "t_blind_n"]
U16 = ["c_nearest_site_mean", "t_nearest_site_mean", "c_spread", "t_spread", "ct_t_centroid_dist",
       "min_enemy_dist", "c_min_bomb_dist", "t_min_bomb_dist"]
U8 = ["c_blind_sum", "t_blind_sum"]
F16 = ["defuse_slack"]
ALL = INT8 + U16 + U8 + F16
assert len(INT8) == 13 and len(U16) == 8 and len(U8) == 2 and len(ALL) == 24

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
assert N == 1058690 and S == 72567700 and E == 36547868, (N, S, E)
assert np.array_equal(pos_off, np.load(V2 + 'pos_offsets.npy').astype(np.int64))
rt = pl.DataFrame({'mid': rm, 'round': rn, 'r': np.arange(N, dtype=np.int64), 'frz': frz, 'nst': nst, 'off': off[:-1]})
assert rt.select(['mid', 'round']).is_unique().all()
log('v2 structure ok: matches', M, 'rounds', N, 'steps', S, 'even steps', E)


def enc_i8(x):
    return np.where(np.isnan(x), -1, np.clip(np.round(x), 0, 5)).astype(np.int8)


def enc_u16(x):
    return np.where(np.isnan(x), 65535, np.clip(np.round(x), 0, 65534)).astype(np.uint16)


def enc_u8(x):
    return np.where(np.isnan(x), 255, np.clip(np.round(x * 10), 0, 254)).astype(np.uint8)


out_i8 = np.full((S, 13), -1, np.int8)
out_u16 = np.full((S, 8), 65535, np.uint16)
out_u8 = np.full((S, 2), 255, np.uint8)
out_f16 = np.full(S, np.nan, np.float16)
written = np.zeros(S, bool)
bucket_mid = np.zeros(M, bool)

tot = dict(rows=0, no_key=0, no_round=0, bad_second=0, bad_tick=0, dup=0, written=0)
per_chunk = []
for f in sorted(glob.glob(CH + 'bucket_c*.parquet')):
    ch = os.path.basename(f)[7:11]
    df = pl.read_parquet(f, columns=['match_key', 'round', 'tick', 'second'] + ALL)
    n0 = df.height
    tot['rows'] += n0
    df = df.with_columns(pl.col('match_key').replace_strict(kid, default=-1, return_dtype=pl.Int64).alias('mid'),
                         pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64), pl.col('second').cast(pl.Int64))
    nk = int((df['mid'] < 0).sum())
    tot['no_key'] += nk
    df = df.filter(pl.col('mid') >= 0)
    bucket_mid[np.unique(df['mid'].to_numpy())] = True
    J = df.join(rt, on=['mid', 'round'], how='left')
    nr = int(J['r'].is_null().sum())
    tot['no_round'] += nr
    J = J.filter(pl.col('r').is_not_null())
    bs = int(((J['second'] < 0) | (J['second'] >= J['nst'])).sum())
    tot['bad_second'] += bs
    J = J.filter((pl.col('second') >= 0) & (pl.col('second') < pl.col('nst')))
    bt = int((J['tick'] != J['frz'] + 64 * J['second']).sum())
    tot['bad_tick'] += bt
    n1 = J.height
    J = J.unique(['r', 'second'], keep='first', maintain_order=True)
    dp = n1 - J.height
    tot['dup'] += dp
    idx = (J['off'] + J['second']).to_numpy().astype(np.int64)
    assert not written[idx].any(), 'overlap between chunks'
    written[idx] = True
    a = {x: J[x].to_numpy().astype(np.float64) for x in ALL}
    out_i8[idx] = np.stack([enc_i8(a[x]) for x in INT8], 1)
    out_u16[idx] = np.stack([enc_u16(a[x]) for x in U16], 1)
    out_u8[idx] = np.stack([enc_u8(a[x]) for x in U8], 1)
    out_f16[idx] = a['defuse_slack'].astype(np.float16)
    tot['written'] += len(idx)
    per_chunk.append((ch, n0, nk, nr, bs, bt, dp, len(idx)))
    log(f'chunk {ch}: rows {n0} no_key {nk} no_round {nr} bad_second {bs} bad_tick {bt} dup {dp} written {len(idx)}')
    del df, J, a

log('bucket totals', tot, 'bucket matches', int(bucket_mid.sum()))
bucket_round = bucket_mid[rm]
lap_round = ~bucket_round
bstep = np.repeat(bucket_round, nst)
S_b = int(bstep.sum())
S_l = S - S_b
assert not (written & ~bstep).any(), 'written step outside bucket rounds'
log('bucket rounds', int(bucket_round.sum()), 'laptop-sourced rounds', int(lap_round.sum()),
    'bucket steps', S_b, 'laptop-sourced steps', S_l, 'bucket steps written', int(written.sum()),
    'bucket steps with no chunk row (sentinel)', S_b - int(written.sum()))

# even-step index maps (v2 even row <-> v4 step row)
rid_e = np.repeat(np.arange(N, dtype=np.int64), cnt)
j_e = np.arange(E, dtype=np.int64) - pos_off[rid_e]
v4_e = off[rid_e] + 2 * j_e
del j_e
v2_i8 = np.load(V2 + 'pos_int8.npy', mmap_mode='r')
v2_u16 = np.load(V2 + 'pos_uint16.npy', mmap_mode='r')
v2_u8 = np.load(V2 + 'pos_uint8.npy', mmap_mode='r')
v2_f16 = np.load(V2 + 'pos_float16.npy', mmap_mode='r')
assert v2_i8.shape == (E, 13) and v2_u16.shape == (E, 8) and v2_u8.shape == (E, 2) and v2_f16.shape == (E,)

# parity: every-second even values on bucket rounds vs v2 even values (before any cone patch)
e_b = np.nonzero(bucket_round[rid_e])[0]
v4_b = v4_e[e_b]
par = {}
A = np.asarray(v2_i8[e_b])
for i, x in enumerate(INT8):
    par[x] = int(np.count_nonzero(out_i8[v4_b, i] != A[:, i]))
del A
A = np.asarray(v2_u16[e_b])
for i, x in enumerate(U16):
    par[x] = int(np.count_nonzero(out_u16[v4_b, i] != A[:, i]))
del A
A = np.asarray(v2_u8[e_b])
for i, x in enumerate(U8):
    par[x] = int(np.count_nonzero(out_u8[v4_b, i] != A[:, i]))
del A
a16 = out_f16[v4_b]
b16 = np.asarray(v2_f16[e_b])
par['defuse_slack'] = int(np.count_nonzero(~((a16 == b16) | (np.isnan(a16) & np.isnan(b16)))))
del a16, b16
log('parity bucket even steps (every-second vs v2, pre-patch) n_even =', len(e_b), 'mismatch counts', {k: v for k, v in par.items() if v}, 'total mismatching features', sum(par.values()))

# laptop-sourced rounds: even steps copied from v2, odd steps stay sentinel
e_l = np.nonzero(lap_round[rid_e])[0]
v4_l = v4_e[e_l]
out_i8[v4_l] = np.asarray(v2_i8[e_l])
out_u16[v4_l] = np.asarray(v2_u16[e_l])
out_u8[v4_l] = np.asarray(v2_u8[e_l])
out_f16[v4_l] = np.asarray(v2_f16[e_l])
assert not written[v4_l].any()
log('laptop even steps copied from v2', len(e_l), '| laptop odd steps (sentinel)', S_l - len(e_l))

# c_exposed / sees_enemy float64 cone patch (39 cells, from fixes/cexp_f64/out/pos_v4_changes.csv)
fcol = {'c_sees_enemy': 6, 't_sees_enemy': 7, 'c_exposed': 8, 't_exposed': 9}
rt_map = {(int(m), int(r)): i for i, (m, r) in enumerate(zip(rm, rn))}
csv_rows = list(csv.DictReader(open(CEXP)))
applied = old_ok = si_pos = si_all = odd = 0
by_feat = {}
cone_in_bucket = 0
for row in csv_rows:
    mid = kid[row['match_key']]
    r = rt_map[(mid, int(row['round']))]
    tick = int(row['tick'])
    dt = tick - int(frz[r])
    assert dt % 64 == 0
    k = dt // 64
    if k % 2:
        odd += 1
        continue
    v4i = int(off[r]) + k
    v2e = int(pos_off[r]) + k // 2
    si = int(row['step_index'])
    si_pos += int(si == v2e)
    si_all += int(si == v4i)
    col = fcol[row['feature']]
    cur = int(out_i8[v4i, col])
    old_ok += int(cur == int(row['old']))
    cone_in_bucket += int(bucket_round[r])
    out_i8[v4i, col] = np.int8(int(row['new']))
    applied += 1
    by_feat[row['feature']] = by_feat.get(row['feature'], 0) + 1
log('cone csv rows', len(csv_rows), 'applied', applied, 'old value matches current', old_ok, 'odd-step rows', odd,
    'step_index equals v2 even index', si_pos, 'equals all-step index', si_all, 'by feature', by_feat, 'in bucket rounds', cone_in_bucket)

# sentinel counts by source (even bucket / odd bucket / laptop even / laptop odd)
bodd = np.zeros(S, bool)
bodd[:] = bstep
bodd[v4_e] = False  # even steps of any round are cleared; re-set below for bucket
even_mask = np.zeros(S, bool)
even_mask[v4_e] = True
bucket_even_mask = np.zeros(S, bool)
bucket_even_mask[v4_b] = True
laptop_even_mask = np.zeros(S, bool)
laptop_even_mask[v4_l] = True
bucket_odd_mask = bstep & ~even_mask
laptop_odd_mask = (~bstep) & ~even_mask
del bodd
sent = {}
for name, msk in [('bucket_even', bucket_even_mask), ('bucket_odd', bucket_odd_mask), ('laptop_even', laptop_even_mask), ('laptop_odd', laptop_odd_mask)]:
    n_msk = int(msk.sum())
    sent[name] = {'steps': n_msk,
                  'int8_sentinel_cells_per_feature': {x: int(np.count_nonzero(out_i8[msk, i] == -1)) for i, x in enumerate(INT8)},
                  'uint16_sentinel_cells_per_feature': {x: int(np.count_nonzero(out_u16[msk, i] == 65535)) for i, x in enumerate(U16)},
                  'uint8_sentinel_cells_per_feature': {x: int(np.count_nonzero(out_u8[msk, i] == 255)) for i, x in enumerate(U8)},
                  'float16_nan_cells': int(np.count_nonzero(np.isnan(out_f16[msk])))}
    log('steps', name, n_msk)
del bucket_odd_mask, laptop_odd_mask
assert S_b + S_l == S

# compare the assembled even steps with the existing export_clean_v4 top-level positions (same v2 base + cone patch)
TOPARR = {}
tp_i8 = np.load(TOP + 'pos_int8.npy', mmap_mode='r')
tp_u16 = np.load(TOP + 'pos_uint16.npy', mmap_mode='r')
tp_u8 = np.load(TOP + 'pos_uint8.npy', mmap_mode='r')
tp_f16 = np.load(TOP + 'pos_float16.npy', mmap_mode='r')
cmp_top = {}
A = np.asarray(tp_i8)
for i, x in enumerate(INT8):
    cmp_top[x] = int(np.count_nonzero(out_i8[v4_e, i] != A[:, i]))
del A
A = np.asarray(tp_u16)
for i, x in enumerate(U16):
    cmp_top[x] = int(np.count_nonzero(out_u16[v4_e, i] != A[:, i]))
del A
A = np.asarray(tp_u8)
for i, x in enumerate(U8):
    cmp_top[x] = int(np.count_nonzero(out_u8[v4_e, i] != A[:, i]))
del A
a16 = out_f16[v4_e]
b16 = np.asarray(tp_f16)
cmp_top['defuse_slack'] = int(np.count_nonzero(~((a16 == b16) | (np.isnan(a16) & np.isnan(b16)))))
del a16, b16
log('compare with export_clean_v4 top-level even positions: mismatch counts', {k: v for k, v in cmp_top.items() if v}, 'total', sum(cmp_top.values()))

# write layout
np.save(OUT + 'pos_int8.npy', out_i8)
np.save(OUT + 'pos_uint16.npy', out_u16)
np.save(OUT + 'pos_uint8.npy', out_u8)
np.save(OUT + 'pos_float16.npy', out_f16)
np.save(OUT + 'pos_offsets.npy', off.astype(np.int64))
src = np.where(bucket_mid, 1, 2).astype(np.uint8)
np.save(OUT + 'match_pos_source.npy', src)
every_keys = [keys[i] for i in np.nonzero(bucket_mid)[0]]
json.dump(every_keys, open(OUT + 'every_second_match_keys.json', 'w'))
log('arrays written', OUT)
log('match source counts: every_second', int(bucket_mid.sum()), 'laptop_even_only', int((~bucket_mid).sum()))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 24), b''):
            h.update(blk)
    return h.hexdigest()


files = {}
for nm in ['pos_int8.npy', 'pos_uint16.npy', 'pos_uint8.npy', 'pos_float16.npy', 'pos_offsets.npy',
           'match_pos_source.npy', 'every_second_match_keys.json']:
    p = OUT + nm
    files[nm] = {'bytes': os.path.getsize(p), 'sha256': sha256_file(p)}
    log(nm, files[nm]['bytes'], files[nm]['sha256'])

v2_hash = {nm: sha256_file(V2 + nm) for nm in ['match_keys.json', 'offsets.npy']}
log('v2 match_keys.json sha256', v2_hash['match_keys.json'], 'v2 offsets.npy sha256', v2_hash['offsets.npy'])

readme = {
    "name": "export_clean_v4 positions (every-second, all steps)",
    "built_on": "2026-11-05",
    "status": "Positions only. Nothing was scored, fit, or evaluated. Holdout B and Oct 26-31 keys were not read (count checks only).",
    "aligned_to": {
        "export": "export_clean_v2 (match_keys.json sha256 " + v2_hash['match_keys.json'] + ", offsets.npy sha256 " + v2_hash['offsets.npy'] + ")",
        "steps": "step s of round r = offsets[r] + k, tick = round_freeze[r] + 64*k, k = 0..n_r-1 (all seconds)",
        "matches": M, "rounds": N, "steps": S, "even_steps_v2": E,
    },
    "layout": {
        "pos_int8.npy": {"shape": [S, 13], "dtype": "int8", "columns": INT8, "sentinel": -1,
                          "encoding": "NaN -> -1; else clip(round(x), 0, 5)"},
        "pos_uint16.npy": {"shape": [S, 8], "dtype": "uint16", "columns": U16, "sentinel": 65535,
                            "encoding": "NaN -> 65535; else clip(round(x), 0, 65534); distances in game units"},
        "pos_uint8.npy": {"shape": [S, 2], "dtype": "uint8", "columns": U8, "sentinel": 255,
                           "encoding": "NaN -> 255; else clip(round(10*x), 0, 254); blindness sums in 0.1 s"},
        "pos_float16.npy": {"shape": [S], "dtype": "float16", "columns": F16, "sentinel": "nan"},
        "pos_offsets.npy": {"shape": [N + 1], "dtype": "int64",
                             "meaning": "offsets over ALL steps (identical to export_clean_v2/offsets.npy), not the even-step offsets used by v2 pos arrays"},
        "match_pos_source.npy": {"shape": [M], "dtype": "uint8", "meaning": "per v2 match index: 1 = every-second bucket build (all steps), 2 = laptop every-2 features (even steps only; odd steps sentinel)"},
        "every_second_match_keys.json": {"meaning": "match keys with every-second positions, in v2 match order", "count": len(every_keys)},
    },
    "sources": {
        "bucket_every_second": "final/pos_all_seconds/bucket_c000..c046.parquet (fixes/pos_every_second/pos_bucket_every_second_keep.py, feature module fixes/pos_every_second/feat/make_pos_features.py, float32 cone test). Rows mapped to v2 rounds by (match_key, round) and placed at offsets[r] + second.",
        "laptop_even": "export_clean_v2 pos arrays at even steps (built from laptop_pos/every2_sites_v1), copied to the even steps of laptop-sourced rounds.",
        "laptop_odd": "sentinel (no laptop every-second build on the space; see not_in_this_build).",
        "cone_patch": "39 cells from fixes/cexp_f64/out/pos_v4_changes.csv (c_sees_enemy 7, t_sees_enemy 5, c_exposed 16, t_exposed 11) set to their float64 values, as in export_clean_v4/pos_int8.npy. All 39 are even steps.",
    },
    "counts": {
        "chunk_rows_total": tot['rows'],
        "chunk_rows_no_key_(v1-only matches removed in v2)": tot['no_key'],
        "chunk_rows_no_round_(v1-only rounds dropped in v2)": tot['no_round'],
        "chunk_rows_bad_second": tot['bad_second'],
        "chunk_rows_bad_tick_vs_freeze_plus_64k": tot['bad_tick'],
        "chunk_rows_duplicate_(round,second)": tot['dup'],
        "chunk_rows_written": tot['written'],
        "bucket_matches": int(bucket_mid.sum()),
        "laptop_sourced_matches": int((~bucket_mid).sum()),
        "bucket_rounds": int(bucket_round.sum()),
        "laptop_rounds": int(lap_round.sum()),
        "bucket_steps": S_b,
        "bucket_steps_with_row": int(written.sum()),
        "bucket_steps_sentinel_no_row": S_b - int(written.sum()),
        "laptop_even_steps_copied": int(len(e_l)),
        "laptop_odd_steps_sentinel": S_l - int(len(e_l)),
        "cone_csv_rows_applied": applied,
    },
    "parity_bucket_even_pre_patch_mismatch": par,
    "compare_with_export_clean_v4_top_level_even_post_patch": cmp_top,
    "sentinels_by_source": sent,
    "chunk_log": [dict(zip(['chunk', 'rows', 'no_key', 'no_round', 'bad_second', 'bad_tick', 'dup', 'written'], list(x))) for x in per_chunk],
    "files": files,
    "known_issues": [
        "Odd steps of bucket matches come from the float32 cone test (make_pos_features.py in fixes/pos_every_second/feat). The float64 cone fix was run only for even steps; odd steps were not checked against float64.",
        "The freeze-tick roster fix (283 rounds) is not applied to these positions: the every-second and every-2 features still use the old roster sides (CHANGES_freeze_cexp.json, pos_features_not_fixed).",
        "The 2,848 keys dated 2026-10-21..25 have laptop positions in v2 (even steps). The owner check listed in export_clean_v4/README.json still applies.",
        "export_clean_v2/pos_assemble.py loaded all 61 every2_sites_v1 day files, including 2026-10-26..31. Those rows dropped out because no export key is dated then (count 0). They were not used for any output.",
    ],
    "not_in_this_build": [
        "Laptop every-second positions for the 25,793 laptop-sourced matches (odd steps). The space has only posport/data/positions_1hz/_plants; the per-second player positions (3.7 GB on the laptop at G:/pureskill/csds-cs2/data/positions_1hz, days 2026-09-01..2026-10-31) are not on the space.",
        "No model was scored, fit, or read for a label. No holdout (2026-10-26..31 or holdout B) rows were read; key counts only.",
    ],
}
json.dump(readme, open(OUT + 'README.json', 'w'), indent=1)
log('README.json written; total seconds', round(time.time() - T0))
