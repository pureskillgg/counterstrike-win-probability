import numpy as np, json, glob, os
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

E = '/home/sagemaker-user/adam/builder/final/export_clean_v2'
F = '/home/sagemaker-user/adam/builder/final'
pr = json.load(open(E + '/pos_README.json'))
keys = json.load(open(E + '/match_keys.json'))
NM = len(keys)
rm = np.load(E + '/round_match.npy').astype(np.int64)
rnum = np.load(E + '/round_num.npy').astype(np.int64)
rfz = np.load(E + '/round_freeze.npy').astype(np.int64)
poff = np.load(E + '/pos_offsets.npy').astype(np.int64)
NR = len(rm)
SE = int(poff[-1])
K = rm * 1024 + rnum
order = np.argsort(K, kind='stable')
Ks = K[order]
print('round keys: unique (match,round_num)', len(np.unique(K)) == NR, '| rnum min', int(rnum.min()), 'max', int(rnum.max()))
print('pos_offsets[-1]', SE, 'steps_even', pr['steps_even'])

# source per export match
cm = pq.read_table(F + '/clean_matches.parquet', columns=['match_key', 'src']).to_pydict()
src_of = {}
for k, s in zip(cm['match_key'], cm['src']):
    src_of.setdefault(k, set()).add(s)
kset = set(keys)
src_arr = np.empty(NM, dtype=object)
multi = 0
for i, k in enumerate(keys):
    s = src_of.get(k, set())
    if len(s) != 1:
        multi += 1
    src_arr[i] = next(iter(s)) if s else 'none'
print('export matches with src missing or multiple', multi)
is_bucket_m = np.array([s == 'bucket' for s in src_arr])
print('export matches src counts', {s: int((src_arr == s).sum()) for s in set(src_arr.tolist())})

is_bucket_r = is_bucket_m[rm]
exp_mask = np.zeros(SE + 1, dtype=np.int64)
br = np.nonzero(is_bucket_r)[0]
np.add.at(exp_mask, poff[br], 1)
np.add.at(exp_mask, poff[br + 1], -1)
exp_mask = np.cumsum(exp_mask)[:SE] > 0
print('bucket-src rounds', len(br), 'expected bucket positions', int(exp_mask.sum()))

pos_int8 = np.load(E + '/pos_int8.npy', mmap_mode='r')
pos_u16 = np.load(E + '/pos_uint16.npy', mmap_mode='r')
pos_u8 = np.load(E + '/pos_uint8.npy', mmap_mode='r')
pos_f16 = np.load(E + '/pos_float16.npy', mmap_mode='r')

feat = []  # (name, group, col_index_in_group, dtype, sentinel)
for j, n in enumerate(pr['int8']): feat.append((n, 'i8', j, np.int8, -1))
for j, n in enumerate(pr['uint16']): feat.append((n, 'u16', j, np.uint16, 65535))
for j, n in enumerate(pr['uint8_tenth_seconds']): feat.append((n, 'u8', j, np.uint8, 255))
for j, n in enumerate(pr['float16']): feat.append((n, 'f16', j, np.float16, float('nan')))
mism = {f[0]: 0 for f in feat}
ex_example = {}

counts = np.zeros(SE, dtype=np.int32)
tot_rows = 0
rows_not_in_export = 0
rows_not_in_export_keys = set()
rows_round_missing = 0
rows_off_grid = 0
rows_odd = 0
rows_out_of_round = 0
rows_neg = 0
files = sorted(glob.glob(F + '/pos_parts/bucket_c*.parquet'))
schema_printed = False
for p in files:
    t = pq.read_table(p)
    if not schema_printed:
        print('pos_parts schema sample', [(f.name, str(f.type)) for f in t.schema][:6], '...', len(t.schema.names), 'cols')
        print('null counts file0 (first 5 feats)', {n: t.column(n).null_count for n in pr['int8'][:3] + ['defuse_slack']})
        schema_printed = True
    tot_rows += t.num_rows
    mk = t.column('match_key')
    idx_arr = pc.index_in(mk, value_set=pa.array(keys))
    found = pc.is_valid(idx_arr).to_numpy(zero_copy_only=False)
    idx = idx_arr.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
    nf = ~found
    rows_not_in_export += int(nf.sum())
    if nf.any():
        for k in set(mk.filter(pa.array(nf)).to_pylist()):
            rows_not_in_export_keys.add(k)
    sel = np.nonzero(found)[0]
    if len(sel) == 0:
        continue
    midx = idx[sel]
    rnd = t.column('round').to_numpy(zero_copy_only=False)[sel].astype(np.int64)
    tick = t.column('tick').to_numpy(zero_copy_only=False)[sel].astype(np.int64)
    rk = midx * 1024 + rnd
    pos = np.searchsorted(Ks, rk)
    pos = np.clip(pos, 0, NR - 1)
    okr = Ks[pos] == rk
    rows_round_missing += int((~okr).sum())
    sel2 = np.nonzero(okr)[0]
    ridx = order[pos[sel2]]
    d = tick[sel2] - rfz[ridx]
    rows_neg += int((d < 0).sum())
    rows_off_grid += int((d % 64 != 0).sum())
    rows_odd += int(((d % 64 == 0) & ((d // 64) % 2 == 1)).sum())
    pidx = poff[ridx] + d // 128
    inr = (pidx >= poff[ridx]) & (pidx < poff[ridx + 1])
    rows_out_of_round += int((~inr).sum())
    good = (d >= 0) & (d % 128 == 0) & inr
    pidx = pidx[good]
    np.add.at(counts, pidx, 1)
    # value comparison for good rows
    rows_sel = sel[sel2][good]
    for name, grp, j, dt, sent in feat:
        col = t.column(name)
        vals_arr = pc.fill_null(col, sent) if grp != 'f16' else pc.fill_null(col, float('nan'))
        pv = vals_arr.to_numpy(zero_copy_only=False)[rows_sel]
        if grp == 'i8':
            ev = np.asarray(pos_int8[pidx])[:, j]
        elif grp == 'u16':
            ev = np.asarray(pos_u16[pidx])[:, j]
        elif grp == 'u8':
            ev = np.asarray(pos_u8[pidx])[:, j]
        else:
            ev = np.asarray(pos_f16[pidx])
        pv = np.asarray(pv, dtype=np.float64)
        ev = np.asarray(ev, dtype=np.float64)
        neq = ~((pv == ev) | (np.isnan(pv) & np.isnan(ev)))
        mism[name] += int(neq.sum())
        if neq.any() and name not in ex_example:
            i0 = int(np.nonzero(neq)[0][0])
            ex_example[name] = (float(pv[i0]), float(ev[i0]))
    print(os.path.basename(p), 'rows', t.num_rows, 'found', int(found.sum()), 'good', int(good.sum()), flush=True)

print('\n== pos_parts totals')
print('pos_parts rows total', tot_rows)
print('rows whose match_key not in export', rows_not_in_export, 'distinct keys', len(rows_not_in_export_keys))
print('rows with (match,round) not in export rounds', rows_round_missing)
print('rows with tick < freeze', rows_neg, '| tick off 64-step grid', rows_off_grid, '| odd step index', rows_odd, '| outside round range', rows_out_of_round)
print('\n== coverage of bucket-src export positions')
cov = counts[exp_mask] > 0
print('expected bucket positions', int(exp_mask.sum()), 'covered', int(cov.sum()), 'missing', int((~cov).sum()))
print('duplicate hits on bucket positions', int((counts[exp_mask] > 1).sum()))
print('hits on non-bucket (laptop) positions', int(counts[~exp_mask].sum()))
print('total hits (should equal good rows)', int(counts.sum()))
print('\n== value agreement pos_parts vs export (bucket rows)')
for name, grp, j, dt, sent in feat:
    print('%-22s mismatches=%d example=%s' % (name, mism[name], ex_example.get(name)))
print('ALL FEATURE MISMATCHES ZERO:', all(v == 0 for v in mism.values()))
