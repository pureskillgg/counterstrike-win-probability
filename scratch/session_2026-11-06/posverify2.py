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

cm = pq.read_table(F + '/clean_matches.parquet', columns=['match_key', 'src']).to_pydict()
src_of = {}
for k, s in zip(cm['match_key'], cm['src']):
    src_of.setdefault(k, set()).add(s)
is_bucket_m = np.array([src_of.get(k, set()) == {'bucket'} for k in keys])
is_bucket_r = is_bucket_m[rm]
exp_mask = np.zeros(SE + 1, dtype=np.int64)
br = np.nonzero(is_bucket_r)[0]
np.add.at(exp_mask, poff[br], 1)
np.add.at(exp_mask, poff[br + 1], -1)
exp_mask = np.cumsum(exp_mask)[:SE] > 0

pos_int8 = np.load(E + '/pos_int8.npy', mmap_mode='r')
pos_u16 = np.load(E + '/pos_uint16.npy', mmap_mode='r')
pos_u8 = np.load(E + '/pos_uint8.npy', mmap_mode='r')
pos_f16 = np.load(E + '/pos_float16.npy', mmap_mode='r')

feat = []
for j, n in enumerate(pr['int8']): feat.append((n, 'i8', j))
for j, n in enumerate(pr['uint16']): feat.append((n, 'u16', j))
for j, n in enumerate(pr['uint8_tenth_seconds']): feat.append((n, 'u8', j))
for j, n in enumerate(pr['float16']): feat.append((n, 'f16', j))
mism = {f[0]: 0 for f in feat}
mism_floor = {f[0]: 0 for f in feat}
ex_example = {}

counts = np.zeros(SE, dtype=np.int32)
tot_rows = 0
rows_not_export_key = 0
miss_keys_only = set()
missing_pairs = set()
missing_pair_rows = 0
rows_off_grid = rows_odd = rows_neg = rows_out = 0
files = sorted(glob.glob(F + '/pos_parts/bucket_c*.parquet'))
for p in files:
    t = pq.read_table(p)
    tot_rows += t.num_rows
    mk = t.column('match_key')
    idx_arr = pc.index_in(mk, value_set=pa.array(keys))
    found = pc.is_valid(idx_arr).to_numpy(zero_copy_only=False)
    idx = idx_arr.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
    nf = ~found
    rows_not_export_key += int(nf.sum())
    if nf.any():
        miss_keys_only |= set(mk.filter(pa.array(nf)).to_pylist())
    sel = np.nonzero(found)[0]
    if len(sel) == 0:
        continue
    midx = idx[sel]
    rnd = t.column('round').to_numpy(zero_copy_only=False)[sel].astype(np.int64)
    tick = t.column('tick').to_numpy(zero_copy_only=False)[sel].astype(np.int64)
    rk = midx * 1024 + rnd
    pos = np.clip(np.searchsorted(Ks, rk), 0, NR - 1)
    okr = Ks[pos] == rk
    if (~okr).any():
        bad = np.nonzero(~okr)[0]
        missing_pair_rows += len(bad)
        missing_pairs |= set(zip(midx[bad].tolist(), rnd[bad].tolist()))
    sel2 = np.nonzero(okr)[0]
    ridx = order[pos[sel2]]
    d = tick[sel2] - rfz[ridx]
    rows_neg += int((d < 0).sum())
    rows_off_grid += int((d % 64 != 0).sum())
    rows_odd += int(((d % 64 == 0) & ((d // 64) % 2 == 1)).sum())
    pidx = poff[ridx] + d // 128
    inr = (pidx >= poff[ridx]) & (pidx < poff[ridx + 1])
    rows_out += int((~inr).sum())
    good = (d >= 0) & (d % 128 == 0) & inr
    pidx = pidx[good]
    np.add.at(counts, pidx, 1)
    rows_sel = sel[sel2][good]
    for name, grp, j in feat:
        col = t.column(name)
        nm = pc.is_null(col).to_numpy(zero_copy_only=False)[rows_sel]
        raw = pc.fill_null(col, 0).to_numpy(zero_copy_only=False)[rows_sel].astype(np.float64)
        if grp == 'i8':
            exp = np.where(nm, -1, np.rint(raw)); exp_f = exp
            ev = np.asarray(pos_int8[pidx])[:, j].astype(np.float64)
        elif grp == 'u16':
            exp = np.where(nm, 65535, np.rint(raw)); exp_f = np.where(nm, 65535, np.floor(raw + 0.5))
            ev = np.asarray(pos_u16[pidx])[:, j].astype(np.float64)
        elif grp == 'u8':
            exp = np.where(nm, 255, np.rint(raw * 10)); exp_f = np.where(nm, 255, np.floor(raw * 10 + 0.5))
            ev = np.asarray(pos_u8[pidx])[:, j].astype(np.float64)
        else:
            exp = np.where(nm, np.nan, raw.astype(np.float16).astype(np.float64)); exp_f = exp
            ev = np.asarray(pos_f16[pidx]).astype(np.float64)
        neq = ~((exp == ev) | (np.isnan(exp) & np.isnan(ev)))
        neq_f = ~((exp_f == ev) | (np.isnan(exp_f) & np.isnan(ev)))
        mism[name] += int(neq.sum())
        mism_floor[name] += int(neq_f.sum())
        if neq.any() and name not in ex_example:
            i0 = int(np.nonzero(neq)[0][0])
            ex_example[name] = (float(raw[i0]), bool(nm[i0]), float(exp[i0]), float(ev[i0]))
    print(os.path.basename(p), 'rows', t.num_rows, 'found', int(found.sum()), 'good', int(good.sum()), flush=True)

found_rows = tot_rows - rows_not_export_key
print('\n== row accounting')
print('pos_parts rows total', tot_rows)
print('rows whose match_key not in export (distinct keys %d)' % len(miss_keys_only), rows_not_export_key)
print('rows in export matches but (match,round) absent from export rounds', missing_pair_rows, '| distinct (match,round) pairs', len(missing_pairs))
print('matched rows used for coverage', int(counts.sum()))
print('tick<freeze', rows_neg, '| off 64-grid', rows_off_grid, '| odd step idx', rows_odd, '| outside round', rows_out)
print('\n== coverage of bucket-src export positions')
exp_n = int(exp_mask.sum())
cov_mask = counts > 0
print('expected bucket positions', exp_n, '| covered', int((cov_mask & exp_mask).sum()), '| missing', int((exp_mask & ~cov_mask).sum()))
print('duplicate hits on bucket positions', int((counts[exp_mask] > 1).sum()), '| hits outside bucket positions (laptop)', int(counts[~exp_mask].sum()))
miss_pos = np.nonzero(exp_mask & ~cov_mask)[0]
mr = np.searchsorted(poff, miss_pos, side='right') - 1
print('missing positions in distinct rounds', len(np.unique(mr)))
c_site_hi_at_missing = np.asarray(pos_int8[miss_pos])[:, 0]
print('missing positions with export c_site_hi == -1', int((c_site_hi_at_missing == -1).sum()), 'of', len(miss_pos))
all_int8_sent = np.all(np.asarray(pos_int8[miss_pos]) == -1, axis=1)
print('missing positions with ALL 13 int8 == -1', int(all_int8_sent.sum()))
print('\n== value agreement pos_parts vs export (bucket rows, after representation transform)')
for name, grp, j in feat:
    print('%-22s mismatches(rint)=%d mismatches(floor+0.5)=%d example(raw,null,exp,export)=%s' % (name, mism[name], mism_floor[name], ex_example.get(name)))
print('ALL FEATURE MISMATCHES ZERO (rint):', all(v == 0 for v in mism.values()))
