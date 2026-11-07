import json, re, glob, collections
import numpy as np
import polars as pl

V2 = '/home/sagemaker-user/adam/builder/final/export_clean_v2/'
FIN = '/home/sagemaker-user/adam/builder/final/'
BUILDER = '/home/sagemaker-user/adam/builder/'

keys = json.load(open(V2 + 'match_keys.json'))
kid = {k: i for i, k in enumerate(keys)}


def kdate(k):
    p = k.split('/')
    return f"{p[1]}-{p[2]}-{p[3]}"


dates = [kdate(k) for k in keys]
print('export keys', len(keys), 'unique', len(kid))
months = collections.Counter(d[:7] for d in dates)
print('export keys by month', sorted(months.items()))
print('export keys dated 2026-10-26..2026-10-31 (count only)', sum(1 for d in dates if '2026-10-26' <= d <= '2026-10-31'))
print('export keys dated 2026-10-21..2026-10-25 (count only)', sum(1 for d in dates if '2026-10-21' <= d <= '2026-10-25'))

hb_lines = [l.strip() for l in open(FIN + 'holdout_b_keys.txt') if l.strip()]
hb = set(hb_lines)
print('holdout_b_keys.txt lines', len(hb_lines), 'unique', len(hb))
print('holdout_b keys found in export keys (count only)', sum(1 for k in hb if k in kid))

off = np.load(V2 + 'offsets.npy')
rm = np.load(V2 + 'round_match.npy')
rn = np.load(V2 + 'round_num.npy').astype(np.int64)
frz = np.load(V2 + 'round_freeze.npy').astype(np.int64)
nst = np.diff(off)
print('rounds', len(rm), 'total steps', int(off[-1]), 'sum nst', int(nst.sum()))
steps_per_match = np.bincount(rm, weights=nst, minlength=len(keys)).astype(np.int64)
print('total steps over matches', int(steps_per_match.sum()))
pairs = set(zip(rm.tolist(), rn.tolist()))
print('unique (match, round_num) pairs', len(pairs), 'of rounds', len(rm))

# every-second bucket chunks: keys, rows, and per-chunk checks
log = open('/home/sagemaker-user/adam/builder/fixes/pos_every_second/keep_run.log').read()
logrows = {}
logmatches = {}
for m in re.finditer(r'^(c\d{3}) matches (\d+) skipped (\d+) rows (\d+) even (\d+) odd (\d+)', log, re.M):
    logmatches[m.group(1)] = int(m.group(2))
    logrows[m.group(1)] = (int(m.group(4)), int(m.group(5)), int(m.group(6)))
print('keep_run.log chunk lines parsed', len(logrows), 'sum matches', sum(logmatches.values()),
      'sum rows', sum(v[0] for v in logrows.values()))

bucket_keys = set()
rows_total = 0
exp_steps_total = 0
mismatch_chunks = []
for f in sorted(glob.glob(FIN + 'pos_all_seconds/bucket_c*.parquet')):
    c = f.split('bucket_')[1][:4]
    df = pl.read_parquet(f, columns=['match_key', 'round', 'tick', 'second'])
    n = df.height
    rows_total += n
    mk_u = df['match_key'].unique().to_list()
    bucket_keys |= set(mk_u)
    ev = int((df['second'] % 2 == 0).sum())
    od = n - ev
    exp = sum(int(steps_per_match[kid[k]]) for k in mk_u if k in kid)
    exp_steps_total += exp
    lr = logrows.get(c)
    ok_log = lr is not None and lr[0] == n and lr[1] == ev and lr[2] == od
    ok_m = logmatches.get(c) == len(mk_u)
    ok_exp = exp == n
    if not (ok_log and ok_m and ok_exp):
        mismatch_chunks.append((c, n, ev, od, exp, lr, len(mk_u), logmatches.get(c)))
    print(c, 'rows', n, 'even', ev, 'odd', od, 'matches', len(mk_u), 'expected_steps', exp,
          'log_ok', ok_log, 'matches_ok', ok_m, 'steps_ok', ok_exp)
print('chunks read', len(glob.glob(FIN + 'pos_all_seconds/bucket_c*.parquet')))
print('chunk rows total', rows_total, 'expected steps total', exp_steps_total)
print('chunks with any mismatch vs keep_run.log or expected steps', len(mismatch_chunks), mismatch_chunks[:5])
print('bucket unique match keys', len(bucket_keys), 'not in export', len(bucket_keys - set(kid)))
print('export keys with every-second rows', sum(1 for k in keys if k in bucket_keys))
laptop_keys = [k for k in keys if k not in bucket_keys]
print('export keys without every-second rows (laptop-sourced or other)', len(laptop_keys))
print('laptop-sourced steps (count)', int(sum(steps_per_match[kid[k]] for k in laptop_keys)))
print('laptop-sourced keys dated 2026-10-26..31 (count only)', sum(1 for k in laptop_keys if '2026-10-26' <= kdate(k) <= '2026-10-31'))
print('bucket keys dated 2026-10-26..31 (count only)', sum(1 for k in bucket_keys if '2026-10-26' <= kdate(k) <= '2026-10-31'))
print('bucket keys dated 2026-10-21..25 (count only)', sum(1 for k in bucket_keys if '2026-10-21' <= kdate(k) <= '2026-10-25'))
print('bucket keys found in holdout_b (count only)', len(bucket_keys & hb))
print('laptop keys found in holdout_b (count only)', sum(1 for k in laptop_keys if k in hb))
print('bucket keys by month', sorted(collections.Counter(kdate(k)[:7] for k in bucket_keys).items()))
print('laptop keys by month', sorted(collections.Counter(kdate(k)[:7] for k in laptop_keys).items()))
