import json, re, collections, glob, os
import pyarrow.parquet as pq

ROOT = '/home/sagemaker-user'
MAN2 = ROOT + '/adam/builder/final/split_manifest_v2.json'
OUT = ROOT + '/adam/builder/fixes/check_unchecked_oct/oct_check_caches_result.json'
KEY_RE = re.compile(r'csds/(\d{4})/(\d{2})/(\d{2})/')


def dk(k):
    m = KEY_RE.search(str(k))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def bucket(d):
    if d is None:
        return 'nodate'
    if d < '2026-10-14':
        return 'before_oct14'
    if d <= '2026-10-20':
        return 'oct14_20'
    if d <= '2026-10-25':
        return 'oct21_25'
    if d <= '2026-10-31':
        return 'oct26_31'
    return 'after_oct31'


R = {'caches': {}, 'manifest': {}}

# A. every parquet cache under the working trees: read the match_key column only
found = []
for sub in ['csds-cs2', 'adam', 'pipe']:
    base = os.path.join(ROOT, sub)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in ('.git', '__pycache__', 'node_modules')]
        for fn in filenames:
            if fn.endswith('.parquet') and ('cache' in fn or 'oct' in fn.lower() or 'ext' in fn or 'pos' in fn):
                found.append(os.path.join(dirpath, fn))
found = sorted(set(found))
print('candidate parquet files', len(found), flush=True)
for f in found:
    try:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        if 'match_key' not in names:
            R['caches'][f] = {'skipped': 'no match_key column', 'cols_head': names[:6]}
            continue
        t = pq.read_table(f, columns=['match_key']).column('match_key').to_pylist()
        keys = set(str(x) for x in t)
        cnt = collections.Counter(bucket(dk(k)) for k in keys)
        ds = sorted(d for d in (dk(k) for k in keys) if d)
        R['caches'][f] = {
            'n_matches': len(keys),
            'min_date': ds[0] if ds else None,
            'max_date': ds[-1] if ds else None,
            'buckets': dict(cnt),
            'size_bytes': os.path.getsize(f),
        }
        print(f, json.dumps(R['caches'][f]), flush=True)
    except Exception as e:
        R['caches'][f] = {'error': type(e).__name__, 'msg': str(e)[:200]}
        print(f, 'ERR', type(e).__name__, str(e)[:160], flush=True)

# B. manifest v2 totals by date window and label (keys and dates only)
m2 = json.load(open(MAN2))
a2 = m2['assign']
per = collections.Counter()
per_label = collections.Counter()
for k, lab in a2.items():
    b = bucket(dk(k))
    per[b] += 1
    per_label[(b, lab)] += 1
R['manifest']['buckets_all_labels'] = dict(per)
R['manifest']['bucket_label'] = {f'{b}|{l}': v for (b, l), v in sorted(per_label.items())}
oct14_25 = [k for k in a2 if dk(k) and '2026-10-14' <= dk(k) <= '2026-10-25']
R['manifest']['oct14_25_total'] = len(oct14_25)
R['manifest']['oct14_25_by_label'] = dict(collections.Counter(a2[k] for k in oct14_25))
dates_by_day = collections.Counter(dk(k) for k in a2 if dk(k) and '2026-10-10' <= dk(k) <= '2026-10-31')
R['manifest']['matches_per_day_oct10_31'] = dict(sorted(dates_by_day.items()))
print('manifest buckets', R['manifest']['buckets_all_labels'], flush=True)
print('manifest oct14_25 total', len(oct14_25), R['manifest']['oct14_25_by_label'], flush=True)
print('manifest per day', R['manifest']['matches_per_day_oct10_31'], flush=True)

with open(OUT, 'w') as f:
    json.dump(R, f, indent=1, default=str)
print('WROTE', OUT, flush=True)
