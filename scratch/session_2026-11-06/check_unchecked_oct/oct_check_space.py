import json, re, collections, sys, os
import pyarrow.parquet as pq

ROOT = '/home/sagemaker-user'
CACHE = ROOT + '/csds-cs2/data/combined/cache_snapshot_2026.parquet'
MAN2 = ROOT + '/adam/builder/final/split_manifest_v2.json'
MAN1 = ROOT + '/adam/builder/final/split_manifest.json'
EXCL = ROOT + '/adam/builder/final/export_clean_v4/TRAINING_EXCLUDE.json'
HB_TXT = ROOT + '/adam/builder/final/holdout_b_keys.txt'
HB_JSON = ROOT + '/adam_holdout_b_keys.json'
OUT = ROOT + '/adam/builder/fixes/check_unchecked_oct/oct_check_space_result.json'

KEY_RE = re.compile(r'csds/(\d{4})/(\d{2})/(\d{2})/')


def dk(k):
    m = KEY_RE.search(str(k))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def win(d):
    return d is not None and '2026-10-21' <= d <= '2026-10-25'


def summ(name, keys):
    keys = set(keys)
    dates = collections.Counter(dk(k) for k in keys)
    ds = sorted(x for x in dates if x)
    out = {
        'n_keys': len(keys),
        'min_date': ds[0] if ds else None,
        'max_date': ds[-1] if ds else None,
        'n_oct21_25': sum(1 for k in keys if win(dk(k))),
        'n_oct26_31': sum(1 for k in keys if dk(k) and '2026-10-26' <= dk(k) <= '2026-10-31'),
        'n_on_or_after_oct26': sum(1 for k in keys if dk(k) and dk(k) >= '2026-10-26'),
        'n_no_date': dates.get(None, 0),
    }
    print(name, json.dumps(out), flush=True)
    return keys, out


R = {}

# 1. cache snapshot used by sprint_prepare (TRAIN_CACHE)
pf = pq.ParquetFile(CACHE)
names = pf.schema_arrow.names
print('cache schema names', names, 'rows', pf.metadata.num_rows, flush=True)
t = pq.read_table(CACHE, columns=[c for c in ['match_key', 'match_date', 'era'] if c in names]).to_pydict()
mk = t['match_key']
cache_keys_all = set(str(x) for x in mk)
cache_keys, R['cache'] = summ('cache_snapshot_2026 (match_key)', cache_keys_all)
if 'match_date' in t:
    md = [str(x) for x in t['match_date']]
    R['cache_match_date_min_max'] = [min(md), max(md)]
    mismatch = sum(1 for k, d in zip(mk, md) if dk(k) and str(d)[:10].replace('/', '-') != dk(k))
    R['cache_match_date_vs_key_mismatch_rows'] = mismatch
    print('cache match_date min/max', min(md), max(md), 'row mismatches vs key date', mismatch, flush=True)
if 'era' in t:
    R['cache_era_counts'] = dict(collections.Counter(str(x) for x in t['era']))
    print('cache era counts', R['cache_era_counts'], flush=True)
R['cache_rows'] = len(mk)

# 2. split manifest v2
m2 = json.load(open(MAN2))
print('manifest v2 top keys', list(m2.keys()), flush=True)
assign2 = m2.get('assign', {})
lab_counts = collections.Counter(assign2.values())
R['man2_label_counts'] = dict(lab_counts)
print('manifest v2 label counts', dict(lab_counts), flush=True)
for lab in sorted(lab_counts):
    keys_l = [k for k, v in assign2.items() if v == lab]
    _, R['man2_' + lab] = summ('man2 label=' + lab, keys_l)
man2_keys = set(assign2.keys())
R['man2_meta'] = {k: (v if not isinstance(v, (dict, list)) or len(json.dumps(v)) < 400 else '<omitted large>') for k, v in m2.items() if k != 'assign'}
print('manifest v2 meta', json.dumps(R['man2_meta'])[:1500], flush=True)
R['cache_vs_man2'] = {
    'cache_not_in_man2': len(cache_keys - man2_keys),
    'man2_not_in_cache': len(man2_keys - cache_keys),
}
print('cache vs man2', R['cache_vs_man2'], flush=True)

# 3. split manifest v1 (first version, copy in final/)
try:
    m1 = json.load(open(MAN1))
    print('manifest v1 top keys', list(m1.keys()), flush=True)
    a1 = m1.get('assign', {})
    R['man1_label_counts'] = dict(collections.Counter(a1.values()))
    print('manifest v1 label counts', R['man1_label_counts'], flush=True)
    R['man1_oct21_25_by_label'] = dict(collections.Counter(v for k, v in a1.items() if win(dk(k))))
    print('manifest v1 oct21-25 by label', R['man1_oct21_25_by_label'], flush=True)
except Exception as e:
    print('manifest v1 ERR', type(e).__name__, str(e)[:200], flush=True)

# 4. training exclusion list (export v4, batch v4_s3_protected)
ex = json.load(open(EXCL))
print('excl top keys', [k for k in ex.keys() if k != 'match_keys'], flush=True)
ex_keys, R['excl'] = summ('TRAINING_EXCLUDE match_keys', ex['match_keys'])
R['excl_stated_n'] = ex.get('n_matches')
R['excl_split_counts_stated'] = ex.get('split_counts')
R['excl_in_man2'] = len(ex_keys & man2_keys)
R['excl_man2_label_counts'] = dict(collections.Counter(assign2.get(k) for k in ex_keys))
R['excl_in_cache'] = len(ex_keys & cache_keys)
print('excl in man2', R['excl_in_man2'], 'by label', R['excl_man2_label_counts'], 'in cache', R['excl_in_cache'], flush=True)

# 5. all Oct 21-25 keys in manifest v2 vs exclusion list
oct_man2 = {k for k in man2_keys if win(dk(k))}
R['oct21_25_in_man2'] = len(oct_man2)
R['oct21_25_in_man2_not_in_excl'] = len(oct_man2 - ex_keys)
R['oct21_25_in_man2_label_counts'] = dict(collections.Counter(assign2[k] for k in oct_man2))
print('oct21-25 in man2', len(oct_man2), 'not in excl', len(oct_man2 - ex_keys), 'labels', R['oct21_25_in_man2_label_counts'], flush=True)
oct_cache = {k for k in cache_keys if win(dk(k))}
R['oct21_25_in_cache'] = len(oct_cache)
print('oct21-25 in cache', len(oct_cache), flush=True)

# 6. holdout B keys (text list, dates only)
hb_lines = [l.strip() for l in open(HB_TXT) if l.strip()]
hb_keys, R['holdB_txt'] = summ('holdout_b_keys.txt', hb_lines)
R['holdB_txt_lines'] = len(hb_lines)
try:
    hbj = json.load(open(HB_JSON))
    hbj_keys_raw = hbj.get('match_keys', hbj) if isinstance(hbj, dict) else hbj
    hbj_keys, R['holdB_json'] = summ('adam_holdout_b_keys.json', hbj_keys_raw)
    R['holdB_json_top'] = list(hbj.keys()) if isinstance(hbj, dict) else 'list'
    print('holdB json top keys', R['holdB_json_top'], flush=True)
except Exception as e:
    print('holdB json ERR', type(e).__name__, str(e)[:200], flush=True)
    hbj_keys = set()
R['holdB_oct21_25_overlap_with_excl'] = len(set(hb_keys) & ex_keys)
R['holdB_oct21_25_overlap_with_man2'] = len(set(hb_keys) & man2_keys)
R['holdB_overlap_with_cache'] = len(set(hb_keys) & cache_keys)
R['holdB_in_man2_label_counts'] = dict(collections.Counter(assign2.get(k) for k in hb_keys))
print('holdB overlap excl', R['holdB_oct21_25_overlap_with_excl'], 'man2', R['holdB_oct21_25_overlap_with_man2'],
      'cache', R['holdB_overlap_with_cache'], 'holdB in man2 by label', R['holdB_in_man2_label_counts'], flush=True)

# 7. export v4 README (owner checks, dates)
try:
    rd = json.load(open(ROOT + '/adam/builder/final/export_clean_v4/README.json'))
    R['export_v4_readme_keys'] = list(rd.keys())
    R['export_v4_owner_checks_open'] = rd.get('owner_checks_open')
    print('export v4 README keys', R['export_v4_readme_keys'], flush=True)
    print('owner_checks_open', json.dumps(rd.get('owner_checks_open'))[:1200], flush=True)
except Exception as e:
    print('readme ERR', type(e).__name__, str(e)[:200], flush=True)

with open(OUT, 'w') as f:
    json.dump(R, f, indent=1, default=str)
print('WROTE', OUT, flush=True)
