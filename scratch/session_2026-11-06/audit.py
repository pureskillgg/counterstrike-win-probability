import numpy as np, json, os, re, subprocess
from collections import Counter

E = '/home/sagemaker-user/adam/builder/final/export_clean_v2'
F = '/home/sagemaker-user/adam/builder/final'
def P(x): return os.path.join(E, x)

rd = json.load(open(P('README.json')))
pr = json.load(open(P('pos_README.json')))
maps = json.load(open(P('maps.json')))
names = json.load(open(P('loadout_names.json')))
keys = json.load(open(P('match_keys.json')))

print('== README / JSON basics')
print('README matches', rd['matches'], 'rounds', rd['rounds'], 'steps', rd['steps'])
print('pos_README steps_even', pr['steps_even'], 'rounds', pr['rounds'])
print('match_keys.json len', len(keys))
print('maps.json', maps, 'n=', len(maps))
print('loadout_names len', len(names), 'CT_', sum(n.startswith('CT_') for n in names), 'T_', sum(n.startswith('T_') for n in names))
print('pos_README feature counts int8', len(pr['int8']), 'uint16', len(pr['uint16']), 'uint8', len(pr['uint8_tenth_seconds']), 'float16', len(pr['float16']))

ms = np.load(P('match_split.npy'))
rm = np.load(P('round_match.npy'))
rmap = np.load(P('round_map.npy'))
off = np.load(P('offsets.npy'))
poff = np.load(P('pos_offsets.npy'))
NR = len(rm)
NM = len(ms)

print('\n== (2) lengths')
print('round arrays len', NR, 'README rounds', rd['rounds'], 'equal', NR == rd['rounds'])
print('match_split len', NM, 'README matches', rd['matches'], 'equal', NM == rd['matches'])
print('offsets len', len(off), 'rounds+1', NR + 1, 'equal', len(off) == NR + 1)
print('pos_offsets len', len(poff), 'rounds+1', NR + 1, 'equal', len(poff) == NR + 1)
pos_int8 = np.load(P('pos_int8.npy'), mmap_mode='r')
pos_u16 = np.load(P('pos_uint16.npy'), mmap_mode='r')
pos_u8 = np.load(P('pos_uint8.npy'), mmap_mode='r')
pos_f16 = np.load(P('pos_float16.npy'), mmap_mode='r')
SE = pr['steps_even']
print('pos_int8', pos_int8.shape, pos_int8.dtype, 'rows==steps_even', pos_int8.shape[0] == SE, 'cols==13', pos_int8.shape[1] == len(pr['int8']))
print('pos_uint16', pos_u16.shape, pos_u16.dtype, 'rows==steps_even', pos_u16.shape[0] == SE, 'cols==8', pos_u16.shape[1] == len(pr['uint16']))
print('pos_uint8', pos_u8.shape, pos_u8.dtype, 'rows==steps_even', pos_u8.shape[0] == SE, 'cols==2', pos_u8.shape[1] == len(pr['uint8_tenth_seconds']))
print('pos_float16', pos_f16.shape, pos_f16.dtype, 'len==steps_even', pos_f16.shape[0] == SE, 'features==1', pos_f16.ndim == 1)

print('\n== (1) steps arithmetic')
steps_r = np.diff(off)
print('offsets[0]', int(off[0]), 'offsets[-1]', int(off[-1]), 'monotonic', bool(np.all(steps_r >= 0)))
total_steps = int(steps_r.sum())
odd_rounds = int((steps_r % 2 == 1).sum())
ceil_sum = int(((steps_r + 1) // 2).sum())
print('sum(steps_r) from offsets', total_steps, '| README steps', rd['steps'], '| equal', total_steps == rd['steps'])
print('odd-step rounds (observed)', odd_rounds, '| fraction of rounds', round(odd_rounds / NR, 6))
print('sum(ceil(steps_r/2))', ceil_sum, '| pos_README steps_even', SE, '| equal', ceil_sum == SE)
print('derived odd-step count = 2*steps_even - steps', 2 * SE - rd['steps'])
print('check: (steps + odd)/2 =', (total_steps + odd_rounds) // 2)
pdiff = np.diff(poff)
print('pos_offsets[0]', int(poff[0]), 'pos_offsets[-1]', int(poff[-1]), 'last==steps_even', int(poff[-1]) == SE, 'monotonic', bool(np.all(pdiff >= 0)))
bad = np.nonzero(pdiff != (steps_r + 1) // 2)[0]
print('rounds where pos diff != ceil(steps_r/2):', len(bad))

print('\n== (3) zero-step rounds')
zs = int((steps_r == 0).sum())
print('rounds with zero stored steps (offsets diff==0):', zs)
print('rounds with zero pos steps (pos diff==0):', int((pdiff == 0).sum()))
print('steps_r min', int(steps_r.min()), 'max', int(steps_r.max()), 'mean', round(float(steps_r.mean()), 4))
if zs:
    zi = np.nonzero(steps_r == 0)[0][:5]
    print('first zero-step round indices', zi.tolist(), 'their match idx', rm[zi].tolist())
rpm = np.bincount(rm, minlength=NM)
print('rm min', int(rm.min()), 'rm max', int(rm.max()), 'in range', bool(rm.min() >= 0 and rm.max() < NM))
print('matches with zero rounds', int((rpm == 0).sum()), 'rounds per match min', int(rpm.min()), 'max', int(rpm.max()))
print('rm non-decreasing', bool(np.all(np.diff(rm) >= 0)))

print('\n== (4) split counts')
u, c = np.unique(ms, return_counts=True)
print('match_split code counts', dict(zip(u.tolist(), c.tolist())))
split_names = ['train', 'val', 'test']
exp_matches = {'train': 47501, 'val': 4633, 'test': 5677}
exp_rounds = {'train': 871748, 'val': 83671, 'test': 103271}
code_counts = dict(zip(u.tolist(), c.tolist()))
rs = ms[rm]
rcount = np.bincount(rs.astype(np.int64), minlength=3)
for i, n in enumerate(split_names):
    mc = code_counts.get(i, 0)
    print(f'{n:5s} code={i} matches={mc} expected={exp_matches[n]} ok={mc == exp_matches[n]} | rounds={int(rcount[i])} expected={exp_rounds[n]} ok={int(rcount[i]) == exp_rounds[n]}')
print('total rounds via split', int(rcount.sum()), 'total matches', NM)

print('\n== (5) maps')
print('round_map unique', np.unique(rmap).tolist(), '| max < 7', int(rmap.max()) < len(maps))
pair = rm.astype(np.int64) * 8 + rmap.astype(np.int64)
upair = np.unique(pair)
matches_with_rounds = int((rpm > 0).sum())
print('unique (match,map) pairs', len(upair), '| matches with rounds', matches_with_rounds, '| one map per match:', len(upair) == matches_with_rounds)
match_map = np.full(NM, -1, dtype=np.int64)
match_map[upair // 8] = upair % 8
rnd_tab = np.bincount(rs.astype(np.int64) * 8 + rmap.astype(np.int64), minlength=24).reshape(3, 8)
valid = match_map >= 0
mat_tab = np.bincount(ms[valid].astype(np.int64) * 8 + match_map[valid], minlength=24).reshape(3, 8)
print('%-12s %9s %9s %9s | %8s %8s %8s' % ('map', 'rnd_train', 'rnd_val', 'rnd_test', 'm_train', 'm_val', 'm_test'))
for m, mn in enumerate(maps):
    print('%-12s %9d %9d %9d | %8d %8d %8d' % (mn, rnd_tab[0, m], rnd_tab[1, m], rnd_tab[2, m], mat_tab[0, m], mat_tab[1, m], mat_tab[2, m]))
print('column totals rounds', rnd_tab[:, :len(maps)].sum(axis=1).tolist(), 'matches', mat_tab[:, :len(maps)].sum(axis=1).tolist())
print('rounds outside maps.json indices', int(rnd_tab[:, len(maps):].sum()))
for m, mn in enumerate(maps):
    if rnd_tab[2, m] == 0 or mat_tab[2, m] == 0:
        print('FLAG zero test rows:', mn, 'test rounds', int(rnd_tab[2, m]), 'test matches', int(mat_tab[2, m]))
print('zero-test-row flags checked over', len(maps), 'maps')

print('\n== (6) duplicate match keys')
cnt = Counter(keys)
dups = {k: v for k, v in cnt.items() if v > 1}
print('unique keys', len(cnt), 'of', len(keys), '| duplicated keys', len(dups), '| extra rows', sum(v - 1 for v in dups.values()))
pat = re.compile(r'^csds/\d{4}/\d{2}/\d{2}/[0-9a-f-]{36}/csds$')
print('keys not matching csds path pattern', sum(1 for k in keys if not pat.match(k)))
uuids = [k.split('/')[-2] for k in keys]
print('unique uuids', len(set(uuids)), 'duplicate uuids', len(uuids) - len(set(uuids)))
oct_keys = [k for k in keys if re.match(r'^csds/2026/10/(2[6-9]|3[01])/', k)]
print('keys dated 2026-10-26..31', len(oct_keys))
print('key date range min', min(k.split('/')[1:4] for k in keys), 'max', max(k.split('/')[1:4] for k in keys))

print('\n== (7) leakage')
ks = set(keys)
ku = set(uuids)
hb_lines = [l.strip() for l in open(F + '/holdout_b_keys.txt', encoding='utf-8') if l.strip()]
hbs = set(hb_lines)
print('holdout_b_keys.txt nonblank', len(hb_lines), 'unique', len(hbs))
print('holdout_b sample', hb_lines[:2])
print('intersection match_keys x holdout_b (exact)', len(ks & hbs))
hb_u = {k.split('/')[-2] for k in hb_lines}
print('intersection match_keys x holdout_b (uuid)', len(ku & hb_u))
S3 = 's3://pureskillgg-csgo-production-sagemaker-0/adam/tomes/holdout_2026-10-26_31/tome/csds/match_keys.txt'
r = subprocess.run(['aws', 's3', 'cp', S3, '-'], capture_output=True, text=True)
print('aws s3 cp rc', r.returncode, 'stderr', r.stderr.strip()[:300])
tome = [l.strip() for l in r.stdout.splitlines() if l.strip()]
print('tome nonblank', len(tome), 'unique', len(set(tome)))
print('tome sample', tome[:2])
print('tome format matches csds pattern', sum(1 for k in tome if pat.match(k)))
print('intersection match_keys x tome (exact)', len(ks & set(tome)))
tome_u = {k.split('/')[-2] for k in tome if '/' in k}
print('intersection match_keys x tome (uuid)', len(ku & tome_u))
print('tome x holdout_b (exact)', len(set(tome) & hbs))

print('\n== (8) sentinels vs missing_per_feature')
mpf = pr['missing_per_feature']
CH = 2000000
def scan(arr, fn):
    n = arr.shape[0]
    cols = arr.shape[1] if arr.ndim == 2 else 1
    cnt = np.zeros(cols, dtype=np.int64)
    vmin = np.full(cols, np.inf)
    vmax = np.full(cols, -np.inf)
    for s in range(0, n, CH):
        b = np.asarray(arr[s:s + CH])
        if b.ndim == 1:
            b = b[:, None]
        sent = fn(b)
        cnt += sent.sum(axis=0)
        v = b.astype(np.float64)
        vmin = np.minimum(vmin, np.where(sent, np.inf, v).min(axis=0))
        vmax = np.maximum(vmax, np.where(sent, -np.inf, v).max(axis=0))
    return cnt, vmin, vmax
specs = [
    ('int8', pos_int8, pr['int8'], lambda b: b == -1),
    ('uint16', pos_u16, pr['uint16'], lambda b: b == 65535),
    ('uint8', pos_u8, pr['uint8_tenth_seconds'], lambda b: b == 255),
    ('float16', pos_f16, pr['float16'], lambda b: np.isnan(b)),
]
all_ok = True
for dt, arr, fl, fn in specs:
    cnt, vmin, vmax = scan(arr, fn)
    for i, f in enumerate(fl):
        exp = mpf.get(f)
        ok = int(cnt[i]) == exp
        all_ok = all_ok and ok
        print('%-8s %-22s sentinel=%10d README=%10s match=%s valid_min=%s valid_max=%s' % (dt, f, int(cnt[i]), exp, ok, vmin[i], vmax[i]))
print('all sentinel counts match README:', all_ok)
print('README keys not checked:', sorted(set(mpf) - set(pr['int8'] + pr['uint16'] + pr['uint8_tenth_seconds'] + pr['float16'])))
