import json, re, subprocess

E = '/home/sagemaker-user/adam/builder/final/export_clean_v2'
F = '/home/sagemaker-user/adam/builder/final'
S3 = 's3://pureskillgg-csgo-production-sagemaker-0/adam/tomes/holdout_2026-10-26_31/match_keys.txt'
pat = re.compile(r'^csds/\d{4}/\d{2}/\d{2}/[0-9a-f-]{36}/csds$')

keys = json.load(open(E + '/match_keys.json'))
ks = set(keys)
ku = {k.split('/')[-2] for k in keys}

hb = [l.strip() for l in open(F + '/holdout_b_keys.txt', encoding='utf-8') if l.strip()]
hbs = set(hb)
hbu = {k.split('/')[-2] for k in hb}

r = subprocess.run(['aws', 's3', 'cp', S3, '-'], capture_output=True, text=True)
print('aws rc', r.returncode, 'stderr', r.stderr.strip()[:200])
tome = [l.strip() for l in r.stdout.splitlines() if l.strip()]
ts = set(tome)
tu = {k.split('/')[-2] for k in tome}
print('tome nonblank lines', len(tome), 'unique', len(ts))
print('tome matches csds path pattern', sum(1 for k in tome if pat.match(k)))
print('tome date range min', min(k.split('/')[1:4] for k in tome), 'max', max(k.split('/')[1:4] for k in tome))
print('tome keys dated 2026-10-26..31', sum(1 for k in tome if re.match(r'^csds/2026/10/(2[6-9]|3[01])/', k)))
print('--- leakage (export match_keys.json x tome) ---')
print('exact string intersection', len(ks & ts))
print('uuid intersection', len(ku & tu))
print('--- leakage (export x holdout_b_keys.txt) ---')
print('holdout_b nonblank', len(hb), 'unique', len(hbs))
print('exact string intersection', len(ks & hbs))
print('uuid intersection', len(ku & hbu))
print('--- cross-check tome x holdout_b ---')
print('tome x holdout_b exact', len(ts & hbs), 'uuid', len(tu & hbu))
print('--- export totals ---')
print('export keys', len(keys), 'unique', len(ks))
print('export keys matching any tome or holdout_b key', len((ks & ts) | (ks & hbs)))
