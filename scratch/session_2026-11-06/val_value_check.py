import subprocess, io, re, collections, sys
import pyarrow.parquet as pq
AWS = ['aws', '--profile', 'runpod', '--endpoint-url', 'https://s3api-eu-ro-1.runpod.io', '--region', 'eu-ro-1']
raw = subprocess.run(AWS + ['s3', 'cp', 's3://z7yvc40ykx/cs2data/keys/val_keys.parquet', '-'], capture_output=True, check=True).stdout
t = pq.read_table(io.BytesIO(raw))
print('rows:', t.num_rows, 'cols:', t.column_names)
UUIDPATH = re.compile(r'^csds/\d{4}/\d{2}/\d{2}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/csds$')
STEAM64 = re.compile(r'7656119\d{10}')
mk = t.column('match_key').to_pylist()
print('distinct match_key:', len(set(mk)), ' shape-conformant (uuid csds path):', sum(1 for x in mk if UUIDPATH.match(x)), ' non-conformant:', sum(1 for x in mk if not UUIDPATH.match(x)))
steam_cells = 0
for name in t.column_names:
    col = t.column(name)
    if str(col.type) in ('large_string', 'string'):
        steam_cells += sum(1 for x in col.to_pylist() if x and STEAM64.search(x))
    elif 'int' in str(col.type):
        vals = col.to_numpy()
        steam_cells += int(((vals >= 76561197960265728) & (vals <= 76561202255233023)).sum())
print('steam64-pattern or steam64-range hits across all cells:', steam_cells)
print('label_ct_win distinct values:', sorted(set(t.column('label_ct_win').to_pylist())))
