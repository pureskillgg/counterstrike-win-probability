import os, sys, json, re, collections
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
import pyarrow.parquet as pq
ROOT = 'G:/pureskill/csds-cs2/evac_2026-10-13/tomes'
COLS = {
 'basic_player_personal': ['name', 'clan_tag', 'steam_id'],
 'basic_player_name': ['name_new', 'name_old'],
 'basic_player_chat': ['text'],
 'header': ['demo_id', 'sharecode', 'client_name', 'server_name'],
}
SHAPES = [
 ('steamid64', re.compile(r'^7656119\d{10}$')),
 ('steam2_or_3', re.compile(r'^(STEAM_\d:\d:\d+|\[U:1:\d+\])$')),
 ('sharecode', re.compile(r'^CSGO-')),
 ('email', re.compile(r'@')),
 ('url', re.compile(r'https?://|www\.')),
 ('lower8', re.compile(r'^[a-z]{8}$')),
 ('letter1', re.compile(r'^[A-Za-z]$')),
 ('mixed13_space', re.compile(r'^[A-Za-z][A-Za-z ]{11,13}$')),
]
def classify(v):
    if v is None: return 'null'
    s = str(v)
    for name, rx in SHAPES:
        if rx.search(s): return name
    return 'other'
res = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
distinct = collections.defaultdict(lambda: collections.defaultdict(set))
nrows = collections.Counter(); files_by = collections.Counter(); skipped_json = []
for dp, dn, fn in os.walk(ROOT):
    for f in fn:
        p = os.path.join(dp, f)
        parts = p.replace(chr(92), '/').split('/')
        if 'csds' not in parts: continue
        seg = parts[parts.index('csds') + 1] if parts.index('csds') + 1 < len(parts) else ''
        table = seg.split('.')[0]
        if table not in COLS: continue
        build = parts[parts.index('tome') - 1] if 'tome' in parts else '?'
        key = (build, table)
        with open(p, 'rb') as fh:
            head = fh.read(4)
        values = {}
        if head == b'PAR1':
            cols = COLS[table]
            t = pq.read_table(p, columns=[c for c in cols if c in pq.read_schema(p).names])
            for c in t.column_names:
                values[c] = t.column(c).to_pylist()
        else:
            try:
                d = json.load(open(p, encoding='utf-8'))
            except Exception:
                skipped_json.append((build, table, 'not-json-or-parquet')); continue
            if isinstance(d, dict):
                for c in COLS[table]:
                    if c in d and isinstance(d[c], list): values[c] = d[c]
                if not values:
                    skipped_json.append((build, table, 'dict-keys:' + ','.join(list(d.keys())[:8]))); continue
            elif isinstance(d, list) and d and isinstance(d[0], dict):
                for c in COLS[table]:
                    values[c] = [r.get(c) for r in d]
            else:
                skipped_json.append((build, table, 'other-json')); continue
        files_by[key] += 1
        for c, vals in values.items():
            nrows[(build, table, c)] += len(vals)
            for v in vals:
                res[(build, table)][c][classify(v)] += 1
            distinct[(build, table)][c].update(set(map(str, vals)))
for key in sorted(files_by):
    build, table = key
    print(f'== {build} / {table}: files {files_by[key]}')
    for c in COLS[table]:
        if c in res[key]:
            n = sum(res[key][c].values())
            dcount = len(distinct[key][c])
            print(f'   col {c}: rows {n}, distinct values {dcount}, shape classes {dict(res[key][c].most_common(8))}')
print('skipped:', skipped_json[:10], 'count', len(skipped_json))
