import os, sys, collections, re
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
root = 'G:/pureskill/csds-cs2/evac_2026-10-13/tomes'
builds = collections.Counter(); windows = collections.Counter(); jsonish = collections.Counter()
for dp, dn, fn in os.walk(root):
    for f in fn:
        p = os.path.join(dp, f).replace(chr(92), '/')
        parts = p.split('/')
        if 'csds' not in parts: continue
        i = parts.index('csds')
        build = parts[i-2] if i >= 2 else '?'
        seg = parts[i+1] if i+1 < len(parts) else '?'
        table = seg.split('.')[0]
        win = seg.split('.', 1)[1] if '.' in seg else ''
        builds[(parts[i-3] if i>=3 else '?', build)] += 1
        windows[win] += 1
        with open(os.path.join(dp, f), 'rb') as fh:
            m = fh.read(4)
        if m != b'PAR1':
            jsonish[table] += 1
print('build dirs (year-date, tome dir):', dict(builds))
print('date windows of table folders:', dict(windows))
print('non-parquet tables:', dict(jsonish))
