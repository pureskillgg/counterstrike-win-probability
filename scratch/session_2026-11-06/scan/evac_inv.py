import os, sys, collections
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
root = 'G:/pureskill/csds-cs2/evac_2026-10-13'
ext = collections.Counter(); byt = collections.Counter(); cnt = 0; tot = 0
tables = collections.Counter()
for dp, dn, fn in os.walk(root):
    for f in fn:
        p = os.path.join(dp, f); s = os.path.getsize(p); tot += s; cnt += 1
        e = os.path.splitext(f)[1].lower() or '(none)'
        ext[e] += 1; byt[e] += s
        parts = p.replace(chr(92), '/').split('/')
        if 'tome' in parts and 'csds' in parts:
            i = parts.index('csds')
            if i + 1 < len(parts): tables[parts[i+1].split('.')[0]] += 1
print('evac files', cnt, 'total GB', round(tot/1e9, 2))
for e, c in ext.most_common(12): print(f'  {c:8d} {byt[e]/1e6:10.1f} MB  {e}')
top = collections.Counter()
for dp, dn, fn in os.walk(root):
    rel = os.path.relpath(dp, root).replace(chr(92), '/').split('/')[0]
    top[rel] += len(fn)
print('top-level in evac:', dict(top))
print('tome table names (distinct):', len(tables), 'sample:', sorted(tables)[:90])
