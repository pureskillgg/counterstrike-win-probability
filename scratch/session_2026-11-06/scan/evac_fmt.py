import os, sys, collections
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
root = 'G:/pureskill/csds-cs2/evac_2026-10-13/tomes'
want = ('basic_player_personal', 'basic_player_name', 'basic_player_chat', 'header', 'basic_player_info', 'basic_player_status')
magic = collections.Counter(); per = collections.Counter(); sample = {}
for dp, dn, fn in os.walk(root):
    for f in fn:
        p = os.path.join(dp, f).replace(chr(92), '/')
        parts = p.split('/')
        if 'csds' not in parts: continue
        i = parts.index('csds'); tbl = parts[i+1].split('.')[0] if i+1 < len(parts) else '?'
        if tbl not in want: continue
        with open(os.path.join(dp, f), 'rb') as fh:
            head = fh.read(8)
        m = head[:4]
        magic[(m)] += 1; per[tbl] += 1
        sample.setdefault(tbl, p)
print('magic bytes counts (first 4):', {repr(k): v for k, v in magic.items()})
print('files per personal table in evac:', dict(per))
for t, p in sample.items(): print('sample', t, p[-140:])
