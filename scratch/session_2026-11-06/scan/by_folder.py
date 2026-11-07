import sys, json, os
from collections import Counter, defaultdict
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
BS = chr(92)
want = set(sys.argv[2].split(','))
per = defaultdict(Counter); files = defaultdict(list)
for line in open(sys.argv[1], encoding='utf-8', errors='replace'):
    if not line.startswith('{'): continue
    d = json.loads(line)
    if 'hits' not in d: continue
    obj = d['obj'].replace(BS, '/')
    if 's3mirror/' in obj: rel = obj.split('s3mirror/')[-1]
    elif 'csds-cs2/' in obj: rel = obj.split('csds-cs2/')[-1]
    else: rel = obj
    top = rel.split('/')[0] + ('/' + rel.split('/')[1] if rel.count('/') >= 2 else '')
    for c, nm in d['hits'].items():
        for n, v in nm.items():
            if n in want:
                per[top][n] += v
                files[n].append((rel, v))
print('== hits by top folder (pattern: count)')
for t in sorted(per): print(f'  {t}: {dict(per[t])}')
for n in sorted(files):
    print(f'== files with {n}: {len(files[n])}')
    for rel, v in files[n][:12]: print('    ', rel[-120:], v)
