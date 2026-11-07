import re, sys, collections
BS = chr(92)
src = sys.argv[1]
out = []
d = None
with open(src, encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.rstrip('\r\n')
        if line.startswith(' Directory of '):
            d = line[len(' Directory of '):]
            continue
        if '<JUNCTION>' in line and d is not None:
            s = line.split('<JUNCTION>', 1)[1].strip()
            m = re.match(r'^(.*?) \[(.*)\]$', s)
            if m:
                name, tgt = m.group(1), m.group(2)
            else:
                name, tgt = s, ''
            tgt = tgt.replace('??' + BS, '')
            out.append((d + BS + name, tgt))
with open(sys.argv[2], 'w', encoding='utf-8') as g:
    for a, b in out:
        g.write(a + '\t' + b + '\n')
print('junctions parsed:', len(out))
tc = collections.Counter()
for a, b in out:
    parts = b.split(BS)
    tc[BS.join(parts[:5])] += 1
print('--- targets (first 5 parts):')
for k, v in tc.most_common(25):
    print(v, k)
sc = collections.Counter()
for a, b in out:
    parts = a.split(BS)
    sc[BS.join(parts[:4])] += 1
print('--- junction holder dirs (first 4 parts):')
for k, v in sc.most_common(25):
    print(v, k)
