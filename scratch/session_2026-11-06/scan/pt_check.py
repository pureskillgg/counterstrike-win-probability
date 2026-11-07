import sys, os, re, zipfile, io
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
SCAN = r'C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad\scan'
sys.path.insert(0, SCAN)
roots = sys.argv[1:]
terms = [b'pandas', b'DataFrame', b'Series', b'numpy.core', b'numpy._core', b'csds/', b'steam', b'match_key', b'player_name', b'pyarrow', b'sklearn', b'xgboost', b'lightgbm', b'argparse', b'Namespace', b'__main__']
from collections import Counter
files = 0; tot = Counter(); nz = []
for root in roots:
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if not f.endswith('.pt'):
                continue
            p = os.path.join(dp, f)
            files += 1
            data = open(p, 'rb').read()
            c = {t.decode(): data.count(t) for t in terms if data.count(t)}
            is_zip = data[:2] == b'PK'
            for k, v in c.items():
                tot[k] += 1
            if c.get('csds/') or c.get('steam') or c.get('match_key') or c.get('player_name') or c.get('DataFrame') or c.get('Series'):
                nz.append((p.replace(chr(92), '/')[-110:], c))
print('pt files', files)
print('files containing each term (count of files):', dict(tot))
print('files with identifier/data-object terms:', len(nz))
for x in nz[:30]:
    print('  ', x)
