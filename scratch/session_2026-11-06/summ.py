import sys, os, json
sys.path.insert(0, sys.argv[1])
from pqfooter import inspect
for p in sys.argv[2:]:
    try:
        r = inspect(p)
        names = [c['name'] for c in r['columns']]
        feats = [n for n in names if n not in ('match_key', 'round', 'tick')]
        codecs = sorted(set(r['codecs_rg0'].values()))
        print(json.dumps({
            'file': os.path.basename(p), 'dir': os.path.basename(os.path.dirname(p)),
            'MB': round(r['file_bytes']/1e6, 1), 'rows': r['num_rows'], 'rg': r['row_groups'],
            'ncols': r['n_columns'], 'codecs': codecs,
            'has_match_key': 'match_key' in names,
            'cols': names if len(names) <= 30 else names[:30] + ['...'],
        }))
    except Exception as e:
        print(json.dumps({'file': os.path.basename(p), 'error': str(e)[:200]}))
