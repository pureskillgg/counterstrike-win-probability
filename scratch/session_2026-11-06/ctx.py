import json, os, collections
ROOTS = ["G:/pureskill/csds-cs2/data/diagnostics","G:/pureskill/csds-cs2/data/sprint","G:/pureskill/csds-cs2/data/holdout_b","G:/pureskill/csds-cs2/data/combined"]
def load(p):
    return json.loads(open(p,'rb').read().decode('utf-8-sig',errors='replace'), parse_constant=lambda c: None)
combos = collections.Counter()
ex = {}
for root in ROOTS:
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if not f.endswith('.json'): continue
            p = os.path.join(dp,f).replace(chr(92),'/')
            try: o = load(p)
            except Exception: continue
            if isinstance(o, dict) and 'test_rows_in_memory' in o:
                key = (repr(o.get('test_rows_in_memory')), repr(o.get('test_rows_used')), repr(o.get('test_rows_in_arrays')), repr(o.get('test_rows_read')), repr(o.get('test_rows')), 'best_val_log_loss' in o, 'undertrained_flag' in o)
                combos[key] += 1
                ex.setdefault(key, p)
for k,c in combos.most_common():
    print(c, k, ex[k])
