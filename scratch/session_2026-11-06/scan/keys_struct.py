import json, re, sys
KEY = re.compile(r'^csds/\d{4}/\d{2}/\d{2}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/csds$')
for p in sys.argv[1:]:
    try:
        d = json.load(open(p, encoding='utf-8'))
    except Exception as e:
        print(p.split('/')[-1], 'parse error', type(e).__name__); continue
    if isinstance(d, dict):
        top = {k: (type(v).__name__, len(v) if hasattr(v, '__len__') else v) for k, v in d.items()}
        print(p.split('/')[-1], 'dict; top-level fields (name, type, size):', top)
        cand = [v for v in d.values() if isinstance(v, list)]
        vals = cand[0] if cand else []
    else:
        print(p.split('/')[-1], 'list; entries', len(d)); vals = d
    strs = [v for v in vals if isinstance(v, str)]
    fmt = sum(1 for v in strs if KEY.match(v))
    other_types = {type(v).__name__ for v in vals}
    print('   entries', len(vals), 'string keys', len(strs), 'match-key format', fmt, 'entry types', sorted(other_types))
    if strs:
        dates = sorted({v[5:15] for v in strs if KEY.match(v)})
        print('   distinct match dates', len(dates), 'range (YYYY/MM/DD) only; no values printed')
