import json, sys
def load(p):
    return json.loads(open(p,'rb').read().decode('utf-8-sig',errors='replace'), parse_constant=lambda c: None)
def short(v):
    s = json.dumps(v, default=str)
    return s if len(s) < 160 else s[:157] + "..."
for p in sys.argv[1:]:
    o = load(p)
    print("==", p)
    if isinstance(o, dict):
        for k, v in o.items():
            if k == 'history':
                print("  history len", len(v), "first", short(v[0]) if v else None, "last", short(v[-1]) if v else None)
            else:
                print("  ", k, "=", short(v))
