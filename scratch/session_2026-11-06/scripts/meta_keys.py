import sys, json
def shape(o, depth=0, maxd=3):
    if depth >= maxd:
        return type(o).__name__ + (f"[len={len(o)}]" if hasattr(o,'__len__') else "")
    if isinstance(o, dict):
        return {k: shape(v, depth+1, maxd) for k, v in list(o.items())[:40]} | ({"...more_keys": len(o)-40} if len(o)>40 else {})
    if isinstance(o, list):
        return [f"list[len={len(o)}]", shape(o[0], depth+1, maxd) if o else None]
    if isinstance(o, str):
        return f"str(len={len(o)})"
    return f"{type(o).__name__}"
for p in sys.argv[1:]:
    print("==", p.split('/')[-1])
    with open(p, encoding='utf-8') as fh:
        d = json.load(fh)
    print(json.dumps(shape(d, 0, 4), indent=1)[:3000])
