import json, sys
def shape(v, depth=0, maxd=3):
    if isinstance(v, dict):
        return {"__type": "dict", "n_keys": len(v), "keys_sample": list(v.keys())[:25], "child": (shape(next(iter(v.values())), depth+1, maxd) if v and depth < maxd else None)}
    if isinstance(v, list):
        return {"__type": "list", "len": len(v), "elem": (shape(v[0], depth+1, maxd) if v and depth < maxd else None)}
    return {"__type": type(v).__name__}
for p in sys.argv[1:]:
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    print("==", p)
    print(json.dumps(shape(d), indent=1)[:2500])
