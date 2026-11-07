import json, sys
def shape(o, depth=0, name="root"):
    pad = "  " * depth
    if isinstance(o, dict):
        print(f"{pad}{name}: dict({len(o)}) keys={list(o.keys())[:20]}")
        for k in list(o.keys())[:12]:
            if depth < 2:
                shape(o[k], depth + 1, str(k))
    elif isinstance(o, list):
        print(f"{pad}{name}: list(len={len(o)}) first_type={type(o[0]).__name__ if o else None}")
        if o and depth < 2 and isinstance(o[0], (dict, list)):
            shape(o[0], depth + 1, name + "[0]")
    else:
        print(f"{pad}{name}: {type(o).__name__} (value not printed)")
for p in sys.argv[1:]:
    print("FILE", p.split("/")[-1])
    shape(json.load(open(p, encoding="utf-8")))
