import json, sys, re
raw = open(sys.argv[1], "rb").read()
try:
    d = json.loads(raw.decode("utf-8"))
except Exception as e:
    print("not_json", e); sys.exit(0)
def walk(o, depth=0, path="$"):
    if isinstance(o, dict):
        for k, v in o.items():
            print("  " * depth + f"KEY {k} ({type(v).__name__})")
            if depth < 3:
                walk(v, depth + 1, path + "." + k)
    elif isinstance(o, list):
        print("  " * depth + f"LIST len={len(o)} elem={type(o[0]).__name__ if o else None}")
        if o and isinstance(o[0], (dict, list)) and depth < 3:
            walk(o[0], depth + 1, path + "[0]")
walk(d)
