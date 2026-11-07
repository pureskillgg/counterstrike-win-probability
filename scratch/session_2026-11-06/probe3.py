import json, collections
F = "/home/sagemaker-user/adam/builder/final"
for name in ["split_manifest.json", "split_manifest_v2.json"]:
    m = json.load(open(F+"/"+name))
    print("==", name, "top keys", list(m.keys())[:12])
    a = m.get("assign", {})
    print("  assign len", len(a), "value types", collections.Counter(type(v).__name__ for v in list(a.values())[:50]))
    print("  sample value", repr(list(a.items())[0])[:200])
    if "rule" in m: print("  rule", str(m["rule"])[:200])
ek = json.load(open("/home/sagemaker-user/adam/builder/final/export_clean_v2/match_keys.json"))
print("export match_keys.json type", type(ek).__name__, "len", len(ek))
if isinstance(ek, dict):
    print("  dict keys", list(ek.keys())[:10])
    for k in list(ek.keys())[:3]:
        v = ek[k]; print("  ", k, type(v).__name__, (len(v) if hasattr(v,'__len__') else v), str(v)[:150])
else:
    print("  sample", str(ek[:2])[:200])
mk = json.load(open("/home/sagemaker-user/adam/builder/final/export_clean_v2/maps.json"))
print("maps.json", str(mk)[:300])
nm = json.load(open("/home/sagemaker-user/adam/builder/final/export_clean_v2/loadout_names.json"))
print("loadout_names.json type", type(nm).__name__, "len", len(nm), str(nm)[:600])
