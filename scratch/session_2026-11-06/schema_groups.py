import json
import collections
import re

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
d = json.load(open(SCR + "/inventory.json", encoding="utf-8"))

pq = [r for r in d if r.get("kind") == "parquet"]
groups = collections.OrderedDict()
for r in pq:
    if "error" in r:
        print("ERROR", r["path"][-90:], r["error"])
        continue
    sig = tuple((c, t) for c, t in r["columns"])
    groups.setdefault(sig, []).append(r)

print("parquet files:", len(pq), "distinct column signatures:", len(groups))
for i, (sig, files) in enumerate(groups.items()):
    names = [c for c, _ in sig]
    types = {c: t for c, t in sig}
    short = ", ".join(f"{c}:{t.replace('string','str').replace('int64','i64').replace('int32','i32').replace('double','f64').replace('float','f32')}" for c, t in sig)
    if len(short) > 420:
        short = short[:420] + "..."
    rows = sum(f["rows"] for f in files)
    screen = [c for c in names if files[0].get("col_screen_hits") and c in files[0]["col_screen_hits"]]
    kvhits = sum(v["meta_pat_hits"] for f in files for v in f.get("kv_meta", {}).values())
    kvkeys = sorted({k for f in files for k in f.get("kv_meta", {}).keys()})
    print(f"\n[{i}] files={len(files)} rows={rows} ncols={len(sig)} kvmeta_pat_hits={kvhits}")
    print("   cols:", short)
    print("   screen:", screen)
    print("   kv_keys:", kvkeys[:12])
    print("   e.g.:", "/".join(files[0]["path"].replace("\\", "/").split("/")[-5:]))
    print("   all:", "; ".join(sorted({"/".join(f["path"].split("/")[-3:-1]) for f in files}))[:300])
