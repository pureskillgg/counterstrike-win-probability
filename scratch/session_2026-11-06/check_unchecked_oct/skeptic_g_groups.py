import os, csv, json, re, collections

ROOT = "G:/pureskill/csds-cs2"
W = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
targets = {}
for r in csv.DictReader(open(W + "undetermined_matches.csv", encoding="utf-8")):
    targets[r["match_key"].split("/")[4]] = r["match_key"]

groups = collections.defaultdict(set)   # group -> set of target uuids
examples = {}
for dp, dn, fn in os.walk(ROOT):
    if ".pixi" in dp or "holdout" in dp.lower():
        dn[:] = []
        continue
    rel = os.path.relpath(dp, ROOT).replace("\\", "/")
    parts = rel.split("/")
    for name in list(dn) + list(fn):
        for u in UUID_RE.findall(name):
            if u in targets:
                g = "/".join(parts[:3]) if parts != ["."] else "."
                groups[g].add(u)
                if g not in examples:
                    examples[g] = os.path.join(dp, name)

out = {}
for g, us in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    out[g] = {"target_uuids": len(us), "example": examples[g]}
    print(g, len(us), examples[g], flush=True)
covered = set().union(*groups.values()) if groups else set()
print("total distinct target uuids seen in G: names", len(covered), flush=True)
with open(W + "skeptic_g_groups.json", "w") as f:
    json.dump({"total": len(covered), "groups": out}, f, indent=1)
print("DONE", flush=True)
