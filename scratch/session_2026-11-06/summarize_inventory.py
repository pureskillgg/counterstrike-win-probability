import json
import collections
import sys

SCR = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
d = json.load(open(SCR + "/inventory.json", encoding="utf-8"))


def group(p):
    p = p.replace("\\", "/")
    if "pureskill_internship/csds-cs2/data/" in p:
        rest = p.split("pureskill_internship/csds-cs2/data/")[1]
        return "DATA/" + rest.split("/")[0]
    if "/csds-cs2/" in p:
        rest = p.split("/csds-cs2/")[1]
        return "/".join(rest.split("/")[:2])
    return "/".join(p.split("/")[:3])


agg = collections.defaultdict(collections.Counter)
rows = collections.defaultdict(int)
for r in d:
    g = group(r["path"])
    agg[g][r.get("kind", "?")] += 1
    if r.get("kind") == "parquet":
        rows[g] += r.get("rows", 0)
for g in sorted(agg):
    print(f"{g:55s} {dict(agg[g])} parquet_rows={rows[g]}")
print("errors:", [(r["path"][-80:], r["error"]) for r in d if "error" in r][:30])
