"""A7: the WINMM matches whose first recorded-winner flip is not at halftime+1."""
import csv, collections, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
with open(ROOT / "logs" / "swarm2" / "A7_match_detail.log", encoding="utf-8") as fh:
    rows = [r for r in csv.DictReader(fh) if r["tome"] == "2024" and int(r["winmm_rounds"]) > 0]
rest = []
for r in rows:
    win = [int(x) for x in r["winmm_list"].split(",") if x]
    mr = int(r["mr"])
    if min(win) != mr // 2 + 1:
        rest.append((r["match"][-30:], r["mr"], r["winmm_list"], r["dis_list"][:40]))
print("WINMM matches:", len(rows), "not at halftime+1 first:", len(rest))
print(collections.Counter((x[1], x[2]) for x in rest).most_common(10))
for x in rest[:5]:
    print("  ", x)
