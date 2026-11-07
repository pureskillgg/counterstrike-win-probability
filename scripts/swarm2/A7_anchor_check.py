"""A7: characterise the 2024 matches with no round 1 (re-anchored chain): day, MR, flips, DIS."""
import collections
import csv
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
rows = []
with open(ROOT / "logs" / "swarm2" / "A7_match_detail.log", encoding="utf-8") as fh:
    rd = csv.DictReader(fh)
    rows = [r for r in rd if r["tome"] == "2024"]
anch = [r for r in rows if int(r["anchors"]) > 0]
non = [r for r in rows if int(r["anchors"]) == 0]
print(f"2024 anchored matches {len(anch)}; non-anchored {len(non)}")
print("anchored: day dist", dict(sorted(collections.Counter(r["day"] for r in anch).items())))
print("non-anchored: day dist", dict(sorted(collections.Counter(r["day"] for r in non).items())))
print("anchored: mr", dict(collections.Counter(r["mr"] for r in anch)))
print("non-anchored: mr", dict(collections.Counter(r["mr"] for r in non)))
for name, g in (("anchored", anch), ("non-anchored", non)):
    print(f"{name}: DIS>0 {sum(1 for r in g if int(r['dis_rounds'])>0)}; WINMM>0 {sum(1 for r in g if int(r['winmm_rounds'])>0)}; "
          f"ATT>0 {sum(1 for r in g if int(r['att'])>0)}; compared rounds {sum(int(r['compared_rounds']) for r in g)}; "
          f"freeze rows {sum(int(r['rounds_freeze']) for r in g)}")
print("anchored with WINMM by day:", dict(sorted(collections.Counter(r["day"] for r in anch if int(r["winmm_rounds"])>0).items())))
print("anchored: first_dis_round dist", collections.Counter(r["first_dis_round"] for r in anch if int(r["dis_rounds"])>0).most_common(5))
print("anchored: sample rows:")
for r in anch[:3]:
    print("  ", r["match"][-40:], r["day"], "mr", r["mr"], "rounds_end", r["rounds_end"], "freeze", r["rounds_freeze"], "compared", r["compared_rounds"], "anchors", r["anchors"], "dis", r["dis_rounds"], "winmm", r["winmm_rounds"])
