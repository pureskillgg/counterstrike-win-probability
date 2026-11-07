"""A7: internal consistency of the per-match results (reads logs/swarm2/A7_match_detail.log)."""
import csv
import collections
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
with open(ROOT / "logs" / "swarm2" / "A7_match_detail.log", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
def L(s):
    return [int(x) for x in s.split(",") if x]
c = collections.Counter()
for r in rows:
    tome, mr = r["tome"], int(r["mr"])
    dis, win = L(r["dis_list"]), L(r["winmm_list"])
    if tome == "2024":
        if win and not dis:
            c["WINMM_without_DIS"] += 1
        if win and dis:
            c["WINMM_and_DIS"] += 1
        if dis and not win:
            c["DIS_without_WINMM"] += 1
            c[f"DIS_without_WINMM first={r['first_dis_round']} mr={mr}"] += 1
        for w in win:
            c[f"WINMM round {w} minus first sched swap+1 (mr {mr})"] += 0
            c[f"WINMM at expected flip round mr{mr}->{mr//2+1 if mr<=24 else None}"] += int(w == mr // 2 + 1)
        if win:
            c["WINMM matches with first WINMM at halftime+1"] += int(min(win) == mr // 2 + 1)
for k, v in sorted(c.items()):
    if v:
        print(f"{k}: {v}")
ct = collections.Counter((r["tome"], r["group"]) for r in rows)
print("groups:", dict(ct))
