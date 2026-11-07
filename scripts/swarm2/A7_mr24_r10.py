"""A7: the 12 matches with chosen MR24 whose first disagreement is round 10 and have no WINMM."""
import csv
import importlib.util
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
spec = importlib.util.spec_from_file_location("A7", ROOT / "scripts" / "swarm2" / "A7_score_coverage.py")
A7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(A7)
RULE = A7.half_rule(*A7.TOMES["2024"])
with open(ROOT / "logs" / "swarm2" / "A7_match_detail.log", encoding="utf-8") as fh:
    rows = [r for r in csv.DictReader(fh) if r["tome"] == "2024" and r["mr"] == "24"
            and r["first_dis_round"] == "10" and r["winmm_rounds"] == "0"]
print("count", len(rows))
H, W, F, info = A7.load_tome("2024", *A7.TOMES["2024"])
for r in rows[:12]:
    m = r["match"]
    Fm, Wm = F[m], W[m]
    fit = A7.mr_fit_full(Fm, (16, 24))
    syms = []
    for k in range(1, 13):
        if k in Fm and k + 1 in Fm:
            t, c = Fm[k]; t2, c2 = Fm[k+1]
            syms.append(A7.boundary_class(t, c, t2, c2)[0])
        else:
            syms.append("-")
    print(m[-38:], "day", r["day"], "fit(bad,ok) 16:", fit[16], "24:", fit[24], "syms1-12:", "".join(syms))
    print("   freeze r7-r11:", [Fm.get(k) for k in range(7, 12)], "W r7-r11:", [Wm.get(k) for k in range(7, 12)])
