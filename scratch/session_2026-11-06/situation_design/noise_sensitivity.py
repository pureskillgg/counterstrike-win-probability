"""Sensitivity of the crude z-counts to the per-datapoint sd (sigma). Reads only by_situation.csv."""
import csv, math
BASE = "C:/Users/13144/Documents/pureskill/counterstrike-win-probability/docs/eval_reports/positions_validation_gdt"
rows = list(csv.DictReader(open(f"{BASE}/by_situation.csv", newline="", encoding="utf-8")))
N = sum(int(r["datapoints"]) for r in rows)
for m in ["p_full_trainctr", "p_base"]:
    cells = [(int(r["datapoints"]), float(r[f"gain_{m}"])) for r in rows]
    print(f"model {m}")
    for sigma in [0.21, 0.30, 0.42, 0.60, 0.84]:
        neg2 = [(n, g) for n, g in cells if g / (sigma / math.sqrt(n)) < -2]
        pos2 = [(n, g) for n, g in cells if g / (sigma / math.sqrt(n)) > 2]
        neg1 = [(n, g) for n, g in cells if g / (sigma / math.sqrt(n)) < -1]
        print(f"  sigma {sigma:.2f}: z<-1 {len(neg1):4d} cells; z<-2 {len(neg2):4d} cells ({sum(n for n,_ in neg2):5d} dp, "
              f"GDT part {sum(n*g for n,g in neg2)/N:+.5f}); z>+2 {len(pos2):4d} cells")
