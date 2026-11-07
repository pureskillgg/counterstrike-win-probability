"""Crude per-cell noise check. sigma_eff is inferred from the overall GDT interval in summary.json (absolute GDT for
p_full_trainctr: GDT_ci) with 297 matches and 50,471 datapoints, assuming independence within a match. Upper-ish guide."""
import csv, json, math
BASE = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/docs/eval_reports/positions_validation_gdt"
summ = json.load(open(f"{BASE}/summary.json", encoding="utf-8"))
m = "p_full_trainctr"
lo, hi = summ["models"][m]["GDT_ci"]
se_overall = (hi - lo) / (2 * 1.96)
M, N = summ["counts"]["matches"], summ["counts"]["datapoints"]
sigma_eff = se_overall * math.sqrt(N)
print(f"SE overall GDT {m} = {se_overall:.5f} (GDT_ci {lo:.5f} to {hi:.5f}); sigma_eff = SE*sqrt(N) = {sigma_eff:.3f}")
for n in [1, 10, 50, 100, 500]:
    print(f"  cell n={n:4d}: noise SE of mean gain ~ {sigma_eff/math.sqrt(n):.4f}")
rows = list(csv.DictReader(open(f"{BASE}/by_situation.csv", newline="", encoding="utf-8")))
weak = [(int(r['datapoints']), float(r[f'gain_{m}'])) for r in rows if float(r[f'gain_{m}']) < 0]
allc = [(int(r['datapoints']), float(r[f'gain_{m}'])) for r in rows]
def zcount(cells, thr):
    return sum(1 for n, g in cells if g / (sigma_eff / math.sqrt(n)) < -thr)
print(f"weak cells: {len(weak)}")
for thr in [1, 2]:
    print(f"  weak cells with z < -{thr} (crude): {zcount(weak, thr)} of {len(weak)} ({100*zcount(weak,thr)/len(weak):.1f}%)")
print(f"all cells with z < -2: {zcount(allc, 2)}; z > +2: {sum(1 for n,g in allc if g/(sigma_eff/math.sqrt(n)) > 2)}")
# datapoints in cells with z < -2
dp_bad = sum(n for n, g in allc if g/(sigma_eff/math.sqrt(n)) < -2)
print(f"datapoints in cells with z < -2: {dp_bad} = {100*dp_bad/N:.3f}% of all")
gain_bad = sum(n*g for n, g in allc if g/(sigma_eff/math.sqrt(n)) < -2)/N
print(f"GDT contribution of cells with z < -2: {gain_bad:+.5f}")
weak_small = [(n,g) for n,g in weak if n <= 50]
print(f"weak cells with n<=50: {len(weak_small)}; datapoints {sum(n for n,_ in weak_small)}")
