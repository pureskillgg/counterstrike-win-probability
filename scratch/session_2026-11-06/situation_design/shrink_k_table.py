import csv, math, statistics
path = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/evaluation/intern_gdt/table.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))
n = [int(r["n_train"]) for r in rows if int(r["n_train"]) > 0]
zero_cells = sum(1 for r in rows if int(r["n_train"]) == 0)
print("cells with n_train 0 (no training visits, carry no weight):", zero_cells)
print("cells", len(n), "total visits (sum n_train)", sum(n))
s = sorted(n)
q = lambda p: s[min(len(s)-1, int(p*len(s)))]
print("min", s[0], "p10", q(0.10), "median", statistics.median(n), "p90", q(0.90), "max", s[-1])
print("cells n<10", sum(1 for x in n if x < 10), "n<50", sum(1 for x in n if x < 50), "n<500", sum(1 for x in n if x < 500))
V = sum(n)
for k in [254, 253.5]:
    if k == 0 and False: pass
    # per-visit weight u = 1/(N+k); cell total T = N*u; normalise so mean visit weight = 1
    raw = {x: 1.0/(x+k) for x in set(n)}
    W = sum(x*raw[x] for x in n)
    scale = V / W  # makes sum over visits of weight = number of visits
    tot = [(x*raw[x]*scale) for x in n]          # cell totals after normalising
    sumw = sum(tot); sumw2 = sum(x*(raw[x]*scale)**2 for x in n)
    ess = sumw**2 / sumw2
    small = sum(t for x, t in zip(n, tot) if x < 50) / sumw
    big = sum(t for x, t in zip(n, tot) if x >= 500) / sumw
    per_w_max = max(raw.values())*scale; per_w_min = min(raw.values())*scale
    print(f"k={k:>5}: Kish ESS over visits {ess:,.0f} ({ess/V:.1%} of {V:,}); share of weight in cells n<50 {small:.1%} (visits {sum(x for x in n if x<50)/V:.1%}); "
          f"cells n>=500 {big:.1%} (visits {sum(x for x in n if x>=500)/V:.1%}); per-visit weight max/min {per_w_max/per_w_min:,.1f}")
