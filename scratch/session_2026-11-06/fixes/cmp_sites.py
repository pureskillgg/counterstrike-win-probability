import json
a = json.load(open("laptop_sites_v1.json", encoding="utf-8"))
b = json.load(open("space_sites_v1.json", encoding="utf-8"))
print("maps laptop", sorted(a), "| space", sorted(b), "| same map set", set(a) == set(b))
worst = 0.0
for m in sorted(set(a) & set(b)):
    d = max(abs(x - y) for pa, pb in zip(a[m], b[m]) for x, y in zip(pa, pb))
    worst = max(worst, d)
    print(f"  {m:12s} laptop {[[round(v, 1) for v in p] for p in a[m]]} | space max abs diff {d:.6f}")
print("worst per-coordinate difference (game units):", round(worst, 9))
