import os, sys, csv
from concurrent.futures import ThreadPoolExecutor
SRC = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02"
base = sys.argv[1]
want = set(l.strip() for l in open(sys.argv[2], encoding="utf-8-sig") if l.strip()) | {"header", "csds"}
rows = [r for r in csv.DictReader(open(sys.argv[3], encoding="utf-8")) if r["status"] == "ok" and r["day"] != "01"]
def size(r):
    d = os.path.join(SRC, r["day"], r["match_folder"])
    all_b = need_b = 0
    for e in os.scandir(d):
        if e.is_file():
            all_b += e.stat().st_size
            if e.name in want:
                need_b += e.stat().st_size
    return all_b, need_b
with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(size, rows))
A = sum(a for a, _ in res); N = sum(n for _, n in res)
print(f"matches={len(rows)} all_bytes={A/1e9:.1f} GB needed_bytes={N/1e9:.1f} GB ratio={N/A:.2f}")
