import gzip, json, os, sys, collections, glob
from concurrent.futures import ThreadPoolExecutor
stage = os.path.join(sys.argv[1], "csds", "2024", "02")
want = [l.strip() for l in open(sys.argv[2], encoding="utf-8-sig") if l.strip()]
manifests = glob.glob(os.path.join(stage, "*", "*", "csds"))
def decl(p):
    raw = open(p, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return {c["channel"] for c in json.loads(raw)["channels"]}
with ThreadPoolExecutor(16) as ex:
    sets = list(ex.map(decl, manifests))
print("staged manifests:", len(manifests))
out_tomes = set(n.split(".")[0][len("basic_"):] for n in os.listdir(os.path.join(sys.argv[3], "tome", "csds")) if n.startswith("basic_"))
rows = []
for ch in want:
    n = sum(ch in s for s in sets)
    rows.append((ch, n, ch in out_tomes))
for ch, n, built in rows:
    print(f"{ch:20s} declared_in={n:5d} built={built}")
