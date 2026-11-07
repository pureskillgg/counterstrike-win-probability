import json, glob, collections
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
FIN = B + "/final"
EXP = FIN + "/export_clean_v2"

assign = json.load(open(FIN + "/split_manifest.json", encoding="utf-8"))["assign"]
print("V1 manifest assign len", len(assign), "labels", dict(collections.Counter(assign.values())))
holdout_b = [l.strip() for l in open(FIN + "/holdout_b_keys.txt", encoding="utf-8") if l.strip()]

mp = {}
lm = pq.read_table(FIN + "/laptop_matches.parquet", columns=["match_key", "map_name"]).to_pydict()
for k, v in zip(lm["match_key"], lm["map_name"]):
    mp.setdefault(k, v)
bm = pq.read_table(FIN + "/bucket_matches.parquet", columns=["match_key", "map_name"]).to_pydict()
for k, v in zip(bm["match_key"], bm["map_name"]):
    mp.setdefault(k, v)

mirage_test = [k for k, v in assign.items() if v == "test" and mp.get(k) == "de_mirage"]
excl = set(mirage_test) | set(holdout_b)
print("MIRAGE_TEST", len(mirage_test), "| HOLDOUT_B", len(holdout_b),
      "| overlap", len(set(mirage_test) & set(holdout_b)), "| excluded union", len(excl))

mk = json.load(open(EXP + "/match_keys.json"))
mks = set(mk)
print("EXPORT keys", len(mks), "| EXPORT intersect excluded", len(mks & excl))

cl = pq.read_table(FIN + "/clean_matches.parquet", columns=["match_key"]).column("match_key").to_pylist()
print("CLEAN_matches rows", len(cl), "distinct", len(set(cl)), "| CLEAN intersect excluded", len(set(cl) & excl))

up = {}
for d in sorted(glob.glob(B + "/data/c[0-9][0-9][0-9]")):
    t = pq.read_table(d + "/matches.parquet", columns=["match_key", "upload_day"]).to_pydict()
    for k, v in zip(t["match_key"], t["upload_day"]):
        up.setdefault(k, str(v)[:10].replace("-", "/"))
t = pq.read_table(FIN + "/laptop_matches.parquet", columns=["match_key", "upload_day"]).to_pydict()
for k, v in zip(t["match_key"], t["upload_day"]):
    up.setdefault(k, str(v)[:10].replace("-", "/"))

ex_up = [up.get(k) for k in mk]
known = [u for u in ex_up if u]
print("EXPORT upload_day known", len(known), "of", len(mk), "| range", min(known), max(known))
print("EXPORT keys uploaded 2026/10/26..2026/10/31", sum(1 for u in known if "2026/10/26" <= u <= "2026/10/31"))

ht = [l.strip() for l in open("/home/sagemaker-user/holdout_tome/match_keys.txt") if l.strip()]
print("HOLDOUT_TOME keys uploaded 2026/10/26..2026/10/31",
      sum(1 for k in ht if up.get(k) and "2026/10/26" <= up[k] <= "2026/10/31"), "of", len(ht))
