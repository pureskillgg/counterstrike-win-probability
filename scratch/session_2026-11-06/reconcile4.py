import json, os, glob
from collections import Counter
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
F = B + "/final"
X = F + "/export_clean_v2"
HOME = "/home/sagemaker-user"

export = set(json.load(open(X + "/match_keys.json")))
V = set(json.load(open(F + "/export_clean/match_keys.json")))

cr = pq.read_table(F + "/clean_rounds.parquet", columns=["match_key", "use", "src"]).to_pydict()
bkeys = set(k for k, u, s in zip(cr["match_key"], cr["use"], cr["src"]) if u and s == "bucket")
src_by_key = {}
for k, u, s in zip(cr["match_key"], cr["use"], cr["src"]):
    if u:
        src_by_key.setdefault(k, set()).add(s)
print("CLEAN_ROUNDS use&bucket keys=%d" % len(bkeys))
print("V_in_bkeys=%d V_not_in_bkeys=%d" % (len(V & bkeys), len(V - bkeys)))
print("EXPORT_in_bkeys=%d EXPORT_not_in_bkeys=%d" % (len(export & bkeys), len(export - bkeys)))

tot = Counter()
per = {}
for i in range(68):
    c = "c%03d" % i
    mk = pq.read_table("%s/data/%s/matches.parquet" % (B, c), columns=["match_key"])["match_key"].to_pylist()
    s = set(mk)
    vc = V & s
    in_b = len(vc & bkeys)
    per[c] = (len(vc), in_b, len(vc) - in_b)
    tot["v"] += len(vc)
    tot["v_in_bkeys"] += in_b
print("PER_CHUNK c: v_in_export_clean, v_in_bkeys(src=bucket use), v_src_not_bucket")
for c in sorted(per):
    print("PB", c, per[c])
print("TOTAL_V=%d V_in_bkeys=%d" % (tot["v"], tot["v_in_bkeys"]))

# laptop_pos coverage for the export keys with no bucket pos rows
lp_dir = B + "/laptop_pos"
files = sorted(glob.glob(lp_dir + "/every2_sites_v1/*.parquet"))
print("LAPTOP_POS_FILES n=%d dir=%s" % (len(files), lp_dir + "/every2_sites_v1"))
lp_keys = set()
for f in files:
    try:
        lp_keys |= set(pq.read_table(f, columns=["match_key"]).column("match_key").to_pylist())
    except Exception as e:
        print("READ_ERR", f, str(e)[:120])
print("LAPTOP_POS_DISTINCT_MATCH_KEYS=%d" % len(lp_keys))
no_bucket_pos = (export - bkeys)
print("EXPORT_NOT_IN_BKEYS (no bucket pos) =%d covered_by_laptop_pos=%d" % (len(no_bucket_pos), len(no_bucket_pos & lp_keys)))
print("EXPORT_NOT_IN_BKEYS sample_src=%s" % Counter(tuple(sorted(src_by_key.get(k, ['NA']))) for k in no_bucket_pos).most_common(4))
print("EXPORT_IN_BKEYS covered_by_laptop_pos=%d" % len((export & bkeys) & lp_keys))
