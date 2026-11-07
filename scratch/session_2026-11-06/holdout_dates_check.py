import glob, os, collections
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
FIN = B + "/final"
EXP = FIN + "/export_clean_v2"

ht = [l.strip() for l in open("/home/sagemaker-user/holdout_tome/match_keys.txt") if l.strip()]
hts = set(ht)
meta = {}
for d in sorted(glob.glob(B + "/data/c[0-9][0-9][0-9]")):
    t = pq.read_table(d + "/matches.parquet", columns=["match_key", "match_date", "upload_day"]).to_pydict()
    for k, md, ud in zip(t["match_key"], t["match_date"], t["upload_day"]):
        meta.setdefault(k, (str(md)[:10], str(ud)[:10]))
t = pq.read_table(FIN + "/laptop_matches.parquet", columns=["match_key", "match_date", "upload_day"]).to_pydict()
for k, md, ud in zip(t["match_key"], t["match_date"], t["upload_day"]):
    meta.setdefault(k, (str(md)[:10], str(ud)[:10]))

cl = pq.read_table(FIN + "/clean_matches.parquet", columns=["match_key", "src", "in_scope", "n_ok_rounds"]).to_pydict()
clm = {k: (s, i, n) for k, s, i, n in zip(cl["match_key"], cl["src"], cl["in_scope"], cl["n_ok_rounds"])}

print("HOLDOUT_TOME keys", len(ht), "| with metadata", sum(1 for k in ht if k in meta), "| in clean_matches", sum(1 for k in ht if k in clm))
win = [k for k in ht if k in meta and "2026-10-26" <= meta[k][0].replace("/", "-") <= "2026-10-31"]
out = [k for k in ht if k in meta and not ("2026-10-26" <= meta[k][0].replace("/", "-") <= "2026-10-31")]
print("in window by match_date", len(win), "| outside window", len(out))
print("outside-window match_date month counts", collections.Counter(meta[k][0][:7] for k in out).most_common(6))
print("outside-window match_date range", min(meta[k][0] for k in out), max(meta[k][0] for k in out))
print("outside-window upload_day sample counts", collections.Counter(meta[k][1][:10] for k in out).most_common(4))
print("outside-window in clean_matches", sum(1 for k in out if k in clm), "| src/in_scope/n_ok_rounds sample",
      [clm[k] for k in out if k in clm][:5])
print("in-window in clean_matches", sum(1 for k in win if k in clm))
