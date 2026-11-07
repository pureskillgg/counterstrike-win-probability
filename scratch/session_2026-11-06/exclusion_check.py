import json, glob, os, collections
import numpy as np
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
FIN = B + "/final"
EXP = FIN + "/export_clean_v2"

mk = json.load(open(EXP + "/match_keys.json"))
mks = set(mk)
print("EXPORT keys", len(mk), "distinct", len(mks))

# holdout tome (Oct 26-31) key list
ht = [l.strip() for l in open("/home/sagemaker-user/holdout_tome/match_keys.txt") if l.strip()]
print("HOLDOUT_TOME match_keys.txt lines", len(ht), "distinct", len(set(ht)), "| in export", len(set(ht) & mks))

# holdout B key list (two copies)
hb_txt = set(l.strip() for l in open(FIN + "/holdout_b_keys.txt") if l.strip())
hbj = json.load(open("/home/sagemaker-user/adam_holdout_b_keys.json"))
print("ADAM_HOLDOUT_B_KEYS json type", type(hbj).__name__,
      "len", len(hbj) if hasattr(hbj, "__len__") else "n/a")
if isinstance(hbj, dict):
    print("  json dict keys (first 5)", list(hbj.keys())[:5])
    hbj_list = []
    for v in hbj.values():
        if isinstance(v, list):
            hbj_list.extend(v)
    hb_json = set(hbj_list) if hbj_list else set()
else:
    hb_json = set(hbj)
print("HOLDOUT_B txt", len(hb_txt), "| json-derived distinct", len(hb_json),
      "| txt==json", hb_txt == hb_json, "| export∩txt", len(mks & hb_txt), "| export∩json", len(mks & hb_json))

# dates for export keys (chunk matches + laptop matches)
dates = {}
for d in sorted(glob.glob(B + "/data/c[0-9][0-9][0-9]")):
    t = pq.read_table(d + "/matches.parquet", columns=["match_key", "match_date"]).to_pydict()
    for k, v in zip(t["match_key"], t["match_date"]):
        dates.setdefault(k, str(v)[:10])
lap_names = pq.read_schema(FIN + "/laptop_matches.parquet").names
print("LAPTOP_MATCHES columns (first 25)", lap_names[:25])
if "match_date" in lap_names:
    t = pq.read_table(FIN + "/laptop_matches.parquet", columns=["match_key", "match_date"]).to_pydict()
    for k, v in zip(t["match_key"], t["match_date"]):
        dates.setdefault(k, str(v)[:10])
print("DATE sample", list(dates.items())[:2])
known = [k for k in mk if k in dates]
print("EXPORT keys with a known match_date", len(known), "of", len(mk))
oct_win = [k for k in known if "2026-10-26" <= dates[k].replace("/", "-") <= "2026-10-31"]
print("EXPORT keys dated 2026-10-26..2026-10-31", len(oct_win))
print("EXPORT max match_date", max(dates[k] for k in known))
ht_known = [k for k in ht if k in dates]
print("HOLDOUT_TOME keys with known date", len(ht_known),
      "| dated in 2026-10-26..31", sum(1 for k in ht_known if "2026-10-26" <= dates[k].replace("/", "-") <= "2026-10-31"))

# per-match pos row count: export vs pos_parts (identity-checked chunks c000-c046)
pp_cnt = collections.Counter()
for p in sorted(glob.glob(FIN + "/pos_parts/bucket_c*.parquet")):
    pp_cnt.update(pq.read_table(p, columns=["match_key"]).column("match_key").to_pylist())
rm = np.asarray(np.load(EXP + "/round_match.npy"))
po = np.asarray(np.load(EXP + "/pos_offsets.npy"))
r_rows = np.diff(po)
m_rows = np.bincount(rm, weights=r_rows, minlength=len(mk)).astype(np.int64)
idx = {k: i for i, k in enumerate(mk)}
inA = [k for k in pp_cnt if k in idx]
mism = [(k, pp_cnt[k], int(m_rows[idx[k]])) for k in inA if pp_cnt[k] != int(m_rows[idx[k]])]
print("PER-MATCH pos rows: pos_parts keys in export", len(inA), "| mismatched counts", len(mism), "| sample", mism[:5])
notin = [k for k in pp_cnt if k not in idx]
print("POS_PARTS keys NOT in export", len(notin), "| their pos rows", sum(pp_cnt[k] for k in notin))
print("EXPORT rows for pos_parts keys", int(sum(m_rows[idx[k]] for k in inA)), "| pos_parts rows total", sum(pp_cnt.values()))
