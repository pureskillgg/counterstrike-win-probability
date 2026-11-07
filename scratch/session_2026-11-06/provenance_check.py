import json, glob, os, collections
import numpy as np
import pyarrow.parquet as pq

B = "/home/sagemaker-user/adam/builder"
FIN = B + "/final"
EXP = FIN + "/export_clean_v2"


def keys_of(path):
    return pq.read_table(path, columns=["match_key"]).column("match_key").to_pylist()


mk = json.load(open(EXP + "/match_keys.json"))
mks = set(mk)

bucket_keys = {}
for d in sorted(glob.glob(B + "/data/c[0-9][0-9][0-9]")):
    c = os.path.basename(d)
    bucket_keys[c] = set(keys_of(d + "/matches.parquet"))
lo_keys = set().union(*[v for c, v in bucket_keys.items() if int(c[1:]) <= 46])
hi_keys = set().union(*[v for c, v in bucket_keys.items() if int(c[1:]) >= 47])
pp_keys = set()
for p in sorted(glob.glob(FIN + "/pos_parts/bucket_c*.parquet")):
    pp_keys |= set(keys_of(p))
print("LO keys", len(lo_keys), "HI keys", len(hi_keys), "lo&hi", len(lo_keys & hi_keys), "pos_parts keys", len(pp_keys))

lap_paths = [B + "/data/laptop/matches.parquet", B + "/data/laptop_old/matches.parquet", FIN + "/laptop_matches.parquet"]
lap = {}
for p in lap_paths:
    if os.path.exists(p):
        try:
            lap[p] = set(keys_of(p))
        except Exception as e:
            print("LAPTOP read error", p, str(e)[:120])
for p, s in lap.items():
    print("LAPTOP", p.replace(B, "B"), "keys", len(s),
          "in export", len(s & mks), "in hi", len(s & hi_keys), "in lo", len(s & lo_keys))

grp = {}
for k in mk:
    if k in pp_keys:
        grp[k] = "A_pos_parts_c000-c046"
    elif k in lo_keys:
        grp[k] = "B_lo_chunk_not_in_pos"
    elif k in hi_keys:
        grp[k] = "C_hi_chunk_c047-c067_only"
    else:
        grp[k] = "D_no_bucket_chunk"
print("GROUPS", dict(collections.Counter(grp.values())))
lap_all = set().union(*lap.values()) if lap else set()
for g in sorted(set(grp.values())):
    ks = [k for k in mk if grp[k] == g]
    print("GROUP", g, "n", len(ks), "in any laptop file", sum(k in lap_all for k in ks))

rm = np.asarray(np.load(EXP + "/round_match.npy"))
po = np.asarray(np.load(EXP + "/pos_offsets.npy"))
pi8 = np.load(EXP + "/pos_int8.npy", mmap_mode="r")
chi = np.asarray(pi8[:, 0])
sent = (chi == -1).astype(np.int64)
cs = np.concatenate([[0], np.cumsum(sent)])
r_sent = cs[po[1:]] - cs[po[:-1]]
r_rows = np.diff(po)
m_rows = np.bincount(rm, weights=r_rows, minlength=len(mk))
m_sent = np.bincount(rm, weights=r_sent, minlength=len(mk))
print("TOTAL pos rows", int(r_rows.sum()), "total sentinel c_site_hi", int(r_sent.sum()))

stat = collections.defaultdict(lambda: [0, 0, 0, 0])
for i, k in enumerate(mk):
    s = stat[grp[k]]
    s[0] += 1
    s[1] += int(m_rows[i])
    s[2] += int(m_sent[i])
    s[3] += int(m_rows[i] == 0)
for g in sorted(stat):
    n, rows, snt, zero = stat[g]
    print("SENT", g, "matches", n, "pos_rows", rows, "sentinel_c_site_hi", snt,
          "sentinel_share", round(snt / max(rows, 1), 5), "zero_row_matches", zero)

pp_rows = 0
for p in sorted(glob.glob(FIN + "/pos_parts/bucket_c*.parquet")):
    pp_rows += pq.ParquetFile(p).metadata.num_rows
print("POS_PARTS total rows", pp_rows, "| export pos rows for group A", sum(int(m_rows[i]) for i, k in enumerate(mk) if grp[k].startswith("A_")))

for d in [FIN + "/pos_parity", FIN + "/pos_check"]:
    for p in sorted(glob.glob(d + "/*.parquet")):
        ks = set(keys_of(p))
        print("PARITY/CHECK", p.replace(B, "B"), "rows", pq.ParquetFile(p).metadata.num_rows,
              "keys", len(ks), "in hi", len(ks & hi_keys), "in lo", len(ks & lo_keys), "in export", len(ks & mks))
