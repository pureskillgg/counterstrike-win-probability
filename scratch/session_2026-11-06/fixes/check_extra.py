import glob
import json
import os
import re
from collections import Counter

import numpy as np
import polars as pl

FIN = "/home/sagemaker-user/adam/builder/final"
A = FIN + "/export_clean"
B = FIN + "/export_clean_v2"
ka = json.load(open(A + "/match_keys.json"))
kb = json.load(open(B + "/match_keys.json"))
sa, sb = set(ka), set(kb)
extra = sa - sb
print("export_clean matches", len(sa), "| v2 matches", len(sb), "| in clean not v2", len(extra), "| in v2 not clean", len(sb - sa))

rm_a = np.load(A + "/round_match.npy")
cnt_rounds = np.bincount(rm_a, minlength=len(ka))
extra_ids = [i for i, k in enumerate(ka) if k in extra]
print("rounds in clean for the extra matches", int(sum(cnt_rounds[i] for i in extra_ids)))


def ym(k):
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})/", k)
    return (m.group(1), m.group(2), m.group(3)) if m else ("?", "?", "?")


dates = Counter("%s-%s" % ym(k)[1:] for k in extra)
print("extra matches by month-day counts", dict(sorted(dates.items())))
print("extra matches dated 2026-10-26 or later (count)", sum(1 for k in extra if ym(k) >= ("2026", "10", "26")))
print("extra matches dated 2026-10-21 to 10-25 (count)", sum(1 for k in extra if ("2026", "10", "21") <= ym(k) <= ("2026", "10", "25")))

hb_raw = open(FIN + "/holdout_b_keys.txt").read().strip()
try:
    hb = set(json.loads(hb_raw))
except Exception:
    hb = set(x.strip() for x in hb_raw.splitlines() if x.strip())
print("holdout_b_keys.txt entries", len(hb), "| holdout B in extra (count)", len(hb & extra), "| holdout B in v2 export (count)", len(hb & sb), "| holdout B in clean (count)", len(hb & sa))

cr = pl.read_parquet(FIN + "/clean_rounds.parquet", columns=["match_key", "use", "src"])
ex_cr = cr.filter(pl.col("match_key").is_in(list(extra)))
print("clean_rounds rows for extra matches", ex_cr.height, "| distinct matches", ex_cr["match_key"].n_unique())
print("extra matches with src/use combos (count of distinct matches):")
print(ex_cr.group_by("src", "use").agg(pl.col("match_key").n_unique().alias("matches")).sort("src").to_dicts())

present = set()
for f in sorted(glob.glob(FIN + "/pos_parts/bucket_c*.parquet")):
    present |= set(pl.read_parquet(f, columns=["match_key"])["match_key"].unique().to_list())
print("pos_parts distinct match keys", len(present), "| extra matches present in pos_parts (count)", len(present & extra))
print("pos_parts keys not in export_clean (count)", len(present - sa))
