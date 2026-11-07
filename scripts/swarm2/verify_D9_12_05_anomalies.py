"""verify D9-12 step 5: examine the 2024 round_end score anomalies found in step 4 (dsum 0 / 2, sum-label +1),
and check whether the round_start offset is uniform across matches or confined to some matches.

Read-only. Same label definitions as step 4 (rs = basic round_start min tick; re = basic round_end min tick and
winner; fe = round_state round_freeze_end min tick; score sum = t_score + ct_score from round_state rows).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def scan(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    lf = pl.scan_parquet(fs)
    have = set(lf.collect_schema().names())
    return lf.select([c for c in cols if c in have]).collect()


rs_raw = scan("round_start", ["match_key", "round", "tick"])
re_raw = scan("round_end", ["match_key", "round", "tick", "winner_team_code"])
st = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])

rs = rs_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("rs"))
re = re_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("re"),
                                                 pl.col("winner_team_code").first().alias("winner"))
sc = (st.filter(pl.col("t_score").is_not_null() & pl.col("ct_score").is_not_null())
      .with_columns((pl.col("t_score") + pl.col("ct_score")).alias("ssum"))
      .select(["match_key", "tick", "ssum", "t_score", "ct_score"]).sort("tick"))

# per match: first round_end label present (1 or 2) and number of round_end labels
mstat = re.group_by("match_key").agg(pl.col("round").min().alias("re_min"), pl.len().alias("n_re"),
                                     pl.col("round").max().alias("re_max"))

# score sum just before and at the round_end tick
pre = (re.with_columns((pl.col("re") - 1).alias("_t")).sort("_t")
       .join_asof(sc.rename({"tick": "_t"}).select(["match_key", "_t", "ssum"]),
                  on="_t", by="match_key", strategy="backward")
       .rename({"ssum": "sum_pre"}))
post = (re.sort("re").with_columns(pl.col("re").alias("_t"))
        .join_asof(sc.rename({"tick": "_t"}).select(["match_key", "_t", "ssum"]),
                   on="_t", by="match_key", strategy="backward")
        .rename({"ssum": "sum_re"}))
W = (re.join(pre.select(["match_key", "round", "sum_pre"]), on=["match_key", "round"])
     .join(post.select(["match_key", "round", "sum_re"]), on=["match_key", "round"])
     .join(mstat, on="match_key", how="left")
     .with_columns((pl.col("sum_re") - pl.col("round")).alias("dev_re"),
                   (pl.col("sum_re") - pl.col("sum_pre")).alias("dsum")))
print("round_end rows:", W.height)
print("dev_re (sum after end - label) by re_min class:")
print(W.group_by(["re_min", "dev_re"]).len().sort(["re_min", "dev_re"]))
print("dsum by re_min class:")
print(W.group_by(["re_min", "dsum"]).len().sort(["re_min", "dsum"]))

# where do the +1 cases sit: round label distribution and per-match concentration
odd = W.filter(pl.col("dev_re") == 1)
print("\n+1 (dev_re) cases: distinct matches", odd["match_key"].n_unique(),
      "| round label dist (top 10):", odd.group_by("round").len().sort("len", descending=True).head(10).rows())
per_match = odd.group_by("match_key").len().sort("len", descending=True)
print("cases per match (top 10 counts):", per_match["len"].head(10).to_list(),
      "| matches with >=1:", per_match.height)

dz = W.filter(pl.col("dsum") == 0)
print("\ndsum==0 cases: distinct matches", dz["match_key"].n_unique(),
      "| round label dist (top 10):", dz.group_by("round").len().sort("len", descending=True).head(10).rows())

# show raw score rows around three example +1 rounds and one normal round in the same match
examples = odd.select(["match_key", "round", "re"]).head(3).rows()
normal = W.filter((pl.col("dev_re") == 0) & (pl.col("dsum") == 1)).select(["match_key", "round", "re"]).head(1).rows()
for mk, rnd, tick in examples + normal:
    tag = "ODD(+1)" if (mk, rnd, tick) in examples else "NORMAL"
    print(f"\n-- {tag} match {mk[-37:-1] if len(mk) > 40 else mk} label {rnd} round_end tick {tick}")
    prev_fe = st.filter((pl.col("match_key") == mk) & (pl.col("event_type") == "round_freeze_end")
                        & (pl.col("round") >= rnd - 1) & (pl.col("round") <= rnd + 1)).select(["round", "tick"])
    print("   freeze_end rows (label, tick):", prev_fe.sort("tick").rows())
    rsr = rs.filter((pl.col("match_key") == mk) & (pl.col("round") >= rnd - 1) & (pl.col("round") <= rnd + 1))
    print("   basic round_start rows (label, tick):", rsr.sort("round").select(["round", "rs"]).rows())
    near = sc.filter((pl.col("match_key") == mk) & (pl.col("tick") >= tick - 1500) & (pl.col("tick") <= tick + 1500))
    print("   score rows within +-1500 ticks (tick, t, ct, sum):",
          near.select(["tick", "t_score", "ct_score", "ssum"]).rows()[:14])

# uniformity: fraction of matches that carry the round_start offset (rs label r vs score sum r)
fe = st.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("fe"))
rs_sc = (rs.sort("rs").with_columns(pl.col("rs").alias("_t"))
         .join_asof(sc.rename({"tick": "_t"}).select(["match_key", "_t", "ssum"]),
                    on="_t", by="match_key", strategy="backward"))
rs_dev = rs_sc.with_columns((pl.col("ssum") - pl.col("round")).alias("dev"))
per_m = rs_dev.group_by("match_key").agg(pl.col("dev").n_unique().alias("nd"), pl.col("dev").min().alias("dmin"),
                                         pl.col("dev").max().alias("dmax"), pl.len().alias("n"))
print("\nround_start, matches with a single offset value:", per_m.filter(pl.col("nd") == 1).height,
      "of", per_m.height, "| offset values:", per_m.group_by(["dmin", "dmax"]).len().rows())
