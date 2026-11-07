"""verify D9-12 step 8: 2024 breakdown of the round_end winner vs side-score check by round label (1..24),
split by failure type, to see whether the failures are the round_start offset, the halftime swap, or timing.

  type 'ok'      : winner side score rises by exactly 1 across the round_end tick
  type 'nochg'   : no score change across the round_end tick (score row already updated before the tick)
  type 'cross'   : a score change occurs, but on the other side from the winner code
  type 'other'   : any other pattern

ASSUMPTION: winner_team_code 2 = T side, 3 = CT side, and t_score / ct_score are side-based (they swap at halftime).
Read-only.
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


re_raw = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
st = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
re = re_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("re"),
                                                 pl.col("winner_team_code").first().alias("winner"),
                                                 pl.col("player_count").first().alias("pc"))
sc = (st.filter(pl.col("t_score").is_not_null() & pl.col("ct_score").is_not_null())
      .select(["match_key", "tick", "t_score", "ct_score"]).sort("tick"))


def side_at(df, col, pfx):
    left = df.select(["match_key", "round", col]).sort(col).with_columns(pl.col(col).alias("_t"))
    out = left.join_asof(sc.rename({"tick": "_t"}), on="_t", by="match_key", strategy="backward")
    return out.select(["match_key", "round", pl.col("t_score").alias(pfx + "t"), pl.col("ct_score").alias(pfx + "ct")])


post = side_at(re.select(["match_key", "round", "re"]), "re", "post_")
pre = side_at(re.select(["match_key", "round", "re"]).with_columns((pl.col("re") - 1).alias("re_m1")), "re_m1", "pre_")
W = re.join(post, on=["match_key", "round"]).join(pre, on=["match_key", "round"])
W = W.with_columns((pl.col("post_t") - pl.col("pre_t")).alias("dt"),
                   (pl.col("post_ct") - pl.col("pre_ct")).alias("dct"))
ok = (((pl.col("winner") == 2) & (pl.col("dt") == 1) & (pl.col("dct") == 0)) |
      ((pl.col("winner") == 3) & (pl.col("dct") == 1) & (pl.col("dt") == 0)))
nochg = (pl.col("dt") == 0) & (pl.col("dct") == 0)
cross = (((pl.col("winner") == 2) & (pl.col("dct") == 1) & (pl.col("dt") == 0)) |
         ((pl.col("winner") == 3) & (pl.col("dt") == 1) & (pl.col("dct") == 0)))
W = W.with_columns(pl.when(ok).then(pl.lit("ok")).when(nochg).then(pl.lit("nochg"))
                   .when(cross).then(pl.lit("cross")).otherwise(pl.lit("other")).alias("kind"))

print("overall kind counts:", W.group_by("kind").len().sort("kind").rows())
print("\nkind by label (labels 1-24):")
tab = (W.filter(pl.col("round") <= 24).group_by(["round", "kind"]).len()
       .pivot(on="kind", index="round", values="len").fill_null(0).sort("round"))
print(tab)
print("\npc (player_count) distribution of 'cross' rounds:", W.filter(pl.col("kind") == "cross").group_by("pc").len()
      .sort("len", descending=True).head(5).rows())
print("pc distribution of 'nochg' rounds:", W.filter(pl.col("kind") == "nochg").group_by("pc").len()
      .sort("len", descending=True).head(5).rows())
print("'cross' rounds by winner code:", W.filter(pl.col("kind") == "cross").group_by("winner").len().sort("winner").rows())
# split by build_num (header key == match_key) and by whether round 1 is present in the match
ROOT_H = BASE / "header"
import glob as _g
H = pl.scan_parquet(sorted(_g.glob(str(ROOT_H / "dataframe_*")))).select(["key", "build_num"]).unique(subset=["key"]).collect()
Wb = W.join(H.rename({"key": "match_key"}), on="match_key", how="left")
first = re.group_by("match_key").agg(pl.col("round").min().alias("first_re_label"))
Wb = Wb.join(first, on="match_key", how="left")
print("\nbuild_num x first round_end label (1 = round 1 present):",
      first.join(H.rename({"key": "match_key"}), on="match_key", how="left").group_by(["build_num", "first_re_label"])
      .len().sort(["build_num", "first_re_label"]).rows())
print("kind counts by build_num (all labels):", Wb.group_by(["build_num", "kind"]).len().sort(["build_num", "kind"]).rows())
print("'cross' rounds by build_num and label (labels 9 and 13 only):",
      Wb.filter((pl.col("kind") == "cross") & pl.col("round").is_in([9, 13])).group_by(["build_num", "round"]).len()
      .sort(["build_num", "round"]).rows())
print("'cross' rounds by build_num, all labels:", Wb.filter(pl.col("kind") == "cross").group_by("build_num").len().sort("build_num").rows())
print("label-13 rounds by build_num and kind:", Wb.filter(pl.col("round") == 13).group_by(["build_num", "kind"]).len()
      .sort(["build_num", "kind"]).rows())
print("'cross' example rows (match tail, label, winner, dt, dct, pc):")
ex = W.filter(pl.col("kind") == "cross").select(["match_key", "round", "winner", "dt", "dct", "pc"]).head(6)
for r in ex.iter_rows(named=True):
    print("   ", r["match_key"][-12:], r["round"], r["winner"], r["dt"], r["dct"], r["pc"])
