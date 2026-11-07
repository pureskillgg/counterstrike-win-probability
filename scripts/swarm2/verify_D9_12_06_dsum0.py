"""verify D9-12 step 6: where the score sum changes for 2024 rounds whose round_end tick shows no change
(dsum == 0, all in matches where round_end label 1 exists), and how many rounds per match are affected.
Also counts the 2024 round_end rows that cross_period-style membership would drop (no round_start label).

Read-only. Uses the same label definitions as step 4.
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
re_raw = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
st = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
sc = (st.filter(pl.col("t_score").is_not_null() & pl.col("ct_score").is_not_null())
      .with_columns((pl.col("t_score") + pl.col("ct_score")).alias("ssum"))
      .select(["match_key", "tick", "ssum", "t_score", "ct_score", "event_type"]).sort("tick"))
rs = rs_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("rs"))
re = re_raw.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("re"),
                                                 pl.col("winner_team_code").first().alias("winner"),
                                                 pl.col("player_count").first().alias("pc"))
fe = st.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("fe"))

pre = (re.with_columns((pl.col("re") - 1).alias("_t")).sort("_t")
       .join_asof(sc.rename({"tick": "_t"}).select(["match_key", "_t", "ssum"]), on="_t", by="match_key",
                  strategy="backward").rename({"ssum": "sum_pre"}))
post = (re.with_columns(pl.col("re").alias("_t")).sort("_t")
        .join_asof(sc.rename({"tick": "_t"}).select(["match_key", "_t", "ssum"]), on="_t", by="match_key",
                   strategy="backward").rename({"ssum": "sum_re"}))
W = (re.join(pre.select(["match_key", "round", "sum_pre"]), on=["match_key", "round"])
     .join(post.select(["match_key", "round", "sum_re"]), on=["match_key", "round"])
     .with_columns((pl.col("sum_re") - pl.col("sum_pre")).alias("dsum")))

mstat = re.group_by("match_key").agg(pl.col("round").min().alias("re_min"))
W = W.join(mstat, on="match_key")
print("dsum by re_min:", W.group_by(["re_min", "dsum"]).len().sort(["re_min", "dsum"]).rows())

z = W.filter(pl.col("dsum") == 0)
print("dsum==0 rounds:", z.height, "| matches:", z["match_key"].n_unique(),
      "| per-match counts (top 8):", z.group_by("match_key").len().sort("len", descending=True)["len"].head(8).to_list())
print("dsum==0 rounds by player_count (2024 modal pc is 20):", z.group_by("pc").len().sort("len", descending=True).head(4).rows(), "| winner counts:",
      z.group_by("winner").len().sort("winner").rows())

# show the score-row timeline between fe(r) and rs(r+1) for three dsum==0 rounds
samp = z.join(fe, on=["match_key", "round"], how="left").sort(["match_key", "round"]).head(3)
for row in samp.iter_rows(named=True):
    mk, r, t_end = row["match_key"], row["round"], row["re"]
    print(f"\n-- match {mk[-40:]} label {r}: fe={row['fe']} re={t_end} winner={row['winner']} sum_pre={row['sum_pre']} sum_re={row['sum_re']}")
    win = sc.filter((pl.col("match_key") == mk) & (pl.col("tick") >= (row["fe"] or t_end) - 200)
                    & (pl.col("tick") <= t_end + 400))
    print("   score rows (tick, t, ct, sum, event):", win.select(["tick", "t_score", "ct_score", "ssum", "event_type"]).rows())
    evs = st.filter((pl.col("match_key") == mk) & (pl.col("tick") >= t_end - 5) & (pl.col("tick") <= t_end + 5)
                    & pl.col("event_type").is_in(["round_end", "cs_win_panel_round", "round_officially_ended"]))
    print("   end-time state events:", evs.select(["tick", "round", "event_type"]).rows())

# membership view used by cross_period / bakeoff (label-keyed): 2024 round_end rows with no round_start label
miss = re.join(rs.select(["match_key", "round"]).with_columns(pl.lit(1).alias("has_rs")),
               on=["match_key", "round"], how="left").filter(pl.col("has_rs").is_null())
print("\n2024 round_end (match,label) with no round_start label:", miss.height,
      "| of which pc==20:", miss.filter(pl.col("pc") == 20).height,
      "| distinct matches:", miss["match_key"].n_unique())
lastr = re.group_by("match_key").agg(pl.col("round").max().alias("rmax"))
mm = miss.join(lastr, on="match_key").with_columns((pl.col("round") == pl.col("rmax")).alias("is_last"))
print("   of which is the match's last round_end label:", mm.filter(pl.col("is_last")).height)
print("   of which pc==20 and last:", mm.filter(pl.col("is_last") & (pl.col("pc") == 20)).height)
# round-1 view: matches with a round_end label 1 (physical round 1 present) and label-1 start row present
r1 = re.filter(pl.col("round") == 1).height
r1_fe = fe.filter(pl.col("round") == 1).height
print("physical round-1 candidates: round_end label 1 rows:", r1, "| freeze_end label 1 rows:", r1_fe,
      "| round_start label 1 rows (physically round 2 by score):", rs.filter(pl.col("round") == 1).height)
