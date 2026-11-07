"""verify_D1_5 debug: why do round_end-derived team totals miss the header in clean matches?
Prints per-round winners, score side (A4) and team-A side for a few complete clean matches."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols=None):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fl(ch)], how="vertical_relaxed")


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def side_a(r, mr):
    n = sum(1 for s in range(1, r) if swap_expected(s, mr))
    return 2 if n % 2 == 0 else 3


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count", "win_reason_message"]).unique(
    ["match_key", "round"], keep="first")
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
HD0 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))],
                how="vertical_relaxed")
HD = HD0.select([pl.col("key").alias("match_key"), "t_starters_score_final", "ct_starters_score_final"]).unique("match_key")

# per round: score side from round_state first/last tick (A4)
sc = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("t_score").sort_by("tick").first().alias("t0"), pl.col("ct_score").sort_by("tick").first().alias("c0"),
    pl.col("t_score").sort_by("tick").last().alias("t1"), pl.col("ct_score").sort_by("tick").last().alias("c1")))
sc = sc.with_columns(pl.when((pl.col("t1") - pl.col("t0")) == 1).then(2)
                     .when((pl.col("c1") - pl.col("c0")) == 1).then(3).otherwise(None).cast(pl.Int64).alias("score_side"),
                     (pl.col("t1") - pl.col("t0")).alias("dT"), (pl.col("c1") - pl.col("c0")).alias("dC"))

# final round_state values per match
fin = (RS.sort(["match_key", "tick"]).group_by("match_key").agg(
    pl.col("t_score").last().alias("t_fin"), pl.col("ct_score").last().alias("c_fin")))

J = (RE.join(HD, on="match_key", how="inner").join(sc.select(["match_key", "round", "score_side", "dT", "dC"]),
                                                  on=["match_key", "round"], how="left")
     .join(fin, on="match_key", how="left"))
J = J.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))

# complete contiguous matches with n == tf + cf (A6), 24 max rounds
cnt = J.group_by("match_key").agg(pl.len().alias("n"), pl.col("round").max().alias("maxr"),
                                   pl.col("t_starters_score_final").first().alias("tf"),
                                   pl.col("ct_starters_score_final").first().alias("cf"),
                                   pl.col("t_fin").first().alias("t_fin"), pl.col("c_fin").first().alias("c_fin"))
print("matches", cnt.height)
print("n vs maxr equal:", cnt.filter(pl.col("n") == pl.col("maxr")).height)
print("n == tf+cf:", cnt.filter(pl.col("n") == pl.col("tf") + pl.col("cf")).height)
print("max round distribution (top):")
print(cnt.group_by("maxr").len().sort("len", descending=True).head(8))
print("header tf+cf vs n for mismatched count (first 8):")
print(cnt.filter(pl.col("n") != pl.col("tf") + pl.col("cf")).head(8))

# sample clean-looking 'neither' matches: n == tf+cf and contiguous
good = cnt.filter((pl.col("n") == pl.col("tf") + pl.col("cf")) & (pl.col("n") == pl.col("maxr")))
print("complete contiguous matches:", good.height)
print("final rs totals vs header (T field, CT field):")
print(good.with_columns(((pl.col("t_fin") == pl.col("tf")) & (pl.col("c_fin") == pl.col("cf"))).alias("rs_ok_direct"),
                        ((pl.col("t_fin") == pl.col("cf")) & (pl.col("c_fin") == pl.col("tf"))).alias("rs_ok_swapped"))
        .group_by(["rs_ok_direct", "rs_ok_swapped"]).len())

sample = good.sort("match_key").head(4)["match_key"].to_list()
for mk in sample:
    x = J.filter(pl.col("match_key") == mk).sort("round").select(
        ["round", "winner_team_code", "win_reason_message", "score_side", "dT", "dC", "player_count"])
    tf = cnt.filter(pl.col("match_key") == mk)["tf"][0]
    cf = cnt.filter(pl.col("match_key") == mk)["cf"][0]
    print("\n=====", mk, "header tf (team A on T at start) =", tf, "cf =", cf)
    print(x)
    w = dict(zip(x["round"].to_list(), x["winner_team_code"].to_list()))
    n = len(w)
    rd_a = sum(1 for r in range(1, n + 1) if w[r] == side_a(r, 24))
    print("side_a per round:", [side_a(r, 24) for r in range(1, n + 1)])
    print("rd_a (round_end) =", rd_a, " rd_b =", n - rd_a)
