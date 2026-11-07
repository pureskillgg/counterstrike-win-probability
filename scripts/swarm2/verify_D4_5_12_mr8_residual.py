"""verify D4-5, step 12: MR8 round-9 residual. Complete 2024 matches with an MR8 halftime event (label 7), days 12-25.

For each match: max round label (regulation 16 or overtime), disputed rounds, X_fix - hT (score-side fix applied to
rounds 9, 13, 25, 28 with the per-match convention), and whether the match has any round-9 disagreement.
Also the MR12 pattern-'13' matches with an overtime-free finish, for contrast.
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, T, CT  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
base = TOMES["2024"]
DISP = [9, 13, 25, 28]
rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code"]).collect().sort(
    ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
H = header(base, ["key", "t_starters_score_final", "ct_starters_score_final"]).rename(
    {"key": "match_key", "t_starters_score_final": "hT", "ct_starters_score_final": "hC"})
bs = scan(base, "bomb_state", ["match_key", "round", "tick", "event_type"]).collect()
bo = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
      .sort("tick").group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("bev")))
half = rs.filter(pl.col("event_type") == "round_announce_last_round_half").group_by("match_key").agg(
    pl.col("round").min().alias("half_label"))
fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"]))
fe_next = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t1", "ct_score": "c1"})
Dp = fe.join(fe_next, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t1") - pl.col("t_score")).alias("dT"), (pl.col("c1") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("ss"))
M = re_.group_by("match_key").agg(pl.len().alias("n_RE"), pl.col("round").min().alias("min_label"),
                                  pl.col("round").max().alias("max_label")).join(H, on="match_key").join(
    half, on="match_key", how="left")
M = M.with_columns(
    pl.when(pl.col("half_label") == 7).then(pl.lit(16)).otherwise(pl.lit(24)).alias("mr"),
    (pl.col("hT") + pl.col("hC") == pl.col("n_RE")).alias("complete"),
    (pl.col("min_label") == 1).alias("label1"),
    pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32).alias("day"))
M = M.filter(pl.col("complete") & pl.col("label1") & pl.col("day").is_between(12, 25) & (pl.col("mr") == 16))
keep = M["match_key"].to_list()
R = re_.filter(pl.col("match_key").is_in(keep)).join(Dp.select(["match_key", "round", "ss"]),
                                                       on=["match_key", "round"], how="left").join(
    bo, on=["match_key", "round"], how="left").join(M.select(["match_key", "mr"]), on="match_key")
R = R.with_columns(
    pl.when(pl.col("bev") == "bomb_exploded").then(pl.lit(T))
    .when(pl.col("bev") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(0)).alias("bside"),
    ((pl.col("round") == pl.col("mr") // 2) | ((pl.col("round") > pl.col("mr")) &
                                               (((pl.col("round") - pl.col("mr") - 3) % 6) == 0))).alias("sw"))
R = R.sort(["match_key", "round"]).with_columns(
    pl.col("sw").cast(pl.Int64).cum_sum().over("match_key").alias("cs"))
R = R.with_columns((pl.col("cs") - pl.col("sw").cast(pl.Int64)).alias("s_before"))
R = R.with_columns(
    pl.when(pl.col("round").is_in(DISP) & (pl.col("bside") != 0)).then(pl.col("bside"))
    .when(pl.col("round").is_in(DISP) & (~pl.col("sw")) & (pl.col("ss") != 0)).then(pl.col("ss"))
    .otherwise(pl.col("winner_team_code")).alias("w_fix"),
    (pl.col("round").is_in(DISP) & (pl.col("ss") != 0) & (~pl.col("sw")) & (pl.col("ss") != pl.col("winner_team_code")))
    .alias("dis"))


def xt(df, w):
    return (df.with_columns(((((pl.col(w) == T) & (pl.col("s_before") % 2 == 0)) |
                              ((pl.col(w) == CT) & (pl.col("s_before") % 2 == 1)))).cast(pl.Int64).alias("xw"))
            .group_by("match_key").agg(pl.col("xw").sum().alias("X")))


X = xt(R, "winner_team_code").rename({"X": "X_RE"}).join(xt(R, "w_fix").rename({"X": "X_fix"}), on="match_key")
X = X.join(M.select(["match_key", "hT", "hC", "max_label"]), on="match_key").join(
    R.group_by("match_key").agg(pl.col("dis").sum().alias("n_dis"),
                                pl.col("round").filter(pl.col("dis")).sort().alias("dis_rounds")), on="match_key")
X = X.with_columns((pl.col("X_fix") - pl.col("hT")).alias("fix_minus_h"),
                   (pl.col("X_fix") == pl.col("hT")).alias("ok"),
                   pl.when(pl.col("max_label") > 16).then(pl.lit("OT")).otherwise(pl.lit("reg16")).alias("finish"))
print(f"MR8 complete d12-25 matches (label 1 present): {X.height}")
print(X.group_by("finish").agg(pl.len().alias("n"), pl.col("ok").mean().alias("fix_ok_rate")).sort("finish"))
print("\nfix_minus_h distribution (ok rate by sign):")
print(X.group_by("fix_minus_h").len().sort("fix_minus_h"))
print("\nfailures: disputed rounds and finish (first 15)")
fails = X.filter(~pl.col("ok"))
print(fails.with_columns(pl.col("dis_rounds").cast(pl.List(pl.Utf8)).list.join(",").alias("dr")).group_by(
    ["dr", "finish"]).len().sort("len", descending=True).head(15))
print("\nround-9 detail of 6 failing matches (round, winner_RE, ss, bside):")
ex = fails["match_key"].head(6).to_list()
det = R.filter(pl.col("match_key").is_in(ex) & pl.col("round").is_between(7, 10)).select(
    ["match_key", "round", "winner_team_code", "ss", "bside", "s_before", "sw"]).with_columns(
    pl.col("match_key").str.slice(-40, 8).alias("mk")).drop("match_key").sort(["mk", "round"])
print(det)
