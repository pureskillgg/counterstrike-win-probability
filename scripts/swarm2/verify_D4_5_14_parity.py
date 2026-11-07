"""verify D4-5, step 14: is the MR8 residual a team-parity inversion (X and Y totals swapped)?

Complete 2024 matches, days 12-25, labels from 1. For each match and each parity hypothesis
  normal   : X is on T in round r iff s(r) even  (s = swaps before r; per-match convention, mr from halftime event)
  inverted : X is on T in round r iff s(r) odd
compute the header agreement (X == hT) for:
  X_SC  : state final score (t_last / c_last) under the parity
  X_RE  : round_end winners under the parity
  X_fix : claim fix applied to rounds 9, 13, 25, 28 (bomb side, else score side, else round_end), under the parity
Also: for normal-parity failures, does X_SC equal hC (i.e. the teams' totals are swapped)?
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
M = M.filter(pl.col("complete") & pl.col("label1") & pl.col("day").is_between(12, 25))
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
    .otherwise(pl.col("winner_team_code")).alias("w_fix"))
last = R.group_by("match_key").agg(pl.col("s_before").last().alias("s_last"), pl.col("round").max().alias("L"))
st = rs.filter(pl.col("match_key").is_in(keep)).sort(["match_key", "tick"]).group_by("match_key").agg(
    pl.col("t_score").last().alias("t_last"), pl.col("ct_score").last().alias("c_last"))


def xt(df, w, inv):
    par = 1 if inv else 0
    return (df.with_columns(((((pl.col(w) == T) & (((pl.col("s_before") + par) % 2) == 0)) |
                               ((pl.col(w) == CT) & (((pl.col("s_before") + par) % 2) == 1)))).cast(pl.Int64).alias("xw"))
            .group_by("match_key").agg(pl.col("xw").sum().alias("X")))


out = M.select(["match_key", "hT", "hC", "mr", "half_label"])
for w, tag in [("winner_team_code", "RE"), ("w_fix", "fix")]:
    for inv, ptag in [(False, "norm"), (True, "inv")]:
        out = out.join(xt(R, w, inv).rename({"X": f"{tag}_{ptag}"}), on="match_key", how="left")
out = out.join(last, on="match_key").join(st, on="match_key")
ls = out["s_last"]
out = out.with_columns(
    pl.when(((pl.col("s_last") % 2) == 0)).then(pl.col("t_last")).otherwise(pl.col("c_last")).alias("SC_norm"),
    pl.when(((pl.col("s_last") % 2) == 1)).then(pl.col("t_last")).otherwise(pl.col("c_last")).alias("SC_inv"))
for c in ["RE_norm", "RE_inv", "fix_norm", "fix_inv", "SC_norm", "SC_inv"]:
    out = out.with_columns((pl.col(c) == pl.col("hT")).alias(c + "_ok"))
out = out.with_columns(pl.when(pl.col("mr") == 16).then(pl.lit("MR8")).otherwise(pl.lit("MR12")).alias("kind"))
print("share of complete d12-25 matches where each hypothesis reproduces the header X total")
print(out.group_by("kind").agg(pl.len().alias("n"),
                               *[pl.col(c + "_ok").mean().alias(c) for c in
                                 ["RE_norm", "RE_inv", "fix_norm", "fix_inv", "SC_norm", "SC_inv"]]).sort("kind"))
f8 = out.filter((pl.col("kind") == "MR8") & (~pl.col("SC_norm_ok")))
print(f"\nMR8 normal-parity state failures: {f8.height}; of which X_SC equals hC (teams swapped): "
      f"{int((f8['SC_norm'] == f8['hC']).sum())}")
print("MR8 with X_SC inverted matching header:", int(out.filter(pl.col("kind") == "MR8")["SC_inv_ok"].sum()), "of",
      int((out["kind"] == "MR8").sum()))
print("\nMR8 by parity: best hypothesis counts (fix_norm / fix_inv / RE_norm / RE_inv)")
m8 = out.filter(pl.col("kind") == "MR8")
print(m8.select([pl.col("fix_norm_ok").sum().alias("fix_norm"), pl.col("fix_inv_ok").sum().alias("fix_inv"),
                 pl.col("RE_norm_ok").sum().alias("RE_norm"), pl.col("RE_inv_ok").sum().alias("RE_inv"),
                 pl.col("SC_norm_ok").sum().alias("SC_norm"), pl.col("SC_inv_ok").sum().alias("SC_inv"),
                 pl.len().alias("n")]))
print("\nMR12 counts (for contrast):")
m12 = out.filter(pl.col("kind") == "MR12")
print(m12.select([pl.col("fix_norm_ok").sum().alias("fix_norm"), pl.col("fix_inv_ok").sum().alias("fix_inv"),
                  pl.col("RE_norm_ok").sum().alias("RE_norm"), pl.col("RE_inv_ok").sum().alias("RE_inv"),
                  pl.col("SC_norm_ok").sum().alias("SC_norm"), pl.col("SC_inv_ok").sum().alias("SC_inv"),
                  pl.len().alias("n")]))
