"""verify D4-5, step 8: residual header disagreements. 2024, complete matches (labels from 1, round_end rows ==
header total), days 12-25. Convention swap mapping only (empirical field-swap detection was too noisy, see step 7).

(a) by disputed pattern, split by regulation (max label <= 24) vs overtime (max label > 24):
    header agreement for round_end-based X and for fix-based X (claim's rule applied to rounds 9, 13, 25, 28).
(b) residual size for pattern '9' matches: X_RE - hT and X_fix - hT.
(c) round 9 detail for a few pattern-'9' matches: round, round_end winner, reason, score side, bomb side.
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, swap_expected_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)
base = TOMES["2024"]
MR = 24
DISP = [9, 13, 25, 28]

rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect().sort(
    ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
H = header(base, ["key", "t_starters_score_final", "ct_starters_score_final"]).rename(
    {"key": "match_key", "t_starters_score_final": "hT", "ct_starters_score_final": "hC"})
bs = scan(base, "bomb_state", ["match_key", "round", "tick", "event_type"]).collect()
bo = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
      .sort("tick").group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("bev")))
fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"]))
fe_next = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t1", "ct_score": "c1"})
Dp = fe.join(fe_next, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t1") - pl.col("t_score")).alias("dT"), (pl.col("c1") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("ss"))

R = re_.join(Dp.select(["match_key", "round", "ss"]), on=["match_key", "round"], how="left").join(
    bo, on=["match_key", "round"], how="left")
R = R.with_columns(
    pl.when(pl.col("bev") == "bomb_exploded").then(pl.lit(T))
    .when(pl.col("bev") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(0)).alias("bside"),
    swap_expected_expr(pl.lit(MR)).cast(pl.Int64).alias("swf"))
R = R.sort(["match_key", "round"]).with_columns(
    (pl.col("swf").cum_sum().over("match_key") - pl.col("swf")).alias("s_before"),
    day_expr().alias("day"))
M0 = R.group_by("match_key").agg(pl.len().alias("n_RE"), pl.col("round").min().alias("min_label"),
                                 pl.col("round").max().alias("max_label"))
M0 = M0.join(H, on="match_key").with_columns((pl.col("hT") + pl.col("hC") == pl.col("n_RE")).alias("complete"),
                                             (pl.col("min_label") == 1).alias("label1"))
M0 = M0.join(R.select(["match_key", "day"]).unique(), on="match_key", how="left")
comp = M0.filter(pl.col("complete") & pl.col("label1") & pl.col("day").is_between(12, 25))
Rc = R.filter(pl.col("match_key").is_in(comp["match_key"].to_list())).with_columns(
    pl.when(pl.col("round").is_in(DISP) & (pl.col("bside") != 0)).then(pl.col("bside"))
    .when(pl.col("round").is_in(DISP) & (~pl.col("swf").cast(pl.Boolean)) & (pl.col("ss") != 0)).then(pl.col("ss"))
    .otherwise(pl.col("winner_team_code")).alias("w_fix"))


def xtot(df, wcol):
    return (df.with_columns(
        (((pl.col(wcol) == T) & (pl.col("s_before") % 2 == 0)) | ((pl.col(wcol) == CT) & (pl.col("s_before") % 2 == 1)))
        .cast(pl.Int64).alias("xw")).group_by("match_key").agg(pl.col("xw").sum().alias("X")))


X = xtot(Rc, "winner_team_code").rename({"X": "X_RE"}).join(xtot(Rc, "w_fix").rename({"X": "X_fix"}), on="match_key")
X = X.join(comp.select(["match_key", "hT", "hC", "max_label"]), on="match_key")
disp = (Dp.join(re_.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner")
        .filter((pl.col("ss") != 0) & pl.col("round").is_in(DISP) & (pl.col("ss") != pl.col("winner_team_code")))
        .group_by("match_key").agg(pl.col("round").sort().alias("disp")))
X = X.join(disp, on="match_key", how="left").with_columns(
    pl.col("disp").cast(pl.List(pl.Utf8)).list.join(",").fill_null("none").alias("pat"),
    pl.when(pl.col("max_label") > 24).then(pl.lit("OT")).otherwise(pl.lit("reg")).alias("kind"),
    (pl.col("X_RE") == pl.col("hT")).alias("RE_ok"), (pl.col("X_fix") == pl.col("hT")).alias("fix_ok"),
    (pl.col("X_RE") - pl.col("hT")).alias("RE_minus_h"), (pl.col("X_fix") - pl.col("hT")).alias("fix_minus_h"))
print("(a) header agreement by disputed pattern and regulation/OT (complete days 12-25)")
print(X.group_by(["pat", "kind"]).agg(pl.len().alias("n"), pl.col("RE_ok").mean().alias("RE_ok"),
                                      pl.col("fix_ok").mean().alias("fix_ok")).sort(["pat", "kind"]))

print("\n(b) residual sizes for pattern '9' (X - header)")
p9 = X.filter(pl.col("pat") == "9")
print("RE_minus_h:", p9.group_by("RE_minus_h").len().sort("RE_minus_h").rows())
print("fix_minus_h:", p9.group_by("fix_minus_h").len().sort("fix_minus_h").rows())
print("pattern 'none' RE_minus_h:", X.filter(pl.col("pat") == "none").group_by("RE_minus_h").len().rows())

print("\n(c) round 9 detail for 5 pattern-'9' matches")
ex = p9.head(5)["match_key"].to_list()
det = (R.filter(pl.col("match_key").is_in(ex) & pl.col("round").is_between(7, 11))
       .select(["match_key", "round", "winner_team_code", "win_reason_code", "ss", "bside", "bev", "s_before"])
       .with_columns(pl.col("match_key").str.slice(-40, 8).alias("mk")).drop("match_key"))
print(det.sort(["mk", "round"]))
print(H.filter(pl.col("match_key").is_in(ex)).select(["match_key", "hT", "hC"]).with_columns(
    pl.col("match_key").str.slice(-40, 8).alias("mk")).drop("match_key"))
