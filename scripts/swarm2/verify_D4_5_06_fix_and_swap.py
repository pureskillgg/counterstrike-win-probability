"""verify D4-5, step 6: (i) does the claimed fix reproduce the header final score? (ii) is the physical side swap
really after round 12 in the disputed matches? (iii) do round_end labels look shifted by one round?

2024 tome, complete matches only (round labels start at 1 and round_end rows == header total, see step 5).
Fix under test (claim, applied to rounds 9, 13, 25, 28 in days 12-25 only):
  winner := bomb side if a bomb_defused / bomb_exploded row sits in the round (bomb_state round attribute),
            else score side (fe(r) -> fe(r+1), unambiguous, no swap after r), else round_end winner unchanged.
Team X (started on T) total under a winner list: rounds where X's side (T iff s(r) even) won.
ASSUMPTION: swap convention mr = 24. ASSUMPTION: header t_starters_score_final is the engine's final score of X.
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, swap_expected_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
base = TOMES["2024"]
MR = 24
DISP = [9, 13, 25, 28]

rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code"]).collect().sort(
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
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("ss"),
    swap_expected_expr(pl.lit(MR)).alias("sw"))

# ---------- per-round table over all round_end rows ----------
R = re_.join(Dp.select(["match_key", "round", "ss", "sw"]), on=["match_key", "round"], how="left").join(
    bo, on=["match_key", "round"], how="left").with_columns(
    pl.when(pl.col("bev") == "bomb_exploded").then(pl.lit(T))
    .when(pl.col("bev") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(0)).alias("bside"),
    swap_expected_expr(pl.lit(MR)).cast(pl.Int64).alias("swf"))
R = R.sort(["match_key", "round"]).with_columns(
    (pl.col("swf").cum_sum().over("match_key") - pl.col("swf")).alias("s_before"))
R = R.with_columns(day_expr().alias("day"))

# complete matches: labels start at 1, round_end rows == header total
M0 = R.group_by("match_key").agg(pl.len().alias("n_RE"), pl.col("round").min().alias("min_label"))
M0 = M0.join(H, on="match_key", how="left").with_columns(
    (pl.col("hT") + pl.col("hC") == pl.col("n_RE")).alias("complete"),
    (pl.col("min_label") == 1).alias("label1"))
M0 = M0.join(R.select(["match_key", "day"]).unique(), on="match_key", how="left")
complete = M0.filter(pl.col("complete") & pl.col("label1") & pl.col("day").is_between(12, 25))["match_key"]
Rc = R.filter(pl.col("match_key").is_in(complete.to_list()))
print(f"complete matches in days 12-25 with labels from 1: {Rc['match_key'].n_unique()}")

# claim's fix winner
Rc = Rc.with_columns(
    pl.when(pl.col("round").is_in(DISP) & (pl.col("bside") != 0)).then(pl.col("bside"))
    .when(pl.col("round").is_in(DISP) & (~pl.col("sw")) & (pl.col("ss") != 0)).then(pl.col("ss"))
    .otherwise(pl.col("winner_team_code")).alias("w_fix"))


def x_total(df, wcol):
    return (df.with_columns(
        (((pl.col(wcol) == T) & (pl.col("s_before") % 2 == 0)) |
         ((pl.col(wcol) == CT) & (pl.col("s_before") % 2 == 1))).cast(pl.Int64).alias("xw"))
        .group_by("match_key").agg(pl.col("xw").sum().alias("X")))


XR = x_total(Rc, "winner_team_code").rename({"X": "X_RE"})
XF = x_total(Rc, "w_fix").rename({"X": "X_fix"})
MM = XR.join(XF, on="match_key").join(H, on="match_key").join(
    Rc.group_by("match_key").agg(pl.col("round").filter(pl.col("round").is_in(DISP) & (pl.col("ss") != 0)
                                                        & (pl.col("ss") != pl.col("winner_team_code"))
                                                        ).sort().alias("disp_rounds")), on="match_key", how="left")
MM = MM.with_columns(
    (pl.col("X_RE") == pl.col("hT")).alias("RE_ok"),
    (pl.col("X_fix") == pl.col("hT")).alias("fix_ok"),
    pl.col("disp_rounds").list.len().alias("n_disp"))
print("\n== (i) header agreement, complete matches days 12-25")
print(MM.group_by("n_disp").agg(pl.len().alias("n"), pl.col("RE_ok").mean().alias("frac_RE_eq_header"),
                                 pl.col("fix_ok").mean().alias("frac_fix_eq_header")).sort("n_disp"))
print("overall: RE ok", int(MM["RE_ok"].sum()), "fix ok", int(MM["fix_ok"].sum()), "of", MM.height)
bad = MM.filter(~pl.col("fix_ok"))
print("\nmatches where the fix still does not reproduce the header (sample of 12):")
print(bad.select(["match_key", "hT", "hC", "X_RE", "X_fix", "disp_rounds"]).head(12))

# disputed-round pattern among the fix failures
print("\nfix failures by disputed-round pattern:")
print(bad.group_by(bad["disp_rounds"].list.join(",")).len().sort("len", descending=True).head(12))

# ---------- (ii) physical swap position from score-field swaps ----------
print("\n== (ii) label r where the score fields swap values between fe(r) and fe(r+1) (physical side swap)")
fs = Dp.with_columns(
    ((pl.col("t1") == pl.col("ct_score")) & (pl.col("c1") == pl.col("t_score")) &
     (pl.col("t_score") != pl.col("ct_score"))).alias("field_swap"))
fs = fs.join(M0.select(["match_key", "complete", "label1", "day"]), on="match_key", how="left")
fs = fs.with_columns(pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).when(pl.col("day").is_between(12, 25))
                     .then(pl.lit("d12-25")).otherwise(pl.lit("other")).alias("grp"))
sw_tab = fs.filter(pl.col("field_swap")).group_by(["grp", "round"]).len().sort(["grp", "round"])
print(sw_tab.filter(pl.col("round") <= 30))
mm = fs.filter(pl.col("field_swap")).select(["match_key"]).unique()
print("matches with any field swap:", mm.height)
disp_matches = Rc.filter(pl.col("round").is_in(DISP) & (pl.col("ss") != 0)
                         & (pl.col("ss") != pl.col("winner_team_code"))).select("match_key").unique()
fs_disp = fs.filter(pl.col("field_swap") & pl.col("match_key").is_in(disp_matches["match_key"].implode()))
print("field-swap labels in complete disputed matches (d12-25):")
print(fs_disp.group_by("round").len().sort("round"))

# ---------- (iii) shift test: does round_end(r) match score side at r-1 or r+1? ----------
print("\n== (iii) shift test in disputed rounds (d12-25, valid unambiguous score side)")
S = R.filter(pl.col("day").is_between(12, 25) & (~pl.col("sw")) & (pl.col("ss") != 0)).select(
    ["match_key", "round", "ss", "winner_team_code"])
S_prev = S.with_columns((pl.col("round") + 1).alias("round")).rename({"ss": "ss_prev", "winner_team_code": "w_prev"})
S_next = S.with_columns((pl.col("round") - 1).alias("round")).rename({"ss": "ss_next", "winner_team_code": "w_next"})
SS = S.join(S_prev.select(["match_key", "round", "ss_prev"]), on=["match_key", "round"], how="left").join(
    S_next.select(["match_key", "round", "ss_next"]), on=["match_key", "round"], how="left")
SS = SS.filter(pl.col("round").is_in(DISP))
print(SS.group_by("round").agg(
    pl.len().alias("n"),
    (pl.col("ss") != pl.col("winner_team_code")).sum().alias("dis_same_label"),
    (pl.col("ss_prev") == pl.col("winner_team_code")).sum().alias("RE_eq_score_prev_round"),
    (pl.col("ss_next") == pl.col("winner_team_code")).sum().alias("RE_eq_score_next_round"),
).sort("round"))
