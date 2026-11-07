"""verify D4-5, step 7: empirical physical side swaps and their effect on the header test.

(a) Physical swap label per match: the label r where the score fields swap values between fe(r) and fe(r+1)
    (t(r+1) == c(r) and c(r+1) == t(r), with t(r) != c(r)). Compare with the convention swap after r = 12.
(b) Header test with X's side mapping from the EMPIRICAL swap labels instead of the convention.
(c) Pattern breakdown of the claim's fix and of round_end against the header, complete matches days 12-25.
ASSUMPTION: a field swap marks a physical side swap after round r. Matches with no detected swap are reported.
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
    # a physical swap after round r: the new T-side field is the old CT-side field, plus one point if the
    # round-r winner was on the CT side (or the reverse). Exactly one point is added in total.
    ((((pl.col("t1") - pl.col("ct_score")) == 0) & ((pl.col("c1") - pl.col("t_score")) == 1)) |
     (((pl.col("t1") - pl.col("ct_score")) == 1) & ((pl.col("c1") - pl.col("t_score")) == 0))).alias("fswap"))

# (a) physical swap labels per match
SW = Dp.filter(pl.col("fswap")).group_by("match_key").agg(pl.col("round").sort().alias("swap_labels"))
allm = fe.select("match_key").unique()
SW = allm.join(SW, on="match_key", how="left").with_columns(
    pl.col("swap_labels").fill_null(pl.lit([]).cast(pl.List(pl.Int64))))
SW = SW.with_columns(
    pl.col("swap_labels").list.len().alias("n_swaps"),
    pl.col("swap_labels").list.first().alias("first_swap"))
SW = SW.with_columns(day_expr().alias("day"))
print("(a) physical swap labels (score-field swap between fe(r) and fe(r+1)), all 2024 matches")
print("distribution of number of detected swaps:")
print(SW.group_by("n_swaps").len().sort("n_swaps"))
print("\nfirst detected swap label (matches with >=1 swap):")
print(SW.filter(pl.col("n_swaps") >= 1).group_by("first_swap").len().sort("len", descending=True).head(15))
print("\nmatches with no detected swap, by day group:")
SW = SW.with_columns(pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).when(pl.col("day").is_between(12, 25))
                     .then(pl.lit("d12-25")).otherwise(pl.lit("other")).alias("grp"))
print(SW.group_by(["grp", "n_swaps"]).len().sort(["grp", "n_swaps"]))
print("\nswap-label sets (top 12 patterns):")
print(SW.with_columns(pl.col("swap_labels").map_elements(lambda l: ",".join(map(str, l)), return_dtype=pl.Utf8)
                      .alias("pat")).group_by("pat").len().sort("len", descending=True).head(12))

# per-round table over all round_end labels (team mapping from the empirical swap labels)
R = re_.join(Dp.select(["match_key", "round", "ss"]), on=["match_key", "round"], how="left").join(
    bo, on=["match_key", "round"], how="left").join(SW.select(["match_key", "swap_labels"]), on="match_key",
                                                  how="left")
R = R.with_columns(
    pl.when(pl.col("bev") == "bomb_exploded").then(pl.lit(T))
    .when(pl.col("bev") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(0)).alias("bside"),
    swap_expected_expr(pl.lit(MR)).alias("sw_conv"))
# s_before from empirical swap labels: count of physical swaps strictly before round r
R = R.with_columns(pl.struct(["round", "swap_labels"]).map_elements(
    lambda s: sum(1 for k in (s["swap_labels"] or []) if k < s["round"]), return_dtype=pl.Int64).alias("s_emp"))
R = R.with_columns(pl.col("sw_conv").cast(pl.Int64).alias("swf"))
R = R.sort(["match_key", "round"]).with_columns(
    (pl.col("swf").cum_sum().over("match_key") - pl.col("swf")).alias("s_conv"))
R = R.with_columns(day_expr().alias("day"))

M0 = R.group_by("match_key").agg(pl.len().alias("n_RE"), pl.col("round").min().alias("min_label"),
                                 pl.col("round").max().alias("max_label"))
M0 = M0.join(H, on="match_key", how="left").with_columns(
    (pl.col("hT") + pl.col("hC") == pl.col("n_RE")).alias("complete"),
    (pl.col("min_label") == 1).alias("label1"))
M0 = M0.join(R.select(["match_key", "day"]).unique(), on="match_key", how="left")
M0 = M0.join(SW.select(["match_key", "n_swaps", "first_swap", "swap_labels"]), on="match_key", how="left")
comp = M0.filter(pl.col("complete") & pl.col("label1") & pl.col("day").is_between(12, 25))
print(f"\nC: complete days 12-25 with labels from 1: {comp.height}")
print("   first physical swap label distribution in these:")
print(comp.group_by("first_swap").len().sort("len", descending=True).head(10))

Rc = R.filter(pl.col("match_key").is_in(comp["match_key"].to_list()))
Rc = Rc.with_columns(
    pl.when(pl.col("round").is_in(DISP) & (pl.col("bside") != 0)).then(pl.col("bside"))
    .when(pl.col("round").is_in(DISP) & (~pl.col("sw_conv")) & (pl.col("ss") != 0)).then(pl.col("ss"))
    .otherwise(pl.col("winner_team_code")).alias("w_fix"))


def xtot(df, wcol, smap):
    return (df.with_columns(
        (((pl.col(wcol) == T) & (pl.col(smap) % 2 == 0)) | ((pl.col(wcol) == CT) & (pl.col(smap) % 2 == 1)))
        .cast(pl.Int64).alias("xw"))
        .group_by("match_key").agg(pl.col("xw").sum().alias("X")))


out = None
for name, w, smap in [("RE_conv", "winner_team_code", "s_conv"), ("RE_emp", "winner_team_code", "s_emp"),
                      ("fix_conv", "w_fix", "s_conv"), ("fix_emp", "w_fix", "s_emp")]:
    t = xtot(Rc, w, smap).rename({"X": name})
    out = t if out is None else out.join(t, on="match_key")
Z = out.join(comp.select(["match_key", "hT", "hC", "n_swaps", "first_swap"]), on="match_key")
for c in ["RE_conv", "RE_emp", "fix_conv", "fix_emp"]:
    Z = Z.with_columns((pl.col(c) == pl.col("hT")).alias(c + "_ok"))
print("\n(b) header agreement with X's mapping, complete days 12-25:")
print(Z.select([c + "_ok" for c in ["RE_conv", "RE_emp", "fix_conv", "fix_emp"]]).mean())
print("\n(b2) same, split by n_swaps detected:")
print(Z.group_by("n_swaps").agg(pl.len().alias("n"),
                                pl.col("RE_conv_ok").mean().alias("RE_conv"),
                                pl.col("RE_emp_ok").mean().alias("RE_emp"),
                                pl.col("fix_conv_ok").mean().alias("fix_conv"),
                                pl.col("fix_emp_ok").mean().alias("fix_emp")).sort("n_swaps"))

# (c) pattern breakdown of disputed-round set (rounds where score side != round_end winner)
disp_rounds = (Dp.filter((~pl.col("ss").eq(0)))
               .join(re_.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner")
               .filter(pl.col("round").is_in(DISP) & (pl.col("ss") != pl.col("winner_team_code")))
               .group_by("match_key").agg(pl.col("round").sort().alias("disp")))
Z = Z.join(disp_rounds, on="match_key", how="left").with_columns(
    pl.col("disp").cast(pl.List(pl.Utf8)).list.join(",").fill_null("none").alias("pat"))
print("\n(c) by disputed pattern (complete days 12-25), header agreement:")
print(Z.group_by("pat").agg(pl.len().alias("n"), pl.col("RE_conv_ok").mean().alias("RE_conv"),
                            pl.col("RE_emp_ok").mean().alias("RE_emp"),
                            pl.col("fix_conv_ok").mean().alias("fix_conv"),
                            pl.col("fix_emp_ok").mean().alias("fix_emp")).sort("n", descending=True))
