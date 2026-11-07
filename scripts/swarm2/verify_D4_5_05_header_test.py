"""verify D4-5, step 5: match-level header test (corrected). 2024 tome.

For each match:
  RE   : round_end winners for ALL rounds present (no pairing; fixes the last-round drop in an earlier draft).
  X    : the team that started on T. Side of X in round label r is T iff s(r) is even, where
         s(r) = number of swap labels k < r (swap_expected(k, 24): k == 12 or k > 24 and (k-27) % 6 == 0).
  X_RE : rounds X won by round_end labels (X on T and winner T, or X on CT and winner CT).
  X_SC : X's final cumulative score from round_state's LAST row: t_score if X on T in the last round, else
         ct_score (the side-based score fields follow the side, see verify_D4_5_04 debug output).
  hT   : header t_starters_score_final (the starting-T team's final score).
  Disputed match: >= 1 valid round (no swap after r, unambiguous) where score side != round_end winner.
ASSUMPTION: swap convention for 2024 (mr = 24). Overtime swaps follow the same convention as bakeoff.py.
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, swap_expected_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
base = TOMES["2024"]
MR = 24

rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code"]).collect().sort(
    ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
H = header(base, ["key", "t_starters_score_final", "ct_starters_score_final"]).rename(
    {"key": "match_key", "t_starters_score_final": "hT", "ct_starters_score_final": "hC"})

fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"]))
fe_next = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t1", "ct_score": "c1"})
Dp = fe.join(fe_next, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t1") - pl.col("t_score")).alias("dT"), (pl.col("c1") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("ss"),
    swap_expected_expr(pl.lit(MR)).alias("sw"))

# per-round table over all round_end labels
R = re_.with_columns(swap_expected_expr(pl.lit(MR)).cast(pl.Int64).alias("swf"))
R = R.sort(["match_key", "round"]).with_columns(
    (pl.col("swf").cum_sum().over("match_key") - pl.col("swf")).alias("s_before"))
R = R.with_columns(
    (((pl.col("winner_team_code") == T) & (pl.col("s_before") % 2 == 0)) |
     ((pl.col("winner_team_code") == CT) & (pl.col("s_before") % 2 == 1))).cast(pl.Int64).alias("x_won"))
M = R.group_by("match_key").agg(
    pl.len().alias("n_RE"), pl.col("x_won").sum().alias("X_RE"), pl.col("round").min().alias("min_label"),
    pl.col("round").max().alias("max_label"), pl.col("round").n_unique().alias("n_distinct"))
M = M.with_columns((pl.col("n_RE") - pl.col("X_RE")).alias("Y_RE"))

last_s = R.sort(["match_key", "round"]).group_by("match_key").agg(pl.col("s_before").last().alias("s_last"),
                                                                    pl.col("round").last().alias("last_label"))
st = rs.sort(["match_key", "tick"]).group_by("match_key").agg(
    pl.col("t_score").last().alias("t_last"), pl.col("ct_score").last().alias("c_last"))
M = M.join(st, on="match_key", how="left").join(last_s, on="match_key", how="left")
M = M.with_columns(
    pl.when(pl.col("s_last") % 2 == 0).then(pl.col("t_last")).otherwise(pl.col("c_last")).alias("X_SC"),
    (pl.col("t_last") + pl.col("c_last")).alias("state_total"))
M = M.join(H, on="match_key", how="left")
disp_rounds = (Dp.filter((~pl.col("sw")) & (pl.col("ss") != 0))
               .join(re_.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner")
               .filter(pl.col("ss") != pl.col("winner_team_code")))
disp_m = disp_rounds.group_by("match_key").agg(pl.len().alias("n_disp"),
                                              pl.col("round").sort().alias("disp_rounds"))
M = M.join(disp_m, on="match_key", how="left").with_columns(pl.col("n_disp").fill_null(0))
M = M.with_columns(day_expr().alias("day"),
                   (pl.col("n_disp") > 0).alias("disputed"),
                   (pl.col("hT") + pl.col("hC") == pl.col("n_RE")).alias("hdr_eq_nRE"),
                   (pl.col("hT") + pl.col("hC") == pl.col("state_total")).alias("hdr_eq_state"),
                   (pl.col("X_RE") == pl.col("hT")).alias("hdrT_eq_RE"),
                   (pl.col("X_SC") == pl.col("hT")).alias("hdrT_eq_SC"),
                   (pl.col("min_label") == 1).alias("starts_at_1"))
M = M.with_columns(pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).when(pl.col("day").is_between(12, 25))
                   .then(pl.lit("d12-25")).otherwise(pl.lit("other")).alias("grp"))

print(f"matches: {M.height}; with header: {int(M['hT'].is_not_null().sum())}")
print("\n== (1) header total vs round_end row count and round_state total, by group and disputed")
print(M.group_by(["grp", "disputed"]).agg(
    pl.len().alias("matches"),
    pl.col("hdr_eq_nRE").mean().alias("frac_hdr_eq_nRE"),
    pl.col("hdr_eq_state").mean().alias("frac_hdr_eq_state_total"),
    pl.col("starts_at_1").mean().alias("frac_label1_present"),
).sort(["grp", "disputed"]))

C = M.filter(pl.col("hdr_eq_nRE") & pl.col("hdr_eq_state") & pl.col("starts_at_1"))
print(f"\n== (2) complete matches: header == round_end rows == round_state total, labels start at 1: n={C.height}")
print(C.group_by(["grp", "disputed"]).agg(
    pl.len().alias("n"),
    pl.col("hdrT_eq_RE").mean().alias("frac_hdrT_eq_roundend_X"),
    pl.col("hdrT_eq_SC").mean().alias("frac_hdrT_eq_state_X"),
).sort(["grp", "disputed"]))

CD = C.filter(pl.col("disputed") & (pl.col("grp") == "d12-25"))
print(f"\n== (3) complete disputed matches, days 12-25: n={CD.height}")
print("   header X matches round_end-derived X:", int(CD["hdrT_eq_RE"].sum()),
      "| header X matches state-derived X:", int(CD["hdrT_eq_SC"].sum()))
print("   neither:", int((~CD["hdrT_eq_RE"] & ~CD["hdrT_eq_SC"]).sum()),
      " both:", int((CD["hdrT_eq_RE"] & CD["hdrT_eq_SC"]).sum()))
print("   disputed round labels among these matches:")
print(CD.explode("disp_rounds").group_by("disp_rounds").len().sort("len", descending=True).head(12))

print("\n== (4) matches where round labels do not start at 1 (day, disputed, header total vs n_RE)")
ns = M.filter(~pl.col("starts_at_1"))
print(ns.group_by(["day", "disputed"]).agg(pl.len().alias("n"), pl.col("min_label").min().alias("min_label_min"),
                                           pl.col("hdr_eq_nRE").mean().alias("frac_hdr_eq_nRE")).sort(["day", "disputed"]))

print("\n== (5) first 12 disputed days 12-25 complete matches with the three totals (sanity)")
print(CD.select(["match_key", "n_RE", "hT", "hC", "X_RE", "X_SC", "t_last", "c_last", "disp_rounds"]).head(12))

print("\n== (6) round-13-disputed matches: how many also have round_end and state-final consistent with each other?")
D13 = M.filter(pl.col("disp_rounds").list.contains(13) & (pl.col("grp") == "d12-25"))
print("n:", D13.height, " complete:", int(D13["hdr_eq_nRE"].sum()), " header-T vs RE:",
      int(D13["hdrT_eq_RE"].sum()), " header-T vs SC:", int(D13["hdrT_eq_SC"].sum()))
