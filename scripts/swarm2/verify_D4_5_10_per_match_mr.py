"""verify D4-5, step 10: rerun the claim's counts and the header test with a PER-MATCH swap convention.

Per-match max_rounds (mr) is inferred from the round_state halftime announcement (round_announce_last_round_half):
  label 11 -> halftime after round 12 -> mr = 24   (MR12 style)
  label 7  -> halftime after round 8  -> mr = 16   (MR8 style)
  no event -> mr = 24 (ASSUMPTION, flagged as 'noevent'; see the header test for these)
Swap rule (bakeoff.swap_expected): r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0).
The claim used mr = 24 for every 2024 match (A12). This step reports both versions.
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(70)
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

M = re_.group_by("match_key").agg(pl.len().alias("n_RE"), pl.col("round").min().alias("min_label")).join(
    H, on="match_key").join(half, on="match_key", how="left")
M = M.with_columns(
    pl.when(pl.col("half_label") == 7).then(pl.lit(16)).when(pl.col("half_label") == 11).then(pl.lit(24))
    .otherwise(pl.lit(24)).alias("mr"),
    pl.when(pl.col("half_label") == 7).then(pl.lit("MR8_lbl7"))
    .when(pl.col("half_label") == 11).then(pl.lit("MR12_lbl11")).otherwise(pl.lit("noevent")).alias("kind"),
    (pl.col("hT") + pl.col("hC") == pl.col("n_RE")).alias("complete"),
    (pl.col("min_label") == 1).alias("label1"))
M = M.join(re_.group_by("match_key").agg(pl.col("round").max().alias("max_label")), on="match_key")
M = M.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32).alias("day"))
M = M.with_columns(pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).when(pl.col("day").is_between(12, 25))
                   .then(pl.lit("d12-25")).otherwise(pl.lit("other")).alias("grp"))

# per-round table with per-match swap flags for both conventions
R = re_.join(Dp.select(["match_key", "round", "ss"]), on=["match_key", "round"], how="left").join(
    bo, on=["match_key", "round"], how="left").join(M.select(["match_key", "mr", "kind", "grp"]), on="match_key")
R = R.with_columns(
    pl.when(pl.col("bev") == "bomb_exploded").then(pl.lit(T))
    .when(pl.col("bev") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(0)).alias("bside"),
    (pl.col("round") == pl.col("mr") // 2).alias("sw_pm"),
    ((pl.col("round") > pl.col("mr")) & (((pl.col("round") - pl.col("mr") - 3) % 6) == 0)).alias("sw_pm2"),
    (pl.col("round") == 12).alias("sw_24"))
R = R.with_columns((pl.col("sw_pm") | pl.col("sw_pm2")).alias("sw_pm_all"))
R = R.with_columns((pl.col("sw_24") | ((pl.col("round") > 24) & (((pl.col("round") - 27) % 6) == 0))).alias("sw_24_all"))

# ----- (1) claim counts: valid = no swap after r, unambiguous score side -----
V_claim = R.filter((~pl.col("sw_24_all")) & (pl.col("ss") != 0) & pl.col("grp").is_in(["d12-25", "d01-07"]))
V_pm = R.filter((~pl.col("sw_pm_all")) & (pl.col("ss") != 0) & pl.col("grp").is_in(["d12-25", "d01-07"]))
for name, V in [("claim (mr=24 for all)", V_claim), ("per-match mr", V_pm)]:
    V = V.with_columns((pl.col("ss") != pl.col("winner_team_code")).alias("dis"))
    print(f"\n== (1) disagreements by round, {name}, days 12-25 (disputed rounds + total) and days 01-07")
    g = V.filter(pl.col("grp") == "d12-25").group_by("round").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"))
    print(g.filter(pl.col("dis") > 0).sort("round"))
    print("d12-25 total dis:", int(V.filter(pl.col("grp") == "d12-25")["dis"].sum()), "of",
          V.filter(pl.col("grp") == "d12-25").height, "| d01-07 dis:",
          int(V.filter(pl.col("grp") == "d01-07")["dis"].sum()), "of", V.filter(pl.col("grp") == "d01-07").height)
    print("d12-25 disagreements by match kind (mr event):")
    print(V.filter(pl.col("grp") == "d12-25").group_by("kind").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"),
                                                                  pl.col("dis").mean().alias("rate")).sort("kind"))
    print("d12-25 disagreements at round 13 by kind:")
    print(V.filter((pl.col("grp") == "d12-25") & (pl.col("round") == 13)).group_by("kind").agg(
        pl.len().alias("n"), pl.col("dis").sum().alias("dis"), pl.col("dis").mean().alias("rate")).sort("kind"))

# ----- (2) header test with per-match convention, complete matches -----
Rc = R.filter(pl.col("match_key").is_in(M.filter(pl.col("complete") & pl.col("label1") & (pl.col("grp") == "d12-25"))
                                       ["match_key"].to_list())).sort(["match_key", "round"])
Rc = Rc.with_columns(
    (pl.col("sw_pm_all").cast(pl.Int64)).alias("swp"),
    (pl.col("sw_24_all").cast(pl.Int64)).alias("sw24"))
Rc = Rc.with_columns(
    (pl.col("swp").cum_sum().over("match_key") - pl.col("swp")).alias("s_pm"),
    (pl.col("sw24").cum_sum().over("match_key") - pl.col("sw24")).alias("s_24"))
Rc = Rc.with_columns(
    pl.when(pl.col("round").is_in(DISP) & (pl.col("bside") != 0)).then(pl.col("bside"))
    .when(pl.col("round").is_in(DISP) & (~pl.col("sw_pm_all")) & (pl.col("ss") != 0)).then(pl.col("ss"))
    .otherwise(pl.col("winner_team_code")).alias("w_fix_pm"))


def xt(df, w, s):
    return (df.with_columns(((((pl.col(w) == T) & (pl.col(s) % 2 == 0)) | ((pl.col(w) == CT) & (pl.col(s) % 2 == 1)))
                             ).cast(pl.Int64).alias("xw"))
            .group_by("match_key").agg(pl.col("xw").sum().alias("X")))


X = (xt(Rc, "winner_team_code", "s_24").rename({"X": "RE_24"})
     .join(xt(Rc, "winner_team_code", "s_pm").rename({"X": "RE_pm"}), on="match_key")
     .join(xt(Rc, "w_fix_pm", "s_pm").rename({"X": "fix_pm"}), on="match_key"))
X = X.join(M.select(["match_key", "hT", "hC", "kind", "mr"]), on="match_key")
disp = (Dp.join(re_.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner")
        .join(M.select(["match_key", "mr"]), on="match_key")
        .filter((pl.col("ss") != 0) & pl.col("round").is_in(DISP) & (pl.col("ss") != pl.col("winner_team_code")))
        .group_by("match_key").agg(pl.col("round").sort().alias("disp")))
X = X.join(disp, on="match_key", how="left").with_columns(
    pl.col("disp").cast(pl.List(pl.Utf8)).list.join(",").fill_null("none").alias("pat"))
for c in ["RE_24", "RE_pm", "fix_pm"]:
    X = X.with_columns((pl.col(c) == pl.col("hT")).alias(c + "_ok"))
print("\n== (2) header agreement, complete d12-25 matches (labels from 1), per-match convention")
print(X.group_by("kind").agg(pl.len().alias("n"), pl.col("RE_24_ok").mean().alias("RE_mr24"),
                             pl.col("RE_pm_ok").mean().alias("RE_per_match_mr"),
                             pl.col("fix_pm_ok").mean().alias("fix_per_match_mr")).sort("kind"))
print("\n== (2b) by disputed pattern x kind")
print(X.group_by(["kind", "pat"]).agg(pl.len().alias("n"), pl.col("RE_pm_ok").mean().alias("RE_pm"),
                                      pl.col("fix_pm_ok").mean().alias("fix_pm")).sort(["kind", "n"], descending=[False, True]))

# ----- (3) header total and state total by kind (all 2024 matches) -----
print("\n== (3) header total vs round_end count, all 2024 matches, by kind and day group")
print(M.group_by(["grp", "kind"]).agg(pl.len().alias("n"), (pl.col("hT") + pl.col("hC") == pl.col("n_RE")).mean().alias(
    "frac_hdr_eq_nRE"), pl.col("label1").mean().alias("frac_label1")).sort(["grp", "kind"]))
