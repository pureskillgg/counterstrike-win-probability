"""verify D4-5, step 16: are disputes exactly at the first round after a physical side swap? Plus death-based and
header-attribute checks. 2024, days 12-25, all matches (not only complete ones).

Per-match swap set S (physical swaps AFTER round k):
  MR12 (halftime label 11): k = 12; if the match goes past 24 rounds, also k = 24 (regulation end, convention B
  verified in verify_D4_5_15) and k = 27, 33, ... (bakeoff rule).
  MR8 (halftime label 7): k = 8, and k = 16 if past 16 (bakeoff rule with mr 16: 19, 25, ...). Overtime after 16 for
  MR8 uses the bakeoff rule (k = 19, 25, ...). Matches with no halftime event (noevent): S from mr 24 convention (flagged).
Table A: disputed rate (score side != round_end) at rounds r with r-1 in S (first round after a swap) vs others.
Table B: death-based winner (opposite of latest player_death player_team_code) vs score side vs round_end for
         elimination-ended rounds (reason 8 or 9) with a dispute at 9, 13, 25 or 28, and undisputed ones.
Table C: header attributes (rushb_version, ppp_version, build_num, providence) for disputed vs undisputed matches.
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
base = TOMES["2024"]
DISP = [9, 13, 25, 28]
rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect().sort(
    ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
Hf = header(base, ["key", "rushb_version", "ppp_version", "build_num", "providence", "match_date"]).rename(
    {"key": "match_key"})
half = rs.filter(pl.col("event_type") == "round_announce_last_round_half").group_by("match_key").agg(
    pl.col("round").min().alias("half_label"))
fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"]))
fe_next = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t1", "ct_score": "c1"})
Dp = fe.join(fe_next, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t1") - pl.col("t_score")).alias("dT"), (pl.col("c1") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("ss"))
M = re_.group_by("match_key").agg(pl.col("round").max().alias("max_label")).join(
    half, on="match_key", how="left").join(Hf, on="match_key", how="left")
M = M.with_columns(
    pl.when(pl.col("half_label") == 7).then(pl.lit("MR8")).when(pl.col("half_label") == 11).then(pl.lit("MR12"))
    .otherwise(pl.lit("noevent")).alias("kind"),
    day_expr().alias("day"))
M = M.with_columns(pl.when(pl.col("kind") == "MR8").then(pl.lit(16)).otherwise(pl.lit(24)).alias("mr"))
M = M.filter(pl.col("day").is_between(12, 25))
keep = M["match_key"].to_list()
mr_of = dict(M.select(["match_key", "mr"]).rows())
kind_of = dict(M.select(["match_key", "kind"]).rows())
maxl = dict(M.select(["match_key", "max_label"]).rows())


def in_S(k, mr, maxlabel):
    # physical swap after round k (per-match set)
    if mr == 24:
        base_set = {12}
        if maxlabel > 24:
            base_set.add(24)
    else:
        base_set = {8}
    if k in base_set:
        return True
    if k > mr and (k - mr - 3) % 6 == 0 and mr == 24 and k >= 27:
        return True
    if k > mr and (k - mr - 3) % 6 == 0 and mr == 16 and k >= 19:
        return True
    return False


R = re_.filter(pl.col("match_key").is_in(keep)).join(Dp.select(["match_key", "round", "ss"]),
                                                       on=["match_key", "round"], how="left")
R = R.with_columns(
    pl.struct(["match_key", "round"]).map_elements(
        lambda s: in_S(s["round"] - 1, mr_of[s["match_key"]], maxl[s["match_key"]]),
        return_dtype=pl.Boolean).alias("after_swap"))
R = R.with_columns(
    (pl.col("ss") != 0).alias("amb_ok"),
    ((pl.col("ss") != 0) & (pl.col("ss") != pl.col("winner_team_code"))).alias("dis"),
    pl.col("match_key").replace_strict(kind_of, default="noevent").alias("kind"))
V = R.filter(pl.col("amb_ok") & pl.col("round").is_between(2, 29))
print("== A. disputed rate by whether the previous round ended with a physical swap (days 12-25)")
print(V.group_by(["kind", "after_swap"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"),
                                             pl.col("dis").mean().alias("rate")).sort(["kind", "after_swap"]))
print("\nall disputed rounds by round number and after_swap flag:")
print(V.filter(pl.col("dis")).group_by(["round", "after_swap"]).len().sort(["round", "after_swap"]))

# ---------- B. death-based winner for elimination rounds ----------
pdth = scan(base, "player_death", ["match_key", "round", "tick", "player_team_code"]).collect()
last_d = pdth.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("player_team_code").last().alias("victim_team"))
E = R.join(last_d, on=["match_key", "round"], how="inner").filter(
    pl.col("win_reason_code").is_in([8, 9]) & pl.col("amb_ok") & pl.col("round").is_between(2, 29))
E = E.with_columns(
    (5 - pl.col("victim_team")).alias("death_winner"),
    pl.when(pl.col("round").is_in(DISP)).then(pl.lit("disp_round")).otherwise(pl.lit("other_round")).alias("rb"))
E = E.with_columns(pl.when(pl.col("round").is_in(DISP) & pl.col("dis")).then(pl.lit("dis_here"))
                   .when(pl.col("round").is_in(DISP)).then(pl.lit("disp_round_agree"))
                   .otherwise(pl.lit("not_disp_round")).alias("cls"))
print("\n== B. elimination-ended rounds (reason 8/9): death-based winner vs score side vs round_end (days 12-25)")
print(E.group_by(["cls", "kind", "round"]).agg(
    pl.len().alias("n"),
    (pl.col("death_winner") == pl.col("ss")).mean().alias("death_eq_score"),
    (pl.col("death_winner") == pl.col("winner_team_code")).mean().alias("death_eq_roundend"),
).filter(pl.col("n") >= 5).sort(["cls", "kind", "round"]))

# ---------- C. header attributes for disputed vs undisputed matches ----------
disp_m = V.filter(pl.col("dis")).select("match_key").unique().with_columns(pl.lit(True).alias("disputed"))
Mx = M.join(disp_m, on="match_key", how="left").with_columns(pl.col("disputed").fill_null(False))
print("\n== C. header attributes, days 12-25, disputed vs undisputed (counts of distinct values)")
for col in ["rushb_version", "ppp_version", "build_num", "providence"]:
    t = Mx.group_by([col, "disputed"]).len().sort([col, "disputed"])
    print(f"-- {col}: distinct values {Mx[col].n_unique()}")
    print(t.head(12))
