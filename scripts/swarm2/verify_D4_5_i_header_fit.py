r"""Verifier for D4-5, step I: header final-score fit, 2024 complete regulation matches.

Ground-truth candidate: header t_starters_score_final (T-starting team's final rounds won) and
ct_starters_score_final (CT-starting team). Under the standard schedule (no swap before round 12, swap after 12,
no OT in matches with n <= 24) the team that started on T won rounds where the winner side equals its side:
  X (T-starter) side at round r = T if r <= 12 else CT ;  Y (CT-starter) = the other side.
Per match, count X_wins and Y_wins under each labelling and compare with the header totals.

Labellings for each round:
  RE    = round_end.winner_team_code
  SC    = score side (freeze_end r -> r+1; unambiguous rounds only; rounds with ambiguous score are taken as RE)
  HYB   = SC where SC is defined, else RE  (what the D4 fix would produce where it applies)

Scope: 2024 matches with n_distinct rounds = 1..n (contiguous, round_end present), n <= 24, n == header total,
and NO observed side flip at boundary 8 (player side codes; excludes the round-8 swap family), so the
S12 schedule is the observed one. Reports fit counts by labelling and by which of rounds 9/13/25/28 disagree.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_i_header_fit.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(18)
pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final"]).with_columns(
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("fsum"))
re_ = (read("round_end", ["match_key", "round", "tick", "winner_team_code"])
       .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first"))
fe = (read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first"))

# contiguous complete matches
pm = re_.group_by("match_key").agg(pl.len().alias("n_rows"), pl.col("round").max().alias("mx"), pl.col("round").min().alias("mn"))
pm = pm.join(H.select(["key", "fsum", "t_starters_score_final", "ct_starters_score_final"]), left_on="match_key",
             right_on="key", how="left")
pm = pm.filter((pl.col("mx") == pl.col("n_rows")) & (pl.col("mn") == 1) & (pl.col("fsum") == pl.col("mx")) & (pl.col("mx") <= 24))
print(f"complete contiguous matches n<=24 with header total == rounds: {pm.height}")
keep = set(pm["match_key"].to_list())

# side-flip evidence at boundary 8 (exclude round-8 swap family)
de = read("player_death", ["match_key", "round", "player_id", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))
pr = de.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").first().alias("side"))
a = pr
b = pr.with_columns((pl.col("round") - 1).alias("round")).rename({"side": "side_next"})
pairs = a.join(b, on=["match_key", "round", "player_id"], how="inner").with_columns(
    (pl.col("side") != pl.col("side_next")).alias("flip"))
f8 = pairs.filter(pl.col("round") == 8).group_by("match_key").agg(pl.len().alias("n8"), pl.col("flip").mean().alias("fr8"))
f12 = pairs.filter(pl.col("round") == 12).group_by("match_key").agg(pl.len().alias("n12"), pl.col("flip").mean().alias("fr12"))
flipinfo = f8.join(f12, on="match_key", how="full", coalesce=True)
bad8 = set(flipinfo.filter((pl.col("n8") >= 2) & (pl.col("fr8") > 0.5))["match_key"].to_list())
ok12 = set(flipinfo.filter((pl.col("n12") >= 2) & (pl.col("fr12") > 0.5))["match_key"].to_list())
print(f"matches with flip at 8 (excluded): {len(bad8 & keep)}; matches with observed flip at 12: {len(ok12 & keep)}")
keep = keep - bad8

# labels per round
nxt = fe.select(["match_key", "round", "t_score", "ct_score"]).with_columns((pl.col("round") - 1).alias("round")).rename(
    {"t_score": "t2", "ct_score": "c2"})
sd = fe.select(["match_key", "round", "t_score", "ct_score"]).join(nxt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
L = (re_.filter(pl.col("match_key").is_in(list(keep))).join(
    sd.select(["match_key", "round", "score_side"]), on=["match_key", "round"], how="left")
    .join(pm.select(["match_key", "t_starters_score_final", "ct_starters_score_final"]), on="match_key", how="inner"))
L = L.with_columns(
    pl.col("score_side").fill_null(0).alias("score_side"),
    pl.when(pl.col("score_side") != 0).then(pl.col("score_side")).otherwise(pl.col("winner_team_code")).alias("hyb"))
# X = T-starter: side at round r
L = L.with_columns(pl.when(pl.col("round") <= 12).then(pl.lit(T)).otherwise(pl.lit(CT)).alias("xside"))
L = L.with_columns(
    (pl.col("winner_team_code") == pl.col("xside")).alias("x_re"),
    (pl.col("score_side") == pl.col("xside")).alias("x_sc"),
    (pl.col("hyb") == pl.col("xside")).alias("x_hy"))
agg = L.group_by("match_key").agg(
    pl.col("x_re").sum().alias("x_re_w"), pl.col("x_sc").sum().alias("x_sc_w"), pl.col("x_hy").sum().alias("x_hy_w"),
    pl.col("t_starters_score_final").first().alias("tA"), pl.col("ct_starters_score_final").first().alias("tB"),
    pl.len().alias("n"),
    (pl.col("score_side") != 0).sum().alias("n_amb_free"),
    (pl.col("score_side") != pl.col("winner_team_code")).filter(pl.col("score_side") != 0).sum().alias("n_dis_all"),
    pl.col("round").filter((pl.col("score_side") != pl.col("winner_team_code")) & (pl.col("score_side") != 0)).alias("dis_rounds"),
)
agg = agg.with_columns(
    ((pl.col("x_re_w") == pl.col("tA")) & (pl.col("n") - pl.col("x_re_w") == pl.col("tB"))).alias("fit_re"),
    ((pl.col("x_sc_w") == pl.col("tA")) & (pl.col("n") - pl.col("x_sc_w") == pl.col("tB"))).alias("fit_sc"),
    ((pl.col("x_hy_w") == pl.col("tA")) & (pl.col("n") - pl.col("x_hy_w") == pl.col("tB"))).alias("fit_hy"),
    pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
)
agg = agg.with_columns(pl.col("dis_rounds").list.eval(pl.element().is_in([9, 13, 25, 28])).list.any().alias("has_spec"))

print("\nfit to header totals, all complete matches (n<=24, no flip at 8):")
print(agg.select(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_roundend"),
                 pl.col("fit_sc").sum().alias("fit_score"), pl.col("fit_hy").sum().alias("fit_hybrid")))

print("\nfit by matches that have >=1 score!=round_end disagreement in rounds 9/13/25/28:")
sub = agg.filter(pl.col("has_spec"))
print(sub.select(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_roundend"),
                 pl.col("fit_sc").sum().alias("fit_score"), pl.col("fit_hy").sum().alias("fit_hybrid")))

print("\nfit by matches with NO score!=round_end disagreement (control):")
ctl = agg.filter(~pl.col("has_spec"))
print(ctl.select(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_roundend"),
                 pl.col("fit_sc").sum().alias("fit_score"), pl.col("fit_hy").sum().alias("fit_hybrid")))

print("\nmatches where disagreements are exactly one round (r in 9,13,25,28) -- the cleanest test:")
one = agg.filter(pl.col("n_dis_all") == 1).with_columns(pl.col("dis_rounds").list.first().alias("dr"))
print(one.group_by("dr").agg(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_roundend"),
                              pl.col("fit_sc").sum().alias("fit_score"), pl.col("fit_hy").sum().alias("fit_hybrid")).sort("dr"))

print("\nby day group, single-disagreement matches:")
print(one.with_columns(pl.when(pl.col("day") == "12").then(pl.lit("d12")).otherwise(pl.lit("d13-25")).alias("g")).group_by(["g", "dr"]).agg(
    pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_roundend"), pl.col("fit_sc").sum().alias("fit_score")).sort(["g", "dr"]))

print("\nall matches: fit rate by day group (round_end vs score)")
print(agg.with_columns(pl.when(pl.col("day") == "12").then(pl.lit("d12")).when(pl.col("day").is_in(["01", "02", "03", "04", "05", "06", "07"])).then(pl.lit("d01-07")).otherwise(pl.lit("d13-25"))
                       .alias("g")).group_by("g").agg(pl.len().alias("matches"), pl.col("fit_re").mean().alias("fit_re_rate"),
                                                       pl.col("fit_sc").mean().alias("fit_sc_rate")).sort("g"))
