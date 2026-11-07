r"""Verifier for D4-5, step M: which side-swap mapping and which labelling the header totals support, per family.

Complete contiguous 2024 matches (n = header total, n <= 30), families from observed first swap (F8, F12, none).
Candidate X-side mappings (X = T-starting team, X starts on T):
  M_obs   : swaps at observed boundaries (player side codes)
  M_s8    : swap after 8 only (no other swap)
  M_s12   : swap after 12 (and OT swap after 27 for n > 24), the D4 schedule
  M_none  : no swap at all
Labellings: RE (round_end winner), HYB (score side where unambiguous, else RE).
For each family, reports the number of matches whose header totals fit (X wins == tA and Y wins == tB) for each
(mapping, labelling) pair, and the best mapping per labelling.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_m_mapping.py
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
pm = re_.group_by("match_key").agg(pl.len().alias("n_rows"), pl.col("round").max().alias("mx"), pl.col("round").min().alias("mn"))
pm = pm.join(H.select(["key", "fsum", "t_starters_score_final", "ct_starters_score_final"]), left_on="match_key",
             right_on="key", how="left")
pm = pm.filter((pl.col("mx") == pl.col("n_rows")) & (pl.col("mn") == 1) & (pl.col("fsum") == pl.col("mx")) & (pl.col("mx") <= 30))

de = read("player_death", ["match_key", "round", "player_id", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))
pr = de.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").first().alias("side"))
b = pr.with_columns((pl.col("round") - 1).alias("round")).rename({"side": "side_next"})
pairs = pr.join(b, on=["match_key", "round", "player_id"], how="inner").with_columns(
    (pl.col("side") != pl.col("side_next")).alias("flip"))
bd = pairs.group_by(["match_key", "round"]).agg(pl.len().alias("np"), pl.col("flip").mean().alias("fr"))
swaps = bd.filter((pl.col("np") >= 1) & (pl.col("fr") > 0.5)).select(["match_key", "round"]).rename({"round": "sb"})
first = swaps.group_by("match_key").agg(pl.col("sb").min().alias("first_swap"))
fam = pm.select(["match_key", "t_starters_score_final", "ct_starters_score_final", "n_rows"]).join(
    first, on="match_key", how="left").with_columns(pl.col("first_swap").fill_null(-1))
fam = fam.with_columns(pl.when(pl.col("first_swap") == 8).then(pl.lit("F8"))
                       .when(pl.col("first_swap") == 12).then(pl.lit("F12"))
                       .when(pl.col("first_swap") == -1).then(pl.lit("none")).otherwise(pl.lit("other")).alias("fam"))

nxt = fe.select(["match_key", "round", "t_score", "ct_score"]).with_columns((pl.col("round") - 1).alias("round")).rename(
    {"t_score": "t2", "ct_score": "c2"})
sd = fe.select(["match_key", "round", "t_score", "ct_score"]).join(nxt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("sc"))
L = re_.join(sd.select(["match_key", "round", "sc"]), on=["match_key", "round"], how="left").join(
    fam.select(["match_key", "fam", "t_starters_score_final", "ct_starters_score_final", "first_swap"]), on="match_key", how="inner")
L = L.with_columns(pl.col("sc").fill_null(0)).with_columns(
    pl.when(pl.col("sc") != 0).then(pl.col("sc")).otherwise(pl.col("winner_team_code")).alias("hyb"))
L = L.with_columns(
    pl.when(pl.col("round") <= 12).then(pl.lit(T)).otherwise(pl.lit(CT)).alias("x_s12"))
# observed mapping: count observed swaps before round
cnt = (L.select(["match_key", "round"]).join(swaps, on="match_key", how="left")
       .filter(pl.col("sb") < pl.col("round")).group_by(["match_key", "round"]).agg(pl.len().alias("nsw")))
L = L.join(cnt, on=["match_key", "round"], how="left").with_columns(pl.col("nsw").fill_null(0))
L = L.with_columns(
    pl.when(pl.col("nsw") % 2 == 0).then(pl.lit(T)).otherwise(pl.lit(CT)).alias("x_obs"),
    pl.when(pl.col("round") <= 8).then(pl.lit(T)).otherwise(pl.lit(CT)).alias("x_s8"),
    pl.lit(T).alias("x_none"))
# OT for n > 24 under S12: swap after 27 => side flips for rounds >= 28
L = L.with_columns(pl.when((pl.col("round") > 27) & (pl.col("round") <= 30)).then(pl.lit(T)).otherwise(pl.col("x_s12")).alias("x_s12"))

# per match fit counts for each mapping x labelling
res = []
for mapping in ["x_obs", "x_s8", "x_s12", "x_none"]:
    Lm = L.with_columns(
        (pl.col("winner_team_code") == pl.col(mapping)).cast(pl.Int64).alias("re_x"),
        (pl.col("hyb") == pl.col(mapping)).cast(pl.Int64).alias("hy_x"))
    A = Lm.group_by("match_key").agg(
        pl.col("re_x").sum().alias("re_w"), pl.col("hy_x").sum().alias("hy_w"), pl.len().alias("n"),
        pl.col("fam").first(), pl.col("t_starters_score_final").first().alias("tA"), pl.col("ct_starters_score_final").first().alias("tB"))
    A = A.with_columns(
        ((pl.col("re_w") == pl.col("tA")) & (pl.col("n") - pl.col("re_w") == pl.col("tB"))).alias("fit_re"),
        ((pl.col("hy_w") == pl.col("tA")) & (pl.col("n") - pl.col("hy_w") == pl.col("tB"))).alias("fit_hy"),
        pl.lit(mapping).alias("mapping"))
    res.append(A)
R = pl.concat(res)
print("fit counts by family x mapping (matches fitting the header):")
print(R.group_by(["fam", "mapping"]).agg(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_roundend"),
                                          pl.col("fit_hy").sum().alias("fit_hybrid")).sort(["fam", "mapping"]))
print("\nF8 and F12 matches: best mapping per labelling (share of matches fitting):")
print(R.filter(pl.col("fam").is_in(["F8", "F12"])).group_by(["fam", "mapping"]).agg(
    pl.len().alias("n"), pl.col("fit_re").mean().alias("re_rate"), pl.col("fit_hy").mean().alias("hy_rate")).sort(["fam", "mapping"]))
