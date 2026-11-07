r"""Verifier for D4-5, step L: match-format families in 2024 (observed first swap boundary) and where
score-vs-round_end disagreements sit, with the header-fit control per family.

Family = first observed swap boundary among complete contiguous matches (n = header total, n <= 30):
  8 (MR8-type: swap after round 8), 12 (MR12-type: swap after round 12), none.
Per family and per round: valid & unambiguous score rounds (mr=24 valid rule as in D4), disagreements with round_end.
Per family: header fit (observed-swap X mapping) for RE and hybrid, for matches with 0 disagreements (control)
and for matches with disagreements.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_l_family.py
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


def swap_expected(r, mr=24):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


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
fam = pm.select(["match_key", "t_starters_score_final", "ct_starters_score_final", "mx"]).join(
    first, on="match_key", how="left").with_columns(pl.col("first_swap").fill_null(-1))
fam = fam.with_columns(pl.when(pl.col("first_swap") == 8).then(pl.lit("F8"))
                       .when(pl.col("first_swap") == 12).then(pl.lit("F12"))
                       .when(pl.col("first_swap") == -1).then(pl.lit("none")).otherwise(pl.lit("other")).alias("fam"))
print("families (complete matches):", fam.group_by("fam").len().sort("fam").rows())

nxt = fe.select(["match_key", "round", "t_score", "ct_score"]).with_columns((pl.col("round") - 1).alias("round")).rename(
    {"t_score": "t2", "ct_score": "c2"})
sd = fe.select(["match_key", "round", "t_score", "ct_score"]).join(nxt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("sc"))
L = re_.join(sd.select(["match_key", "round", "sc"]), on=["match_key", "round"], how="left").join(
    fam.select(["match_key", "fam", "t_starters_score_final", "ct_starters_score_final"]), on="match_key", how="inner")
L = L.with_columns(pl.col("sc").fill_null(0))
L = L.with_columns(
    pl.struct(["round"]).map_elements(lambda s: not swap_expected(s["round"]), return_dtype=pl.Boolean).alias("valid_mr24"))
V = L.filter(pl.col("valid_mr24") & (pl.col("sc") != 0)).with_columns(
    (pl.col("sc") != pl.col("winner_team_code")).alias("dis"))
print("\nvalid (mr24) unambiguous score rounds with disagreement, by family and round (rounds with any disagreement):")
dd = V.group_by(["fam", "round"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).filter(pl.col("dis") > 0).sort(["fam", "round"])
print(dd)
print("\nfamily F8: all rounds 1..16 valid unambiguous n and disagreements:")
print(V.filter(pl.col("fam") == "F8").group_by("round").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("round"))

# header fit per family with observed swaps
cnt = (L.select(["match_key", "round"]).join(swaps, on="match_key", how="left")
       .filter(pl.col("sb") < pl.col("round")).group_by(["match_key", "round"]).agg(pl.len().alias("nsw_before")))
L2 = L.join(cnt, on=["match_key", "round"], how="left").with_columns(pl.col("nsw_before").fill_null(0)).with_columns(
    pl.when(pl.col("nsw_before") % 2 == 0).then(pl.lit(T)).otherwise(pl.lit(CT)).alias("xside"),
    pl.when(pl.col("sc") != 0).then(pl.col("sc")).otherwise(pl.col("winner_team_code")).alias("hyb"))
L2 = L2.with_columns(
    (pl.col("winner_team_code") == pl.col("xside")).cast(pl.Int64).alias("xre"),
    (pl.col("hyb") == pl.col("xside")).cast(pl.Int64).alias("xhy"),
    ((pl.col("sc") != 0) & (pl.col("sc") != pl.col("winner_team_code"))).cast(pl.Int64).alias("dis_all"))
A = L2.group_by("match_key").agg(
    pl.col("xre").sum().alias("xre_w"), pl.col("xhy").sum().alias("xhy_w"), pl.col("dis_all").sum().alias("ndis"),
    pl.len().alias("n"), pl.col("fam").first(), pl.col("t_starters_score_final").first().alias("tA"),
    pl.col("ct_starters_score_final").first().alias("tB"))
A = A.with_columns(
    ((pl.col("xre_w") == pl.col("tA")) & (pl.col("n") - pl.col("xre_w") == pl.col("tB"))).alias("fit_re"),
    ((pl.col("xhy_w") == pl.col("tA")) & (pl.col("n") - pl.col("xhy_w") == pl.col("tB"))).alias("fit_hy"))
print("\nheader fit by family and disagreement status (matches):")
print(A.with_columns(pl.when(pl.col("ndis") == 0).then(pl.lit("no_dis")).otherwise(pl.lit("has_dis")).alias("st")).group_by(
    ["fam", "st"]).agg(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_re"), pl.col("fit_hy").sum().alias("fit_hy")).sort(["fam", "st"]))
print("\nfamily F8 header fit by round-length n (no_dis control):")
print(A.filter((pl.col("fam") == "F8") & (pl.col("ndis") == 0)).group_by("n").agg(
    pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_re"), pl.col("fit_hy").sum().alias("fit_hy")).sort("n"))
