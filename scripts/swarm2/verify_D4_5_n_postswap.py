r"""Verifier for D4-5, step N: is the round_end-vs-score disagreement tied to the FIRST round after an observed side
swap? All 2024 matches (not only complete ones), by observed swap boundary, for d12-25 and the d01-07 control.

Observed swap boundary b: player side-code flip rate > 0.5 over >= 1 player pair in boundary b -> b+1 (player
deaths in both rounds). For each match and each round r with a valid unambiguous score side (no swap expected after
r under mr=24, as in D4), records whether r-1 is an observed swap boundary (r is the first round after a swap) and
whether score disagrees with round_end.

Also tests the lag alternative in F8: SC(9) vs RE(8) and RE(9).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_n_postswap.py
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


re_ = (read("round_end", ["match_key", "round", "tick", "winner_team_code"])
       .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first"))
fe = (read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first"))
de = read("player_death", ["match_key", "round", "player_id", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))
pr = de.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").first().alias("side"))
b = pr.with_columns((pl.col("round") - 1).alias("round")).rename({"side": "side_next"})
pairs = pr.join(b, on=["match_key", "round", "player_id"], how="inner").with_columns(
    (pl.col("side") != pl.col("side_next")).alias("flip"))
bd = pairs.group_by(["match_key", "round"]).agg(pl.len().alias("np"), pl.col("flip").mean().alias("fr"))
# boundary b (round b -> b+1): swap if majority flipped
swb = bd.filter((pl.col("np") >= 1) & (pl.col("fr") > 0.5)).select(["match_key", "round"]).with_columns(
    (pl.col("round") + 1).alias("post_round")).select(["match_key", "post_round"]).unique()
swb = swb.with_columns(pl.lit(True).alias("is_post_swap"))

nxt = fe.select(["match_key", "round", "t_score", "ct_score"]).with_columns((pl.col("round") - 1).alias("round")).rename(
    {"t_score": "t2", "ct_score": "c2"})
sd = fe.select(["match_key", "round", "t_score", "ct_score"]).join(nxt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("sc"))
sd = sd.with_columns(pl.struct(["round"]).map_elements(lambda s: not swap_expected(s["round"]), return_dtype=pl.Boolean).alias("valid"))
L = re_.join(sd.select(["match_key", "round", "sc", "valid"]), on=["match_key", "round"], how="left")
L = L.join(swb.rename({"post_round": "round"}), on=["match_key", "round"], how="left").with_columns(
    pl.col("is_post_swap").fill_null(False))
L = L.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day")).with_columns(
    pl.when(pl.col("day") == "12").then(pl.lit("d12"))
    .when(pl.col("day").is_in(["01", "02", "03", "04", "05", "06", "07"])).then(pl.lit("d01-07"))
    .otherwise(pl.lit("d13-25")).alias("grp"))
V = L.filter(pl.col("valid") & (pl.col("sc") != 0)).with_columns(
    (pl.col("sc") != pl.col("winner_team_code")).alias("dis"))
print("valid unambiguous rounds, disagreement by round class x is_post_swap (observed), d12-25 and d01-07:")
Vr = V.with_columns(pl.when(pl.col("round").is_in([9, 13, 25, 28])).then(pl.lit("spec")).otherwise(pl.lit("other")).alias("rc"))
print(Vr.group_by(["grp", "rc", "is_post_swap"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"), pl.col("dis").mean().alias("rate")).sort(["grp", "rc", "is_post_swap"]))
print("\nwhich rounds are post-swap rounds with disagreements (d12-25):")
print(V.filter((pl.col("grp") != "d01-07") & pl.col("is_post_swap")).group_by("round").agg(
    pl.len().alias("n"), pl.col("dis").sum().alias("dis"), pl.col("dis").mean().alias("rate")).sort("round").filter(pl.col("round") <= 30))
print("\nnon-post-swap rounds with disagreements (d12-25), round 9/13/25/28 only:")
print(V.filter((pl.col("grp") != "d01-07") & ~pl.col("is_post_swap") & pl.col("round").is_in([9, 13, 25, 28])).group_by("round").agg(
    pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("round"))

# lag test in matches with observed swap at boundary 8 (post-swap round 9): SC(9) vs RE(8) and RE(9)
F8 = swb.filter(pl.col("post_round") == 9)["match_key"].to_list()
W8 = (re_.filter(pl.col("round").is_in([8, 9])).select(["match_key", "round", "winner_team_code"]).pivot(
    on="round", index="match_key", values="winner_team_code").rename({"8": "re8", "9": "re9"}))
S9 = sd.filter(pl.col("round") == 9).select(["match_key", "sc"])
lag = W8.join(S9, on="match_key", how="inner").filter(pl.col("match_key").is_in(F8)).filter(pl.col("sc") != 0)
print(f"\nlag test, F8-type matches with unambiguous SC(9): n={lag.height}")
print(lag.select(
    (pl.col("sc") == pl.col("re8")).mean().alias("SC9_eq_RE8"),
    (pl.col("sc") == pl.col("re9")).mean().alias("SC9_eq_RE9")))
