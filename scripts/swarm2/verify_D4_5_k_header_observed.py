r"""Verifier for D4-5, step K: header final-score fit using OBSERVED swap boundaries (data-driven, no fixed schedule).

For each 2024 complete contiguous match (round_end rounds = 1..n, n = header total, n <= 30):
  observed swap boundary b (between round b and b+1) when the player side codes flip for >= 50% of players who died
  in both rounds (requires >= 1 pair). Side of the T-starting team X:
     X_side(r) = T before the first observed swap, flips at each observed swap.
  Y (CT-starter) = the other side. Header totals: tA = X final, tB = Y final.
Labellings: RE (round_end winner), SC (score side; ambiguous rounds fall back to RE), and the count of rounds where
SC != RE. Fit = (X wins == tA and Y wins == tB) for each labelling, and also fit with tA and tB swapped.

Reports fit rates overall, by number of observed swaps, by day group, and for one-disagreement matches by round.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_k_header_observed.py
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
print(f"complete contiguous matches (n<=30, n == header total): {pm.height}")

de = read("player_death", ["match_key", "round", "player_id", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))
pr = de.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").first().alias("side"))
b = pr.with_columns((pl.col("round") - 1).alias("round")).rename({"side": "side_next"})
pairs = pr.join(b, on=["match_key", "round", "player_id"], how="inner").with_columns(
    (pl.col("side") != pl.col("side_next")).alias("flip"))
bd = pairs.group_by(["match_key", "round"]).agg(pl.len().alias("np"), pl.col("flip").mean().alias("fr"))
swaps = bd.filter((pl.col("np") >= 1) & (pl.col("fr") > 0.5)).select(["match_key", "round"]).rename({"round": "sb"})
swaps_per = swaps.group_by("match_key").agg(pl.col("sb").sort().alias("swap_bounds"))

# per round labels
nxt = fe.select(["match_key", "round", "t_score", "ct_score"]).with_columns((pl.col("round") - 1).alias("round")).rename(
    {"t_score": "t2", "ct_score": "c2"})
sd = fe.select(["match_key", "round", "t_score", "ct_score"]).join(nxt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("sc"))
L = re_.join(sd.select(["match_key", "round", "sc"]), on=["match_key", "round"], how="left").join(
    pm.select(["match_key", "n_rows", "t_starters_score_final", "ct_starters_score_final"]), on="match_key", how="inner")
L = L.with_columns(pl.col("sc").fill_null(0))
# number of observed swap boundaries strictly before round r, per (match, round)
cnt = (L.select(["match_key", "round"]).join(swaps, on="match_key", how="left")
       .filter(pl.col("sb") < pl.col("round")).group_by(["match_key", "round"]).agg(pl.len().alias("nsw_before")))
L = L.join(cnt, on=["match_key", "round"], how="left").with_columns(pl.col("nsw_before").fill_null(0))
L = L.join(swaps_per, on="match_key", how="left").with_columns(
    pl.col("swap_bounds").fill_null(pl.lit([], dtype=pl.List(pl.Int64))))
L = L.with_columns(
    pl.when(pl.col("nsw_before") % 2 == 0).then(pl.lit(T)).otherwise(pl.lit(CT)).alias("xside"),
    pl.when((pl.col("sc") != 0)).then(pl.col("sc")).otherwise(pl.col("winner_team_code")).alias("hyb"))
L = L.with_columns(
    (pl.col("winner_team_code") == pl.col("xside")).cast(pl.Int64).alias("xre"),
    (pl.col("hyb") == pl.col("xside")).cast(pl.Int64).alias("xhy"),
    ((pl.col("sc") != 0) & (pl.col("sc") != pl.col("winner_team_code"))).cast(pl.Int64).alias("dis"))
agg = L.group_by("match_key").agg(
    pl.col("xre").sum().alias("xre_w"), pl.col("xhy").sum().alias("xhy_w"), pl.col("dis").sum().alias("ndis"),
    pl.col("round").filter(pl.col("dis") == 1).alias("dis_rounds"),
    pl.col("t_starters_score_final").first().alias("tA"), pl.col("ct_starters_score_final").first().alias("tB"),
    pl.col("n_rows").first().alias("n"), pl.col("swap_bounds").first().alias("swaps"),
)
agg = agg.with_columns(
    ((pl.col("xre_w") == pl.col("tA")) & (pl.col("n") - pl.col("xre_w") == pl.col("tB"))).alias("fit_re"),
    ((pl.col("xhy_w") == pl.col("tA")) & (pl.col("n") - pl.col("xhy_w") == pl.col("tB"))).alias("fit_hy"),
    ((pl.col("xre_w") == pl.col("tB")) & (pl.col("n") - pl.col("xre_w") == pl.col("tA"))).alias("fit_re_swapped_hdr"),
    ((pl.col("xhy_w") == pl.col("tB")) & (pl.col("n") - pl.col("xhy_w") == pl.col("tA"))).alias("fit_hy_swapped_hdr"),
    pl.col("swaps").list.len().alias("n_swaps"),
    pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
)
agg = agg.with_columns(pl.when(pl.col("day") == "12").then(pl.lit("d12")).when(pl.col("day").is_in(["01", "02", "03", "04", "05", "06", "07"])).then(pl.lit("d01-07")).otherwise(pl.lit("d13-25")).alias("grp"))
print("\nfit summary (header totals as X/Y, observed swaps):")
print(agg.select(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_roundend"), pl.col("fit_hy").sum().alias("fit_hybrid"),
                 pl.col("fit_re_swapped_hdr").sum().alias("fit_re_hdr_swapped"), pl.col("fit_hy_swapped_hdr").sum().alias("fit_hy_hdr_swapped")))
print("\nby number of observed swaps:")
print(agg.group_by("n_swaps").agg(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_re"), pl.col("fit_hy").sum().alias("fit_hy")).sort("n_swaps"))
print("\nby day group:")
print(agg.group_by("grp").agg(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_re"), pl.col("fit_hy").sum().alias("fit_hy")).sort("grp"))
print("\nby observed swap boundaries (first swap round) for matches with disagreements:")
dd = agg.filter(pl.col("ndis") > 0).with_columns(pl.col("swaps").list.first().alias("first_swap"))
print(dd.group_by("first_swap").agg(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_re"), pl.col("fit_hy").sum().alias("fit_hy")).sort("first_swap"))
print("\nsingle-disagreement matches: round, matches, fit_re, fit_hybrid:")
one = agg.filter(pl.col("ndis") == 1).with_columns(pl.col("dis_rounds").list.first().alias("dr"))
print(one.group_by("dr").agg(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_re"), pl.col("fit_hy").sum().alias("fit_hy")).sort("dr"))
print("\nmatches with NO disagreement (control): fit_re, fit_hybrid:")
ctl = agg.filter(pl.col("ndis") == 0)
print(ctl.select(pl.len().alias("matches"), pl.col("fit_re").sum().alias("fit_re"), pl.col("fit_hy").sum().alias("fit_hy")))
print("\nall complete matches, multi-disagreement count:", agg.filter(pl.col("ndis") > 1).height)
agg.select(["match_key", "n", "tA", "tB", "n_swaps", "ndis", "fit_re", "fit_hy"]).head(3)
