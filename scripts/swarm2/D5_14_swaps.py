"""D5 step 14: where do the freeze-end score fields swap sides, and does the wrong-label site follow the first swap?

For each match and each boundary r -> r+1 (r = 1..29), classify the freeze-end score fields using round_end
winner w_r (a = [w_r == 2], b = [w_r == 3]):
  no_swap    t(r+1) == t(r) + a  and  ct(r+1) == ct(r) + b       (fields follow the team on each side)
  side_swap  t(r+1) == ct(r) + b and  ct(r+1) == t(r) + a         (fields follow the side; halftime-style swap)
  ambig      both hold; other      neither holds
Then per match: the list of boundaries classified side_swap, and the first such boundary.
Compares with the wrong-label site (from D5_13: w9 / w13 / w25 / w28 using valid steps).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_14_swaps.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet").select(["match_key", "round", "fe_t_score", "fe_ct_score", "winner"]).sort(
    ["match_key", "round"])
M = pl.read_parquet(SCR / "matches.parquet")
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
mk = pl.read_parquet(SCR / "rounds.parquet").group_by("match_key").agg(
    pl.col("n_round_announce_match_start").fill_null(0).sum().alias("mstart"))
M = (M.join(mk, on="match_key", how="left")
     .with_columns(regime=pl.when(pl.col("mstart") > 0).then(pl.lit("A")).otherwise(pl.lit("B")),
                   grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other"))))

N = R.with_columns(nt=pl.col("fe_t_score").shift(-1).over("match_key"),
                   nct=pl.col("fe_ct_score").shift(-1).over("match_key"),
                   nr=pl.col("round").shift(-1).over("match_key"))
N = N.filter((pl.col("nr") == pl.col("round") + 1) & pl.col("fe_t_score").is_not_null() & pl.col("nt").is_not_null()
             & pl.col("winner").is_not_null())
a = pl.when(pl.col("winner") == 2).then(1).otherwise(0)
b = pl.when(pl.col("winner") == 3).then(1).otherwise(0)
N = N.with_columns(
    ns=(pl.col("nt") == pl.col("fe_t_score") + a) & (pl.col("nct") == pl.col("fe_ct_score") + b),
    sw=(pl.col("nt") == pl.col("fe_ct_score") + b) & (pl.col("nct") == pl.col("fe_t_score") + a))
N = N.with_columns(cls=pl.when(pl.col("ns") & pl.col("sw")).then(pl.lit("ambig"))
                   .when(pl.col("ns")).then(pl.lit("no_swap"))
                   .when(pl.col("sw")).then(pl.lit("side_swap")).otherwise(pl.lit("other")))
print("boundary classification counts (all boundaries 1..29, all groups):")
print(N.group_by("cls").len().sort("cls"))

# boundaries classified side_swap per group
SW = N.filter(pl.col("cls") == "side_swap").select(["match_key", "round"])
print("\nside_swap boundaries by round (label r where r -> r+1 swaps), by group:")
SWg = SW.join(M.select(["match_key", "grp", "regime"]), on="match_key", how="left")
print(SWg.group_by(["grp", "regime", "round"]).agg(pl.len().alias("matches")).sort(["grp", "regime", "round"]).filter(
    pl.col("round").is_in([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29])))

# per-match first swap boundary
first = SW.group_by("match_key").agg(pl.col("round").min().alias("first_swap_r"),
                                     pl.col("round").n_unique().alias("n_swaps"))
F = M.select(["match_key", "grp", "regime"]).join(first, on="match_key", how="left")
print("\nper-match first side-swap boundary (late regime A / B and early):")
print(F.group_by(["grp", "regime", "first_swap_r"]).agg(pl.len().alias("matches")).sort(["grp", "regime", "first_swap_r"], nulls_last=True)
      .filter(pl.col("matches") > 0).head(80))
print("\nn_swaps per match:")
print(F.group_by(["grp", "regime", "n_swaps"]).agg(pl.len().alias("matches")).sort(["grp", "regime", "n_swaps"], nulls_last=True))

# wrong-label site from valid steps: label 9 or 13 (from D5_13 logic), per match
# valid steps only (clean +1 steps)
V = N.filter(pl.col("round").is_in([9, 13, 25, 28]) & (pl.col("cls") != "ambig")).with_columns(
    dt=pl.col("nt") - pl.col("fe_t_score"), dct=pl.col("nct") - pl.col("fe_ct_score")).with_columns(
    ch=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
    .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)))
V = V.filter(pl.col("ch") != 0).with_columns(wrong=(pl.col("ch") != pl.col("winner")).cast(pl.Int8))
site = V.filter(pl.col("wrong") == 1).group_by("match_key").agg(pl.col("round").min().alias("first_wrong_label"),
                                                                pl.col("round").sort().alias("wrong_labels"))
site = site.with_columns(wrong_str=pl.col("wrong_labels").cast(pl.List(pl.String)).list.join(","))
G = F.join(site.select(["match_key", "first_wrong_label", "wrong_str"]), on="match_key", how="left")
print("\nlate regime-A ctrl: first_swap_r vs first_wrong_label (matches with both):")
print(G.filter((pl.col("grp") == "ctrl_late") & (pl.col("regime") == "A") & pl.col("first_swap_r").is_not_null()
               & pl.col("first_wrong_label").is_not_null())
      .group_by(["first_swap_r", "first_wrong_label"]).agg(pl.len().alias("matches")).sort(["first_swap_r", "first_wrong_label"]))
print("\nlate regime-A ctrl: wrong-label pattern vs first side-swap boundary (matches with a wrong label):")
print(G.filter((pl.col("grp") == "ctrl_late") & (pl.col("regime") == "A") & pl.col("first_wrong_label").is_not_null())
      .group_by(["first_swap_r", "wrong_str"]).agg(pl.len().alias("matches")).sort(["first_swap_r", "wrong_str"]))
print("\naff (regime A, all halftime): first_swap_r and wrong pattern:")
print(G.filter(pl.col("grp") == "aff_r13").group_by(["first_swap_r", "wrong_str"]).agg(pl.len().alias("matches"))
      .sort(["first_swap_r", "wrong_str"]))
print("\nregime B (ctrl and early): first_swap_r distribution:")
print(G.filter(pl.col("regime") == "B").group_by(["grp", "first_swap_r"]).agg(pl.len().alias("matches"))
      .sort(["grp", "first_swap_r"], nulls_last=True))
