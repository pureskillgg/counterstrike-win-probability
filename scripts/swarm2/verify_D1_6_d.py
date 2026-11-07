"""verify D1-6 (swarm2 verifier), part D: game-produced header totals as an independent referee for each match.
Uses ALL rounds of each 2024 match (every player_count), because the header counts every round.
  W  = round_end winner_team_code.  L = freeze-to-freeze ledger (ledger_side, falling back to W where null).
  Two orientation conventions for the starters' totals, chosen from the control group (no mismatch anywhere in 1-24):
    'side'   : header t_starters_score_final == number of rounds in 1..n won by code 2 (T) (no halftime swap).
    'swap'   : starters-T team wins rounds 1..12 when code 2, rounds 13..n when code 3 (halftime swap).
  Matches used: regulation-only (round count n <= 24 with rounds 1..n complete, and header t+ct final == n).
ASSUMPTIONS:
  D-A1 header t_starters_score_final / ct_starters_score_final are the game's final totals per starting side.
  D-A2 the freeze ledger is the game's own score state (independent of round_end).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
pl.Config.set_fmt_str_lengths(50)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def f24(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


def rd(files, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


hf = sorted(glob.glob(str(T24 / "header" / "dataframe_*")))
H = pl.concat([pl.read_parquet(f, columns=["key", "t_starters_score_final", "ct_starters_score_final"]) for f in hf],
              how="vertical_relaxed").select([pl.col("key").alias("match_key"),
                                             pl.col("t_starters_score_final").alias("S_T"),
                                             pl.col("ct_starters_score_final").alias("S_CT")]).unique("match_key")

RE = (rd(f24("round_end"), ["match_key", "round", "tick", "winner_team_code", "player_count"])
      .unique(["match_key", "round"], keep="first", maintain_order=True).rename({"tick": "end_tick"}))
RS = rd(f24("round_state"), ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
FZ = (RS.filter(pl.col("event_type") == "round_freeze_end")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t0"),
                                            pl.col("ct_score").first().alias("c0")))
FZn = FZ.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("t0").alias("t1"), pl.col("c0").alias("c1")])
LED = FZ.join(FZn, on=["match_key", "round"], how="left").with_columns(
    pl.when(((pl.col("t1") - pl.col("t0")) == 1) & ((pl.col("c1") - pl.col("c0")) == 0)).then(pl.lit(2))
      .when(((pl.col("c1") - pl.col("c0")) == 1) & ((pl.col("t1") - pl.col("t0")) == 0)).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("ledger_side")).select(["match_key", "round", "ledger_side"])

A = (RE.join(LED, on=["match_key", "round"], how="left")
     .filter(pl.col("round") <= 24)
     .with_columns(pl.coalesce(["ledger_side", "winner_team_code"]).alias("L"),
                   ((pl.col("ledger_side").is_not_null()) & (pl.col("ledger_side") != pl.col("winner_team_code")))
                   .alias("mm"))
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
A = A.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

M = A.group_by("match_key").agg(
    pl.len().alias("nr"), pl.col("round").max().alias("maxr"), pl.col("dg").first().alias("dg"),
    pl.col("mm").sum().alias("n_mm"),
    (pl.col("mm") & (pl.col("round") == 9)).sum().alias("mm9"),
    (pl.col("mm") & (pl.col("round") == 13)).sum().alias("mm13"),
    (pl.col("winner_team_code") == 2).sum().alias("W2"),
    (pl.col("winner_team_code") == 3).sum().alias("W3"),
    (pl.col("L") == 2).sum().alias("L2"),
    (pl.col("L") == 3).sum().alias("L3"),
    pl.when(pl.col("round") <= 12).then(pl.col("winner_team_code") == 2).otherwise(pl.col("winner_team_code") == 3)
      .sum().alias("Wswap"),
    pl.when(pl.col("round") <= 12).then(pl.col("L") == 2).otherwise(pl.col("L") == 3).sum().alias("Lswap"),
)
M = M.join(H, on="match_key", how="inner")
REG = M.filter((pl.col("maxr") == pl.col("nr")) & (pl.col("nr") <= 24) & ((pl.col("S_T") + pl.col("S_CT")) == pl.col("nr")))
hdr("D0. universe")
print("matches with rounds 1..24 contiguous and header final sum == n (regulation-only):", REG.height,
      " total matches:", M.height)
print("by dg:", REG.group_by("dg").len().sort("dg").to_dicts())
print("n_mm distribution (all matches, rounds 1..24, ledger vs winner):",
      M.group_by("n_mm").len().sort("n_mm").to_dicts())

# group labels: which round is the single mismatch (if exactly one)
REG = REG.with_columns(
    pl.when(pl.col("n_mm") == 0).then(pl.lit("G0_clean"))
      .when((pl.col("n_mm") == 1) & (pl.col("mm9") == 1)).then(pl.lit("G9_only"))
      .when((pl.col("n_mm") == 1) & (pl.col("mm13") == 1)).then(pl.lit("G13_only"))
      .when(pl.col("n_mm") == 1).then(pl.lit("G_other_single"))
      .otherwise(pl.lit("G_multi")).alias("grp"))

hdr("D1. convention check on the clean group (n_mm == 0): which header convention matches")
for dgv in ["d12-25", "d01-07"]:
    g = REG.filter((pl.col("grp") == "G0_clean") & (pl.col("dg") == dgv))
    print(f"{dgv} clean matches: {g.height}"
          f"  S_T==W2 (side): {int((g['S_T'] == g['W2']).sum())}"
          f"  S_T==Wswap (swap): {int((g['S_T'] == g['Wswap']).sum())}"
          f"  S_T==L2 (side, ledger): {int((g['S_T'] == g['L2']).sum())}"
          f"  S_T==Lswap (swap, ledger): {int((g['S_T'] == g['Lswap']).sum())}")

hdr("D2. header referee on matches with exactly one mismatch round (single-locus groups)")
for grp in ["G9_only", "G13_only", "G_other_single", "G0_clean", "G_multi"]:
    g = REG.filter(pl.col("grp") == grp)
    if g.height == 0:
        print(f"{grp}: 0 matches")
        continue
    sw_W = int((g["S_T"] == g["Wswap"]).sum())
    sw_L = int((g["S_T"] == g["Lswap"]).sum())
    sd_W = int((g["S_T"] == g["W2"]).sum())
    sd_L = int((g["S_T"] == g["L2"]).sum())
    print(f"{grp:<16} matches={g.height:>5}  swap: header==W {sw_W:>5}  header==L {sw_L:>5}"
          f"   side: header==W {sd_W:>5}  header==L {sd_L:>5}   (dg counts {g.group_by('dg').len().sort('dg').to_dicts()})")

hdr("D3. G9_only matches: detail (first 6)")
print(REG.filter(pl.col("grp") == "G9_only").select(
    ["match_key", "nr", "S_T", "S_CT", "W2", "L2", "Wswap", "Lswap", "mm9", "mm13"]).head(6))
