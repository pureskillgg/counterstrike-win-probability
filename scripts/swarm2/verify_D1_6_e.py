"""verify D1-6 (swarm2 verifier), part E: completion filter for the header referee, d01-07 format check, and round sequences.
  E1 completed regulation matches: max(S_T, S_CT) == 13, nr <= 24, S_T + S_CT == nr (rounds 1..nr contiguous).
     Re-run the header referee (headers vs W and L tallies, swap convention) by group: G0 clean, G9_only, G13_only.
  E2 header sum vs round count and max score distribution by day group (format check: MR12 vs other).
  E3 round sequences (winner, ledger, reason, freeze scores) for the first G9_only matches, completed ones if any.
ASSUMPTIONS as in part D; E-A1 completed MR12 match ends when one team reaches 13 rounds (max == 13).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
pl.Config.set_fmt_str_lengths(40)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]
REASON_SIDE = {1: 2, 7: 3, 8: 3, 9: 2, 11: 3, 12: 3, 13: 2, 17: 3, 18: 2}


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

RE = (rd(f24("round_end"), ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
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
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("ledger_side")).select(
    ["match_key", "round", "ledger_side", "t0", "c0"])
A = (RE.join(LED, on=["match_key", "round"], how="left")
     .with_columns(pl.coalesce(["ledger_side", "winner_team_code"]).alias("L"),
                   ((pl.col("ledger_side").is_not_null()) & (pl.col("ledger_side") != pl.col("winner_team_code")))
                   .alias("mm"),
                   pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
A = A.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
A = A.filter(pl.col("round") <= 24)

M = A.group_by("match_key").agg(
    pl.len().alias("nr"), pl.col("round").max().alias("maxr"), pl.col("dg").first().alias("dg"),
    pl.col("mm").sum().alias("n_mm"),
    (pl.col("mm") & (pl.col("round") == 9)).sum().alias("mm9"),
    (pl.col("mm") & (pl.col("round") == 13)).sum().alias("mm13"),
    pl.when(pl.col("round") <= 12).then(pl.col("winner_team_code") == 2).otherwise(pl.col("winner_team_code") == 3)
      .sum().alias("Wswap"),
    pl.when(pl.col("round") <= 12).then(pl.col("L") == 2).otherwise(pl.col("L") == 3).sum().alias("Lswap"))
M = M.join(H, on="match_key", how="inner")
M = M.with_columns(pl.max_horizontal(["S_T", "S_CT"]).alias("mx"))
M = M.with_columns(
    (pl.col("mx") - pl.col("nr")).alias("dummy"))  # placeholder column removed below
M = M.drop("dummy")

hdr("E2. format check by day group: header sum vs round count, max score (all 2024 matches, rounds 1..24 only)")
fmt = M.with_columns(
    ((pl.col("S_T") + pl.col("S_CT")) == pl.col("nr")).alias("sum_eq_nr"),
    (pl.col("maxr") == pl.col("nr")).alias("contig"),
    (pl.col("mx") == 13).alias("mx13"))
print(fmt.group_by(["dg", "sum_eq_nr", "contig", "mx13"]).len().sort(["dg", "sum_eq_nr", "contig", "mx13"]))

hdr("E1. completed regulation matches (max == 13, contiguous, sum == nr): header referee by group")
C = M.filter((pl.col("mx") == 13) & (pl.col("maxr") == pl.col("nr")) & ((pl.col("S_T") + pl.col("S_CT")) == pl.col("nr")))
C = C.with_columns(
    pl.when(pl.col("n_mm") == 0).then(pl.lit("G0_clean"))
      .when((pl.col("n_mm") == 1) & (pl.col("mm9") == 1)).then(pl.lit("G9_only"))
      .when((pl.col("n_mm") == 1) & (pl.col("mm13") == 1)).then(pl.lit("G13_only"))
      .when(pl.col("n_mm") == 1).then(pl.lit("G_other_single"))
      .otherwise(pl.lit("G_multi")).alias("grp"))
print("completed regulation matches:", C.height, " by dg:", C.group_by("dg").len().sort("dg").to_dicts())
for grp in ["G0_clean", "G9_only", "G13_only", "G_other_single", "G_multi"]:
    g = C.filter(pl.col("grp") == grp)
    if g.height == 0:
        print(f"{grp:<16} 0 matches")
        continue
    print(f"{grp:<16} matches={g.height:>5}  header==Wswap {int((g['S_T'] == g['Wswap']).sum()):>5}"
          f"  header==Lswap {int((g['S_T'] == g['Lswap']).sum()):>5}"
          f"  (both {int(((g['S_T'] == g['Wswap']) & (g['S_T'] == g['Lswap'])).sum())})"
          f"  dg {g.group_by('dg').len().sort('dg').to_dicts()}")

hdr("E3. round sequences for the first 3 completed G9_only matches (round, winner, ledger, reason, freeze t0/c0)")
ids = C.filter(pl.col("grp") == "G9_only")["match_key"].head(3).to_list()
for mk in ids:
    print("\n" + mk)
    seq = (A.filter(pl.col("match_key") == mk).select(["round", "winner_team_code", "ledger_side", "win_reason_code", "t0", "c0"])
           .sort("round"))
    print(seq)
    print("header S_T / S_CT:", C.filter(pl.col("match_key") == mk).select(["S_T", "S_CT"]).to_dicts())
