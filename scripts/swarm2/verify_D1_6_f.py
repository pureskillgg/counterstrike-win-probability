"""verify D1-6 (swarm2 verifier), part F: loci, completeness, death-code consistency, d01-07 structure.
  F1 per-match mismatch loci over ALL rounds (incl. overtime), d12-25 and d01-07, all player_counts: count of mismatched
     rounds per match and which round index.
  F2 for the 305 claim matches (pc20, round 9 score side vs winner): completeness (max(S) == 13?), header sum vs round count.
  F3 death-team-code consistency: round-9 death rows (player_death) vs player_info team_code for the same player and round,
     in the 305 mismatch matches vs the 2944 agreeing matches. (The elimination test depends on death team codes.)
  F4 round-count structure by day group (contiguity and header sum), all matches.
  F5 for complete matches with a round-13 mismatch: header == W2 / L2 (side) and header == Wswap / Lswap (swap).
ASSUMPTIONS: as in parts B-E. Overtime rounds are included in F1 only.
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
SC = RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("t_score").sort_by("tick").first().alias("tb"), pl.col("ct_score").sort_by("tick").first().alias("cb"),
    pl.col("t_score").sort_by("tick").last().alias("ta"), pl.col("ct_score").sort_by("tick").last().alias("ca"))
SC = SC.with_columns(
    pl.when((pl.col("ta") - pl.col("tb")) == 1).then(pl.lit(2))
      .when((pl.col("ca") - pl.col("cb")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side")).select(["match_key", "round", "score_side"])
A = (RE.join(LED, on=["match_key", "round"], how="left").join(SC, on=["match_key", "round"], how="left")
     .with_columns(pl.coalesce(["ledger_side", "winner_team_code"]).alias("L"),
                   ((pl.col("ledger_side").is_not_null()) & (pl.col("ledger_side") != pl.col("winner_team_code")))
                   .alias("mm"),
                   ((pl.col("score_side").is_not_null()) & (pl.col("score_side") != pl.col("winner_team_code")))
                   .alias("mmC"),
                   pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
A = A.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

# ----------------------------------------------------------------------------- F1 loci, all rounds
hdr("F1. mismatched rounds per match (ledger vs winner), ALL rounds incl. overtime, all player_counts")
PM = A.group_by("match_key").agg(pl.col("dg").first().alias("dg"), pl.col("mm").sum().alias("n_mm"),
                                 pl.col("mm").cast(pl.Int64).sum().alias("n_mm2"),
                                 pl.col("round").max().alias("maxr"), pl.len().alias("nr"))
print("matches by dg and n_mm:")
print(PM.group_by(["dg", "n_mm"]).len().sort(["dg", "n_mm"]))
locus = A.filter(pl.col("mm")).group_by(["dg", "round"]).agg(pl.len().alias("mismatched_rows"),
                                                              pl.col("match_key").n_unique().alias("matches")).sort(["dg", "round"])
print("mismatched rows by round (ledger vs winner):")
print(locus)
ledger_rows = A.filter(pl.col("ledger_side").is_not_null()).group_by(["dg", "round"]).agg(pl.len().alias("rows_with_ledger"))
print("ledger rows by round, d12-25 (denominator):")
print(ledger_rows.filter(pl.col("dg") == "d12-25").sort("round"))

# ----------------------------------------------------------------------------- F2 the 305 claim matches
hdr("F2. completeness of the 305 claim matches (pc20, round-9 score side != winner, d12-25)")
PC = A.filter(pl.col("player_count") == 20).filter(pl.col("round") == 9)
claim305 = PC.filter(pl.col("mmC") & (pl.col("dg") == "d12-25"))["match_key"].to_list()
print("claim matches:", len(claim305))
Mx = (A.filter(pl.col("match_key").is_in(claim305)).group_by("match_key")
      .agg(pl.len().alias("nr_all"), pl.col("round").max().alias("maxr_all"),
           pl.col("mm").sum().alias("n_mm_all")))
Mx = Mx.join(H, on="match_key", how="left").with_columns(pl.max_horizontal(["S_T", "S_CT"]).alias("mx"),
                                                         ((pl.col("S_T") + pl.col("S_CT")) == pl.col("nr_all")).alias("sum_eq_nr"))
print("nr_all distribution:", Mx.group_by("nr_all").len().sort("nr_all").to_dicts())
print("max header score distribution:", Mx.group_by("mx").len().sort("mx").to_dicts())
print("sum(header) == number of rounds:", int(Mx["sum_eq_nr"].sum()), "of", Mx.height)
print("claim matches with max header score == 13 (complete MR12):", int((Mx["mx"] == 13).sum()))
print("claim matches with n_mm_all (ledger vs winner, all rounds) distribution:",
      Mx.group_by("n_mm_all").len().sort("n_mm_all").to_dicts())

# ----------------------------------------------------------------------------- F3 death-team codes
hdr("F3. death team codes vs player_info team codes (round 9, d12-25)")
PI = rd(f24("player_info"), ["match_key", "round", "player_id_fixed", "team_code"]).unique(
    ["match_key", "round", "player_id_fixed"], keep="first")
PD = (rd(f24("player_death"), ["match_key", "round", "tick", "player_id_fixed", "player_team_code"])
      .drop_nulls(["player_id_fixed", "player_team_code"]).with_columns(pl.col("player_id_fixed").cast(pl.Int64))
      .unique(["match_key", "round", "player_id_fixed", "tick"], keep="first"))
PDJ = (PD.filter(pl.col("round") == 9).join(PI, on=["match_key", "round", "player_id_fixed"], how="inner")
       .with_columns((pl.col("player_team_code") == pl.col("team_code")).alias("same"))
       .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
PDJ = PDJ.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
PDJ = PDJ.with_columns(pl.col("match_key").is_in(claim305).alias("in_claim305"))
print(PDJ.group_by(["dg", "in_claim305"]).agg(pl.len().alias("death_rows_round9"), pl.col("same").sum().alias("codes_agree"),
                                              pl.col("match_key").n_unique().alias("matches")).sort(["dg", "in_claim305"]))
hdr("F3b. death team codes vs player_info team codes, round 13 (d12-25), all matches")
PD13 = (PD.filter(pl.col("round") == 13).join(PI, on=["match_key", "round", "player_id_fixed"], how="inner")
        .with_columns((pl.col("player_team_code") == pl.col("team_code")).alias("same")))
print("round-13 death rows matched to player_info:", PD13.height, " codes agree:", int(PD13["same"].sum()))

# ----------------------------------------------------------------------------- F4 structure
hdr("F4. round-count structure by day group (all matches)")
S = PM.join(H, on="match_key", how="left")
S = S.with_columns(((pl.col("S_T") + pl.col("S_CT")) == pl.col("nr")).alias("sum_eq_nr"),
                   (pl.col("maxr") == pl.col("nr")).alias("contig"),
                   pl.max_horizontal(["S_T", "S_CT"]).alias("mx"))
print(S.group_by(["dg", "sum_eq_nr", "contig"]).len().sort(["dg", "sum_eq_nr", "contig"]))
print("d01-07 matches: nr distribution (top):", S.filter(pl.col("dg") == "d01-07").group_by("nr").len()
      .sort("len", descending=True).head(8).to_dicts())
print("d12-25 matches: nr distribution (top):", S.filter(pl.col("dg") == "d12-25").group_by("nr").len()
      .sort("len", descending=True).head(8).to_dicts())

# ----------------------------------------------------------------------------- F5 complete matches with round-13 mismatch
hdr("F5. complete MR12 matches (max header score 13, rounds 1..n contiguous, sum == n, n <= 24), d12-25, pc-agnostic")
B = A.filter(pl.col("round") <= 24).group_by("match_key").agg(
    pl.len().alias("nr"), pl.col("round").max().alias("maxr"),
    (pl.col("mm") & (pl.col("round") == 13)).sum().alias("mm13"),
    (pl.col("mm") & (pl.col("round") == 9)).sum().alias("mm9"),
    pl.col("mm").sum().alias("n_mm"),
    (pl.col("winner_team_code") == 2).sum().alias("W2"), (pl.col("L") == 2).sum().alias("L2"),
    pl.when(pl.col("round") <= 12).then(pl.col("winner_team_code") == 2).otherwise(pl.col("winner_team_code") == 3)
      .sum().alias("Wswap"),
    pl.when(pl.col("round") <= 12).then(pl.col("L") == 2).otherwise(pl.col("L") == 3).sum().alias("Lswap"),
    pl.col("dg").first().alias("dg"))
B = B.join(H, on="match_key", how="inner").with_columns(pl.max_horizontal(["S_T", "S_CT"]).alias("mx"))
CM = B.filter((pl.col("mx") == 13) & (pl.col("maxr") == pl.col("nr")) & ((pl.col("S_T") + pl.col("S_CT")) == pl.col("nr")))
print("complete MR12 matches:", CM.height, " by dg:", CM.group_by("dg").len().sort("dg").to_dicts())
print("complete matches with n_mm == 0 (clean):", int((CM["n_mm"] == 0).sum()),
      "  with mm13 == 1:", int((CM["mm13"] == 1).sum()), "  other:", int(((CM["n_mm"] > 0) & (CM["mm13"] == 0)).sum()))
for lab, cond in [("clean", CM["n_mm"] == 0), ("mm13 only", (CM["mm13"] == 1) & (CM["n_mm"] == 1))]:
    g = CM.filter(cond)
    if g.height == 0:
        print(lab, "0 matches")
        continue
    print(f"{lab:<11} n={g.height:>4}  header==W2 {int((g['S_T'] == g['W2']).sum()):>4}  header==L2 {int((g['S_T'] == g['L2']).sum()):>4}"
          f"  header==Wswap {int((g['S_T'] == g['Wswap']).sum()):>4}  header==Lswap {int((g['S_T'] == g['Lswap']).sum()):>4}")
