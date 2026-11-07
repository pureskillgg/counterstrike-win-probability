"""D9 probe 10: per-match flip pattern on build 9961 (pc20), non-bomb rounds 8-14 and 24-30.

Per round: R = recorded consistent, F = only flipped consistent, N = neither, '-' = not scored (bomb or no roster).
Counts the distinct per-match patterns, to see whether flips are a per-match mode or a per-round rule.
ASSUMPTION: same consistency rule as D9_tests4.py.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9")
R = pl.read_parquet(SCR / "D9_round.parquet")
M = pl.read_parquet(SCR / "D9_match.parquet")


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, want):
    fd = BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in want if c in have]).collect()


DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"])
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dead = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead")))
BS = scan("bomb_state", ["match_key", "round", "event_type"])
bomb_rounds = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("bomb_any")))
T = (R.join(M.select(["match_key", "build_num"]), on="match_key", how="left")
       .join(dead, on=["match_key", "round"], how="left")
       .join(bomb_rounds, on=["match_key", "round"], how="left")
       .with_columns(pl.col("bomb_any").fill_null(False), pl.col("T_dead").fill_null(0),
                     pl.col("CT_dead").fill_null(0))
       .filter((pl.col("build_num") == 9961) & (pl.col("player_count") == 20) & pl.col("n_CT").is_not_null()
               & (pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30]))))
T = T.with_columns(
    pl.when(pl.col("bomb_any")).then(pl.lit("-"))
      .when(pl.col("winner_team_code") == 2).then(
          pl.when(pl.col("CT_dead") >= pl.col("n_CT")).then(pl.lit("R"))
          .when(pl.col("T_dead") >= pl.col("n_T")).then(pl.lit("F")).otherwise(pl.lit("N")))
      .otherwise(
          pl.when(pl.col("T_dead") >= pl.col("n_T")).then(pl.lit("R"))
          .when(pl.col("CT_dead") >= pl.col("n_CT")).then(pl.lit("F")).otherwise(pl.lit("N")))
      .alias("c"))
pat = (T.sort("round").group_by("match_key", maintain_order=True)
       .agg(pl.col("round").cast(pl.String).alias("rs"), pl.col("c").alias("cs")))
pat = pat.with_columns(pl.struct(["rs", "cs"]).map_elements(
    lambda s: " ".join(f"{r}{c}" for r, c in zip(s["rs"], s["cs"])), return_dtype=pl.String).alias("pattern"))
print("matches with any scored round here:", pat.height)
top = pat.group_by("pattern").len().sort("len", descending=True).head(25)
print(top)
pat2 = pat.with_columns(
    pl.col("pattern").str.count_matches(r"\dF").alias("n_flip_rounds"))
print("\nmatches by number of flipped rounds (F):", pat2.group_by("n_flip_rounds").len().sort("n_flip_rounds").rows())

print("\n== matches containing a flipped round (F), by round label")
for rr in [9, 13, 25, 28]:
    n_f = pat.filter(pl.col("pattern").str.contains(rf"\b{rr}F\b")).height
    n_s = pat.filter(pl.col("pattern").str.contains(rf"\b{rr}[RF]\b")).height
    print(f"   round {rr}: flipped in {n_f} matches (scored in {n_s})")
n_13_only = pat.filter(pl.col("pattern").str.contains(r"\b13F\b") & ~pl.col("pattern").str.contains(r"\dF\b.*\b13F") ).height
n_13_and_9 = pat.filter(pl.col("pattern").str.contains(r"\b9F\b") & pl.col("pattern").str.contains(r"\b13F\b")).height
n_9_no_13 = pat.filter(pl.col("pattern").str.contains(r"\b9F\b") & ~pl.col("pattern").str.contains(r"\b13F\b")).height
print(f"   matches with 9F and 13F together: {n_13_and_9}; with 9F but not 13F: {n_9_no_13}")
print("   matches where a flipped round is followed by an R round (single-round inversions):",
      pat.filter(pl.col("pattern").str.contains(r"\dF \d+R")).height)
