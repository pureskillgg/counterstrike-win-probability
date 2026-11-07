"""verify D9-10 (g): last checks before the verdict.

  G1: is the 8-boundary format a real short match, or truncated data? Compare round count with the header's
      final score sum per format (sw8 vs sw12 vs other) and the max round index. A truncated 24-round match
      would show a large gap between final-score sum and round count.
  G2: co-occurrence claimed in D9_tests10.log: matches with a 9F, and of those how many have a 13F.
  G3: per-stratum summary of the inversion evidence (bomb-verified and death-test), build 9961 only, for the log.
ASSUMPTIONS: switch format from player_info as in verify_D9_10_c_schedule.py; death test as in D9_tests10.py.
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
ROUNDS_OF_INTEREST = [8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30]


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    missing = [c for c in cols if c not in have]
    if missing:
        raise SystemExit(f"[{name}] missing columns {missing}")
    return lf.select(cols).collect()


H = scan("header", ["key", "build_num", "t_starters_score_final", "ct_starters_score_final"]).rename(
    {"key": "match_key"})
H9 = H.filter(pl.col("build_num") == 9961)
B9 = H9["match_key"].to_list()

PI = scan("player_info", ["match_key", "round", "player_id", "team_code"]).filter(pl.col("match_key").is_in(B9))
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
a = PIu.rename({"round": "r", "team_code": "tA"})
b = PIu.with_columns((pl.col("round") - 1).alias("round")).rename({"round": "r", "team_code": "tB"})
pairs = a.join(b, on=["match_key", "r", "player_id"], how="inner").with_columns((pl.col("tA") != pl.col("tB")).alias("chg"))
bd = (pairs.group_by(["match_key", "r"]).agg(pl.len().alias("nc"), pl.col("chg").mean().alias("fch"))
      .filter(pl.col("nc") >= 3).with_columns((pl.col("fch") >= 0.5).alias("sw")))
first_sw = bd.filter(pl.col("sw")).group_by("match_key").agg(pl.col("r").min().alias("first_switch"))
fmt = (H9.select(["match_key", "t_starters_score_final", "ct_starters_score_final"])
       .join(first_sw, on="match_key", how="left")
       .with_columns(pl.when(pl.col("first_switch") == 8).then(pl.lit("sw8"))
                     .when(pl.col("first_switch") == 12).then(pl.lit("sw12"))
                     .otherwise(pl.lit("other_or_none")).alias("fmt")))
RE = scan("round_end", ["match_key", "round", "player_count", "winner_team_code"]).unique(
    subset=["match_key", "round"], keep="first")
RE = RE.filter(pl.col("match_key").is_in(B9))
nr = RE.group_by("match_key").agg(pl.len().alias("n_rounds"), pl.col("round").max().alias("max_round"))
G = fmt.join(nr, on="match_key", how="left").with_columns(
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("final_sum"))
G = G.with_columns((pl.col("final_sum") - pl.col("n_rounds")).alias("gap"))
print("== G1 round count vs header final-score sum by format (build 9961)")
print(G.group_by("fmt").agg(pl.len().alias("n"),
                            pl.col("n_rounds").mean().round(2).alias("mean_rounds"),
                            pl.col("n_rounds").max().alias("max_rounds"),
                            pl.col("max_round").max().alias("max_round_idx"),
                            pl.col("gap").mean().round(2).alias("mean_gap_final_minus_rounds"),
                            pl.col("gap").median().alias("median_gap"),
                            (pl.col("max_round") == pl.col("n_rounds")).mean().round(3).alias("contiguous_share"))
      .sort("fmt").rows())
print("   sw8 n_rounds distribution:", G.filter(pl.col("fmt") == "sw8").group_by("n_rounds").len()
      .sort("n_rounds").rows())

# ---- G2: 9F and 13F co-occurrence (death test, pc20, build 9961)
DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"]).filter(pl.col("match_key").is_in(B9))
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dead = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead")))
tc = (PIu.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("n_T"),
                                              (pl.col("team_code") == 3).sum().alias("n_CT")))
BS = scan("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("match_key").is_in(B9))
bomb = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("bomb_any")))
T = (RE.join(tc, on=["match_key", "round"], how="left").join(dead, on=["match_key", "round"], how="left")
       .join(bomb, on=["match_key", "round"], how="left")
       .with_columns(pl.col("bomb_any").fill_null(False), pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0))
       .filter((pl.col("player_count") == 20) & pl.col("n_CT").is_not_null()
               & pl.col("round").is_in(ROUNDS_OF_INTEREST)))
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
n9 = pat.filter(pl.col("pattern").str.contains(r"\b9F\b")).height
n913 = pat.filter(pl.col("pattern").str.contains(r"\b9F\b") & pl.col("pattern").str.contains(r"\b13F\b")).height
print("\n== G2 co-occurrence: matches with 9F:", n9, "| of those with 13F too:", n913, "| without 13F:", n9 - n913)
print("   matches with 13F and 9F-free:", pat.filter(pl.col("pattern").str.contains(r"\b13F\b")
                                                   & ~pl.col("pattern").str.contains(r"\b9F\b")).height)
