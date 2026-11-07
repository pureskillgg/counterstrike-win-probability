"""Alternative explanations for the D4-6 flagged rounds.

Tests (2024 days 12-25 population unless stated):
  T1. Win reason vs death counts. A round won by elimination must have 5 deaths of the losing side in the live window.
      Reason 8 = CTs_Win (T eliminated), 9 = Terrorists_Win (CT eliminated), 1 = Target_Bombed, 7 = Bomb_Defused,
      12 = Target_Saved (time out, CT wins). Compared for flagged vs unflagged rounds.
  T2. Gap regime. Is the 0 vs 429 gap a per-match tick offset? Per-match distinct gap values.
  T3. Flagged-round clustering by match: are flags confined to specific rounds within matches that otherwise agree?
  T4. Score side S derived from raw vs cleaned score columns (S vs S_raw) on the flagged set.
ASSUMPTIONS: A1 (64 ticks/s). Win-reason mapping taken from the 2024 full-round table (reason code 8 CTs_Win -> CT 3, 9 Terrorists_Win -> T 2),
             verified in verify_D4_6_probe.py output.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
POP = R.filter((pl.col("player_count") == 20) & (pl.col("dd") >= 12) & (pl.col("S") != 0)
               & (pl.col("n_victim_sides") == 1) & pl.col("victim_side").is_not_null()).with_columns(
    (pl.col("S") != pl.col("W")).alias("flag"))

print("== T1: win reason x deaths in window (CTs_Win/Terrorists_Win/Target_Bombed/Bomb_Defused/Target_Saved), POP")
t1 = POP.filter(pl.col("win_reason_code").is_in([1, 7, 8, 9, 12])).with_columns(
    pl.when(pl.col("round").is_in([13, 25, 28, 27, 9])).then(pl.col("round").cast(pl.Utf8)).otherwise(pl.lit("other")).alias("rb"))
print(t1.group_by(["rb", "flag", "win_reason_code"]).agg(
    pl.len().alias("n"),
    (pl.col("n_T_deaths_win") == 5).mean().alias("P_5T_deaths"),
    (pl.col("n_CT_deaths_win") == 5).mean().alias("P_5CT_deaths"),
    pl.col("n_T_deaths_win").mean().alias("mean_T_deaths"),
    pl.col("n_CT_deaths_win").mean().alias("mean_CT_deaths"),
    pl.col("gap").median().alias("median_gap"),
).sort(["rb", "flag", "win_reason_code"]))

print("\n== T1b: flagged rows only: W, reason, S, victim, deaths by side in window (sample of round 13)")
f13 = POP.filter((pl.col("round") == 13) & pl.col("flag"))
print(f13.group_by(["W", "S", "win_reason_code"]).len().sort("len", descending=True))
print("flagged round-13 deaths by side in window (mean):",
      f13.select(pl.col("n_T_deaths_win").mean().alias("T"), pl.col("n_CT_deaths_win").mean().alias("CT")).row(0))

print("\n== T2: gap regime per match (2024 full rounds, all days, all rounds with a last death in window)")
G = R.filter((pl.col("player_count") == 20) & pl.col("gap").is_not_null())
print("gap value shares (all rows):")
print(G.with_columns(pl.when(pl.col("gap") == 0).then(pl.lit("0"))
                     .when(pl.col("gap") == 429).then(pl.lit("429"))
                     .otherwise(pl.lit("other")).alias("gb")).group_by("gb").len().sort("len", descending=True))
per_match = G.group_by("match_key").agg(pl.col("gap").n_unique().alias("n_distinct_gap"),
                                         (pl.col("gap") == 429).mean().alias("share429"),
                                         (pl.col("gap") == 0).mean().alias("share0"),
                                         pl.len().alias("n_rounds"))
print("matches by distinct gap count:")
print(per_match.group_by("n_distinct_gap").len().sort("n_distinct_gap").head(10))
print("matches where all rounds have gap 429 or 0 (mixed-free):", per_match.filter(
    (pl.col("share429") + pl.col("share0") == 1.0)).height, "of", per_match.height)
print("matches where gap==0 for >80% rounds:", per_match.filter(pl.col("share0") > 0.8).height,
      " gap==429 for >80%:", per_match.filter(pl.col("share429") > 0.8).height)

print("\n== T3: flag by match, days 12-25, POP: matches with any flag, and which rounds are flagged in them")
fm = POP.filter(pl.col("flag")).group_by("match_key").agg(pl.col("round").sort().alias("flag_rounds"),
                                                           pl.len().alias("n_flag"))
print("matches with >=1 flagged round:", fm.height)
print("number of flagged rounds per match (distribution):")
print(fm.group_by("n_flag").len().sort("n_flag"))
print("sample of 8 matches with flagged rounds:")
print(fm.head(8))
allm = POP.group_by("match_key").agg(pl.len().alias("n_rounds_pop"), pl.col("flag").sum().alias("n_flag_all"))
print("matches in POP:", allm.height, " of which with flags:", allm.filter(pl.col("n_flag_all") > 0).height)
print("share of flagged rounds in matches where round 13 is flagged:",
      POP.filter(pl.col("match_key").is_in(POP.filter((pl.col("round") == 13) & pl.col("flag"))["match_key"].to_list()))
      .select(pl.col("flag").mean()).row(0))

print("\n== T4: S (cleaned score) vs S_raw on POP; flagged rounds 9,13,25,27,28")
print(POP.filter(pl.col("round").is_in([9, 13, 25, 27, 28])).group_by(["round"]).agg(
    pl.len().alias("n"), (pl.col("S") != pl.col("S_raw")).sum().alias("S_ne_Sraw"),
    pl.col("flag").mean().alias("flag_rate")).sort("round"))
