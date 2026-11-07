"""verify D5-6, fourth pass.

  W1  header test in regime A ctrl_late (round 13 present): which round-13 rows must be flipped?
      Variants: none, all round-13 rows, only no-plant round-13 rows (the claim's no-bomb scope),
      only planted round-13 rows. If planted round-13 rows are also inverted, the claim's no-bomb-only
      fix leaves them wrong.
  W2  header test extended to overtime matches (max round 25..30) with the swap pattern
      A on T in rounds 1-12 and 25-27, A on CT in rounds 13-24 and 28-30 (ASSUMPTION A6: side swaps at
      the start of each overtime half, as the label pattern 13/25/28 suggests). Regime B is the control.
      Variants: none, flip {13, 25, 28}, flip {25, 28}.
Input: scratchpad D5_6_indep/rounds.parquet (written by verify_D5_6_indep_check.py) and the header.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_indep_check4.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_6_indep")

R = pl.read_parquet(SCR / "rounds.parquet")
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))[0]
H = (pl.read_parquet(hf, columns=["key", "t_starters_score_final", "ct_starters_score_final"])
     .rename({"key": "match_key"}))
MM = R.group_by("match_key").agg(
    pl.col("round").max().alias("max_round"), pl.col("round").n_unique().alias("n_rounds"),
    pl.col("regime").first().alias("regime"), pl.col("grp").first().alias("grp")).join(H, on="match_key", how="left")
MM = MM.with_columns(final_sum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))

# ---------------- W1 ----------------
print("=" * 100)
print("W1. Regime A ctrl_late, header test with round-13 flip scopes (eligible: <=24 rounds, contiguous, header sum ok)")
print("=" * 100)
elig = MM.filter((pl.col("regime") == "A") & (pl.col("grp") == "ctrl_late") & (pl.col("max_round") <= 24)
                 & (pl.col("n_rounds") == pl.col("max_round")) & (pl.col("final_sum") == pl.col("max_round")))
X = R.filter(pl.col("match_key").is_in(elig["match_key"]) & pl.col("rw").is_not_null()).with_columns(
    a_as=pl.when(pl.col("round") <= 12).then((pl.col("rw") == 2).cast(pl.Int32)).otherwise((pl.col("rw") == 3).cast(pl.Int32)))
tot = X.group_by("match_key").agg(pl.col("a_as").sum().alias("tot"))
r13 = X.filter(pl.col("round") == 13).select(["match_key", pl.col("a_as").alias("a13"), "has_plant"])
W = (tot.join(r13, on="match_key", how="inner").join(H, on="match_key", how="left"))
W = W.with_columns(
    s_none=(pl.col("tot") == pl.col("t_starters_score_final")),
    s_all13=(pl.col("tot") + 1 - 2 * pl.col("a13") == pl.col("t_starters_score_final")),
    s_nobomb13=pl.when(pl.col("has_plant")).then(pl.col("tot") == pl.col("t_starters_score_final"))
    .otherwise(pl.col("tot") + 1 - 2 * pl.col("a13") == pl.col("t_starters_score_final")),
    s_bomb13=pl.when(~pl.col("has_plant")).then(pl.col("tot") == pl.col("t_starters_score_final"))
    .otherwise(pl.col("tot") + 1 - 2 * pl.col("a13") == pl.col("t_starters_score_final")))
print("matches with a round 13 (regime A ctrl_late eligible):", W.height,
      " planted round 13:", int(W["has_plant"].sum()), " no-plant round 13:", int((~W["has_plant"]).sum()))
print(W.select([pl.col("s_none").mean().round(3).alias("consistent_as_is"),
                pl.col("s_all13").mean().round(3).alias("consistent_flip_all13"),
                pl.col("s_nobomb13").mean().round(3).alias("consistent_flip_noplant13_only"),
                pl.col("s_bomb13").mean().round(3).alias("consistent_flip_planted13_only")]))
print("split by planted status of round 13 (consistent rate, n):")
print(W.group_by("has_plant").agg(pl.len().alias("n"), pl.col("s_none").mean().round(3).alias("as_is"),
                                  pl.col("s_all13").mean().round(3).alias("flip_all13")).sort("has_plant"))

# ---------------- W2 ----------------
print("\n" + "=" * 100)
print("W2. Overtime matches (max round 25..30): header test with swap pattern (ASSUMPTION A6), regime A vs B")
print("=" * 100)
ot = MM.filter((pl.col("max_round") > 24) & (pl.col("max_round") <= 30) & (pl.col("n_rounds") == pl.col("max_round"))
               & (pl.col("final_sum") == pl.col("max_round")))
print("eligible OT matches by regime/grp:")
print(ot.group_by(["regime", "grp"]).agg(pl.len().alias("matches")).sort(["regime", "grp"]))
XO = R.filter(pl.col("match_key").is_in(ot["match_key"]) & pl.col("rw").is_not_null())
# A is T in rounds 1-12 and 25-27; A is CT in 13-24 and 28-30
XO = XO.with_columns(
    a_ot=pl.when(pl.col("round").is_in(list(range(1, 13)) + [25, 26, 27])).then((pl.col("rw") == 2).cast(pl.Int32))
    .otherwise((pl.col("rw") == 3).cast(pl.Int32)))
XO = XO.with_columns(a_fl=pl.when(pl.col("round").is_in([13, 25, 28])).then(1 - pl.col("a_ot")).otherwise(pl.col("a_ot")),
                     a_fl2=pl.when(pl.col("round").is_in([25, 28])).then(1 - pl.col("a_ot")).otherwise(pl.col("a_ot")))
TO = XO.group_by("match_key").agg(pl.col("a_ot").sum().alias("tot"), pl.col("a_fl").sum().alias("tot_fl"),
                                  pl.col("a_fl2").sum().alias("tot_fl2"))
TO = TO.join(ot.select(["match_key", "regime", "grp", "t_starters_score_final"]), on="match_key", how="left")
TO = TO.with_columns(
    ok_none=(pl.col("tot") == pl.col("t_starters_score_final")),
    ok_fl_13_25_28=(pl.col("tot_fl") == pl.col("t_starters_score_final")),
    ok_fl_25_28=(pl.col("tot_fl2") == pl.col("t_starters_score_final")))
print(TO.group_by(["regime", "grp"]).agg(
    pl.len().alias("matches"),
    pl.col("ok_none").mean().round(3).alias("as_is"),
    pl.col("ok_fl_25_28").mean().round(3).alias("flip_25_28"),
    pl.col("ok_fl_13_25_28").mean().round(3).alias("flip_13_25_28")).sort(["regime", "grp"]))

print("\nDONE")
