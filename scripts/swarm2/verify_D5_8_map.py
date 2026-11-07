"""Verifier for D5-8, map pass: where in the round index does the round_end winner disagree with independent evidence?

For every round r (all rounds, both regimes, late and early days):
  bomb check: bomb outcome side vs round_end winner (bomb outcome from bomb_state; ground truth by game rules)
  step check: naive freeze-end step (fe r -> fe r+1, +1 on t -> 2, +1 on ct -> 3) vs round_end winner, only when a clean step exists
Also: missing-round pattern (rounds in freeze-end data but absent from round_end), by regime.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_8_map.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_8")
R = pl.read_parquet(SCR / "rounds2024.parquet").rename({"winner_team_code": "winner"})
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]

R = R.with_columns(grp=pl.when(pl.col("day").is_in(LATE)).then(pl.lit("late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
R = R.filter(pl.col("grp") != "other")
R = R.with_columns(
    bomb_dis=pl.when(pl.col("bomb_side").is_null()).then(None).otherwise((pl.col("bomb_side") != pl.col("winner")).cast(pl.Int8)),
    step_dis=pl.when(pl.col("step_side").is_null()).then(None).otherwise((pl.col("step_side") != pl.col("winner")).cast(pl.Int8)),
)

print("MAP-1. per-round disagreement, regime A (late) and regime B (late + early). bomb: n / disagree; step: n / disagree")
rows = []
for rg in [True, False]:
    for grp in ["late", "early"] if not rg else ["late"]:
        sub = R.filter((pl.col("regA") == rg) & (pl.col("grp") == grp))
        g = sub.group_by("round").agg(
            pl.col("bomb_dis").count().alias("bomb_n"), pl.col("bomb_dis").sum().alias("bomb_dis"),
            pl.col("step_dis").count().alias("step_n"), pl.col("step_dis").sum().alias("step_dis"))
        g = g.with_columns(regA=pl.lit(rg), grp=pl.lit(grp)).filter(pl.col("round") <= 30)
        rows.append(g)
G = pl.concat(rows).sort(["regA", "grp", "round"])
# compact view: only rounds with any disagreement, plus round 1-30 totals in one table per regime
for rg in [True, False]:
    for grp in (["late"] if rg else ["late", "early"]):
        v = G.filter((pl.col("regA") == rg) & (pl.col("grp") == grp))
        print(f"\n  regime {'A' if rg else 'B'} / {grp}:")
        print(v.select(["round", "bomb_n", "bomb_dis", "step_n", "step_dis"]).with_columns(
            step_pct=(100.0 * pl.col("step_dis") / pl.col("step_n")).round(2)).to_pandas().to_string(index=False)
              if False else v.select(["round", "bomb_n", "bomb_dis", "step_n", "step_dis"]))

print("\nMAP-2. rounds where any disagreement occurs (step or bomb), regime A late, with counts:")
A = G.filter((pl.col("regA") == True) & (pl.col("grp") == "late")).filter((pl.col("step_dis") > 0) | (pl.col("bomb_dis") > 0))
print(A.select(["round", "bomb_n", "bomb_dis", "step_n", "step_dis"]))
print("   regime B (late+early) rounds with any disagreement:")
Bv = G.filter(pl.col("regA") == False).group_by("round").agg(pl.col("bomb_n").sum(), pl.col("bomb_dis").sum(),
                                                             pl.col("step_n").sum(), pl.col("step_dis").sum())
print(Bv.filter((pl.col("step_dis") > 0) | (pl.col("bomb_dis") > 0)))

# ---- missing rounds: freeze-end rounds vs round_end rounds, per match ----
M = R.group_by("match_key").agg(pl.col("round").n_unique().alias("n_end"), pl.col("round").max().alias("max_end"),
                                pl.col("regA").first(), pl.col("grp").first())
print("\nMAP-3. round_end coverage per match, by regime (late and early):")
print(M.group_by(["regA", "grp"]).agg(pl.len().alias("matches"), pl.col("n_end").mean().round(2).alias("mean_rounds_in_end"),
                                       (pl.col("max_end") > pl.col("n_end")).sum().alias("gap_matches")).sort(["regA", "grp"]))
