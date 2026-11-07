"""Verifier exploration for D5-8: header score sums vs round counts, and raw freeze-end sequences.

Reads the per-round table from verify_D5_8_build.py (scratchpad). No repo writes.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_8_explore.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(40)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_8")
R = pl.read_parquet(SCR / "rounds2024.parquet").rename({"winner_team_code": "winner"})

# ---- A. header sum minus round count, by regime ----
M = R.group_by("match_key").agg(
    pl.len().alias("n_rounds"), pl.col("round").max().alias("max_round"), pl.col("round").n_unique().alias("n_uniq"),
    pl.col("regA").first(), pl.col("day").first(),
    pl.col("t_starters_score_final").first().alias("hdr_t"), pl.col("ct_starters_score_final").first().alias("hdr_ct"),
    pl.col("map_name").first(),
).with_columns(diff=pl.col("hdr_t") + pl.col("hdr_ct") - pl.col("n_rounds"))
print("A. header(t+ct) minus round_end row count, by regime (distribution of diff):")
print(M.group_by(["regA", "diff"]).len().sort(["regA", "len"], descending=[False, True]).head(40))
print("   max_round vs n_rounds equal (regime, count):")
print(M.with_columns(eq=pl.col("max_round") == pl.col("n_rounds")).group_by(["regA", "eq"]).len().sort(["regA", "eq"]))

# ---- B. raw freeze-end sequences for sample matches ----
def show(mk, label):
    S = (R.filter(pl.col("match_key") == mk).sort("round")
         .select(["round", "winner", "reason_side", "bomb_side", "t_fe", "ct_fe", "t_fe_next", "ct_fe_next", "step_side"]))
    print(f"\nB. {label}: {mk.split('/')[-1][:8]}  hdr t/ct final = {M.filter(pl.col('match_key') == mk)['hdr_t'][0]}/"
          f"{M.filter(pl.col('match_key') == mk)['hdr_ct'][0]}  regA={M.filter(pl.col('match_key') == mk)['regA'][0]}")
    print(S)

# pick examples: regime A with bomb-disagree at 13, regime A clean at 13 (step agrees), regime B late, early B
ex_aff = R.filter((pl.col("regA")) & (pl.col("round") == 13) & pl.col("bomb_side").is_not_null()
                  & (pl.col("bomb_side") != pl.col("winner"))).select("match_key").head(1)["match_key"][0]
ex_clean13 = R.filter((pl.col("regA")) & (pl.col("round") == 13) & (pl.col("step_side") == pl.col("winner"))
                      & pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)])).select("match_key").head(1)["match_key"][0]
ex_B_late = R.filter((~pl.col("regA")) & pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)])
                     & (pl.col("round") == 13)).select("match_key").head(1)["match_key"][0]
show(ex_aff, "regime A, bomb-disagrees at round 13")
show(ex_clean13, "regime A, step agrees with winner at round 13")
show(ex_B_late, "regime B late, round 13")

# ---- C. regime A round 13: step agreement vs header-complete matches ----
r13 = R.filter((pl.col("round") == 13) & pl.col("regA") & pl.col("step_side").is_not_null())
r13 = r13.with_columns(step_agrees=(pl.col("step_side") == pl.col("winner")))
print("\nC. regime A round 13 (all days) step agreement counts:", r13.group_by("step_agrees").len().sort("step_agrees").to_dicts())
print("   round 13 step agreement by day group:")
print(r13.with_columns(g=pl.when(pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)])).then(pl.lit("late"))
                       .otherwise(pl.lit("early"))).group_by(["g", "step_agrees"]).len().sort(["g", "step_agrees"]))
