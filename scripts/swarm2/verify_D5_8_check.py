"""Verifier for D5-8: does round 13 (first post-halftime round) carry an inverted winner label in regime A?

Reads the per-round table written by verify_D5_8_build.py (scratchpad, not the repo).
Three independent sources per round, all side-coded (2 = T, 3 = CT):
  winner  = round_end winner_team_code
  reason  = side implied by round_end win_reason_code (ASSUMPTION: standard CS2 reason-code mapping)
  bomb    = bomb outcome (exploded -> 2, defused -> 3); only rounds with a bomb outcome
  step    = freeze-end score step r -> r+1 (+1 on t field -> 2, +1 on ct field -> 3); needs a clean step
Also a header test: header final team scores vs round totals derived from each source.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_8_check.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_8")
R = pl.read_parquet(SCR / "rounds2024.parquet").rename({"winner_team_code": "winner"})
H = pl.read_parquet(SCR / "header2024.parquet")
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]

# ---- 1. regime by day (matches) ----
M = R.select(["match_key", "day", "regA"]).unique("match_key")
print("1. regime by day group (matches; regime A = round_announce_match_start present):")
print(M.with_columns(grp=pl.when(pl.col("day").is_in(LATE)).then(pl.lit("late 12-25"))
                     .when(pl.col("day").is_in(EARLY)).then(pl.lit("early 01-07")).otherwise(pl.lit("other")))
      .group_by(["grp", "regA"]).len().sort(["grp", "regA"]))

# ---- 2. per-source agreement at rounds 12, 13, 14, 25, 28, and at all other rounds, by regime ----
R = R.with_columns(
    agree_bomb=pl.when(pl.col("bomb_side").is_null()).then(None).otherwise(pl.col("bomb_side") == pl.col("winner")),
    agree_reason=pl.when(pl.col("reason_side").is_null()).then(None).otherwise(pl.col("reason_side") == pl.col("winner")),
    agree_step=pl.when(pl.col("step_side").is_null()).then(None).otherwise(pl.col("step_side") == pl.col("winner")),
)
print("\n2. agreement of round_end winner with each source, by round and regime (non-null comparisons only):")
print("   columns: n_rows | reason: n, agree% | step: n, agree% | bomb: n, agree%")
for rnd in [12, 13, 14, 24, 25, 26, 27, 28]:
    for rg in [True, False]:
        sub = R.filter((pl.col("round") == rnd) & (pl.col("regA") == rg) & pl.col("day").is_in(LATE + EARLY))
        n = sub.height
        rr = sub.filter(pl.col("agree_reason").is_not_null())
        ss = sub.filter(pl.col("agree_step").is_not_null())
        bb = sub.filter(pl.col("agree_bomb").is_not_null())
        pct = lambda d, c: (f"{100.0 * d[c].mean():.1f}%" if d.height else "n/a")
        print(f"   r={rnd:>2} regA={str(rg):5} rows={n:>5} | reason n={rr.height:>5} {pct(rr,'agree_reason'):>7}"
              f" | step n={ss.height:>5} {pct(ss,'agree_step'):>7} | bomb n={bb.height:>4} {pct(bb,'agree_bomb'):>7}")

# all other rounds (late and early), regime A vs B
oth = R.filter(~pl.col("round").is_in([12, 13, 14, 24, 25, 26, 27, 28]) & pl.col("day").is_in(LATE + EARLY))
for rg in [True, False]:
    sub = oth.filter(pl.col("regA") == rg)
    rr = sub.filter(pl.col("agree_reason").is_not_null())
    ss = sub.filter(pl.col("agree_step").is_not_null())
    bb = sub.filter(pl.col("agree_bomb").is_not_null())
    print(f"   other rounds regA={str(rg):5} rows={sub.height:>6} | reason n={rr.height:>6} "
          f"{100.0 * rr['agree_reason'].mean():.2f}% | step n={ss.height:>6} {100.0 * ss['agree_step'].mean():.2f}%"
          f" | bomb n={bb.height:>5} {100.0 * bb['agree_bomb'].mean():.2f}%")

# ---- 3. three-way cross-tab at round 13 (the decisive test) ----
print("\n3. round 13, three-way: bomb vs winner, reason vs winner, step vs winner (late days only):")
r13 = R.filter((pl.col("round") == 13) & pl.col("day").is_in(LATE))
x = r13.with_columns(
    bomb_s=pl.when(pl.col("agree_bomb").is_null()).then(pl.lit("none")).when(pl.col("agree_bomb")).then(pl.lit("agree"))
    .otherwise(pl.lit("DISAGREE")),
    step_s=pl.when(pl.col("agree_step").is_null()).then(pl.lit("none")).when(pl.col("agree_step")).then(pl.lit("agree"))
    .otherwise(pl.lit("DISAGREE")),
    reason_s=pl.when(pl.col("agree_reason").is_null()).then(pl.lit("none")).when(pl.col("agree_reason")).then(pl.lit("agree"))
    .otherwise(pl.lit("DISAGREE")),
)
print(x.group_by(["regA", "bomb_s", "reason_s", "step_s"]).len().sort(["regA", "bomb_s", "reason_s", "step_s"]))

# ---- 4. round_end internal consistency: winner_team_code vs reason-implied side, by regime and round bin ----
R = R.with_columns(rbin=pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                   .when(pl.col("round") == 12).then(pl.lit("r12"))
                   .when(pl.col("round").is_in([25, 28])).then(pl.lit("r25/28"))
                   .otherwise(pl.lit("other")))
print("\n4. round_end internal consistency: rows where winner_team_code != reason-implied side (late days):")
L = R.filter(pl.col("day").is_in(LATE))
print(L.filter(pl.col("reason_side").is_not_null()).group_by(["regA", "rbin"])
      .agg(pl.len().alias("rows"), (pl.col("reason_side") != pl.col("winner")).sum().alias("winner_vs_reason_mismatch"))
      .sort(["regA", "rbin"]))
print("   early days (01-07), same check:")
E = R.filter(pl.col("day").is_in(EARLY))
print(E.filter(pl.col("reason_side").is_not_null()).group_by(["regA", "rbin"])
      .agg(pl.len().alias("rows"), (pl.col("reason_side") != pl.col("winner")).sum().alias("winner_vs_reason_mismatch"))
      .sort(["regA", "rbin"]))

# ---- 5. header test: do round totals from each source reproduce the header final scores? ----
# side of the starting-T team in each round (ASSUMPTION: MR12, sides swap at 12->13; OT blocks of 6, T for first 3)
def side_x(r):
    return pl.when((pl.col(r) <= 12) | ((pl.col(r) >= 25) & (((pl.col(r) - 25) % 6) < 3))).then(pl.lit(2)).otherwise(pl.lit(3))


P = R.with_columns(sideX=side_x("round"),
                   src_step=pl.coalesce(["step_side", "reason_side"]),
                   src_reason=pl.col("reason_side"),
                   src_winner=pl.col("winner"))
P = P.with_columns(
    xw_winner=(pl.col("sideX") == pl.col("src_winner")).cast(pl.Int64),
    xw_reason=(pl.col("sideX") == pl.col("src_reason")).cast(pl.Int64),
    xw_step=(pl.col("sideX") == pl.col("src_step")).cast(pl.Int64),
)
T = P.group_by("match_key").agg(
    pl.len().alias("n_rounds"),
    pl.col("xw_winner").sum().alias("X_wins_winner"),
    pl.col("xw_reason").sum().alias("X_wins_reason"),
    pl.col("xw_step").sum().alias("X_wins_step"),
    pl.col("regA").first(), pl.col("day").first(),
    pl.col("t_starters_score_final").first().alias("hdr_t"),
    pl.col("ct_starters_score_final").first().alias("hdr_ct"),
    pl.col("round").max().alias("max_round"),
)
T = T.with_columns(complete=(pl.col("hdr_t") + pl.col("hdr_ct") == pl.col("n_rounds")))
print("\n5. header test (matches with header final sum == number of round_end rows, i.e. complete round set):")
print(T.group_by("regA").agg(pl.len().alias("matches"), pl.col("complete").sum().alias("complete")).sort("regA"))
C = T.filter(pl.col("complete"))
for src in ["X_wins_winner", "X_wins_reason", "X_wins_step"]:
    ok_x = C.filter(pl.col(src) == pl.col("hdr_t"))
    ok_swap = C.filter((C["n_rounds"] - C[src]) == pl.col("hdr_t"))
    print(f"   {src:15s}: exact match (starting-T team = header t) by regime:",
          C.with_columns(ok=(pl.col(src) == pl.col("hdr_t"))).group_by("regA").agg(
              pl.len().alias("n"), pl.col("ok").sum().alias("exact"),
              (100.0 * pl.col("ok").mean()).round(2).alias("pct")).sort("regA").to_dicts())
