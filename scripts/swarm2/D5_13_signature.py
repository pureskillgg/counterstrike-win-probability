"""D5 step 13: per-match error signature across the first post-swap labels (9, 13, 25, 28).

Truth = freeze-end score step r -> r+1 (valid for r = 9, 13, 25, 28: no side swap between r and r+1).
A label is 'wrong' when the valid step is a clean +1 for one team and disagrees with the round_end winner.
Matches are split by regime (A: round_announce_match_start present; B: absent) and by group.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_13_signature.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
mk = R.group_by("match_key").agg(pl.col("n_round_announce_match_start").fill_null(0).sum().alias("mstart"))
M = (M.join(mk, on="match_key", how="left")
     .with_columns(regime=pl.when(pl.col("mstart") > 0).then(pl.lit("A")).otherwise(pl.lit("B")),
                   grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other"))))

fe = R.select(["match_key", "round", "fe_t_score", "fe_ct_score", "winner"]).sort(["match_key", "round"])
step = (fe.with_columns(nt=pl.col("fe_t_score").shift(-1).over("match_key"),
                        nct=pl.col("fe_ct_score").shift(-1).over("match_key"),
                        nr=pl.col("round").shift(-1).over("match_key"))
        .filter((pl.col("nr") == pl.col("round") + 1) & pl.col("round").is_in([9, 13, 25, 28])
                & pl.col("fe_t_score").is_not_null() & pl.col("nt").is_not_null() & pl.col("winner").is_not_null())
        .with_columns(dt=pl.col("nt") - pl.col("fe_t_score"), dct=pl.col("nct") - pl.col("fe_ct_score"))
        .with_columns(ch=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
                      .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)))
        .filter(pl.col("ch") != 0)
        .with_columns(wrong=(pl.col("ch") != pl.col("winner")).cast(pl.Int8))
        .select(["match_key", "round", "wrong"]))
W = step.pivot(on="round", index="match_key", values="wrong", aggregate_function="first")
W = W.rename({c: f"w{c}" for c in W.columns if c != "match_key"})
for c in ["w9", "w13", "w25", "w28"]:
    if c not in W.columns:
        W = W.with_columns(pl.lit(None, dtype=pl.Int8).alias(c))
S = M.select(["match_key", "grp", "regime", "day"]).join(W, on="match_key", how="left").filter(
    pl.col("grp").is_in(["ctrl_late", "aff_r13", "early"]))
print("usable label coverage (match counts with a valid step):")
print(S.group_by(["grp", "regime"]).agg(pl.len().alias("matches"),
                                       pl.col("w9").is_not_null().sum().alias("has9"),
                                       pl.col("w13").is_not_null().sum().alias("has13"),
                                       pl.col("w25").is_not_null().sum().alias("has25"),
                                       pl.col("w28").is_not_null().sum().alias("has28")).sort(["grp", "regime"]))

S = S.with_columns(
    sig=pl.concat_str([
        pl.when(pl.col("w9") == 1).then(pl.lit("9")).otherwise(pl.lit("")),
        pl.when(pl.col("w13") == 1).then(pl.lit("13")).otherwise(pl.lit("")),
        pl.when(pl.col("w25") == 1).then(pl.lit("25")).otherwise(pl.lit("")),
        pl.when(pl.col("w28") == 1).then(pl.lit("28")).otherwise(pl.lit("")),
    ], separator=",").str.strip_chars(","))
S = S.with_columns(sig=pl.when(pl.col("sig") == "").then(pl.lit("none")).otherwise(pl.col("sig")))
S = S.with_columns(pattern_complete=(pl.col("w9").is_not_null() & pl.col("w13").is_not_null()))
print("\nwrong-label signature by group and regime (matches with at least one usable step among 9/13/25/28):")
print(S.filter(pl.col("w9").is_not_null() | pl.col("w13").is_not_null() | pl.col("w25").is_not_null() | pl.col("w28").is_not_null())
      .group_by(["grp", "regime", "sig"]).agg(pl.len().alias("matches")).sort(["grp", "regime", "matches"], descending=[False, False, True]))

print("\nlate regime-A matches with 9 and 13 both usable: joint pattern")
J = S.filter((pl.col("grp") == "ctrl_late") & (pl.col("regime") == "A") & pl.col("w9").is_not_null() & pl.col("w13").is_not_null())
print(J.group_by(["w9", "w13"]).agg(pl.len().alias("matches")).sort(["w9", "w13"]))
print("\nlate regime-A aff matches (13 usable):", S.filter((pl.col("grp") == "aff_r13") & pl.col("w13").is_not_null()).height,
      " w13 == 1:", S.filter((pl.col("grp") == "aff_r13") & (pl.col("w13") == 1)).height)
print("early regime-B matches with any usable wrong label:",
      S.filter((pl.col("grp") == "early") & ((pl.col("w9") == 1) | (pl.col("w13") == 1) | (pl.col("w25") == 1) | (pl.col("w28") == 1))).height)
