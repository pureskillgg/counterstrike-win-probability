"""D5 step 17: score-lead check (D5_10 part 3) restricted to matches without a side-field swap at 8->9.

D5_10 maps team A with swaps at 12->13, 24->25 and 27->28 only. Matches with a side-field swap at 8->9
(D5_14: 253 regime A, 168 regime B) break that mapping, so they are reported separately here.
Matches are split: (i) no 8->9 swap (the mapping is valid), (ii) 8->9 swap present.
ASSUMPTION: the 8->9 swap is detected with the round_end winner (side_swap classification, D5_14 rule).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_17_scorelead_no8.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet").select(
    ["match_key", "round", "fe_t_score", "fe_ct_score", "winner", "n_round_announce_match_start"]).sort(["match_key", "round"])
M = pl.read_parquet(SCR / "matches.parquet")
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
mk = R.group_by("match_key").agg(pl.col("n_round_announce_match_start").fill_null(0).sum().alias("mstart"))
M = (M.join(mk, on="match_key", how="left")
     .with_columns(regime=pl.when(pl.col("mstart") > 0).then(pl.lit("A")).otherwise(pl.lit("B")),
                   grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other"))))

# side_swap at boundary 8 -> 9 (sw true and ns false, using round_end winner 8)
N8 = R.filter(pl.col("round") == 8).with_columns(nt=pl.col("fe_t_score"), nct=pl.col("fe_ct_score"))
nx = R.filter(pl.col("round") == 9).select(["match_key", pl.col("fe_t_score").alias("t9"), pl.col("fe_ct_score").alias("ct9")])
J8 = N8.join(nx, on="match_key", how="inner").filter(pl.col("winner").is_not_null() & pl.col("t9").is_not_null())
a = pl.when(pl.col("winner") == 2).then(1).otherwise(0)
b = pl.when(pl.col("winner") == 3).then(1).otherwise(0)
J8 = J8.with_columns(
    ns=(pl.col("t9") == pl.col("fe_t_score") + a) & (pl.col("ct9") == pl.col("fe_ct_score") + b),
    sw=(pl.col("t9") == pl.col("fe_ct_score") + b) & (pl.col("ct9") == pl.col("fe_t_score") + a))
swap8 = J8.filter(pl.col("sw") & ~pl.col("ns")).select("match_key").with_columns(has_swap8=pl.lit(True))
M = M.join(swap8, on="match_key", how="left").with_columns(has_swap8=pl.col("has_swap8").fill_null(False))
print("matches with side_swap at 8->9:", M.filter(pl.col("has_swap8")).height)
print(M.group_by(["grp", "regime", "has_swap8"]).agg(pl.len().alias("matches")).sort(["grp", "regime", "has_swap8"]))

# team-A check (same mapping as D5_10)
RX = R.join(M.select(["match_key", "grp", "regime", "has_swap8"]), on="match_key", how="inner")
fe2 = R.filter(pl.col("round") == 2).select(["match_key", pl.col("fe_t_score").alias("t2"), pl.col("fe_ct_score").alias("ct2")])
RX = RX.join(fe2, on="match_key", how="left").sort(["match_key", "round"])


def team_a_is_t(col):
    return (pl.col(col) <= 12) | pl.col(col).is_between(25, 27)


RX = RX.with_columns(
    a_fe=pl.when(team_a_is_t("round")).then(pl.col("fe_t_score")).otherwise(pl.col("fe_ct_score")),
    a_won=pl.when(pl.col("round") == 1).then(
        pl.when(team_a_is_t("round")).then(pl.col("t2")).otherwise(pl.col("ct2")))
    .otherwise(pl.when(team_a_is_t("round") & (pl.col("winner") == 2)).then(1)
               .when((~team_a_is_t("round")) & (pl.col("winner") == 3)).then(1).otherwise(0)))
RX = RX.with_columns(cum_before=pl.col("a_won").fill_null(0).cum_sum().over("match_key") - pl.col("a_won").fill_null(0))
RX = RX.with_columns(diff=pl.col("a_fe") - pl.col("cum_before"))
ck = RX.filter(pl.col("a_fe").is_not_null() & (pl.col("round") >= 2))
ck = ck.filter(pl.col("winner").is_not_null() | (pl.col("regime") == "B"))
print("\nteam-A score-lead check, matches WITHOUT an 8->9 side swap:")
for reg in ["A", "B"]:
    c = ck.filter((pl.col("regime") == reg) & (~pl.col("has_swap8")))
    nm = c["match_key"].n_unique()
    bad_m = c.filter(pl.col("diff") != 0)["match_key"].n_unique()
    print(f"  regime {reg}: matches {nm}; states {c.height}; diff!=0 states {c.filter(pl.col('diff') != 0).height}; "
          f"matches with any diff!=0 {bad_m}")
    print("    diff!=0 by round (first 25 labels):")
    print(c.filter(pl.col("diff") != 0).group_by("round").agg(pl.len().alias("states")).sort("round").head(25))
print("\nteam-A score-lead check, matches WITH an 8->9 side swap (mapping not valid; reported for completeness):")
for reg in ["A", "B"]:
    c = ck.filter((pl.col("regime") == reg) & pl.col("has_swap8"))
    print(f"  regime {reg}: matches {c['match_key'].n_unique()}; states {c.height}; diff!=0 states {c.filter(pl.col('diff') != 0).height}")
