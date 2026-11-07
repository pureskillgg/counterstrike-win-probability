"""verify D7-5 step 4: orientation-free halftime swap detection from player identities.

For each match, a player's side (player_team_code on player_death and bomb_state rows, keyed by player_id) is
read per phase: rounds 1-12 (phase pre) and rounds 13-24 (phase post). A halftime swap shows up as players
whose modal side flips between the two phases. This uses neither the freeze_end scores nor round_end.
ASSUMPTION: player_id is a stable identity within one match; team codes on death/bomb rows are side codes
(checked in step 2 through planter code = 2 on planted rounds).
Also: for each round-9 and round-27 disagreement between winner_team_code and the DP ledger, which side the
elimination/bomb evidence supports.
Reads the raw tome channels (projection only) and the scratchpad R table.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_5")


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


PD = pl.scan_parquet(files(BASE / f"basic_player_death{SUF}")).select(
    ["match_key", "round", "player_id", "player_team_code"]).collect()
BS = pl.scan_parquet(files(BASE / f"basic_bomb_state{SUF}")).select(
    ["match_key", "round", "player_id", "player_team_code", "event_type"]).filter(
    pl.col("event_type") == "bomb_planted").drop("event_type").collect()
PT = pl.concat([PD, BS]).filter(pl.col("player_team_code").is_not_null() & pl.col("player_id").is_not_null()
                                 & pl.col("round").is_not_null()).unique(["match_key", "round", "player_id", "player_team_code"])
PT = PT.with_columns(pl.when(pl.col("round") <= 12).then(pl.lit("pre")).when(pl.col("round") <= 24)
                     .then(pl.lit("post")).otherwise(pl.lit("ot")).alias("phase"))
print("player-side observations:", PT.height)

# modal side per (match, player, phase)
MODE = (PT.group_by(["match_key", "player_id", "phase", "player_team_code"]).len()
        .sort(["match_key", "player_id", "phase", "len"], descending=[False, False, False, True])
        .unique(["match_key", "player_id", "phase"], keep="first")
        .select(["match_key", "player_id", "phase", "player_team_code"]))
pre = MODE.filter(pl.col("phase") == "pre").select(["match_key", "player_id", pl.col("player_team_code").alias("pre_side")])
post = MODE.filter(pl.col("phase") == "post").select(["match_key", "player_id", pl.col("player_team_code").alias("post_side")])
J = pre.join(post, on=["match_key", "player_id"], how="inner")
F = (J.group_by("match_key").agg(
    pl.len().alias("common_players"),
    (pl.col("pre_side") != pl.col("post_side")).sum().alias("flips"),
).with_columns((pl.col("flips") / pl.col("common_players")).alias("flip_frac")))
print("matches with any common pre/post players:", F.height)
print("flip_frac distribution (all matches):", F["flip_frac"].describe().rows())

R = pl.read_parquet(str(SCR / "R_all_rounds.parquet")).with_columns(
    pl.col("build_num").cast(pl.Utf8).alias("build"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("path_day"))
M = (R.filter(pl.col("player_count") == 20).group_by("match_key")
     .agg(pl.col("build").first(), pl.col("path_day").first(), pl.col("mdt").first()))
# structural score-signature at pair 12->13 per match
S12 = (R.filter((pl.col("round") == 12) & (pl.col("player_count") == 20))
       .select(["match_key", "noswap_valid", "swap_valid"]))
S12 = S12.with_columns(
    pl.when(pl.col("noswap_valid") & pl.col("swap_valid")).then(pl.lit("both"))
      .when(pl.col("noswap_valid")).then(pl.lit("noswap_only"))
      .when(pl.col("swap_valid")).then(pl.lit("swap_only"))
      .otherwise(pl.lit("neither")).alias("pair12_type"))
X = M.join(F, on="match_key", how="left").join(S12.select(["match_key", "pair12_type"]), on="match_key", how="left")
X = X.with_columns(
    pl.col("flip_frac").is_not_null().alias("has_player_phase"),
    pl.when(pl.col("flip_frac").is_null()).then(pl.lit("no_players"))
      .when(pl.col("flip_frac") >= 0.5).then(pl.lit("players_swap"))
      .otherwise(pl.lit("players_no_swap")).alias("player_class"))
print("\n== (1) player-based halftime class x build x score pair12 type (5v5 matches) ==")
print(X.group_by(["build", "pair12_type", "player_class"]).len().sort(["build", "pair12_type", "player_class"]).rows())
print("\n== (2) flip_frac quantiles by build and pair12 type ==")
print(X.filter(pl.col("flip_frac").is_not_null()).group_by(["build", "pair12_type"]).agg(
    pl.len().alias("n"),
    pl.col("flip_frac").median().alias("median_flip"),
    pl.col("flip_frac").quantile(0.1).alias("p10"),
    pl.col("flip_frac").quantile(0.9).alias("p90"),
).sort(["build", "pair12_type"]))

print("\n== (3) noswap_only matches (9961): player evidence and death/bomb coverage ==")
NS = X.filter((pl.col("build") == "9961") & (pl.col("pair12_type") == "noswap_only"))
print("noswap_only 9961 matches:", NS.height, "| with player phase data:", NS.filter(pl.col("has_player_phase")).height,
      "| player_class:", NS.group_by("player_class").len().sort("player_class").rows())
deaths_ns = PD.join(NS.select("match_key"), on="match_key", how="inner").group_by("match_key").len()
print("noswap_only matches with any death rows:", deaths_ns.height, "| median deaths per match:",
      deaths_ns["len"].median() if deaths_ns.height else None)
print("noswap_only matches path days:", NS.group_by("path_day").len().sort("path_day").rows())
print("noswap_only matches mdt range:", NS["mdt"].min(), NS["mdt"].max())
# swap_only comparison: deaths per match
deaths_so = PD.join(X.filter((pl.col("build") == "9961") & (pl.col("pair12_type") == "swap_only")).select("match_key"),
                    on="match_key", how="inner").group_by("match_key").len()
print("swap_only 9961 matches with death rows:", deaths_so.height, "| median deaths per match:",
      deaths_so["len"].median())

print("\n== (4) round 9 and round 27 disagreements (winner vs DP): evidence split (5v5, 9961 and 9842) ==")
Q = R.filter(pl.col("player_count") == 20)
for rd in [9, 27]:
    T = Q.filter((pl.col("round") == rd) & pl.col("dp_side").is_not_null())
    T = T.with_columns(
        (pl.col("winner_team_code") == pl.col("dp_side")).alias("w_eq_dp"),
        pl.when(pl.col("bomb_side").is_not_null()).then(pl.lit("bomb"))
          .when(pl.col("elim_side").is_not_null()).then(pl.lit("elim"))
          .otherwise(pl.lit("none")).alias("evidence"),
    )
    T = T.with_columns(
        pl.when(pl.col("evidence") == "bomb").then(pl.col("bomb_side"))
          .when(pl.col("evidence") == "elim").then(pl.col("elim_side"))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("evid_side"))
    T = T.with_columns(
        (pl.col("winner_team_code") == pl.col("evid_side")).alias("w_eq_evid"),
        (pl.col("dp_side") == pl.col("evid_side")).alias("dp_eq_evid"))
    print(f"round {rd}: rows with dp side, by build x w_eq_dp x evidence:")
    print(T.group_by(["build", "w_eq_dp", "evidence"]).agg(
        pl.len().alias("n"),
        pl.col("w_eq_evid").sum().alias("win_eq_evid"),
        pl.col("dp_eq_evid").sum().alias("dp_eq_evid"),
    ).sort(["build", "w_eq_dp", "evidence"]))

print("\n== (5) round 9 in 9961: does the disagreement line up with a halftime-style pair signature at 8->9? ==")
S89 = Q.filter((pl.col("round") == 8) & (pl.col("build") == "9961")).select(["match_key", "noswap_valid", "swap_valid"])
T9 = Q.filter((pl.col("round") == 9) & (pl.col("build") == "9961") & pl.col("dp_side").is_not_null()).join(
    S89, on="match_key", how="left").with_columns(
    (pl.col("winner_team_code") == pl.col("dp_side")).alias("w_eq_dp"),
    pl.when(pl.col("noswap_valid") & pl.col("swap_valid")).then(pl.lit("both"))
      .when(pl.col("noswap_valid")).then(pl.lit("noswap_only"))
      .when(pl.col("swap_valid")).then(pl.lit("swap_only"))
      .otherwise(pl.lit("neither")).alias("pair8_type"))
print(T9.group_by(["pair8_type", "w_eq_dp"]).len().sort(["pair8_type", "w_eq_dp"]).rows())
