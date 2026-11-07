"""verify D7-5 step 5: coverage and physical-consistency probes for the 9961 noswap_only matches.

(1) Death/bomb coverage by round bucket for noswap_only matches vs swap_only matches (build 9961).
(2) Physical-consistency test: a round with no bomb outcome and no elimination cannot be a T win
    (T can only win by elimination of CT or bomb explosion). Count winner_team_code and dp_side == 2 in such rounds,
    by build, using only rounds that have complete death coverage (all deaths in the window have a team code).
(3) Per-round printout for 2 noswap_only matches (score sequence, winner, dp, bomb, elim, death side counts).
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
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_5")


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


PD = pl.scan_parquet(files(BASE / f"basic_player_death{SUF}")).select(
    ["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
BSA = pl.scan_parquet(files(BASE / f"basic_bomb_state{SUF}")).select(
    ["match_key", "round", "tick", "event_type", "player_team_code"]).collect()

R = pl.read_parquet(str(SCR / "R_all_rounds.parquet")).with_columns(
    pl.col("build_num").cast(pl.Utf8).alias("build"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("path_day"))
Q = R.filter(pl.col("player_count") == 20)

# pair12 type per match (structural score signature)
S12 = Q.filter(pl.col("round") == 12).select(["match_key", "noswap_valid", "swap_valid"]).with_columns(
    pl.when(pl.col("noswap_valid") & pl.col("swap_valid")).then(pl.lit("both"))
      .when(pl.col("noswap_valid")).then(pl.lit("noswap_only"))
      .when(pl.col("swap_valid")).then(pl.lit("swap_only"))
      .otherwise(pl.lit("neither")).alias("pair12_type")).select(["match_key", "pair12_type"])
M = Q.group_by("match_key").agg(pl.col("build").first()).join(S12, on="match_key", how="left")
M9961 = M.filter(pl.col("build") == "9961")

# per-match counts by round bucket
def bucket(col):
    return (pl.when(pl.col("round") <= 12).then(pl.lit("r01_12"))
            .when(pl.col("round") <= 24).then(pl.lit("r13_24"))
            .otherwise(pl.lit("r25p")))

pdb = (PD.join(M9961.select(["match_key", "pair12_type"]), on="match_key", how="inner")
       .with_columns(bucket("round").alias("bk")))
cov = pdb.group_by(["pair12_type", "bk"]).agg(
    pl.len().alias("deaths"), pl.col("match_key").n_unique().alias("matches")).sort(["pair12_type", "bk"])
print("== (1) 9961 death rows by pair12 type and round bucket ==")
print(cov)
bsb = (BSA.filter(pl.col("event_type") == "bomb_planted")
       .join(M9961.select(["match_key", "pair12_type"]), on="match_key", how="inner")
       .with_columns(bucket("round").alias("bk")))
print("9961 bomb_planted rows by pair12 type and round bucket:")
print(bsb.group_by(["pair12_type", "bk"]).agg(pl.len().alias("plants"),
                                              pl.col("match_key").n_unique().alias("matches")).sort(["pair12_type", "bk"]))

# physical consistency: rounds with no bomb outcome and no elimination
print("\n== (2) physical consistency: rounds without bomb or elimination evidence ==")
# complete death coverage in window: count null-team deaths in [fe_tick, re_tick]
NT = (PD.filter(pl.col("player_team_code").is_null())
      .join(Q.select(["match_key", "round", "fe_tick", "re_tick"]), on=["match_key", "round"], how="inner")
      .filter((pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
      .group_by(["match_key", "round"]).len().rename({"len": "null_team_deaths"}))
Z = Q.join(NT, on=["match_key", "round"], how="left").with_columns(pl.col("null_team_deaths").fill_null(0))
Z = Z.with_columns(
    (pl.col("bomb_side").is_null() & pl.col("elim_side").is_null()).alias("no_evidence"),
    (pl.col("null_team_deaths") == 0).alias("complete_team_codes"),
)
NE = Z.filter(pl.col("no_evidence") & pl.col("re_tick").is_not_null())
print("rounds with no bomb and no elimination (5v5, round_end present):")
print(NE.group_by(["build", "complete_team_codes"]).agg(
    pl.len().alias("n"),
    (pl.col("winner_team_code") == 2).sum().alias("winner_T"),
    (pl.col("dp_side") == 2).sum().alias("dp_T"),
    pl.col("dp_side").is_not_null().sum().alias("n_dp"),
    pl.col("nT").mean().alias("mean_nT_deaths"),
    pl.col("nCT").mean().alias("mean_nCT_deaths"),
).sort(["build", "complete_team_codes"]))
print("winner_team_code==2 (T win) rounds with no bomb and no elim, by round number (9961, complete codes):")
print(NE.filter((pl.col("build") == "9961") & pl.col("complete_team_codes") & (pl.col("winner_team_code") == 2))
      .group_by("round").len().sort("round").rows())
print("dp_side==2 (T win) rounds with no bomb and no elim, by round number (9961, complete codes):")
print(NE.filter((pl.col("build") == "9961") & pl.col("complete_team_codes") & (pl.col("dp_side") == 2))
      .group_by("round").len().sort("round").rows())

# (3) printout for two noswap_only matches
print("\n== (3) per-round printout, two 9961 noswap_only matches ==")
ns = M9961.filter(pl.col("pair12_type") == "noswap_only")["match_key"].head(2).to_list()
rows = (Q.filter(pl.col("match_key").is_in(ns) & pl.col("round").is_between(1, 16))
        .select(["match_key", "round", "fe_tick", "re_tick", "t_score", "ct_score", "winner_team_code",
                 "dp_side", "bomb_side", "elim_side", "nT", "nCT"]).sort(["match_key", "round"]))
print(rows.with_columns(pl.col("match_key").str.slice(0, 26)))
