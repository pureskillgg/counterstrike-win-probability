"""D1 (swarm2) hypothesis test, v2.

Hypothesis D1: in the affected 2024 matches the side codes do NOT flip at halftime, so the
bomb-side mapping disagrees with winner_team_code from round 13 on.

Tests (all on the 2024 tome, G:\\pureskill\\csds-cs2\\tomes\\2024-02-g\\tome\\csds):
  T1 per-match team_code flip at 12->13 (player_info), split by round-13 mismatch status
  T2 flip profile at every boundary 1->2 .. 24->25, affected vs clean vs no-bomb matches
  T3 distribution of the 12->13 flip fraction in the mismatch vs non-mismatch matches
  T4 mismatch round pattern per affected match (is it only 13 and 25, or 'from 13 on')
  T5 team split sanity at rounds 12 and 13 (count of code 2 and code 3 players)
  T6 planter cross-check: bomb_state player_team_code vs player_info; planter flip 12->13
  T7 player_spawn cross-check: spawn team vs player_info, and spawn flip 12->13
  T8 score-based side winner from round_state (independent of round_end and team codes),
     compared with winner_team_code and with the bomb outcome; halftime score-swap check

ASSUMPTIONS (marked in the report):
  A1 team_code is a side code (2 = T, 3 = CT); a flip is a change of code between two rounds.
  A2 covered player = present in player_info at both rounds of a boundary.
  A3 flip class: flip_full >= 0.9, no_flip <= 0.1, else partial.
  A4 matches with duplicate (match, round, player) keys in player_info are excluded (3 matches).
  A5 affected = at least one mismatched round with a bomb outcome (player_count == 20).
  A6 evaluable = at least 6 covered players.
  A7 side winner from round_state = side whose score rose between the first and last tick of the round.
  A8 mismatch definition copied from scripts/swarm/e2_side_check_r13.py.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
OUT = Path(r"G:\pureskill\csds-cs2\scripts\swarm2")
FULL, NO = 0.9, 0.1      # A3
MIN_COV = 6              # A6
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fl(ch)], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ----------------------------------------------------------------------------- load
RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"])
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_id_fixed", "player_team_code"])
PI_raw = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"])
SP = rd("player_spawn", ["match_key", "round", "tick", "player_id_fixed", "player_team_code"])
RS = rd("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])

k = ["match_key", "round", "player_id_fixed"]
dup_matches = sorted(PI_raw.filter(pl.len().over(k) > 1)["match_key"].unique().to_list())
PI = PI_raw.filter(~pl.col("match_key").is_in(dup_matches))
hdr("SETUP")
print(f"A4 dup matches excluded: {len(dup_matches)}")
print("  " + "\n  ".join(dup_matches))
print(f"player_info rows total={PI_raw.height} kept={PI.height}; round_state rows={RS.height}; spawn rows={SP.height}")

# bomb outcome per round (identical to e2 script)
outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
RO = (RE.filter(pl.col("player_count") == 20)
      .join(outcome, on=["match_key", "round"], how="inner")
      .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
      .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
print(f"[check] outcome rounds={RO.height} mismatches={int(RO['mismatch'].sum())} (expect 15436 / 499)")

# ----------------------------------------------------------------------------- T1 flip per boundary
parts = []
for b in range(1, 25):
    a = PI.filter(pl.col("round") == b).select(["match_key", "player_id_fixed", pl.col("team_code").alias("ta")])
    c = PI.filter(pl.col("round") == b + 1).select(["match_key", "player_id_fixed", pl.col("team_code").alias("tb")])
    g = (a.join(c, on=["match_key", "player_id_fixed"], how="inner")
          .with_columns((pl.col("ta") != pl.col("tb")).alias("flip"))
          .group_by("match_key").agg(pl.len().alias("n_both"), pl.col("flip").sum().alias("n_flip"))
          .with_columns(pl.lit(b).alias("b")))
    parts.append(g)
FL = pl.concat(parts).with_columns((pl.col("n_flip") / pl.col("n_both")).alias("ff"))
FL.write_parquet(OUT / "D1_boundary_flips_v2.parquet")

MATCHES = PI.select("match_key").unique().with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
b1213 = (FL.filter(pl.col("b") == 12).select(["match_key", "n_both", "n_flip", "ff"])
         .rename({"n_both": "n_both_12_13", "n_flip": "n_flip_12_13", "ff": "ff_12_13"}))
has = PI.select(["match_key", "round"]).unique().with_columns(pl.lit(1).alias("p"))
r12 = has.filter(pl.col("round") == 12).select(["match_key", pl.col("p").alias("has_r12")])
r13 = has.filter(pl.col("round") == 13).select(["match_key", pl.col("p").alias("has_r13")])
mm_match = RO.group_by("match_key").agg(
    pl.len().alias("n_outcome_rounds"),
    pl.col("mismatch").sum().alias("n_mismatch"),
    pl.col("round").filter(pl.col("mismatch")).sort().alias("mismatch_rounds"),
    (pl.col("mismatch") & (pl.col("round") == 13)).any().alias("r13_mismatch_any"),
    (pl.col("round") == 13).sum().alias("r13_outcome_n"))
M = (MATCHES.join(r12, on="match_key", how="left").join(r13, on="match_key", how="left")
     .join(b1213, on="match_key", how="left").join(mm_match, on="match_key", how="left")
     .with_columns(pl.col("n_outcome_rounds").fill_null(0), pl.col("n_mismatch").fill_null(0),
                   pl.col("r13_outcome_n").fill_null(0), pl.col("r13_mismatch_any").fill_null(False),
                   pl.col("has_r12").fill_null(0), pl.col("has_r13").fill_null(0),
                   pl.col("n_both_12_13").fill_null(0))
     .with_columns(
         pl.when(pl.col("n_both_12_13") < MIN_COV).then(pl.lit("not_evaluable"))
           .when(pl.col("ff_12_13") >= FULL).then(pl.lit("flip_full"))
           .when(pl.col("ff_12_13") <= NO).then(pl.lit("no_flip"))
           .otherwise(pl.lit("partial")).alias("flip_class_12_13"),
         pl.when(pl.col("n_outcome_rounds") == 0).then(pl.lit("no_bomb_evidence"))
           .when(pl.col("n_mismatch") > 0).then(pl.lit("affected"))
           .otherwise(pl.lit("clean")).alias("match_status"),
         pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"),
         pl.when(pl.col("r13_outcome_n") == 0).then(pl.lit("no_r13_outcome"))
           .when(pl.col("r13_mismatch_any")).then(pl.lit("r13_MISMATCH"))
           .otherwise(pl.lit("r13_clean")).alias("r13_status")))
M.write_parquet(OUT / "D1_match_table_v2.parquet")

hdr("T1 per-match team_code flip at 12->13 (player_info), by round-13 status")
print(M.group_by(["r13_status", "flip_class_12_13"]).len().sort(["r13_status", "flip_class_12_13"]))
print("\nmatch_status x flip class (all matches):")
print(M.group_by(["match_status", "flip_class_12_13"]).len().sort(["match_status", "flip_class_12_13"]))
print("\nday group x flip class:")
print(M.group_by(["dg", "flip_class_12_13"]).len().sort(["dg", "flip_class_12_13"]))
print("\ncoverage: matches with r12 =", int(M["has_r12"].sum()), " r13 =", int(M["has_r13"].sum()),
      " both =", int((M["has_r12"] * M["has_r13"]).sum()), " total =", M.height)

# ----------------------------------------------------------------------------- T2 boundary profile
hdr("T2 flip profile at each boundary b->b+1, by match_status (ff = fraction of covered players whose code changed)")
FLS = FL.join(M.select(["match_key", "match_status", "r13_status"]), on="match_key", how="inner")
prof = (FLS.filter(pl.col("n_both") >= MIN_COV)
        .group_by(["b", "match_status"])
        .agg(pl.len().alias("n_eval"),
             (pl.col("ff") >= FULL).sum().alias("n_full"),
             (pl.col("ff") <= NO).sum().alias("n_none"),
             pl.col("ff").median().alias("median_ff"),
             pl.col("ff").mean().round(4).alias("mean_ff"))
        .sort(["b", "match_status"]))
print(prof)

# ----------------------------------------------------------------------------- T3 distribution at 12->13
hdr("T3 distribution of flip fraction at 12->13, r13 mismatch vs r13 clean matches")
ev = M.filter(pl.col("flip_class_12_13") != "not_evaluable")
print(ev.group_by("r13_status").agg(
    pl.len().alias("n_matches_evaluable"),
    pl.col("ff_12_13").min().alias("ff_min"),
    pl.col("ff_12_13").median().alias("ff_median"),
    pl.col("ff_12_13").max().alias("ff_max"),
    pl.col("n_both_12_13").sum().alias("covered_players_total"),
    pl.col("n_flip_12_13").sum().alias("flipped_players_total"),
    (pl.col("n_both_12_13") - pl.col("n_flip_12_13")).sum().alias("unflipped_players_total"),
    pl.col("n_both_12_13").median().alias("median_covered"),
).sort("r13_status"))
nr = M.filter((pl.col("r13_status") == "r13_MISMATCH") & (pl.col("flip_class_12_13") != "flip_full"))
print("r13 mismatch matches NOT flip_full:", nr.height)
print("r13 mismatch matches with r12 or r13 missing from player_info:",
      M.filter((pl.col("r13_status") == "r13_MISMATCH") & ((pl.col("has_r12") == 0) | (pl.col("has_r13") == 0))).height)
print("r13 clean matches NOT flip_full (evaluable only):",
      M.filter((pl.col("r13_status") == "r13_clean") & (pl.col("flip_class_12_13") != "flip_full")).height)

# ----------------------------------------------------------------------------- T4 mismatch round pattern
hdr("T4 mismatch round pattern per affected match")
aff = M.filter(pl.col("n_mismatch") > 0).with_columns(
    pl.col("mismatch_rounds").list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("rounds_str"))
print("affected matches:", aff.height)
print(aff.group_by("rounds_str").len().sort("len", descending=True).head(40))
rounds_flat = aff.explode("mismatch_rounds").group_by("mismatch_rounds").len().sort("mismatch_rounds")
print("\nmismatched rounds by round number across affected matches (rounds with bomb outcome, all day groups):")
print(rounds_flat)
print("\nmismatch rounds in 14..24 (should be ~0 if 'from 13 on' were true):",
      RO.filter(pl.col("mismatch") & pl.col("round").is_between(14, 24)).height)
print("mismatch rounds in 1..12:", RO.filter(pl.col("mismatch") & (pl.col("round") <= 12)).height)
print("mismatch rounds at 13:", RO.filter(pl.col("mismatch") & (pl.col("round") == 13)).height,
      " at 25:", RO.filter(pl.col("mismatch") & (pl.col("round") == 25)).height,
      " other:", RO.filter(pl.col("mismatch") & ~pl.col("round").is_in([13, 25])).height)

# ----------------------------------------------------------------------------- T5 team split sanity
hdr("T5 team split sanity: number of players with code 2 (T) and code 3 (CT) at rounds 12 and 13")
split = (PI.group_by(["match_key", "round"]).agg(
    (pl.col("team_code") == 2).sum().alias("nT"), (pl.col("team_code") == 3).sum().alias("nCT")))
s12 = split.filter(pl.col("round") == 12).select(["match_key", pl.col("nT").alias("nT12"), pl.col("nCT").alias("nCT12")])
s13 = split.filter(pl.col("round") == 13).select(["match_key", pl.col("nT").alias("nT13"), pl.col("nCT").alias("nCT13")])
ss = s12.join(s13, on="match_key", how="inner").join(M.select(["match_key", "r13_status"]), on="match_key")
print(ss.with_columns(((pl.col("nT12") == 5) & (pl.col("nCT12") == 5)).alias("r12_5v5"),
                      ((pl.col("nT13") == 5) & (pl.col("nCT13") == 5)).alias("r13_5v5"))
        .group_by(["r13_status", "r12_5v5", "r13_5v5"]).len().sort(["r13_status"]))
print("\nexample pattern (nT12,nCT12 -> nT13,nCT13) counts, all matches:")
print(ss.group_by(["nT12", "nCT12", "nT13", "nCT13"]).len().sort("len", descending=True).head(12))

# ----------------------------------------------------------------------------- T6 planter cross-check
hdr("T6 planter cross-check (bomb_state planter team vs player_info team, same round)")
PL = (BS.filter(pl.col("event_type") == "bomb_planted")
      .with_columns(pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("pid"))
      .filter(pl.col("pid").is_not_null())
      .group_by(["match_key", "round"]).agg(pl.col("pid").first(), pl.col("player_team_code").first().alias("bs_team")))
PL = PL.filter(~pl.col("match_key").is_in(dup_matches))
pi_same = PI.select(["match_key", "round", pl.col("player_id_fixed").alias("pid"), pl.col("team_code").alias("pi_team")])
PL = PL.join(pi_same, on=["match_key", "round", "pid"], how="left")
PL = PL.join(M.select(["match_key", "r13_status"]), on="match_key", how="left")
print("planted rounds:", PL.height, " with player_info match for planter:", PL.filter(pl.col("pi_team").is_not_null()).height)
print("planter bs_team value counts:", PL.group_by("bs_team").len().sort("bs_team").to_dicts())
print("planter bs_team == pi_team:", PL.filter(pl.col("pi_team").is_not_null()).select((pl.col("bs_team") == pl.col("pi_team")).mean().round(5)).item())
print(PL.filter(pl.col("round") == 13).group_by(["r13_status", "bs_team", "pi_team"]).len().sort(["r13_status", "bs_team", "pi_team"]))
# planter flip 12->13 at round 13
pl12 = PI.filter(pl.col("round") == 12).select(["match_key", pl.col("player_id_fixed").alias("pid"), pl.col("team_code").alias("pi12")])
p13 = PL.filter(pl.col("round") == 13).join(pl12, on=["match_key", "pid"], how="left")
print("\nplanter at round 13 with player_info at round 12 available:")
print(p13.group_by(["r13_status", "pi12", "pi_team"]).len().sort(["r13_status", "pi12", "pi_team"]))

# ----------------------------------------------------------------------------- T7 spawn cross-check
hdr("T7 player_spawn team vs player_info team; spawn flip at 12->13")
spw = (SP.with_columns(pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("pid"))
       .filter(pl.col("pid").is_not_null() & ~pl.col("match_key").is_in(dup_matches))
       .sort("tick").group_by(["match_key", "round", "pid"]).agg(pl.col("player_team_code").first().alias("sp_team")))
cmp = spw.join(pi_same, on=["match_key", "round", "pid"], how="inner")
print("spawn rows (first per player-round):", spw.height, " matched to player_info:", cmp.height)
print("spawn team == player_info team:", cmp.select((pl.col("sp_team") == pl.col("pi_team")).mean().round(5)).item())
print("agreement by r13 status (rounds 12 and 13 only):")
print(cmp.filter(pl.col("round").is_in([12, 13])).join(M.select(["match_key", "r13_status"]), on="match_key")
        .group_by(["r13_status", "round"]).agg(pl.len().alias("n"),
                                              (pl.col("sp_team") == pl.col("pi_team")).mean().round(5).alias("agree"))
        .sort(["r13_status", "round"]))
sp12 = spw.filter(pl.col("round") == 12).select(["match_key", "pid", pl.col("sp_team").alias("s12")])
sp13 = spw.filter(pl.col("round") == 13).select(["match_key", "pid", pl.col("sp_team").alias("s13")])
sf = sp12.join(sp13, on=["match_key", "pid"], how="inner").with_columns((pl.col("s12") != pl.col("s13")).alias("flip"))
sfm = (sf.group_by("match_key").agg(pl.len().alias("n"), pl.col("flip").mean().alias("ff"))
       .join(M.select(["match_key", "r13_status"]), on="match_key", how="left"))
print("\nspawn-based 12->13 flip fraction by r13 status:")
print(sfm.group_by("r13_status").agg(pl.len().alias("matches"), pl.col("ff").min().alias("min"),
                                     pl.col("ff").median().alias("median"), pl.col("ff").max().alias("max")))

# ----------------------------------------------------------------------------- T8 score-based side winner
hdr("T8 score-based side winner per round (round_state) vs winner_team_code and vs bomb outcome")
RSs = RS.sort(["match_key", "round", "tick"])
sc = (RSs.group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_before"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_before"),
        pl.col("t_score").sort_by("tick").last().alias("t_after"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_after")))
sc = sc.with_columns(
    pl.when((pl.col("t_after") - pl.col("t_before")) == 1).then(pl.lit(2))
      .when((pl.col("ct_after") - pl.col("ct_before")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side"))
print("score_side counts (all rounds in round_state):")
print(sc.group_by("score_side", maintain_order=True).len())

RE_sc = (RE.join(sc, on=["match_key", "round"], how="inner")
         .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
                       pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                         .when(pl.col("round") == 25).then(pl.lit("r25"))
                         .otherwise(pl.lit("other")).alias("rbin"))
         .join(M.select(["match_key", "r13_status"]), on="match_key", how="left"))
RE_sc = RE_sc.filter(pl.col("score_side").is_not_null())
RE_sc = RE_sc.with_columns(
    pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"),
    (pl.col("score_side") == pl.col("winner_team_code")).alias("score_eq_winner"))
print("\nwinner_team_code == score-based side winner, all rounds with player_count==20 (by bin and day group):")
print(RE_sc.filter(pl.col("player_count") == 20).group_by(["dg", "rbin"])
        .agg(pl.len().alias("n"), (~pl.col("score_eq_winner")).sum().alias("wt_disagrees_score"),
             (~pl.col("score_eq_winner")).mean().round(4).alias("disagree_rate"))
        .sort(["dg", "rbin"]))
print("\nround 13 only: winner_team_code vs score side winner, by r13_status (player_count==20):")
print(RE_sc.filter((pl.col("round") == 13) & (pl.col("player_count") == 20)).group_by(["r13_status", "score_eq_winner"])
        .len().sort(["r13_status", "score_eq_winner"]))

RO_sc = RO.join(sc.select(["match_key", "round", "score_side"]), on=["match_key", "round"], how="left")
print("\nbomb outcome vs score side winner on bomb rounds, by r13 status and round bin:")
RO_sc = RO_sc.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                             .when(pl.col("round") == 25).then(pl.lit("r25")).otherwise(pl.lit("other")).alias("rbin"))
RO_sc = RO_sc.join(M.select(["match_key", "r13_status"]), on="match_key", how="left")
print(RO_sc.filter(pl.col("score_side").is_not_null())
        .group_by(["rbin", "r13_status", "mismatch", pl.col("bomb_side") == pl.col("score_side")])
        .len().sort(["rbin", "r13_status", "mismatch"]))

# halftime score-swap check
hdr("T8b halftime score swap: does t_score at round 13 start equal ct_score at round 12 end (swap)?")
sw = (sc.filter(pl.col("round").is_in([12, 13])).select(["match_key", "round", "t_before", "ct_before", "t_after", "ct_after"]))
s12x = sw.filter(pl.col("round") == 12).select(["match_key", pl.col("t_after").alias("t12_after"), pl.col("ct_after").alias("ct12_after")])
s13x = sw.filter(pl.col("round") == 13).select(["match_key", pl.col("t_before").alias("t13_before"), pl.col("ct_before").alias("ct13_before")])
hx = s12x.join(s13x, on="match_key", how="inner").join(M.select(["match_key", "r13_status"]), on="match_key", how="left")
hx = hx.with_columns(
    ((pl.col("t13_before") == pl.col("ct12_after")) & (pl.col("ct13_before") == pl.col("t12_after")) &
     (pl.col("t13_before") != pl.col("t12_after"))).alias("swap"),
    ((pl.col("t13_before") == pl.col("t12_after")) & (pl.col("ct13_before") == pl.col("ct12_after"))).alias("noswap"))
print(hx.group_by(["r13_status", "swap", "noswap"]).len().sort(["r13_status", "swap", "noswap"]))

# ----------------------------------------------------------------------------- save per-match table
M.filter(pl.col("n_outcome_rounds") > 0).select(
    ["match_key", "day", "dg", "n_outcome_rounds", "n_mismatch", "r13_outcome_n", "r13_status",
     "n_both_12_13", "n_flip_12_13", "ff_12_13", "flip_class_12_13", "match_status"]
).with_columns(pl.col("n_mismatch").cast(pl.Int64)).write_csv(OUT / "D1_match_flip_vs_mismatch_v2.csv")
print("\nwrote D1_match_table_v2.parquet, D1_boundary_flips_v2.parquet, D1_match_flip_vs_mismatch_v2.csv")
