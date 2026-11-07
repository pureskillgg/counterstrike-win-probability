"""D1 hypothesis test: do player team codes flip at halftime (12 -> 13) in the 2024 matches
where the round-13 bomb outcome mismatches winner_team_code?

Data: 2024 tome, player_info (round, player_id_fixed, team_code), round_end, bomb_state.
Mismatch definition copied from scripts/swarm/e2_side_check_r13.py (player_count == 20,
bomb_defused -> 3, bomb_exploded -> 2, compared with winner_team_code).
ASSUMPTIONS:
  A1 team_code is a side code (2 = T, 3 = CT) per player per round; a flip = code changes between rounds.
  A2 a player is 'covered' at a boundary if present in player_info at both rounds.
  A3 flip class thresholds: flip_full >= 0.9 of covered players; no_flip <= 0.1; else partial.
  A4 matches with duplicate (match, round, player) keys are excluded from the primary test.
  A5 'affected' = at least one mismatched round with a bomb outcome (player_count == 20).
  A6 a match needs >= 6 covered players at 12->13 to be evaluable.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(70)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
OUT = Path(r"G:\pureskill\csds-cs2\scripts\swarm2")
FULL, NO = 0.9, 0.1          # ASSUMPTION A3
MIN_COVERED = 6              # ASSUMPTION A6


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fl(ch)], how="vertical_relaxed")


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"])
BS = rd("bomb_state", ["match_key", "round", "event_type", "player_id_fixed", "player_team_code"])
PI_raw = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"])

# ---- A4: duplicate-key matches
k = ["match_key", "round", "player_id_fixed"]
dup_matches = sorted(PI_raw.filter(pl.len().over(k) > 1)["match_key"].unique().to_list())
PI = PI_raw.filter(~pl.col("match_key").is_in(dup_matches))
print(f"[A4] dup matches excluded from primary: {len(dup_matches)} -> {dup_matches}")
print(f"player_info rows total={PI_raw.height}, kept={PI.height}")

# ---- bomb outcome per round (mirrors e2 script)
outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
RO = (RE.filter(pl.col("player_count") == 20)
      .join(outcome, on=["match_key", "round"], how="inner")
      .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
      .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
print(f"[check] outcome rounds={RO.height} mismatches={int(RO['mismatch'].sum())} (expect 15436 / 499)")

# ---- per-boundary flip table: round b -> b+1
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
FL.write_parquet(OUT / "D1_boundary_flips.parquet")

# ---- match universe
MATCHES = PI.select("match_key").unique().with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
print(f"matches in primary universe: {MATCHES.height}")

b1213 = (FL.filter(pl.col("b") == 12)
         .select(["match_key", "n_both", "n_flip", "ff"])
         .rename({"n_both": "n_both_12_13", "n_flip": "n_flip_12_13", "ff": "ff_12_13"}))
rounds_present = PI.group_by("match_key").agg(pl.col("round").n_unique().alias("n_rounds_pi"))
has = PI.select(["match_key", "round"]).unique().with_columns(pl.lit(1).alias("p"))
r12 = has.filter(pl.col("round") == 12).select(["match_key", pl.col("p").alias("has_r12")])
r13 = has.filter(pl.col("round") == 13).select(["match_key", pl.col("p").alias("has_r13")])

mm_match = RO.group_by("match_key").agg(
    pl.len().alias("n_outcome_rounds"),
    pl.col("mismatch").sum().alias("n_mismatch"),
    pl.col("round").filter(pl.col("mismatch")).sort().alias("mismatch_rounds"),
    (pl.col("mismatch") & (pl.col("round") == 13)).any().alias("r13_mismatch_any"),
    (pl.col("round") == 13).sum().alias("r13_outcome_n"))

M = (MATCHES.join(rounds_present, on="match_key", how="left")
     .join(r12, on="match_key", how="left")
     .join(r13, on="match_key", how="left")
     .join(b1213, on="match_key", how="left")
     .join(mm_match, on="match_key", how="left")
     .with_columns(pl.col("n_outcome_rounds").fill_null(0),
                   pl.col("n_mismatch").fill_null(0),
                   pl.col("r13_outcome_n").fill_null(0),
                   pl.col("r13_mismatch_any").fill_null(False),
                   pl.col("has_r12").fill_null(0),
                   pl.col("has_r13").fill_null(0)))


def cls(ff, n):
    if ff is None or n is None or n < MIN_COVERED:
        return "not_evaluable"
    if ff >= FULL:
        return "flip_full"
    if ff <= NO:
        return "no_flip"
    return "partial"


M = M.with_columns(
    pl.struct(["ff_12_13", "n_both_12_13"])
      .map_elements(lambda s: cls(s["ff_12_13"], s["n_both_12_13"]), return_dtype=pl.Utf8)
      .alias("flip_class_12_13"))
M = M.with_columns(
    pl.when(pl.col("n_outcome_rounds") == 0).then(pl.lit("no_bomb_evidence"))
      .when(pl.col("n_mismatch") > 0).then(pl.lit("affected"))
      .otherwise(pl.lit("clean")).alias("match_status"),
    pl.when(pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)])).then(pl.lit("d12-25"))
      .otherwise(pl.lit("d01-07")).alias("dg"))
M.write_parquet(OUT / "D1_match_table.parquet")

print("\n=== 1. Coverage of rounds 12 and 13 in player_info (primary matches) ===")
print(M.select(pl.col("has_r12").sum().alias("matches_with_r12"),
               pl.col("has_r13").sum().alias("matches_with_r13"),
               (pl.col("has_r12") * pl.col("has_r13")).sum().alias("matches_with_both"),
               pl.len().alias("total")))
print("\n=== 2. Match status (bomb evidence) ===")
print(M.group_by("match_status").len().sort("match_status"))
print("\n=== 3. Flip class at 12->13 by match status ===")
print(M.group_by(["match_status", "flip_class_12_13"]).len().sort(["match_status", "flip_class_12_13"]))
print("\n=== 4. Flip class at 12->13 by day group ===")
print(M.group_by(["dg", "flip_class_12_13"]).len().sort(["dg", "flip_class_12_13"]))
print("\n=== 5. Matches with an r13 bomb outcome: flip class x r13 mismatch ===")
R13 = M.filter(pl.col("r13_outcome_n") > 0)
print(R13.group_by(["flip_class_12_13", "r13_mismatch_any"]).len().sort(["flip_class_12_13", "r13_mismatch_any"]))
print("\n=== 6. All matches with any bomb outcome: flip class x any mismatch ===")
print(M.filter(pl.col("n_outcome_rounds") > 0)
        .group_by(["flip_class_12_13", pl.col("n_mismatch") > 0]).len()
        .sort(["flip_class_12_13"]))

M.filter(pl.col("n_outcome_rounds") > 0).select(
    ["match_key", "day", "n_outcome_rounds", "n_mismatch", "mismatch_rounds", "r13_outcome_n",
     "r13_mismatch_any", "n_both_12_13", "n_flip_12_13", "ff_12_13", "flip_class_12_13", "match_status"]
).write_csv(OUT / "D1_match_flip_vs_mismatch.csv")
print("\nwrote D1_match_flip_vs_mismatch.csv and D1_match_table.parquet")
