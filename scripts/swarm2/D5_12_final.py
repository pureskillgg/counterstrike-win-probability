"""D5 step 12: final per-match counts for the report (2024 tome).

  1. Regime A (round_announce_match_start present) late matches: per-match flags on the best truth
     (bomb side if a bomb outcome exists, else last-victim death side if no bomb plant, else none):
       flip13 / flip9 / flip25 / flip28 with the number of matches where truth was available.
  2. Map x regime (days 12-25 and all): is the regime (and so the defect) map-specific?
  3. Duplicate freeze-end rows per (match, round), labels 1..15, by group.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_12_final.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D5_common import read  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
B = pl.read_parquet(SCR / "bomb.parquet")

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
mk = R.group_by("match_key").agg(pl.col("n_round_announce_match_start").fill_null(0).sum().alias("mstart"))
M = (M.join(mk, on="match_key", how="left")
     .with_columns(regime=pl.when(pl.col("mstart") > 0).then(pl.lit("A")).otherwise(pl.lit("B")),
                   grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other"))))
r1 = R.filter(pl.col("round") == 1).select(["match_key", pl.col("in_end").alias("r1_end")])

DE = read("player_death", ["match_key", "round", "tick", "player_team_code"]).filter(pl.col("player_team_code").is_in([2, 3]))
last = (DE.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("player_team_code").last().alias("last_victim"))
    .with_columns(death_implied=pl.when(pl.col("last_victim") == 2).then(pl.lit(3)).otherwise(pl.lit(2)))
    .select(["match_key", "round", "death_implied"]))
planted = B.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique().with_columns(
    has_plant=pl.lit(True))
bside = (B.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
         .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
         .with_columns(bomb_side=pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)))
         .select(["match_key", "round", "bomb_side"]))
T = (R.filter(pl.col("in_end") & pl.col("winner").is_not_null())
     .select(["match_key", "round", "winner"])
     .join(last, on=["match_key", "round"], how="left")
     .join(planted, on=["match_key", "round"], how="left")
     .join(bside, on=["match_key", "round"], how="left")
     .with_columns(has_plant=pl.col("has_plant").fill_null(False)))
T = T.with_columns(
    truth=pl.when(pl.col("bomb_side").is_not_null()).then(pl.col("bomb_side"))
    .when(~pl.col("has_plant") & pl.col("death_implied").is_not_null()).then(pl.col("death_implied"))
    .otherwise(pl.lit(None)))
T = T.with_columns(flip=pl.when(pl.col("truth").is_null()).then(None)
                   .otherwise((pl.col("truth") != pl.col("winner")).cast(pl.Int8)))
T = T.join(M.select(["match_key", "grp", "regime", "map_name", "day"]), on="match_key", how="left")

print("== 1. regime A, per-match flags (any usable round of that label) ==")
A = M.filter(pl.col("regime") == "A").select(["match_key", "grp", "day", "map_name"]).join(r1, on="match_key", how="left")
A = A.with_columns(pl.col("r1_end").fill_null(False))
per_lab = {}
for lab in [9, 13, 25, 28]:
    t = (T.filter((pl.col("round") == lab) & pl.col("flip").is_not_null())
         .group_by("match_key").agg(pl.col("flip").max().alias(f"f{lab}"), pl.len().alias(f"n{lab}")))
    per_lab[lab] = t
Z = A
for lab, t in per_lab.items():
    Z = Z.join(t, on="match_key", how="left")
Z = Z.with_columns([pl.col(f"f{lab}").fill_null(-1) for lab in per_lab])
print(Z.group_by(["grp", "r1_end"]).agg(
    pl.len().alias("matches"),
    (pl.col("f13") == 1).sum().alias("flip13_any"),
    (pl.col("f13") == 0).sum().alias("noflip13_usable"),
    (pl.col("f13") == -1).sum().alias("no_truth13"),
    (pl.col("f9") == 1).sum().alias("flip9_any"),
    (pl.col("f9") == 0).sum().alias("noflip9_usable"),
    (pl.col("f25") == 1).sum().alias("flip25"),
    (pl.col("f28") == 1).sum().alias("flip28"),
).sort(["grp", "r1_end"]))
print("\nregime A late matches: flip9 x flip13 (usable both):")
print(Z.filter((pl.col("f13") >= 0) & (pl.col("f9") >= 0) & pl.col("grp").is_in(["ctrl_late", "aff_r13"]))
      .group_by(["grp", "f9", "f13"]).agg(pl.len().alias("matches")).sort(["grp", "f9", "f13"]))
print("\nregime A late matches: flip13 by flip9 status for matches with usable round-13 truth:")
print(Z.filter((pl.col("f13") >= 0) & pl.col("grp").is_in(["ctrl_late", "aff_r13"]))
      .group_by(["f9", "f13"]).agg(pl.len().alias("matches")).sort(["f9", "f13"]))

print("\n== 2. map x regime (late days 12-25 and all) ==")
mm = M.with_columns(late=pl.col("day").is_in(LATE))
print(mm.group_by(["map_name", "regime"]).agg(pl.len().alias("matches")).sort(["map_name", "regime"]))
print(mm.filter(pl.col("late")).group_by(["map_name"]).agg(
    pl.len().alias("late_matches"),
    (pl.col("regime") == "A").sum().alias("regime_A"),
    pl.col("aff_r13").sum().alias("aff_r13")).sort("late_matches", descending=True))
flip_map = (T.filter((pl.col("round") == 13) & pl.col("flip").is_not_null() & (pl.col("regime") == "A"))
            .group_by("map_name").agg(pl.len().alias("usable_r13"), pl.col("flip").sum().alias("flipped_r13"))
            .with_columns((pl.col("flipped_r13") / pl.col("usable_r13")).round(3).alias("flip_rate")))
print("\nround-13 flip rate by map, regime A (any truth source):")
print(flip_map.sort("usable_r13", descending=True))

print("\n== 3. duplicate freeze-end rows per (match, round), labels 1..15 ==")
print(R.filter(pl.col("round").is_between(1, 15)).join(M.select(["match_key", "grp"]), on="match_key", how="left")
      .group_by(["grp", "round"]).agg((pl.col("n_fe_rows").fill_null(0) > 1).sum().alias("rounds_with_2plus_fe"),
                                      pl.col("n_fe_rows").fill_null(0).sum().alias("fe_rows"))
      .sort(["grp", "round"]).filter(pl.col("rounds_with_2plus_fe") > 0))
print("total rounds with 2+ freeze-end rows (all labels):", R.filter(pl.col("n_fe_rows").fill_null(0) > 1).height)
print("total rounds with 2+ round_start rows (channel):", R.filter(pl.col("n_start_rows").fill_null(0) > 1).height)
