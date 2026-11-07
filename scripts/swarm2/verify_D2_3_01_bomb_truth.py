"""verify_D2_3_01: independent recompute of the 2024 round-13/25/28 flip claim with a bomb-only truth.

Truth used here does NOT use score fields, team-side tracking (sx), or elimination inference:
  B = side implied by the first in-window bomb outcome event
      bomb_exploded -> 2 (T), bomb_defused -> 3 (CT)
  flip = recorded winner_team_code != B

ASSUMPTIONS (marked, all reported in the log):
  A1  round window = [round_freeze_end tick, round_end tick + TOL], TOL = 40 ticks (copied from D2_truth.py)
  A2  the first in-window outcome event is the round outcome
  A3  exploded => T (code 2), defused => CT (code 3)
  A4  winner_team_code is a side code (2 = T, 3 = CT); supported by rounds 14-24 where it agrees with the bomb at ~100%

Scope: 2024 tome only (G:\\pureskill\\csds-cs2\\tomes\\2024-02-g\\tome\\csds). Read-only.
Memory: column projection on every channel, one channel at a time, well under 3 GB.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D2_3_01_bomb_truth.py
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
MK = "match_key"
TOL = 40  # ASSUMPTION A1


def files(ch):
    fd = sorted(glob.glob(str(BASE / f"basic_{ch}.*")))
    assert len(fd) == 1, (ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def scan(ch, cols):
    return pl.scan_parquet(files(ch)).select(cols)


def rbin_expr():
    r = pl.col("round")
    return (pl.when(r <= 12).then(pl.lit("r01_12"))
            .when(r == 13).then(pl.lit("r13"))
            .when(r <= 24).then(pl.lit("r14_24"))
            .when(r == 25).then(pl.lit("r25"))
            .when(r <= 27).then(pl.lit("r26_27"))
            .when(r == 28).then(pl.lit("r28"))
            .otherwise(pl.lit("r29plus")).alias("rbin"))


# ---------------------------------------------------------------- load
re_raw = scan("round_end", [MK, "round", "tick", "winner_team_code", "win_reason_code",
                            "win_reason_message", "player_count"]).collect()
print("round_end rows:", re_raw.height,
      " duplicate (match,round) rows:", re_raw.height - re_raw.unique([MK, "round"]).height)
re_ = (re_raw.sort("tick").unique([MK, "round"], keep="first")
       .rename({"tick": "t_end"}))
print("distinct (match,round) in round_end:", re_.height, " matches:", re_[MK].n_unique())
print("winner_team_code counts:", re_["winner_team_code"].value_counts().sort("winner_team_code").rows())
print("player_count counts:", re_["player_count"].value_counts().sort("player_count").rows())

rs = (scan("round_state", [MK, "round", "tick", "event_type"])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
fe = (rs.sort("tick").unique([MK, "round"], keep="first")
      .select([MK, "round", pl.col("tick").alias("t_fe")]))
print("freeze_end rows:", rs.height, " distinct (match,round):", fe.height)

bs = scan("bomb_state", [MK, "round", "tick", "event_type", "player_team_code"]).collect()
print("bomb_state event types:", bs["event_type"].value_counts().sort("count", descending=True).rows())

# ---------------------------------------------------------------- window and outcome
bs = (bs.join(fe, on=[MK, "round"], how="left")
      .join(re_.select([MK, "round", "t_end"]), on=[MK, "round"], how="left")
      .with_columns(((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + TOL))
                    .alias("in_win")))
outc = bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
print("\noutcome events:", outc.height,
      " in window (A1):", outc.filter(pl.col("in_win")).height,
      " outside window:", outc.filter(~pl.col("in_win")).height,
      " with no freeze_end or round_end row:", outc.filter(pl.col("t_fe").is_null() | pl.col("t_end").is_null()).height)

ob = (outc.filter(pl.col("in_win")).sort("tick")
      .group_by([MK, "round"]).agg(
          pl.col("event_type").first().alias("bo_first"),
          pl.len().alias("n_outcome_in"),
          pl.col("event_type").n_unique().alias("n_outcome_types")))
ob = ob.with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2))
                     .otherwise(pl.lit(3)).cast(pl.Int64).alias("B"))  # A2, A3
print("rounds with an in-window outcome:", ob.height,
      " with >1 outcome type:", ob.filter(pl.col("n_outcome_types") > 1).height,
      " with >1 outcome event:", ob.filter(pl.col("n_outcome_in") > 1).height)

plant = (bs.filter((pl.col("event_type") == "bomb_planted") & pl.col("in_win"))
         .sort("tick").group_by([MK, "round"])
         .agg(pl.col("player_team_code").first().alias("planter_side"), pl.len().alias("n_plant")))
print("rounds with an in-window bomb_planted:", plant.height,
      " planter_side counts:", plant["planter_side"].value_counts().sort("planter_side").rows())

# ---------------------------------------------------------------- base table
base = (re_.select([MK, "round", "t_end", "winner_team_code", "win_reason_code",
                    "win_reason_message", "player_count"])
        .join(ob.select([MK, "round", "B", "n_outcome_in", "bo_first"]), on=[MK, "round"], how="left")
        .join(plant, on=[MK, "round"], how="left")
        .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
                      rbin_expr()))
print("\nbase rows:", base.height, " rows with B:", base.filter(pl.col("B").is_not_null()).height,
      " rows with B and player_count==20:",
      base.filter(pl.col("B").is_not_null() & (pl.col("player_count") == 20)).height)

k = base.filter(pl.col("B").is_not_null()).with_columns(
    (pl.col("winner_team_code") != pl.col("B")).alias("flip"))
k20 = k.filter(pl.col("player_count") == 20)

# ---------------------------------------------------------------- headline table
print("\n== H1. Bomb-only truth: recorded winner vs bomb side, by round bin (all rows with B)")
t = (k.group_by("rbin").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"),
                            pl.col("flip").mean().round(4).alias("rate")).sort("rbin"))
print(t)
print("   claim (full truth set, D2_final): r01_12 14/41497 (0.03%) | r13 1888/3270 (57.7%) | "
      "r14_24 6/22549 | r25 114/200 (57.0%) | r28 115/202 (56.9%) | r29+ 0/674")

print("\n== H1b. Same, player_count == 20 only")
print(k20.group_by("rbin").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"),
                               pl.col("flip").mean().round(4).alias("rate")).sort("rbin"))

print("\n== H2. Flips per round number (B-truth, all rows) for rounds 1-35")
pr = (k.group_by("round").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"))
      .sort("round"))
print(pr.filter(pl.col("round") <= 35).rows())

# ---------------------------------------------------------------- match-level pattern
print("\n== H3. Match-level pattern at rounds 13, 25, 28 (B-truth; matches with all three present)")
piv = (k.filter(pl.col("round").is_in([13, 25, 28]))
       .select([MK, pl.col("round").cast(pl.Utf8), "flip"])
       .pivot(on="round", index=MK, values="flip", aggregate_function="first"))
print("matches with any of 13/25/28 in B-truth:", piv.height)
full3 = piv.drop_nulls(["13", "25", "28"])
print("matches with B at 13, 25 and 28:", full3.height)
print(full3.group_by(["13", "25", "28"]).len().sort(["13", "25", "28"]))
pair = piv.drop_nulls(["13"])
print("matches with B at 13:", pair.height, " flipped at 13:", pair.filter(pl.col("13")).height)
print("crosstab 13 x 25 (where both present):")
print(piv.drop_nulls(["13", "25"]).group_by(["13", "25"]).len().sort(["13", "25"]))
print("crosstab 13 x 28 (where both present):")
print(piv.drop_nulls(["13", "28"]).group_by(["13", "28"]).len().sort(["13", "28"]))

# ---------------------------------------------------------------- round-13 structure
r13 = k.filter(pl.col("round") == 13)
print("\n== H4. Round 13 (B-truth): flips by day")
print(r13.group_by("day").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"),
                              pl.col("flip").mean().round(3).alias("rate")).sort("day"))

print("\n== H5. Round 13 (B-truth): planter side by flip status (T should be 2)")
print(r13.group_by(["flip", "planter_side"]).len().sort(["flip", "planter_side"]))

print("\n== H6. Round 13 (B-truth): win_reason_code by flip status and B")
print(r13.group_by(["flip", "B", "win_reason_code", "winner_team_code"]).len()
      .sort(["flip", "B", "win_reason_code", "winner_team_code"]))

print("\n== H7. Win reason message (all rounds with B), by flip")
print(k.group_by(["flip", "win_reason_message"]).len().sort(["flip", "len"], descending=[False, True]).head(30))

print("\n== H8. Round 13 (B-truth): flip rate by winner_team_code (recorded) and B")
print(r13.group_by(["winner_team_code", "B"]).len().sort(["winner_team_code", "B"]))

# ---------------------------------------------------------------- robustness of the bomb window
print("\n== R1. Robustness of round-13 flip count to the window (A1 variants)")
ob_all = (outc.sort("tick").group_by([MK, "round"]).agg(
    pl.col("event_type").first().alias("bo_first"), pl.len().alias("n_out_any"))
    .with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
                  .cast(pl.Int64).alias("B_any")))
ob_tol0 = (outc.filter(pl.col("tick") <= pl.col("t_end")).sort("tick")
           .group_by([MK, "round"]).agg(pl.col("event_type").first().alias("bo_first"))
           .with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
                         .cast(pl.Int64).alias("B_tol0")))
for label, bb, col in [("A1 TOL=40 (main)", ob, "B"), ("TOL=0 (no slack after round_end)", ob_tol0, "B_tol0"),
                       ("no window at all (first outcome of any tick)", ob_all, "B_any")]:
    j = (re_.select([MK, "round", "winner_team_code"]).join(bb.select([MK, "round", col]), on=[MK, "round"], how="inner")
         .filter(pl.col("round").is_in([13, 25, 28, 9, 12, 14])))
    j = j.with_columns((pl.col("winner_team_code") != pl.col(col)).alias("flip"))
    g = j.group_by("round").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"),
                                pl.col("flip").mean().round(3).alias("rate")).sort("round")
    print(f"  {label}: {g.rows()}")

print("\nend of H/R tables (no files written)")
