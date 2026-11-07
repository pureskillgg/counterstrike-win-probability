"""verify D9-8 step 9: apply the same score-vs-round_end test to the 2026 tome.

Claim under test (from the task context): 2026 has 0 round-winner mismatches in 7,427 rounds.
That claim is based on bomb outcomes only. This script checks the stronger, bomb-free test:
round winner from round_state score deltas (round_officially_ended) vs round_end winner, and the
bomb-outcome side vs both, on the 2026-09-18..10-08 tome. Read-only, projected columns, lazy scans.

ASSUMPTIONS (same as steps 5-6): A1 side-based score delta; A2 round 1 starts at 0-0; A3 first
round_officially_ended row per (match, round); round 27-type OT swap artifacts are reported separately
and not treated as proof either way. player_count == 20 is the pc filter used throughout.
Output goes to stdout (tee'd by the caller). Scratch parquet only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUF = ".2026-09-18,2026-10-08"
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")


def files_of(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


re_s = pl.scan_parquet(files_of("round_end"))
print("2026 round_end schema:", dict(re_s.collect_schema()))
re26 = (re_s.select(["match_key", "round", "tick", "winner_team_code", "win_reason_message", "player_count"])
        .collect())
print("2026 round_end rows:", re26.height, " matches:", re26["match_key"].n_unique())
print("pc values:", re26.group_by("player_count").agg(pl.len().alias("n")).sort("player_count").rows()[:12])
re20 = re26.filter(pl.col("player_count") == 10).group_by(["match_key", "round"]).agg(
    pl.col("winner_team_code").first(), pl.col("win_reason_message").first(), pl.col("tick").min().alias("end_tick"))
print("2026 pc20 (match, round) rows:", re20.height)
print("message x winner (2026):")
print(re26.group_by(["win_reason_message", "winner_team_code"]).agg(pl.len().alias("n")).sort(["win_reason_message"]))

bs_s = pl.scan_parquet(files_of("bomb_state")).select(["match_key", "round", "tick", "event_type"])
bs26 = bs_s.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"]).agg(
    pl.col("event_type").unique().alias("evs"))
bs26 = bs26.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side")).collect()
print("\n2026 bomb outcome (match, round) rows:", bs26.height)

rst_s = pl.scan_parquet(files_of("round_state")).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
print("2026 round_state schema:", dict(pl.scan_parquet(files_of("round_state")).collect_schema()))
rst = rst_s.filter(pl.col("event_type") == "round_officially_ended").group_by(["match_key", "round"]).agg(
    pl.col("t_score").first(), pl.col("ct_score").first()).collect()
print("2026 round_officially_ended (match, round) rows:", rst.height)
prev = rst.with_columns((pl.col("round") + 1).alias("round")).select(
    ["match_key", "round", pl.col("t_score").alias("t_prev"), pl.col("ct_score").alias("ct_prev")])
sc = rst.join(prev, on=["match_key", "round"], how="left").with_columns(
    pl.when((pl.col("round") == 1)).then(0).otherwise(pl.col("t_prev")).alias("t_prev"),
    pl.when((pl.col("round") == 1)).then(0).otherwise(pl.col("ct_prev")).alias("ct_prev"))
sc = sc.with_columns(((pl.col("t_score") - pl.col("t_prev"))).alias("dT"),
                     ((pl.col("ct_score") - pl.col("ct_prev"))).alias("dCT"))
sc = sc.with_columns(pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(2)
                     .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(3).otherwise(None).alias("score_winner"))
print("\n2026 score delta patterns (top):")
print(sc.group_by(["dT", "dCT"]).agg(pl.len().alias("n")).sort("n", descending=True).head(8))

J = re20.join(sc.select(["match_key", "round", "score_winner"]), on=["match_key", "round"], how="left")
J = J.join(bs26.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")
J = J.with_columns(pl.col("match_key").str.extract(r"csds/(\d{4})/(\d\d)/(\d\d)/", 2).alias("month"),
                   pl.col("match_key").str.extract(r"csds/(\d{4})/(\d\d)/(\d\d)/", 3).alias("day"))
print("\n2026 month/day sample:", J.select(["month", "day"]).unique().head(5).rows())
J = J.with_columns(
    pl.col("score_winner").is_not_null().alias("has_score"),
    pl.col("bomb_side").is_not_null().alias("has_bomb"),
    (pl.col("score_winner") != pl.col("winner_team_code")).alias("score_disagree"),
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("bomb_disagree"),
)
print("\n2026 pc20 rounds:", J.height, " with clean score winner:", int(J["has_score"].sum()),
      " with bomb outcome:", int(J["has_bomb"].sum()))
print("bomb-outcome mismatch (2026):", int((J["has_bomb"] & J["bomb_disagree"]).sum()),
      "of", int(J["has_bomb"].sum()))
print("score-vs-round_end disagreement (2026):", int((J["has_score"] & J["score_disagree"]).sum()),
      "of", int(J["has_score"].sum()))
print("\nscore-vs-round_end disagreement by round number (2026, top):")
print(J.filter(J["has_score"] & J["score_disagree"]).group_by("round").agg(pl.len().alias("n")).sort("n", descending=True).head(15))
print("\nmessages on 2026 score disagreements:")
print(J.filter(J["has_score"] & J["score_disagree"]).group_by("win_reason_message").agg(pl.len().alias("n")).sort("n", descending=True))
print("\nbomb rounds with score: score==bomb / score==round_end:")
bb = J.filter(J["has_bomb"] & J["has_score"])
print("  n=", bb.height, " score==bomb:", int((bb["score_winner"] == bb["bomb_side"]).sum()),
      " score==round_end:", int((bb["score_winner"] == bb["winner_team_code"]).sum()))
J.write_parquet(SCRATCH / "verify_D9_8_2026_rounds.parquet")
