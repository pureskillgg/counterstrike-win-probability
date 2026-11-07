"""Verifier D3-5, probe 3: is round_end internally consistent (winner_team_code vs win_reason_code side)?
Plus the full round_state timeline for one round-13 case. Read only. Prints to stdout.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_width_chars(230)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
# win reason -> side of the winner (CS2 round_end reasons)
REASON_SIDE = {1: T, 9: T, 13: T, 17: T, 7: CT, 8: CT, 11: CT, 12: CT, 18: CT}


def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


RE = g_read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
RE = RE.with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
    pl.col("win_reason_code").replace_strict(REASON_SIDE, default=-1).alias("reason_side"),
)
RE = RE.with_columns((pl.col("reason_side") == pl.col("winner_team_code")).alias("agree"),
                     pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 1).then(pl.lit("r1"))
                     .otherwise(pl.lit("other")).alias("rbin"))
print("reason codes not mapped:", RE.filter(pl.col("reason_side") == -1).height)
print("\nwinner vs reason-side agreement, all rows:", RE.group_by("agree").len().rows())
print("\nagreement by round bin:")
print(RE.group_by(["rbin", "agree"]).len().sort(["rbin", "agree"]).rows())
d_grp = RE.with_columns(pl.when(pl.col("day").cast(pl.Int64) >= 12).then(pl.lit("day12_25")).otherwise(pl.lit("day01_07")).alias("grp"))
print("\nagreement by day group x round bin:")
print(d_grp.group_by(["grp", "rbin", "agree"]).len().sort(["grp", "rbin", "agree"]).rows())
print("\nround-13 disagreement by day (day12-25):")
print(d_grp.filter((pl.col("round") == 13) & (pl.col("grp") == "day12_25"))
      .group_by(["day", "agree"]).len().sort(["day", "agree"]).rows())
print("\ndisagreement by round number (day12-25), rounds 1..26:")
print(d_grp.filter((pl.col("grp") == "day12_25") & (pl.col("round") <= 26))
      .group_by(["round", "agree"]).len().sort(["round", "agree"]).rows())
print("\ndisagreement by round number (day01-07), rounds 1..26:")
print(d_grp.filter((pl.col("grp") == "day01_07") & (pl.col("round") <= 26))
      .group_by(["round", "agree"]).len().sort(["round", "agree"]).rows())

# detail: the disagreeing rows, by reason
dis = RE.filter(~pl.col("agree"))
print("\nreason x winner for disagreeing rows:", dis.group_by(["win_reason_code", "winner_team_code"]).len().sort("len", descending=True).rows()[:20])

# one round-13 case: 01648186 full round_state timeline rounds 12..14, with player deaths
M = "csds/2024/02/12/01648186-3840-41d9-8090-4bea9e926412/csds"
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
tl = RS.filter((pl.col("match_key") == M) & (pl.col("round").is_between(12, 14))).sort("tick")
print("\nround_state timeline, match", M[-40:], "rounds 12-14 (score-relevant events):")
keep_ev = ["round_freeze_end", "round_end", "round_start", "bomb_planted", "bomb_defused", "bomb_exploded",
           "round_officially_ended", "cs_win_panel_round", "round_announce_match_point"]
print(tl.filter(pl.col("event_type").is_in(keep_ev)).select(["round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).rows())
print("all event types in rounds 12-14:", tl.group_by("event_type").len().sort("len", descending=True).rows())
print("score changes (tick, event, t, ct) rounds 12-14:")
chg = tl.with_columns(((pl.col("t_score") != pl.col("t_score").shift(1)) | (pl.col("ct_score") != pl.col("ct_score").shift(1))).alias("chg"))
print(chg.filter(pl.col("chg")).select(["round", "tick", "event_type", "t_score", "ct_score"]).rows())
RD = g_read("player_death", ["match_key", "round", "tick", "player_team_code", "attacker_id", "player_id"])
rd13 = RD.filter((pl.col("match_key") == M) & (pl.col("round") == 13)).sort("tick")
print("\nplayer_death round 13 (tick, victim team):", rd13.select(["tick", "player_team_code"]).rows())
