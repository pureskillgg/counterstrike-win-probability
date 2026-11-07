"""Timeline of round 12-14 for sample 2024 matches: round_state events with scores, round_end rows, deaths, bomb events.
Purpose: see WHERE the score changes relative to round_end and to the last death (tick level).
Samples: 2 matches with a flagged round 13 (reasons 8 and 9), 1 match where round 13 agrees.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(40)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SUF = ".2023-12-28,2024-02-25"
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


flag13 = R.filter((pl.col("round") == 13) & (pl.col("player_count") == 20) & (pl.col("S") != pl.col("W"))
                  & (pl.col("S") != 0) & (pl.col("dd") >= 12))
ok13 = R.filter((pl.col("round") == 13) & (pl.col("player_count") == 20) & (pl.col("S") == pl.col("W"))
                & (pl.col("S") != 0) & (pl.col("dd") >= 12))
picks = []
for reason in [8, 9]:
    m = flag13.filter(pl.col("win_reason_code") == reason).select("match_key").head(1)
    picks.append(("flag_r13_reason%d" % reason, m["match_key"][0]))
m = ok13.select("match_key").head(1)
picks.append(("agree_r13", m["match_key"][0]))

rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase"])
de = read("player_death", ["match_key", "round", "tick", "player_team_code", "player_id", "attacker_id"])
bs = read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
re_ = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
for label, mk in picks:
    print("\n==========", label, mk)
    sub = R.filter(pl.col("match_key") == mk).filter(pl.col("round").is_in([12, 13, 14]))
    print(sub.select(["round", "live", "end", "S", "W", "win_reason_code", "t_score", "ct_score", "t_next", "ct_next",
                      "last_tick", "gap", "n_T_deaths_win", "n_CT_deaths_win", "bomb_side", "plant_tick", "bomb_out_tick", "planter_side"]))
    r_events = rs.filter((pl.col("match_key") == mk) & (pl.col("round").is_in([12, 13, 14])) &
                         pl.col("event_type").is_in(["round_freeze_end", "round_end", "round_officially_ended",
                                                     "cs_win_panel_round", "round_start", "bomb_planted", "bomb_defused",
                                                     "bomb_exploded"]))
    print("round_state events (freeze_end / end / bomb):")
    print(r_events.sort("tick").select(["round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase"]))
    print("round_end rows:")
    print(re_.filter((pl.col("match_key") == mk) & pl.col("round").is_in([12, 13, 14])).sort("tick"))
    d = de.filter((pl.col("match_key") == mk) & pl.col("round").is_in([13])).sort("tick")
    print("deaths in round 13 (tick, victim side):", d.select(["tick", "player_team_code"]).rows())
    b = bs.filter((pl.col("match_key") == mk) & pl.col("round").is_in([13])).sort("tick")
    print("bomb rows round 13:", b.select(["tick", "event_type", "player_team_code"]).rows())
