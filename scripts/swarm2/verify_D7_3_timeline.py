"""verify D7-3: timeline of a few build-9961 round-13 bomb-mismatch matches, and of control matches.

Purpose: decide whether the bomb_state outcome for a mismatched round belongs to that round (tick window),
or to a neighbouring round, and what round_state scores say about the round winner (independent of round_end).
Read-only. Prints only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(260)
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def chan_dirs(base, ch):
    return sorted(p for p in base.glob(f"basic_{ch}.*") if p.is_dir())


def read_chan(base, ch, cols, where=None):
    parts = []
    for d in chan_dirs(base, ch):
        for f in sorted(glob.glob(str(d / "dataframe_*"))):
            lf = pl.scan_parquet(f).select(cols)
            if where is not None:
                lf = lf.filter(where)
            parts.append(lf.collect())
    return pl.concat(parts, how="vertical_relaxed")


R = pl.read_parquet(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_3_rounds.parquet")
print("rows:", R.height)

# round_state: event types and score columns
RS = read_chan(T24, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                                    "t_score_raw", "ct_score_raw", "phase", "is_warmup"])
print("round_state event_type counts:", RS.group_by("event_type").len().sort("len", descending=True).rows())
RST = read_chan(T24, "round_start", ["match_key", "round", "tick"]).rename({"tick": "start_tick"})

# pick example matches: build 9961, round 13, bomb mismatch
ex = (R.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & pl.col("mm_bomb"))
      .sort("mdt").select(["match_key", "map_name", "mdt", "lag_bomb_minus_end"]).head(4))
print("\nexample mismatched matches:")
print(ex)
ctrl = (R.filter((pl.col("build_num") == 9842) & (pl.col("round") == 13) & pl.col("has_bomb"))
        .sort("mdt").select(["match_key", "map_name", "mdt"]).head(2))
print("control matches (build 9842, round 13 with bomb):")
print(ctrl)

for mk in pl.concat([ex.select("match_key"), ctrl.select("match_key")])["match_key"].to_list():
    print("\n" + "=" * 110)
    print("MATCH", mk)
    sub = R.filter(pl.col("match_key") == mk).filter(pl.col("round").is_between(11, 15)).sort("round")
    print(sub.select(["round", "tick", "winner_team_code", "win_reason_message", "reason_side", "bomb_side",
                      "bomb_tick", "lag_bomb_minus_end", "mm_reason", "mm_bomb", "player_count"]))
    st = RST.filter((pl.col("match_key") == mk) & pl.col("round").is_between(11, 15)).sort("round")
    print("round_start:", st.rows())
    bs = read_chan(T24, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"],
                   where=pl.col("match_key") == mk).filter(pl.col("round").is_between(11, 15)).sort("tick")
    print("bomb_state events (round, tick, type, player_team_code):")
    print(bs.select(["round", "tick", "event_type", "player_team_code"]).rows())
    rs = RS.filter((pl.col("match_key") == mk) & pl.col("round").is_between(11, 15)).sort("tick")
    print("round_state rows (round, tick, event, t_score, ct_score, t_raw, ct_raw, phase):")
    print(rs.select(["round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase"]).rows())
