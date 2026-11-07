"""verify D1-5: trace round-13 mismatch matches (2024 tome). Reads only; prints to stdout."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def rd(ch, cols=None):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


RE = rd("round_end").unique(["match_key", "round"])
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase"])
RST = rd("round_start", ["match_key", "round", "tick"]).unique(["match_key", "round"]).rename({"tick": "start_tick"})

outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("bomb_tick")))
RO = (RE.filter(pl.col("player_count") == 20).join(outc, on=["match_key", "round"], how="inner")
      .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
      .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch")))
mm13 = RO.filter((pl.col("round") == 13) & pl.col("mismatch")).select("match_key").to_series().to_list()
print("round-13 mismatch matches:", len(mm13))
ex = mm13[:3]
for mk in ex:
    print("\n====", mk)
    re_ = RE.filter((pl.col("match_key") == mk) & pl.col("round").is_between(10, 15)).select(
        ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "legacy_code", "second"]).sort("round")
    print("round_end:"); print(re_)
    rs_ = RS.filter((pl.col("match_key") == mk) & pl.col("round").is_between(11, 15))
    agg = (rs_.sort("tick").group_by("round").agg(
        pl.col("tick").min().alias("tick_min"), pl.col("tick").max().alias("tick_max"),
        pl.col("t_score").first().alias("t_first"), pl.col("ct_score").first().alias("ct_first"),
        pl.col("t_score").last().alias("t_last"), pl.col("ct_score").last().alias("ct_last"),
        pl.col("t_score_raw").first().alias("traw_first"), pl.col("ct_score_raw").first().alias("craw_first"),
        pl.col("t_score_raw").last().alias("traw_last"), pl.col("ct_score_raw").last().alias("craw_last"),
        pl.col("phase").first().alias("phase")).sort("round"))
    print("round_state first/last per round (tick window, scores):"); print(agg)
    ev = rs_.filter(pl.col("event_type").is_in(["round_start", "round_end", "round_officially_ended", "round_freeze_end", "bomb_planted", "bomb_defused", "bomb_exploded", "cs_win_panel_round"]))
    print("round_state events (round,tick,event,t,ct):"); print(ev.select(["round", "tick", "event_type", "t_score", "ct_score"]).sort(["tick"]))
    bs_ = BS.filter((pl.col("match_key") == mk) & pl.col("round").is_between(11, 15)).sort("tick")
    print("bomb_state events:"); print(bs_)
