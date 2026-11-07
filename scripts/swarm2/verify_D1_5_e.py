"""verify_D1_5_e: coverage of bomb outcomes. bomb_state holds exploded/defused for only part of the planted rounds.
Question: do basic_bomb_action / basic_bomb_defuse hold explosion or defuse events for the planted rounds that
have no bomb_state outcome? Reads the 2024 tome only. Appends to logs\\swarm2\\verify_D1_5.log.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
LOG = ROOT / "logs" / "swarm2" / "verify_D1_5.log"
_fh = open(LOG, "a", encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)


def emit(*parts):
    s = " ".join(str(p) for p in parts)
    print(s, flush=True)
    _fh.write(s + "\n")
    _fh.flush()


def fdir(ch):
    d = sorted(BASE.glob(f"basic_{ch}.*"))
    assert len(d) == 1, (ch, d)
    return d[0]


def rd(ch, cols):
    fs = sorted(glob.glob(str(fdir(ch) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


BS = rd("bomb_state", ["match_key", "round", "tick", "event_type"])
PLANT = BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique()
OUT_BS = (BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
          .select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("has_bs_outcome")))
PL = PLANT.join(OUT_BS, on=["match_key", "round"], how="left").with_columns(pl.col("has_bs_outcome").fill_null(False))
MISS = PL.filter(~pl.col("has_bs_outcome")).select(["match_key", "round"])
emit("planted (match,round) keys:", PL.height, " with bomb_state outcome:", PL.height - MISS.height,
     " without:", MISS.height)

BA = rd("bomb_action", ["match_key", "round", "tick", "event_type"])
emit("bomb_action event types (all rows):", BA.group_by("event_type").len().sort("event_type").rows())
BAm = BA.join(MISS, on=["match_key", "round"], how="inner")
emit("bomb_action rows in the planted-without-outcome rounds, by event type:",
     BAm.group_by("event_type").len().sort("len", descending=True).rows())
emit("distinct planted-without-outcome rounds that have any bomb_action row:",
     BAm.select(["match_key", "round"]).unique().height, "of", MISS.height)

BD = rd("bomb_defuse", ["match_key", "round", "tick", "event_type"])
emit("bomb_defuse event types (all rows):", BD.group_by("event_type").len().sort("event_type").rows())
BDm = BD.join(MISS, on=["match_key", "round"], how="inner")
emit("bomb_defuse rows in the planted-without-outcome rounds, by event type:",
     BDm.group_by("event_type").len().sort("len", descending=True).rows())
emit("distinct planted-without-outcome rounds with a bomb_defuse row:",
     BDm.select(["match_key", "round"]).unique().height, "of", MISS.height)
_fh.close()
print("appended", LOG)
