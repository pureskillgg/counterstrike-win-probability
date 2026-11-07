"""verify D1-7 (swarm2 verifier): per-match dump of score counters, winner and bomb around rounds 22-30 (2024) and 25-34 (2026)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"


def scan(base, name, suf=""):
    return pl.scan_parquet(sorted(glob.glob(str(base / f"basic_{name}{suf}*" / "dataframe_*"))))


def rounds_table(base, suf, rmin, rmax, pc_filter=None):
    re_ = (scan(base, "round_end", suf).select(["match_key", "round", "winner_team_code", "player_count"])
           .unique(["match_key", "round"]).collect())
    if pc_filter is not None:
        re_ = re_.filter(pl.col("player_count") == pc_filter)
    rs = scan(base, "round_state", suf).select(["match_key", "round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "event_type"]).collect()
    rs = rs.sort(["match_key", "round", "tick"])
    per = (rs.group_by(["match_key", "round"]).agg(
        pl.col("t_score").first().alias("t0"), pl.col("t_score").last().alias("t1"),
        pl.col("ct_score").first().alias("c0"), pl.col("ct_score").last().alias("c1"),
        pl.col("t_score_raw").first().alias("tr0"), pl.col("t_score_raw").last().alias("tr1"),
        pl.col("ct_score_raw").first().alias("cr0"), pl.col("ct_score_raw").last().alias("cr1"),
        pl.len().alias("n_rs")))
    bs = (scan(base, "bomb_state", suf).filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
          .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).collect())
    bs = bs.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side")).select(["match_key", "round", "bomb_side"])
    out = (re_.join(per, on=["match_key", "round"], how="left").join(bs, on=["match_key", "round"], how="left")
           .filter(pl.col("round").is_between(rmin, rmax)))
    return out


# 2024 sample: matches where round 27 score side disagrees with winner (pc 20 as in claim)
t24 = rounds_table(T24, SUF24, 22, 30)
t24 = t24.with_columns(((pl.col("t1") - pl.col("t0")) == 1).alias("tw"), ((pl.col("c1") - pl.col("c0")) == 1).alias("cw"))
t24 = t24.with_columns(pl.when(pl.col("tw") & ~pl.col("cw")).then(2).when(pl.col("cw") & ~pl.col("tw")).then(3).otherwise(None).alias("score_side"))
r27 = t24.filter((pl.col("round") == 27) & pl.col("score_side").is_not_null() & (pl.col("score_side") != pl.col("winner_team_code")) & pl.col("bomb_side").is_not_null())
print("2024 sample: matches with round-27 bomb-verified score disagreement:", r27.height)
sample = r27.select("match_key").head(2)["match_key"].to_list()
for mk in sample:
    print("\n==== 2024", mk)
    print(t24.filter(pl.col("match_key") == mk).select(["round", "player_count", "winner_team_code", "bomb_side", "score_side", "t0", "t1", "c0", "c1", "tr0", "tr1", "cr0", "cr1", "n_rs"]).sort("round"))

# 2024 sample: a match with round-28 winner-vs-bomb mismatch
r28 = t24.filter((pl.col("round") == 28) & pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code")))
print("\n2024 sample: matches with round-28 winner-bomb mismatch:", r28.height)
sample2 = r28.select("match_key").head(2)["match_key"].to_list()
for mk in sample2:
    print("\n==== 2024 (r28 winner-bomb mismatch)", mk)
    print(t24.filter(pl.col("match_key") == mk).select(["round", "player_count", "winner_team_code", "bomb_side", "score_side", "t0", "t1", "c0", "c1", "tr0", "tr1", "cr0", "cr1", "n_rs"]).sort("round"))
