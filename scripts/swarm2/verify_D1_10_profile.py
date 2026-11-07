"""D1-10 verifier: profile categorical values and score-column relations in the 2026 tome.
Column projection only; each table is under ~100 MB in memory.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def fl(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}*" / "dataframe_*")))


def lf(ch, cols):
    return pl.scan_parquet(fl(ch)).select(cols)


print("== round_state event_type counts")
rs_lazy = lf("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase", "is_warmup"])
rs = rs_lazy.collect()
print(rs.group_by("event_type").len().sort("len", descending=True))
print("== round_state phase counts")
print(rs.group_by("phase").len().sort("len", descending=True))
print("== is_warmup counts")
print(rs.group_by("is_warmup").len())
print("== null counts in score cols")
print(rs.select([pl.col(c).is_null().sum().alias(c) for c in ["t_score", "ct_score", "t_score_raw", "ct_score_raw"]]))
print("== fraction t_score == t_score_raw and ct_score == ct_score_raw:")
print(rs.select(
    (pl.col("t_score") == pl.col("t_score_raw")).mean().alias("t_eq_t_raw"),
    (pl.col("ct_score") == pl.col("ct_score_raw")).mean().alias("ct_eq_ct_raw"),
))
print("== rows where t_score != t_score_raw, by round (top 15)")
print(rs.filter(pl.col("t_score") != pl.col("t_score_raw")).group_by("round").len().sort("round").head(40))

print("\n== score_update team_code and rows per round")
su = lf("score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"]).collect()
print(su.group_by("team_code").len())
print("score_update delta counts:")
print(su.with_columns((pl.col("new_score") - pl.col("old_score")).alias("delta")).group_by("delta").len().sort("delta"))
print("score_update rows per (match, round) max:", su.group_by(["match_key", "round"]).len()["len"].max())

print("\n== bomb_action event_type")
ba = lf("bomb_action", ["match_key", "round", "tick", "event_type", "player_team_code", "site_code"]).collect()
print(ba.group_by("event_type").len().sort("len", descending=True))
print("== bomb_defuse event_type")
bd = lf("bomb_defuse", ["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
print(bd.group_by("event_type").len().sort("len", descending=True))
print("== bomb_state event_type")
bs = lf("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code", "site_code"]).collect()
print(bs.group_by("event_type").len().sort("len", descending=True))

print("\n== round_end win_reason_code/message distribution")
re_ = pl.scan_parquet(fl("round_end")).select(["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"]).collect()
print(re_.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len().sort("len", descending=True).head(30))
print("duplicate (match_key, round) rows in round_end:", re_.height - re_.unique(["match_key", "round"]).height)
print("distinct matches in round_end:", re_["match_key"].n_unique())
print("distinct matches in round_state:", rs["match_key"].n_unique())
print("distinct matches in score_update:", su["match_key"].n_unique())
