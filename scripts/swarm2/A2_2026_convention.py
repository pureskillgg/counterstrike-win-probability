# A2: does the 2026 tome use the same round_start label convention as 2024-02-g? Counts and label relation per match.
import os, sys, json, collections
import polars as pl
T26 = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
def cdir(ch):
    c = [d for d in os.listdir(T26) if d.startswith("basic_" + ch + ".")]
    return os.path.join(T26, c[0]) if c else None
def ch_frame(ch, cols):
    dp = cdir(ch)
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect() for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
re_ = ch_frame("round_end", ["round", "tick", "winner_team_code", "match_key"])
rs = ch_frame("round_start", ["round", "tick", "match_key"])
st = ch_frame("round_state", ["round", "tick", "event_type", "match_key"])
fe = st.filter(pl.col("event_type") == "round_freeze_end")
print("2026 rows: round_end", re_.height, "round_start", rs.height, "round_freeze_end", fe.height, flush=True)
print("2026 round_end labels not in round_start (same match):", re_.join(rs.select(["match_key", "round"]).unique().with_columns(pl.lit(1).alias("h")), on=["match_key", "round"], how="left").filter(pl.col("h").is_null()).height, flush=True)
print("2026 round_start labels not in round_end (same match):", rs.select(["match_key", "round"]).unique().join(re_.select(["match_key", "round"]).unique().with_columns(pl.lit(1).alias("h")), on=["match_key", "round"], how="left").filter(pl.col("h").is_null()).height, flush=True)
print("2026 round_start labels not in freeze_end (same match):", rs.select(["match_key", "round"]).unique().join(fe.select(["match_key", "round"]).unique().with_columns(pl.lit(1).alias("h")), on=["match_key", "round"], how="left").filter(pl.col("h").is_null()).height, flush=True)
print("2026 freeze_end labels not in round_end (same match):", fe.select(["match_key", "round"]).unique().join(re_.select(["match_key", "round"]).unique().with_columns(pl.lit(1).alias("h")), on=["match_key", "round"], how="left").filter(pl.col("h").is_null()).height, flush=True)
# tick relation: round_start label k vs round_end label k
j = rs.select(["match_key", "round", "tick"]).rename({"tick": "ts"}).join(re_.select(["match_key", "round", "tick"]).rename({"tick": "te"}), on=["match_key", "round"], how="inner")
print("2026 same-label pairs (round_start, round_end):", j.height, " round_end tick < round_start tick:", j.filter(pl.col("te") < pl.col("ts")).height, " round_end tick > round_start tick:", j.filter(pl.col("te") > pl.col("ts")).height, flush=True)
print("DONE", flush=True)
