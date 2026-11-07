"""E1 verification: check the E1 cache against the G: tome metadata and profile the inputs.

Reads: G: tome parquet metadata (row counts) and logs\\swarm\\E1\\cache (built by E1_cache.py).
Reads no E: files. Writes only a text log under logs\\swarm\\E1\\verify_inputs.txt.
Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I -X utf8 scripts\\swarm\\E1_verify_inputs.py
"""

import glob
import os

import polars as pl
import pyarrow.parquet as pq

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
CACHE = os.path.join(ROOT, "logs", "swarm", "E1", "cache")
OUT_TXT = os.path.join(ROOT, "logs", "swarm", "E1", "verify_inputs.txt")
CHANNELS = ["item_equip", "item_pickup", "item_dropped", "item_refund", "player_info",
            "round_state", "round_start", "round_end", "header"]

lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def pages(name):
    if name == "header":
        return sorted(glob.glob(os.path.join(TOME, "header", "dataframe_*")))
    d = [x for x in glob.glob(os.path.join(TOME, "basic_%s.*" % name)) if os.path.isdir(x)]
    assert len(d) == 1, (name, d)
    return sorted(glob.glob(os.path.join(d[0], "dataframe_*")))


out("== 1. cache row counts vs G: tome page metadata")
for ch in CHANNELS:
    tome_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in pages(ch))
    cache_rows = pq.ParquetFile(os.path.join(CACHE, ch + ".parquet")).metadata.num_rows
    out("%-13s tome_pages=%d tome_rows=%d cache_rows=%d match=%s" % (
        ch, len(pages(ch)), tome_rows, cache_rows, tome_rows == cache_rows))

rs = pl.read_parquet(os.path.join(CACHE, "round_state.parquet"))
out("\n== 2. round_state event types (count of rows)")
out(rs.group_by("event_type").len().sort("len", descending=True).head(60))

fe = rs.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).agg(
    pl.len().alias("n"))
out("\nround_freeze_end rows per (match, round): distribution of n")
out(fe.group_by("n").len().sort("n"))
out("distinct (match, round) with freeze_end:", fe.height)

ri = pl.read_parquet(os.path.join(CACHE, "round_start.parquet"))
out("\nround_start rows per (match, round): distribution")
out(ri.group_by(["match_key", "round"]).agg(pl.len().alias("n")).group_by("n").agg(pl.len().alias("n_pairs")).sort("n"))
out("round_state event_type full list:", sorted(rs["event_type"].unique().to_list()))

eq = pl.read_parquet(os.path.join(CACHE, "item_equip.parquet"))
out("\n== 3. item_equip weapon_type_code by item (top 40 by rows)")
out(eq.group_by(["item", "weapon_type_code"]).len().sort("len", descending=True).head(40))

out("\n== 4. item_equip player_team_code vs player_info team_code (per match, round, player_id)")
pi = pl.read_parquet(os.path.join(CACHE, "player_info.parquet"))
pi_k = pi.select(["match_key", "round", "player_id", "team_code"]).unique()
out("player_info rows", pi.height, "unique (match,round,player_id)", pi_k.height)
dup_pi = pi.group_by(["match_key", "round", "player_id"]).len().filter(pl.col("len") > 1)
out("player_info duplicated keys:", dup_pi.height)
j = eq.select(["match_key", "round", "player_id", "player_team_code"]).join(
    pi_k, on=["match_key", "round", "player_id"], how="left")
out("item_equip rows", j.height, "unmapped (no player_info)", j.filter(pl.col("team_code").is_null()).height)
jj = j.filter(pl.col("team_code").is_not_null())
out("agreement player_team_code == team_code:", round(float((jj["player_team_code"] == jj["team_code"]).mean()), 6))
out("team_code values in player_info:", pi.group_by("team_code").len().sort("team_code").to_dicts())

out("\n== 5. match and round universe")
hd = pl.read_parquet(os.path.join(CACHE, "header.parquet"))
out("header rows", hd.height, "is_wingman:", hd.group_by("is_wingman").len().to_dicts())
out("max_rounds:", hd.group_by("max_rounds").len().sort("max_rounds").to_dicts())
out("match_type:", hd.group_by("match_type").len().to_dicts())
re_ = pl.read_parquet(os.path.join(CACHE, "round_end.parquet"))
out("round_end rows", re_.height, "matches", re_["match_key"].n_unique())
out("winner_team_code:", re_.group_by("winner_team_code").len().sort("winner_team_code").to_dicts())
out("player_count in round_end:", re_.group_by("player_count").len().sort("player_count").to_dicts())

os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
out("wrote", OUT_TXT)
