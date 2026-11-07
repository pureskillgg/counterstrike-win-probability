# A2: final checks: duplicate groups by folder day; build share by folder day; excluded matches; no-round_start match; 2026 empty channels.
import os, sys, json, csv, collections
import polars as pl
import pyarrow.parquet as pq
BASE = r"G:\pureskill\csds-cs2"
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
hdr = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"), columns=["key", "match_date", "map_name", "build_num", "t_starters_score_final", "ct_starters_score_final", "number_of_points", "match_id"]).rename({"key": "match_key"})
hdr = hdr.with_columns(pl.col("match_key").str.split("/").list.get(3).alias("day"), pl.col("match_date").str.slice(0, 10).alias("mdate"))
# build share by folder day
print("build_num by folder day:", hdr.group_by(["day", "build_num"]).len().sort(["day", "build_num"]).rows(), flush=True)
# match month by folder day
print("match_date month by folder day (counts 2023-12/2024-01/2024-02):")
mm = hdr.with_columns(pl.col("mdate").str.slice(0, 7).alias("mon")).group_by(["day", "mon"]).len().sort(["day", "mon"])
print(mm.rows(), flush=True)
# duplicate groups by folder day
sig = hdr.with_columns(pl.concat_str([pl.col("mdate"), pl.col("map_name"), pl.col("t_starters_score_final").cast(pl.Utf8), pl.col("ct_starters_score_final").cast(pl.Utf8), pl.col("number_of_points").cast(pl.Utf8)], separator="|").alias("sig"))
dg = sig.group_by("sig").agg(pl.col("day").alias("days"), pl.col("match_id").alias("mids"), pl.len().alias("n")).filter(pl.col("n") > 1)
print("dup groups:", dg.height, "excess copies:", int((dg["n"] - 1).sum()), flush=True)
cross = collections.Counter()
for r in dg.iter_rows(named=True):
    cross["same_folder_day" if len(set(r["days"])) == 1 else "cross_folder_day"] += 1
print("dup groups within one folder day vs across folder days:", dict(cross), flush=True)
print("dup group size distribution:", dg.group_by("n").len().sort("n").rows(), flush=True)
# excluded matches (prescan not ok, day != 01)
rows = list(csv.DictReader(open(os.path.join(BASE, "logs", "prescan_2024.csv"), encoding="utf-8-sig")))
exc = [r for r in rows if r["status"] != "ok" and r["day"] != "01"]
print("excluded non-day01 folders:", len(exc))
for r in exc: print("  ", r["day"], r["match_folder"], r["status"], r["detail"][:120], flush=True)
# is any excluded match present in the 2024 tome header?
ids = set(hdr["match_id"].to_list())
print("excluded ids present in tome:", [r["match_folder"] for r in exc if r["match_folder"] in ids], flush=True)
# no-round_start match
mk = "csds/2024/02/12/m_56c0f3b373d8adecd20c6222/csds"
h = hdr.filter(pl.col("match_key") == mk).to_dicts()
print("match m_56c0f3b3 header:", {k: h[0][k] for k in ["mdate", "map_name", "build_num", "t_starters_score_final", "ct_starters_score_final"]} if h else None, flush=True)
st = pl.read_parquet(os.path.join(TOME, "basic_round_state.2023-12-28,2024-02-25", "dataframe_00000"), columns=["match_key", "event_type"]).filter(pl.col("match_key") == mk)
print("match m_56c0f3b3 round_state event types:", st.group_by("event_type").len().sort("len", descending=True).rows()[:10], flush=True)
# 2026 empty channels
T26 = os.path.join(BASE, "tomes", "2026-10-01_to_08", "tome", "csds")
for ch in ["basic_bot_takeover", "basic_player_name"]:
    cands = [d for d in os.listdir(T26) if d.startswith(ch + ".")]
    for d in cands:
        p = os.path.join(T26, d, "dataframe_00000")
        if os.path.isfile(p):
            print("2026", ch, "rows", pq.ParquetFile(p).metadata.num_rows, "cols", len(pq.read_schema(p).names), flush=True)
print("DONE", flush=True)
