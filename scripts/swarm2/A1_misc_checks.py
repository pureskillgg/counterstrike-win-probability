# A1: header date range, network_protocol, page-schema drift effect on naive scan, rounds in the 90 disagreeing matches.
import os, glob
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"), columns=["network_protocol", "bomb_time", "match_date", "key"])
print("match_date min/max:", h.get_column("match_date").min(), h.get_column("match_date").max(), "| parse ok:", h.get_column("match_date").str.slice(0,10).n_unique(), "distinct days")
print("network_protocol: dtype", h.schema["network_protocol"], "distinct", h.get_column("network_protocol").n_unique(), "nulls", h.get_column("network_protocol").null_count(), "values", h.get_column("network_protocol").unique().to_list()[:3])
print("bomb_time: dtype", h.schema["bomb_time"], "nulls", h.get_column("bomb_time").null_count())
# downstream: naive scan of item_equip pages (as a reader would do with a glob)
pages = sorted(glob.glob(os.path.join(T, "basic_item_equip.2026-09-18,2026-10-08", "dataframe_*")))
try:
    n = pl.scan_parquet(pages).select(pl.len()).collect().item()
    print("naive scan_parquet(glob) item_equip: OK rows", n)
except Exception as e:
    print("naive scan_parquet(glob) item_equip FAILED:", type(e).__name__, str(e).splitlines()[0][:200])
for c in ["basic_item_pickup.2026-09-18,2026-10-08", "basic_player_hurt.2026-09-18,2026-10-08"]:
    pages = sorted(glob.glob(os.path.join(T, c, "dataframe_*")))
    try:
        n = pl.scan_parquet(pages).select(pl.len()).collect().item()
        print("naive scan", c.split(".")[0], "OK rows", n)
    except Exception as e:
        print("naive scan", c.split(".")[0], "FAILED:", type(e).__name__, str(e).splitlines()[0][:200])
# which pages carry which item_equip schema
import pyarrow.parquet as pq
for p in sorted(glob.glob(os.path.join(T, "basic_item_equip.2026-09-18,2026-10-08", "dataframe_*"))):
    s = pq.read_schema(p)
    types = dict(zip(s.names, [str(t) for t in s.types]))
    print("  item_equip", os.path.basename(p), "ncols", len(s.names), "has def_index", "def_index" in types, "player_id_fixed", types.get("player_id_fixed"), "first col", s.names[0])
for c in ["basic_item_pickup.2026-09-18,2026-10-08", "basic_player_hurt.2026-09-18,2026-10-08"]:
    for p in sorted(glob.glob(os.path.join(T, c, "dataframe_*"))):
        s = pq.read_schema(p)
        types = dict(zip(s.names, [str(t) for t in s.types]))
        k = "attacker_id_fixed" if "hurt" in c else "player_id_fixed"
        print("  ", c.split(".")[0], os.path.basename(p), "ncols", len(s.names), k, types.get(k))
# rounds in the 90 disagreeing matches
d = pl.read_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_header_vs_rounds.csv", infer_schema_length=0)
bad_keys = d.filter(pl.col("round_match") == "false").get_column("key").to_list()
pages = sorted(glob.glob(os.path.join(T, "basic_round_end.2026-09-18,2026-10-08", "dataframe_*")))
re_ = pl.concat([pl.scan_parquet(p) for p in pages], how="diagonal_relaxed").select("match_key").collect()
n_bad = re_.filter(pl.col("match_key").is_in(bad_keys)).height
print("round_end rows in the 90 disagreeing matches:", n_bad, "of", re_.height, f"({100*n_bad/re_.height:.2f}%)")
short_keys = d.filter((pl.col("round_match") == "false") & (pl.col("n").cast(pl.Int64) < pl.col("S").cast(pl.Int64))).get_column("key").to_list()
print("round_end rows in the 80 shortfall matches:", re_.filter(pl.col("match_key").is_in(short_keys)).height)
print("total header S (sum t+ct):", int(h.height and d.select((pl.col("S").cast(pl.Int64))).sum().item()), "| total round_end rows:", re_.height)
