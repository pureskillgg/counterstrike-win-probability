import glob, os
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
d = os.path.join(ROOT, "basic_item_equip.2026-09-18,2026-10-08")
df = pq.read_table(os.path.join(d, "dataframe_00000"), columns=["match_key", "round", "tick", "player_id"])
print("match_key sample:", df.column("match_key").slice(0, 3).to_pylist())
print("distinct match_key in page0:", pc.count_distinct(df.column("match_key")).as_py())
print("null match_key page0:", df.column("match_key").null_count)
ht = pq.read_table(os.path.join(ROOT, "header", "dataframe_00000"))
print("header rows:", ht.num_rows)
print("header key sample:", ht.column("key").slice(0, 3).to_pylist())
print("header match_id sample:", ht.column("match_id").slice(0, 3).to_pylist())
print("header distinct key:", pc.count_distinct(ht.column("key")).as_py())
print("header distinct match_id:", pc.count_distinct(ht.column("match_id")).as_py())
ks = pq.read_table(os.path.join(ROOT, "header", "keyset_00000"))
print("header keyset rows:", ks.num_rows, ks.column("_").slice(0,2).to_pylist())
keys_header = set(ht.column("key").to_pylist())
keys_chan_p0 = set(df.column("match_key").to_pylist())
print("channel page0 keys in header.key:", len(keys_chan_p0 & keys_header), "of", len(keys_chan_p0))
ksk = set(pq.read_table(os.path.join(d, "keyset_00000")).column("_").to_pylist())
print("channel keyset0 size:", len(ksk), "page0 match_keys in keyset0:", len(keys_chan_p0 & ksk), "of", len(keys_chan_p0))
mk_sample = list(keys_chan_p0)[:2]
print("chan mk sample2:", mk_sample)
