# verify_A2_4_rawsample.py: read-only check of the raw 2024 tail copy for a sample of build-9842 and build-9961 matches.
# Per match: round-1 rows in raw round_state (event types), raw round_start/round_end min label, raw tick-channel min tick
# vs the round-1 end bundle tick. Reads small channel files only (round_state, round_start, round_end; tick for first 25 each).
import os
import polars as pl
pl.Config.set_tbl_rows(30)
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
RAWBASE = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02"
hdr = pl.scan_parquet(os.path.join(TOME, "header", "dataframe_00000")).select(["key", "match_id", "build_num"]).collect()
hdr = hdr.with_columns(pl.col("key").str.split("/").list.get(3).alias("day"), pl.col("build_num").cast(pl.Utf8).alias("build"))
rows = []
for b, n in [("9842", 60), ("9961", 60)]:
    sub = hdr.filter(pl.col("build") == b).sample(n=n, seed=0) if hdr.filter(pl.col("build") == b).height >= n else hdr.filter(pl.col("build") == b)
    for r in sub.iter_rows(named=True):
        rows.append((b, r["day"], r["match_id"]))
stats = {}
present = {"9842": 0, "9961": 0}
for b, day, mid in rows:
    d = os.path.join(RAWBASE, day, mid)
    if not os.path.isdir(d):
        stats.setdefault((b, "folder_missing"), 0)
        stats[(b, "folder_missing")] += 1
        continue
    rs_p = os.path.join(d, "round_state")
    if not os.path.exists(rs_p):
        stats.setdefault((b, "no_round_state"), 0)
        stats[(b, "no_round_state")] += 1
        continue
    present[b] += 1
    st = pl.read_parquet(rs_p, columns=["round", "tick", "event_type"])
    r1 = st.filter(pl.col("round") == 1)
    rs_min_label = pl.read_parquet(os.path.join(d, "round_start"), columns=["round"])["round"].min() if os.path.exists(os.path.join(d, "round_start")) else None
    re_min_label = pl.read_parquet(os.path.join(d, "round_end"), columns=["round"])["round"].min() if os.path.exists(os.path.join(d, "round_end")) else None
    evs = tuple(sorted(set(r1["event_type"].to_list())))
    key = (b, "r1_events=" + ",".join(evs) if evs else "r1_events=none", f"rs_min={rs_min_label}", f"re_min={re_min_label}")
    stats[key] = stats.get(key, 0) + 1
    ob = r1.filter(pl.col("event_type") == "round_officially_ended")["tick"].to_list()
    if os.path.exists(os.path.join(d, "tick")) and rows.index((b, day, mid)) < 25 * 2:
        tk = pl.read_parquet(os.path.join(d, "tick"), columns=["round", "tick"])
        tmin = tk["tick"].min()
        key2 = (b, "tick_min_minus_r1_bundle_s", round((tmin - ob[0]) / 64, 1) if ob else None)
        stats[key2] = stats.get(key2, 0) + 1
print("sampled per build:", {b: sum(1 for x in rows if x[0] == b) for b in ["9842", "9961"]}, " with round_state present:", present, flush=True)
for k, v in sorted(stats.items(), key=lambda kv: (kv[0][0], str(kv[0][1:]))):
    print(k, v, flush=True)
print("DONE", flush=True)
