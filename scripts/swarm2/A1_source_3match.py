# A1: row counts (and full-row equality) for 3 seeded matches: tome channel pages vs raw E: parquet.
# Tome: polars lazy scan over every dataframe_* page of a channel, predicate match_key == key (pushdown).
# Raw: E:\csds_final\csds\2026\10\<DD>\<match_id>\<channel> (extensionless parquet).
import os, random, glob, json, sys, time
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
E = r"E:\csds_final\csds\2026\10"
t0 = time.time()
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"), columns=["key"])
keys = sorted(h.get_column("key").to_list())
rng = random.Random(20261013)
picked = rng.sample(keys, 3)
chans = sorted(d for d in os.listdir(T) if d != "header" and not d.startswith("."))
out = {}
for key in picked:
    dd, mid = key.split("/")[3], key.split("/")[4]
    base = os.path.join(E, dd, mid)
    rows_report = []
    mismatches = []
    for c in chans:
        raw_name = c.split(".")[0].replace("basic_", "", 1)
        pages = sorted(glob.glob(os.path.join(T, c, "dataframe_*")))
        tome_lf = pl.concat([pl.scan_parquet(p) for p in pages], how="diagonal_relaxed").filter(pl.col("match_key") == key)
        tome_n = tome_lf.select(pl.len()).collect().item()
        raw_path = os.path.join(base, raw_name)
        if os.path.exists(raw_path):
            raw_lf = pl.scan_parquet(raw_path)
            raw_n = raw_lf.select(pl.len()).collect().item()
            raw_cols = raw_lf.collect_schema().names()
        else:
            raw_n, raw_cols = None, None
        tome_cols = pl.concat([pl.scan_parquet(p) for p in pages[:1]]).collect_schema().names() if pages else []
        extra_tome = sorted(set(tome_cols) - set(raw_cols or []) - {"match_key"}) if raw_cols else None
        extra_raw = sorted(set(raw_cols or []) - set(tome_cols)) if raw_cols else None
        ok = (raw_n == tome_n)
        # full content comparison (only if counts match and both nonzero or both zero)
        content = None
        if raw_cols is not None and tome_n > 0 and raw_n == tome_n:
            cols = [x for x in raw_cols if x in tome_cols]
            a = tome_lf.select(cols).collect()
            b = raw_lf.select(cols).collect()
            try:
                a2 = a.sort(cols); b2 = b.sort(cols)
                content = a2.equals(b2)
                if not content:
                    # try casting to string for mismatch diagnosis
                    content = "NOT_EQUAL"
            except Exception as e:
                content = f"cmp_error:{type(e).__name__}"
        rows_report.append(dict(channel=raw_name, tome_rows=tome_n, raw_rows=raw_n, count_equal=ok,
                                content=content, extra_in_tome=extra_tome, extra_in_raw=extra_raw))
        if not ok or content not in (True, None):
            mismatches.append(raw_name)
    out[key] = dict(rows=rows_report, mismatches=mismatches)
    print("MATCH", key, "channels:", len(rows_report), "count mismatches or content issues:", mismatches or "none", flush=True)
    for r in rows_report:
        print("  ", r["channel"], r["tome_rows"], r["raw_rows"], r["count_equal"], r["content"], r["extra_in_tome"], r["extra_in_raw"], flush=True)
    print("  elapsed", round(time.time()-t0,1), flush=True)
json.dump(out, open(r"G:\pureskill\csds-cs2\logs\swarm2\A1_source_3match.json", "w"), indent=1, default=str)
