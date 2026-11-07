import polars as pl, sys, json
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
R = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/refute_oct/"
base = "G:/pureskill/csds-cs2/data/combined/"
pm = pl.read_csv(S + "batch_v4_per_match.csv")
rd = pl.read_csv(S + "batch_v4_rounds.csv").select(["match_key", pl.col("round_num").alias("round"), "steps"])
kl = pl.DataFrame({"match_key": pm["match_key"].to_list()}).lazy()
res = {}
for f in ["cache_snapshot_2026_oct23_31.parquet", "cache_snapshot_2026_ext.parquet", "cache_snapshot_2026.parquet"]:
    lf = pl.scan_parquet(base + f).select(["match_key", "round", "second"]).join(kl, on="match_key", how="semi")
    agg = lf.group_by(["match_key", "round"]).agg(pl.len().alias("cache_rows"), pl.col("second").n_unique().alias("cache_sec")).collect()
    m_present = agg["match_key"].n_unique()
    print(f, "batch matches", m_present, "rounds", agg.height, "rows", int(agg["cache_rows"].sum()), flush=True)
    j = rd.join(agg, on=["match_key", "round"], how="full", coalesce=True)
    only_v4 = j.filter(pl.col("cache_rows").is_null())
    only_cache = j.filter(pl.col("steps").is_null())
    both = j.filter(pl.col("cache_rows").is_not_null() & pl.col("steps").is_not_null())
    mism = both.filter(pl.col("cache_rows") != pl.col("steps"))
    print("  v4 rounds absent from cache:", only_v4.height, "seconds in those", int(only_v4["steps"].sum()), flush=True)
    print("  cache rounds not in v4:", only_cache.height, flush=True)
    print("  rounds in both with row count != v4 steps:", mism.height, "sum v4 steps there", int(mism["steps"].sum()), "sum cache rows", int(mism["cache_rows"].sum()), flush=True)
    miss_matches = only_v4["match_key"].unique().to_list()
    print("  v4 matches with any missing round:", len(miss_matches), flush=True)
    if only_v4.height:
        print("  missing-round detail (match_key, round, steps):", only_v4.select(["match_key", "round", "steps"]).head(30).rows(), flush=True)
    res[f] = {"matches": m_present, "rounds": agg.height, "rows": int(agg["cache_rows"].sum()),
              "v4_rounds_absent": only_v4.height, "v4_seconds_absent": int(only_v4["steps"].sum()),
              "cache_only_rounds": only_cache.height, "rounds_row_mismatch": mism.height,
              "missing_match_keys": miss_matches}
    if f.endswith("oct23_31.parquet"):
        mism.write_csv(R + "refute_oct23_31_row_mismatch.csv")
        only_v4.write_csv(R + "refute_oct23_31_v4_absent_rounds.csv")
json.dump(res, open(R + "refute_cache_batch.json", "w"), indent=1)
