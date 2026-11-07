import polars as pl, sys, json
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/"
base = "G:/pureskill/csds-cs2/data/"
pm = pl.read_csv(S + "batch_v4_per_match.csv")
rd = pl.read_csv(S + "batch_v4_rounds.csv")
print("v4 per-match rows", pm.height, "sum n_rounds", pm["n_rounds"].sum(), "sum n_seconds", pm["n_seconds"].sum())
print("v4 rounds rows", rd.height, "sum steps", rd["steps"].sum())
print("split counts", pm.group_by("match_split").len().sort("match_split").rows())
keys = pm.select("match_key").unique()
out = {}
for src in ["positions_1hz", "status_1hz"]:
    tot_m = set(); tot_r = 0; tot_s = 0; per = {}
    for d in ["21", "22", "23", "24", "25"]:
        f = f"{base}{src}/2026-10-{d}.parquet"
        lf = pl.scan_parquet(f).select(["match_key", "round", "second"])
        g = (lf.join(pl.from_pandas(keys.to_pandas()).lazy() if False else lf.limit(0), on="match_key", how="inner").limit(0))
        # filter to batch keys via semi join
        kl = pl.DataFrame({"match_key": keys["match_key"].to_list()}).lazy()
        sub = lf.join(kl, on="match_key", how="semi").group_by(["match_key", "round"]).agg(
            pl.col("second").n_unique().alias("n_sec"), pl.col("second").min().alias("smin"), pl.col("second").max().alias("smax"))
        sub = sub.collect()
        m = sub["match_key"].unique().to_list()
        tot_m.update(m)
        per[d] = {"matches": len(m), "rounds": sub.height, "seconds": int(sub["n_sec"].sum())}
        print(src, d, per[d], flush=True)
    out[src] = {"distinct_matches": len(tot_m), "per_file": per}
    print(src, "TOTAL distinct batch matches in files:", len(tot_m), flush=True)
    print("batch keys not found in", src, ":", len(set(keys["match_key"].to_list()) - tot_m), flush=True)
json.dump(out, open(S + "refute_local_pos_status.json", "w"), indent=1)
