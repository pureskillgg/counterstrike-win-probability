import polars as pl, sys, json, csv, glob, os
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/"
CK = S + "check_unchecked_oct/"
G = "G:/pureskill/csds-cs2/"
batch = set(r["match_key"] for r in csv.DictReader(open(CK + "batch_v4_per_match.csv", encoding="utf-8")))
hb = set(l.strip() for l in open(G + "data/holdout_b/holdout_keys.txt", encoding="utf-8") if l.strip())
# 1. sprint validation keys
vk = pl.read_parquet("C:/Users/<user>/sprint/matrix_trainctr/val_keys.parquet")
col = [c for c in vk.columns if "key" in c.lower()][0]
vks = set(vk[col].to_list())
print("val_keys", vk.height, "batch overlap", len(vks & batch), "HB overlap", len(vks & hb), flush=True)
# 2. model feature counts
for d in [G + "models/", "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/models/"]:
    counts = {}
    for f in sorted(glob.glob(d + "*.meta.json")):
        try:
            m = json.load(open(f, encoding="utf-8"))
            ft = m.get("features")
            if isinstance(ft, str):
                import ast; ft = ast.literal_eval(ft)
            n = len(ft) if ft is not None else None
            counts[os.path.basename(f)] = n
        except Exception as e:
            counts[os.path.basename(f)] = "ERR"
    print(d, "feature counts", Counter(counts.values()), flush=True)
    print("  77-feature metas:", [k for k, v in counts.items() if v == 77], flush=True)
    print("  83-feature metas:", [k for k, v in counts.items() if v == 83][:10], flush=True)
# 3. cache_pos_2026 for batch keys: rows, per-round rows vs steps, tick stride
v4 = {}
for r in csv.DictReader(open(CK + "batch_v4_rounds.csv", encoding="utf-8")):
    v4[(r["match_key"], int(float(r["round_num"])))] = int(r["steps"])
kl = pl.DataFrame({"match_key": list(batch)}).lazy()
cp = pl.scan_parquet(G + "data/combined/cache_pos_2026.parquet").select(["match_key", "round", "tick"]).join(kl, on="match_key", how="semi").collect()
print("cache_pos_2026 batch rows", cp.height, "keys", cp["match_key"].n_unique(), flush=True)
per = cp.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("tick").min().alias("tmin"), pl.col("tick").max().alias("tmax")).collect() if False else cp.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
rr = per.with_columns(pl.struct(["match_key", "round"]).map_elements(lambda s: v4.get((s["match_key"], int(s["round"]))), return_dtype=pl.Int64).alias("steps"))
print("rounds in cache_pos", rr.height, "matched to v4", rr.filter(pl.col("steps").is_not_null()).height, flush=True)
rr2 = rr.filter(pl.col("steps").is_not_null())
print("ratio rows/steps overall", round(rr2["n"].sum() / rr2["steps"].sum(), 4), "rows == ceil(steps/2) for", int((rr2["n"] == (rr2["steps"] + 1) // 2).sum()), "of", rr2.height, flush=True)
ticks = cp.sort(["match_key", "round", "tick"]).with_columns((pl.col("tick") - pl.col("tick").shift(1).over(["match_key", "round"])).alias("dt")).filter(pl.col("dt").is_not_null())
print("tick diffs top", ticks.group_by("dt").len().sort("len", descending=True).head(5).rows(), flush=True)
# 4. plants coverage for batch
pl_files = sorted(glob.glob(G + "data/positions_1hz/_plants/*"))
print("plants files", len(pl_files), [os.path.basename(p) for p in pl_files][-6:], flush=True)
ks = set()
for f in pl_files:
    if f.endswith(".parquet"):
        try:
            ks.update(pl.scan_parquet(f).select("match_key").unique().collect()["match_key"].to_list())
        except Exception as e:
            print("err", os.path.basename(f), str(e)[:80], flush=True)
print("plants distinct keys", len(ks), "batch covered", len(batch & ks), "batch missing", len(batch - ks), flush=True)
