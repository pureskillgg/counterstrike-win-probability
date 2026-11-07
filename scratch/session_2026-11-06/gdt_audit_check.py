"""Read-only audit check for the GDT objective. Reads round-state split membership, cache state columns, and training-split
outcome-free fields only. Does not read test or validation labels, does not train or score."""
import sys
import numpy as np
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
sys.path.insert(0, ROOT + "/scripts/model")
import gdt_grader as sg  # noqa: E402

MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
CACHE = ROOT + "/data/combined/cache_snapshot_2026_ext.parquet"
TRAIN, VAL, TEST = 0, 1, 2

# (A) split membership per match (round-state arrays; no label arrays loaded)
frames = []
for m in MAPS:
    z = np.load(f"{ROOT}/data/round_state/{m}_ext.npz")
    split = z["split"].astype(np.int64)
    midx = z["match_idx"].astype(np.int64)
    mk = z["match_keys"].astype(str)
    frames.append(pl.DataFrame({
        "mk": mk[midx], "split": split,
        "round": z["round"].astype(np.int64), "second": z["second"].astype(np.int64),
    }).with_columns(pl.lit(m).alias("map")))
allr = pl.concat(frames)
allr = allr.with_columns((pl.col("map") + "|" + pl.col("mk")).alias("mkey"))
print("rows by split:", allr.group_by("split").len().sort("split").rows())
pm = allr.group_by("mkey").agg(pl.col("split").n_unique().alias("ns"), pl.col("split").min().alias("smin"))
print("matches total:", pm.height, " matches with mixed split labels:", int(pm.filter(pl.col("ns") > 1).height))
single = pm.filter(pl.col("ns") == 1)
print("matches by split (single-split matches):", single.group_by("smin").len().sort("smin").rows())
tr_keys = set(single.filter(pl.col("smin") == TRAIN)["mkey"].to_list())
va_keys = set(single.filter(pl.col("smin") == VAL)["mkey"].to_list())
print("train/val match overlap:", len(tr_keys & va_keys))
raw_mk = allr.select(["map", "mk"]).unique()
print("unique match keys across maps (raw):", raw_mk["mk"].n_unique(), "of", raw_mk.height, "map-match pairs")
print("match keys containing '|':", int(allr.filter(pl.col("mk").str.contains(r"\|")).select("mkey").n_unique()))

# (B) schemas: names and dtypes only
cs = pl.read_parquet_schema(CACHE)
need = ["match_key", "round", "second", "alive_ct", "alive_t", "bomb_planted", "seconds_since_plant", "st_equip_ct", "st_equip_t"]
print("cache has needed cols:", [c for c in need if c in cs], "missing:", [c for c in need if c not in cs])
print("cache dtypes:", {c: str(cs[c]) for c in need if c in cs})
rs = pl.read_parquet_schema(ROOT + "/data/diagnostics/gdt_final_2026-11-06/rows_gdt.parquet")
print("grader rows columns:", list(rs.keys()))

# (C) groups over training-split rows (usable not applied: rows without a cache row would be dropped here, counted below)
tr = allr.filter(pl.col("split") == TRAIN).select(["mk", "round", "second"])
N0 = tr.height
base = pl.DataFrame({"match": tr["mk"], "round": tr["round"], "second": tr["second"],
                     "row": pl.Series(np.arange(N0, dtype=np.int64))})
cache = (pl.scan_parquet(CACHE)
         .select(["match_key", "round", "second", "alive_ct", "alive_t", "bomb_planted", "seconds_since_plant",
                  "st_equip_ct", "st_equip_t"])
         .rename({"match_key": "match"})
         .filter(pl.col("match").is_in(sorted(set(tr["mk"].to_list()))))
         .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
         .collect())
print("cache rows for train matches:", cache.height, " duplicate (match,round,second) keys:",
      cache.height - cache.unique(["match", "round", "second"]).height)
df = base.with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)).join(
    cache, on=["match", "round", "second"], how="left")
print("train rows:", N0, " joined rows:", df.height, " rows without cache row:", int(df["alive_ct"].null_count()))
df = df.filter(pl.col("alive_ct").is_not_null())
N = df.height
df = df.with_columns(
    pl.col("alive_ct").cast(pl.Int64), pl.col("alive_t").cast(pl.Int64),
    pl.col("bomb_planted").cast(pl.Int64).alias("planted"),
    pl.col("st_equip_ct").alias("equip_ct"), pl.col("st_equip_t").alias("equip_t"),
    pl.when(pl.col("bomb_planted") == 0).then(pl.lit(0.0)).otherwise(pl.col("seconds_since_plant").cast(pl.Float64)).alias("since_plant"),
    pl.lit(0).alias("y"),
)
print("planted==1 rows with null seconds_since_plant:", int(df.filter((pl.col("planted") == 1) & pl.col("since_plant").is_null()).height))
print("null st_equip_ct/t:", int(df["equip_ct"].null_count()), int(df["equip_t"].null_count()))
sit = sg.situations(df.select(["match", "round", "second", "alive_ct", "alive_t", "planted", "y", "since_plant",
                               "equip_ct", "equip_t", "row"]))
sit = sit.with_columns(pl.concat_str([pl.col("match"), pl.col("round").cast(pl.Utf8)] + [pl.col(c).cast(pl.Utf8) for c in sg.CELL],
                                     separator="|").alias("gkey"))
counts = sit.group_by("gkey").agg(pl.len().alias("n_g"))
G = counts.height
sit = sit.join(counts, on="gkey", how="left").sort("row")
ng = sit["n_g"].to_numpy().astype(np.float64)
w = N / (G * ng)
ess = float(w.sum() ** 2 / (w ** 2).sum())
print(f"N={N:,} G={G:,} mean seconds per group={N / G:.2f}")
print("n_g quantiles (min,p50,p90,p99,max):", np.percentile(ng, [0, 50, 90, 99, 100]).round(1).tolist())
print("w quantiles (min,p50,p99,max):", np.percentile(w, [0, 50, 99, 100]).round(3).tolist(), " mean w:", round(float(w.mean()), 9))
print(f"Kish effective sample size under uniform rows: {ess:,.0f} of {N:,} ({ess / N:.1%})")
print(f"singleton-group rows: {int((ng == 1).sum()):,}; rows with w > 5: {int((w > 5).sum()):,}")
print("groups with 1 second:", int((counts['n_g'] == 1).sum()), " groups with n_g>=500:", int((counts['n_g'] >= 500).sum()))
print("planted share among train rows:", round(float(df['planted'].mean()), 4))
