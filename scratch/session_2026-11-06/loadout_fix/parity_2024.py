import glob
import polars as pl

RIFLE = (7, 8, 10, 13, 16, 39, 60); M4 = (16, 60); AK = (7,); AWP = (9,)
SNIPER_OTHER = (11, 38, 40); SMG = (17, 19, 23, 24, 26, 33, 34); HEAVY = (14, 25, 27, 28, 29, 35)
FIELDS = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
          "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy", "equip"]
DIFF = ["hp", "rifle", "equip", "armor_n", "awp", "smg"]
KEYS = ["match_key", "round", "tick"]

files = sorted(glob.glob("G:/pureskill/csds-cs2/data/status_1hz_2024/*.parquet"))
st = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
st = st.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
print("status files", len(files), "rows", st.height, "matches", st["match_key"].n_unique())
print("side values", st["side"].unique().sort().to_list())

inv = pl.col("inv_primary").fill_null(0).cast(pl.Int64)
agg = st.group_by(KEYS + ["side"]).agg([
    pl.col("health").fill_null(0).cast(pl.Float64).sum().alias("hp"),
    (pl.col("armor").fill_null(0) > 0).cast(pl.Int64).sum().alias("armor_n"),
    pl.col("armor").fill_null(0).cast(pl.Float64).sum().alias("armor_sum"),
    pl.col("has_helmet").fill_null(False).cast(pl.Int64).sum().alias("helmet_n"),
    pl.col("has_defuser").fill_null(False).cast(pl.Int64).sum().alias("kit_n"),
    pl.col("is_defusing").fill_null(False).cast(pl.Int64).sum().alias("defusing"),
    (inv == AK[0]).cast(pl.Int64).sum().alias("ak"),
    inv.is_in(list(M4)).cast(pl.Int64).sum().alias("m4"),
    (inv == AWP[0]).cast(pl.Int64).sum().alias("awp"),
    inv.is_in(list(RIFLE)).cast(pl.Int64).sum().alias("rifle"),
    inv.is_in(list(SNIPER_OTHER)).cast(pl.Int64).sum().alias("sniper_other"),
    inv.is_in(list(SMG)).cast(pl.Int64).sum().alias("smg"),
    inv.is_in(list(HEAVY)).cast(pl.Int64).sum().alias("heavy"),
    (inv == 0).cast(pl.Int64).sum().alias("no_primary"),
    pl.col("inv_flashbang").fill_null(0).cast(pl.Float64).sum().alias("flash"),
    pl.col("inv_smokegrenade").fill_null(0).cast(pl.Float64).sum().alias("smoke"),
    pl.col("inv_hegrenade").fill_null(0).cast(pl.Float64).sum().alias("he"),
    (pl.col("inv_molotov").fill_null(0) + pl.col("inv_incgrenade").fill_null(0)).cast(pl.Float64).sum().alias("fire"),
    pl.col("inv_decoy").fill_null(0).cast(pl.Float64).sum().alias("decoy"),
    pl.col("equipment_value_calc").fill_null(0).cast(pl.Float64).sum().alias("equip"),
    pl.len().alias("n_rows"),
])
ct = agg.filter(pl.col("side") == 3).drop("side").rename({**{f: f"st_{f}_ct" for f in FIELDS}, "n_rows": "n_ct"})
t = agg.filter(pl.col("side") == 2).drop("side").rename({**{f: f"st_{f}_t" for f in FIELDS}, "n_rows": "n_t"})
wide = ct.join(t, on=KEYS, how="full", coalesce=True)
for f in DIFF:
    wide = wide.with_columns((pl.col(f"st_{f}_ct").fill_null(0) - pl.col(f"st_{f}_t").fill_null(0)).alias(f"st_{f}_diff"))
wide = wide.fill_null(0)
print("wide ticks (round,tick with any status row)", wide.height)

snap_cols = KEYS + ["second", "alive_ct", "alive_t"] + [f"st_{f}_{s}" for f in FIELDS for s in ("ct", "t")] + [f"st_{f}_diff" for f in DIFF]
snap = pl.read_parquet("G:/pureskill/csds-cs2/data/combined/cache_snapshot_2024.parquet", columns=snap_cols)
snap = snap.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
print("snapshot rows", snap.height, "matches", snap["match_key"].n_unique())
print("snapshot match_key example", snap["match_key"][0], "| status example", st["match_key"][0])
j = snap.join(wide, on=KEYS, how="left", suffix="_srv")
print("snapshot rows with no status tick match (left join, no srv cols):", j.filter(pl.col("n_ct").is_null() & pl.col("n_t").is_null()).height)
j = j.fill_null(0)

names = [f"st_{f}_{s}" for f in FIELDS for s in ("ct", "t")] + [f"st_{f}_diff" for f in DIFF]
print("--- mismatches per feature (|snapshot - serving formula| > 1e-6) ---")
bad_total = {}
for n in names:
    mm = j.filter((pl.col(n) - pl.col(n + "_srv")).abs() > 1e-6).height
    bad_total[n] = mm
print({k: v for k, v in bad_total.items() if v})
print("features with zero mismatches:", sum(1 for v in bad_total.values() if v == 0), "of", len(names))
print("--- alive counts vs status rows per side ---")
print("alive_ct == n_ct:", j.filter(pl.col("alive_ct") == pl.col("n_ct")).height, "of", j.height)
print("alive_t == n_t:", j.filter(pl.col("alive_t") == pl.col("n_t")).height, "of", j.height)
print("alive_ct - n_ct distribution:")
print((j["alive_ct"] - j["n_ct"]).value_counts().sort("alive_ct" if False else "count", descending=True).head(8))
