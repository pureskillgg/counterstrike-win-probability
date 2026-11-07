import polars as pl
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
P = "C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/positions_validation"
G = "G:/pureskill/csds-cs2"
rows = pl.read_parquet(f"{P}/validation_gdt_rows.parquet")
print("ROWS file:", rows.height, rows.columns, rows.dtypes)
print("  matches", rows["match"].n_unique(), "rounds", rows.select(["match","round"]).n_unique(), "dup keys", rows.select(["match","round","second"]).is_duplicated().sum())
print("  sample match", rows["match"][0], "second dtype", rows.schema["second"])
rows2 = pl.read_parquet(f"{P}/validation_gdt_rows_recheck.parquet")
print("RECHECK same shape", rows2.shape, "equal frame:", rows.equals(rows2))
for f in ["base_seed0","full_trainctr_seed0","full_oldctr_seed0"]:
    a = pl.read_parquet(f"{P}/{f}.parquet")
    print(f, a.height, a.columns, a.dtypes[:4], "matches", a["match_key"].n_unique())
r57 = pl.read_parquet(f"{S}/gdt57v83/val_rescored_v6_drop_heavy_57.parquet")
print("57 file", r57.height, r57.columns, r57.dtypes, "matches", r57["match_key"].n_unique(), "dupkeys", r57.select(["match_key","round","second"]).is_duplicated().sum())
print("  sample match_key", r57["match_key"][0])
vk = pl.read_parquet(f"{G}/data/sprint/matrix_trainctr/val_keys.parquet")
print("val_keys", vk.height, vk.columns, vk.dtypes)
s83 = pl.read_parquet(f"{G}/data/sprint/run/gpu_fast/preds/full_d4_e0.03_m20_s0.parquet")
print("sprint83 pred", s83.height, s83.columns, s83.dtypes)
print("  matches", s83["match_key"].n_unique() if "match_key" in s83.columns else "NA")
