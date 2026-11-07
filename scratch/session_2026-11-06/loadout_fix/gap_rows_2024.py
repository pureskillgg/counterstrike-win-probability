import glob
import polars as pl
KEYS = ["match_key", "round", "tick"]
files = sorted(glob.glob("G:/pureskill/csds-cs2/data/status_1hz_2024/*.parquet"))
st = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "second", "side", "player_id", "health"]) for f in files], how="vertical_relaxed")
st = st.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
snap = pl.read_parquet("G:/pureskill/csds-cs2/data/combined/cache_snapshot_2024.parquet",
                       columns=KEYS + ["second", "alive_ct", "alive_t", "st_hp_ct", "st_hp_t", "st_no_primary_ct"])
snap = snap.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
stk = st.select(KEYS).unique()
stk = stk.with_columns(pl.lit(1).alias("in_status"))
j = snap.join(stk, on=KEYS, how="left")
gap = j.filter(pl.col("in_status").is_null())
print("gap rows", gap.height)
print("gap rows with st_hp_ct > 0:", gap.filter(pl.col("st_hp_ct") > 0).height)
print("gap rows with alive_ct > 0:", gap.filter(pl.col("alive_ct") > 0).height)
print("gap matches", gap["match_key"].n_unique(), "gap rounds", gap.select(["match_key", "round"]).unique().height)
# does the round exist in status at all?
rk = st.select(["match_key", "round"]).unique().with_columns(pl.lit(1).alias("round_in_status"))
g2 = gap.join(rk, on=["match_key", "round"], how="left")
print("gap rows whose round is absent from status entirely:", g2.filter(pl.col("round_in_status").is_null()).height)
print("gap rows whose match is absent from status entirely:", gap.join(st.select("match_key").unique().with_columns(pl.lit(1).alias("m")), on="match_key", how="left").filter(pl.col("m").is_null()).height)
print("sample gap rows:")
print(gap.select(KEYS + ["second", "alive_ct", "alive_t", "st_hp_ct", "st_hp_t", "st_no_primary_ct"]).head(8))
# for the first sample, show status ticks of that round
s0 = gap.select(["match_key", "round"]).row(0)
print("sample round", s0)
print(st.filter((pl.col("match_key") == s0[0]) & (pl.col("round") == s0[1])).group_by("tick").agg(pl.len().alias("n")).sort("tick").head(12))
