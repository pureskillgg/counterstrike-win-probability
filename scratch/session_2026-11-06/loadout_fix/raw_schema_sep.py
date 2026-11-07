import polars as pl
d = "E:/csds_final/csds/2026/09/01/01f19adb-166f-4666-95fb-fc75f186c486"
pi = pl.read_parquet(f"{d}/player_info")
print("player_info shape", pi.shape)
print("player_info cols", pi.columns)
print(pi.head(12))
ps = pl.read_parquet(f"{d}/player_status")
print("player_status shape", ps.shape)
print("player_status cols", ps.columns)
rs = pl.read_parquet(f"{d}/round_state")
print("round_state cols", rs.columns)
print("round_state event types", rs["event_type"].unique().to_list() if "event_type" in rs.columns else None)
