import glob, os, sys, collections, pyarrow.parquet as pq
tome = os.path.join(sys.argv[1], "tome", "csds", "basic_round_end.2026-09-18,2026-10-08")
t = pq.read_table(sorted(glob.glob(os.path.join(tome, "dataframe_*")))[0], columns=["player_count", "win_reason_code", "winner_team_code"])
print("rows:", t.num_rows)
print("player_count:", sorted(collections.Counter(t.column("player_count").to_pylist()).items()))
print("win_reason_code:", sorted(collections.Counter(t.column("win_reason_code").to_pylist()).items()))
print("winner_team_code:", sorted(collections.Counter(t.column("winner_team_code").to_pylist()).items()))
