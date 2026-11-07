import glob, os, collections, pyarrow.parquet as pq
for label, root in [("2024-02-g", r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"), ("2026-10-01_to_08", r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")]:
    d = [x for x in os.listdir(root) if x.startswith("basic_round_end.")][0]
    t = pq.read_table(sorted(glob.glob(os.path.join(root, d, "dataframe_*")))[0], columns=["player_count", "win_reason_code", "winner_team_code"])
    n = t.num_rows
    pc = collections.Counter(t.column("player_count").to_pylist())
    wr = collections.Counter(t.column("win_reason_code").to_pylist())
    print(f"## {label}: round_end rows={n}")
    print("  player_count top:", pc.most_common(6))
    print("  win_reason_code:", sorted(wr.items()))
    print("  winner codes:", sorted(collections.Counter(t.column("winner_team_code").to_pylist()).items()))
