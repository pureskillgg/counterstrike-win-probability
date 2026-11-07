# Probe: why buy-end snapshots are dropped (reads the round diagnostics written by economy_snapshot.py).
import os
import pandas as pd
ROOT = r"G:\pureskill\csds-cs2"
d = pd.read_csv(os.path.join(ROOT, "logs", "eda", "economy_snapshot_round_diag.csv"))
b = d[d["buy_status"].notna()]
print("rounds with a buy tick:", len(b))
print(b["buy_status"].value_counts().to_string())
print(b.groupby("buy_status")[["buy_players", "buy_n_T", "buy_n_CT"]].agg(["min", "median", "max"]).to_string())
print(b.groupby(["buy_status"])["buy_players"].value_counts().to_string())
print("freeze status among buy-dropped:", b.loc[b["buy_status"] == "dropped", "status"].value_counts().to_string())
