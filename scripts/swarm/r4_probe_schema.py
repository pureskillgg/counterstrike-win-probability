"""R4 probe (read-only): schema and within-round / within-match structure of data/state_sample.

Reads only the small parquet tables under G:\\pureskill\\csds-cs2\\data\\state_sample.
Writes nothing. Prints row counts, column names, rounds per match, and snapshot rows per round
for positions_snapshots (if it carries match_key and round columns).
"""
import polars as pl

BASE = r"G:\pureskill\csds-cs2\data\state_sample"

for name in ["rounds.parquet", "positions_snapshots.parquet"]:
    df = pl.read_parquet(f"{BASE}\\{name}")
    print(f"== {name}: rows={df.height} cols={df.width}")
    print("columns:", df.columns)
    print(df.head(3))

rounds = pl.read_parquet(f"{BASE}\\rounds.parquet")
keys = [c for c in ["match_key", "round"] if c in rounds.columns]
if "match_key" in rounds.columns:
    per_match = rounds.group_by("match_key").agg(pl.len().alias("n_rounds"))
    print("== rounds per match:", per_match["n_rounds"].describe())
    print("matches:", per_match.height)

snap = pl.read_parquet(f"{BASE}\\positions_snapshots.parquet")
if {"match_key", "round"}.issubset(snap.columns):
    per_round = snap.group_by(["match_key", "round"]).agg(pl.len().alias("n_states"))
    print("== states per round in positions_snapshots:", per_round["n_states"].describe())
    print("rounds with snapshots:", per_round.height)
    if "tick" in snap.columns:
        print("tick column present; min/max tick:", snap["tick"].min(), snap["tick"].max())
else:
    print("positions_snapshots lacks match_key/round; states-per-round not computed.")
