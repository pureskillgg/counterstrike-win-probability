"""D1-10 verifier: one 2024 match and one 2026 match, printed as compact tuples.
Shows the round window anchors (round_start channel, round_state round_freeze_end and round_end, and the
round_end channel tick) next to the bomb_state outcome rows, so the anchor that contains the bomb tick
can be identified. Read-only, column projection, one match per tome.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
MATCHES = [
    (T24, "csds/2024/02/12/05d51edb-04bc-420c-91b9-4bb95317ca19/csds", "2024 bomb-dis round 13 match"),
    (T26, "csds/2026/10/08/002aa219-53f3-4d2f-9bed-b9bdb8236ea4/csds", "2026 swap example match"),
]


def fl(root, ch):
    return sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))


for root, key, label in MATCHES:
    print(f"\n==== {label}: {key.split('/')[4][:8]}")
    st = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick"]).collect() for f in fl(root, "round_start")])
    st = st.filter(pl.col("match_key") == key).sort(["round", "tick"])
    en = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "winner_team_code"]).collect() for f in fl(root, "round_end")])
    en = en.filter(pl.col("match_key") == key).sort(["round", "tick"])
    rs = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type"]).collect() for f in fl(root, "round_state")])
    rs = rs.filter((pl.col("match_key") == key) & pl.col("event_type").is_in(["round_freeze_end", "round_end", "round_start", "round_officially_ended"]))
    bs = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type"]).collect() for f in fl(root, "bomb_state")])
    bs = bs.filter((pl.col("match_key") == key) & pl.col("event_type").is_in(["bomb_exploded", "bomb_defused", "bomb_planted"]))
    print("   round_start channel (round, tick):", [(r, t) for r, t in st.select(["round", "tick"]).iter_rows() if 10 <= r <= 15])
    print("   round_end channel (round, tick, winner):", [(r, t, w) for r, t, w in en.select(["round", "tick", "winner_team_code"]).iter_rows() if 10 <= r <= 15])
    rsx = rs.filter(pl.col("round").is_between(11, 14)).sort(["round", "tick"])
    print("   round_state anchors (round, tick, event):")
    for r in rsx.iter_rows(named=True):
        print("     ", r["round"], r["tick"], r["event_type"])
    bsx = bs.filter(pl.col("round").is_between(11, 14)).sort(["round", "tick"])
    print("   bomb_state (round, tick, event):", [(r["round"], r["tick"], r["event_type"]) for r in bsx.iter_rows(named=True)])
print("\nDONE")
