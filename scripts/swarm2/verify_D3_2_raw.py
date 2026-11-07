"""D3-2 verifier: does the E: raw 2024 copy carry the same round-13 round_end rows as the G: tome?

Reads, for five sample matches (three with a round-13 bomb mismatch, two with a round-13 non-bomb
score disagreement), the round_end / bomb_state / round_state files under the E: tail copy and the same
channels from the G: 2024 tome, and compares rows on the key columns. Read-only on E:.
Source demos (.dem) are not present under this E: tree, so this checks copy equality only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")

SAMPLES = [
    ("12", "dd1fcc82-4dd1-46b0-9cfe-58178154871d", "bomb mismatch r13"),
    ("24", "5e09cc8f-46f7-4f8f-a3d3-d95d2246d5a8", "bomb mismatch r13"),
    ("25", "952a47e6-ff78-486d-bd7b-4c67c580ff50", "bomb mismatch r13"),
    ("12", "c781d3d6-7990-4621-b12f-65e8a9e2fd91", "non-bomb r13 score disagreement"),
    ("15", "34b87e2b-049a-412d-b572-914d7c47732c", "non-bomb r13 score disagreement"),
]
CH = ["round_end", "bomb_state", "round_state"]
KEEP = {
    "round_end": ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"],
    "bomb_state": ["match_key", "round", "tick", "event_type", "player_team_code"],
    "round_state": ["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
}


def g_rows(ch, mk):
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    df = pl.concat([pl.read_parquet(f, columns=KEEP[ch]) for f in fs], how="vertical_relaxed")
    return df.filter(pl.col("match_key") == mk).sort(["round", "tick", "event_type"] if ch != "round_end"
                                                       else ["round", "tick"])


for dd, uuid, label in SAMPLES:
    mk = f"csds/2024/02/{dd}/{uuid}/csds"
    print(f"\n== match {uuid} day {dd}: {label}")
    for ch in CH:
        epath = RAW / dd / uuid / ch
        try:
            e = pl.read_parquet(str(epath), columns=[c for c in KEEP[ch] if c != "match_key"])
            e_cols = list(e.columns)
        except Exception as ex:  # noqa: BLE001
            print(f"  {ch}: E: read failed: {type(ex).__name__}: {str(ex)[:160]}")
            continue
        g = g_rows(ch, mk).drop("match_key")
        ecols = [c for c in KEEP[ch] if c in e_cols and c != "match_key"]
        ee = e.select(ecols).sort([c for c in ["round", "tick", "event_type"] if c in ecols])
        gg = g.select(ecols).sort([c for c in ["round", "tick", "event_type"] if c in ecols])
        same = ee.height == gg.height and ee.equals(gg)
        print(f"  {ch}: E rows {ee.height}  G rows {gg.height}  identical on {ecols}: {same}")
        if ch == "round_end":
            print("  E round_end:", ee.filter(pl.col("round") == 13).rows() if "round" in ecols else "")
            print("  G round_end:", gg.filter(pl.col("round") == 13).rows() if "round" in ecols else "")

print("\n== DONE")
