"""verify_D7_8 step B: ledger-free signals. Which round numbers carry the game's own halftime / match-point
announcements, what t_score_raw / ct_score_raw look like at freeze_end, and phase/is_warmup values.
Reads raw 2024 tome only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


RS = (pl.concat([pl.scan_parquet(f).select(["match_key", "round", "event_type", "t_score", "ct_score",
                                            "t_score_raw", "ct_score_raw", "phase", "is_warmup", "tick"])
                 for f in files(BASE / f"basic_round_state{SUF}")]).collect())
print("round_state slim rows:", RS.height)

print("\nphase values (all rows):", RS.group_by("phase").len().sort("len", descending=True).head(20).rows())
print("is_warmup counts:", RS.group_by("is_warmup").len().rows())

for ev in ["round_announce_last_round_half", "round_announce_match_point", "round_announce_final",
           "cs_win_panel_match", "begin_new_match"]:
    e = RS.filter(pl.col("event_type") == ev)
    per = e.group_by("match_key").agg(pl.len().alias("len"))
    print(f"\n== {ev}: rows {e.height}, matches {e['match_key'].n_unique()}, rows per match dist:",
          per.group_by("len").agg(pl.len().alias("n_matches")).sort("len").rows())
    print("   round numbers (count):", e.group_by("round").len().sort("round").rows()[:40])

# per-match halftime-announcement round(s)
LH = (RS.filter(pl.col("event_type") == "round_announce_last_round_half")
        .group_by("match_key").agg(pl.col("round").sort().alias("lh_rounds")))
LH = LH.with_columns(pl.col("lh_rounds").list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("lh_pattern"))
print("\nlast_round_half round pattern per match (top 20):", LH.group_by("lh_pattern").len().sort("len", descending=True).head(20).rows())

# raw vs cooked scores at freeze_end
FE = RS.filter(pl.col("event_type") == "round_freeze_end")
print("\nfreeze_end: t_score_raw == t_score:", FE.filter(pl.col("t_score_raw") == pl.col("t_score")).height,
      "of", FE.height)
print("freeze_end: ct_score_raw == ct_score:", FE.filter(pl.col("ct_score_raw") == pl.col("ct_score")).height)
print("freeze_end: t_score_raw == ct_score (side-swapped look):",
      FE.filter(pl.col("t_score_raw") == pl.col("ct_score")).height)

# phase at freeze_end: first round with phase 'second' per match (ledger-free halftime marker)
PH = (FE.sort(["match_key", "round"]).group_by("match_key", maintain_order=True)
        .agg(pl.col("round").filter(pl.col("phase") == "second").min().alias("first_second_round"),
             pl.col("round").filter(pl.col("phase") == "first").max().alias("last_first_round"),
             pl.col("phase").unique().sort().alias("phases"),
             pl.col("round").max().alias("maxr")))
print("\nfreeze_end phase sets per match (top 10):", PH.group_by(pl.col("phases").list.join(",")).len()
      .sort("len", descending=True).head(10).rows())
print("first 'second'-phase freeze_end round, per match (top 15):",
      PH.group_by("first_second_round").len().sort("len", descending=True).head(15).rows())
print("last 'first'-phase freeze_end round, per match (top 15):",
      PH.group_by("last_first_round").len().sort("len", descending=True).head(15).rows())
print("first_second_round minus last_first_round (top 10):",
      PH.with_columns((pl.col("first_second_round") - pl.col("last_first_round")).alias("gap"))
        .group_by("gap").len().sort("len", descending=True).head(10).rows())

# where do 'halftime' rows sit (round numbers)
HT = RS.filter(pl.col("phase") == "halftime")
print("\nhalftime-phase rows:", HT.height, "matches:", HT["match_key"].n_unique(),
      "| round counts (top 20):", HT.group_by("round").len().sort("len", descending=True).head(20).rows())
print("event types on halftime-phase rows (top 10):",
      HT.group_by("event_type").len().sort("len", descending=True).head(10).rows())

# show one example match: freeze_end sequence
ex = FE["match_key"].unique().sort()[0]
print("\nexample match:", ex)
print(FE.filter(pl.col("match_key") == ex).sort("round").select(
    ["round", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase", "is_warmup"]).head(30))
print(LH.filter(pl.col("match_key") == ex))
