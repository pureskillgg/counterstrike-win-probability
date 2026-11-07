"""D9-12 independent verifier, score-based absolute test. Read-only on the tomes.

Question: physically, does a round_start / round_prestart row labelled r come before or after round r's result is
banked? Score totals are set by round outcomes, not by round labels, so they give an absolute reference:
  completed rounds at a moment = t_score + ct_score at that moment (ASSUMPTION S: scores count completed rounds
  one for one; checked below against freeze_end rows).
For a row labelled r:
  * if completed rounds = r - 1, the row sits in round r's grid (label correct);
  * if completed rounds = r, the row sits after round r finished, i.e. it belongs to round r + 1 (labelled one early).

Two score readings are used:
  (i) the row's own t_score + ct_score (round_state rows carry scores; basic round_start has none);
  (ii) as-of reading: latest round_state score row at or before the row's tick (works for basic round_start).
Output: diff = completed_rounds - label. Expected: freeze_end diff = -1 (correct labels). diff = 0 for a start family
means that family is labelled one round early.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024-02-g": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026-10-01_to_08": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}


def scan(base, suf, ch, cols):
    files = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    return pl.scan_parquet(files).select(cols).collect()


def show(j, title):
    print(f"   {title}")
    print("      overall diff = completed - label:", j.group_by("sdiff").len().sort("sdiff").rows() if j.height else [])
    if j.height:
        print("      by build:", j.group_by(["build_num", "sdiff"]).len().sort(["build_num", "sdiff"]).rows())


for tome, (base, suf) in TOMES.items():
    print(f"\n########## tome {tome}")
    H = pl.concat([pl.read_parquet(f, columns=["key", "build_num"]) for f in
                   sorted(glob.glob(str(base / "header" / "dataframe_*")))]).rename({"key": "match_key"})

    RS = scan(base, suf, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    nn = RS.select([pl.col("t_score").is_null().sum().alias("t_null"),
                    pl.col("ct_score").is_null().sum().alias("ct_null")]).rows()
    print("   round_state null score counts (t, ct):", nn)
    RS = RS.filter(pl.col("t_score").is_not_null() & pl.col("ct_score").is_not_null()) \
        .with_columns((pl.col("t_score") + pl.col("ct_score")).alias("s"))
    score_rows = RS.select(["match_key", "tick", "s"]).sort("tick")

    def own_scores(event):
        d = RS.filter(pl.col("event_type") == event).select(["match_key", "round", "s"])
        return d.with_columns((pl.col("s") - pl.col("round")).alias("sdiff")).join(H, on="match_key", how="left")

    print("\n  -- (i) own-row scores (round_state rows carry t_score and ct_score)")
    for ev in ["round_freeze_end", "round_prestart", "round_poststart", "round_start", "round_end",
               "round_officially_ended"]:
        show(own_scores(ev), f"round_state {ev}")

    print("\n  -- (ii) as-of scores at the tick of each family (latest round_state score row at or before tick)")

    def asof_scores(tbl, name):
        d = tbl.select(["match_key", "round", "tick"]).sort("tick")
        j = d.join_asof(score_rows.rename({"tick": "tick_sc"}), left_on="tick", right_on="tick_sc",
                        by="match_key", strategy="backward")
        j = j.with_columns((pl.col("s").fill_null(-99) - pl.col("round")).alias("sdiff"))
        j = j.join(H, on="match_key", how="left")
        show(j, name)

    asof_scores(RS.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]),
                "freeze_end (as-of, sanity: expect -1)")
    asof_scores(scan(base, suf, "round_start", ["match_key", "round", "tick"]),
                "basic round_start TABLE (as-of)")
    BS = scan(base, suf, "bomb_state", ["match_key", "round", "tick", "event_type"])
    asof_scores(BS.filter(pl.col("event_type") == "bomb_planted"), "bomb_planted (as-of; live rounds expect -1)")
