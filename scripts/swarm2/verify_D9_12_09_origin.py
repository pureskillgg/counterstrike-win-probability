"""verify D9-12 step 9: where the 2024 round_start offset originates, and round-1 coverage.

  (a) Do round_state rows of event_type round_start / round_prestart carry the same (match, label, tick) as the
      basic round_start rows? If yes, the offset is already in the labelled round_state events (upstream of the
      basic round_start channel). Control: 2026 tome.
  (b) 2024: matches with no basic round_start rows at all, and their round_end row counts.
  (c) 2024: matches whose first round_end / round_freeze_end label is 1 (round 1 present) or 2 (round 1 absent).
Read-only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}


def scan(base, suf, ch, cols):
    fs = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    lf = pl.scan_parquet(fs)
    have = set(lf.collect_schema().names())
    return lf.select([c for c in cols if c in have]).collect()


for year, (base, suf) in TOMES.items():
    print(f"\n################ {year}")
    rs_raw = scan(base, suf, "round_start", ["match_key", "round", "tick"])
    rs = rs_raw.select(["match_key", "round", "tick"]).unique()
    st = scan(base, suf, "round_state", ["match_key", "round", "tick", "event_type"])
    re_raw = scan(base, suf, "round_end", ["match_key", "round", "tick"])
    for ev in ["round_start", "round_prestart"]:
        s = st.filter(pl.col("event_type") == ev).select(["match_key", "round", "tick"]).unique()
        m = s.join(rs, on=["match_key", "round", "tick"], how="inner").height
        print(f"[a] round_state '{ev}' rows (unique): {s.height}; exact (match,label,tick) match to basic round_start: {m}")
    # label-shifted match (same tick, label + 1)
    rs_shift = rs.with_columns((pl.col("round") + 1).alias("round"))
    s = st.filter(pl.col("event_type") == "round_prestart").select(["match_key", "round", "tick"]).unique()
    print(f"    round_prestart rows matching basic round_start at label+1 instead: "
          f"{s.join(rs_shift, on=['match_key', 'round', 'tick'], how='inner').height}")

    ms = re_raw.group_by("match_key").agg(pl.len().alias("n_re"))
    rsm = rs_raw.group_by("match_key").agg(pl.len().alias("n_rs"))
    no_rs = ms.join(rsm, on="match_key", how="anti")
    print(f"[b] matches with round_end rows: {ms.height}; matches with NO round_start rows: {no_rs.height}; "
          f"their round_end rows: {no_rs['n_re'].to_list()}")

    first_re = re_raw.group_by("match_key").agg(pl.col("round").min().alias("first_re_label"))
    print(f"[c] first round_end label per match (1 = round 1 present, 2 = round 1 absent):",
          first_re.group_by("first_re_label").len().sort("first_re_label").rows())
