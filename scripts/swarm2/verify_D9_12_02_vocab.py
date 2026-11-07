"""verify D9-12 step 2: event/phase vocabularies, tick-second relation, per-match label structure (2024 and 2026).

Read-only. One channel at a time, column projection only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
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
    print(f"\n######## {year}")
    rs = scan(base, suf, "round_state", ["match_key", "round", "tick", "event_type", "phase", "second"])
    print("round_state event_type counts:")
    print(rs.group_by("event_type").len().sort("len", descending=True))
    print("round_state phase counts:")
    print(rs.group_by("phase").len().sort("len", descending=True))

    rst = scan(base, suf, "round_start", ["match_key", "round", "tick", "second"])
    ren = scan(base, suf, "round_end", ["match_key", "round", "tick", "second"])
    ps = scan(base, suf, "player_spawn", ["match_key", "round", "tick", "second"])
    # tick vs second relation: ratio tick/second on round_end rows
    rat = ren.filter(pl.col("second") > 0).with_columns((pl.col("tick") / pl.col("second")).alias("tps"))
    print("round_end tick/second ratio quantiles (ticks per second):",
          rat.select(pl.col("tps").quantile(0.05).alias("p05"), pl.col("tps").median().alias("p50"),
                     pl.col("tps").quantile(0.95).alias("p95")).rows())
    # offset between tick and second (tick - 64*second): check constancy
    off = ren.with_columns((pl.col("tick") - 64.0 * pl.col("second")).alias("off64"))
    print("round_end tick - 64*second quantiles:",
          off.select(pl.col("off64").quantile(0.05).alias("p05"), pl.col("off64").median().alias("p50"),
                     pl.col("off64").quantile(0.95).alias("p95")).rows())

    # per-match label structure
    n_match = rst.select("match_key").n_unique()
    print("distinct matches in round_start:", n_match, "| in round_end:", ren.select("match_key").n_unique())
    def labelstats(df, name):
        g = df.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                                         pl.len().alias("rows"), pl.col("round").n_unique().alias("nuniq"))
        print(f"{name}: matches={g.height} rows={int(g['rows'].sum())} "
              f"rmin dist={g.group_by('rmin').len().sort('rmin').rows()[:5]} "
              f"rows-vs-uniq mismatch matches={g.filter(pl.col('rows') != pl.col('nuniq')).height}")
        return g
    gs = labelstats(rst, "round_start")
    ge = labelstats(ren, "round_end")
    # matches where round_start max label differs from round_end max label
    j = gs.join(ge, on="match_key", how="full", suffix="_end")
    j = j.with_columns((pl.col("rmax") - pl.col("rmax_end")).alias("max_diff"))
    print("round_start max label minus round_end max label, per match:",
          j.group_by("max_diff").len().sort("max_diff").rows())
    print("round_start min label per match:", gs.group_by("rmin").len().sort("rmin").rows())
    # per-match count difference rows
    j2 = j.with_columns((pl.col("rows") - pl.col("rows_end")).alias("row_diff"))
    print("round_start rows minus round_end rows, per match:",
          j2.group_by("row_diff").len().sort("row_diff").head(15).rows())
