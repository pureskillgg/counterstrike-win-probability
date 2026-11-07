# verify_A1_3_naive2.py
# Follow-up checks for A1-3 (2026 tome). Tests the claimed naive-read failure on player_hurt's
# attacker_id_fixed, whether ScanCastOptions can absorb the missing def_index column of item_equip,
# and whether a per-page projection with diagonal concat recovers a usable frame. Lazy/aggregated or
# two-column projections only; memory stays far below 3 GB.
import os, glob, collections
import polars as pl
import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def folder(ch):
    return os.path.join(TOME, f"basic_{ch}.2026-09-18,2026-10-08")


def pages(ch):
    return sorted(glob.glob(os.path.join(folder(ch), "dataframe_*")))


def P(*a):
    print(*a, flush=True)


def tryit(label, fn):
    try:
        P(f"   {label}: OK -> {fn()}")
    except Exception as e:
        P(f"   {label}: FAIL {type(e).__name__}: {str(e).splitlines()[0][:200]}")


P(f"verify_A1_3_naive2 (polars {pl.__version__})")
P("\n-- A. player_hurt attacker_id_fixed (the column the claim names for this channel)")
ph = pages("player_hurt")
tryit("scan.select(null_count(attacker_id_fixed)).collect()",
      lambda: pl.scan_parquet(ph).select(pl.col("attacker_id_fixed").null_count()).collect().item())
tryit("scan.select(null_count(player_id_fixed)).collect()  [control]",
      lambda: pl.scan_parquet(ph).select(pl.col("player_id_fixed").null_count()).collect().item())

P("\n-- B. item_equip: can ScanCastOptions absorb the missing def_index column (pages 2-7)?")
ie = pages("item_equip")
opts = pl.ScanCastOptions(integer_cast="allow-float")
tryit("scan(cast allow-float).select(len)", lambda: pl.scan_parquet(ie, cast_options=opts).select(pl.len()).collect().item())
tryit("scan(cast allow-float).select(null_count(def_index))",
      lambda: pl.scan_parquet(ie, cast_options=opts).select(pl.col("def_index").null_count()).collect().item())
tryit("scan(no cast).select(null_count(def_index))",
      lambda: pl.scan_parquet(ie).select(pl.col("def_index").null_count()).collect().item())

P("\n-- C. item_equip: per-page projection [player_id_fixed, def_index] + diagonal_relaxed concat")
frames = []
for p in ie:
    cols = ["player_id_fixed"] + (["def_index"] if "def_index" in pq.read_schema(p).names else [])
    frames.append(pl.read_parquet(p, columns=cols))
dfc = pl.concat(frames, how="diagonal_relaxed")
P(f"   rows={dfc.height} dtypes={dict(dfc.schema)}")
P(f"   player_id_fixed nulls={dfc.get_column('player_id_fixed').null_count()} "
  f"def_index nulls={dfc.get_column('def_index').null_count()} "
  f"def_index populated={dfc.height - dfc.get_column('def_index').null_count()}")
P("   rows where def_index is absent (pages 2-7, column missing) = rows whose page lacks the column:")
absent = sum(pq.read_metadata(p).num_rows for p in ie if "def_index" not in pq.read_schema(p).names)
P(f"   absent-column rows={absent} (pages 2-7 sum to 6,996,880 rows, so this should equal 6,996,880)")
del dfc, frames

P("\n-- D. item_equip page 1 (mixed day 05-08): def_index populated share by day (projection: match_key, def_index)")
p1 = ie[1]
d1 = pl.read_parquet(p1, columns=["match_key", "def_index"])
d1 = d1.with_columns(pl.col("match_key").str.extract(r"/(\d{2})/[0-9a-fA-F-]{36}/", 1).alias("day"))
g = d1.group_by("day").agg(pl.len().alias("rows"), pl.col("def_index").null_count().alias("def_nulls")).sort("day")
for row in g.iter_rows(named=True):
    pop = row["rows"] - row["def_nulls"]
    P(f"   day {row['day']}: rows={row['rows']} def_index populated={pop} ({100.0 * pop / row['rows']:.1f}%)")
del d1, g

P("\nDONE naive2")
