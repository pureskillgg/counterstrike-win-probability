"""D1 map/format check (swarm2): does the 2024 header carry map name or max_rounds, and do the
switch-signature groups (flip at 12 vs flip at 8) differ by map or max_rounds?
Also checks round_end.player_count convention in the 2026 tome (bakeoff.py expects 10).
ASSUMPTION: the signature table D1_match_signature_table.csv was written by D1_followup3.py.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

HDR = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds\header")
OUT = Path(r"G:\pureskill\csds-cs2\scripts\swarm2")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


f = sorted(glob.glob(str(HDR / "dataframe_*")))[0]
H = pl.read_parquet(f)
hdr("HEADER schema and sample")
print(H.schema)
print(H.head(3))

# common candidates
cands = [c for c in H.columns if any(s in c.lower() for s in ["map", "round", "mode", "format", "type", "date", "game"])]
print("candidate columns:", cands)
for c in cands:
    if H[c].dtype in (pl.Utf8, pl.Int64, pl.Int32, pl.Float64):
        vc = H[c].value_counts().sort("count", descending=True).head(15)
        print(f"\n-- {c}: n_unique={H[c].n_unique()}")
        print(vc)

X = pl.read_csv(OUT / "D1_match_signature_table.csv", infer_schema_length=0)
X = X.with_columns(pl.col("flip_bounds_str").fill_null(""), pl.col("bomb_mm").fill_null("false"))
X = X.with_columns(pl.when(pl.col("flip_bounds_str").str.contains("8") & ~pl.col("flip_bounds_str").str.contains("12"))
                   .then(pl.lit("flip8_only"))
                   .when(pl.col("flip_bounds_str").str.contains("12")).then(pl.lit("flip12"))
                   .otherwise(pl.lit("no_flip_seen")).alias("sig"))
print("\nsignature counts:")
print(X.group_by("sig").len().sort("sig"))

mk = H.columns[0]
hdr("HEADER join: candidate map/format fields by signature group")
key_col = None
for c in H.columns:
    if H[c].dtype == pl.Utf8 and H[c].str.contains("csds/2024").any():
        key_col = c
        break
print("header key column (guess):", key_col)
if key_col is not None:
    HH = H.with_columns(pl.col(key_col).alias("match_key"))
    J = X.join(HH, on="match_key", how="left")
    for c in cands:
        if c in (key_col, "match_key"):
            continue
        print(f"\n-- {c} by signature")
        try:
            print(J.group_by(["sig", c]).len().sort(["sig", "len"], descending=[False, True]).head(25))
        except Exception as e:
            print("skip", c, e)

hdr("2026 tome round_end.player_count convention (bakeoff.py expects 10)")
try:
    f26 = sorted(glob.glob(str(T26 / "basic_round_end*" / "dataframe_*")))
    print("2026 round_end files:", len(f26))
    R26 = pl.concat([pl.read_parquet(p, columns=["player_count"]) for p in f26[:10]], how="vertical_relaxed")
    print("2026 player_count (first 10 files) distribution:")
    print(R26.group_by("player_count").len().sort("len", descending=True).head(8))
except Exception as e:
    print("2026 read failed:", e)
