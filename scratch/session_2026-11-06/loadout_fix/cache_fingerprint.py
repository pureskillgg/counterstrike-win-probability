"""Fingerprint a parity directory: row counts, match counts, null counts and per-column sums. Read-only.
Run with the local pixi python (local copies) or the space venv python (space copies); same output shape."""
import sys

import polars as pl

base = sys.argv[1]
for c in ["c050", "c060"]:
    d = pl.read_parquet(f"{base}/bucket_{c}.parquet")
    print(f"{c} rows {d.height} matches {d['match_key'].n_unique()} cols {d.width}", flush=True)
    print(f" schema_ok {all(str(d.schema[x]) == 'Float32' for x in d.columns if x not in ('match_key', 'round', 'tick'))}", flush=True)
    print(f" null_total {sum(int(d[x].null_count()) for x in d.columns)}", flush=True)
    s = {x: round(float(d[x].cast(pl.Float64).sum()), 4) for x in d.columns if x not in ("match_key",)}
    print(" sums", s, flush=True)
