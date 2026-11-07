"""Debug helper 2 for verify_D7_8: print the P-table join columns for one match (stdout only)."""
import io
import contextlib
import sys
from pathlib import Path

import polars as pl

src = Path(r"G:\pureskill\csds-cs2\scripts\swarm2\verify_D7_8_main.py").read_text(encoding="utf-8")
src = src.replace('sys.stdout.reconfigure(encoding="utf-8")', "pass")
g = {"__name__": "verify_main_debug2"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, "verify_D7_8_main.py", "exec"), g)
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(260)
FE, SA, U, N, P = (g[k] for k in ["FE", "SA", "U", "N", "P"])
K = [k for k in FE["match_key"].unique().to_list() if k.endswith("m_799e15539078498e49fcd3b6/csds")][0]
print("match:", K[-40:])
print("FE rounds 10-14:")
print(FE.filter((pl.col("match_key") == K) & pl.col("round").is_between(10, 14)).sort("round"))
print("SA rounds 10-14:")
print(SA.filter((pl.col("match_key") == K) & pl.col("round").is_between(10, 14)).sort("round"))
print("U rounds 11-13:")
print(U.filter((pl.col("match_key") == K) & pl.col("round").is_between(11, 13)).sort("round"))
print("N rounds 10-13 (keyed by round-1):")
print(N.filter((pl.col("match_key") == K) & pl.col("round").is_between(10, 13)).sort("round"))
print("P rounds 11-13:")
print(P.filter((pl.col("match_key") == K) & pl.col("round").is_between(11, 13)).sort("round"))
