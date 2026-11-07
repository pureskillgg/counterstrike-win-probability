r"""D5-8 verifier: control for the freeze-end step method. Step-vs-label agreement by round (regime A and B),
restricted to no-phase-change boundaries. Control rounds (not 13) should agree ~100%; round 13 should not.
Also: within regime A round-13 matches whose header final fits the flipped label, the non-bomb step disagreement.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_stepcontrol.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_8v")
R = pl.read_parquet(OUT / "rounds_v2.parquet").sort(["match_key", "round"])
X = R.with_columns(nt=pl.col("fe_t").shift(-1).over("match_key"), nct=pl.col("fe_ct").shift(-1).over("match_key"),
                   nr=pl.col("round").shift(-1).over("match_key"), nph=pl.col("fe_phase").shift(-1).over("match_key"))
X = X.filter((pl.col("nr") == pl.col("round") + 1) & (pl.col("fe_phase") == pl.col("nph"))
             & pl.col("fe_t").is_not_null() & pl.col("nt").is_not_null())
X = X.with_columns(dt=pl.col("nt") - pl.col("fe_t"), dct=pl.col("nct") - pl.col("fe_ct")).with_columns(
    st=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(2).when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(3)
    .otherwise(None).cast(pl.Int64))
X = X.filter(pl.col("st").is_not_null() & pl.col("round").is_between(6, 16))
t = X.group_by(["grp", "round"]).agg(pl.len().alias("n"), (pl.col("st") != pl.col("winner")).sum().alias("ne"),
                                    ((pl.col("st") != pl.col("winner")).mean()).round(3).alias("disagree")).sort(["grp", "round"])
print("== step-vs-label disagreement by round (clean no-phase-change steps), rounds 6..16:")
for g in ["lateA", "lateB", "earlyB"]:
    rows = t.filter(pl.col("grp") == g).select(["round", "n", "ne", "disagree"]).rows()
    print(" ", g, rows)
