"""D6-6 verifier step 10: three-way agreement on post-boundary rounds in build 9961 and in build 9842.
Which source is the odd one out: round_end winner, score-derived winner, or bomb outcome?
Read-only. Uses the long table from step 02b (scratchpad) plus the post-boundary flag re-derived from player_info.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_10_three_way.py
"""
import collections
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(os.environ.get("VERIFY_SCRATCH", r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"))
BASE24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def ch_files(base, ch):
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read(base, ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in ch_files(base, ch)], how="vertical_relaxed")


def bounds_all(rm):
    rounds = sorted(rm)
    ref = next((r for r in rounds if len(rm[r]) >= 8), rounds[0])
    t0 = rm[ref]
    seq = []
    for r in rounds:
        cur = rm[r]
        common = [p for p in cur if p in t0]
        if len(common) < 4:
            continue
        frac = sum(1 for p in common if cur[p] != t0[p]) / len(common)
        seq.append((r, 1 if frac > 0.5 else 0))
    return [a[0] for a, b in zip(seq, seq[1:]) if a[1] != b[1] and b[0] == a[0] + 1]


pi = read(BASE24, "player_info", ["match_key", "round", "player_id", "team_code"]).unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
pmap = collections.defaultdict(lambda: collections.defaultdict(dict))
for m, r, p, t in pi.iter_rows():
    pmap[m][r][p] = t
rows = [(m, b + 1) for m, rm in pmap.items() for b in bounds_all(rm)]
PB = pl.DataFrame(rows, schema=["match", "round"], orient="row").with_columns(pl.lit(True).alias("post_any"))
H = pl.concat([pl.read_parquet(f, columns=["key", "build_num"]) for f in sorted(glob.glob(str(BASE24 / "header" / "dataframe_*")))],
              how="vertical_relaxed").unique(["key"], keep="first", maintain_order=True).rename({"key": "match"})
L = (pl.read_parquet(SCR / "verify_D6_6_long_2024.parquet").join(PB, on=["match", "round"], how="left")
     .with_columns(pl.col("post_any").fill_null(False)).join(H, on="match", how="left"))

for build in (9842, 9961):
    sub = L.filter((pl.col("build_num") == build) & pl.col("post_any") & pl.col("bomb").is_not_null()
                   & pl.col("w_re").is_not_null() & pl.col("w_sc").is_not_null())
    n = sub.height
    print(f"build {build}: post-boundary rows with bomb, round_end and score winner: n={n}")
    print(f"   score == bomb: {int((sub['w_sc'] == sub['bomb']).sum())}   round_end == bomb: {int((sub['w_re'] == sub['bomb']).sum())}"
          f"   score == round_end: {int((sub['w_sc'] == sub['w_re']).sum())}")
sub = L.filter((pl.col("build_num") == 9961) & pl.col("post_any") & pl.col("bomb").is_not_null() & pl.col("w_re").is_not_null())
print(f"build 9961 post-boundary rows with bomb and round_end (score may be absent): n={sub.height}; "
      f"round_end == bomb: {int((sub['w_re'] == sub['bomb']).sum())}")
