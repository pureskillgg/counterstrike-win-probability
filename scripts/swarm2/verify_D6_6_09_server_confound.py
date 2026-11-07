"""D6-6 verifier step 09: strongest alternative to 'build' is the game/server directory (srcds host or game version).
If the post-boundary defect is uniform across all game_directory values inside build 9961, and absent in all
values inside build 9842, the build (not a server or a single host) is the discriminator.
Read-only. Uses the long table from step 02b and header game_directory.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_09_server_confound.py
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
SCR = Path(os.environ.get("VERIFY_SCRATCH", r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"))
BASE24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"

H = pl.concat([pl.read_parquet(f, columns=["key", "build_num", "game_directory", "server_name", "network_protocol"])
               for f in sorted(glob.glob(str(BASE24 / "header" / "dataframe_*")))], how="vertical_relaxed")
H = H.unique(["key"], keep="first", maintain_order=True).rename({"key": "match"}).with_columns(
    pl.col("server_name").str.extract(r"(srcds\d+)", 1).alias("srcds_host"))
L = pl.read_parquet(SCR / "verify_D6_6_long_2024.parquet").join(H, on="match", how="left")

# post-boundary rows from the D6 orientation rule: round immediately after any boundary (recomputed in step 06/07)
rows = []  # post flags re-derived here so this step is self-contained


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
for m, rm in pmap.items():
    for b in bounds_all(rm):
        rows.append((m, b + 1))
PB = pl.DataFrame(rows, schema=["match", "round"], orient="row").with_columns(pl.lit(True).alias("post_any"))
L = L.join(PB, on=["match", "round"], how="left").with_columns(pl.col("post_any").fill_null(False))

sc = L.filter(pl.col("scorable"))
print("== post-boundary scorable rounds by build x game_directory ==")
print(sc.filter(pl.col("post_any")).group_by(["build_num", "game_directory"]).agg(
    pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort(["build_num", "game_directory"]))
print("\n== non-post scorable rounds by build (dis count) ==")
print(sc.filter(~pl.col("post_any")).group_by("build_num").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort("build_num"))
print("\n== post-boundary scorable rounds by srcds host within build 9961 (top 10 by n) ==")
print(sc.filter(pl.col("post_any") & (pl.col("build_num") == 9961)).group_by("srcds_host").agg(
    pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort("n", descending=True).head(10))
print("\n== distinct srcds hosts carrying BOTH builds (count of hosts) ==")
hb = H.group_by("srcds_host").agg(pl.col("build_num").n_unique().alias("nb"), pl.len().alias("n"))
print("hosts:", hb.height, " hosts with >1 build:", hb.filter(pl.col("nb") > 1).height)
print(hb.filter(pl.col("nb") > 1).sort("n", descending=True).head(8))
