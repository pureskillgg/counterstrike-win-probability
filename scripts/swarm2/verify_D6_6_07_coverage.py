"""D6-6 verifier step 07: coverage of the post-boundary rule. How many post-boundary rounds (round after any swap
boundary) have a round_end row but no score winner, by build? And what are the 6 non-post disagreements in build 9961?
Also the bomb-based coverage on post-boundary rounds (e2 definition, pc==20) by build, which does not use scores.
Read-only.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_07_coverage.py
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


def ch_files(base, ch):
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read(base, ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in ch_files(base, ch)], how="vertical_relaxed")


def bounds_all(rounds_map):
    rounds = sorted(rounds_map)
    ref = next((r for r in rounds if len(rounds_map[r]) >= 8), rounds[0])
    t0 = rounds_map[ref]
    seq = []
    for r in rounds:
        cur = rounds_map[r]
        common = [p for p in cur if p in t0]
        if len(common) < 4:
            continue
        frac = sum(1 for p in common if cur[p] != t0[p]) / len(common)
        seq.append((r, 1 if frac > 0.5 else 0))
    return [a[0] for a, b in zip(seq, seq[1:]) if a[1] != b[1] and b[0] == a[0] + 1]


pi = read(BASE24, "player_info", ["match_key", "round", "player_id", "team_code"])
pi = pi.unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
pmap = collections.defaultdict(lambda: collections.defaultdict(dict))
for m, r, p, t in pi.iter_rows():
    pmap[m][r][p] = t
B = {m: bounds_all(rm) for m, rm in pmap.items()}
rows = [(m, b + 1) for m, bl in B.items() for b in bl]
PB = pl.DataFrame(rows, schema=["match", "round"], orient="row").with_columns(pl.lit(True).alias("post_any"))

H = pl.concat([pl.read_parquet(f, columns=["key", "build_num", "map_name", "match_date"])
               for f in sorted(glob.glob(str(BASE24 / "header" / "dataframe_*")))], how="vertical_relaxed")
H = H.unique(["key"], keep="first", maintain_order=True)

L = pl.read_parquet(SCR / "verify_D6_6_long_2024.parquet").join(PB, on=["match", "round"], how="left").with_columns(
    pl.col("post_any").fill_null(False)).join(H.rename({"key": "match"}).select(["match", "build_num", "map_name"]),
                                             on="match", how="left")

print("== (a) post-boundary rounds with a round_end row: scorable vs not, by build ==")
post = L.filter(pl.col("post_any") & pl.col("w_re").is_not_null())
print(post.group_by(["build_num", "scorable"]).agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort(["build_num", "scorable"]))

print("\n== (b) bomb-based coverage on post-boundary rounds (e2 definition, pc==20, bomb outcome present) ==")
bb = L.filter(pl.col("post_any") & (pl.col("pc") == 20) & pl.col("bomb").is_not_null() & pl.col("w_re").is_not_null())
print(bb.group_by("build_num").agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm"),
                                    pl.col("bomb_mm").mean().round(4).alias("rate")).sort("build_num"))
print("bomb-based non-post rounds (pc==20, bomb present), by build:")
nb = L.filter((~pl.col("post_any")) & (pl.col("pc") == 20) & pl.col("bomb").is_not_null() & pl.col("w_re").is_not_null())
print(nb.group_by("build_num").agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm")).sort("build_num"))
print("bomb-based non-post mismatches, build 9961, by round (all rounds):")
print(nb.filter((pl.col("build_num") == 9961) & pl.col("bomb_mm")).group_by("round").len().sort("round").rows())

print("\n== (c) the 6 build-9961 scorable disagreements on non-post rounds ==")
six = L.filter((pl.col("build_num") == 9961) & (~pl.col("post_any")) & pl.col("sc_dis"))
print(six.select(["match", "round", "map_name", "w_sc", "w_re", "bomb", "pc"]).rows())
for m in six["match"].to_list():
    print("  boundaries (all):", B.get(m), " match:", m[:50])

print("\n== (d) post-boundary rounds in build 9961 whose round_end is inverted per bomb (bomb rows only) ==")
print(L.filter(pl.col("post_any") & (pl.col("build_num") == 9961) & pl.col("bomb").is_not_null()).select(
    [pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm")]))
