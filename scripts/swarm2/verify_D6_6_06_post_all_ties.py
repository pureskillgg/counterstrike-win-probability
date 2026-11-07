"""D6-6 verifier step 06: (a) does the defect sit on EVERY round right after any swap boundary (not only the first),
(b) what explains the unscorable rounds (W1 ambiguity at tied round-start scores, or missing freeze_end),
and (c) does the map split add anything beyond the build split.

Uses the 2024 tome directly for player_info bounds (all boundaries, D6 rule) and the long table from step 02b.
Read-only. Writes nothing to the repo.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_06_post_all_ties.py
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

L = pl.read_parquet(SCR / "verify_D6_6_long_2024.parquet")
H = pl.concat([pl.read_parquet(f, columns=["key", "build_num", "map_name", "match_date"])
               for f in sorted(glob.glob(str(BASE24 / "header" / "dataframe_*")))], how="vertical_relaxed")
H = H.unique(["key"], keep="first", maintain_order=True)
L = L.join(H.rename({"key": "match"}), on="match", how="left")

# (a) post-any-boundary flag
rows = []
for m, bl in B.items():
    for b in bl:
        rows.append((m, b + 1, b))
PB = pl.DataFrame(rows, schema=["match", "round", "boundary"], orient="row").with_columns(pl.lit(True).alias("post_any"))
L2 = L.join(PB.select(["match", "round", "post_any", "boundary"]), on=["match", "round"], how="left").with_columns(
    pl.col("post_any").fill_null(False))
sc = L2.filter(pl.col("scorable"))
print("== (a) scorable rounds: disagreement by build x (round immediately after ANY swap boundary) ==")
print(sc.group_by(["build_num", "post_any"]).agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis"),
                                                  pl.col("sc_dis").mean().round(4).alias("rate")).sort(["build_num", "post_any"]))
print("build 9961 disagreements NOT on a post-boundary round:", int(sc.filter((pl.col("build_num") == 9961) & (~pl.col("post_any")))["sc_dis"].sum()))
print("build 9961 post-boundary rounds with NO disagreement:", int(sc.filter((pl.col("build_num") == 9961) & pl.col("post_any") & (~pl.col("sc_dis"))).height))
nb = sc.filter((pl.col("build_num") == 9961) & pl.col("post_any") & (~pl.col("sc_dis")))
print("   those rows (match, round, boundary):", nb.select(["match", "round", "boundary"]).rows()[:10])
print("post-boundary rounds by boundary value, build 9961:")
print(sc.filter((pl.col("build_num") == 9961) & pl.col("post_any")).group_by("boundary").agg(
    pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort("boundary"))

# (b) unscorable composition: tied round-start score?
print("\n== (b) unscorable rounds with a round_end row: is the round-start score tied? (2024, all builds) ==")
SCM = read(BASE24, "round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
SCM = SCM.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first", maintain_order=True)
sc_map = {(m, r): (t, c) for m, r, t, c in SCM.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
un = L.filter(pl.col("w_re").is_not_null() & pl.col("w_sc").is_null())
cnt = collections.Counter()
for m, r in un.select(["match", "round"]).iter_rows():
    s0 = sc_map.get((m, r))
    s1 = sc_map.get((m, r + 1))
    if s0 is None:
        cnt["no freeze_end at r"] += 1
    elif s1 is None:
        cnt["no freeze_end at r+1 (match end or missing)"] += 1
    elif s0[0] == s0[1]:
        cnt["tied at round start (W1 ambiguous)"] += 1
    else:
        cnt["other (not tied, score change not +1)"] += 1
print(dict(cnt))
print("round_end rows in 2024 with no score winner, by round (top 8):")
print(un.group_by("round").len().sort("len", descending=True).head(8))

# (c) map split vs build split for disagreements (scorable, all rounds)
print("\n== (c) map vs build: scorable disagreement rate on build-9961 rows, by map ==")
print(sc.filter(pl.col("build_num") == 9961).group_by("map_name").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis"),
                                                                     pl.col("sc_dis").mean().round(3).alias("rate")).sort("map_name"))
print("map split within build 9842 (scorable disagreements):")
print(sc.filter(pl.col("build_num") == 9842).group_by("map_name").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort("map_name"))
