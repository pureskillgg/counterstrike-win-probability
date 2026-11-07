"""D3-4 verifier control: is the (8,) swap group map-specific in 2026 too, and what are its final scores?
Uses the 2026 freeze-delta strict signature (same rule as 2024). READ-ONLY. Header via glob of header folders.
"""
import glob, sys, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
def files(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}.*" / "dataframe_*")))
hfiles = sorted(glob.glob(str(T26 / "header" / "dataframe_*")))
print("header files:", len(hfiles))
H = pl.concat([pl.read_parquet(f, columns=["key", "map_name", "t_starters_score_final", "ct_starters_score_final"]) for f in hfiles], how="vertical_relaxed")
hdr = {k: (mp, int(t), int(c)) for k, mp, t, c in H.select(["key", "map_name", "t_starters_score_final", "ct_starters_score_final"]).iter_rows()}
RSL = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("event_type") == "round_freeze_end").collect() for f in files("round_state")], how="vertical_relaxed")
FE = RSL.sort("tick").group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first())
fe = {(m, r): (t, c) for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
rounds = collections.defaultdict(set)
for (m, r) in fe: rounds[m].add(r)
S = {}
for m in rounds:
    s = []
    for r in sorted(rounds[m]):
        if (m, r + 1) in fe:
            a, b = fe[(m, r)], fe[(m, r + 1)]
            dt, dc = b[0] - a[0], b[1] - a[1]
            if dt + dc == 1 and min(dt, dc) < 0: s.append(r)
    S[m] = tuple(s)
g8 = [m for m in S if S[m] == (8,)]
print("2026 matches:", len(S), "| (8,) group:", len(g8))
tot = collections.Counter(hdr[m][0] for m in S if m in hdr)
c8 = collections.Counter(hdr[m][0] for m in g8 if m in hdr)
print("map | total | (8,) | share (8,) :")
for mp in sorted(tot, key=lambda x: -tot[x]):
    print(f"  {mp:<12} {tot[mp]:>5} {c8.get(mp, 0):>5}  {c8.get(mp, 0) / tot[mp]:.3f}")
print("(8,) final score pairs (top 10):", collections.Counter(tuple(sorted(hdr[m][1:], reverse=True)) for m in g8 if m in hdr).most_common(10))
print("(12,) group count:", sum(1 for m in S if S[m] == (12,)), "| () group:", sum(1 for m in S if S[m] == ()))
print("DONE")
