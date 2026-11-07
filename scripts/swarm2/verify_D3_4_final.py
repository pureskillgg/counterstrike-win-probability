"""D3-4 verifier: (1) do '()' matches (no detected swap) fit a 12-swap better than no swap (hidden-swap test)?
(2) final-score profile of the (8,) group (is a complete MR12 match possible at 8-16 rounds?).
(3) same header-fit test for the (12,) group as a control.
ASSUMPTIONS: as in verify_D3_4_header.py.
"""
import glob, sys, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def folder(ch):
    return TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
def read(ch, cols=None):
    fs = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")
H = read("header", ["key", "t_starters_score_final", "ct_starters_score_final", "map_name"]).with_columns(
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day")).filter(pl.col("day") >= 12)
keys = H["key"].to_list()
hdr = {k: (int(t), int(c), mp) for k, t, c, mp in H.select(["key", "t_starters_score_final", "ct_starters_score_final", "map_name"]).iter_rows()}
RE = read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
RE = RE.filter(pl.col("match_key").is_in(keys))
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(pl.col("match_key").is_in(keys))
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").group_by(["match_key", "round"])
      .agg(pl.col("t_score").first(), pl.col("ct_score").first()))
del RS
re = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
fe = {(m, r): (t, c) for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
mr = collections.defaultdict(set)
for (m, r) in re: mr[m].add(r)
def dside(m, r):
    if (m, r) in fe and (m, r + 1) in fe:
        a, b = fe[(m, r)], fe[(m, r + 1)]
        dt, dc = b[0] - a[0], b[1] - a[1]
        if (dt, dc) == (1, 0): return 2
        if (dt, dc) == (0, 1): return 3
    return None
def strict(m, r):
    if (m, r) in fe and (m, r + 1) in fe:
        a, b = fe[(m, r)], fe[(m, r + 1)]
        dt, dc = b[0] - a[0], b[1] - a[1]
        return dt + dc == 1 and min(dt, dc) < 0
    return False
S = {m: tuple(r for r in sorted(mr[m]) if strict(m, r)) for m in keys}
def clean(m):
    rs = mr.get(m, set()); N = max(rs) if rs else 0
    return bool(N) and rs == set(range(1, N + 1)) and N == hdr[m][0] + hdr[m][1]
def starters_T(P, w_of, N):
    tot = 0
    for r in range(1, N + 1):
        on_T = (sum(1 for p in P if p < r) % 2 == 0)
        w = w_of(r)
        if w is None: return None
        if (on_T and w == 2) or ((not on_T) and w == 3): tot += 1
    return tot
def fit_delta(m, P):
    N = max(mr[m])
    return starters_T(P, lambda r: (dside(m, r) if dside(m, r) is not None else re[(m, r)]), N) == hdr[m][0]
def fit_code(m, P):
    N = max(mr[m])
    return starters_T(P, lambda r: re[(m, r)], N) == hdr[m][0]
print("(1) hidden-swap test, clean matches, delta-based fit for each schedule:")
for g in [(), (12,), (8,)]:
    ms = [m for m in keys if S[m] == g and clean(m)]
    row = {P: sum(fit_delta(m, P) for m in ms) for P in [(), (12,), (8,), (12, 27)]}
    print(f"  group {str(g):<8} clean n={len(ms):>4} | fit(): {row[()]:>4} | fit(12,): {row[(12,)]:>4} | fit(8,): {row[(8,)]:>4} | fit(12,27): {row[(12, 27)]:>4}")
ms0 = [m for m in keys if S[m] == () and clean(m)]
gap12 = {m: abs(fe[(m, 12)][0] - fe[(m, 12)][1]) for m in ms0 if (m, 12) in fe}
print("  () clean matches with freeze(12) present:", len(gap12), "| gap==1:", sum(1 for g in gap12.values() if g == 1), "| gap>=3:", sum(1 for g in gap12.values() if g >= 3))
for lab, sel in [("gap==1", lambda g: g == 1), ("gap>=3", lambda g: g >= 3)]:
    sub = [m for m in gap12 if sel(gap12[m])]
    print(f"    {lab}: n={len(sub)} | fit(): {sum(fit_delta(m, ()) for m in sub)} | fit(12,): {sum(fit_delta(m, (12,)) for m in sub)}")
print("\n(2) (8,) group final-score profile (all 333; header final scores):")
g8 = [m for m in keys if S[m] == (8,)]
mx = collections.Counter(max(hdr[m][0], hdr[m][1]) for m in g8)
tot = collections.Counter(hdr[m][0] + hdr[m][1] for m in g8)
print("  max team final score distribution:", sorted(mx.items()))
print("  header total distribution:", sorted(tot.items()))
ge13 = sum(1 for m in g8 if max(hdr[m][0], hdr[m][1]) >= 13)
print(f"  matches with a team at >=13 wins (complete MR12 possible): {ge13} of {len(g8)}")
print("  N (round_end count) vs header total, (8,) group: N==total:", sum(1 for m in g8 if len(mr[m]) == hdr[m][0] + hdr[m][1]),
      "| N<total:", sum(1 for m in g8 if len(mr[m]) < hdr[m][0] + hdr[m][1]), "| N>total:", sum(1 for m in g8 if len(mr[m]) > hdr[m][0] + hdr[m][1]))
print("\n(3) control: (12,) group, clean matches fit(): / fit(12,) / fit(12,27)")
g12 = [m for m in keys if S[m] == (12,) and clean(m)]
print("  n clean:", len(g12), "| fit():", sum(fit_delta(m, ()) for m in g12), "| fit(12,):", sum(fit_delta(m, (12,)) for m in g12), "| fit(12,27):", sum(fit_delta(m, (12, 27)) for m in g12))
print("\nDONE")
