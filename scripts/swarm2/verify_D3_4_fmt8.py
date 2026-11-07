"""D3-4 verifier: (8,)-group schedule search and final-score pairs. Clean matches only (contiguous 1..N, N == header total).
Searches subsets (size<=3) of candidate halftime/OT positions for the delta-based header fit. READ-ONLY.
"""
import glob, sys, collections, itertools
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
H = read("header", ["key", "t_starters_score_final", "ct_starters_score_final"]).with_columns(
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day")).filter(pl.col("day") >= 12)
keys = H["key"].to_list()
hdr = {k: (int(t), int(c)) for k, t, c in H.select(["key", "t_starters_score_final", "ct_starters_score_final"]).iter_rows()}
RE = read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first").filter(pl.col("match_key").is_in(keys))
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(pl.col("match_key").is_in(keys))
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
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
g8 = [m for m in keys if S[m] == (8,)]
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
def fit(m, P):
    N = max(mr[m])
    return starters_T(P, lambda r: (dside(m, r) if dside(m, r) is not None else re[(m, r)]), N) == hdr[m][0]
print("(8,) group n=", len(g8), "| clean:", sum(clean(m) for m in g8))
pairs = collections.Counter(tuple(sorted(hdr[m], reverse=True)) for m in g8)
print("final score pairs (sorted high-low), top 15:", pairs.most_common(15))
print("max-team final distribution:", sorted(collections.Counter(max(hdr[m]) for m in g8).items()))
cands = [7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 24, 25]
cl = [m for m in g8 if clean(m)]
res = collections.Counter()
for n in range(0, 4):
    for P in itertools.combinations(cands, n):
        c = sum(fit(m, P) for m in cl)
        if c: res[P] = c
print("best schedules for clean (8,) matches (fit count of", len(cl), "):", res.most_common(10))
# how many clean (8,) fit (8,) at all; list example unfit
unfit = [m for m in cl if not fit(m, (8,))]
print("clean (8,) not fitting (8,):", len(unfit))
for m in unfit[:6]:
    print("  ", m.split("/")[4][:8], "hdr", hdr[m], "N", max(mr[m]), "rounds", len(mr[m]), "S", S[m], "codes", [re[(m, r)] for r in sorted(mr[m])][:20])
