"""D3-4 verifier diagnostics: group-level claims. (A) round-count/header-clean profile per signature group.
(B) whether the () group hides a 12-swap (delta(12) availability / header fit under (12,)). (C) map specificity
of the (8,) group (chi-square vs map share). (D) round-13 code-vs-delta disagreement by map. (E) label-shift test:
winner code at 13 vs bomb at 12. (F) groups of clean matches the delta-based (12,) reconstruction fails on.
"""
import glob, sys, collections, math
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(240)
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
hdr = {k: (int(t), int(c), mp, int(d)) for k, t, c, mp, d in H.select(["key", "t_starters_score_final", "ct_starters_score_final", "map_name", "day"]).iter_rows()}
RE = read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
RE = RE.filter(pl.col("match_key").is_in(keys))
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(pl.col("match_key").is_in(keys))
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").group_by(["match_key", "round"])
      .agg(pl.col("t_score").first(), pl.col("ct_score").first()))
BX = set(RS.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round"]).iter_rows())
BD = set(RS.filter(pl.col("event_type") == "bomb_defused").select(["match_key", "round"]).iter_rows())
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
def fits(m, P, src):
    N = max(mr[m])
    if src == "delta":
        f = lambda r: (dside(m, r) if dside(m, r) is not None else re[(m, r)])
    else:
        f = lambda r: re[(m, r)]
    return starters_T(P, f, N) == hdr[m][0]
print("(A) signature group profile (day12-25):")
groups = collections.defaultdict(list)
for m in keys: groups[S[m]].append(m)
for g in sorted(groups, key=lambda g: -len(groups[g])):
    ms = groups[g]
    cl = [m for m in ms if clean(m)]
    Ns = [max(mr[m]) for m in ms]
    tot = [hdr[m][0] + hdr[m][1] for m in ms]
    print(f"  {str(g):<12} n={len(ms):>5}  clean(N==hdr_total, contiguous)={len(cl):>5}  N med={sorted(Ns)[len(Ns)//2]} min={min(Ns)} max={max(Ns)}  hdr_total med={sorted(tot)[len(tot)//2]} min={min(tot)} max={max(tot)}")
print("\n(B) () group: is a 12-swap hidden? delta(12) availability and header fit with (12,) delta-based (clean matches):")
g0 = groups[()]
av = collections.Counter()
for m in g0:
    has12 = (m, 12) in fe and (m, 13) in fe
    cl = clean(m)
    if cl:
        av[("has_fe12_fe13" if has12 else "no_fe12_or_fe13", "clean", "fit(12,)_delta" if fits(m, (12,), "delta") else "nofit(12,)_delta")] += 1
    else:
        av[("has_fe12_fe13" if has12 else "no_fe12_or_fe13", "unclean")] += 1
for k, v in sorted(av.items()): print("  ", k, v)
gap12 = collections.Counter()
for m in g0:
    if (m, 12) in fe and (m, 13) in fe:
        a, b = fe[(m, 12)], fe[(m, 13)]
        gap12[min(abs(a[0] - a[1]), 3)] += 1
print("  () group freeze score gap |t-ct| at round 12 (3 means >=3):", sorted(gap12.items()))
print("\n(C) (8,) group: header fit under (8,) and (12,) delta-based (clean only); map specificity:")
g8 = groups[(8,)]
c8 = [m for m in g8 if clean(m)]
print("  (8,) clean:", len(c8), "| fit (8,) delta:", sum(fits(m, (8,), "delta") for m in c8), "| fit (12,) delta:", sum(fits(m, (12,), "delta") for m in c8))
tot_map = collections.Counter(hdr[m][2] for m in keys)
map8 = collections.Counter(hdr[m][2] for m in g8)
n = len(keys); k8 = len(g8)
print("  map | total | (8,) count | expected(8,) | share in (8,)")
chi = 0.0
for mp in sorted(tot_map, key=lambda x: -tot_map[x]):
    e = tot_map[mp] * k8 / n; o = map8.get(mp, 0)
    chi += (o - e) ** 2 / e if e > 0 else 0
    print(f"  {mp:<12} {tot_map[mp]:>5}  {o:>5}   {e:>8.1f}   {o / tot_map[mp]:.3f}")
print(f"  chi-square (8,) vs map share (df={len(tot_map) - 1}): {chi:.1f}")
print("\n(D) round-13 code vs clean-delta disagreement by map (all matches, clean delta(13)):")
by_map = collections.defaultdict(lambda: [0, 0])
for m in keys:
    if 13 in mr[m] and dside(m, 13) is not None:
        by_map[hdr[m][2]][1] += 1
        by_map[hdr[m][2]][0] += int(dside(m, 13) != re[(m, 13)])
for mp in sorted(by_map, key=lambda x: -by_map[x][1]):
    d, t = by_map[mp]
    print(f"  {mp:<12} disagree={d:>4}/{t:<4} rate={d / t:.3f}")
print("\n(E) label-shift test at round 13: code(13) vs bomb(12) and vs delta-side(12) where clean:")
sh = collections.Counter()
for m in keys:
    if 12 in mr[m] and 13 in mr[m]:
        b12 = 2 if (m, 12) in BX else (3 if (m, 12) in BD else None)
        if b12 is not None:
            sh["code13==bomb12" if re[(m, 13)] == b12 else "code13!=bomb12"] += 1
print("  ", sorted(sh.items()))
print("\n(F) clean matches failing delta-based (12,) reconstruction, by group:")
fail = collections.Counter(); fail_ex = []
for m in keys:
    if clean(m) and not fits(m, (12,), "delta"):
        fail[S[m]] += 1
        if len(fail_ex) < 6: fail_ex.append((m, S[m], hdr[m][0], hdr[m][1], max(mr[m])))
print("  ", fail.most_common())
for ex in fail_ex: print("   example:", ex)
