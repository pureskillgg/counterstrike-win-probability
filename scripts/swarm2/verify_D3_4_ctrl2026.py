"""D3-4 verifier control: same freeze-delta vs round_end-code test on the 2026 tome (G: tome, read-only).
Tests whether the first-post-swap disagreement (abs round 13 in MR12) also appears in 2026. Channel folders are
basic_<channel>.<date range>; glob by prefix. Narrow projection; filter freeze rows lazily.
"""
import glob, sys, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
def files(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}.*" / "dataframe_*")))
print("round_end files:", len(files("round_end")), "| round_state files:", len(files("round_state")))
RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code"]) for f in files("round_end")], how="vertical_relaxed")
RE = RE.unique(["match_key", "round"], keep="first")
RSL = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("event_type") == "round_freeze_end").collect() for f in files("round_state")], how="vertical_relaxed")
FE = RSL.sort("tick").group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first())
re = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
fe = {(m, r): (t, c) for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
print("2026 round_end (m,r):", len(re), "| freeze (m,r):", len(fe), "| matches:", len({m for m, _ in re}))
mr = collections.defaultdict(set)
for (m, r) in re: mr[m].add(r)
def delta(m, r):
    if (m, r) in fe and (m, r + 1) in fe:
        a, b = fe[(m, r)], fe[(m, r + 1)]
        return b[0] - a[0], b[1] - a[1]
    return None
def side(d):
    if d == (1, 0): return 2
    if d == (0, 1): return 3
    return None
sig = collections.Counter(); off = collections.Counter(); off_dis = collections.Counter()
abs13 = collections.defaultdict(lambda: [0, 0])
anom = collections.Counter()
for m in mr:
    S = []
    for r in sorted(mr[m]):
        d = delta(m, r)
        if d is None: continue
        dt, dc = d
        if dt + dc == 1 and min(dt, dc) < 0: S.append(r)
        if not (dt + dc == 1 and min(dt, dc) >= 0) and not (dt + dc == 1 and min(dt, dc) < 0): anom[(dt + dc)] += 1
    sig[tuple(S)] += 1
    for r in sorted(mr[m]):
        d = delta(m, r)
        if d is None or side(d) is None: continue
        prev = [s for s in S if s < r]
        k = (r - max(prev)) if prev else 0
        off[k] += 1
        if side(d) != re[(m, r)]: off_dis[k] += 1
        if r == 13:
            abs13[tuple(S)][1] += 1
            abs13[tuple(S)][0] += int(side(d) != re[(m, r)])
print("signature groups (strict rule), top 8:", sig.most_common(8))
print("non-clean non-swap freeze-delta sums (sum:count):", sorted(anom.items())[:10])
print("clean-delta disagreement by offset k (0=no prior swap):")
for k in sorted(off)[:8]:
    print(f"  k={k:>2} rounds={off[k]:>7} disagree={off_dis[k]:>6} rate={off_dis[k] / off[k]:.4f}")
print("round 13 code-vs-delta disagreement by signature (disagree/total):")
for g, (d, t) in sorted(abs13.items(), key=lambda x: -x[1][1])[:6]:
    print(f"  {str(g):<12} {d}/{t}")
print("\nDONE")
