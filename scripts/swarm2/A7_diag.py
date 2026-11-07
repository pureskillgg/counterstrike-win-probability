"""A7 diagnostics: examples behind the 2024 round-9 first disagreements and the ATT boundaries."""
import collections
import importlib.util
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
spec = importlib.util.spec_from_file_location("A7", ROOT / "scripts" / "swarm2" / "A7_score_coverage.py")
A7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(A7)
T, CT = A7.T, A7.CT
H, W, F, info = A7.load_tome("2024", *A7.TOMES["2024"])
print("2024 loaded", info)

def mr_of(m):
    fit = A7.mr_fit_2024(F.get(m, {}), A7.MR24_CANDIDATES)
    return min(A7.MR24_CANDIDATES, key=lambda k: (fit[k], k)), fit

wcode = {2: "T", 3: "CT"}
# 1. ATT boundaries by round number and kind
att_r = collections.Counter(); att_kind = collections.Counter()
att_examples = []
for m in sorted(H):
    mr, fit = mr_of(m)
    Fm = F.get(m, {})
    for r in sorted(Fm):
        if r + 1 not in Fm: continue
        t, c = Fm[r]; t2, c2 = Fm[r+1]
        cls = A7.boundary_class(t, c, t2, c2)
        if cls in ("ambig", "neither"): continue
        sched = A7.swap_expected(r, mr)
        if cls != ("swap" if sched else "noswap"):
            kind = "sched_noswap" if sched else "unsched_swap"
            att_r[(r, kind)] += 1
            att_kind[kind] += 1
            if len(att_examples) < 40:
                att_examples.append((m, mr, r, kind, t, c, t2, c2, W.get(m, {}).get(r), W.get(m, {}).get(r+1)))
print("ATT boundaries by (round, kind) top:", sorted(att_r.items(), key=lambda x: -x[1])[:25])
print("ATT kinds:", dict(att_kind))
print("\nATT examples (match, mr, r, kind, t_r, c_r, t_r+1, c_r+1, W_r, W_r+1):")
for e in att_examples[:25]:
    print("  ", e[0][-40:], e[1:])

# 2. first disagreement round 9: show freeze rows, winners and derived lead
print("\n== first DIS at round 9: examples ==")
shown = 0
cnt9 = 0
for m in sorted(H):
    mr, fit = mr_of(m)
    Fm, Wm = F.get(m, {}), W.get(m, {})
    res, det = A7.analyze_match(m, mr, Wm, Fm)
    if res.get("first_dis_round", 0) == 9:
        cnt9 += 1
        if shown < 4:
            shown += 1
            print("match", m[-50:], "mr", mr, "fit", fit)
            for r in range(1, 14):
                print(f"   r{r:>2} freeze t,c={Fm.get(r)}  winner={Wm.get(r)}")
            print("   details:", [d for d in det if d[0] in ('DIS', 'WINMM', 'ATT_SCHED', 'ATT_UNSCHED')][:8])
print("matches with first DIS at 9:", cnt9)

# 3. round-9 winner mismatch: winner code vs score inferred, counts by round-9 freeze pattern
print("\n== WINMM at round 9: boundary 9->10 examples ==")
shown = 0
for m in sorted(H):
    mr, fit = mr_of(m)
    Fm, Wm = F.get(m, {}), W.get(m, {})
    res, det = A7.analyze_match(m, mr, Wm, Fm)
    if 9 in [int(x) for x in res.get("winmm_list", "").split(",") if x]:
        if shown < 3:
            shown += 1
            print("match", m[-50:], "mr", mr)
            for r in range(6, 13):
                print(f"   r{r:>2} freeze={Fm.get(r)} winner={Wm.get(r)}")
print("done")
