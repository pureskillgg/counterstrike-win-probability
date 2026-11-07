"""verify_D8_5_check: independent check of claim D8-5 (2024 tome).

Claim under test: non-round-13 header misfit (other_misfit, 116 complete matches) is concentrated on
de_inferno, de_vertigo, de_overpass, de_nuke; tracks the detected (8,) halftime signature; map is not the key
(the same four maps host 1,155 standard (12,) matches); proposed fix: key on per-match detected swap schedule.

Input: scratchpad rounds.parquet written by scripts\\swarm2\\verify_D8_5_build.py (read-only derivative of the tome).
Output: logs\\swarm2\\verify_D8_5.log
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_check.py
"""
import collections
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
LOG = ROOT / "logs" / "swarm2" / "verify_D8_5.log"
OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D8_5")
T, CT = 2, 3
B = 20000        # ASSUMPTION: permutation replicates (claim used 5000; smallest attainable p = 1/20001)
SEED = 20261014  # ASSUMPTION: fixed RNG seed
RNG = np.random.default_rng(SEED)
FOUR = ["de_inferno", "de_vertigo", "de_overpass", "de_nuke"]
LOG.parent.mkdir(parents=True, exist_ok=True)
LOGF = open(LOG, "w", encoding="utf-8", newline="\n")


def say(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    LOGF.write(s + "\n")
    LOGF.flush()


# ----------------------------------------------------------------- load per-match structures
R = pl.read_parquet(OUT / "rounds.parquet").sort(["match_key", "round"])
MAT = {}
for x in R.iter_rows(named=True):
    m = MAT.get(x["match_key"])
    if m is None:
        m = MAT[x["match_key"]] = {"key": x["match_key"], "map": x["map_name"], "tA": int(x["tA"]),
                                   "tB": int(x["tB"]), "day": int(x["day"]), "rd": []}
    m["rd"].append(x)
for m in MAT.values():
    rd = m["rd"]
    n = len(rd)
    m["n"] = n
    m["contig"] = [x["round"] for x in rd] == list(range(1, n + 1))
    m["complete"] = (n == m["tA"] + m["tB"]) and n <= 24 and m["contig"]
    m["dg"] = "d01-07" if m["day"] <= 7 else "d12-25"

say("=== verify_D8_5 check (independent recomputation) ===")
say(f"matches in 2024 tome: {len(MAT)}; complete regulation (n==tA+tB, n<=24, contiguous): "
    f"{sum(m['complete'] for m in MAT.values())}")
say(f"ASSUMPTION: permutation B={B}, seed={SEED}; day groups d01-07 (day<=7) and d12-25 (day>=12)")


# ----------------------------------------------------------------- D4 class rule (same rule as D4_map_day.py)
def d4_class(m):
    rd, n, tA, tB = m["rd"], m["n"], m["tA"], m["tB"]
    mm13 = any(x["round"] == 13 and x["bside"] is not None and x["bside"] != x["winner"] for x in rd)
    wa = sum(1 for x in rd if x["winner"] == (T if x["round"] <= 12 else CT))
    misfit = (wa, n - wa) != (tA, tB)
    r13 = next((x for x in rd if x["round"] == 13), None)
    flip = False
    if misfit and r13 is not None:
        nwa = wa - 1 if r13["winner"] == CT else wa + 1
        flip = (nwa, n - nwa) == (tA, tB)
    if mm13:
        return "affected"
    if flip:
        return "r13_flip_misfit"
    if misfit:
        return "other_misfit"
    return "fits"


COMP = [m for m in MAT.values() if m["complete"]]
for m in COMP:
    m["cls"] = d4_class(m)
say("\n--- A. D4 classes on complete matches (claim: 1246 total; affected 211, r13_flip 825, other_misfit 116, fits 94)")
tab =pl.DataFrame([{"cls": m["cls"], "dg": m["dg"]} for m in COMP])
say(tab.group_by("cls").len().sort("cls"))
say("complete matches by day group:")
say(tab.group_by("dg").len().sort("dg"))
say(tab.group_by(["dg", "cls"]).len().sort(["dg", "cls"]))


# ----------------------------------------------------------------- D8 signature (same logic as D8_map_test/D8_format_check)
def d8_sig(m):
    live = {x["round"]: (x["tb"], x["cb"]) for x in m["rd"] if x["tb"] is not None}
    swaps = []
    for r in sorted(live):
        if r + 1 not in live:
            continue
        t1, c1 = live[r]
        t2, c2 = live[r + 1]
        dt, dc = t2 - t1, c2 - c1
        if (dc == 1 and dt == 0) or (dt == 1 and dc == 0) or (dt == 0 and dc == 0):
            continue
        if dt * dc <= 0:
            swaps.append(r)
    return tuple(swaps)


for m in MAT.values():
    m["sig"] = d8_sig(m)


# ----------------------------------------------------------------- F-1 replication (all matches, day 12-25)
say("\n--- B. F-1 replication: D8 signature by map, all matches day 12-25 (claim: (8,) 333 only on the four maps)")
D12 = [m for m in MAT.values() if m["dg"] == "d12-25"]
sc = pl.DataFrame([{"map": m["map"], "sig": str(m["sig"])} for m in D12])
say("matches by signature:", sc.group_by("sig").len().sort("len", descending=True).rows())
say(sc.group_by(["map", "sig"]).len().pivot(on="sig", index="map", values="len").fill_null(0).sort("map"))
s8 = [m for m in D12 if m["sig"] == (8,)]
say(f"(8,) matches days 12-25: {len(s8)}; maps: {sorted(set(m['map'] for m in s8))}")
s8_all = [m for m in MAT.values() if m["sig"] == (8,)]
say(f"(8,) matches all days: {len(s8_all)}; by day group: "
    f"{collections.Counter(m['dg'] for m in s8_all)}; maps: {sorted(set(m['map'] for m in s8_all))}")
# round count and header total for (8,) matches
say("(8,) matches: round count n vs header total tA+tB (all days), top patterns:")
say(collections.Counter((m["n"], m["tA"] + m["tB"], m["complete"]) for m in s8_all).most_common(12))
say("(8,) matches: rounds with a bomb outcome (any round), mean per match: "
    f"{np.mean([sum(1 for x in m['rd'] if x['bside'] is not None) for m in s8_all]):.2f}; "
    f"round-13 bomb rows in (8,) matches: {sum(1 for m in s8_all for x in m['rd'] if x['round'] == 13 and x['bside'] is not None)}")


# ----------------------------------------------------------------- C. other_misfit by map (claim numbers)
OM = [m for m in COMP if m["cls"] == "other_misfit"]
say(f"\n--- C. other_misfit (complete): {len(OM)} total; by day group "
    f"{collections.Counter(m['dg'] for m in OM)}; by D8 signature {collections.Counter(m['sig'] for m in OM)}")
say("other_misfit by map, complete matches, all days (claim gives rates for the 1246 set):")
say(f"{'map':<13}{'n_complete':>11}{'other_misfit':>13}{'rate':>8}   n_complete_d12-25 / other_misfit_d12-25")
for mp in sorted(set(m["map"] for m in COMP)):
    n_all = sum(1 for m in COMP if m["map"] == mp)
    k_all = sum(1 for m in OM if m["map"] == mp)
    n12 = sum(1 for m in COMP if m["map"] == mp and m["dg"] == "d12-25")
    k12 = sum(1 for m in OM if m["map"] == mp and m["dg"] == "d12-25")
    say(f"{mp:<13}{n_all:>11}{k_all:>13}{k_all / n_all if n_all else float('nan'):>8.3f}   {n12} / {k12}")
say("other_misfit by map x D8 signature (complete, all days):")
say(collections.Counter((m["map"], m["sig"]) for m in OM).most_common())


# ----------------------------------------------------------------- D. permutation tests (claim: p=0.0002 both)
def het(g, s, y, G, S):
    ns = np.bincount(s, minlength=S).astype(float)
    ys = np.bincount(s, weights=y, minlength=S)
    ps = np.divide(ys, ns, out=np.zeros(S), where=ns > 0)
    pe = ps[s]
    og = np.bincount(g, weights=y, minlength=G)
    eg = np.bincount(g, weights=pe, minlength=G)
    vg = np.bincount(g, weights=pe * (1 - pe), minlength=G)
    ok = vg > 0
    return float(np.sum((og[ok] - eg[ok]) ** 2 / vg[ok]))


def perm_p(g, s, y, G, S, label):
    obs = het(g, s, y, G, S)
    idx = [np.where(s == v)[0] for v in range(S)]
    null = np.empty(B)
    for b in range(B):
        gp = g.copy()
        for ix in idx:
            gp[ix] = RNG.permutation(g[ix])
        null[b] = het(gp, s, y, G, S)
    p = (1 + np.sum(null >= obs - 1e-9)) / (B + 1)
    say(f"  {label:<40} stat={obs:8.2f}  perm p={p:.6f}  (floor 1/{B + 1}={1 / (B + 1):.6f}; "
        f"null exceedances={int(np.sum(null >= obs - 1e-9))})")
    return obs, p


def codes(vals):
    u = sorted(set(vals))
    ix = {v: i for i, v in enumerate(u)}
    return np.array([ix[v] for v in vals], dtype=np.int64), len(u)


say("\n--- D. permutation tests on other_misfit, complete matches, all days (same universe as claim)")
yO = np.array([1.0 if m["cls"] == "other_misfit" else 0.0 for m in COMP])
gM, GM = codes([m["map"] for m in COMP])
gS, GS = codes([str(m["sig"]) for m in COMP])
gD, GD = codes([m["day"] for m in COMP])
z = np.zeros(len(COMP), dtype=np.int64)
perm_p(gM, z, yO, GM, 1, "map, unadjusted")
perm_p(gM, gD, yO, GM, GD, "map | day")
perm_p(gS, z, yO, GS, 1, "signature, unadjusted")
perm_p(gS, gM, yO, GS, GM, "signature | map")
perm_p(gM, gS, yO, GM, GS, "map | signature")

# restricted to the four maps, complete matches
sel4 = np.array([m["map"] in FOUR for m in COMP])
say("within the four maps only (complete, all days):")
g4, G4 = codes([m["sig"] for m in COMP if m["map"] in FOUR])
mp4, GMP = codes([m["map"] for m in COMP if m["map"] in FOUR])
y4 = yO[sel4]
perm_p(g4, mp4, y4, G4, GMP, "signature | map (four maps)")
perm_p(mp4, g4, y4, GMP, G4, "map | signature (four maps)")


# ----------------------------------------------------------------- E. header-fit schedule search (model-free)
def wins_A(rd, swaps, flip_r=None):
    """Wins of the team that started on T (side A), with side A flipping after each swap round in swaps.
    flip_r: optionally flip one round's winner (round-13 style defect)."""
    wa = 0
    for x in rd:
        k = sum(1 for s in swaps if s < x["round"])
        side_a = T if k % 2 == 0 else CT
        won = x["winner"] == side_a
        if flip_r is not None and x["round"] == flip_r:
            won = not won
        wa += int(won)
    return wa


def cands(n):
    return [()] + [(s,) for s in range(1, n)]


def fits(m, swaps, flip_r=None, orient=False):
    n = m["n"]
    wa = wins_A(m["rd"], swaps, flip_r)
    tA, tB = (m["tB"], m["tA"]) if orient else (m["tA"], m["tB"])
    return (wa, n - wa) == (tA, tB)


for m in COMP:
    m["exact"] = [c for c in cands(m["n"]) if fits(m, c)]
    m["flip13"] = [c for c in cands(m["n"]) if any(x["round"] == 13 for x in m["rd"]) and fits(m, c, flip_r=13)]
    m["orient"] = [c for c in cands(m["n"]) if fits(m, c, orient=True)]
    m["exact_default12"] = fits(m, (12,))

say("\n--- E. header-fit schedule search over complete matches (candidates: no swap, or one swap after round s)")
say("E-0 MR12 default (swap after 12): exact fit count by class:")
say(collections.Counter((m["cls"], m["exact_default12"]) for m in COMP))
say("E-1 other_misfit matches: which single-swap candidates fit exactly (round_end winners, header totals):")
say(collections.Counter(tuple(c) for m in OM for c in m["exact"]).most_common(10))
say(f"    other_misfit with NO exact-fit candidate at all: {sum(1 for m in OM if not m['exact'])} of {len(OM)}")
say(f"    other_misfit fitting (8,) exactly: {sum(1 for m in OM if (8,) in m['exact'])} of {len(OM)}")
say(f"    other_misfit fitting (12,) with round-13 flip: {sum(1 for m in OM if (12,) in m['flip13'])} of {len(OM)}")
say(f"    other_misfit fitting any candidate with round-13 flip: {sum(1 for m in OM if m['flip13'])} of {len(OM)}")
say(f"    other_misfit fitting any candidate under swapped header orientation: "
    f"{sum(1 for m in OM if m['orient'])} of {len(OM)}")
say("E-2 (8,) complete matches (all classes): exact fit under (8,) / under (12,) / with (12,)+flip13:")
s8c = [m for m in COMP if m["sig"] == (8,)]
say(f"    n={len(s8c)}; fit (8,) exact={sum(1 for m in s8c if (8,) in m['exact'])}; "
    f"fit (12,) exact={sum(1 for m in s8c if (12,) in m['exact'])}; "
    f"fit (8,)+flip13={sum(1 for m in s8c if fits(m, (8,), flip_r=13))}")
say(f"    class of (8,) complete matches: {collections.Counter(m['cls'] for m in s8c)}")
say("E-3 () (no D8 swap) complete matches by class and fitting candidates:")
s0c = [m for m in COMP if m["sig"] == ()]
say(f"    n={len(s0c)}; classes {collections.Counter(m['cls'] for m in s0c)}")
say(f"    other_misfit with sig (): by map {collections.Counter(m['map'] for m in OM if m['sig'] == ())}")
say(f"    other_misfit with sig (): best candidates {collections.Counter(tuple(c) for m in OM if m['sig'] == () for c in m['exact']).most_common(6)}")
s12c = [m for m in COMP if m["sig"] == (12,)]
say(f"E-4 (12,) complete matches: n={len(s12c)}; classes {collections.Counter(m['cls'] for m in s12c)}")
say(f"    (12,) complete matches on the four maps: {sum(1 for m in s12c if m['map'] in FOUR)}; "
    f"other_misfit among them: {sum(1 for m in s12c if m['cls'] == 'other_misfit')}")
say(f"    (12,) complete on other maps: {sum(1 for m in s12c if m['map'] not in FOUR)}; "
    f"other_misfit among them: {sum(1 for m in s12c if m['cls'] == 'other_misfit' and m['map'] not in FOUR)}")


# ----------------------------------------------------------------- F. winner-consistent step census (independent detector)
def steps(m):
    rd = m["rd"]
    out = []
    for x, y in zip(rd[:-1], rd[1:]):
        if y["round"] != x["round"] + 1:
            out.append((x["round"], "gap"))
            continue
        if None in (x["tb"], x["cb"], y["tb"], y["cb"]):
            out.append((x["round"], "missing"))
            continue
        tT = 1 if x["winner"] == T else 0
        tC = 1 - tT
        ns = (y["tb"] == x["tb"] + tT) and (y["cb"] == x["cb"] + tC)
        sw = (y["tb"] == x["cb"] + tC) and (y["cb"] == x["tb"] + tT)
        out.append((x["round"], "both" if (ns and sw) else "noswap" if ns else "swap" if sw else "anomaly"))
    return out


say("\n--- F. winner-consistent round_state step census (a step must match no-swap or swap given round_end winner)")
say("F-0 step types over complete matches, by class:")
rows = []
for m in COMP:
    st = steps(m)
    rows.append({"cls": m["cls"], "map": m["map"], "n_anom": sum(1 for _, t in st if t == "anomaly"),
                 "n_both": sum(1 for _, t in st if t == "both"), "n_swap": sum(1 for _, t in st if t == "swap"),
                 "n_gap_miss": sum(1 for _, t in st if t in ("gap", "missing")),
                 "anom_rounds": str([r for r, t in st if t == "anomaly"])})
F = pl.DataFrame(rows)
say(F.group_by("cls").agg(pl.len().alias("matches"), (pl.col("n_anom") > 0).sum().alias("with_anomaly"),
                           pl.col("n_swap").sum().alias("swap_steps"), pl.col("n_both").sum().alias("both_steps"),
                           pl.col("n_gap_miss").sum().alias("gap_or_missing")).sort("cls"))
say("F-1 anomaly rounds (step after round r) for other_misfit complete matches (top 10):")
say(F.filter(pl.col("cls") == "other_misfit").group_by("anom_rounds").len().sort("len", descending=True).head(10))
say("F-2 anomaly rounds for r13_flip_misfit (top 6), for comparison:")
say(F.filter(pl.col("cls") == "r13_flip_misfit").group_by("anom_rounds").len().sort("len", descending=True).head(6))

# ----------------------------------------------------------------- G. ASSUMPTION sensitivity: MR12 header fit with any single round flip
say("\n--- G. other_misfit: does any single-round winner flip under MR12 reconcile the header? (ASSUMPTION: one error)")
cnt_any = 0
where = collections.Counter()
for m in OM:
    ok_rounds = []
    for r in range(1, m["n"] + 1):
        if fits(m, (12,), flip_r=r):
            ok_rounds.append(r)
    if ok_rounds:
        cnt_any += 1
        where[tuple(ok_rounds)] += 1
say(f"other_misfit with some single-round flip fitting header under MR12: {cnt_any} of {len(OM)}")
say(f"flip rounds that fit (top): {where.most_common(8)}")

LOGF.close()
print("CHECK DONE")
