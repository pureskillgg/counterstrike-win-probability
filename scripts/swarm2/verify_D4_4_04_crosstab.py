r"""Verifier D4-4 cross-tabs: halftime position from freeze-end score steps vs label, bomb and header.

Usage: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_4_04_crosstab.py 2024
       (or 2026 for the control era)

Halftime position per complete regulation match is taken from the freeze-end boundary classes
(swap = side fields cross between r and r+1), not from labels, bombs or header.
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ERA = sys.argv[1] if len(sys.argv) > 1 else "2024"
ROOT = Path(r"G:\pureskill\csds-cs2\tomes")
BASE = ROOT / ("2024-02-g" if ERA == "2024" else "2026-10-01_to_08") / "tome" / "csds"
T, CT = 2, 3


def files(ch):
    fd = BASE / "header" if ch == "header" else sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def classify(a, b):
    t0, c0 = a
    t1, c1 = b
    ns = (t1 - t0, c1 - c0)
    sw = (t1 - c0, c1 - t0)
    ns_ok = ns in ((1, 0), (0, 1))
    sw_ok = sw in ((1, 0), (0, 1))
    if ns_ok and sw_ok:
        return "ambig", None
    if ns_ok:
        return "noswap", (T if ns == (1, 0) else CT)
    if sw_ok:
        return "swap", (CT if sw == (1, 0) else T)
    return "bad", None


hcols = ["key", "t_starters_score_final", "ct_starters_score_final"]
H = pl.scan_parquet(files("header")).collect()
print(f"era={ERA} header columns include max_rounds: {'max_rounds' in H.columns}; rows={H.height}")
HD = {r["key"]: r for r in H.select(hcols).iter_rows(named=True)}
RE = pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code"]).collect()
RE1 = RE.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
BS = (pl.scan_parquet(files("bomb_state")).select(["match_key", "round", "tick", "event_type"])
      .filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).collect())
BO = BS.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_exploded").any().alias("exp"),
    (pl.col("event_type") == "bomb_defused").any().alias("dfs"))
FE_raw = (pl.scan_parquet(files("round_state")).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
          .filter(pl.col("event_type") == "round_freeze_end").collect())
print(f"freeze_end event rows={FE_raw.height}")
FE = FE_raw.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)

rows = defaultdict(dict)
for r in RE1.iter_rows(named=True):
    rows[r["match_key"]].setdefault(r["round"], {})["L"] = r["winner_team_code"]
for r in BO.iter_rows(named=True):
    d = rows[r["match_key"]].setdefault(r["round"], {})
    d["B"] = T if (r["exp"] and not r["dfs"]) else (CT if (r["dfs"] and not r["exp"]) else None)
    d["hasB"] = True
for r in FE.iter_rows(named=True):
    rows[r["match_key"]].setdefault(r["round"], {})["fe"] = (r["t_score"], r["ct_score"])


def day_of(key):
    p = key.split("/")
    return f"{p[1]}-{p[2]}-{p[3]}"


def fit(wa, n, tA, tB):
    if (wa, n - wa) == (tA, tB):
        return "ok"
    if (wa, n - wa) == (tB, tA):
        return "swapped"
    return "bad"


def wins_A(rr, n, sched, bomb_r=None):
    wa = 0; swaps = 0
    for r in range(1, n + 1):
        d = rr[r]
        w = d["L"]
        if bomb_r is not None and r == bomb_r and d.get("B") is not None:
            w = d["B"]
        side_A = T if swaps % 2 == 0 else CT
        wa += (w == side_A)
        if r in sched:
            swaps += 1
    return wa


C = []  # per complete regulation match
for m in rows:
    if m not in HD:
        continue
    rr = rows[m]
    Lr = sorted(r for r in rr if "L" in rr[r])
    n = len(Lr)
    tA = int(HD[m]["t_starters_score_final"]); tB = int(HD[m]["ct_starters_score_final"])
    if not (Lr == list(range(1, n + 1)) and n == tA + tB and n <= 24):
        continue
    swaps = []; ambig_or_missing = []; bnd = {}
    for r in range(1, n):
        a = rr.get(r, {}).get("fe"); b = rr.get(r + 1, {}).get("fe")
        if a is None or b is None:
            bnd[r] = "missing"
        else:
            bnd[r], _ = classify(a, b)
        if bnd[r] == "swap":
            swaps.append(r)
    if swaps == [12]:
        grp = "swap@12"
    elif swaps == [8]:
        grp = "swap@8"
    elif swaps == []:
        grp = "no-swap-found"
    else:
        grp = "other:" + ",".join(map(str, swaps))
    # label vs score-step disagreement by round
    disagree = []
    for r in range(1, n):
        a = rr.get(r, {}).get("fe"); b = rr.get(r + 1, {}).get("fe")
        if a is None or b is None:
            continue
        cls, s = classify(a, b)
        if cls in ("noswap", "swap") and s != rr[r]["L"]:
            disagree.append(r)
    b13 = rr.get(13, {})
    aff = b13.get("B") is not None and "L" in b13 and b13["B"] != b13["L"]
    C.append(dict(m=m, n=n, tA=tA, tB=tB, grp=grp, swaps=swaps, disagree=disagree, aff=aff,
                  day=day_of(m), rr=rr, bnd=bnd))

print(f"\n== {ERA}: complete regulation (n<=24) matches: {len(C)}")
print("halftime position from score steps (group -> count):")
for g, c in Counter(x["grp"] for x in C).most_common():
    print(f"   {g:<16} {c}")

print("\nlabel vs score-step disagreement, by halftime group (disagree-round set -> count):")
dsets = Counter((x["grp"], tuple(x["disagree"])) for x in C)
for k, v in sorted(dsets.items(), key=lambda kv: (kv[0][0], -kv[1])):
    print(f"   {k[0]:<16} disagree_at={list(k[1])}  n={v}")

print("\nbomb-affected (round-13 bomb side != label) by halftime group:")
ag = Counter((x["grp"], "affected" if x["aff"] else ("bomb-agrees" if (x["rr"].get(13, {}).get("hasB")) else "no-bomb13")) for x in C)
for k, v in sorted(ag.items()):
    print(f"   {k[0]:<16} {k[1]:<12} n={v}")

print("\nheader fit (complete regulation) by halftime group:")
for g in sorted(set(x["grp"] for x in C)):
    grp = [x for x in C if x["grp"] == g]
    c12 = Counter(); cs = Counter(); cfix = Counter()
    for x in grp:
        rr = x["rr"]; n = x["n"]
        c12[fit(wins_A(rr, n, {12}), n, x["tA"], x["tB"])] += 1
        sch = set(x["swaps"]) if x["swaps"] else {12}
        cs[fit(wins_A(rr, n, sch), n, x["tA"], x["tB"])] += 1
        # label corrected at the disagreement rounds (score-step winner substituted)
        wa = 0; swaps_n = 0
        for r in range(1, n + 1):
            w = rr[r]["L"]
            if r in x["disagree"]:
                w = 5 - w
            side_A = T if swaps_n % 2 == 0 else CT
            wa += (w == side_A)
            if r in sch:
                swaps_n += 1
        cfix[fit(wa, n, x["tA"], x["tB"])] += 1
    tot = len(grp)
    print(f"   {g:<16} n={tot:>5}  S12-end ok={c12['ok']:>4} swapped={c12['swapped']:>3} bad={c12['bad']:>4}   "
          f"score-swap-end ok={cs['ok']:>4} swapped={cs['swapped']:>3} bad={cs['bad']:>4}   "
          f"score-swap + inverted-label-fix ok={cfix['ok']:>4} swapped={cfix['swapped']:>3} bad={cfix['bad']:>4}")

print("\nother-round checks: bomb side != label, by round (all matches with header, any n):")
mm = Counter(); tot = Counter()
for m in rows:
    if m not in HD:
        continue
    for r, d in rows[m].items():
        if "L" in d and d.get("hasB") and d.get("B") is not None:
            tot[r] += 1
            if d["B"] != d["L"]:
                mm[r] += 1
print("   " + ", ".join(f"r{r}:{mm[r]}/{tot[r]}" for r in sorted(tot)))
print(f"   non-13 total: {sum(v for k, v in mm.items() if k != 13)} of {sum(v for k, v in tot.items() if k != 13)}")
print(f"   round 13 total: {mm[13]} of {tot[13]}")
