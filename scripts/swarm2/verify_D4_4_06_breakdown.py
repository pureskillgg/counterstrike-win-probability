r"""Verifier D4-4 final breakdown of the claimed sets by independent score-step status.

Score-step status at round 13 uses only freeze-end fields: winner = T if (dt,dc)==(1,0) and CT if (0,1)
on boundary 13->14 (no halftime is ever detected at 13->14 in the data; see verify_D4_4_03 / _04 output).
Halftime group = score-step swap position (swap@12, swap@8 or undetermined).

Usage: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_4_06_breakdown.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
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


H = pl.scan_parquet(files("header")).select(["key", "t_starters_score_final", "ct_starters_score_final"]).collect()
HD = {r["key"]: r for r in H.iter_rows(named=True)}
RE = pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code"]).collect()
RE1 = RE.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
BS = (pl.scan_parquet(files("bomb_state")).select(["match_key", "round", "tick", "event_type"])
      .filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).collect())
BO = BS.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_exploded").any().alias("exp"),
    (pl.col("event_type") == "bomb_defused").any().alias("dfs"))
FE = (pl.scan_parquet(files("round_state")).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
FE = FE.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)

rows = defaultdict(dict)
for r in RE1.iter_rows(named=True):
    rows[r["match_key"]].setdefault(r["round"], {})["L"] = r["winner_team_code"]
for r in BO.iter_rows(named=True):
    d = rows[r["match_key"]].setdefault(r["round"], {})
    d["B"] = T if (r["exp"] and not r["dfs"]) else (CT if (r["dfs"] and not r["exp"]) else None)
    d["hasB"] = True
for r in FE.iter_rows(named=True):
    rows[r["match_key"]].setdefault(r["round"], {})["fe"] = (r["t_score"], r["ct_score"])


def fit(wa, n, tA, tB):
    if (wa, n - wa) == (tA, tB):
        return "ok"
    if (wa, n - wa) == (tB, tA):
        return "swapped"
    return "bad"


def wins_A(rr, n, sched):
    wa = 0; swaps = 0
    for r in range(1, n + 1):
        side_A = T if swaps % 2 == 0 else CT
        wa += (rr[r]["L"] == side_A)
        if r in sched:
            swaps += 1
    return wa


def halftime(rr, n):
    sw = []
    for r in range(1, n):
        a = rr.get(r, {}).get("fe"); b = rr.get(r + 1, {}).get("fe")
        if a is None or b is None:
            continue
        if classify(a, b)[0] == "swap":
            sw.append(r)
    return tuple(sw)


GN = {(12,): "swap@12", (8,): "swap@8", (): "undetermined"}
CR = []
for m in rows:
    if m not in HD:
        continue
    rr = rows[m]
    Lr = sorted(r for r in rr if "L" in rr[r])
    n = len(Lr)
    tA = int(HD[m]["t_starters_score_final"]); tB = int(HD[m]["ct_starters_score_final"])
    if Lr == list(range(1, n + 1)) and n == tA + tB and n <= 24 and 13 in rr and "L" in rr[13]:
        CR.append((m, n, tA, tB))
print(f"complete regulation n<=24 with round-13 label: {len(CR)}")

aff_status = Counter(); union_status = Counter(); misfit_status = Counter(); flip_status = Counter()
mismatch_by_group = Counter()
for m, n, tA, tB in CR:
    rr = rows[m]
    g = GN.get(halftime(rr, n), "other")
    L13 = rr[13]["L"]; B13 = rr[13].get("B"); hasB = bool(rr[13].get("hasB"))
    fe13 = rr.get(13, {}).get("fe"); fe14 = rr.get(14, {}).get("fe")
    w13 = None
    if fe13 is not None and fe14 is not None:
        dt, dc = fe14[0] - fe13[0], fe14[1] - fe13[1]
        w13 = T if (dt, dc) == (1, 0) else (CT if (dt, dc) == (0, 1) else None)
    sc = "no-score" if w13 is None else ("score=label" if w13 == L13 else "score!=label")
    affected = hasB and B13 is not None and B13 != L13
    if affected:
        bm = "bomb=score" if (w13 is not None and B13 == w13) else ("bomb!=score" if w13 is not None else "bomb-no-score")
        aff_status[(g, bm)] += 1
    s12 = fit(wins_A(rr, n, {12}), n, tA, tB)
    if s12 != "ok" and not affected:
        misfit_status[(g, "bomb13" if hasB else "no-bomb13", sc)] += 1
        if not hasB:
            wa = wins_A(rr, n, {12})
            isA = (L13 == CT)  # S12: team A on CT in round 13
            new_wa = wa - (1 if isA else 0) + (0 if isA else 1)
            if (new_wa, n - new_wa) == (tA, tB):
                flip_status[(g, sc)] += 1
    if not affected and not hasB and s12 != "ok":
        pass
    union_status[(g, "affected" if affected else ("bomb-agrees" if hasB else "no-bomb13"), sc)] += 1

print("\nclaimed 211 affected (round-13 bomb side != label): bomb vs independent score-step at round 13")
for k, v in sorted(aff_status.items()):
    print(f"  {k[0]:<13} {k[1]:<14} n={v}")
print(f"  total affected = {sum(aff_status.values())}")

print("\nclaimed 941 unaffected S12 misfits: halftime group x bomb row x score-step round 13")
for k, v in sorted(misfit_status.items()):
    print(f"  {k[0]:<13} {k[1]:<10} {k[2]:<14} n={v}")
print(f"  total misfits = {sum(misfit_status.values())}")

print("\nclaimed 825 round-13 single-flip fixes (no bomb row): halftime group x score-step round 13")
for k, v in sorted(flip_status.items()):
    print(f"  {k[0]:<13} {k[1]:<14} n={v}")
print(f"  total = {sum(flip_status.values())}")
print(f"  of which score-step confirms the round-13 label inverted: {sum(v for k, v in flip_status.items() if k[1] == 'score!=label')}")
print(f"  of which score-step says round 13 label is correct: {sum(v for k, v in flip_status.items() if k[1] == 'score=label')}")
print(f"  of which no usable score-step at round 13: {sum(v for k, v in flip_status.items() if k[1] == 'no-score')}")

print("\nall complete regulation matches: halftime group x affected/bomb/no-bomb x score-step round 13")
for k, v in sorted(union_status.items()):
    print(f"  {k[0]:<13} {k[1]:<12} {k[2]:<14} n={v}")
