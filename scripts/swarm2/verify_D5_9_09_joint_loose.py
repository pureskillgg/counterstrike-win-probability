"""verify D5-9, step 9: can the claim's joint table (1,621 13-only; 161 9-only; none both) be reproduced?

Tries looser usability rules on the late regime-A matches (days 12-25), labels 9 and 13:
  rule S (strict): boundary unambiguous (noswap or swap), truth from that class           (step 3 rule)
  rule L (loose):  any fit (noswap or swap or ambig); truth = noswap-fit winner if the boundary fits noswap,
                   else swap-fit winner
  rule X (fe-tie excluded but hidden halftimes kept): as S, plus matches whose boundary at 12 or 8 is ambiguous
Reports the joint counts for each rule. No writes outside scratchpad.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_09_joint_loose.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_9")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
meta = {r["match_key"]: r for r in M.to_dicts()}
LATE = {f"{d:02d}" for d in range(12, 26)}

fe = defaultdict(dict)
rend = defaultdict(dict)
for r in R.iter_rows(named=True):
    if r["fe_t"] is not None:
        fe[r["match_key"]][r["round"]] = (r["fe_t"], r["fe_ct"])
    if r["end_winner"] is not None:
        rend[r["match_key"]][r["round"]] = r["end_winner"]


def fit(mk, r):
    """return (cls, truth_noswap_fit, truth_swap_fit) for boundary r"""
    F = fe[mk]
    if r not in F or r + 1 not in F:
        return ("none", None, None)
    t0, c0 = F[r]
    t1, c1 = F[r + 1]
    dt, dc = t1 - t0, c1 - c0
    s = (dt + dc) == 1
    ns = s and dt in (0, 1) and dc in (0, 1)
    sw = s and (t1 - c0) in (0, 1) and (c1 - t0) in (0, 1)
    tn = (2 if dt == 1 else 3) if ns else None
    ts = (2 if (c1 - t0) == 1 else 3) if sw else None
    if ns and sw:
        return ("ambig", tn, ts)
    if ns:
        return ("noswap", tn, None)
    if sw:
        return ("swap", None, ts)
    return ("bad", None, None)


def amb_at(mk, r):
    return fit(mk, r)[0] == "ambig"


late_A = [mk for mk in meta if meta[mk]["regime"] == "A" and meta[mk]["day"] in LATE]
swap8 = {mk for mk in meta if fit(mk, 8)[0] == "swap"}
swap12 = {mk for mk in meta if fit(mk, 12)[0] == "swap"}


def truth_rule(mk, r, rule):
    cls, tn, ts = fit(mk, r)
    if rule == "S":
        if cls == "noswap":
            return tn
        if cls == "swap":
            return ts
        return None
    if rule == "L":
        if tn is not None:
            return tn
        if ts is not None:
            return ts
        return None


print("late regime-A matches:", len(late_A))
for rule in ["S", "L"]:
    joint = Counter()
    for mk in late_A:
        t9 = truth_rule(mk, 9, rule)
        t13 = truth_rule(mk, 13, rule)
        if t9 is None or t13 is None or 9 not in rend[mk] or 13 not in rend[mk]:
            joint["not both usable"] += 1
            continue
        i9 = rend[mk][9] != t9
        i13 = rend[mk][13] != t13
        joint[(f"9inv={i9}", f"13inv={i13}")] += 1
    print(f"rule {rule} (strict={rule=='S'}):", {str(k): v for k, v in sorted(joint.items(), key=str)})

# rule X: hidden MR12 halftimes (boundary 12 ambiguous) with usable 13 and 9
joint = Counter()
for mk in late_A:
    t9 = truth_rule(mk, 9, "S")
    t13 = truth_rule(mk, 13, "L")
    if t9 is None or t13 is None or 9 not in rend[mk] or 13 not in rend[mk]:
        joint["not both"] += 1
        continue
    joint[(f"9inv={rend[mk][9] != t9}", f"13inv={rend[mk][13] != t13}", "hidden12" if amb_at(mk, 12) else "clean12")] += 1
print("rule X (9 strict, 13 loose, with hidden-12 flag):", {str(k): v for k, v in sorted(joint.items(), key=str)})
print("claim: 13-inverted-only 1,621; 9-inverted-only 161; none both")
print("\nDONE joint loose")
