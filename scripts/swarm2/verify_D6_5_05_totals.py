"""D6-5 verifier step 5: totals for the defect set under determinate resolution, with and without content copies.

Set D (2024, d12-25 only, all disagreements found):
  R1 scorable arithmetic winner != round_end                       (claim's population)
  R2 gap-1 aligned 'both' rows, winner != round_end                 (determinate: same under both orientations)
  R3 gap-0 tie rows at rounds 9, 13, 25 (side-attached, A1)         (round 27 excluded: boundary, A1 invalid there)
Rounds 9, 13, 25, 28 only; all other rounds produce zero disagreements in R1/R2 and zero bomb mismatches.
Later content copies: match_keys whose round-end fingerprint equals an earlier key (step 3 definition).
Run: G:\pureskill\csds-cs2\.pixi\envs\default\python.exe -I -X utf8 scripts\swarm2\verify_D6_5_05_totals.py
"""
import collections
import pickle
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
T, CT = 2, 3

with open(SCRATCH / "verify_D6_5_state.pkl", "rb") as f:
    S = pickle.load(f)
sc, rew, dup_later = S["sc"], S["rew"], S["dup_later_fp"]


def d_of(m):
    return int(m.split("/")[3])


def resolve(scm, r):
    """Return (kind, winner) with kind in scorable, gap1, tie, no_next, incons."""
    if r + 1 not in scm:
        return "no_next", None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    dT, dC = t1 - t0, c1 - c0
    no = (dT, dC) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and sw:
        if t0 == c0:
            return "tie", (T if (dT, dC) == (1, 0) else CT)
        return "gap1", (T if (dT, dC) == (1, 0) else CT)
    if no:
        return "scorable", (T if (dT, dC) == (1, 0) else CT)
    if sw:
        return "scorable", (T if (t1 - c0, c1 - t0) == (0, 1) else CT)
    return "incons", None


rows_all = 0
for m, scm in sc.items():
    if d_of(m) >= 12 and d_of(m) <= 25:
        rows_all += len(scm)
rows_all_all = sum(len(scm) for scm in sc.values())
print(f"round rows (freeze_end) all days: {rows_all_all}; d12-25: {rows_all}")

copy_rows = sum(len(sc[m]) for m in dup_later if m in sc)
print(f"later-copy keys: {len(dup_later)}; their round rows: {copy_rows}")

D = collections.Counter()
Dcopy = collections.Counter()
ROUNDS = (9, 13, 25, 28)
per_kind = collections.Counter()
for m, scm in sc.items():
    if not (12 <= d_of(m) <= 25):
        continue
    for r in ROUNDS:
        if r not in scm:
            continue
        kind, w = resolve(scm, r)
        w_re = rew.get((m, r))
        if w is None or w_re is None:
            continue
        if kind == "tie" and r == 27:
            continue
        if kind in ("scorable", "gap1", "tie") and w != w_re:
            D[r] += 1
            per_kind[(r, kind)] += 1
            if m in dup_later:
                Dcopy[r] += 1
print(f"\ndefect set (R1+R2+R3, rounds 9/13/25/28, d12-25): total {sum(D.values())}; by round {dict(sorted(D.items()))}")
print(f"  by round and kind: {dict(sorted(per_kind.items()))}")
print(f"  later-copy members: {sum(Dcopy.values())}; by round {dict(sorted(Dcopy.items()))}")
prim = sum(D.values()) - sum(Dcopy.values())
print(f"  primary-only (after content dedupe): {prim}; by round {dict(sorted((r, D[r] - Dcopy[r]) for r in D))}")
print(f"  rate on claim's denominator (83,382 rows): all {sum(D.values()) / rows_all_all:.4f}; primary-only "
      f"{prim / (rows_all_all - copy_rows):.4f} (denominator minus later-copy rows)")
print(f"  claim's 2,042 as share of 83,382: {2042 / rows_all_all:.4f}")

# Round 27 artifact size if side-attached were (wrongly) applied at the OT boundary
art = 0
for m, scm in sc.items():
    if not (12 <= d_of(m) <= 25) or 27 not in scm:
        continue
    kind, w = resolve(scm, 27)
    w_re = rew.get((m, 27))
    if kind == "tie" and w is not None and w_re is not None and w != w_re:
        art += 1
print(f"\nboundary check: round-27 tie rows (d12-25) where side-attached disagrees with round_end: {art} "
      "(artifact; A1 does not hold across the OT swap after round 27, so these are excluded)")

# Partition of the claim's 22,753 by cause
parts = collections.Counter()
for m, scm in sc.items():
    for r in scm:
        kind, _ = resolve(scm, r)
        if kind == "tie":
            parts["ties (gap 0, odd rounds)"] += 1
        elif kind == "gap1":
            parts["gap-1 aligned (even rounds)"] += 1
        elif kind == "no_next":
            parts["no next freeze_end (last round of match)"] += 1
print(f"\nclaim's unscorable 22,753 decomposed: {dict(parts)}; sum = {sum(parts.values())}")
print("  note: gap-1 rows are determinate (step 4 section 1: 7,920 of 7,920 same winner under both orientations)")
