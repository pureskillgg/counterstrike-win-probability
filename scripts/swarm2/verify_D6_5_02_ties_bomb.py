"""D6-5 verifier step 2: where do the unscorable rows sit, and does the round_end defect extend into them?

Uses the cached state from step 1 (score transitions, round_end winners, player_count).
Side-attached winner (no swap between r and r+1) is applied ONLY to rounds 9, 13, 25, 28, where the
transition r -> r+1 contains no halftime or overtime side swap (swap boundaries are after 8, 12, 27).
ASSUMPTION A1: no side swap between freeze_end(r) and freeze_end(r+1) for r in {9, 13, 25, 28}.
  Checked below: arithmetic 'swap' labels at these rounds should be zero.
Bomb outcome (independent of score and round_end): bomb_defused -> CT (3), bomb_exploded -> T (2).
Run: G:\pureskill\csds-cs2\.pixi\envs\default\python.exe -I -X utf8 scripts\swarm2\verify_D6_5_02_ties_bomb.py
"""
import collections
import glob
import pickle
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
T, CT = 2, 3


def files(ch):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


with open(SCRATCH / "verify_D6_5_state.pkl", "rb") as f:
    S = pickle.load(f)
sc, rew, pc = S["sc"], S["rew"], S["pc"]


def day_group(m):
    d = int(m.split("/")[3])
    return "d01-07" if d <= 7 else "d12-25" if 12 <= d <= 25 else "other"


def arith(scm, r):
    """Score-only arithmetic label and winner, same rule as step 1."""
    if r + 1 not in scm:
        return "no_next", None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    dT, dC = t1 - t0, c1 - c0
    no = (dT, dC) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and sw:
        return "both", None
    if no:
        return "no", (T if (dT, dC) == (1, 0) else CT)
    if sw:
        return "swap", (T if (t1 - c0, c1 - t0) == (0, 1) else CT)
    return "inconsistent", None


def side_attached(scm, r):
    """Winner assuming no side swap between r and r+1 (ASSUMPTION A1). None if increments are not a single point."""
    if r + 1 not in scm:
        return None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    if (t1 - t0, c1 - c0) == (1, 0):
        return T
    if (t1 - t0, c1 - c0) == (0, 1):
        return CT
    return None


BS = read("bomb_state", ["match_key", "round", "event_type"])
oc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
both_ev = oc.filter(pl.col("evs").list.len() > 1).height
bomb = {}
for m, r, evs in oc.iter_rows():
    bomb[(m, r)] = CT if "bomb_defused" in evs else T
print(f"bomb outcome rounds: {len(bomb)}; rounds with both defused and exploded rows: {both_ev}")

print("\n== 0. Assumption A1 check: arithmetic 'swap' labels at rounds 9, 13, 25, 28 (should be zero) ==")
lab = collections.Counter()
for m, scm in sc.items():
    for rr in (9, 13, 25, 28):
        if rr in scm:
            lab[(rr, arith(scm, rr)[0])] += 1
for k in sorted(lab):
    print(f"  round {k[0]} label {k[1]:<12} n={lab[k]}")

print("\n== 1. Round 13 and other defect rounds: bomb-verified rows by score status (pc == 20 as in e2, and all pc) ==")
rows = collections.defaultdict(collections.Counter)
for (m, r), b in bomb.items():
    if r not in (9, 12, 13, 14, 25, 28):
        continue
    if m not in sc or r not in sc[m]:
        rows[(r, day_group(m), "pc" + str(pc.get((m, r))), "no_freeze_end")]["n"] += 1
        continue
    scm = sc[m]
    st, w_arith = arith(scm, r)
    w_side = side_attached(scm, r)
    w_re = rew.get((m, r))
    if st == "no":
        status = "score_scorable"
        w_sc = w_arith
    elif st == "swap":
        status = "score_swap_label"
        w_sc = w_arith
    else:
        status = f"score_{st}"
        w_sc = w_side
    rows[(r, day_group(m), "pc" + str(pc.get((m, r))), status)]["n"] += 1
    if w_re is None:
        rows[(r, day_group(m), "pc" + str(pc.get((m, r))), status)]["re_missing"] += 1
        continue
    rows[(r, day_group(m), "pc" + str(pc.get((m, r))), status)]["re_eq_bomb" if w_re == b else "re_ne_bomb"] += 1
    if w_sc is None:
        rows[(r, day_group(m), "pc" + str(pc.get((m, r))), status)]["sc_none"] += 1
    else:
        rows[(r, day_group(m), "pc" + str(pc.get((m, r))), status)]["sc_eq_bomb" if w_sc == b else "sc_ne_bomb"] += 1
        rows[(r, day_group(m), "pc" + str(pc.get((m, r))), status)]["sc_eq_re" if w_sc == w_re else "sc_ne_re"] += 1
print("  round  day      pc    status                 n    re!=bomb  re==bomb  sc!=bomb  sc==bomb  sc_none  re_miss")
for k in sorted(rows):
    d = rows[k]
    print(f"  {k[0]:>5}  {k[1]:<7}  {k[2]:<5} {k[3]:<21} {d['n']:>4}  {d['re_ne_bomb']:>8}  {d['re_eq_bomb']:>8}  "
          f"{d['sc_ne_bomb']:>8}  {d['sc_eq_bomb']:>8}  {d['sc_none']:>7}  {d['re_missing']:>7}")

print("\n== 2. e2 reproduction on round 13 (pc == 20, bomb present) ==")
for grp in ("d01-07", "d12-25"):
    n = mm = 0
    for (m, r), b in bomb.items():
        if r != 13 or day_group(m) != grp or pc.get((m, r)) != 20:
            continue
        w_re = rew.get((m, r))
        if w_re is None:
            continue
        n += 1
        mm += int(w_re != b)
    print(f"  {grp}: round-13 bomb rows pc20 = {n}, round_end != bomb = {mm}")

print("\n== 3. Claim's 370 / 1,372 split: round 13 disagreements, bomb present vs absent (score scorable, pc any) ==")
cnt = collections.Counter()
for m, scm in sc.items():
    if 13 not in scm:
        continue
    st, w_sc = arith(scm, 13)
    w_re = rew.get((m, 13))
    if st != "no" or w_re is None:
        continue
    if w_sc != w_re:
        cnt["disagree_with_bomb" if (m, 13) in bomb else "disagree_no_bomb"] += 1
        if (m, 13) in bomb and bomb[(m, 13)] != w_sc:
            cnt["score_ne_bomb_in_disagree"] += 1
print(f"  scorable round-13 disagreements: {dict(cnt)}")

print("\n== 4. Round-13 bomb mismatches outside the scorable set (claim covers only scorable rows) ==")
bm = collections.Counter()
for (m, r), b in bomb.items():
    if r != 13 or pc.get((m, r)) != 20 or day_group(m) != "d12-25":
        continue
    w_re = rew.get((m, r))
    if w_re is None:
        continue
    scm = sc.get(m, {})
    st = arith(scm, 13)[0] if 13 in scm else "no_freeze_end"
    bm["mismatch" if w_re != b else "match"] += 1
    if w_re != b:
        bm[f"mismatch_status_{st}"] += 1
print(f"  d12-25 round-13 pc20 bomb rows by status: {dict(bm)}")

print("\n== 5. Round 25 (all freeze_end ties at 12-12): bomb-verified round_end vs bomb, all pc ==")
c25 = collections.Counter()
for (m, r), b in bomb.items():
    if r != 25:
        continue
    w_re = rew.get((m, r))
    scm = sc.get(m, {})
    w_side = side_attached(scm, 25) if 25 in scm else None
    c25[(day_group(m), "re!=bomb" if w_re != b else "re==bomb", "side_att_none" if w_side is None else
         ("side==bomb" if w_side == b else "side!=bomb"))] += 1
for k in sorted(c25):
    print(f"  {k[0]:<7} {k[1]:<9} {k[2]:<13} n={c25[k]}")

print("\n== 6. Round-13 side-attached winner on tied rounds (no bomb) vs round_end, by day group ==")
tie = collections.Counter()
for m, scm in sc.items():
    if 13 not in scm:
        continue
    st, _ = arith(scm, 13)
    if st not in ("both", "no_next"):
        continue
    w_side = side_attached(scm, 13)
    w_re = rew.get((m, 13))
    key = (day_group(m), st, "bomb" if (m, 13) in bomb else "no_bomb")
    if w_re is None or w_side is None:
        tie[key + ("unresolved",)] += 1
        continue
    tie[key + ("re!=side" if w_re != w_side else "re==side",)] += 1
for k in sorted(tie):
    print(f"  {k[0]:<7} {k[1]:<8} {k[2]:<8} {k[3]:<11} n={tie[k]}")

print("\n== 7. Round-13 tie status: freeze_end 13 gap in the d12-25 subset ==")
gaps = collections.Counter()
for m, scm in sc.items():
    if 13 not in scm or day_group(m) != "d12-25":
        continue
    t0, c0 = scm[13]
    gaps["tie" if t0 == c0 else ("gap1" if abs(t0 - c0) == 1 else "gap2plus")] += 1
print(f"  d12-25 freeze_end 13 gaps: {dict(gaps)}")
