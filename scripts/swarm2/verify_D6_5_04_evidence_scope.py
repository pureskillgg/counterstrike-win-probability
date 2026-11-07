"""D6-5 verifier, part 3: full-population defect scope using team-code-free evidence.

Evidence winner E(m, r), in priority order (none of these use round_end.winner_team_code):
  1. bomb outcome event: bomb_defused -> CT (3), bomb_exploded -> T (2)
  2. planted, no outcome event: if all five CT players are dead and not all T players -> T wins
     (after a plant, CT can only win by defusing; a CT wipe with no defuse ends the round for T)
  3. not planted, clean 5v5 roster: exactly one side fully dead -> the other side wins
  4. not planted, nobody wiped: time-out -> CT wins (REPORTED SEPARATELY; calibration is weaker)
Score winner SC(m, r): from round_state round_freeze_end t_score/ct_score (None when the transition
is 'both' or 'none'; every round that starts from a tied score is 'both' by construction, so it is None).

Outputs (per era):
  - RE vs E over all rounds, by source and by round number (the defect scope, lower bound)
  - split by tie-start (freeze_end t == c) vs non-tie-start, so the score method's blind spot is visible
  - calibration: E vs RE on rounds where SC == RE (both assumed correct)
  - SC vs E where both exist

Run:
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_04_evidence_scope.py 2024
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_04_evidence_scope.py 2026
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
T, CT = 2, 3
BASES = {
    "2024": Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"),
    "2026": Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"),
}
ERA = sys.argv[1] if len(sys.argv) > 1 else "2024"
BASE = BASES[ERA]


def lazy(ch):
    dirs = sorted(BASE.glob(f"basic_{ch}.*"))
    assert len(dirs) == 1, (ch, dirs)
    fs = sorted(glob.glob(str(dirs[0] / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


def trans(scm, r):
    if r not in scm or (r + 1) not in scm:
        return None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    no = (t1 - t0, c1 - c0) in ((1, 0), (0, 1))
    sw = (t1 - c0, c1 - t0) in ((1, 0), (0, 1))
    if no and sw:
        return "both"
    if no:
        return "no"
    if sw:
        return "swap"
    return "none"


def win_from_score(scm, r):
    lab = trans(scm, r)
    if lab not in ("no", "swap"):
        return None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    if lab == "no":
        return T if (t1 - t0) == 1 else CT
    return T if (c1 - t0) == 1 else CT


print(f"=== ERA {ERA}")
fe = (lazy("round_state").filter(pl.col("event_type") == "round_freeze_end")
      .select(["match_key", "round", "t_score", "ct_score"]).collect()
      .unique(["match_key", "round"], keep="first"))
re_ = (lazy("round_end").select(["match_key", "round", "winner_team_code", "player_count"]).collect()
       .unique(["match_key", "round"], keep="first"))
bs = (lazy("bomb_state").filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "event_type"]).collect())
pinfo = (lazy("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
         .unique(["match_key", "round", "player_id"], keep="first"))
death = lazy("player_death").select(["match_key", "round", "player_id"]).collect()

scm_all = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm_all[m][r] = (t, c)
rew = {}
for m, r, w, pc in re_.select(["match_key", "round", "winner_team_code", "player_count"]).iter_rows():
    rew[(m, r)] = (w, pc)
bomb = collections.defaultdict(set)
for m, r, e in bs.iter_rows():
    bomb[(m, r)].add(e)
roster = collections.defaultdict(dict)
for m, r, p, t in pinfo.iter_rows():
    if t in (T, CT):
        roster[(m, r)][p] = t
dead = collections.defaultdict(set)
for m, r, p in death.iter_rows():
    dead[(m, r)].add(p)
planted = {k for k, v in bomb.items() if "bomb_planted" in v}


def bomb_side(m, r):
    ev = bomb.get((m, r), set())
    if "bomb_defused" in ev:
        return CT
    if "bomb_exploded" in ev:
        return T
    return None


def evidence(m, r):
    bsd = bomb_side(m, r)
    if bsd is not None:
        return bsd, "bomb"
    ro = roster.get((m, r), {})
    nT = sum(1 for s in ro.values() if s == T)
    nC = len(ro) - nT
    ok55 = nT == 5 and nC == 5
    dset = dead.get((m, r), set())
    wT = ok55 and all(p in dset for p, s in ro.items() if s == T)
    wC = ok55 and all(p in dset for p, s in ro.items() if s == CT)
    if (m, r) in planted:
        if wC and not wT:
            return T, "planted_ct_wipe"
        return None, "planted_unresolved"
    if not ok55:
        return None, "not_5v5"
    if wT and wC:
        return None, "both_wiped"
    if wT:
        return CT, "wipe"
    if wC:
        return T, "wipe"
    return CT, "timeout"


# ------------------------------------------------------------------ build per-round table over all round_end rows
rows = []
for (m, r), (w_re, pc) in rew.items():
    scm = scm_all.get(m, {})
    w_sc = win_from_score(scm, r) if scm else None
    tie_start = (r in scm) and (scm[r][0] == scm[r][1])
    E, src = evidence(m, r)
    day = m.split("/")[3] if ERA == "2024" else "all"
    rows.append((m, r, day, w_re, w_sc, tie_start, E, src, r in scm))

print(f"round_end rows (unique match,round): {len(rows)}")
print(f"rows with freeze_end present: {sum(1 for x in rows if x[8])}; tie-start (t==c at freeze_end): "
      f"{sum(1 for x in rows if x[5])}")

# ------------------------------------------------------------------ A. evidence coverage and RE vs E
print("\n== A. round_end vs evidence winner E, all rounds (team-code free evidence only) ==")
A = collections.Counter()
for m, r, day, w_re, w_sc, tie, E, src, hasfe in rows:
    A[("src", src)] += 1
    if E is None:
        continue
    A[("E_avail", src)] += 1
    A[("RE==E" if w_re == E else "RE!=E", src)] += 1
for k in sorted(A, key=str):
    print(f"  {str(k):<45} n={A[k]}")
tot_E = sum(v for k, v in A.items() if k[0] == "E_avail")
tot_dis = sum(v for k, v in A.items() if k[0] == "RE!=E")
print(f"  total rounds with evidence E: {tot_E}; RE!=E: {tot_dis} ({100.0 * tot_dis / tot_E:.2f}% of evidence rounds)")

# ------------------------------------------------------------------ B. RE != E by round number and tie state
print("\n== B. RE != E by absolute round number (all sources) and by tie-start flag ==")
B_n = collections.Counter()
B_dis = collections.Counter()
B_tie = collections.Counter()
for m, r, day, w_re, w_sc, tie, E, src, hasfe in rows:
    if E is None:
        continue
    B_n[r] += 1
    if w_re != E:
        B_dis[r] += 1
        B_tie[("tie_start" if tie else "non_tie", "RE!=E")] += 1
    else:
        B_tie[("tie_start" if tie else "non_tie", "RE==E")] += 1
top = sorted(B_dis.items(), key=lambda kv: -kv[1])[:14]
print("  round: RE!=E / evidence n (top 14 by count):")
print("   ", [(r, f"{d}/{B_n[r]}") for r, d in top])
print("  by tie-start flag:", dict(sorted(B_tie.items())))
if ERA == "2024":
    print("  round-number groups (count RE!=E, count evidence):")
    for lo, hi in [(1, 8), (9, 9), (10, 12), (13, 13), (14, 24), (25, 25), (26, 28), (29, 99)]:
        d = sum(B_dis[r] for r in range(lo, hi + 1))
        n = sum(B_n[r] for r in range(lo, hi + 1))
        print(f"    rounds {lo:>2}-{hi:<2}: {d}/{n}")

# ------------------------------------------------------------------ C. by day group for 2024 (rounds 9, 13, 25, 28 only, others pooled)
if ERA == "2024":
    print("\n== C. RE!=E by match day, rounds 9/13/25/28 vs all other rounds ==")
    C = collections.Counter()
    for m, r, day, w_re, w_sc, tie, E, src, hasfe in rows:
        if E is None:
            continue
        grp = f"r{r}" if r in (9, 13, 25, 28) else "other"
        C[(day, grp, "dis" if w_re != E else "ok")] += 1
    days = sorted({k[0] for k in C})
    print("  day   r9(dis/n)    r13(dis/n)   r25(dis/n)   r28(dis/n)   other(dis/n)")
    for dd in days:
        cells = []
        for grp in ("r9", "r13", "r25", "r28", "other"):
            d = C[(dd, grp, "dis")]
            n = d + C[(dd, grp, "ok")]
            cells.append(f"{d}/{n}")
        print(f"  {dd}   " + "   ".join(f"{c:<10}" for c in cells))

# ------------------------------------------------------------------ D. calibration and SC vs E
print("\n== D. Calibration: E vs RE where SC == RE (both presumed right); SC vs E where both exist ==")
D = collections.Counter()
for m, r, day, w_re, w_sc, tie, E, src, hasfe in rows:
    if w_sc is None:
        continue
    if w_sc == w_re and E is not None:
        D[("SC==RE, E avail", src, "E==RE" if E == w_re else "E!=RE")] += 1
    if w_sc != w_re and E is not None:
        D[("SC!=RE, E avail", src, "E==SC" if E == w_sc else ("E==RE" if E == w_re else "?"))] += 1
    if w_sc != w_re and E is None:
        D[("SC!=RE, E none", src)] += 1
for k in sorted(D, key=str):
    print(f"  {str(k):<90} n={D[k]}")

# ------------------------------------------------------------------ E. unscorable rounds: evidence coverage
print("\n== E. Rounds the score method cannot score (w_sc None): evidence coverage and RE!=E ==")
Ev = collections.Counter()
for m, r, day, w_re, w_sc, tie, E, src, hasfe in rows:
    if w_sc is not None:
        continue
    Ev[("tie_start" if tie else "non_tie", "E avail" if E is not None else "no E")] += 1
    if E is not None:
        Ev[("tie_start" if tie else "non_tie", "RE!=E" if w_re != E else "RE==E")] += 1
for k in sorted(Ev, key=str):
    print(f"  {str(k):<50} n={Ev[k]}")
print(f"  unscorable rounds total: {sum(1 for x in rows if x[4] is None)}")
