"""D6-5 verifier step 4: scope of the defect across all rounds, including rows the claim calls unscorable.

Three resolution rules, each checked against bomb and elimination evidence:
  R1 scorable (arithmetic, no swap assumption needed)          -> claim's population
  R2 gap-1 'both' (aligned increments): no-swap winner == swap winner (checked here for every row)
  R3 gap-0 tie: side-attached winner (ASSUMPTION A1: no side swap between r and r+1; checked on R1/R2 rows and on bomb rows)
Last-round rows (no freeze_end r+1) are reported as unresolved by score.
Bomb: bomb_defused -> 3, bomb_exploded -> 2. Elimination: as step 3 (no bomb plant in the round).
Run: G:\pureskill\csds-cs2\.pixi\envs\default\python.exe -I -X utf8 scripts\swarm2\verify_D6_5_04_scope.py
"""
import collections
import glob
import pickle
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
T, CT = 2, 3


def files(ch):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


with open(SCRATCH / "verify_D6_5_state.pkl", "rb") as f:
    S = pickle.load(f)
sc, rew, pc, dup_later = S["sc"], S["rew"], S["pc"], S["dup_later_fp"]


def day_group(m):
    d = int(m.split("/")[3])
    return "d01-07" if d <= 7 else "d12-25" if 12 <= d <= 25 else "other"


def classify(scm, r):
    """Return dict: kind in {no_next, scorable, gap1_both, tie, incons}; winner_arith; winner_noswap; winner_swap."""
    if r + 1 not in scm:
        return {"kind": "no_next"}
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    dT, dC = t1 - t0, c1 - c0
    no = (dT, dC) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    w_no = (T if (dT, dC) == (1, 0) else CT) if no else None
    w_sw = (T if (t1 - c0, c1 - t0) == (0, 1) else CT) if sw else None
    if no and sw:
        kind = "tie" if t0 == c0 else "gap1_both"
    elif no:
        kind = "scorable"
    elif sw:
        kind = "scorable"
    else:
        kind = "incons"
    return {"kind": kind, "w_no": w_no, "w_sw": w_sw, "gap": abs(t0 - c0)}


# bomb outcomes, planted rounds, elimination inputs
BS = read("bomb_state", ["match_key", "round", "event_type"])
plant_rounds = set(BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).iter_rows())
outcome = {}
for m, r, ev in BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).select(
        ["match_key", "round", "event_type"]).iter_rows():
    outcome[(m, r)] = CT if ev == "bomb_defused" else T
plant_no_outcome = plant_rounds - set(outcome)
PI = read("player_info", ["match_key", "round", "player_id", "team_code"]).unique(["match_key", "round", "player_id"], keep="first")
roster = collections.defaultdict(lambda: collections.defaultdict(set))
for m, r, p, t in PI.iter_rows():
    roster[(m, r)][t].add(p)
PD = read("player_death", ["match_key", "round", "player_id"])
victims = collections.defaultdict(set)
for m, r, p in PD.iter_rows():
    if p is not None:
        victims[(m, r)].add(p)


def elim_winner(m, r):
    ros_t, ros_c = roster.get((m, r), {}).get(T, set()), roster.get((m, r), {}).get(CT, set())
    if not ros_t or not ros_c:
        return None
    wT = ros_t <= victims.get((m, r), set())
    wC = ros_c <= victims.get((m, r), set())
    if wT and not wC:
        return CT
    if wC and not wT:
        return T
    return None


print("== 0. Plant without outcome row (bomb_planted but no defused/exploded row) ==")
print(f"  planted rounds: {len(plant_rounds)}; with outcome: {len(plant_rounds & set(outcome))}; planted without outcome: {len(plant_no_outcome)}")
pno = collections.Counter(day_group(m) for m, _ in plant_no_outcome)
print(f"  planted-without-outcome by day group: {dict(pno)}; by round (top 6): "
      f"{collections.Counter(r for _, r in plant_no_outcome).most_common(6)}")

print("\n== 1. R2 check: gap-1 'both' rows, no-swap winner vs swap winner (must be identical if the claim's gap-1 is determinate) ==")
chk = collections.Counter()
for m, scm in sc.items():
    for r in scm:
        c = classify(scm, r)
        if c["kind"] == "gap1_both":
            chk["same" if c["w_no"] == c["w_sw"] else "differ"] += 1
print(f"  gap-1 'both' rows: {dict(chk)}")

print("\n== 2. Rows by kind and by round: round_end vs resolved winner (R1 scorable; R2 gap-1 both; R3 tie side-attached) ==")
print("  round  kind        day      n  re!=w   bomb_n  bomb_re!=bomb  elim_n  elim!=re  elim!=w")
agg = collections.defaultdict(collections.Counter)
for m, scm in sc.items():
    for r in scm:
        c = classify(scm, r)
        kind = c["kind"]
        if kind == "no_next":
            key = (r, "no_next", day_group(m))
            agg[key]["n"] += 1
            if (m, r) in outcome:
                agg[key]["bomb_n"] += 1
                agg[key]["bomb_re!=bomb"] += int(rew.get((m, r)) != outcome[(m, r)])
            continue
        if kind in ("scorable", "gap1_both", "tie"):
            w = c["w_no"] if kind != "scorable" else (c["w_no"] if c["w_no"] is not None else c["w_sw"])
        else:
            continue
        if w is None:
            continue
        key = (r, kind, day_group(m))
        w_re = rew.get((m, r))
        if w_re is None:
            agg[key]["re_missing"] += 1
            continue
        agg[key]["n"] += 1
        agg[key]["re!=w"] += int(w_re != w)
        if (m, r) in outcome:
            agg[key]["bomb_n"] += 1
            agg[key]["bomb_re!=bomb"] += int(w_re != outcome[(m, r)])
            agg[key]["bomb_w!=bomb"] += int(w != outcome[(m, r)])
        if (m, r) not in plant_rounds:
            e = elim_winner(m, r)
            if e is not None:
                agg[key]["elim_n"] += 1
                agg[key]["elim!=re"] += int(e != w_re)
                agg[key]["elim!=w"] += int(e != w)
for key in sorted(agg, key=lambda x: (x[0], x[1], x[2])):
    d = agg[key]
    if key[0] not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25, 27, 28, 29, 30):
        continue
    if d["n"] == 0 and d["bomb_n"] == 0:
        continue
    print(f"  {key[0]:>5}  {key[1]:<10}  {key[2]:<7} {d['n']:>5}  {d['re!=w']:>5}  {d['bomb_n']:>6}  {d['bomb_re!=bomb']:>13}  "
          f"{d['elim_n']:>6}  {d['elim!=re']:>8}  {d['elim!=w']:>7}")

print("\n== 3. Bomb mismatches by round (all rounds, d12-25 and d01-07): round_end != bomb ==")
bm = collections.Counter()
for (m, r), b in outcome.items():
    w_re = rew.get((m, r))
    if w_re is None:
        continue
    bm[(r, day_group(m), "mismatch" if w_re != b else "match")] += 1
mm_rounds = collections.Counter()
for (r, g, s), n in bm.items():
    if s == "mismatch":
        mm_rounds[(r, g)] += n
print(f"  mismatches by (round, day group): {dict(sorted(mm_rounds.items()))}")
tot_mm = sum(v for k, v in mm_rounds.items())
print(f"  total bomb mismatches (all rounds, all pc): {tot_mm} of {len(outcome)} bomb-outcome rounds")

print("\n== 4. Round-13 tie and no-next rows, d12-25, no bomb: claim's 540 candidates among first-post-swap rounds (9, 13, 28) ==")
cand = collections.Counter()
for rr in (9, 13, 28):
    for m, scm in sc.items():
        if rr not in scm or day_group(m) != "d12-25":
            continue
        c = classify(scm, rr)
        if (m, rr) in outcome:
            continue
        if c["kind"] in ("tie", "gap1_both", "no_next"):
            cand[(rr, c["kind"])] += 1
print(f"  unscorable-or-tie first-post-swap rows in d12-25 without bomb (kind counts): {dict(cand)}; total {sum(cand.values())}")

print("\n== 5. Control-round noise: elimination vs score winner, all non-defect rounds, d12-25 (by round, decisive only) ==")
ctrl = collections.Counter()
for m, scm in sc.items():
    if day_group(m) != "d12-25":
        continue
    for r in scm:
        if r in (9, 13, 25, 28) or (m, r) in plant_rounds:
            continue
        c = classify(scm, r)
        if c["kind"] not in ("scorable", "tie"):
            continue
        w = c["w_no"] if c["kind"] == "tie" else (c["w_no"] if c["w_no"] is not None else c["w_sw"])
        e = elim_winner(m, r)
        if e is None or w is None:
            continue
        ctrl[(r, "agree" if e == w else "disagree")] += 1
bad = collections.Counter()
for (r, s), n in ctrl.items():
    if s == "disagree":
        bad[r] += n
print(f"  control disagreements (elim vs score) by round: {dict(sorted(bad.items()))}")
print(f"  control total decisive: {sum(ctrl.values())}; disagreements: {sum(bad.values())}")

print("\n== 6. Dedupe effect: claim's 2,042 after removing later content copies (scorable R1 only) ==")
dd = collections.Counter()
for m, scm in sc.items():
    for r in scm:
        c = classify(scm, r)
        if c["kind"] != "scorable":
            continue
        w = c["w_no"] if c["w_no"] is not None else c["w_sw"]
        w_re = rew.get((m, r))
        if w_re is None:
            continue
        dd[("copy" if m in dup_later else "primary", "disagree" if w != w_re else "agree")] += 1
print(f"  scorable R1 counts by copy status: {dict(dd)}")

print("\n== 7. Round-13 defect rows: round_end tick inside round 13 freeze window (tick-window check) ==")
RS = read("round_state", ["match_key", "round", "event_type", "tick"])
fe_tick = {}
for m, r, t in RS.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "tick"]).iter_rows():
    fe_tick.setdefault((m, r), t)
RE_T = read("round_end", ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
re_tick = {(m, r): t for m, r, t in RE_T.iter_rows()}
win = collections.Counter()
for m, scm in sc.items():
    if day_group(m) != "d12-25":
        continue
    for rr in (13,):
        if (m, rr) not in outcome or rr not in scm:
            continue
        a, b = fe_tick.get((m, rr)), fe_tick.get((m, rr + 1))
        t = re_tick.get((m, rr))
        if a is None or t is None:
            win["missing"] += 1
            continue
        inside = (b is None) or (a <= t < b)
        win["in_window" if inside else "outside"] += 1
print(f"  d12-25 round-13 bomb rows: round_end tick in own window: {dict(win)}")
