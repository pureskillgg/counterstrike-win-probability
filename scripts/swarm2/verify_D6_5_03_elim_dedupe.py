"""D6-5 verifier step 3: elimination test (independent of bomb and score) and content-level duplicate check.

Elimination test (rounds with no bomb_planted row, so the round cannot have ended by a bomb):
  roster(side, r) = distinct player_ids with that team_code in player_info for (match, round)
  wiped(side, r)  = roster nonempty and every roster player appears as a victim in player_death for (match, round)
  elim winner     = the side that is NOT wiped, when exactly one side is wiped (loser = wiped side)
Compare elim winner with round_end winner and with the score-based winner (step 1/2 rules, side-attached for ties).
ASSUMPTION E1: a wiped side is the losing side of an elimination round. Baseline (d01-07) checks this.
Content duplicates: fingerprint = per-match sequence of (round, round_end tick, winner); copies across keys.
Run: G:\pureskill\csds-cs2\.pixi\envs\default\python.exe -I -X utf8 scripts\swarm2\verify_D6_5_03_elim_dedupe.py
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


def score_winner(scm, r):
    """(status, winner): scorable from arithmetic; 'both' resolved side-attached only when gap is 0 (tie)."""
    if r + 1 not in scm:
        return "no_next", None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    dT, dC = t1 - t0, c1 - c0
    no = (dT, dC) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and sw:
        if t0 == c0:
            return "tie", (T if (dT, dC) == (1, 0) else CT) if no else None
        return "both", None
    if no:
        return "scorable", (T if (dT, dC) == (1, 0) else CT)
    if sw:
        return "scorable", (T if (t1 - c0, c1 - t0) == (0, 1) else CT)
    return "inconsistent", None


# --- inputs
PI = read("player_info", ["match_key", "round", "player_id", "team_code"]).unique(["match_key", "round", "player_id"], keep="first")
roster = collections.defaultdict(lambda: collections.defaultdict(set))
for m, r, p, t in PI.iter_rows():
    roster[(m, r)][t].add(p)
PD = read("player_death", ["match_key", "round", "player_id"])
victims = collections.defaultdict(set)
for m, r, p in PD.iter_rows():
    if p is not None:
        victims[(m, r)].add(p)
BS = read("bomb_state", ["match_key", "round", "event_type"])
plant_rounds = set(BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).iter_rows())
outcome = {}
for m, r, ev in BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).select(
        ["match_key", "round", "event_type"]).iter_rows():
    outcome[(m, r)] = CT if ev == "bomb_defused" else T


def wiped(m, r, side):
    ros = roster.get((m, r), {}).get(side, set())
    if not ros:
        return None
    return ros <= victims.get((m, r), set())


def elim_winner(m, r):
    wT, wC = wiped(m, r, T), wiped(m, r, CT)
    if wT is None or wC is None:
        return "no_roster", None
    if wT and not wC:
        return "elim", CT
    if wC and not wT:
        return "elim", T
    if wT and wC:
        return "both_wiped", None
    return "neither_wiped", None


print("== 0. Roster size distribution (distinct player_ids per side per round, rounds with freeze_end) ==")
rs = collections.Counter()
for m, scm in sc.items():
    for r in scm:
        for side in (T, CT):
            rs[len(roster.get((m, r), {}).get(side, set()))] += 1
print(f"  roster sizes: {dict(sorted(rs.items()))}")

print("\n== 1. Elimination test, no-bomb-plant rounds, rounds 9 / 13 / 25 / 28 and controls ==")
tab = collections.Counter()
for m, scm in sc.items():
    for r in scm:
        if (m, r) in plant_rounds:
            continue
        st, w_sc = score_winner(scm, r)
        w_re = rew.get((m, r))
        est, w_el = elim_winner(m, r)
        if est != "elim":
            tab[(r if r in (9, 13, 25, 28) else "ctrl", day_group(m), "elim_unavailable", est)] += 1
            continue
        key_r = r if r in (9, 13, 25, 28) else "ctrl"
        if w_re is not None:
            tab[(key_r, day_group(m), "re_vs_elim", "agree" if w_re == w_el else "disagree")] += 1
        if w_sc is not None:
            tab[(key_r, day_group(m), "score_vs_elim", "agree" if w_sc == w_el else "disagree")] += 1
        if w_re is not None and w_sc is not None and w_re != w_sc:
            tab[(key_r, day_group(m), "re_ne_score_rows", "elim_matches_score" if w_sc == w_el else
                 ("elim_matches_re" if w_re == w_el else "elim_matches_neither"))] += 1
for k in sorted(tab, key=lambda x: (str(x[0]), x[1], x[2], x[3])):
    print(f"  r={str(k[0]):<5} {k[1]:<7} {k[2]:<17} {k[3]:<22} n={tab[k]}")

print("\n== 2. Round-13 no-bomb disagreements: which side is wiped (score winner vs round_end winner) ==")
w13 = collections.Counter()
for m, scm in sc.items():
    if 13 not in scm or (m, 13) in plant_rounds or day_group(m) != "d12-25":
        continue
    st, w_sc = score_winner(scm, 13)
    w_re = rew.get((m, 13))
    if w_sc is None or w_re is None or w_sc == w_re:
        continue
    wt, wc = wiped(m, 13, T), wiped(m, 13, CT)
    who = "none" if wt is None else ("T" if wt and not wc else "CT" if wc and not wt else "both" if wt and wc else "neither")
    w13[(st, "wiped=" + who)] += 1
print(f"  d12-25 round-13 no-bomb disagreement rows (scorable, score != round_end): {dict(w13)}")

print("\n== 3. Content-level duplicate check (per-match fingerprint over round, tick, winner) ==")
RE_full = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
RE_full = RE_full.unique(["match_key", "round"], keep="first").sort(["match_key", "round"])
fp = collections.defaultdict(list)
for m, grp in RE_full.group_by("match_key", maintain_order=True):
    key = m if isinstance(m, str) else m[0]
    seq = tuple((r, t, w, rr) for r, t, w, rr in grp.select(["round", "tick", "winner_team_code", "win_reason_code"]).iter_rows())
    fp[seq].append(key)
groups = [v for v in fp.values() if len(v) > 1]
copies = sum(len(v) - 1 for v in groups)
print(f"  fingerprint groups with >1 match_key: {len(groups)}; extra copies: {copies}; unique matches: {len(fp)}")
dup_later = set()
for v in groups:
    for k in sorted(v)[1:]:
        dup_later.add(k)
dd = collections.Counter(day_group(m) for m in dup_later)
print(f"  later-copy keys by day group: {dict(dd)}")
S["dup_later_fp"] = dup_later
with open(SCRATCH / "verify_D6_5_state.pkl", "wb") as f:
    pickle.dump({**S, "dup_later_fp": dup_later}, f)

print("\n== 4. Disagreements (scorable, round 9/13/28) after removing later content copies ==")
dis = collections.Counter()
for m, scm in sc.items():
    for r in (9, 13, 25, 28):
        if r not in scm:
            continue
        st, w_sc = score_winner(scm, r)
        w_re = rew.get((m, r))
        if st != "scorable" or w_re is None:
            continue
        if w_sc != w_re:
            dis[(r, "copy" if m in dup_later else "primary")] += 1
print(f"  scorable disagreements by round and copy status: {dict(sorted(dis.items()))}")
