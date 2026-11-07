"""D6-5 verifier, part 4.

  A. Round 9 in 16-round matches (score-derived swap after round 8): score vs round_end vs team-code-free
     evidence, compared with 24-round matches and matches without a swap at 8.
  B. Header test: starters' total rounds won from round winners (round_end, and evidence-with-fallback)
     vs header t_starters_score_final / ct_starters_score_final, per match. Header is a separate table
     (not round_end), so it is a third reference. Mapping: orientation per round from player_info
     (the claim's method), starters-T = the side-T team at the first round with >= 8 players.
  C. The claimed 540 (first player_info post-swap round, days 12-25, unscorable): decisive-evidence coverage.

Run:
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_05_header_round9.py 2024
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_05_header_round9.py 2026
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
re_ = (lazy("round_end").select(["match_key", "round", "winner_team_code"]).collect()
       .unique(["match_key", "round"], keep="first"))
bs = (lazy("bomb_state").filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "event_type"]).collect())
pinfo = (lazy("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
         .unique(["match_key", "round", "player_id"], keep="first"))
death = lazy("player_death").select(["match_key", "round", "player_id"]).collect()
header = pl.concat([pl.scan_parquet(f) for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))],
                   how="vertical_relaxed").select(["key", "t_starters_score_final", "ct_starters_score_final"]).collect()
H = {k: (t, c) for k, t, c in header.iter_rows()}

scm_all = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm_all[m][r] = (t, c)
rew = {(m, r): w for m, r, w in re_.iter_rows()}
bomb = collections.defaultdict(set)
for m, r, e in bs.iter_rows():
    bomb[(m, r)].add(e)
roster = collections.defaultdict(dict)
rm_all = collections.defaultdict(lambda: collections.defaultdict(dict))
for m, r, p, t in pinfo.iter_rows():
    rm_all[m][r][p] = t
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


def orientation(rm):
    """Per-round orientation (1 = sides flipped vs reference round); reference = first round with >= 8 players."""
    rounds = sorted(rm)
    ref = next((r for r in rounds if len(rm[r]) >= 8), rounds[0])
    t0 = rm[ref]
    out = {}
    for r in rounds:
        cur = rm[r]
        common = [p for p in cur if p in t0]
        if len(common) < 4:
            continue
        frac = sum(1 for p in common if cur[p] != t0[p]) / len(common)
        out[r] = 1 if frac > 0.5 else 0
    return out


def detect_bounds(rm):
    o = orientation(rm)
    rr = sorted(o)
    return [a for a, b in zip(rr, rr[1:]) if o[a] != o[b] and b == a + 1]


# ------------------------------------------------------------------ A. round 9 by score-derived swap at 8
print("\n== A. Round 9 by score-derived swap group (2024 or 2026) ==")
A = collections.Counter()
for m, scm in scm_all.items():
    if 9 not in scm:
        continue
    w_re = rew.get((m, 9))
    if w_re is None:
        continue
    grp = "s8_swap(16-round)" if trans(scm, 8) == "swap" else ("s8_both_or_none" if trans(scm, 8) in ("both", "none") else "no_s8")
    w_sc = win_from_score(scm, 9)
    E, src = evidence(m, 9)
    A[(grp, "SC" if w_sc is not None else "SC_none")] += 1
    if w_sc is not None and w_sc != w_re:
        A[(grp, "SC!=RE")] += 1
    if E is not None:
        A[(grp, "E_avail")] += 1
        A[(grp, "RE!=E" if w_re != E else "RE==E")] += 1
        if w_sc is not None:
            A[(grp, "SC==E" if w_sc == E else "SC!=E")] += 1
    else:
        A[(grp, f"no_E:{src}")] += 1
for k in sorted(A, key=str):
    print(f"  {str(k):<70} n={A[k]}")

# ------------------------------------------------------------------ B. header test
print("\n== B. Header test per match: starters-T / starters-CT rounds won vs header final ==")
stats = collections.Counter()
mism_examples = []
for m, rm in rm_all.items():
    if m not in H:
        stats["no_header"] += 1
        continue
    o = orientation(rm)
    if not o:
        stats["no_orientation"] += 1
        continue
    rounds = sorted(o)
    if any((m, r) not in rew for r in rounds):
        stats["incomplete_round_end"] += 1
        continue
    ref = next((r for r in sorted(rm) if len(rm[r]) >= 8), sorted(rm)[0])
    # side-T at the reference round is the starters-T team; orientation flips that team's side
    sT_re = sT_best = sT_e = 0
    defect_rounds = 0
    for r in rounds:
        side_of_starters_T = T if o[r] == 0 else CT
        w_re = rew[(m, r)]
        E, src = evidence(m, r)
        w_best = E if E is not None else w_re
        if E is not None and E != w_re:
            defect_rounds += 1
        sT_re += int(w_re == side_of_starters_T)
        sT_best += int(w_best == side_of_starters_T)
    n = len(rounds)
    ft, fc = H[m]
    sC_re, sC_best = n - sT_re, n - sT_best
    ok_re = (sT_re, sC_re) == (ft, fc)
    ok_re_sw = (sT_re, sC_re) == (fc, ft)
    ok_best = (sT_best, sC_best) == (ft, fc)
    ok_best_sw = (sT_best, sC_best) == (fc, ft)
    stats["n_matches"] += 1
    stats[f"RE_match_as_T:{ok_re}"] += 1
    stats[f"RE_match_swapped:{ok_re_sw}"] += 1
    stats[f"best_match_as_T:{ok_best}"] += 1
    stats[f"best_match_swapped:{ok_best_sw}"] += 1
    has_defect = defect_rounds > 0
    stats[f"defect_matches(E!=RE)={has_defect}: RE_match_any={ok_re or ok_re_sw}"] += 1
    stats[f"defect_matches(E!=RE)={has_defect}: best_match_any={ok_best or ok_best_sw}"] += 1
    if has_defect and not (ok_best or ok_best_sw) and len(mism_examples) < 4:
        mism_examples.append((m[-44:], n, (sT_best, sC_best), (ft, fc), defect_rounds))
for k in sorted(stats, key=str):
    print(f"  {str(k):<90} {stats[k]}")
for ex in mism_examples:
    print("   example best-mismatch:", ex)

# ------------------------------------------------------------------ C. claimed 540 coverage with decisive evidence
if ERA == "2024":
    print("\n== C. Claimed 540: player_info first post-swap round, days 12-25, unscorable: evidence coverage ==")
    C = collections.Counter()
    for m, rm in rm_all.items():
        bounds = detect_bounds(rm)
        if not bounds:
            continue
        day = m.split("/")[3]
        if not ("12" <= day <= "25"):
            continue
        rp = bounds[0] + 1
        scm = scm_all.get(m, {})
        if rp not in scm or (m, rp) not in rew:
            C["missing_row"] += 1
            continue
        if win_from_score(scm, rp) is not None:
            C["scorable"] += 1
            continue
        E, src = evidence(m, rp)
        C[("unscorable", "E:" + src)] += 1
        if E is not None:
            C[("unscorable", "RE!=E" if rew[(m, rp)] != E else "RE==E", src)] += 1
    for k in sorted(C, key=str):
        print(f"  {str(k):<70} n={C[k]}")
