"""D6-5 verifier, part 2: scope and partition of the round_end defect.

  A. Wipe rule (team-code free except roster side) vs round_end on ALL rounds, by round number, both eras.
  B. Partition of the 2024 score-vs-round_end disagreements by independent evidence (bomb / wipe / none).
  C. Reproduce the claimed 540 'unscorable first-post-swap rounds, days 12-25' using the claim's
     player_info-based swap boundaries (first detected boundary only, as in D6_swap_detect.py section 17).
  D. Round-25 bomb-vs-round_end mismatches (outside the claimed round set): match-ending? wipe evidence?

Run:
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_03_scope.py 2024
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_03_scope.py 2026
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


def detect_bounds(rm):
    """Claim's method (D6_swap_detect.detect_swaps): orientation vs reference round; swap = orientation change."""
    rounds = sorted(rm)
    ref = next((r for r in rounds if len(rm[r]) >= 8), rounds[0])
    t0 = rm[ref]
    seq = []
    for r in rounds:
        cur = rm[r]
        common = [p for p in cur if p in t0]
        if len(common) < 4:
            continue
        frac = sum(1 for p in common if cur[p] != t0[p]) / len(common)
        seq.append((r, 1 if frac > 0.5 else 0))
    return [a[0] for a, b in zip(seq, seq[1:]) if a[1] != b[1] and b[0] == a[0] + 1]


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
pc_of = {}
for m, r, w, pc in re_.select(["match_key", "round", "winner_team_code", "player_count"]).iter_rows():
    rew[(m, r)] = w
    pc_of[(m, r)] = pc
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


def wipe(m, r):
    """Returns ('wipe', winner) / ('timeout', CT) / ('planted'|'not_5v5'|'both_wiped', None)."""
    if (m, r) in planted:
        return "planted", None
    ro = roster.get((m, r), {})
    nT = sum(1 for s in ro.values() if s == T)
    nC = len(ro) - nT
    if nT != 5 or nC != 5:
        return "not_5v5", None
    dset = dead.get((m, r), set())
    wT = all(p in dset for p, s in ro.items() if s == T)
    wC = all(p in dset for p, s in ro.items() if s == CT)
    if wT and wC:
        return "both_wiped", None
    if wT:
        return "wipe", CT
    if wC:
        return "wipe", T
    return "timeout", CT


# ------------------------------------------------------------------ A. wipe rule vs round_end, all rounds
print("\n== A. Wipe rule vs round_end on ALL rounds (5v5, no bomb planted), by round number ==")
A = collections.Counter()
A_round_wipe_dis = collections.Counter()
A_round_wipe_n = collections.Counter()
A_round_timeout_dis = collections.Counter()
A_round_timeout_n = collections.Counter()
for (m, r), w_re in rew.items():
    kind, imp = wipe(m, r)
    if kind in ("planted", "not_5v5", "both_wiped"):
        A[f"skip_{kind}"] += 1
        continue
    if kind == "wipe":
        A["wipe_n"] += 1
        A_round_wipe_n[r] += 1
        if imp != w_re:
            A["wipe_disagree_round_end"] += 1
            A_round_wipe_dis[r] += 1
    else:
        A["timeout_n"] += 1
        A_round_timeout_n[r] += 1
        if imp != w_re:
            A["timeout_disagree_round_end"] += 1
            A_round_timeout_dis[r] += 1
print(f"  totals: {dict(A)}")
print("  wipe-decisive disagreements by round (round: disagree/n):")
print("   ", {r: f"{A_round_wipe_dis[r]}/{A_round_wipe_n[r]}" for r in sorted(A_round_wipe_n) if A_round_wipe_dis[r]})
print("  wipe-decisive agreement, rounds with n>0 (count of rounds where all wipe-decisive rounds agree):",
      sum(1 for r in A_round_wipe_n if A_round_wipe_dis[r] == 0))
print("  timeout-rule disagreements by round (round: disagree/n), top 12 by disagree count:")
print("   ", sorted(((r, f"{A_round_timeout_dis[r]}/{A_round_timeout_n[r]}") for r in A_round_timeout_n if A_round_timeout_dis[r]),
                    key=lambda x: -int(x[1].split('/')[0]))[:12])

# ------------------------------------------------------------------ B. partition of score-vs-round_end disagreements
print("\n== B. Partition of score-vs-round_end disagreements by independent evidence ==")
dis = []
for m, scm in scm_all.items():
    for r in scm:
        w_sc = win_from_score(scm, r)
        w_re = rew.get((m, r))
        if w_sc is not None and w_re is not None and w_sc != w_re:
            dis.append((m, r, w_sc, w_re))
print(f"  disagreements: {len(dis)}")
B = collections.Counter()
B_by_round = collections.Counter()
for m, r, w_sc, w_re in dis:
    bsd = bomb_side(m, r)
    kind, imp = wipe(m, r)
    if bsd is not None:
        key = f"bomb_known(bomb==score:{bsd == w_sc})"
    elif kind == "wipe":
        key = f"wipe_decisive(wipe==score:{imp == w_sc}, wipe==round_end:{imp == w_re})"
    elif kind == "timeout":
        key = f"timeout(rule==score:{imp == w_sc}, rule==round_end:{imp == w_re})"
    else:
        key = f"unresolved({kind})"
    B[key] += 1
    B_by_round[(r, key.split("(")[0])] += 1
for k in sorted(B, key=str):
    print(f"  {k:<70} n={B[k]}")
print("  by round:", dict(sorted(B_by_round.items(), key=str)))

# ------------------------------------------------------------------ C. claimed 540 (player_info first-post-swap, days 12-25)
if ERA == "2024":
    print("\n== C. Claimed 540: first detected (player_info) post-swap round, days 12-25, unscorable ==")
    C = collections.Counter()
    C_all = collections.Counter()
    for m, rm in rm_all.items():
        bounds = detect_bounds(rm)
        if not bounds:
            continue
        day = m.split("/")[3]
        s1 = bounds[0]
        rp = s1 + 1
        scm = scm_all.get(m, {})
        if rp not in scm:
            C_all["no_freeze_end_at_post_round"] += 1
            continue
        w_sc = win_from_score(scm, rp)
        w_re = rew.get((m, rp))
        if w_sc is not None and w_re is not None:
            C_all["scorable"] += 1
            continue
        has_bomb = bomb_side(m, rp) is not None
        kind, imp = wipe(m, rp)
        status = "wipe_decisive" if kind == "wipe" else kind
        C[(day, "unscorable")] += 1
        if "12" <= day <= "25":
            C_all[("days12_25_unscorable")] += 1
            C_all[("days12_25 bomb_outcome", has_bomb)] += 1
            C_all[("days12_25 evidence", "bomb" if has_bomb else status)] += 1
            if not has_bomb and kind == "wipe":
                C_all[("days12_25 wipe-implied winner == round_end", imp == w_re if w_re is not None else "no_re")] += 1
    print(f"  unscorable first-post-swap rounds, days 12-25 (player_info boundaries, first only): {C_all[('days12_25_unscorable')]}")
    for k in sorted(C_all, key=str):
        if isinstance(k, tuple):
            print(f"    {str(k):<70} n={C_all[k]}")
    print(f"  scorable first-post-swap (score and round_end usable), all days: {C_all['scorable']}")
    print(f"  first-post-swap rounds without freeze_end at r+1 position: {C_all['no_freeze_end_at_post_round']}")

# ------------------------------------------------------------------ D. round-25 bomb mismatches
print("\n== D. Bomb-vs-round_end mismatches at round 25 (player_count==20) ==")
D = collections.Counter()
for (m, r), bsd in ((k, bomb_side(*k)) for k in bomb):
    if r != 25 or bsd is None:
        continue
    w_re = rew.get((m, r))
    if w_re is None or pc_of.get((m, r)) != 20:
        continue
    scm = scm_all.get(m, {})
    has26 = 26 in scm
    kind, imp = wipe(m, r)
    if bsd != w_re:
        D[("bomb!=round_end", f"round26_exists={has26}")] += 1
        if kind == "wipe":
            D[("bomb!=round_end", "wipe_decisive", f"wipe==bomb:{imp == bsd}", f"wipe==round_end:{imp == w_re}")] += 1
        else:
            D[("bomb!=round_end", kind)] += 1
    else:
        D[("bomb==round_end", f"round26_exists={has26}")] += 1
for k in sorted(D, key=str):
    print(f"  {str(k):<100} n={D[k]}")
