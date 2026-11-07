"""D6-5 verifier, part 5: propagation of a wrong round winner through the cumulative score lead (2024).

The bake-off lead is cumulative over round winners (see scripts/model/sufficiency/score_lead_fix.py header:
'winner adds one to the winning team'). A wrong winner at round d therefore makes the lead wrong for every
later round of the same match (ASSUMPTION: the feature at live start of round r uses winners of rounds < r).
This script measures:
  - decisive defect set (team-code-free evidence: bomb, planted CT-wipe, wipe; timeouts excluded)
  - score-based defect set (SC != RE on scorable rounds, the claim's 2,042)
  - matches affected, and round rows whose lead is downstream of the first defect (r > first defect round)
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_06_propagation.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
T, CT = 2, 3
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")


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


fe = (lazy("round_state").filter(pl.col("event_type") == "round_freeze_end")
      .select(["match_key", "round", "t_score", "ct_score"]).collect().unique(["match_key", "round"], keep="first"))
re_ = lazy("round_end").select(["match_key", "round", "winner_team_code"]).collect().unique(["match_key", "round"], keep="first")
bs = (lazy("bomb_state").filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "event_type"]).collect())
pinfo = (lazy("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
         .unique(["match_key", "round", "player_id"], keep="first"))
death = lazy("player_death").select(["match_key", "round", "player_id"]).collect()

scm_all = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm_all[m][r] = (t, c)
rew = {(m, r): w for m, r, w in re_.iter_rows()}
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


def evidence(m, r):
    ev = bomb.get((m, r), set())
    if "bomb_defused" in ev:
        return CT, "bomb"
    if "bomb_exploded" in ev:
        return T, "bomb"
    ro = roster.get((m, r), {})
    nT = sum(1 for s in ro.values() if s == T)
    ok55 = nT == 5 and len(ro) - nT == 5
    dset = dead.get((m, r), set())
    wT = ok55 and all(p in dset for p, s in ro.items() if s == T)
    wC = ok55 and all(p in dset for p, s in ro.items() if s == CT)
    if (m, r) in planted:
        return (T, "planted_ct_wipe") if (wC and not wT) else (None, "planted_unresolved")
    if not ok55:
        return None, "not_5v5"
    if wT and wC:
        return None, "both_wiped"
    if wT:
        return CT, "wipe"
    if wC:
        return T, "wipe"
    return CT, "timeout"


dec_def = collections.defaultdict(list)   # match -> defect rounds (decisive evidence)
sc_def = collections.defaultdict(list)    # match -> defect rounds (score-vs-round_end, claim's set)
for (m, r), w_re in rew.items():
    E, src = evidence(m, r)
    if E is not None and src != "timeout" and E != w_re:
        dec_def[m].append(r)
    scm = scm_all.get(m, {})
    w_sc = win_from_score(scm, r) if scm else None
    if w_sc is not None and w_sc != w_re:
        sc_def[m].append(r)

all_rows = collections.Counter(m for (m, r) in rew)
total_rows = sum(all_rows.values())


def summarize(name, defs):
    aff = {m: min(rs) for m, rs in defs.items() if rs}
    down = sum(sum(1 for r in range(f + 1, 200) if (m, r) in rew) for m, f in aff.items())
    down_rows = sum(1 for (m, r) in rew if m in aff and r > aff[m])
    print(f"  {name}: defect rounds={sum(len(v) for v in defs.values())}, matches affected={len(aff)} "
          f"of {len(all_rows)} matches; round rows downstream of first defect (r > first defect) = {down_rows} "
          f"= {100.0 * down_rows / total_rows:.2f}% of {total_rows} round_end rows")
    first = collections.Counter(f for f in aff.values())
    print(f"    first defect round distribution (top 8): {first.most_common(8)}")
    return aff


print("== Propagation through the cumulative lead (2024 tome) ==")
aff_dec = summarize("decisive-evidence defects (bomb/planted_ct_wipe/wipe)", dec_def)
aff_sc = summarize("score-vs-round_end defects (claim's set, scorable only)", sc_def)
both = set(aff_dec) | set(aff_sc)
print(f"  union of matches affected by either: {len(both)}")
print("  NOTE: the lead is only wrong for rounds downstream of the first wrong winner; a single flipped round")
print("  flips the lead for all later rounds in that match, so the affected share of states exceeds the 2.4%")
print("  row share quoted by the claim.")
