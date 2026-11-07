"""D6-5 verifier, part 9: exact decisive-evidence counts used in the verdict (2024 and 2026 controls).

Decisive evidence = bomb outcome, planted CT-wipe (T wins), or non-planted wipe. Timeouts are excluded.
Categories for each round_end disagreement with decisive evidence (E != round_end):
  A. score-scorable and score != round_end   (the claim's set; E == score in all such cases)
  B. score-scorable and score == round_end   (evidence-only disagreement: noise or joint error)
  C. score-unscorable                        (outside the claim's scope)
Also round 25 on decisive evidence only, and the claimed-540 subset with decisive evidence.

Run:
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_10_summary_counts.py 2024
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_5_10_summary_counts.py 2026
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


C = collections.Counter()
for (m, r), w_re in rew.items():
    scm = scm_all.get(m, {})
    w_sc = win_from_score(scm, r) if scm else None
    E, src = evidence(m, r)
    if E is None or src == "timeout":
        if w_sc is not None and w_sc != w_re:
            C["A_claim_set: SC!=RE, no decisive E"] += 1
        continue
    if w_sc is not None:
        if w_sc != w_re:
            C[f"A_claim_set: SC!=RE, decisive E ({src}); E==SC:{E == w_sc}"] += 1
        else:
            if E != w_re:
                C[f"B_scorable SC==RE but decisive E!=RE ({src})"] += 1
            else:
                C["scorable, SC==RE==E"] += 1
    else:
        if E != w_re:
            C[f"C_unscorable decisive E!=RE ({src})"] += 1
        else:
            C["C_unscorable decisive E==RE"] += 1
print(f"=== ERA {ERA}: decisive-evidence summary (timeouts excluded)")
for k in sorted(C, key=str):
    print(f"  {k:<70} n={C[k]}")
tot_dis = sum(v for k, v in C.items() if "!=RE" in k or "!=RE" in k or ("SC!=RE" in k and "E==SC" in k))
print(f"  decisive round_end-vs-evidence disagreements (all categories): "
      f"{sum(v for k, v in C.items() if ('E!=RE' in k) or ('SC!=RE, decisive' in k and 'E==SC:True' in k))}")

# round 25 on decisive evidence
R25 = collections.Counter()
for (m, r), w_re in rew.items():
    if r != 25:
        continue
    E, src = evidence(m, r)
    if E is None or src == "timeout":
        R25["no decisive evidence"] += 1
        continue
    R25["RE!=E" if w_re != E else "RE==E"] += 1
    R25[f"src {src} {'RE!=E' if w_re != E else 'RE==E'}"] += 1
print(f"  round 25, decisive evidence only: {dict(sorted(R25.items(), key=str))}")
