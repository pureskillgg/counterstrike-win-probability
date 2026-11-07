"""verify D3-3 controls (read-only).

A. 2026 tome control: the claim's exact definition (side delta vs round_end winner, swap rule with 24) on the
   2026-10-01..08 tome, counted by round and by 5v5 status (5 T and 5 CT in player_info).
B. 2024 side-label consistency in claim-disagreement 5v5 rounds: victim side at death vs player_info side for the
   same player and round; planter side at bomb_begin_plant vs player_info side for the planter; and the same
   checks for the agreeing 5v5 rounds at the same round numbers, for comparison.
C. 2024 event-based adjudication (bomb or elimination) restricted to 5v5 rounds at rounds 13, 25, 28 and controls.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
T, CT = 2, 3
TOMES = {
    "2024": (Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"), ".2023-12-28,2024-02-25"),
    "2026": (Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"), ".2026-09-18,2026-10-08"),
}


def scan(tome, suf, ch):
    fs = sorted(glob.glob(str(tome / f"basic_{ch}{suf}" / "dataframe_*")))
    return pl.scan_parquet(fs, missing_columns="insert", extra_columns="ignore")


def swap_claim(r, mr=24):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def load(year):
    tome, suf = TOMES[year]
    RE = (scan(tome, suf, "round_end").select(["match_key", "round", "tick", "winner_team_code"]).collect()
          .sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True))
    W = {(m, r): (w, tk) for m, r, tk, w in RE.select(["match_key", "round", "tick", "winner_team_code"]).iter_rows()}
    RS = scan(tome, suf, "round_state").filter(pl.col("event_type") == "round_freeze_end").select(
        ["match_key", "round", "tick", "t_score", "ct_score"]).collect()
    FE = RS.sort("tick").group_by(["match_key", "round"]).agg(
        pl.col("tick").first().alias("ftick"), pl.col("t_score").first(), pl.col("ct_score").first())
    F = {(m, r): (ft, t, c) for m, r, ft, t, c in FE.select(
        ["match_key", "round", "ftick", "t_score", "ct_score"]).iter_rows()}
    PI = scan(tome, suf, "player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
    TS, CS = collections.defaultdict(set), collections.defaultdict(set)
    side_of = {}
    for m, r, pid, tc in PI.iter_rows():
        side_of[(m, r, pid)] = tc
        if tc == T:
            TS[(m, r)].add(pid)
        elif tc == CT:
            CS[(m, r)].add(pid)
    return W, F, TS, CS, side_of, tome, suf


# ---------- A. 2026 control
W26, F26, TS26, CS26, _, _, _ = load("2026")
D26 = {}
for (m, r), (ft, t1, c1) in F26.items():
    if (m, r + 1) not in F26 or swap_claim(r):
        continue
    _, t2, c2 = F26[(m, r + 1)]
    dt, dc = t2 - t1, c2 - c1
    if dc == 1 and dt == 0:
        D26[(m, r)] = CT
    elif dt == 1 and dc == 0:
        D26[(m, r)] = T
cnt = collections.Counter()
dis_r = collections.Counter()
for (m, r), d in D26.items():
    if (m, r) not in W26:
        continue
    five = len(TS26.get((m, r), ())) == 5 and len(CS26.get((m, r), ())) == 5
    cnt["valid"] += 1
    cnt["valid_5v5"] += int(five)
    if d != W26[(m, r)][0]:
        cnt["disagree"] += 1
        cnt["disagree_5v5"] += int(five)
        dis_r[r] += 1
print("== A. 2026 control (claim definition): valid, disagree, 5v5 variants:", dict(cnt))
print("   2026 disagreement by round:", sorted(dis_r.items()))

# ---------- B. 2024 side-label consistency
W, F, TS, CS, side_of, tome24, suf24 = load("2024")
D = {}
for (m, r), (ft, t1, c1) in F.items():
    if (m, r + 1) not in F or swap_claim(r):
        continue
    _, t2, c2 = F[(m, r + 1)]
    dt, dc = t2 - t1, c2 - c1
    if dc == 1 and dt == 0:
        D[(m, r)] = CT
    elif dt == 1 and dc == 0:
        D[(m, r)] = T


def day_grp(m):
    return "day12_25" if int(m.split("/")[3]) >= 12 else "day01_07"


five2024 = {(m, r) for (m, r) in W if len(TS.get((m, r), ())) == 5 and len(CS.get((m, r), ())) == 5}
dis5 = {k for k, d in D.items() if k in W and k in five2024 and day_grp(k[0]) == "day12_25" and d != W[k][0]}
agr5 = {k for k, d in D.items() if k in W and k in five2024 and day_grp(k[0]) == "day12_25" and d == W[k][0]}

PD = scan(tome24, suf24, "player_death").select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
PD = PD.filter(pl.col("player_team_code").is_in([T, CT])).unique(["match_key", "round", "player_id"], keep="first")
lab = collections.Counter()
for m, r, pid, side in PD.select(["match_key", "round", "player_id", "player_team_code"]).iter_rows():
    if (m, r) not in five2024:
        continue
    key = "dis5" if (m, r) in dis5 else ("agr5" if (m, r) in agr5 else None)
    if key is None:
        continue
    pi = side_of.get((m, r, pid))
    if pi is None:
        lab[(key, r if r in (13, 25, 28) else "rest", "victim_not_in_player_info")] += 1
    else:
        lab[(key, r if r in (13, 25, 28) else "rest", "victim_side_matches_player_info" if pi == side else "victim_side_mismatch")] += 1
print("\n== B1. death victim side vs player_info side, 5v5 rounds in day12_25 (dis5 = claim disagreement, agr5 = agree):")
for k, v in sorted(lab.items(), key=lambda kv: str(kv[0])):
    print("  ", k, v)

BA = scan(tome24, suf24, "bomb_action").select(["match_key", "round", "tick", "event_type", "player_id", "player_team_code"]).collect()
plant = {}
for m, r, tk, ev, pid, side in BA.filter(pl.col("event_type") == "bomb_begin_plant").select(
        ["match_key", "round", "tick", "event_type", "player_id", "player_team_code"]).iter_rows():
    plant.setdefault((m, r), (pid, side))
pl_cnt = collections.Counter()
for (m, r), (pid, side) in plant.items():
    if (m, r) not in five2024:
        continue
    key = "dis5" if (m, r) in dis5 else ("agr5" if (m, r) in agr5 else None)
    if key is None:
        continue
    pi = side_of.get((m, r, pid))
    pl_cnt[(key, r if r in (13, 25, 28) else "rest",
            "planter_in_player_info_T" if pi == T else ("planter_in_player_info_CT" if pi == CT else "planter_missing"),
            "event_side_T" if side == T else "event_side_CT")] += 1
print("\n== B2. planter side: player_info side of the planter and the event side label, 5v5 day12_25:")
for k, v in sorted(pl_cnt.items(), key=lambda kv: str(kv[0])):
    print("  ", k, v)

# ---------- C. 2024 event-based adjudication restricted to 5v5 rounds
RS2 = F
PDd = collections.defaultdict(list)
for m, r, tk, pid, side in PD.select(["match_key", "round", "tick", "player_id", "player_team_code"]).iter_rows():
    PDd[(m, r)].append((tk, side, pid))
BS = scan(tome24, suf24, "bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(
    ["match_key", "round", "tick", "event_type"]).collect()
bomb = {}
for m, r, tk, ev in BS.select(["match_key", "round", "tick", "event_type"]).iter_rows():
    bomb.setdefault((m, r), (tk, T if ev == "bomb_exploded" else CT))


def elim(m, r):
    ftk = F[(m, r)][0]
    etk = W[(m, r)][1]
    nt, nc = len(TS[(m, r)]), len(CS[(m, r)])
    dead = collections.defaultdict(set)
    done = {}
    for tk, side, pid in sorted(PDd.get((m, r), [])):
        if not (ftk < tk <= etk):
            continue
        dead[side].add(pid)
        need = nt if side == T else nc
        if len(dead[side]) == need and side not in done:
            done[side] = tk
    if T in done and CT in done:
        return None if done[T] == done[CT] else (CT if done[T] < done[CT] else T)
    if T in done:
        return CT
    if CT in done:
        return T
    return None


tab = collections.Counter()
for (m, r) in five2024:
    if (m, r) not in W or (m, r) not in F or day_grp(m) != "day12_25":
        continue
    if r not in (8, 9, 10, 12, 13, 14, 24, 25, 26, 27, 28, 29):
        continue
    etk = W[(m, r)][1]
    if (m, r) in bomb and F[(m, r)][0] < bomb[(m, r)][0] <= etk:
        k, src = bomb[(m, r)][1], "bomb"
    else:
        k, src = elim(m, r), "elim"
    if k is None:
        tab[(r, "K_undefined")] += 1
        continue
    w = W[(m, r)][0]
    d = D.get((m, r))
    tab[(r, "K!=W" if k != w else "K=W", "D_def" if d is not None else "D_none",
         ("D=K" if d == k else "D!=K") if d is not None else "")] += 1
print("\n== C. 5v5 day12_25: event outcome K vs W and D by round:")
for r in (8, 9, 10, 12, 13, 14, 24, 25, 26, 27, 28, 29):
    rows = {k[1:]: v for k, v in tab.items() if k[0] == r}
    if rows:
        print(f"   r={r:2d}", {k: v for k, v in sorted(rows.items(), key=lambda kv: str(kv[0]))})
