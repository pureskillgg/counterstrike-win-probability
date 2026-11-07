"""verify D3-3 structure (2024 G: tome, read-only).

1. Event side-label sanity: planter side (bomb_begin_plant) must be T (2) and defuser side (bomb_begin_defuse) CT (3).
   Checked per round number for claim-disagreement rounds vs agreeing rounds.
2. Flip vs empirical swap: for each match, the empirical halftime/OT swap transitions from player_info.
   Does the claimed W-flip (K=D and K!=W) sit at the first round after an empirical swap?
3. Recount the claim's totals with the bakeoff's own 5v5 filter: round_end.player_count == 10 and 5 T and 5 CT
   in player_info (instead of the claim's distinct-player count).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def chan_files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def scan(ch):
    return pl.scan_parquet(chan_files(ch), missing_columns="insert", extra_columns="ignore")


def grp(m):
    return "day12_25" if int(m.split("/")[3]) >= 12 else "day01_07"


def swap_claim(r, mr=24):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


RE = (scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
      .sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True))
W = {(m, r): (w, tk, pc) for m, r, tk, w, pc in RE.select(
    ["match_key", "round", "tick", "winner_team_code", "player_count"]).iter_rows()}

RS = scan("round_state").filter(pl.col("event_type") == "round_freeze_end").select(
    ["match_key", "round", "tick", "t_score", "ct_score"]).collect()
FE = RS.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("tick").first().alias("ftick"), pl.col("t_score").first(), pl.col("ct_score").first())
F = {(m, r): (ft, t, c) for m, r, ft, t, c in FE.select(
    ["match_key", "round", "ftick", "t_score", "ct_score"]).iter_rows()}

PI = scan("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
TS, CS = collections.defaultdict(set), collections.defaultdict(set)
for m, r, pid, tc in PI.iter_rows():
    if tc == T:
        TS[(m, r)].add(pid)
    elif tc == CT:
        CS[(m, r)].add(pid)

PD = scan("player_death").select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
PD = PD.filter(pl.col("player_team_code").is_in([T, CT])).unique(["match_key", "round", "player_id"], keep="first")
deaths = collections.defaultdict(list)
for m, r, tk, pid, side in PD.select(["match_key", "round", "tick", "player_id", "player_team_code"]).iter_rows():
    deaths[(m, r)].append((tk, side, pid))

BA = scan("bomb_action").select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
planter = {}
for m, r, tk, ev, side in BA.filter(pl.col("event_type") == "bomb_begin_plant").select(
        ["match_key", "round", "tick", "event_type", "player_team_code"]).iter_rows():
    planter.setdefault((m, r), side)
BD = scan("bomb_defuse").select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
defuser = {}
for m, r, tk, ev, side in BD.select(["match_key", "round", "tick", "event_type", "player_team_code"]).iter_rows():
    defuser.setdefault((m, r), side)

BS = scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(
    ["match_key", "round", "tick", "event_type"]).collect()
bomb = {}
for m, r, tk, ev in BS.select(["match_key", "round", "tick", "event_type"]).iter_rows():
    bomb.setdefault((m, r), (tk, T if ev == "bomb_exploded" else CT))


def elim_outcome(m, r):
    if (m, r) not in F or (m, r) not in W:
        return None
    ftk = F[(m, r)][0]
    etk = W[(m, r)][1]
    nt, nc = len(TS.get((m, r), ())), len(CS.get((m, r), ()))
    if nt == 0 or nc == 0:
        return None
    dead = collections.defaultdict(set)
    done = {}
    for tk, side, pid in sorted(deaths.get((m, r), [])):
        if not (ftk < tk <= etk):
            continue
        dead[side].add(pid)
        need = nt if side == T else nc
        if len(dead[side]) == need and side not in done:
            done[side] = tk
    if T in done and CT in done:
        if done[T] == done[CT]:
            return None
        return CT if done[T] < done[CT] else T
    if T in done:
        return CT
    if CT in done:
        return T
    return None


K = {}
for (m, r), (w, etk, pc) in W.items():
    if (m, r) not in F:
        continue
    if (m, r) in bomb and F[(m, r)][0] < bomb[(m, r)][0] <= etk:
        K[(m, r)] = bomb[(m, r)][1]
    else:
        k = elim_outcome(m, r)
        if k is not None:
            K[(m, r)] = k

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

# claim disagreement set (side delta vs W), day12_25
claim_dis = {k for k, d in D.items() if k in W and grp(k[0]) == "day12_25" and d != W[k][0]}
claim_agree = {k for k, d in D.items() if k in W and grp(k[0]) == "day12_25" and d == W[k][0]}

# ---------- 1. planter/defuser side labels
pl_cnt = collections.Counter()
for (m, r), side in planter.items():
    key = "dis" if (m, r) in claim_dis else ("agree" if (m, r) in claim_agree else "other")
    if grp(m) != "day12_25":
        key = "day01_07_" + key
    pl_cnt[(r if r in (9, 13, 25, 28) else "rest", key, "planter_T" if side == T else "planter_CT" if side == CT else "planter_other")] += 1
print("== 1. planter side (bomb_begin_plant player_team_code) by round class and claim disagreement (day12_25):")
for k, v in sorted(pl_cnt.items(), key=lambda kv: str(kv[0])):
    if "day01" in k[1]:
        continue
    print("  ", k, v)

ex_cnt = collections.Counter()
for (m, r), (tk, side) in bomb.items():
    if (m, r) not in W:
        continue
    key = "dis" if (m, r) in claim_dis else ("agree" if (m, r) in claim_agree else "other")
    pls = planter.get((m, r))
    if pls is None:
        ex_cnt[(r if r in (9, 13, 25, 28) else "rest", key, "no_plant_row")] += 1
        continue
    ex_cnt[(r if r in (9, 13, 25, 28) else "rest", key,
             "planter_T" if pls == T else "planter_CT", "bomb_side_T" if side == T else "bomb_side_CT",
             "W_T" if W[(m, r)][0] == T else "W_CT")] += 1
print("\n   bomb outcome rounds: planter side, bomb side, W side. 'dis' and 'agree' are day12_25 claim classes;"
      " 'other' = all remaining rounds with a bomb outcome, all days:")
for k, v in sorted(ex_cnt.items(), key=lambda kv: str(kv[0])):
    if k[0] in (9, 13, 25, 28, "rest"):
        print("  ", k, v)

# ---------- 2. flips vs empirical swap transitions
swap_set = collections.defaultdict(set)  # match -> set of r where team on T changes r -> r+1
first_ok = {}
for m in {m for (m, _) in W}:
    rounds = sorted(r for (mm, r) in W if mm == m)
    A = None
    for r in rounds:
        if TS.get((m, r)) and CS.get((m, r)):
            A = TS[(m, r)]
            break
    if A is None:
        continue
    on = {}
    for r in rounds:
        if TS.get((m, r)) and CS.get((m, r)):
            on[r] = "A" if len(TS[(m, r)] & A) >= len(CS[(m, r)] & A) else "B"
    for r in rounds:
        if r in on and (r + 1) in on and on[r] != on[r + 1]:
            swap_set[m].add(r)

flip = collections.Counter()
for (m, r), k in K.items():
    if grp(m) != "day12_25" or r not in (8, 9, 10, 12, 13, 14, 24, 25, 26, 27, 28, 29):
        continue
    if (m, r) not in W:
        continue
    w = W[(m, r)][0]
    prev_swap = (r - 1) in swap_set.get(m, set())
    flipped = (k != w)
    flip[(r, "prev_swap" if prev_swap else "no_prev_swap", "flip" if flipped else "same")] += 1
print("\n== 2. K != W (round_end winner contradicted by event outcome) by round, and whether the previous round had an empirical side swap (day12_25)")
for r in (8, 9, 10, 12, 13, 14, 24, 25, 26, 27, 28, 29):
    row = {}
    for ps in ("prev_swap", "no_prev_swap"):
        f = flip[(r, ps, "flip")]
        s = flip[(r, ps, "same")]
        row[ps] = f"{f}/{f + s}"
    print(f"   r={r:2d}  prev_swap {row['prev_swap']:>12}   no_prev_swap {row['no_prev_swap']:>12}")

# matches with empirical swap at 8 and their round-9 flips
m8 = {m for m in swap_set if 8 in swap_set[m]}
m12 = {m for m in swap_set if 12 in swap_set[m]}
print(f"\n   matches (day12_25) with empirical swap after round 8: {len([m for m in m8 if grp(m)=='day12_25'])}; "
      f"after round 12: {len([m for m in m12 if grp(m)=='day12_25'])}")
c = collections.Counter()
for (m, r) in claim_dis:
    if r == 9:
        c["r9_claimdis_in_swap8" if m in m8 else "r9_claimdis_not_swap8"] += 1
    if r == 13:
        c["r13_claimdis_in_swap12" if m in m12 else "r13_claimdis_not_swap12"] += 1
print("   claim disagreement rounds by swap structure:", dict(c))
print("   round-13 claim disagreements in matches with swap after 12:", c["r13_claimdis_in_swap12"],
      "of", c["r13_claimdis_in_swap12"] + c["r13_claimdis_not_swap12"])

# ---------- 3. recount with the bakeoff 5v5 filter
tot = collections.Counter()
for (m, r), d in D.items():
    if (m, r) not in W or grp(m) != "day12_25":
        continue
    w, tk, pc = W[(m, r)]
    five = len(TS.get((m, r), ())) == 5 and len(CS.get((m, r), ())) == 5
    tot["valid_all"] += 1
    tot["valid_5v5"] += int(five)
    tot["valid_pc20"] += int(pc == 20)
    if d != w:
        tot["claim_disagree_all"] += 1
        tot["claim_disagree_5v5"] += int(five)
        tot["claim_disagree_pc20"] += int(pc == 20)
print("\n== 3. recount (day12_25): 5v5 = 5 T and 5 CT in player_info; pc20 = round_end.player_count == 20 (5v5 code in this tome):")
for k, v in sorted(tot.items()):
    print("  ", k, v)
