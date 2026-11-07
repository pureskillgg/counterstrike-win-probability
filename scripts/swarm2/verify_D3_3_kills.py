"""verify D3-3: three-way adjudication at each round between round_end winner W, freeze score delta D,
and an event-based outcome K (2024 G: tome, read-only).

K (event-based round outcome, side code):
  1. bomb outcome from bomb_state inside the round window (exploded -> T=2, defused -> CT=3) if present;
  2. else elimination: a side is eliminated when every player of that side (count from player_info at
     that round) has a player_death with victim side = that side inside (freeze_end tick, round_end tick].
     The side that completes elimination first loses. Ties or no elimination -> K undefined.
  Victim side comes from player_death.player_team_code (side code at death).

D (claim's side delta): freeze(r+1) - freeze(r) on (t_score, ct_score), only where exactly one side +1 and other 0,
  and r not in the claim's swap set (12, 27 for MR24) - same as the claim, for comparison.

Also reports, for the claim's disagreement rounds, how often K agrees with D vs with W.
Run from repo root with the project interpreter.
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


RE = (scan("round_end").select(["match_key", "round", "tick", "winner_team_code"]).collect()
      .sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True))
W = {(m, r): (w, tk) for m, r, tk, w in RE.select(["match_key", "round", "tick", "winner_team_code"]).iter_rows()}

RS = scan("round_state").filter(pl.col("event_type") == "round_freeze_end").select(
    ["match_key", "round", "tick", "t_score", "ct_score"]).collect()
FE = RS.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("tick").first().alias("ftick"), pl.col("t_score").first(), pl.col("ct_score").first())
F = {(m, r): (ft, t, c) for m, r, ft, t, c in FE.select(
    ["match_key", "round", "ftick", "t_score", "ct_score"]).iter_rows()}

PI = scan("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
nT, nC = collections.Counter(), collections.Counter()
for m, r, pid, tc in PI.iter_rows():
    if tc == T:
        nT[(m, r)] += 1
    elif tc == CT:
        nC[(m, r)] += 1

PD = scan("player_death").select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
PD = PD.filter(pl.col("player_team_code").is_in([T, CT])).unique(["match_key", "round", "player_id"], keep="first")
deaths = collections.defaultdict(list)  # (m,r) -> [(tick, side, pid)]
for m, r, tk, pid, side in PD.select(["match_key", "round", "tick", "player_id", "player_team_code"]).iter_rows():
    deaths[(m, r)].append((tk, side, pid))

BS = scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(
    ["match_key", "round", "tick", "event_type"]).collect()
bomb = {}
for m, r, tk, ev in BS.select(["match_key", "round", "tick", "event_type"]).iter_rows():
    bomb.setdefault((m, r), (tk, T if ev == "bomb_exploded" else CT))


def elim_outcome(m, r):
    ftk = F[(m, r)][0]
    etk = W[(m, r)][1]
    nt, nc = nT.get((m, r), 0), nC.get((m, r), 0)
    if nt == 0 or nc == 0:
        return None, "no_side_count"
    dead = collections.defaultdict(set)
    done_tick = {}
    for tk, side, pid in sorted(deaths.get((m, r), [])):
        if not (ftk < tk <= etk):
            continue
        dead[side].add(pid)
        need = nt if side == T else nc
        if len(dead[side]) == need and side not in done_tick:
            done_tick[side] = tk
    if T in done_tick and CT in done_tick:
        if done_tick[T] == done_tick[CT]:
            return None, "tie"
        return (CT if done_tick[T] < done_tick[CT] else T), "elim"
    if T in done_tick:
        return CT, "elim"
    if CT in done_tick:
        return T, "elim"
    return None, "no_elim"


K = {}
Kkind = collections.Counter()
for (m, r), (w, etk) in W.items():
    if (m, r) not in F:
        continue
    if (m, r) in bomb and F[(m, r)][0] < bomb[(m, r)][0] <= etk:
        K[(m, r)] = bomb[(m, r)][1]
        Kkind[(grp(m), "bomb")] += 1
        continue
    k, why = elim_outcome(m, r)
    Kkind[(grp(m), why)] += 1
    if k is not None:
        K[(m, r)] = k
print("K source counts (bomb inside window vs elimination vs none):")
for k, v in sorted(Kkind.items()):
    print("  ", k, v)

# D (claim's side delta)
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

# 1. K vs W and K vs D, by round, all rounds with a valid K and W
rows = collections.defaultdict(lambda: collections.Counter())
for (m, r), k in K.items():
    if (m, r) not in W:
        continue
    g = grp(m)
    w = W[(m, r)][0]
    rows[(g, r)]["K_defined"] += 1
    rows[(g, r)]["K_eq_W" if k == w else "K_ne_W"] += 1
    if (m, r) in D:
        rows[(g, r)]["D_defined"] += 1
        d = D[(m, r)]
        rows[(g, r)]["K_eq_D" if k == d else "K_ne_D"] += 1
        if d != w:
            rows[(g, r)]["claim_disagree_rounds"] += 1
            rows[(g, r)]["claim_dis_K_eq_D" if k == d else "claim_dis_K_eq_W" if k == w else "claim_dis_other"] += 1

print("\n== 1. event outcome K vs round_end W and vs claim's side delta D, rounds with K defined")
for g in ["day01_07", "day12_25"]:
    print(f"  {g}")
    for r in [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29]:
        c = rows.get((g, r))
        if not c:
            continue
        print(f"    r={r:2d} K_def={c['K_defined']:5d}  K=W {c['K_eq_W']:5d}  K!=W {c['K_ne_W']:5d}  "
              f"| D_def={c['D_defined']:5d} K=D {c['K_eq_D']:5d} K!=D {c['K_ne_D']:5d}  "
              f"| claim-disagree {c['claim_disagree_rounds']:5d}: K=D {c['claim_dis_K_eq_D']:5d} "
              f"K=W {c['claim_dis_K_eq_W']:5d} other {c['claim_dis_other']:5d}")

# 2. totals over the whole claim disagreement set and over all agreeing rounds
tot = collections.Counter()
for (m, r), k in K.items():
    if (m, r) not in W or (m, r) not in D:
        continue
    g = grp(m)
    d, w = D[(m, r)], W[(m, r)][0]
    tot[(g, "dis" if d != w else "agree", "K=D" if k == d else "K=W" if k == w else "other")] += 1
print("\n== 2. totals: claim disagreement rounds vs agreeing rounds (where K defined)")
for k, v in sorted(tot.items()):
    print("  ", k, v)
