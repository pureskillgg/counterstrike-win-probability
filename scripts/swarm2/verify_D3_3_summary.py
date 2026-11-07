"""verify D3-3 summary (2024 G: tome, read-only): the core numbers in one place, claim definition plus 5v5 and
event-evidence coverage.

Definitions (ASSUMPTIONS marked):
  W  round_end winner_team_code, first row by tick per (match, round)
  D  claim side delta: freeze(r+1) - freeze(r) on (t_score, ct_score), exactly one side +1; skip swap_claim(r) (MR24 proxy,
     ASSUMPTION) and missing next round
  K  event outcome: bomb_state exploded -> T, defused -> CT, if the event lies in (freeze_end, round_end] of the round;
     else elimination (all players of one side dead in the window, the side that completes first loses)
  5v5 = 5 T and 5 CT in player_info for the round (round_end.player_count is 20 for 5v5 in this tome)
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
nT, nC = collections.Counter(), collections.Counter()
for m, r, pid, tc in PI.iter_rows():
    if tc == T:
        nT[(m, r)] += 1
    elif tc == CT:
        nC[(m, r)] += 1
PD = scan("player_death").select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
PD = PD.filter(pl.col("player_team_code").is_in([T, CT])).unique(["match_key", "round", "player_id"], keep="first")
deaths = collections.defaultdict(list)
for m, r, tk, pid, side in PD.select(["match_key", "round", "tick", "player_id", "player_team_code"]).iter_rows():
    deaths[(m, r)].append((tk, side, pid))
BS = scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(
    ["match_key", "round", "tick", "event_type"]).collect()
bomb = {}
for m, r, tk, ev in BS.select(["match_key", "round", "tick", "event_type"]).iter_rows():
    bomb.setdefault((m, r), (tk, T if ev == "bomb_exploded" else CT))


def elim(m, r):
    ftk, etk = F[(m, r)][0], W[(m, r)][1]
    nt, nc = nT.get((m, r), 0), nC.get((m, r), 0)
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
        return None if done[T] == done[CT] else (CT if done[T] < done[CT] else T)
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
        k = elim(m, r)
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

tab = collections.Counter()
for (m, r), d in D.items():
    if (m, r) not in W or grp(m) != "day12_25":
        continue
    w = W[(m, r)][0]
    five = nT.get((m, r), 0) == 5 and nC.get((m, r), 0) == 5
    k = K.get((m, r))
    if r not in (9, 13, 25, 28):
        tab[("all_rounds_other", "disagree" if d != w else "agree")] += 1
        continue
    key = f"r{r}"
    tab[(key, "valid_all")] += 1
    tab[(key, "valid_5v5")] += int(five)
    if d != w:
        tab[(key, "disagree_all")] += 1
        tab[(key, "disagree_5v5")] += int(five)
        if five:
            if k is None:
                tab[(key, "dis5v5_no_event_evidence")] += 1
            else:
                tab[(key, "dis5v5_K_eq_D" if k == d else "dis5v5_K_ne_D")] += 1
                tab[(key, "dis5v5_K_eq_W" if k == w else "dis5v5_K_ne_W")] += 1
    else:
        if five and k is not None:
            tab[(key, "agree5v5_K_eq_W" if k == w else "agree5v5_K_ne_W")] += 1

print("day12_25, claim definition (valid = D defined, W present):")
for r in (9, 13, 25, 28):
    key = f"r{r}"
    print(f"  round {r:2d}: valid {tab[(key, 'valid_all')]:5d}, disagree {tab[(key, 'disagree_all')]:4d}"
          f"  | 5v5 valid {tab[(key, 'valid_5v5')]:5d}, 5v5 disagree {tab[(key, 'disagree_5v5')]:4d}"
          f" = K-backed {tab[(key, 'dis5v5_K_eq_D')] + tab[(key, 'dis5v5_K_ne_D')]} "
          f"(K=D {tab[(key, 'dis5v5_K_eq_D')]}, K!=D {tab[(key, 'dis5v5_K_ne_D')]}, K=W {tab[(key, 'dis5v5_K_eq_W')]})"
          f" + no event evidence {tab[(key, 'dis5v5_no_event_evidence')]}")
    print(f"           agreeing 5v5 rounds with K defined: K=W {tab[(key, 'agree5v5_K_eq_W')]}, "
          f"K!=W {tab[(key, 'agree5v5_K_ne_W')]}")
other = {k: v for k, v in tab.items() if k[0] == "all_rounds_other"}
print("  other rounds (claim definition):", {k[1]: v for k, v in other.items()})
tot5 = sum(tab[(f"r{r}", "disagree_5v5")] for r in (9, 13, 25, 28))
tot_all = sum(tab[(f"r{r}", "disagree_all")] for r in (9, 13, 25, 28))
print(f"  totals rounds 9/13/25/28: all disagree {tot_all}, 5v5 disagree {tot5}, non-5v5 {tot_all - tot5}")
k_backed = sum(tab[(f"r{r}", "dis5v5_K_eq_D")] + tab[(f"r{r}", "dis5v5_K_ne_D")] for r in (9, 13, 25, 28))
print(f"  5v5 disagreements with event evidence: {k_backed}; of these K=D: "
      f"{sum(tab[(f'r{r}', 'dis5v5_K_eq_D')] for r in (9, 13, 25, 28))}; K=W: "
      f"{sum(tab[(f'r{r}', 'dis5v5_K_eq_W')] for r in (9, 13, 25, 28))}")
