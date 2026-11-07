"""Verifier D3-5, probe 6: is round_state (score fields) side-attached in every physical-swap class?
Per match class (physical swap rounds vs field swap rounds), compare the elimination winner (E, death feed)
and the bomb winner (D) with the score-field winner (C, unambiguous transitions only) and with round_end (A).
Also checks whether the death feed's victim team codes agree with player_info team codes in the same round.
Read only. Prints to stdout.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


PI = g_read("player_info", ["match_key", "round", "player_id", "team_code"])
DE = g_read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
RE = g_read("round_end", ["match_key", "round", "tick", "winner_team_code"]).unique(["match_key", "round"], keep="first")
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = RS.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
FE = {(m, int(r)): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
A = {(m, int(r)): int(w) for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}

# player side per (match, round, player)
PSIDE = {}
side = collections.defaultdict(lambda: {T: set(), CT: set()})
for m, r, pid, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
    PSIDE[(m, int(r), pid)] = tc
    if tc in (T, CT):
        side[(m, int(r))][tc].add(pid)

# (1) death feed team code vs player_info team code, same round
agree = disagree = missing = 0
for m, r, pid, tc in DE.select(["match_key", "round", "player_id", "player_team_code"]).iter_rows():
    p = PSIDE.get((m, int(r), pid))
    if p is None:
        missing += 1
    elif p == tc:
        agree += 1
    else:
        disagree += 1
print(f"death feed victim team == player_info team (same round): agree {agree}, disagree {disagree}, no player_info row {missing}")

# bomb planter team code vs player_info (planted rows)
pl_agree = pl_dis = pl_none = 0
for m, r, pid, tc, ev in BS.filter(pl.col("event_type") == "bomb_planted").select(
        ["match_key", "round", "player_id" if "player_id" in BS.columns else "tick", "player_team_code", "event_type"]).iter_rows():
    pl_none += 1
print("bomb_planted rows:", pl_none, "(planter id not in this projection; planter side checked via player_team_code only)")
print("planter player_team_code counts:", BS.filter(pl.col("event_type") == "bomb_planted")
      .group_by("player_team_code").len().sort("player_team_code").rows())

# (2) per-match physical swap rounds (player_info), with a complete 5v5 round on both sides
phys = collections.defaultdict(list)
rounds_by_m = collections.defaultdict(list)
for (m, r) in side:
    rounds_by_m[m].append(r)
for m, rs in rounds_by_m.items():
    rs = sorted(rs)
    for r in rs:
        if (m, r + 1) not in side:
            continue
        s0, s1 = side[(m, r)], side[(m, r + 1)]
        if s0[T] and s1[T] and s1[T] == s0[CT] and s1[CT] == s0[T]:
            phys[m].append(r)


def classify(t, c, t2, c2):
    out = set()
    if (t2, c2) == (t + 1, c):
        out.add((T, 0))
    if (t2, c2) == (c, t + 1):
        out.add((T, 1))
    if (t2, c2) == (t, c + 1):
        out.add((CT, 0))
    if (t2, c2) == (c + 1, t):
        out.add((CT, 1))
    return out


fe_by_m = collections.defaultdict(dict)
for (m, r), v in FE.items():
    fe_by_m[m][r] = v
field = {}
C_unamb = {}
for m, fes in fe_by_m.items():
    fs = []
    for r in sorted(fes):
        if (r + 1) not in fes:
            continue
        t, c = fes[r]
        t2, c2 = fes[r + 1]
        cs = classify(t, c, t2, c2)
        if len(cs) == 1:
            w, s = next(iter(cs))
            C_unamb[(m, r)] = w
            if s == 1:
                fs.append(r)
    field[m] = tuple(fs)

# (3) elimination winner per round (physical victim codes), bomb winner
dead = collections.defaultdict(lambda: collections.defaultdict(set))
for m, r, pid, tc in DE.select(["match_key", "round", "player_id", "player_team_code"]).iter_rows():
    dead[(m, int(r))][tc].add(pid)
bombw = {}
bm = BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused", "bomb_planted"]))
bev = collections.defaultdict(set)
for m, r, ev in bm.select(["match_key", "round", "event_type"]).iter_rows():
    bev[(m, int(r))].add(ev)
for k, evs in bev.items():
    if "bomb_exploded" in evs:
        bombw[k] = T
    elif "bomb_defused" in evs:
        bombw[k] = CT


def elim_winner(k):
    if "bomb_planted" in bev.get(k, set()):
        return None
    nT, nCT = len(dead[k].get(T, set())), len(dead[k].get(CT, set()))
    if nCT >= 5 and nT < 5:
        return T
    if nT >= 5 and nCT < 5:
        return CT
    return None


# class table
cls = collections.defaultdict(collections.Counter)
for m in fe_by_m:
    grp = "day12_25" if int(m.split("/")[3]) >= 12 else "day01_07"
    ps = tuple(phys.get(m, ()))
    fs = field.get(m, ())
    key = (grp, "phys" + str(ps), "field" + str(fs))
    for r in sorted(fe_by_m[m]):
        k = (m, r)
        if k not in C_unamb:
            continue
        c = C_unamb[k]
        a = A.get(k)
        e = elim_winner(k)
        d = bombw.get(k)
        cls[key]["rounds_with_C"] += 1
        if a is not None:
            cls[key]["A==C"] += int(a == c)
            cls[key]["A!=C"] += int(a != c)
        if e is not None:
            cls[key]["E_avail"] += 1
            cls[key]["E==C"] += int(e == c)
            cls[key]["E!=C"] += int(e != c)
            if a is not None:
                cls[key]["E!=A"] += int(e != a)
        if d is not None:
            cls[key]["D_avail"] += 1
            cls[key]["D==C"] += int(d == c)
            cls[key]["D!=C"] += int(d != c)
print("\nclass (group, physical swap rounds, field swap rounds) -> counts")
for key, cnt in sorted(cls.items(), key=lambda kv: -kv[1]["rounds_with_C"])[:30]:
    print(key, dict(cnt))

# (4) the classes where physical swap exists but the field shows no swap at that round
tot = collections.Counter()
for key, cnt in cls.items():
    tot.update(cnt)
print("\nall classes total:", dict(tot))
