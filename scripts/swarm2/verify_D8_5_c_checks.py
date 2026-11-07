"""verify D8-5 (part c): residual specifics. Independent loaders (same conventions as verify_D8_5_b_mr8.py).

C1 orientation split among complete MR8 matches, and whether 'both orientations fit' is just tA == tB (8-8 ties)
C2 complete matches with detected (8,): exact fits with no flip (as-is), and with the round-9 flip
C3 the MR12-labelled matches among complete other_misfit (details)
C4 bomb side vs round_end vs score side at the first post-swap round (MR8 round 9, MR12 round 13), by build
C5 event-MR8 matches: round_end vs score side at round 9, split by whether the score detector flagged (8,)
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_c_checks.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def fo(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def ch_files(ch):
    return fo(BASE / f"basic_{ch}{SUF}")


HDR = pl.concat([pl.read_parquet(f, columns=["key", "map_name", "build_num", "t_starters_score_final",
                                             "ct_starters_score_final"]) for f in fo(BASE / "header")],
                how="vertical_relaxed")
H = {r["key"]: r for r in HDR.iter_rows(named=True)}
RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code"]) for f in ch_files("round_end")],
               how="vertical_relaxed").unique(["match_key", "round"], keep="first")
rw_all = collections.defaultdict(dict)
for r in RE.iter_rows(named=True):
    rw_all[r["match_key"]][int(r["round"])] = int(r["winner_team_code"])
BS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"])
                .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])) for f in ch_files("bomb_state")],
               how="vertical_relaxed")
ev = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
bside = {(r["match_key"], int(r["round"])): (CT if "bomb_defused" in r["evs"] else T) for r in ev.iter_rows(named=True)}
HT = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type"])
                .filter(pl.col("event_type") == "round_announce_last_round_half").select(["match_key", "round"])
                for f in ch_files("round_state")], how="vertical_relaxed")
HTL = collections.defaultdict(set)
for r in HT.iter_rows(named=True):
    HTL[r["match_key"]].add(int(r["round"]))


def ev_h(m):
    return (min(HTL[m]) + 1) if HTL.get(m) else None


RSF = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
                 .filter(pl.col("event_type") == "round_freeze_end")
                 .select(["match_key", "round", "tick", "t_score", "ct_score"]) for f in ch_files("round_state")],
                how="vertical_relaxed")
FE = RSF.sort("tick").unique(["match_key", "round"], keep="first")
live = {(m, r): (int(t), int(c)) for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
del RSF, FE
by_m = collections.defaultdict(list)
for (m, r) in live:
    by_m[m].append(r)
SIG = {}
for m, rs in by_m.items():
    sw = []
    for r in sorted(rs):
        if (m, r + 1) not in live:
            continue
        t1, c1 = live[(m, r)]
        t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if (dc == 1 and dt == 0) or (dt == 1 and dc == 0) or (dt == 0 and dc == 0):
            continue
        if dt * dc <= 0:
            sw.append(r)
    SIG[m] = tuple(sw)


def side_a(r, swaps):
    k = sum(1 for s in swaps if s < r)
    return T if k % 2 == 0 else CT


def wa_of(rw, n, swaps, flips=()):
    wa = sum(1 for r in range(1, n + 1) if rw[r] == side_a(r, swaps))
    for r in flips:
        wa = wa - 1 if rw[r] == side_a(r, swaps) else wa + 1
    return wa


def fits(rw, n, tot, swaps, orient, flips=()):
    if any(r not in rw for r in flips):
        return False
    tA = tot[0] if orient == 0 else tot[1]
    return wa_of(rw, n, swaps, flips) == tA


def tot_of(m):
    return (int(H[m]["t_starters_score_final"]), int(H[m]["ct_starters_score_final"]))


def score_winner(m, r):
    if (m, r) not in live or (m, r + 1) not in live:
        return None
    t1, c1 = live[(m, r)]
    t2, c2 = live[(m, r + 1)]
    dt, dc = t2 - t1, c2 - c1
    if dt == 1 and dc == 0:
        return T
    if dc == 1 and dt == 0:
        return CT
    return None


COMPLETE = []
for m, rw in rw_all.items():
    if m not in H:
        continue
    tA, tB = tot_of(m)
    n = len(rw)
    if n == tA + tB and n <= 24 and sorted(rw) == list(range(1, n + 1)):
        COMPLETE.append(m)
c8 = [m for m in COMPLETE if ev_h(m) == 8]
c12 = [m for m in COMPLETE if ev_h(m) == 12]
print(f"complete {len(COMPLETE)}; complete event-MR8 {len(c8)}; complete event-MR12 {len(c12)}")

# ---------------- C1 orientation split among complete MR8
print("\n=== C1 orientation split, complete event-MR8 matches (MR8 model: swap after 8, round-9 flip) ===")
o_only0 = o_only1 = both = none = 0
for m in c8:
    rw, n, tot = rw_all[m], len(rw_all[m]), tot_of(m)
    f0 = fits(rw, n, tot, (8,), 0, (9,))
    f1 = fits(rw, n, tot, (8,), 1, (9,))
    if f0 and f1:
        both += 1
    elif f0:
        o_only0 += 1
    elif f1:
        o_only1 += 1
    else:
        none += 1
ties = [m for m in c8 if tot_of(m)[0] == tot_of(m)[1]]
print(f"  orientation 0 only {o_only0}; orientation 1 only {o_only1}; both {both}; neither {none}")
print(f"  complete event-MR8 with header tA == tB (tie): {len(ties)}")
print(f"  'both' group size {both}; of which tie matches {sum(1 for m in ties if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 0, (9,)))}")
print(f"  orientation-1-only matches by map: " + str(sorted(collections.Counter(
    H[m]["map_name"] for m in c8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 1, (9,))
    and not fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 0, (9,))).items())))

# ---------------- C2 detected (8,) complete matches
print("\n=== C2 complete matches with detected (8,) (the claim's F-3 population) ===")
d8 = [m for m in COMPLETE if SIG.get(m) == (8,)]
print(f"  detected (8,) complete: {len(d8)}")
print(f"  exact fit, as-is orientation, no flip (schedule = detected): "
      f"{sum(1 for m in d8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 0))}")
print(f"  exact fit, reversed orientation, no flip: "
      f"{sum(1 for m in d8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 1))}")
print(f"  exact fit with round-9 flip, as-is: "
      f"{sum(1 for m in d8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 0, (9,)))}; "
      f"reversed: {sum(1 for m in d8 if fits(rw_all[m], len(rw_all[m]), tot_of(m), (8,), 1, (9,)))}")

# ---------------- C3 MR12-labelled complete matches that the MR12 baseline does not fit
print("\n=== C3 MR12-labelled complete matches not fitted by the MR12 baseline (as-is, or round-13 flip) ===")
for m in c12:
    rw, n, tot = rw_all[m], len(rw_all[m]), tot_of(m)
    mm13 = 13 in rw and (m, 13) in bside and bside[(m, 13)] != rw[13]
    if mm13 or fits(rw, n, tot, (12,), 0) or fits(rw, n, tot, (12,), 0, (13,)):
        continue
    wstr = "".join("A" if rw[r] == T else "B" for r in range(1, n + 1))
    print(f"  {m} map={H[m]['map_name']} n={n} header=({tot[0]},{tot[1]}) labels={sorted(HTL[m])} "
          f"detected={SIG.get(m)} sequence(A=team on T in round 1)={wstr}")
    print(f"    fits MR8 model (8, round-9 flip): o0={fits(rw, n, tot, (8,), 0, (9,))} o1={fits(rw, n, tot, (8,), 1, (9,))}; "
          f"MR8 no flip: o0={fits(rw, n, tot, (8,), 0)} o1={fits(rw, n, tot, (8,), 1)}")

# ---------------- C4 bomb side vs round_end vs score side at first post-swap round
print("\n=== C4 first post-swap round: bomb side vs round_end vs score side (MR8 round 9; MR12 round 13) ===")
c4 = collections.Counter()
for m in H:
    if m not in rw_all:
        continue
    h = ev_h(m)
    if h not in (8, 12):
        continue
    r = h + 1
    if (m, r) not in bside or r not in rw_all[m]:
        continue
    b = H[m]["build_num"]
    bs = bside[(m, r)]
    re_w = rw_all[m][r]
    sc = score_winner(m, r)
    c4[(b, h, "bomb rows")] += 1
    c4[(b, h, "bomb != round_end")] += 1 if bs != re_w else 0
    if sc is not None:
        c4[(b, h, "score-compared")] += 1
        c4[(b, h, "bomb != score")] += 1 if bs != sc else 0
        c4[(b, h, "round_end != score")] += 1 if re_w != sc else 0
for k_ in sorted(c4):
    print(f"  build={k_[0]} halftime_after={k_[1]} {k_[2]:<20} {c4[k_]}")

# ---------------- C5 MR8 round 9: detector-flagged vs not
print("\n=== C5 event-MR8 matches, round 9: round_end vs score side, split by detector flag (header-free) ===")
c5 = collections.Counter()
for m in H:
    if m not in rw_all or ev_h(m) != 8 or 9 not in rw_all[m]:
        continue
    sc = score_winner(m, 9)
    if sc is None:
        continue
    flag = "detected (8,)" if SIG.get(m) == (8,) else "not detected"
    c5[(H[m]["build_num"], flag, "compared")] += 1
    c5[(H[m]["build_num"], flag, "round_end != score")] += 1 if rw_all[m][9] != sc else 0
for k_ in sorted(c5):
    print(f"  build={k_[0]} {k_[1]:<15} {k_[2]:<22} {c5[k_]}")
print("\nDONE (part c)")
