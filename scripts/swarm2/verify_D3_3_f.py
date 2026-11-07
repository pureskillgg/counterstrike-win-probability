"""verify_D3_3 part F: delta-free scan. For every round number k (1..30), compare round_end winner with the
elimination oracle (roster-sized side elimination from player_death) and the bomb outcome, by format and day group.
This does not use score deltas, so it can see rounds the delta test cannot (no next freeze, swap-excluded).
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


def files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def day(m):
    return int(m.split("/")[3])


ann = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).group_by("match_key").agg(pl.col("round").min().alias("r_ann")).collect())
MR = {m: int(r) * 2 + 2 for m, r in ann.iter_rows()}
fe = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_freeze_end")
      .select(["match_key", "round", "tick"]).collect().sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("tick").first()))
FT = {(m, int(r)): int(tk) for m, r, tk in fe.iter_rows()}
re_df = (pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code"]).collect()
         .sort("tick").group_by(["match_key", "round"]).agg(pl.col("tick").first(), pl.col("winner_team_code").first()))
RE = {(m, int(r)): (int(tk), int(w)) for m, r, tk, w in re_df.iter_rows()}
dth = pl.scan_parquet(files("player_death")).select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
dth_g = collections.defaultdict(list)
for m, r, tk, pid, pt in dth.iter_rows():
    dth_g[(m, int(r))].append((int(tk), int(pid), int(pt) if pt is not None else -1))
pi = pl.scan_parquet(files("player_info")).select(["match_key", "round", "player_id", "team_code"]).collect()
roster = collections.defaultdict(lambda: {2: set(), 3: set()})
for m, r, pid, tc in pi.iter_rows():
    if tc in (2, 3):
        roster[(m, int(r))][int(tc)].add(pid)
bs = (pl.scan_parquet(files("bomb_state")).filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
      .select(["match_key", "round", "event_type"]).collect())
bo = {}
for m, r, ev in bs.iter_rows():
    bo[(m, int(r))] = T if ev == "bomb_exploded" else CT


def elim(m, k):
    ro = roster.get((m, k))
    if (m, k) not in RE or (m, k) not in FT or ro is None or not ro[T] or not ro[CT]:
        return None
    t_end = RE[(m, k)][0]
    t_frz = FT[(m, k)]
    vic = {T: set(), CT: set()}
    for tk, pid, pt in dth_g.get((m, k), []):
        if t_frz <= tk <= t_end and pt in (T, CT):
            vic[pt].add(pid)
    dT = len(vic[T] & ro[T])
    dC = len(vic[CT] & ro[CT])
    if dT >= len(ro[T]) and dC < len(ro[CT]):
        return CT
    if dC >= len(ro[CT]) and dT < len(ro[T]):
        return T
    return None


tab = collections.defaultdict(lambda: [0, 0, 0])  # agree, disagree, undetermined (elim)
btab = collections.defaultdict(lambda: [0, 0])    # bomb agree, bomb disagree
for (m, k), (tk, w) in RE.items():
    grp = "d12_25" if day(m) >= 12 else "d01_07"
    fmt = MR.get(m, 24)
    e = elim(m, k)
    key = (grp, fmt, k)
    if e is None:
        tab[key][2] += 1
    elif e == w:
        tab[key][0] += 1
    else:
        tab[key][1] += 1
    if (m, k) in bo:
        btab[key][0 if bo[(m, k)] == w else 1] += 1
print("== delta-free scan: round_end vs elimination (agree/disagree/undet) and vs bomb (agree/disagree) ==")
for grp in ("d01_07", "d12_25"):
    for fmt in (16, 24):
        rows = sorted(k for (g, f, k) in tab if g == grp and f == fmt)
        if not rows:
            continue
        print(f"\n  {grp} MR{fmt}:")
        for k in rows:
            a, d, u = tab[(grp, fmt, k)]
            ba, bd = btab.get((grp, fmt, k), [0, 0])
            if a + d + u + ba + bd == 0:
                continue
            rate = d / (a + d) if (a + d) else float("nan")
            flag = "  <== inverted" if (a + d) >= 20 and rate > 0.2 else ""
            print(f"    r={k:>2}: elim agree {a:>5} disagree {d:>5} undet {u:>5} | disagree rate {rate:6.3f} | bomb agree {ba:>4} disagree {bd:>4}{flag}")
