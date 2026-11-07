"""verify_D3_3 part E: (1) roster sizes by format, (2) does round_end also disagree with the death/bomb oracles on the
rounds the claimed fix would NOT relabel (unclear / no-next / swap-excluded deltas), (3) MR16 round-8 contamination.

Elimination oracle here uses the per-round roster from player_info (team_code 2/3) instead of a fixed 5.
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


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def day(m):
    return int(m.split("/")[3])


ann = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).group_by("match_key").agg(pl.col("round").min().alias("r_ann")).collect())
MR = {m: int(r) * 2 + 2 for m, r in ann.iter_rows()}
fe_df = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_freeze_end")
         .select(["match_key", "round", "tick", "t_score", "ct_score"]).collect().sort("tick")
         .group_by(["match_key", "round"]).agg(pl.col("tick").first(), pl.col("t_score").first(),
                                               pl.col("ct_score").first()))
F, FT = collections.defaultdict(dict), collections.defaultdict(dict)
for m, r, tk, t, c in fe_df.select(["match_key", "round", "tick", "t_score", "ct_score"]).iter_rows():
    F[m][int(r)] = (int(t), int(c))
    FT[m][int(r)] = int(tk)
re_all = pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code",
                                                    "win_reason_code", "player_count"]).collect()
re_df = re_all.sort("tick").group_by(["match_key", "round"]).agg(pl.col("tick").first(),
                                                              pl.col("winner_team_code").first(),
                                                              pl.col("win_reason_code").first(),
                                                              pl.col("player_count").first())
RE = {(m, int(r)): (int(tk), int(w), int(rc), int(p)) for m, r, tk, w, rc, p in
      re_df.select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"]).iter_rows()}

dth = pl.scan_parquet(files("player_death")).select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
dth_g = collections.defaultdict(list)
for m, r, tk, pid, pt in dth.iter_rows():
    dth_g[(m, int(r))].append((int(tk), int(pid), int(pt) if pt is not None else -1))
bs = (pl.scan_parquet(files("bomb_state")).filter(pl.col("event_type").is_in(
      ["bomb_planted", "bomb_exploded", "bomb_defused"])).select(["match_key", "round", "tick", "event_type"]).collect())
bomb = collections.defaultdict(dict)
for m, r, tk, ev in bs.iter_rows():
    bomb[(m, int(r))].setdefault(ev, int(tk))
pi = pl.scan_parquet(files("player_info")).select(["match_key", "round", "player_id", "team_code"]).collect()
roster = collections.defaultdict(lambda: {2: set(), 3: set()})
for m, r, pid, tc in pi.iter_rows():
    if tc in (2, 3):
        roster[(m, int(r))][int(tc)].add(pid)

# (1) roster sizes and player_count by format, day group, round 13 / 9
print("== (1) roster sizes (player_info, team 2/3) and round_end player_count by format ==")
rs_cnt = collections.Counter()
for (m, r), rd in roster.items():
    if (m, r) in RE and r in (9, 13):
        rs_cnt[(MR.get(m, 24), r, len(rd[T]), len(rd[CT]))] += 1
for kk, v in sorted(rs_cnt.items()):
    print("   MR,round,|T|,|CT| roster:", kk, v)
pc_cnt = collections.Counter()
for m in F:
    if day(m) >= 12:
        for r in (9, 13):
            if (m, r) in RE:
                pc_cnt[(MR.get(m, 24), r, RE[(m, r)][3])] += 1
print("   round_end player_count (day12-25) for rounds 9 and 13 (MR, round, player_count): ",
      sorted(pc_cnt.items()))


def elim(m, k):
    if (m, k) not in RE or k not in F[m]:
        return "nofreeze"
    t_end = RE[(m, k)][0]
    t_frz = FT[m][k]
    ro = roster.get((m, k))
    if ro is None or not ro[T] or not ro[CT]:
        return "noroster"
    victims = {T: set(), CT: set()}
    for tk, pid, pt in dth_g.get((m, k), []):
        if t_frz <= tk <= t_end and pt in (T, CT):
            victims[pt].add(pid)
    dead_T = len(victims[T] & ro[T])
    dead_CT = len(victims[CT] & ro[CT])
    if dead_T >= len(ro[T]) and dead_CT < len(ro[CT]):
        return CT
    if dead_CT >= len(ro[CT]) and dead_T < len(ro[T]):
        return T
    return "undetermined"


def delta_status(m, k, mr):
    if k not in F[m] or (k + 1) not in F[m]:
        return "no_next_freeze", None
    if swap_expected(k, mr):
        return "swap_excluded", None
    t, c = F[m][k]
    t2, c2 = F[m][k + 1]
    dT, dC = t2 - t, c2 - c
    if dC == 1 and dT == 0:
        return "valid", CT
    if dT == 1 and dC == 0:
        return "valid", T
    return "unclear", None


# (2) for each MR/round in the claim's set, categorise by delta status and compare round_end with elim and bomb
print("\n== (2) round_end vs oracles by delta status (day 12-25) ==")
targets = [(16, 8), (16, 9), (24, 12), (24, 13), (24, 14), (24, 25), (24, 27), (24, 28)]
for mr_t, k in targets:
    st = collections.Counter()
    for m in F:
        if day(m) < 12 or MR.get(m, 24) != mr_t or (m, k) not in RE:
            continue
        status, dw = delta_status(m, k, 24)  # the claim uses MR24 for every match
        e = elim(m, k)
        rew = RE[(m, k)][1]
        bm = bomb.get((m, k), {})
        bw = None
        if "bomb_exploded" in bm:
            bw = T
        elif "bomb_defused" in bm:
            bw = CT
        key = status
        st[(key, "n")] += 1
        if status == "valid":
            st[(key, "delta_vs_rend", "agree" if dw == rew else "disagree")] += 1
        if e in (T, CT):
            st[(key, "elim_vs_rend", "agree" if e == rew else "disagree")] += 1
            if status == "valid":
                st[(key, "elim_vs_delta", "agree" if e == dw else "disagree")] += 1
        if bw is not None:
            st[(key, "bomb_vs_rend", "agree" if bw == rew else "disagree")] += 1
    print(f"  MR{mr_t} round {k}:")
    for kk, v in sorted(st.items(), key=lambda kv: str(kv[0])):
        print("     ", kk, v)

# (3) MR16 round-8 contamination under the claim's MR24 schedule: boundary 8->9 for MR16 matches
print("\n== (3) MR16 round 8 under the claim's MAX_ROUNDS=24 (not excluded) ==")
c3 = collections.Counter()
for m in F:
    if day(m) < 12 or MR.get(m, 24) != 16 or (m, 8) not in RE:
        continue
    status, dw = delta_status(m, 8, 24)
    c3[status] += 1
    if status == "valid":
        c3["valid_disagree" if dw != RE[(m, 8)][1] else "valid_agree"] += 1
        t, c = F[m][8]
        t2, c2 = F[m][9]
        c3[("valid_tie_t_eq_c", t == c)] += 1
print("   status counts / disagreements:", sorted(c3.items(), key=lambda kv: str(kv[0])))
