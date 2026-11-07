"""verify_D3_3 part C: which source is wrong at the disagreeing rounds?

Checks (all read-only, 2024 tome):
  C1. Per round k (MR24 and MR16 matches, day groups 01-07 and 12-25): winner(k) from round_end vs the score-only
      winner from the freeze boundary k -> k+1, using the schedule-consistent class (noswap unless swap_expected).
      Only unambiguous boundaries counted. This is the claim's test with the schedule applied per match.
  C2. Tick ordering: is round_end(k).tick between freeze(k).tick and freeze(k+1).tick? (round_end misattributed?)
  C3. win_reason_code vs winner_team_code: does the reason code agree with the winner on non-disputed rounds, and
      what does it say on the disputed round-13 rows?
  C4. Bomb cross-check on the same rows: bomb_planted player_team_code, bomb_exploded/defused presence.
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


ann = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).group_by("match_key").agg(pl.col("round").min().alias("r_ann")).collect())
MR = {m: int(r) * 2 + 2 for m, r in ann.iter_rows()}
fe_df = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_freeze_end")
         .select(["match_key", "round", "tick", "t_score", "ct_score"]).collect().sort("tick")
         .group_by(["match_key", "round"]).agg(pl.col("tick").first(), pl.col("t_score").first(),
                                               pl.col("ct_score").first()))
F = collections.defaultdict(dict)
FT = collections.defaultdict(dict)
for m, r, tk, t, c in fe_df.select(["match_key", "round", "tick", "t_score", "ct_score"]).iter_rows():
    F[m][int(r)] = (int(t), int(c))
    FT[m][int(r)] = int(tk)
re_df = (pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code",
                                                    "win_reason_code", "player_count"]).collect()
         .sort("tick").group_by(["match_key", "round"]).agg(pl.col("tick").first(),
                                                            pl.col("winner_team_code").first(),
                                                            pl.col("win_reason_code").first(),
                                                            pl.col("player_count").first()))
RE = {}
for m, r, tk, w, rc, p in re_df.select(["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                       "player_count"]).iter_rows():
    RE[(m, int(r))] = (int(tk), int(w), int(rc), int(p))


def day(m):
    return int(m.split("/")[3])


def sched_winner(k, mr, t, c, t2, c2):
    """Winner from boundary k -> k+1 using the schedule-consistent class; None if the class does not fit."""
    dT, dC = t2 - t, c2 - c
    xT, xC = t2 - c, c2 - t
    nos = dT in (0, 1) and dC in (0, 1) and dT + dC == 1
    swp = xT in (0, 1) and xC in (0, 1) and xT + xC == 1
    if swap_expected(k, mr):
        if swp:
            return CT if xT == 1 else T, "swap"
        return None, "no_swap_fit"
    if nos:
        return (T if dT == 1 else CT), "noswap"
    return None, "no_noswap_fit"


# C1 per round, schedule-consistent, for MR24 and MR16
print("== C1: round_end winner vs schedule-consistent score winner, per round ==")
res = collections.defaultdict(lambda: [0, 0, 0])  # agree, disagree, skipped
for m in F:
    mr = MR.get(m, 24)
    for k in sorted(F[m]):
        if k + 1 not in F[m] or (m, k) not in RE:
            continue
        t, c = F[m][k]
        t2, c2 = F[m][k + 1]
        w, _ = sched_winner(k, mr, t, c, t2, c2)
        key = ("d01_07" if day(m) <= 7 else "d12_25", mr, k)
        if w is None:
            res[key][2] += 1
            continue
        if w == RE[(m, k)][1]:
            res[key][0] += 1
        else:
            res[key][1] += 1
for grp in ("d01_07", "d12_25"):
    for mr in (16, 24):
        rows = sorted((k, v) for (g, mm, k), v in res.items() if g == grp and mm == mr)
        bad = [(k, v[1], v[0] + v[1]) for k, v in rows if v[1] > 0]
        tot_bad = sum(v[1] for _, v in rows)
        tot = sum(v[0] + v[1] for _, v in rows)
        print(f"  {grp} MR{mr}: disagree/compared {tot_bad}/{tot}; rounds with disagreement (k, dis, n): {bad}")

# C2 tick ordering around the disputed rounds
print("\n== C2: tick ordering freeze(k) < round_end(k) < freeze(k+1) ==")
ord_cnt = collections.Counter()
for m in F:
    if day(m) < 12:
        continue
    for k in (8, 9, 12, 13, 14, 25, 28):
        if (m, k) in RE and k in F[m] and k + 1 in F[m]:
            tk = RE[(m, k)][0]
            ok = FT[m][k] < tk < FT[m][k + 1]
            ord_cnt[(MR.get(m, 24), k, "ok" if ok else "outside")] += 1
for key in sorted(ord_cnt):
    print("  ", key, ord_cnt[key])

# C3 win reason vs winner, non-disputed rounds vs disputed round 13 (MR24, day 12-25)
print("\n== C3: win_reason_code vs winner_team_code ==")
rc_win = collections.Counter()
rc_disp = collections.Counter()
disputed = set()
for m in F:
    mr = MR.get(m, 24)
    if day(m) < 12 or mr != 24:
        continue
    k = 13
    if k + 1 in F[m] and k in F[m] and (m, k) in RE:
        t, c = F[m][k]
        t2, c2 = F[m][k + 1]
        if swap_expected(k, mr):
            continue
        dT, dC = t2 - t, c2 - c
        if dC == 1 and dT == 0:
            ds = CT
        elif dT == 1 and dC == 0:
            ds = T
        else:
            continue
        if ds != RE[(m, k)][1]:
            disputed.add(m)
for (m, k), (tk, w, rc, p) in RE.items():
    if k == 13 and m in disputed:
        rc_disp[(w, rc)] += 1
    elif day(m) >= 12 and MR.get(m, 24) == 24 and k not in (13,):
        rc_win[(w, rc)] += 1
print("  reason codes (winner, reason) for non-disputed MR24 d12_25 rounds (top 20):")
for kk, v in rc_win.most_common(20):
    print("   ", kk, v)
print("  reason codes (winner, reason) for disputed round-13 rows:")
for kk, v in rc_disp.most_common(20):
    print("   ", kk, v)

# C4 bomb events on round 13
print("\n== C4: bomb outcome on round 13 (MR24, day 12-25) ==")
bs = (pl.scan_parquet(files("bomb_state")).filter(pl.col("event_type").is_in(
      ["bomb_planted", "bomb_exploded", "bomb_defused"]))
      .select(["match_key", "round", "event_type", "player_team_code"]).collect())
b13 = bs.filter((pl.col("round") == 13)).group_by(["match_key", "event_type"]).agg(pl.col("player_team_code").first())
outc = collections.defaultdict(dict)
for m, ev, pt in b13.iter_rows():
    outc[m][ev] = pt
cnt = collections.Counter()
for m in disputed:
    if m not in outc:
        cnt["disputed_no_bomb_row"] += 1
        continue
    d = outc[m]
    if "bomb_exploded" in d:
        bomb_side = T
    elif "bomb_defused" in d:
        bomb_side = CT
    else:
        cnt["disputed_planted_no_end"] += 1
        continue
    cnt[("bomb_side", bomb_side, "rend_winner", RE[(m, 13)][1])] += 1
    cnt[("planter_side", d.get("bomb_planted"))] += 1
print("  disputed round-13 matches with a bomb row:", sum(v for k, v in cnt.items() if isinstance(k, tuple) and k[0] == "bomb_side"))
for k, v in sorted(cnt.items(), key=lambda kv: str(kv[0])):
    print("   ", k, v)
print("  disputed matches total:", len(disputed))
