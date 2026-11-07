"""verify D3-3 debug2: trace agreeing matches (W13 == side delta D13) through round 14..16 (2024 G: tome).

Purpose: explain why the team-cumulative freeze check (alt script section 2) is 100% bad at round 14.
Read-only.
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
N_SHOW = 3


def chan_files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def scan(ch):
    return pl.scan_parquet(chan_files(ch), missing_columns="insert", extra_columns="ignore")


RE = (scan("round_end").select(["match_key", "round", "tick", "winner_team_code"]).collect()
      .sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True))
W = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
RS = scan("round_state").filter(pl.col("event_type") == "round_freeze_end").select(
    ["match_key", "round", "tick", "t_score", "ct_score"]).collect()
FE = RS.sort("tick").group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first())
F = {(m, r): (t, c) for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
PI = scan("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
TS, CS = collections.defaultdict(set), collections.defaultdict(set)
for m, r, pid, tc in PI.iter_rows():
    if tc == T:
        TS[(m, r)].add(pid)
    elif tc == CT:
        CS[(m, r)].add(pid)

matches = sorted({m for (m, _) in W})
rounds_by_m = collections.defaultdict(list)
for (m, r) in W:
    rounds_by_m[m].append(r)

# agreeing matches: round 13 both present and W13 == side delta D13 (T if t +1, CT if c +1)
agree, dis = [], []
for m in matches:
    if (m, 13) in W and (m, 13) in F and (m, 14) in F:
        t1, c1 = F[(m, 13)]
        t2, c2 = F[(m, 14)]
        dt, dc = t2 - t1, c2 - c1
        if dt == 1 and dc == 0:
            d = T
        elif dc == 1 and dt == 0:
            d = CT
        else:
            continue
        (agree if W[(m, 13)] == d else dis).append(m)
print(f"matches with valid D13: agree {len(agree)}, disagree {len(dis)}")


def A_team(m):
    for r in sorted(rounds_by_m[m]):
        if TS.get((m, r)) and CS.get((m, r)):
            return TS[(m, r)]
    return None


def trace(m, upto=16):
    A = A_team(m)
    print(f"\n==== match ...{m[-40:]} (agree-or-disagree at 13 shown in header)")
    print(" r   W  freeze(t,c)  onT(A/B)  cumA cumB  expectT,expectC  ok?")
    cA = cB = 0
    for r in range(1, upto + 1):
        if (m, r) not in W:
            print(f"{r:2d}  missing round_end")
            break
        ts, cs = TS.get((m, r), set()), CS.get((m, r), set())
        if A is None or not ts or not cs:
            print(f"{r:2d} no player info")
            break
        on_t = "A" if len(ts & A) >= len(cs & A) else "B"
        f = F.get((m, r))
        exp_t = cA if on_t == "A" else cB
        exp_c = cB if on_t == "A" else cA
        ok = f is not None and f == (exp_t, exp_c)
        print(f"{r:2d}  {W[(m, r)]}  {str(f):<12} {on_t:<8}  {cA:4d} {cB:4d}  {str((exp_t, exp_c)):<15} {ok}")
        w = W[(m, r)]
        if w == T:
            winner = on_t
        else:
            winner = "B" if on_t == "A" else "A"
        if winner == "A":
            cA += 1
        else:
            cB += 1


for m in agree[:N_SHOW]:
    trace(m)
for m in dis[:N_SHOW]:
    trace(m)

# overall: for agreeing matches, is the freeze(14) consistent with cumulative W through 13?
ok14 = collections.Counter()
for m in agree:
    A = A_team(m)
    cA = cB = 0
    good = True
    for r in range(1, 15):
        if (m, r) not in W or A is None:
            good = False
            break
        ts, cs = TS.get((m, r), set()), CS.get((m, r), set())
        if not ts or not cs:
            good = False
            break
        on_t = "A" if len(ts & A) >= len(cs & A) else "B"
        if r == 14 and (m, 14) in F:
            f = F[(m, 14)]
            ok14["agree_match_freeze14_ok" if f == ((cA if on_t == "A" else cB), (cB if on_t == "A" else cA))
                 else "agree_match_freeze14_bad"] += 1
            break
        w = W[(m, r)]
        winner = on_t if w == T else ("B" if on_t == "A" else "A")
        if winner == "A":
            cA += 1
        else:
            cB += 1
print("\nagreeing matches, freeze14 vs cumulative W through 13:", dict(ok14))
