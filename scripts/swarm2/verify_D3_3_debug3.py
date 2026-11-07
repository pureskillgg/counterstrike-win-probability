"""verify D3-3 debug3: agreeing matches (W13 == D13) with complete rounds 1..16 (2024 G: tome). Read-only."""
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
full = [m for m in matches if all((m, r) in W and (m, r) in F and TS.get((m, r)) and CS.get((m, r)) for r in range(1, 17))]
print("matches with complete rounds 1..16 (W, freeze, player sides):", len(full))

cls = collections.Counter()
examples = []
for m in full:
    t13, c13 = F[(m, 13)]
    t14, c14 = F[(m, 14)]
    dt, dc = t14 - t13, c14 - c13
    if dt == 1 and dc == 0:
        d = T
    elif dc == 1 and dt == 0:
        d = CT
    else:
        cls["unclear13"] += 1
        continue
    cls["agree13" if W[(m, 13)] == d else "dis13"] += 1
print("class counts:", dict(cls))

# freeze-14 check for agreeing matches, using A = T-side team at round 1 (player_info at round 1)
res = collections.Counter()
shown = 0
for m in full:
    A = TS[(m, 1)]
    cA = cB = 0
    bad_round = None
    for r in range(1, 15):
        on_t = "A" if len(TS[(m, r)] & A) >= len(CS[(m, r)] & A) else "B"
        t, c = F[(m, r)]
        expT = cA if on_t == "A" else cB
        expC = cB if on_t == "A" else cA
        if (t, c) != (expT, expC):
            bad_round = r
            break
        w = W[(m, r)]
        winner = on_t if w == T else ("B" if on_t == "A" else "A")
        if winner == "A":
            cA += 1
        else:
            cB += 1
    # check freeze chain for round 14 only (uses W through 13)
    side13 = None
    t13, c13 = F[(m, 13)]
    t14, c14 = F[(m, 14)]
    d13 = T if (t14 - t13 == 1 and c14 - c13 == 0) else (CT if (c14 - c13 == 1 and t14 - t13 == 0) else None)
    key = "agree13" if W[(m, 13)] == d13 else ("dis13" if d13 is not None else "unclear")
    res[(key, "first_bad_round=" + ("none<=14" if bad_round is None else str(bad_round)))] += 1
    if key == "agree13" and bad_round is not None and shown < 3:
        shown += 1
        print(f"\nagreeing match ...{m[-40:]} first bad round {bad_round}")
        cA = cB = 0
        for r in range(1, 16):
            on_t = "A" if len(TS[(m, r)] & A) >= len(CS[(m, r)] & A) else "B"
            t, c = F[(m, r)]
            expT = cA if on_t == "A" else cB
            expC = cB if on_t == "A" else cA
            print(f"  r={r:2d} W={W[(m, r)]} freeze=({t},{c}) onT={on_t} expect=({expT},{expC}) ok={(t, c) == (expT, expC)}")
            w = W[(m, r)]
            winner = on_t if w == T else ("B" if on_t == "A" else "A")
            if winner == "A":
                cA += 1
            else:
                cB += 1
print("\nfirst-bad-round by agree/dis class (complete matches):")
for k, v in sorted(res.items()):
    print("  ", k, v)
