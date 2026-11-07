"""A13 part 2: whole-match hypothesis test for round_state score attachment (2026 tome).

Why: the boundary test (A13_score_attach.py) leaves ambiguous boundaries unresolved. Here each
round is tested against whole-match hypotheses, using cumulative wins from round winners.

Definitions (per match, per round r that has a round_freeze_end row):
  sx(r)  = side of team X in round r (X = the five players on code 2 at the first round with a
           clean 5 v 5 player_info split). sx(r) is None if the split is not clean.
  S_X(r) = number of rounds q < r with winner_team_code(q) == sx(q)  (team X's wins before r)
  S_Y(r) = (r - 1) - S_X(r)                                       (team Y's wins before r)
  The round is evaluated only if every round q < r has a winner and a known sx(q).
Hypotheses for the freeze pair (t, ct) at round r:
  side  : t = score of the team on the T side  (S_X if sx(r)==2 else S_Y), ct = the other
  teamX : t = S_X(r), ct = S_Y(r)     (field follows team X)
  teamY : t = S_Y(r), ct = S_X(r)     (field follows team Y)
A hypothesis is a candidate for a match if it holds in every evaluated round.
Match classes:
  side_only  : side is a candidate, neither team hypothesis is
  team_only  : a team hypothesis is a candidate, side is not
  undetermined: side and a team hypothesis are both candidates (no swap in the evaluated rounds)
  neither    : no hypothesis fits every evaluated round (data or rule problem)
  insufficient: no round could be evaluated

ASSUMPTION: winner_team_code is side-coded (2 = T, 3 = CT) in 2026. Supported by the win-reason map
(Target_Bombed -> 2, CTs_Win -> 3) and by feature_validation section 3. Not re-checked here.

Read-only. Output: stdout only (captured to logs/swarm2/A13_score_attach_full.log).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes/2026-10-01_to_08/tome/csds"
SUFFIX = ".2026-09-18,2026-10-08"


def read(channel, cols):
    folder = TOME / ("header" if channel == "header" else f"basic_{channel}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


print("== A13 whole-match score attachment test (2026 tome) ==")
hdr = read("header", ["key", "max_rounds"])
keys = list(hdr["key"])
print(f"header matches: {len(keys)}")
RE = read("round_end", ["match_key", "round", "winner_team_code"])
RS = read("round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
PI = read("player_info", ["match_key", "round", "player_id", "team_code"])

win = {}
for x in RE.iter_rows(named=True):
    win[(x["match_key"], int(x["round"]))] = x["winner_team_code"]
frz = {}
for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
    k = (x["match_key"], int(x["round"]))
    if k not in frz and x["t_score"] is not None and x["ct_score"] is not None:
        frz[k] = (int(x["t_score"]), int(x["ct_score"]))
pl_codes = collections.defaultdict(dict)
for x in PI.iter_rows(named=True):
    if x["team_code"] in (2, 3):
        pl_codes[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]

rounds_by_match = collections.defaultdict(set)
for (m, r) in pl_codes:
    rounds_by_match[m].add(r)
for (m, r) in frz:
    rounds_by_match[m].add(r)
for (m, r) in win:
    rounds_by_match[m].add(r)

# team X from the first clean 5 v 5 round
team_x = {}
side_of = {}  # (match, round) -> sx or None
for m in keys:
    rs = sorted(rounds_by_match.get(m, ()))
    x_set = None
    for r in rs:
        codes = pl_codes.get((m, r), {})
        if len(codes) == 10 and sum(1 for v in codes.values() if v == 2) == 5:
            x_set = {p for p, v in codes.items() if v == 2}
            break
    if x_set is None:
        continue
    team_x[m] = x_set
    for r in rs:
        codes = pl_codes.get((m, r), {})
        if len(codes) != 10:
            side_of[(m, r)] = None
            continue
        xs = [codes.get(p) for p in x_set]
        if all(v == 2 for v in xs if v is not None) and all(v is not None for v in xs):
            side_of[(m, r)] = 2
        elif all(v == 3 for v in xs if v is not None) and all(v is not None for v in xs):
            side_of[(m, r)] = 3
        else:
            side_of[(m, r)] = None

print(f"matches with a clean 5 v 5 starting round (team X defined): {len(team_x)}")

cls = {}
evals = {}
mismatch_detail = collections.Counter()
for m in keys:
    if m not in team_x:
        cls[m] = "insufficient"
        continue
    rs = sorted(r for r in rounds_by_match[m] if (m, r) in frz)
    cand = {"side": True, "teamX": True, "teamY": True}
    n_eval = 0
    n_post_swap = 0
    prev_sx = None
    prefix_ok = True  # True while every earlier round has a winner and a known side of team X
    s_x = 0           # team X wins among rounds before r (only valid while prefix_ok)
    max_r = max(rounds_by_match[m]) if rounds_by_match[m] else 0
    for r in range(1, max_r + 1):
        sx = side_of.get((m, r))
        if (m, r) in frz and prefix_ok and sx is not None:
            t, c = frz[(m, r)]
            s_y = (r - 1) - s_x
            t_side, c_side = (s_x, s_y) if sx == 2 else (s_y, s_x)
            side_ok = (t, c) == (t_side, c_side)
            cand["side"] &= side_ok
            cand["teamX"] &= (t, c) == (s_x, s_y)
            cand["teamY"] &= (t, c) == (s_y, s_x)
            n_eval += 1
            if prev_sx is not None and sx != prev_sx:
                n_post_swap += 1
            prev_sx = sx
            if not side_ok:
                mismatch_detail["side_fail"] += 1
        # advance the prefix with this round's winner; the prefix breaks if the winner or side is unknown
        w = win.get((m, r))
        if prefix_ok and w in (2, 3) and sx is not None:
            if w == sx:
                s_x += 1
        else:
            prefix_ok = False
    evals[m] = (n_eval, n_post_swap)
    if n_eval == 0:
        cls[m] = "insufficient"
        continue
    side_c = cand["side"]
    team_c = cand["teamX"] or cand["teamY"]
    if side_c and team_c:
        cls[m] = "undetermined"
    elif side_c:
        cls[m] = "side_only"
    elif team_c:
        cls[m] = "team_only"
    else:
        cls[m] = "neither"

print("\nHypothesis classes (all 2026 matches):")
allc = collections.Counter(cls.get(m, "insufficient") for m in keys)
print(f"  {dict(allc)}")


def block(label, subset):
    c = collections.Counter(cls.get(m, "insufficient") for m in subset)
    n = len(subset)
    share = {k: round(v / n, 4) if n else 0 for k, v in c.items()}
    print(f"  {label}: n={n}; {dict(c)}; shares {share}")


day = lambda k: k.split("/")[3]
print("\nBy key day:")
block("days 01-02", [m for m in keys if day(m) in ("01", "02")])
block("days 03-10", [m for m in keys if day(m) not in ("01", "02")])
for d in sorted({day(m) for m in keys}):
    block(f"day {d}", [m for m in keys if day(m) == d])

post = [m for m in keys if evals.get(m, (0, 0))[1] > 0]
print(f"\nmatches with at least one evaluated post-swap round: {len(post)}")
block("post-swap matches, days 01-02", [m for m in post if day(m) in ("01", "02")])
block("post-swap matches, all", post)

print("\nMismatch counts (rounds failing side hypothesis):", dict(mismatch_detail))
team_only = [m for m in keys if cls.get(m) == "team_only"]
print(f"team_only matches: {len(team_only)}; by day {dict(collections.Counter(day(m) for m in team_only))}")
neither = [m for m in keys if cls.get(m) == "neither"]
print(f"neither matches: {len(neither)}; by day {dict(collections.Counter(day(m) for m in neither))}")
print(f"example neither keys: {[m[:45] for m in neither[:5]]}")
