"""D3-5 verifier, step E: swap-free check of round_end winners against round_state score deltas and bomb outcomes.

For round r, the freeze delta is (t, ct)[freeze r+1] - (t, ct)[freeze r]. Fields are side-based, so a +1 on the
T field means the T side won round r, a +1 on the CT field means the CT side won, provided no side swap happened
between the two freeze rows. Transitions where a swap can happen (round 12 under MR12 halftime, and OT rounds)
are reported separately and are not used for the main count. Only unit deltas (1,0) or (0,1) are used.
Bomb side (bomb_state): bomb_defused -> CT (3), bomb_exploded -> T (2). Score delta is compared with round_end winner
and with bomb side. Reported by round number and day group. Matches without round 1 (starts at 2) are included.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"])
RE = (RE.with_columns((pl.col("player_count") == 20).cast(pl.Int8).alias("p20"))
      .sort(["match_key", "round", "p20"], descending=[False, False, True])
      .unique(["match_key", "round"], keep="first"))
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
BS = rd("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded", "bomb_planted"]))
print("bomb_state event types:", BS.group_by("event_type").len().rows())
bomb = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))

W = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
B = {(m, r): ("ct" if "bomb_defused" in evs else "t") for m, r, evs in bomb.iter_rows()}
F = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
day = lambda m: int(m.split("/")[3])
grp = lambda m: "12-25" if day(m) >= 12 else "<12"

stat = collections.Counter()
by_r = collections.defaultdict(collections.Counter)
mism_matches = collections.defaultdict(set)
for (m, r), (t1, c1) in F.items():
    if (m, r + 1) not in F:
        stat["no_next_freeze"] += 1
        continue
    t2, c2 = F[(m, r + 1)]
    dt, dc = t2 - t1, c2 - c1
    if (dt, dc) == (1, 0):
        D = T
    elif (dt, dc) == (0, 1):
        D = CT
    else:
        D = None
    g = grp(m)
    key = (g, r)
    if D is None:
        by_r[key]["unclear"] += 1
        continue
    by_r[key]["valid"] += 1
    if (m, r) in W:
        if W[(m, r)] == D:
            by_r[key]["winner_agrees"] += 1
        else:
            by_r[key]["winner_disagrees"] += 1
            mism_matches[(g, r)].add(m)
    if (m, r) in B and (m, r) in W:
        by_r[key]["bomb_round"] += 1
        bside = T if B[(m, r)] == "t" else CT
        by_r[key]["bomb_agrees_delta"] += int(bside == D)
        by_r[key]["bomb_agrees_winner"] += int(bside == W[(m, r)])
print("\nper round and day group: valid unit-delta rounds; delta vs winner; delta vs bomb (bomb rounds only)")
print(f"{'grp':6} {'r':>3} {'valid':>6} {'w_agree':>7} {'w_dis':>6} {'bomb_n':>6} {'b_vs_delta':>10} {'b_vs_winner':>11} {'matches_w_dis':>13}")
for key in sorted(by_r, key=lambda k: (k[0], k[1])):
    c = by_r[key]
    if key[1] > 30:
        continue
    print(f"{key[0]:6} {key[1]:>3} {c['valid']:>6} {c['winner_agrees']:>7} {c['winner_disagrees']:>6} "
          f"{c['bomb_round']:>6} {c['bomb_agrees_delta']:>10} {c['bomb_agrees_winner']:>11} "
          f"{len(mism_matches.get(key, set())):>13}")
print("\nstats:", dict(stat))
