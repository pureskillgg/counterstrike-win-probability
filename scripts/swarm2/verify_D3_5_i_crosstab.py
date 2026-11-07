"""D3-5 verifier, step I: cross-tab of round witnesses by engine halftime type and round number (2024 G: tome).

Witnesses per (match, round), all as side codes (2 = T, 3 = CT):
  W   : round_end.winner_team_code (player_count 20 preferred, first row)
  R   : side implied by round_end.win_reason_code (mapping learned from bomb rows: check printed table)
  B   : bomb side (bomb_defused -> CT, bomb_exploded -> T), only where a bomb outcome exists
  D   : round_state freeze delta side (unit delta only; excluded at a swap transition)
Matches grouped by engine halftime h (first round with phase 'second'): h8, h12, unknown. Day group 12-25 vs <12.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count", "win_reason_code"])
RE = (RE.with_columns((pl.col("player_count") == 20).cast(pl.Int8).alias("p20"))
      .sort(["match_key", "round", "p20"], descending=[False, False, True])
      .unique(["match_key", "round"], keep="first"))
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
sec = RS.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("h"))
hmap = {m: h for m, h in sec.iter_rows()}
BS = rd("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bomb = {}
for m, r, ev in BS.iter_rows():
    bomb[(m, r)] = CT if ev == "bomb_defused" else T

W, RR = {}, {}
for m, r, w, rc in RE.select(["match_key", "round", "winner_team_code", "win_reason_code"]).iter_rows():
    W[(m, r)] = w
    RR[(m, r)] = rc
F = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}

# learn reason-code mapping from bomb rows
mp = collections.Counter()
for k, b in bomb.items():
    if k in RR:
        mp[(RR[k], b)] += 1
print("reason code vs bomb side (bomb rows):", sorted(mp.items()))
reason_side_map = {}
for (rc, side), n in mp.items():
    if rc not in reason_side_map or mp[(rc, reason_side_map[rc])] < n:
        reason_side_map[rc] = side
print("reason -> side mapping used:", reason_side_map)

day = lambda m: int(m.split("/")[3])
grp = lambda m: "12-25" if day(m) >= 12 else "<12"
htype = lambda m: f"h{hmap[m]}" if m in hmap else "hNA"

tab = collections.defaultdict(collections.Counter)
for (m, r), w in W.items():
    if (m, r) not in F or (m, r + 1) not in F:
        continue
    t1, c1 = F[(m, r)]
    t2, c2 = F[(m, r + 1)]
    dt, dc = t2 - t1, c2 - c1
    D = T if (dt, dc) == (1, 0) else (CT if (dt, dc) == (0, 1) else None)
    key = (grp(m), htype(m), r)
    if D is None:
        tab[key]["D_unclear"] += 1
        continue
    tab[key]["n"] += 1
    tab[key]["W_eq_D"] += int(w == D)
    rc = RR.get((m, r))
    if rc in reason_side_map:
        tab[key]["R_eq_D"] += int(reason_side_map[rc] == D)
    if (m, r) in bomb:
        tab[key]["bomb_n"] += 1
        tab[key]["bomb_eq_D"] += int(bomb[(m, r)] == D)
        tab[key]["bomb_eq_W"] += int(bomb[(m, r)] == w)

print("\n grp    htype  r   n     W=D    R=D   bomb_n  bomb=D  bomb=W")
for key in sorted(tab, key=lambda k: (k[0], k[1], k[2])):
    if key[2] not in (1, 2, 8, 9, 10, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30):
        continue
    c = tab[key]
    print(f"{key[0]:6} {key[1]:6} {key[2]:>3} {c['n']:>5} {c['W_eq_D']:>6} {c['R_eq_D']:>6} {c['bomb_n']:>7} "
          f"{c['bomb_eq_D']:>7} {c['bomb_eq_W']:>7}   unclear={c['D_unclear']}")
