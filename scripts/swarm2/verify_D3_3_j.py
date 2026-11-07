"""verify_D3_3 part J: are the bomb events on disputed round-13 rows inside that round's own window
[freeze_end(13), round_end(13)]? Also the planter side on those rows. In-scope unique MR24 day 12-25 rows only."""
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


fe = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_freeze_end")
      .select(["match_key", "round", "tick"]).collect().group_by(["match_key", "round"]).agg(pl.col("tick").min()))
FT = {(m, int(r)): int(t) for m, r, t in fe.iter_rows()}
re_u = (pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code"]).collect()
        .group_by(["match_key", "round"]).agg(pl.col("tick").min(), pl.col("winner_team_code").first()))
RE = {(m, int(r)): (int(t), int(w)) for m, r, t, w in re_u.iter_rows()}
bs = (pl.scan_parquet(files("bomb_state")).filter(pl.col("event_type").is_in(
      ["bomb_planted", "bomb_exploded", "bomb_defused"])).select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect())
B = collections.defaultdict(dict)
planter = {}
for m, r, tk, ev, pt in bs.iter_rows():
    B[(m, int(r))].setdefault(ev, int(tk))
    if ev == "bomb_planted":
        planter[(m, int(r))] = pt
cnt = collections.Counter()
for (m, k), (tk, w) in RE.items():
    if k != 13 or int(m.split("/")[3]) < 12 or (m, k) not in FT:
        continue
    ev = B.get((m, k), {})
    if "bomb_exploded" in ev or "bomb_defused" in ev:
        e = "bomb_exploded" if "bomb_exploded" in ev else "bomb_defused"
        t_ev = ev[e]
        inside = FT[(m, k)] <= t_ev <= tk
        cnt[("bomb_in_round_window", inside)] += 1
        cnt[("planter_side", planter.get((m, k)))] += 1
        cnt[("implied_winner_is_round_end", (T if e == "bomb_exploded" else CT) == w)] += 1
print("round-13 MR24 day12-25 rows with an exploded or defused event (all, pre-scope):")
for k, v in sorted(cnt.items(), key=lambda kv: str(kv[0])):
    print("  ", k, v)
