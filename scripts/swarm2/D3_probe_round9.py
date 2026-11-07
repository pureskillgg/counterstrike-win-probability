"""D3 probe: unclear score deltas by round, and round-9 disagreement examples (2024 G: tome)."""
import collections, glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"); SUF = ".2023-12-28,2024-02-25"
def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
RE = g_read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
win = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
unclear = collections.Counter(); unclear_day = collections.Counter(); tot = collections.Counter()
ex = []
for (m, r), (t1, c1) in live.items():
    day = int(m.split("/")[3])
    if day < 12 or (m, r + 1) not in live or r in (12, 27, 33):
        continue
    t2, c2 = live[(m, r + 1)]
    dt, dc = t2 - t1, c2 - c1
    tot[r] += 1
    if not ((dc == 1 and dt == 0) or (dt == 1 and dc == 0)):
        unclear[r] += 1
        if (dt, dc) not in [(0, 0)]:
            unclear_day[(dt, dc)] += 1
print("day12-25 rounds with delta not a single-side +1 (round: unclear/total) for rounds 1..12:")
print([(r, unclear[r], tot[r]) for r in range(1, 13)])
print("unclear delta patterns (dt, dc) counts:", unclear_day.most_common(8))
# round-9 disagreement examples
shown = 0
for (m, r), (t1, c1) in live.items():
    if r != 9 or int(m.split("/")[3]) < 12 or (m, 10) not in live or (m, 9) not in win:
        continue
    t2, c2 = live[(m, 10)]
    dt, dc = t2 - t1, c2 - c1
    ds = 3 if (dc == 1 and dt == 0) else (2 if (dt == 1 and dc == 0) else None)
    if ds is not None and ds != win[(m, 9)] and shown < 4:
        seq = [(rr, win.get((m, rr)), live.get((m, rr))) for rr in range(6, 12)]
        print("\nexample", m, "winners/freeze(t,ct) rounds 6..11:", seq)
        shown += 1
