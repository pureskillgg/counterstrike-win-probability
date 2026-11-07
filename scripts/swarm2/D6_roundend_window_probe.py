"""D6 probe: which round window does the affected round-13 round_end tick fall in? Delta to bomb tick (ticks, 64/s)."""
import sys, glob, collections, statistics
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
def read(ch, cols):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(fd / "dataframe_*")))], how="vertical_relaxed")
RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"]).unique(["match_key","round"], keep="first")
BS = read("bomb_state", ["match_key", "round", "event_type", "tick"])
RS = read("round_state", ["match_key", "round", "event_type", "tick"])
FE = RS.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
fe = {(m, r): tk for m, r, tk in FE.select(["match_key","round","tick"]).iter_rows()}
bo = BS.filter(pl.col("event_type").is_in(["bomb_defused","bomb_exploded"])).group_by(["match_key","round"]).agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("btick"))
R = RE.join(bo, on=["match_key","round"], how="inner").with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bs"))
R = R.filter(pl.col("player_count") == 20)
aff = set(R.filter((pl.col("round")==13) & (pl.col("bs") != pl.col("winner_team_code")))["match_key"].to_list())
ctl = set(R.filter((pl.col("round")==13) & (pl.col("bs") == pl.col("winner_team_code")))["match_key"].to_list())
def window(m, tk):
    # round whose [freeze_end, next freeze_end) window contains tk
    rounds = sorted(r for (mm, r) in fe if mm == m)
    w = None
    for r in rounds:
        if fe[(m, r)] <= tk:
            w = r
    return w
for grp, ms in [("affected", aff), ("control", ctl)]:
    sub = R.filter((pl.col("round") == 13) & pl.col("match_key").is_in(list(ms)))
    win = collections.Counter()
    deltas = []
    for m, tk, bt in sub.select(["match_key", "tick", "btick"]).iter_rows():
        win[window(m, tk)] += 1
        deltas.append((tk - bt) / 64.0)
    print(f"{grp}: n={sub.height}; round_end13 tick falls in window of round: {dict(sorted(win.items(), key=lambda x: (x[0] is None, x[0] or 0)))}")
    print(f"   round_end13 tick minus bomb tick (seconds): median={statistics.median(deltas):.2f} "
          f"min={min(deltas):.2f} max={max(deltas):.2f}")
