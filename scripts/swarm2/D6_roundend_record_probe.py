"""D6 probe: is the affected round-13 round_end record (winner, reason, tick) an exact copy of another round_end
record, in the same match or in another match? Also tick positions of bomb events in round 13 vs round_end 13."""
import sys, glob, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
def read(ch, cols):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(fd / "dataframe_*")))], how="vertical_relaxed")
RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count", "second"])
BS = read("bomb_state", ["match_key", "round", "event_type", "tick"])
RS = read("round_state", ["match_key", "round", "event_type", "tick"])
FE = RS.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
fe = {(m, r): tk for m, r, tk in FE.select(["match_key", "round", "tick"]).iter_rows()}
RE1 = RE.unique(["match_key", "round"], keep="first")
rec = {(m, r): (w, rc, tk) for m, r, tk, w, rc in RE1.select(["match_key","round","tick","winner_team_code","win_reason_code"]).iter_rows()}
# affected set (e2 definition)
bo = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key","round"]).agg(pl.col("event_type").unique().alias("evs"))
r13 = (RE1.filter((pl.col("round")==13) & (pl.col("player_count")==20)).join(bo.filter(pl.col("round")==13), on=["match_key","round"])
       .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bs"))
       .with_columns((pl.col("bs") != pl.col("winner_team_code")).alias("mm")))
aff = set(r13.filter(pl.col("mm"))["match_key"].to_list())
ctl = set(r13.filter(~pl.col("mm"))["match_key"].to_list())
print(f"affected: {len(aff)}, control with bomb: {len(ctl)}")
# 1. same-match copies
same_copy = collections.Counter()
for m in aff:
    w13, rc13, tk13 = rec[(m, 13)]
    hits = [k for (mm, k), v in rec.items() if mm == m and k != 13 and v[:2] == (w13, rc13)]
    same_copy["has same winner+reason in another round of same match" if hits else "no same-match copy"] += 1
print("affected: round_end 13 record copies within match:", dict(same_copy))
# 2. cross-match copies by (winner, reason, tick)
by_tick = collections.defaultdict(list)
for (m, r), (w, rc, tk) in rec.items():
    by_tick[tk].append((m, r))
cross = collections.Counter()
for m in aff:
    w13, rc13, tk13 = rec[(m, 13)]
    others = [x for x in by_tick[tk13] if x != (m, 13)]
    cross["same tick in another match/round" if others else "tick unique"] += 1
print("affected: round_end 13 tick shared with another record:", dict(cross))
# 3. reason codes for affected round_end 13 vs control
rc_aff = collections.Counter(rec[(m, 13)][1] for m in aff)
rc_ctl = collections.Counter(rec[(m, 13)][1] for m in ctl)
print("reason codes, round_end 13, affected:", dict(rc_aff))
print("reason codes, round_end 13, control:", dict(rc_ctl))
# 4. bomb event ticks vs round_end 13 tick and freeze_end 13/14
pos = collections.Counter()
for m in aff | ctl:
    b = BS.filter((pl.col("match_key")==m) & (pl.col("round")==13) & pl.col("event_type").is_in(["bomb_defused","bomb_exploded"]))
    if b.height == 0: continue
    tk_b = b["tick"][0]
    grp = "affected" if m in aff else "control"
    pos[(grp, "bomb tick < round_end13 tick" if tk_b < rec[(m,13)][2] else "bomb tick >= round_end13 tick")] += 1
print("bomb outcome tick vs round_end 13 tick:", dict(pos))
