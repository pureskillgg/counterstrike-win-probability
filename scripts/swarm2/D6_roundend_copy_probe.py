"""D6 probe: is the round-13 round_end row a copy of round 12's result (stale event)? Uses round_end tick,
duplicate counts, and comparison with score-derived winners of rounds 12 and 13 (2024)."""
import sys, glob, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T, CT = 2, 3

def read(ch, cols):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(fd / "dataframe_*")))], how="vertical_relaxed")

RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
RS = read("round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"])
FE = RS.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
dup = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
print("round_end rows per (match, round): ", dup["n"].value_counts().sort("n").rows())
d13 = dup.filter(pl.col("round") == 13)["n"].value_counts().sort("n").rows()
print("round 13 rows per match:", d13)
dupkeys = dup.filter(pl.col("n") > 1)
print("(match, round) pairs with >1 round_end row, by round (top 8):", dupkeys["round"].value_counts().sort("count", descending=True).head(8).rows())

re1 = RE.unique(["match_key", "round"], keep="first")
fe = {(m, r): tk for m, r, tk in FE.select(["match_key", "round", "tick"]).iter_rows()}
sc = collections.defaultdict(dict)
for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    sc[m][r] = (t, c)

def score_winner(scm, r):
    if r not in scm or r + 1 not in scm: return None
    t0, c0 = scm[r]; t1, c1 = scm[r + 1]
    no = (t1 - t0, c1 - c0) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and not sw: return T if (t1 - t0, c1 - c0) == (1, 0) else CT
    if sw and not no: return T if (t1 - c0, c1 - t0) == (0, 1) else CT
    return None

rew = {(m, r): w for m, r, w in re1.select(["match_key", "round", "winner_team_code"]).iter_rows()}
rtk = {(m, r): tk for m, r, tk in re1.select(["match_key", "round", "tick"]).iter_rows()}
# tick position: is round_end[13] before freeze_end[13]?  (before -> it belongs to round 12)
pos = collections.Counter()
copy = collections.Counter()
for m in {k[0] for k in rew if k[1] == 13}:
    if (m, 13) not in rtk or (m, 13) not in fe:
        continue
    before = rtk[(m, 13)] < fe[(m, 13)]
    w13 = rew.get((m, 13)); w12 = rew.get((m, 12))
    sc13 = score_winner(sc.get(m, {}), 13)
    sc12 = score_winner(sc.get(m, {}), 12)
    key = "disagree" if (sc13 is not None and w13 != sc13) else ("agree" if sc13 is not None else "unscorable")
    pos[(key, "rE13_tick_before_freeze13" if before else "rE13_tick_after_freeze13")] += 1
    if key != "unscorable":
        copy[(key, "rE13==rE12" if w13 == w12 else "rE13!=rE12", "rE13==score12" if (sc12 is not None and w13 == sc12) else ("score12_none" if sc12 is None else "rE13!=score12"))] += 1
print("round 13: round_end tick vs freeze_end tick:")
for k in sorted(pos): print("  ", k, pos[k])
print("round 13 round_end winner vs round 12 round_end winner and score-derived round 12 winner:")
for k in sorted(copy): print("  ", k, copy[k])
# same on round 12 and 14 relative placement
tick_rel = collections.Counter()
for (m, r), tk in rtk.items():
    if r in (9, 12, 13, 14, 28) and (m, r) in fe:
        tick_rel[(r, "before" if tk < fe[(m, r)] else "after")] += 1
print("round_end tick before/after own freeze_end, by round:", sorted(tick_rel.items()))
