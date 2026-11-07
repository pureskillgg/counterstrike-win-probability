"""D6 test: compare the cross_period.py inferred 2024 max_rounds with the team-code swap format.
Inferred (cross_period.py lines 283-290): 2 x (min round with round_announce_last_round_half) + 2; default 24.
Team-code format: 2 x first player_info swap round s (from logs/swarm2/D6_per_match.log); for matches with
no player_info swap, the first unambiguous score-field swap (same convention).
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/D6_inferred_maxrounds.py
"""
import sys, glob, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
HALF_EVENT = "round_announce_last_round_half"

fd = sorted(BASE.glob("basic_round_state.*"))[0]
rs = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "event_type", "t_score", "ct_score"])
                for f in sorted(glob.glob(str(fd / "dataframe_*")))], how="vertical_relaxed")
half = (rs.filter(pl.col("event_type") == HALF_EVENT).group_by("match_key")
        .agg(pl.col("round").min().alias("r")))
half = {m: int(r) for m, r in half.iter_rows()}
fr = rs.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
sc = collections.defaultdict(dict)
for m, r, t, c in fr.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    sc[m][r] = (t, c)

def first_score_swap(scm):
    for r in sorted(scm):
        if r + 1 not in scm:
            continue
        t0, c0 = scm[r]; t1, c1 = scm[r + 1]
        no = (t1 - t0, c1 - c0) in [(1, 0), (0, 1)]
        sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
        if sw and not no:
            return r
    return None

rows = []
with open(ROOT / "logs" / "swarm2" / "D6_per_match.log", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        m, date, bounds, grp = p[0], p[1], p[3], p[7]
        s_pi = int(bounds.split(",")[0]) if bounds else None
        s_sc = first_score_swap(sc.get(m, {})) if s_pi is None else None
        s = s_pi if s_pi is not None else s_sc
        src = "player_info" if s_pi is not None else ("score" if s_sc is not None else "none")
        h = half.get(m)
        inferred = h * 2 + 2 if h is not None else 24
        team = 2 * s if s is not None else None
        rows.append((m, date[:10], grp, inferred, "event" if h is not None else "default", team, src))

tot = collections.Counter()
for m, d, grp, inf, isrc, team, src in rows:
    if team is None:
        tot[("undetermined", inf)] += 1
    else:
        tot[("agree" if inf == team else "disagree", inf, team)] += 1
print(f"matches: {len(rows)}")
print("inferred max_rounds source counts:", dict(collections.Counter(r[4] for r in rows)))
print("inferred max_rounds distribution:", dict(collections.Counter(r[3] for r in rows)))
print("team-code format distribution:", dict(collections.Counter(r[5] for r in rows)),
      "| source:", dict(collections.Counter(r[6] for r in rows)))
print("cross-tab (inferred max_rounds, team-code format):")
for k in sorted(tot, key=lambda x: str(x)):
    print(f"  {str(k):<40} n={tot[k]}")
agree = sum(v for k, v in tot.items() if k[0] == "agree")
dis = sum(v for k, v in tot.items() if k[0] == "disagree")
und = sum(v for k, v in tot.items() if k[0] == "undetermined")
print(f"agree={agree} disagree={dis} undetermined={und} agree_rate_on_determined={agree / (agree + dis):.4f}")
aff = [r for r in rows if r[2] == "r13_mismatch"]
print(f"affected (r13 mismatch) matches: n={len(aff)}; inferred max_rounds: "
      f"{dict(collections.Counter(r[3] for r in aff))}; team-code format: {dict(collections.Counter(r[5] for r in aff))}")
