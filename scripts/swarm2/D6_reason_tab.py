"""D6 probe: round-13 win_reason_code vs round_end winner vs score-derived winner vs bomb outcome (2024)."""
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
RE = read("round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "player_count"]).unique(["match_key","round"], keep="first")
RS = read("round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
FE = RS.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
BS = read("bomb_state", ["match_key", "round", "event_type"])
bo = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
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
bomb = {}
for m, r, evs in bo.iter_rows():
    if r == 13:
        bomb[m] = "defused" if "bomb_defused" in evs else "exploded"
tab = collections.Counter()
for m, r, w, rc in RE.filter(pl.col("round") == 13).select(["match_key", "round", "winner_team_code", "win_reason_code"]).iter_rows():
    if m not in bomb:
        continue
    s = score_winner(sc.get(m, {}), 13)
    tab[(f"reason={rc}", f"winner={w}", f"score={s}", f"bomb={bomb[m]}")] += 1
for k in sorted(tab, key=lambda x: (x[0], x[1], x[2], x[3])):
    print(f"  {k[0]:<11} {k[1]:<10} {k[2]:<10} {k[3]:<14} n={tab[k]}")
