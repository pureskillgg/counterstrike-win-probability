"""D6 probe: when is a score transition ambiguous? Check label_pair vs score gap before the round (2024)."""
import sys, glob, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
fd = sorted(BASE.glob("basic_round_state.*"))[0]
rs = pl.concat([pl.read_parquet(f, columns=["match_key","round","event_type","t_score","ct_score"]) for f in sorted(glob.glob(str(fd/"dataframe_*")))], how="vertical_relaxed")
rs = rs.filter(pl.col("event_type")=="round_freeze_end").unique(["match_key","round"], keep="first")
sc = collections.defaultdict(dict)
for m, r, t, c in rs.select(["match_key","round","t_score","ct_score"]).iter_rows():
    sc[m][r] = (t, c)
def label_pair(scm, r):
    if r+1 not in scm or r not in scm: return None
    t0,c0 = scm[r]; t1,c1 = scm[r+1]
    no = (t1-t0, c1-c0) in [(1,0),(0,1)]
    sw = (t1-c0, c1-t0) in [(1,0),(0,1)]
    if no and sw: return "both"
    return "no" if no else ("swap" if sw else "none")
tab = collections.Counter()
for m, scm in sc.items():
    for r in scm:
        lab = label_pair(scm, r)
        if lab is None: continue
        gap = abs(scm[r][0]-scm[r][1])
        tab[(min(gap,4), lab)] += 1
for k in sorted(tab):
    print(f"  |gap_before|={k[0]} label={k[1]:<5} n={tab[k]}")
# check: 'none' labels (data inconsistency with any swap/no-swap)
print("  none-label share:", sum(v for k,v in tab.items() if k[1]=="none"), "of", sum(tab.values()))
