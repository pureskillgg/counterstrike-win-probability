"""D3 secondary: side-swap signature per match from round_state score deltas (2024 G: tome).

A side swap between freeze(r) and freeze(r+1) shows as a delta that is not a single-side +1 and whose
two components have opposite signs (or both nonzero). Swap rounds are r where the delta is unclear.
For each day 12..25 match with contiguous round numbers, list the swap rounds, then check whether the
round_end winner at r+1 disagrees with the engine delta at r+1 (side-based, both freeze rows after the swap).
Also tabulate final round count per signature. ASSUMPTION: an unclear delta with opposite-sign components
marks a side swap; a round with no delta row is not used.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D3_swap_signature.py
"""
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
by_m = collections.defaultdict(list)
for (m, r) in live:
    by_m[m].append(r)
sig_count = collections.Counter()
sig_flip = collections.Counter()   # (sig) -> (flip_rounds_at_swap_plus1, swaps_checked)
rounds_n = {}
for m, rs in by_m.items():
    day = int(m.split("/")[3])
    if day < 12:
        continue
    rs = sorted(rs)
    n_rounds = len(set(r for (mm, r) in win if mm == m)) if False else None
    swaps = []
    for r in rs:
        if (m, r + 1) not in live:
            continue
        t1, c1 = live[(m, r)]; t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if (dc == 1 and dt == 0) or (dt == 1 and dc == 0):
            continue
        if (dt == 0 and dc == 0):
            continue
        if dt * dc <= 0 and (dt, dc) != (0, 0):
            swaps.append(r)
    sig = tuple(swaps)
    sig_count[sig] += 1
    # flip check on first round after each swap, only when the next round's delta is clean
    for s in swaps:
        if (m, s + 1) in win and (m, s + 1) in live and (m, s + 2) in live:
            t1, c1 = live[(m, s + 1)]; t2, c2 = live[(m, s + 2)]
            dt, dc = t2 - t1, c2 - c1
            if (dc == 1 and dt == 0) or (dt == 1 and dc == 0):
                ds = 3 if dc == 1 else 2
                key = (s, "day12_25")
                sig_flip[(s, ds != win[(m, s + 1)])] += 1
print("swap signature (swap rounds) : matches (day 12..25, matches with a live row):")
for sig, n in sig_count.most_common(15):
    print(f"  {str(sig):<22} {n}")
print("\nround_end flip check on the first round after each detected swap (swap round s, flip at s+1):")
print("  (s, flipped?) -> count:", sorted(sig_flip.items()))
