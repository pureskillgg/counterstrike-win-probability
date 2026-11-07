"""D3-5 verifier, step O: the 6 sample rows (2026 1 Hz sample) where the label and the naive freeze delta disagree.
Checks whether each is a swap transition under the bake-off rule (swap_expected(r, max_rounds)). Read only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T, CT = 2, 3


def rd(ch, cols):
    folders = [str(TOME / "header")] if ch == "header" else sorted(glob.glob(str(TOME / f"basic_{ch}.*")))
    files = []
    for fd in folders:
        files += sorted(glob.glob(str(Path(fd) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


S = pl.read_parquet(str(ROOT / "data" / "state_sample" / "rounds.parquet"), columns=["match_key", "round", "ct_win"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
F = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
H = {k: mr for k, mr in rd("header", ["key", "max_rounds"]).iter_rows()}
keys = set(S.select(["match_key", "round"]).iter_rows())
bad = []
for m, r, cw in S.iter_rows():
    if (m, r) in F and (m, r + 1) in F:
        t1, c1 = F[(m, r)]
        t2, c2 = F[(m, r + 1)]
        dd = (t2 - t1, c2 - c1)
        if dd in [(1, 0), (0, 1)]:
            D = 1 if dd == (0, 1) else 0
            if D != cw:
                bad.append((m, r, cw, dd, H.get(m), swap_expected(r, H.get(m, 24))))
print("sample rows where label != naive delta:", len(bad))
for b in bad:
    print("  ", b[0][-45:], "round", b[1], "label ct_win", b[2], "delta", b[3], "max_rounds", b[4],
          "swap_transition_under_bakeoff_rule", b[5])
print("all of them swap transitions:", all(b[5] for b in bad))
print("== end ==")
