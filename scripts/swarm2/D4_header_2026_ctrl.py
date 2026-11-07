r"""D4 control: the same S12 (MR12 halftime) header test on 2026 regulation matches, to confirm the test code
reconciles where the header is known to agree (bakeoff rule). Complete matches with n <= 24, all rounds, round_end winners.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_header_2026_ctrl.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final", "max_rounds"])
Hd = {r["key"]: r for r in H.iter_rows(named=True)}
RE = read_dir(sorted(BASE.glob("basic_round_end.*"))[0], ["match_key", "round", "winner_team_code"]).unique(
    ["match_key", "round"], keep="first")
by = {}
for r in RE.sort(["match_key", "round"]).iter_rows(named=True):
    by.setdefault(r["match_key"], []).append(r)
cnt = {"ok": 0, "swapped": 0, "bad": 0}
cnt_all = {"ok": 0, "swapped": 0, "bad": 0}
for m, rounds in by.items():
    h = Hd.get(m)
    if h is None:
        continue
    tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(rounds)
    if n != tA + tB or [int(x["round"]) for x in rounds] != list(range(1, n + 1)):
        continue
    mr = int(h["max_rounds"])
    for dst, cond in ((cnt, n <= 24), (cnt_all, True)):
        if not cond:
            continue
        wa = wb = 0
        for x in rounds:
            side_A = T if int(x["round"]) <= mr // 2 else CT
            if int(x["winner_team_code"]) == side_A:
                wa += 1
            else:
                wb += 1
        st = "ok" if (wa, wb) == (tA, tB) else ("swapped" if (wa, wb) == (tB, tA) else "bad")
        dst[st] += 1
print("2026 complete matches with S12 (halftime at max_rounds//2, no OT rule):")
print("  n<=24 (regulation only):", cnt)
print("  all complete:", cnt_all)
