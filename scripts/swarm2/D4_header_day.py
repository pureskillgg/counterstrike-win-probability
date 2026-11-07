"""D4 diagnostic: header reconciliation by match day for 2024 (end-source winners), S_bake vs S_mr15 schedules.

Groups complete matches (round count == header total) by day (01-07 vs 12-25) and by affected flag
(>=1 label-based round-13 mismatch). Reports OK rate under MR12 halftime at 12 (plus bakeoff overtime rule)
and under MR15 halftime at 15. A3 caveat: 2024 header has no max_rounds, so the format is inferred here, not read.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_header_day.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final"])
Hd = {r["key"]: r for r in H.iter_rows(named=True)}
re_ = read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
outs = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
R = re_.join(outs.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")
mm13 = set(R.filter((pl.col("round") == 13) & pl.col("bside").is_not_null())
           .filter(pl.col("bside") != pl.col("winner_team_code"))["match_key"].to_list())
by_match = {}
for r in R.sort(["match_key", "round"]).iter_rows(named=True):
    by_match.setdefault(r["match_key"], []).append(r)

SCHED = {"S_bake": lambda r: r == 12 or (r > 24 and (r - 27) % 6 == 0),
         "S_mr15": lambda r: r == 15}
agg = {}
for m, rounds in by_match.items():
    h = Hd.get(m)
    if h is None:
        continue
    tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(rounds)
    if not (n == tA + tB and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    day = m.split("/")[3]
    dgrp = "d01-07" if int(day) <= 7 else "d12-25"
    grp = "affected" if m in mm13 else "unaffected"
    for name, sw in SCHED.items():
        wa = wb = swaps = 0
        for x in rounds:
            side_A = T if swaps % 2 == 0 else CT
            if int(x["winner_team_code"]) == side_A:
                wa += 1
            else:
                wb += 1
            if sw(int(x["round"])):
                swaps += 1
        st = "ok" if (wa, wb) == (tA, tB) else ("swapped" if (wa, wb) == (tB, tA) else "bad")
        k = (dgrp, grp, name)
        agg.setdefault(k, {"ok": 0, "swapped": 0, "bad": 0, "maxr": {}})
        agg[k][st] += 1
        agg[k]["maxr"][n] = agg[k]["maxr"].get(n, 0) + 1
print("complete matches (end-source winners): day group x affected x schedule")
for k in sorted(agg):
    v = agg[k]
    tot = v["ok"] + v["swapped"] + v["bad"]
    print(f"  {k[0]:<8} {k[1]:<11} {k[2]:<7} n={tot:>5} ok={v['ok']:>5} swapped={v['swapped']:>4} "
          f"bad={v['bad']:>5} ok_rate={v['ok'] / tot:.3f}")
