r"""D4 diagnostic: which simple winner-to-team mapping reproduces header totals for unaffected 2024 regulation matches?

Complete matches with n <= 24 and no round-13 label mismatch. Schedules tested for team A (header t_starters):
  NONE   no side swap (team A stays on T for all rounds; winners compared directly)
  S12    swap after round 12 (MR12 halftime)
  S15    swap after round 15 (MR15 halftime)
Also reports how often the T-coded winner count (winner_team_code == 2) equals the header t_starters total.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_header_unaff.py
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
SCHED = {"NONE": None, "S12": 12, "S15": 15}
agg = {s: {"ok": 0, "swapped": 0, "bad": 0} for s in SCHED}
t2 = {"n2_eq_tA": 0, "n2_eq_tB": 0, "neither": 0}
ex = []
for m, rounds in by_match.items():
    h = Hd.get(m)
    if h is None or m in mm13:
        continue
    tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(rounds)
    if not (n == tA + tB and n <= 24 and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    n2 = sum(1 for x in rounds if int(x["winner_team_code"]) == T)
    t2["n2_eq_tA" if n2 == tA else ("n2_eq_tB" if n2 == tB else "neither")] += 1
    for name, sw in SCHED.items():
        wa = wb = 0
        for x in rounds:
            side_A = T if (sw is None or int(x["round"]) <= sw) else CT
            if int(x["winner_team_code"]) == side_A:
                wa += 1
            else:
                wb += 1
        st = "ok" if (wa, wb) == (tA, tB) else ("swapped" if (wa, wb) == (tB, tA) else "bad")
        agg[name][st] += 1
    if len(ex) < 3:
        ex.append((m[:30], tA, tB, "".join("T" if int(x["winner_team_code"]) == T else "C" for x in rounds)))
print("unaffected complete regulation matches (n <= 24):")
for name in SCHED:
    v = agg[name]
    tot = sum(v.values())
    print(f"  {name:<5} ok={v['ok']:>5} swapped={v['swapped']:>4} bad={v['bad']:>5} n={tot} ok_rate={v['ok'] / tot:.3f}")
print("winner-2 count vs header t_starters:", t2)
print("examples (match, tA, tB, winner string T/C by round):")
for e in ex:
    print("  ", e)
