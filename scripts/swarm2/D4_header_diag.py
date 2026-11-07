"""D4 diagnostic for D4_header_check.py: which side-swap schedule reproduces the 2024 header totals?

Candidate schedules (set of rounds r after which sides swap, i.e. swap before r+1):
  S_bake   {12} + {27, 33, 39, ...}           bakeoff.swap_expected with max_rounds 24 (the rule used in D4_header_check)
  S_ot24   {12} + {24, 27, 30, ...}           swap at regulation end and every 3 OT rounds
  S_ot24b  {12} + {24, 30, 36, ...}           swap at regulation end and at each OT period start only
  S_none   {12}                               halftime only
  S_mr15   {15} + {30 ... no OT}              MR15 (max 30 regulation rounds), halftime at 15
For each complete match (round count == header total, rounds 1..n contiguous), the win counts for team A
(header t_starters, starts on T) are recomputed from the round_end winner codes and compared with the header.
Prints OK / SWAPPED / BAD rates for affected (>=1 round-13 label mismatch) and unaffected matches, per schedule.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D4_header_diag.py
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
re_ = read("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    ["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
outs = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
R = re_.join(outs.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")
mm13 = (R.filter((pl.col("round") == 13) & pl.col("bside").is_not_null())
        .filter(pl.col("bside") != pl.col("winner_team_code")).select("match_key").unique())
affected = set(mm13["match_key"].to_list())

SCHED = {
    "S_bake": lambda r: r == 12 or (r > 24 and (r - 24 - 3) % 6 == 0),
    "S_ot24": lambda r: r == 12 or (r >= 24 and r % 3 == 0),
    "S_ot24b": lambda r: r == 12 or (r >= 24 and (r - 24) % 6 == 0),
    "S_none": lambda r: r == 12,
    "S_mr15": lambda r: r == 15,
}

by_match = {}
for r in R.sort(["match_key", "round"]).iter_rows(named=True):
    by_match.setdefault(r["match_key"], []).append(r)

stats = {}
for m, rounds in by_match.items():
    h = Hd.get(m)
    if h is None:
        continue
    tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(rounds)
    if not (n == tA + tB and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    grp = "affected" if m in affected else "unaffected"
    for name, sw in SCHED.items():
        for src in ("end", "bomb"):
            wa = wb = 0
            swaps = 0
            for x in rounds:
                r = int(x["round"])
                w = int(x["winner_team_code"])
                if src == "bomb" and x["bside"] is not None:
                    w = int(x["bside"])
                side_A = T if swaps % 2 == 0 else CT
                if w == side_A:
                    wa += 1
                else:
                    wb += 1
                if sw(r):
                    swaps += 1
            st = "ok" if (wa, wb) == (tA, tB) else ("swapped" if (wa, wb) == (tB, tA) else "bad")
            key = (grp, name, src, st)
            stats[key] = stats.get(key, 0) + 1

print("complete matches by group x schedule x source x status (2024 tome):")
for grp in ["unaffected", "affected"]:
    tot_u = sum(v for k, v in stats.items() if k[0] == grp and k[2] == "end" and k[1] == "S_bake")
    print(f"\n[{grp}] complete matches: {tot_u}")
    for name in SCHED:
        for src in ("end", "bomb"):
            row = {st: stats.get((grp, name, src, st), 0) for st in ("ok", "swapped", "bad")}
            tot = sum(row.values())
            print(f"  {name:<8} src={src:<4} ok={row['ok']:>5} swapped={row['swapped']:>5} bad={row['bad']:>5} "
                  f"ok_rate={row['ok'] / tot if tot else float('nan'):.3f}")
