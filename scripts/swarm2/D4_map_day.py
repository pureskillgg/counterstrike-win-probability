r"""D4 diagnostic: do the round-13 defect matches cluster by map or by day (2024 tome)?

Per complete regulation match (n <= 24, round count == header total): affected (label-based r13 bomb mismatch),
r13_flip_misfit (S12 header misfit that a round-13 flip reconciles, no bomb row in r13), and other.
Tabulated by map_name (header) and by day.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_map_day.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final", "map_name"])
Hd = {r["key"]: r for r in H.iter_rows(named=True)}
re_ = read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
outs = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
R = re_.join(outs.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")
mm13 = set(R.filter((pl.col("round") == 13) & pl.col("bside").is_not_null())
           .filter(pl.col("bside") != pl.col("winner_team_code"))["match_key"].to_list())
by = {}
for r in R.sort(["match_key", "round"]).iter_rows(named=True):
    by.setdefault(r["match_key"], []).append(r)
rows = []
for m, rounds in by.items():
    h = Hd.get(m)
    if h is None:
        continue
    tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(rounds)
    if not (n == tA + tB and n <= 24 and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    wa = sum(1 for x in rounds if int(x["winner_team_code"]) == (T if int(x["round"]) <= 12 else CT))
    misfit = (wa, n - wa) != (tA, tB)
    r13 = next((x for x in rounds if int(x["round"]) == 13), None)
    flip_fix = False
    if misfit and r13 is not None:
        nwa = wa - 1 if int(r13["winner_team_code"]) == CT else wa + 1
        flip_fix = (nwa, n - nwa) == (tA, tB)
    cls = "affected" if m in mm13 else ("r13_flip_misfit" if flip_fix else ("other_misfit" if misfit else "fits"))
    rows.append({"match_key": m, "map": h["map_name"], "day": m.split("/")[3], "cls": cls})
D = pl.DataFrame(rows)
print(f"complete regulation matches: {D.height}")
print(D.group_by("cls").len().sort("cls"))
print("\nby map (counts per class):")
print(D.group_by(["map", "cls"]).len().pivot(on="cls", index="map", values="len").fill_null(0).sort("map"))
print("\nby day group (counts per class):")
print(D.with_columns(pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dg"))
      .group_by(["dg", "cls"]).len().pivot(on="cls", index="dg", values="len").fill_null(0).sort("dg"))
