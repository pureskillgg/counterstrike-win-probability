r"""D4 diagnostic: class (affected / r13_flip_misfit / other_misfit / fits) x round-13 bomb presence, 2024 tome.

Complete regulation matches only (n <= 24, round count == header total). r13_bomb = the match has a bomb outcome row
in round 13. Shows whether the header-flip classes and the bomb-visible mismatches are the same population.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_class_bomb_xtab.py
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


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final"])
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
    r13_bomb = r13 is not None and r13["bside"] is not None
    r13_bomb_mm = r13_bomb and int(r13["bside"]) != int(r13["winner_team_code"])
    cls = "affected" if m in mm13 else ("r13_flip_misfit" if flip_fix else ("other_misfit" if misfit else "fits"))
    rows.append({"cls": cls, "r13_bomb": bool(r13_bomb), "r13_bomb_mismatch": bool(r13_bomb_mm)})
D = pl.DataFrame(rows)
print(D.group_by(["cls", "r13_bomb", "r13_bomb_mismatch"]).len().sort(["cls", "r13_bomb", "r13_bomb_mismatch"]))
