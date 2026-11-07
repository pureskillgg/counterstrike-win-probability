r"""D4 diagnostic: for unaffected 2024 regulation misfits that round 13 alone would fix, what are the round-13 rows like?

For each such match (misfit with S12 and round_end winners; flipping round 13 reconciles the header):
  win_reason_code and winner of round 13, whether round 13 has a bomb outcome row, and whether the last death
  in the live window is within 3 s of round_end (elimination-like) or not (timer-like).
Compared with all unaffected regulation round-13 rows. Affected matches are excluded.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_r13_fixers.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
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
re_ = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"]).unique(
    ["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
outs = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
R = re_.join(outs.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")
mm13 = set(R.filter((pl.col("round") == 13) & pl.col("bside").is_not_null())
           .filter(pl.col("bside") != pl.col("winner_team_code"))["match_key"].to_list())
live = (read("round_state", ["match_key", "round", "tick", "event_type"]).filter(pl.col("event_type") == "round_freeze_end")
        .select(["match_key", "round", pl.col("tick").alias("live")]).unique(["match_key", "round"], keep="first"))
de = read("player_death", ["match_key", "round", "tick", "player_team_code"])
by = {}
for r in R.sort(["match_key", "round"]).iter_rows(named=True):
    by.setdefault(r["match_key"], []).append(r)
fixer_matches = []
for m, rounds in by.items():
    h = Hd.get(m)
    if h is None or m in mm13:
        continue
    tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(rounds)
    if not (n == tA + tB and n <= 24 and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    wa = sum(1 for x in rounds if int(x["winner_team_code"]) == (T if int(x["round"]) <= 12 else CT))
    if (wa, n - wa) == (tA, tB):
        continue
    r13 = [x for x in rounds if int(x["round"]) == 13][0]
    side_A13 = CT
    isA = int(r13["winner_team_code"]) == side_A13
    new_wa = wa - 1 if isA else wa + 1
    if (new_wa, n - new_wa) == (tA, tB):
        fixer_matches.append((m, r13))
print("unaffected regulation misfits fixed by flipping round 13:", len(fixer_matches))
fm = pl.DataFrame({"match_key": [m for m, _ in fixer_matches],
                   "winner_team_code": [int(r["winner_team_code"]) for _, r in fixer_matches],
                   "win_reason_code": [int(r["win_reason_code"]) for _, r in fixer_matches],
                   "round": [13] * len(fixer_matches)})
fm = fm.join(re_.select(["match_key", "round", "end", "player_count"]) if "end" in re_.columns else
             re_.select(["match_key", "round", "tick", "player_count"]).rename({"tick": "end"}),
             on=["match_key", "round"], how="left")
fm = fm.join(live, on=["match_key", "round"], how="left")
bomb13 = set(outs.filter(pl.col("round") == 13)["match_key"].to_list())
fm = fm.with_columns(pl.col("match_key").is_in(list(bomb13)).alias("r13_has_bomb"))
print("win_reason_code of flipped round 13 (count):")
print(fm.group_by(["win_reason_code", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))
print("round 13 has a bomb outcome row:", fm.group_by("r13_has_bomb").len().rows())
last = (de.join(fm.select(["match_key", "round", "live", "end"]), on=["match_key", "round"], how="inner")
        .filter((pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
        .group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("last_death"),
                                              pl.col("end").first(), pl.col("live").first()))
last = last.with_columns(((pl.col("end") - pl.col("last_death")) <= 192).alias("death_at_end"))
print("last death within 3 s of round_end (elimination-like):")
print(last.group_by("death_at_end").len())
print("all unaffected regulation round-13 rows, win_reason_code x winner (for comparison):")
ref = (re_.filter(pl.col("round") == 13).join(
    pl.DataFrame({"match_key": list(set(by.keys()) - mm13)}), on="match_key", how="inner")
       .filter(pl.col("player_count") == 20))
print(ref.group_by(["win_reason_code", "winner_team_code"]).len().sort(["win_reason_code", "winner_team_code"]))
