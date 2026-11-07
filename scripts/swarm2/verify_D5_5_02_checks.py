r"""Verifier for D5-5. Independent recomputation from the 2024 tome (G:) via the
scratch extracts written by verify_D5_5_01_load.py, plus death/spawn channels.

Definitions (ASSUMPTION unless stated):
  window(k)   = [round_freeze_end tick of label k, round_end tick of label k]   (claim's definition)
  step(k)     = clean +1 on exactly one score field between freeze_end(k) and freeze_end(k+1)
                -> 2 (T field rose) / 3 (CT field rose); anything else -> 0
                ASSUMPTION: the step is a valid winner only if no side swap occurs at boundary k->k+1.
  bomb(k)     = 3 if bomb_defused in label k, 2 if bomb_exploded (bomb_state channel); -1 if both
  rw(k)       = round_end winner_team_code, first row for (match, label)
  elimination = distinct player_ids of a team dying inside the window vs distinct players spawned
                for that team in the same round (player_spawn channel).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_5_02_checks.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")


def scan(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


H = pl.read_parquet(SCR / "header.parquet")
RE_raw = pl.read_parquet(SCR / "round_end.parquet")
RS = pl.read_parquet(SCR / "round_state_sub.parquet")
BS = pl.read_parquet(SCR / "bomb_state_sub.parquet")

print("== player_count distribution in round_end:")
print(RE_raw.group_by("player_count").len().sort("player_count"))

FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(
          pl.col("tick").first().alias("fe_tick"), pl.col("t_score").first().alias("t"),
          pl.col("ct_score").first().alias("ct"), pl.len().alias("n_fe")))
RE = (RE_raw.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
          pl.col("tick").first().alias("re_tick"), pl.col("winner_team_code").first().alias("rw"),
          pl.col("win_reason_code").first().alias("reason"), pl.col("player_count").first().alias("pc"),
          pl.len().alias("n_re")))
BOMB_BS = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(
               pl.col("event_type").unique().alias("bevs"), pl.col("tick").min().alias("b_tick")))
BOMB_BS = BOMB_BS.with_columns(
    bomb_bs=pl.when(pl.col("bevs").list.contains("bomb_defused") & pl.col("bevs").list.contains("bomb_exploded"))
    .then(pl.lit(-1))
    .when(pl.col("bevs").list.contains("bomb_defused")).then(pl.lit(3))
    .otherwise(pl.lit(2)))
PLANT_BS = (BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"])
            .agg(pl.len().alias("n_plant"), pl.col("tick").min().alias("plant_tick")))
BOMB_RS = (RS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("revs")))
BOMB_RS = BOMB_RS.with_columns(
    bomb_rs=pl.when(pl.col("revs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)))

SP = (scan("player_spawn").select(["match_key", "round", "player_id", "player_team_code"]).collect()
      .filter(pl.col("player_team_code").is_in([2, 3])))
ROSTER = (SP.group_by(["match_key", "round", "player_team_code"]).agg(
    pl.col("player_id").n_unique().alias("n_spawn")))
DE = (scan("player_death").select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
      .filter(pl.col("player_team_code").is_in([2, 3])))
print("player_death rows (team 2/3):", DE.height, " player_spawn rows (team 2/3):", SP.height)
print("roster size distribution (distinct spawned players per team per round):")
print(ROSTER.group_by("n_spawn").len().sort("n_spawn"))

# ---------- wide per-label table ----------
W = (FE.join(RE, on=["match_key", "round"], how="full", coalesce=True)
     .join(BOMB_BS.select(["match_key", "round", "bomb_bs", "b_tick"]), on=["match_key", "round"], how="left")
     .join(BOMB_RS.select(["match_key", "round", "bomb_rs"]), on=["match_key", "round"], how="left")
     .join(PLANT_BS, on=["match_key", "round"], how="left")
     .rename({"round": "k"}))
NXT = FE.select(["match_key", pl.col("round").alias("k"), pl.col("fe_tick").alias("fe_tick_n"),
                 pl.col("t").alias("t_n"), pl.col("ct").alias("ct_n")]).with_columns(
    k=pl.col("k") - 1)
W = W.join(NXT, on=["match_key", "k"], how="left")
W = W.with_columns(dt=pl.col("t_n") - pl.col("t"), dct=pl.col("ct_n") - pl.col("ct"))
W = W.with_columns(
    step=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
    .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)))
W = W.join(H.select([pl.col("key").alias("match_key"), "map_name", "match_date", "day",
                     "t_starters_score_final", "ct_starters_score_final"]), on="match_key", how="left")
RO = (ROSTER.rename({"round": "k"}).pivot(on="player_team_code", index=["match_key", "k"], values="n_spawn")
      .rename({"2": "nT", "3": "nCT"}))
W = W.join(RO, on=["match_key", "k"], how="left")
W.write_parquet(SCR / "wide.parquet")
print("wide rows", W.height)

Wd = W.filter(pl.col("rw").is_not_null())

# ---------- 1. per-label agreement ----------
print("\n== 1. per label (matches with a round_end row): clean step vs round_end winner vs bomb ==")
rows = []
for lab in range(1, 31):
    sub = Wd.filter(pl.col("k") == lab)
    clean = sub.filter(pl.col("step") != 0)
    bomb = sub.filter(pl.col("bomb_bs").is_not_null())
    bomb_clean = bomb.filter(pl.col("step") != 0)
    rows.append(dict(
        label=lab,
        n_round_end=sub.height,
        step_clean=clean.height,
        step_eq_rw=clean.filter(pl.col("step") == pl.col("rw")).height,
        bomb_rows=bomb.height,
        bomb_eq_rw=bomb.filter(pl.col("bomb_bs") == pl.col("rw")).height,
        bomb_clean_step=bomb_clean.height,
        bomb_eq_step=bomb_clean.filter(pl.col("bomb_bs") == pl.col("step")).height,
    ))
print(pl.DataFrame(rows))

# ---------- 2. label-13 cross-tab ----------
r1_keys = set(Wd.filter(pl.col("k") == 1)["match_key"].to_list())
L13 = Wd.filter((pl.col("k") == 13) & pl.col("fe_tick").is_not_null()).with_columns(
    r1=pl.col("match_key").is_in(list(r1_keys)),
    bomb_status=pl.when(pl.col("bomb_bs").is_null()).then(pl.lit("no_bomb"))
    .when(pl.col("bomb_bs") == pl.col("rw")).then(pl.lit("bomb=rw")).otherwise(pl.lit("bomb!=rw")),
    step_vs_rw=pl.when(pl.col("step") == 0).then(pl.lit("step0"))
    .when(pl.col("step") == pl.col("rw")).then(pl.lit("step=rw")).otherwise(pl.lit("step!=rw")),
)
print("\n== 2. label 13: r1 (round_end label 1 present) x bomb_status x step_vs_rw ==")
print(L13.group_by(["r1", "bomb_status", "step_vs_rw"]).agg(pl.len().alias("matches"))
      .sort(["r1", "bomb_status", "step_vs_rw"]))
L13.write_parquet(SCR / "label13.parquet")
