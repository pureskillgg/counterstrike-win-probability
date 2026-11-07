"""verify D5-9, step 1: independent per-(match, round) table from the 2024 tome, built from raw channels.

Does not reuse D5 intermediates. Projects columns; one channel at a time; lazy scans where possible.
Output (scratchpad, C:, not the project):
  verify_D5_9/rounds.parquet   one row per (match_key, round): round_end, freeze-end scores (side and raw),
                               phase-halftime row count, bomb outcome, death counts, last-death victim team
  verify_D5_9/matches.parquet  one row per header match: map, match_type, regime marker, last-round-half labels,
                               match-point labels, max round_end label, n round_end rows
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_01_build.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_9")
SCR.mkdir(parents=True, exist_ok=True)


def scan(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    if not fs:
        raise FileNotFoundError(ch)
    return pl.concat([pl.scan_parquet(f).select(cols) for f in fs], how="vertical_relaxed")


# ---------- header ----------
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))[0]
H = pl.read_parquet(hf, columns=["key", "match_id", "map_name", "match_date", "match_type", "tick_rate"]).rename(
    {"key": "match_key"}).with_columns(day=pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1))
print("header rows:", H.height, "unique match_key:", H["match_key"].n_unique())

# ---------- round_end (first row per match,round; keep row count) ----------
RE_raw = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"]).collect()
RE = (RE_raw.sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(
          pl.col("tick").first().alias("end_tick"),
          pl.col("winner_team_code").first().alias("end_winner"),
          pl.col("win_reason_code").first().alias("end_reason"),
          pl.col("player_count").first().alias("end_player_count"),
          pl.len().alias("end_nrows"),
          pl.col("winner_team_code").n_unique().alias("end_nwinner_distinct")))
print("round_end (match,round) pairs:", RE.height, " rows:", RE_raw.height)
print("round_end player_count distribution (rows):")
print(RE_raw.group_by("player_count").len().sort("player_count"))

# ---------- round_state ----------
RS = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                          "t_score_raw", "ct_score_raw", "phase"]).collect()
print("round_state rows:", RS.height)
print("round_state phase distribution:")
print(RS.group_by("phase").len().sort("phase"))

FE_raw = RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
FE = FE_raw.group_by(["match_key", "round"]).agg(
    pl.col("tick").first().alias("fe_tick"),
    pl.col("t_score").first().alias("fe_t"),
    pl.col("ct_score").first().alias("fe_ct"),
    pl.col("t_score_raw").first().alias("fe_t_raw"),
    pl.col("ct_score_raw").first().alias("fe_ct_raw"),
    pl.col("phase").first().alias("fe_phase"),
    pl.len().alias("fe_nrows"),
    (pl.col("t_score") != pl.col("t_score").last()).sum().alias("fe_t_changes"),
)
print("freeze_end (match,round) pairs:", FE.height, " rows:", FE_raw.height)
print("freeze_end nrows distribution:")
print(FE.group_by("fe_nrows").len().sort("fe_nrows"))

HALF = (RS.filter(pl.col("phase") == "halftime")
        .group_by(["match_key", "round"]).agg(pl.len().alias("n_halftime_phase_rows"),
                                              pl.col("tick").min().alias("halftime_tick_min")))

# announcement events: per match, the round label at which each occurs
ANN = (RS.filter(pl.col("event_type").is_in(
    ["round_announce_match_start", "round_announce_last_round_half", "round_announce_match_point",
     "round_announce_final", "begin_new_match", "cs_win_panel_match"]))
       .group_by(["match_key", "event_type"]).agg(
           pl.col("round").sort().alias("labels"), pl.len().alias("n")))
ANN_wide = ANN.pivot(on="event_type", index="match_key", values="labels", aggregate_function="first")
ANN_n = ANN.pivot(on="event_type", index="match_key", values="n", aggregate_function="first")

# ---------- bomb_state ----------
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
BO = BS.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
print("bomb rows by event:", BO.group_by("event_type").len().sort("event_type").to_dicts())
BOMB = (BO.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("event_type").filter(pl.col("event_type") == "bomb_defused").len().alias("n_defused"),
    pl.col("event_type").filter(pl.col("event_type") == "bomb_exploded").len().alias("n_exploded"),
    pl.col("event_type").filter(pl.col("event_type") == "bomb_planted").len().alias("n_planted"),
    pl.col("player_team_code").filter(pl.col("event_type") == "bomb_defused").first().alias("defuser_team"),
    pl.col("player_team_code").filter(pl.col("event_type") == "bomb_exploded").first().alias("exploder_team"),
))

# ---------- player_death: counts and last victim (max tick) ----------
DE = scan("player_death", ["match_key", "round", "tick", "player_team_code", "attacker_team_code"]).collect()
print("player_death rows:", DE.height)
DE_s = DE.sort(["match_key", "round", "tick"])
DEATH = DE_s.group_by(["match_key", "round"]).agg(
    pl.len().alias("n_deaths"),
    pl.col("tick").max().alias("last_death_tick"),
    pl.col("player_team_code").last().alias("last_victim_team"),
    pl.col("attacker_team_code").last().alias("last_attacker_team"),
)

# ---------- union of (match, round) keys from round_end and freeze_end ----------
KEYS = pl.concat([RE.select(["match_key", "round"]), FE.select(["match_key", "round"])]).unique()
R = (KEYS.join(RE, on=["match_key", "round"], how="left")
     .join(FE, on=["match_key", "round"], how="left")
     .join(HALF, on=["match_key", "round"], how="left")
     .join(BOMB, on=["match_key", "round"], how="left")
     .join(DEATH, on=["match_key", "round"], how="left")
     .sort(["match_key", "round"]))
R = R.with_columns(
    pl.col("n_halftime_phase_rows").fill_null(0),
    pl.col("n_defused").fill_null(0), pl.col("n_exploded").fill_null(0), pl.col("n_planted").fill_null(0),
    pl.col("n_deaths").fill_null(0))
R.write_parquet(SCR / "rounds.parquet")
print("rounds table rows:", R.height)

# ---------- match table ----------
MX = RE.group_by("match_key").agg(pl.col("round").max().alias("max_end_label"),
                                  pl.col("round").min().alias("min_end_label"),
                                  pl.len().alias("n_end_labels"),
                                  pl.col("end_player_count").mode().first().alias("mode_player_count"))
M = (H.join(MX, on="match_key", how="left")
     .join(ANN_wide, on="match_key", how="left")
     .join(ANN_n.select(["match_key"] + [c for c in ANN_n.columns if c != "match_key"]).rename(
         {c: f"n_{c}" for c in ANN_n.columns if c != "match_key"}), on="match_key", how="left"))
M = M.with_columns(regime=pl.when(pl.col("n_round_announce_match_start").fill_null(0) > 0)
                   .then(pl.lit("A")).otherwise(pl.lit("B")))
M.write_parquet(SCR / "matches.parquet")
print("matches rows:", M.height, " regime A:", int((M["regime"] == "A").sum()),
      " regime B:", int((M["regime"] == "B").sum()))
print("matches with no round_end rows:", M.filter(pl.col("max_end_label").is_null()).height)
print("written:", SCR / "rounds.parquet", SCR / "matches.parquet")
