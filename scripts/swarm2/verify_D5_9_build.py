"""verify D5-9 build: independent per-round tables from the 2024 tome (projected reads).

Reads only G:\\pureskill\\csds-cs2\\tomes\\2024-02-g\\tome\\csds (header + 4 channels).
Writes intermediate parquet only to the scratchpad (C:), never to the project tome, never under E:.
Each channel is read with column projection; no channel is loaded twice.

Outputs (scratchpad VD59\\):
  header.parquet       one row per match
  fe.parquet           freeze-end row per (match, round): t_score, ct_score, tick, n rows
  end.parquet          round_end rows per (match, round): winner, reason, player_count, tick, n rows
  marks.parquet        match-level round_state markers (announce / match-point / final / win panels)
  bomb.parquet         bomb_planted / bomb_defused / bomb_exploded rows
  deaths.parquet       player_death rows: match_key, round, tick, victim team, attacker team

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_build.py
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
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\VD59")
SCR.mkdir(parents=True, exist_ok=True)


def scan(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    if not fs:
        raise FileNotFoundError(ch)
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


# ---------- header ----------
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
H = pl.read_parquet(hf[0], columns=["key", "match_id", "map_name", "match_date", "match_type",
                                    "tick_rate", "t_starters_score_final", "ct_starters_score_final"])
H = H.rename({"key": "match_key"}).with_columns(day=pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1))
H.write_parquet(SCR / "header.parquet")
print("header rows:", H.height, "unique keys:", H["match_key"].n_unique())
print("match_type:", H.group_by("match_type").len().sort("len", descending=True).to_dicts())

# ---------- round_state ----------
RS = scan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"]).collect()
print("round_state rows:", RS.height)

FE_EV = "round_freeze_end"
fe_all = RS.filter(pl.col("event_type") == FE_EV).sort(["match_key", "round", "tick"])
FE = fe_all.group_by(["match_key", "round"], maintain_order=False).agg(
    pl.col("t_score").first().alias("fe_t"),
    pl.col("ct_score").first().alias("fe_ct"),
    pl.col("tick").first().alias("fe_tick"),
    pl.col("t_score").n_unique().alias("fe_distinct_t"),
    pl.len().alias("fe_n"),
)
FE.write_parquet(SCR / "fe.parquet")
print("freeze-end (match,round) pairs:", FE.height, " pairs with >1 fe row:", int((FE["fe_n"] > 1).sum()))

MARK_EV = ["round_announce_match_start", "round_announce_last_round_half", "round_announce_match_point",
           "round_announce_final", "cs_win_panel_match", "cs_win_panel_round", "begin_new_match"]
MK = (RS.filter(pl.col("event_type").is_in(MARK_EV))
      .group_by(["match_key", "event_type"]).agg(
          pl.len().alias("n"),
          pl.col("round").min().alias("min_round"),
          pl.col("round").max().alias("max_round"),
          pl.col("round").sort().alias("rounds")))
MK.write_parquet(SCR / "marks.parquet")
print("marks rows:", MK.height)

# ---------- round_end ----------
RE = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                               "player_count", "legacy_code"]).collect()
print("round_end rows:", RE.height)
RE.write_parquet(SCR / "end_raw.parquet")

# ---------- bomb_state (bomb events only) ----------
BS = scan("bomb_state").select(["match_key", "round", "tick", "event_type", "player_team_code"]).filter(
    pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).collect()
BS.write_parquet(SCR / "bomb.parquet")
print("bomb rows:", BS.height, BS.group_by("event_type").len().sort("event_type").to_dicts())

# ---------- player_death (projected) ----------
DE = scan("player_death").select(["match_key", "round", "tick", "player_team_code", "attacker_team_code"]).collect()
DE.write_parquet(SCR / "deaths.parquet")
print("player_death rows:", DE.height)

print("\nbuild done ->", SCR)
