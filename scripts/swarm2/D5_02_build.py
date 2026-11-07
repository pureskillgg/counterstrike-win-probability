"""D5 step 2: build per-round and per-match tables for the 2024 tome.

Outputs (scratchpad, not the project):
  D5/rounds.parquet   one row per (match_key, round) in the union of round_start,
                      round_state round_freeze_end, and round_end; with counts of
                      every round_state event type per round (restart/pause probes).
  D5/bomb.parquet     bomb_planted / bomb_defused / bomb_exploded rows from bomb_state.
  D5/matches.parquet  one row per header match: day, map, final scores, round counts,
                      r13 mismatch flag (same definition as scripts/swarm/e2_side_check_r13.py).
  D5/deaths.parquet   player_death (match_key, round, tick) for window checks.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_02_build.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D5_common import day_of, header, read  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
SCR.mkdir(parents=True, exist_ok=True)

# ---------- header / matches ----------
H = header().with_columns(
    match_key=pl.col("key"),
    day=day_of(pl.col("key")),
)

# ---------- round_end ----------
RE_raw = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                            "win_reason_message", "legacy_code", "player_count"])
RE_dupe_pairs = (RE_raw.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
                 .filter(pl.col("n") > 1))
print("round_end duplicate (match,round) pairs:", RE_dupe_pairs.height)
print(RE_dupe_pairs)
for row in RE_dupe_pairs.iter_rows(named=True):
    print(RE_raw.filter((pl.col("match_key") == row["match_key"]) & (pl.col("round") == row["round"])))

RE = (RE_raw.sort(["match_key", "round", "tick"])
      .with_columns(pl.int_range(0, pl.len()).over(["match_key", "round"]).alias("end_seq"))
      .group_by(["match_key", "round"], maintain_order=False).agg(
          pl.col("tick").first().alias("end_tick"),
          pl.col("tick").max().alias("end_tick_last"),
          pl.col("winner_team_code").first().alias("winner"),
          pl.col("win_reason_code").first().alias("reason_code"),
          pl.col("win_reason_message").first().alias("reason_msg"),
          pl.col("legacy_code").first().alias("legacy_code"),
          pl.col("player_count").first().alias("player_count"),
          pl.len().alias("n_end_rows"),
          pl.col("winner_team_code").n_unique().alias("n_distinct_winner"),
      ))

# ---------- round_start ----------
RST = read("round_start", ["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("start_tick"),
    pl.len().alias("n_start_rows"))

# ---------- round_state ----------
RS_TYPES = [
    "round_freeze_end", "round_start", "round_end", "round_officially_ended", "round_prestart",
    "round_poststart", "cs_pre_restart", "cs_round_start_beep", "cs_round_final_beep",
    "buytime_ended", "cs_win_panel_round", "round_announce_match_start",
    "round_announce_last_round_half", "round_announce_match_point", "announce_phase_end",
    "round_announce_final", "begin_new_match", "cs_win_panel_match",
    "bomb_planted", "bomb_defused", "bomb_exploded",
]
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
RS = RS.filter(pl.col("event_type").is_in(RS_TYPES))

FE = (RS.filter(pl.col("event_type") == "round_freeze_end")
      .sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(
          pl.col("tick").first().alias("fe_tick"),
          pl.col("tick").last().alias("fe_tick_last"),
          pl.col("t_score").first().alias("fe_t_score"),
          pl.col("ct_score").first().alias("fe_ct_score"),
          pl.col("phase").first().alias("fe_phase"),
          pl.len().alias("n_fe_rows")))

# per (match, round) counts for each round_state event type
cnt = (RS.group_by(["match_key", "round", "event_type"]).agg(pl.len().alias("n"))
       .pivot(on="event_type", index=["match_key", "round"], values="n", aggregate_function="sum")
       .fill_null(0))
cnt = cnt.rename({c: f"n_{c}" for c in cnt.columns if c not in ("match_key", "round")})

# ---------- bomb_state ----------
BS = read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
BOMB = BS.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
BOMB.write_parquet(SCR / "bomb.parquet")
print("bomb_state bomb rows:", BOMB.height, BOMB.group_by("event_type").len().to_dicts())

# ---------- deaths (projected) ----------
DE = read("player_death", ["match_key", "round", "tick"])
DE.write_parquet(SCR / "deaths.parquet")
print("player_death rows:", DE.height)

# ---------- union rounds ----------
ROUNDS = (RST.join(FE, on=["match_key", "round"], how="full", coalesce=True)
          .join(RE, on=["match_key", "round"], how="full", coalesce=True)
          .join(cnt, on=["match_key", "round"], how="left"))
ROUNDS = ROUNDS.with_columns(
    in_start=pl.col("start_tick").is_not_null(),
    in_fe=pl.col("fe_tick").is_not_null(),
    in_end=pl.col("end_tick").is_not_null(),
)
ROUNDS.write_parquet(SCR / "rounds.parquet")
print("rounds union rows:", ROUNDS.height)
print("presence pattern (start, freeze_end, end):")
print(ROUNDS.group_by(["in_start", "in_fe", "in_end"]).agg(pl.len().alias("n")).sort(["in_start", "in_fe", "in_end"]))

# ---------- matches ----------
r13 = (RE.filter(pl.col("player_count") == 20)
       .join(BOMB.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
             .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")),
             on=["match_key", "round"], how="inner")
       .with_columns(
           pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
       .with_columns((pl.col("bomb_side") != pl.col("winner")).alias("mismatch")))
mm_all = r13.filter(pl.col("mismatch"))
print("\nall-round bomb-outcome mismatches (player_count==20):", mm_all.height, "of", r13.height)
print("mismatch round-number distribution (top):")
print(mm_all.group_by("round").agg(pl.len().alias("n")).sort("round"))
aff_r13 = mm_all.filter(pl.col("round") == 13).select("match_key").unique()
print("matches with an r13 mismatch:", aff_r13.height)

M_ROUND_COUNTS = (RE.group_by("match_key").agg(
    pl.len().alias("n_end_pairs"),
    pl.col("round").max().alias("max_end_round"),
    pl.col("round").min().alias("min_end_round"),
    (pl.col("n_end_rows") > 1).sum().alias("n_dup_end_pairs")))
S_ROUND_COUNTS = (RST.group_by("match_key").agg(
    pl.len().alias("n_start_pairs"), pl.col("round").max().alias("max_start_round")))

MATCHES = (H.select(["match_key", "match_id", "day", "map_name", "match_date",
                     "t_starters_score_final", "ct_starters_score_final"])
           .join(M_ROUND_COUNTS, on="match_key", how="left")
           .join(S_ROUND_COUNTS, on="match_key", how="left")
           .with_columns(
               final_sum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"),
               aff_r13=pl.col("match_key").is_in(aff_r13["match_key"].to_list()),
               mm_r13_rows=pl.lit(0)))
MATCHES.write_parquet(SCR / "matches.parquet")
print("\nmatches rows:", MATCHES.height, " aff_r13 matches:", int(MATCHES["aff_r13"].sum()))
print(MATCHES.group_by("day").agg(pl.len().alias("matches"), pl.col("aff_r13").sum().alias("aff")).sort("day"))
