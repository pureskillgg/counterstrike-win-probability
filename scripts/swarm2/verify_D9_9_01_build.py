"""verify_D9_9 step 1: build an independent per-round table from the raw 2024-02 tome (read-only).

Independent of scripts\\swarm2\\D9_build.py (no reuse of its parquet). Columns are projected per channel.
Output (scratchpad, not the project): verify_D9_9_round.parquet, verify_D9_9_playerround.parquet,
verify_D9_9_playerdeath.parquet, verify_D9_9_match.parquet.

ASSUMPTIONS (marked ASSUMPTION in the log):
  - build_num from the header is the build key (as the claim states).
  - bomb side: bomb_defused -> 3 (CT), bomb_exploded -> 2 (T) (standard CS rule; the claim uses the same).
  - round_state t_score / ct_score at event round_officially_ended is the end-of-round score.
  - team codes: 2 = T, 3 = CT.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
SCR.mkdir(parents=True, exist_ok=True)


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, want, header=False):
    fd = BASE / "header" if header else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    missing = [c for c in want if c not in have]
    if missing:
        print(f"[{name}] MISSING columns: {missing}")
    return lf.select([c for c in want if c in have]).collect()


# ---- header: build per match
H = scan("header", ["key", "build_num", "map_name", "match_date", "rushb_version"], header=True)
H = H.with_columns(pl.col("key").alias("match_key"),
                   pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
print("header rows:", H.height, "| distinct keys:", H.select("key").n_unique(),
      "| build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
M = H.select(["match_key", "build_num", "map_name", "day", "rushb_version"])

# ---- round_end
RE_raw = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message",
                            "player_count"])
dupre = RE_raw.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("round_end rows:", RE_raw.height, "| (m,r) with >1 row:", dupre.height)
RE = RE_raw.unique(subset=["match_key", "round"], keep="first")
print("round_end unique (m,r):", RE.height)

# ---- round_state: end-of-round score
RS = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
ROE = (RS.filter(pl.col("event_type") == "round_officially_ended")
       .group_by(["match_key", "round"])
       .agg(pl.col("tick").max().alias("ro_tick"),
            pl.col("t_score").max().alias("t_after"),
            pl.col("ct_score").max().alias("ct_after")))
print("round_officially_ended unique (m,r):", ROE.height)

# previous-round score (for the score delta); round 1 starts at 0-0 (ASSUMPTION; checked below)
ROE = ROE.sort(["match_key", "round"]).with_columns(
    pl.col("t_after").shift(1).over("match_key").alias("t_prev_raw"),
    pl.col("ct_after").shift(1).over("match_key").alias("ct_prev_raw"),
    pl.col("round").shift(1).over("match_key").alias("r_prev"))
ROE = ROE.with_columns(
    pl.when(pl.col("round") == 1).then(pl.lit(0)).otherwise(pl.col("t_prev_raw")).alias("t_prev"),
    pl.when(pl.col("round") == 1).then(pl.lit(0)).otherwise(pl.col("ct_prev_raw")).alias("ct_prev"),
)

# ---- bomb outcome and planter / defuser sides (bomb_state)
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_id", "player_team_code"])
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg((pl.col("event_type") == "bomb_defused").any().alias("defused"),
             (pl.col("event_type") == "bomb_exploded").any().alias("exploded"),
             pl.len().alias("n_outcome_rows")))
plant = (BS.filter(pl.col("event_type") == "bomb_planted")
         .group_by(["match_key", "round"])
         .agg(pl.col("player_team_code").drop_nulls().first().alias("planter_side"),
              pl.col("player_id").first().alias("planter_id"),
              pl.len().alias("n_plant_rows")))
defuse = (BS.filter(pl.col("event_type") == "bomb_defused")
          .group_by(["match_key", "round"])
          .agg(pl.col("player_team_code").drop_nulls().first().alias("defuser_side")))
print("bomb_state event types present:",
      BS.group_by("event_type").len().sort("event_type").rows())

# ---- player_info roster per round (dedupe per player; check conflicts)
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
pi_conf = PI.group_by(["match_key", "round", "player_id"]).agg(pl.col("team_code").n_unique().alias("nteam"))
print("player_info (m,r,p) rows:", PI.height, "| unique (m,r,p):", pi_conf.height,
      "| (m,r,p) with >1 team_code:", pi_conf.filter(pl.col("nteam") > 1).height)
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
roster = (PIu.group_by(["match_key", "round"])
          .agg((pl.col("team_code") == 2).sum().alias("n_T"),
               (pl.col("team_code") == 3).sum().alias("n_CT"),
               (pl.col("team_code") == 0).sum().alias("n_team0"),
               pl.len().alias("n_players_info")))

# ---- deaths (victim side = player_team_code; attacker side = attacker_team_code)
DE = scan("player_death", ["match_key", "round", "tick", "player_id", "player_team_code",
                           "attacker_id", "attacker_team_code"])
print("player_death rows:", DE.height,
      "| null victim team:", DE.filter(pl.col("player_team_code").is_null()).height,
      "| null victim id:", DE.filter(pl.col("player_id").is_null()).height)
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
print("player_death unique (m,r,victim):", DEu.height)
deaths = (DEu.group_by(["match_key", "round"])
          .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
               (pl.col("player_team_code") == 3).sum().alias("CT_dead"),
               (pl.col("player_team_code").is_null()).sum().alias("dead_null_team"),
               pl.len().alias("n_dead_rows")))

# ---- assemble round table
R = (RE.join(M, on="match_key", how="left")
       .join(ROE, on=["match_key", "round"], how="left")
       .join(outc, on=["match_key", "round"], how="left")
       .join(plant, on=["match_key", "round"], how="left")
       .join(defuse, on=["match_key", "round"], how="left")
       .join(roster, on=["match_key", "round"], how="left")
       .join(deaths, on=["match_key", "round"], how="left"))
R = R.with_columns(
    pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False),
    pl.col("n_outcome_rows").fill_null(0),
    pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0), pl.col("n_dead_rows").fill_null(0),
    pl.col("dead_null_team").fill_null(0))
R = R.with_columns(
    (pl.col("defused") | pl.col("exploded")).alias("has_outcome"),
    pl.when(pl.col("defused")).then(pl.lit(3)).when(pl.col("exploded")).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round").is_in([25, 28])).then(pl.lit("r25_28"))
      .when(pl.col("round") == 9).then(pl.lit("r9"))
      .otherwise(pl.lit("other")).alias("rclass"),
)
# recorded side-check mismatch (claim definition): bomb side != recorded winner
R = R.with_columns(
    pl.when(pl.col("has_outcome")).then(pl.col("bomb_side") != pl.col("winner_team_code"))
      .otherwise(None).alias("bomb_mismatch"))

# score-based winner (round end minus round start, end-of-round scores from round_state)
R = R.with_columns(
    (pl.col("t_after") - pl.col("t_prev")).alias("dT_straight"),
    (pl.col("ct_after") - pl.col("ct_prev")).alias("dCT_straight"),
    (pl.col("t_after") - pl.col("ct_prev")).alias("dT_swapped"),
    (pl.col("ct_after") - pl.col("t_prev")).alias("dCT_swapped"),
)
R.write_parquet(SCR / "verify_D9_9_round.parquet")
print("wrote round table rows:", R.height)

# ---- player-round table (for label-frame checks)
PIu.join(M, on="match_key", how="left").write_parquet(SCR / "verify_D9_9_playerround.parquet")
DE.join(M, on="match_key", how="left").write_parquet(SCR / "verify_D9_9_playerdeath.parquet")
BS.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).join(
    M, on="match_key", how="left").write_parquet(SCR / "verify_D9_9_bombevents.parquet")
M.write_parquet(SCR / "verify_D9_9_match.parquet")
print("wrote player/death/bomb/match tables")
