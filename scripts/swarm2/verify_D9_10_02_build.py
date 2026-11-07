"""verify_D9_10 step 2: independent per-round table for the 2024-02-g tome (read-only on the tome).

Writes only to the session scratchpad (not the project). Projected columns, one table at a time.
Definitions (ASSUMPTION where noted):
  - build_num from header (game build), pc20 = round_end.player_count == 20 (same as the claim).
  - n_T / n_CT: distinct players per round in player_info with team_code 2 / 3 (same as D9_build.py).
  - T_dead / CT_dead: distinct players per round in player_death with player_team_code 2 / 3 (same as claim).
  - bomb outcome: bomb_defused -> side 3, bomb_exploded -> side 2 (same rule as the side check).
  - planter_side: player_team_code on the bomb_planted row (a T-side player must plant; ASSUMPTION of game rule).
  - death label agreement: a death row's player_team_code vs the same player's player_info team_code in that round.
  - end_tick: round_end.tick; start_tick: round_start.tick (first row); bomb ticks from bomb_state.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(30)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_10")
SCR.mkdir(parents=True, exist_ok=True)


def scan(name, cols, header=False):
    fd = BASE / "header" if header else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(sorted(glob.glob(str(fd / "dataframe_*"))))
    have = set(lf.collect_schema().names())
    missing = [c for c in cols if c not in have]
    if missing:
        print(f"  [{name}] missing: {missing}")
    return lf.select([c for c in cols if c in have]).collect()


# ---- header
H = scan("header", ["key", "build_num", "match_date"], header=True).rename({"key": "match_key"})
print("header rows", H.height, "| distinct builds:", H.group_by("build_num").len().sort("build_num").rows())

# ---- round_end (one row per match,round expected; check duplicates)
RE_all = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                            "win_reason_message", "player_count"])
dup = RE_all.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("round_end rows", RE_all.height, "| (match,round) groups with >1 row:", dup.height)
if dup.height:
    print(RE_all.join(dup.select(["match_key", "round"]), on=["match_key", "round"]).sort(["match_key", "round", "tick"]).head(10))
RE = RE_all.unique(subset=["match_key", "round"], keep="first")
REASON_T = {"#SFUI_Notice_Terrorists_Win", "#SFUI_Notice_Target_Bombed", "#SFUI_Notice_CTs_Surrender",
            "#SFUI_Notice_Hostages_Not_Resc"}
REASON_CT = {"#SFUI_Notice_CTs_Win", "#SFUI_Notice_Bomb_Defused", "#SFUI_Notice_Target_Saved",
             "#SFUI_Notice_Terrorists_Surren", "#SFUI_Notice_All_Hostages_Resc"}
RE = RE.with_columns(
    pl.when(pl.col("win_reason_message").is_in(list(REASON_T))).then(pl.lit(2))
      .when(pl.col("win_reason_message").str.contains("Terrorists_Win|Target_Bombed|CTs_Surrender|Hostages_Not")).then(pl.lit(2))
      .when(pl.col("win_reason_message").str.contains("CTs_Win|Bomb_Defused|Target_Saved|Terrorists_Surren|All_Hostages")).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("reason_winner"))
print("reason_winner unmapped:", RE.filter(pl.col("reason_winner").is_null()).height,
      "| winner != reason_winner:", RE.filter(pl.col("reason_winner") != pl.col("winner_team_code")).height)

# ---- round_start first tick per round
RS = scan("round_start", ["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("start_tick"))

# ---- bomb: outcome, planter side, ticks
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
bo = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg((pl.col("event_type") == "bomb_defused").any().alias("defused"),
             (pl.col("event_type") == "bomb_exploded").any().alias("exploded"),
             pl.col("tick").max().alias("bomb_end_tick")))
pl_ = (BS.filter(pl.col("event_type") == "bomb_planted")
         .group_by(["match_key", "round"])
         .agg(pl.col("player_team_code").first().alias("planter_side"),
              pl.len().alias("n_plant_rows")))

# ---- player_info: per round distinct players and side
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
tc = (PIu.group_by(["match_key", "round"])
        .agg((pl.col("team_code") == 2).sum().alias("n_T"),
             (pl.col("team_code") == 3).sum().alias("n_CT"),
             pl.len().alias("n_info")))

# ---- player_death: distinct deaths per round and side; per-death label check vs player_info
DE = scan("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dd = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead"),
             pl.col("tick").min().alias("first_death_tick"),
             pl.col("tick").max().alias("last_death_tick"),
             pl.len().alias("n_death_players")))
lab = (DEu.join(PIu.select(["match_key", "round", "player_id", pl.col("team_code").alias("info_team")]),
                on=["match_key", "round", "player_id"], how="left")
          .with_columns(pl.when(pl.col("info_team").is_null()).then(pl.lit("no_info"))
                          .when(pl.col("info_team") == pl.col("player_team_code")).then(pl.lit("agree"))
                          .otherwise(pl.lit("disagree")).alias("lab")))
print("death rows (distinct player per round):", DEu.height)
print("death-label vs player_info team_code:", lab.group_by("lab").len().sort("lab").rows())
lab_r = lab.group_by(["match_key", "round"]).agg(
    (pl.col("lab") == "agree").sum().alias("lab_agree"),
    (pl.col("lab") == "disagree").sum().alias("lab_disagree"))

# ---- assemble per-round table
R = (RE.join(H, on="match_key", how="left")
       .join(RS, on=["match_key", "round"], how="left")
       .join(bo, on=["match_key", "round"], how="left")
       .join(pl_, on=["match_key", "round"], how="left")
       .join(tc, on=["match_key", "round"], how="left")
       .join(dd, on=["match_key", "round"], how="left")
       .join(lab_r, on=["match_key", "round"], how="left"))
R = R.with_columns(
    pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False),
    pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0),
    pl.col("lab_agree").fill_null(0), pl.col("lab_disagree").fill_null(0),
    (pl.col("defused") | pl.col("exploded")).alias("has_bomb_outcome"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
R.write_parquet(SCR / "round_table.parquet")
print("wrote", SCR / "round_table.parquet", "rows", R.height)
print("rounds with build_num 9961:", R.filter(pl.col("build_num") == 9961).height,
      "| matches:", R.filter(pl.col("build_num") == 9961)["match_key"].n_unique())
print("rounds per build (top):", R.group_by("build_num").agg(pl.col("match_key").n_unique().alias("matches"), pl.len().alias("rounds")).sort("matches", descending=True).head(8))
