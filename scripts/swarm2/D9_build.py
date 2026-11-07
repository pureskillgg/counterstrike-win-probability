"""D9 build: per-match and per-round tables for the 2024-02 tome (read-only on the tome).

Writes intermediate parquet to the session scratchpad (not the project).
Reconciles with the side-check population (player_count == 20, bomb outcome rows):
  expected from scripts\\swarm\\out\\e2_side_check_r13.log: rounds with outcome 15436,
  mismatches 499; round 13: 785 rows, 434 mismatches.
Assumptions (ASSUMPTION):
  - bomb outcome: bomb_defused -> side 3 (CT), bomb_exploded -> side 2 (T), same rule as the side check
  - match day from the match_key path csds/2024/02/DD/
  - duplicate signature replicates bakeoff.py (round_end + player_death + player_info events)
  - is_bot on player_disconnect / player_personal is taken as recorded (not verified)
"""
import collections
import glob
import hashlib
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9")
SCR.mkdir(parents=True, exist_ok=True)


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, want):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    cols = [c for c in want if c in have]
    missing = [c for c in want if c not in have]
    if missing:
        print(f"  [{name}] missing columns (skipped): {missing}")
    return lf.select(cols).collect()


print("== load header")
H = scan("header", ["key", "match_id", "match_date", "match_type", "platform", "map_name", "tick_rate",
                    "build_num", "game_directory", "rushb_version", "ppp_version", "providence",
                    "number_of_points", "sharecode", "demo_id", "t_starters_score_final",
                    "ct_starters_score_final", "tick_save_rate", "demo_version_name"])
H = H.with_columns(
    pl.col("key").alias("match_key"),
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
    pl.col("match_date").str.slice(8, 2).alias("date_day"),
    pl.col("game_directory").str.extract(r"csgo_v(\d+)", 1).alias("game_build"),
)
print("header rows", H.height, "| key day-parse nulls:", H.filter(pl.col("day").is_null()).height)

print("== load round_end")
RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                        "win_reason_message", "player_count"])
print("round_end rows", RE.height, "| unique (match,round):", RE.select(["match_key", "round"]).unique().height)
dupre = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("(match,round) groups with >1 round_end row:", dupre.height)

print("== load bomb_state / player_info / disconnect / personal / death / round_start")
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code", "player_id"])
print("bomb_state event types:", BS.group_by("event_type").len().sort("event_type").rows())
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
DC = scan("player_disconnect", ["match_key", "round", "tick", "player_id", "is_bot", "disconnect_reason"])
PP = scan("player_personal", ["match_key", "round", "player_id", "is_bot"])
DE = scan("player_death", ["match_key", "round", "tick", "player_id", "player_team_code",
                           "attacker_id", "weapon_name"])
RST = scan("round_start", ["match_key", "round", "tick", "time_limit", "frag_limit", "objective"])
BT_cols = pl.scan_parquet(files_of(BASE / f"basic_bot_takeover{SUF}")).collect_schema().names()
BT_rows = pl.scan_parquet(files_of(BASE / f"basic_bot_takeover{SUF}")).select(pl.len()).collect().item()
print("bot_takeover columns in 2024 tome:", BT_cols, "| rows:", BT_rows)

# ---- per-round outcome from bomb_state (same rule as e2_side_check_r13.py)
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg((pl.col("event_type") == "bomb_defused").any().alias("defused"),
             (pl.col("event_type") == "bomb_exploded").any().alias("exploded"),
             pl.len().alias("n_outcome_rows")))
plant = (BS.filter(pl.col("event_type") == "bomb_planted")
         .group_by(["match_key", "round"])
         .agg(pl.col("player_team_code").first().alias("planter_side"),
              pl.len().alias("n_plant_rows")))

# ---- team counts from player_info (dedupe per player)
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
tc = (PIu.group_by(["match_key", "round"])
      .agg((pl.col("team_code") == 2).sum().alias("n_T"),
           (pl.col("team_code") == 3).sum().alias("n_CT"),
           pl.len().alias("n_players_info")))

# ---- disconnects per round
dc_r = (DC.group_by(["match_key", "round"])
        .agg(pl.len().alias("dc_rows"),
             pl.col("is_bot").cast(pl.Int64).sum().alias("dc_bot_rows"),
             pl.col("player_id").n_unique().alias("dc_players")))

# ---- bots per round from personal rows (dedupe per player; no names or steam ids used)
PPu = PP.unique(subset=["match_key", "round", "player_id"], keep="last")
bots_r = (PPu.group_by(["match_key", "round"])
          .agg(pl.col("is_bot").cast(pl.Int64).sum().alias("personal_bots"),
               pl.len().alias("personal_players")))

# ---- deaths per round
de_r = DE.group_by(["match_key", "round"]).agg(pl.len().alias("n_deaths"))

# ---- round_start metadata
rst_r = RST.group_by(["match_key", "round"]).agg(pl.len().alias("n_round_start"),
                                                  pl.col("time_limit").first(),
                                                  pl.col("frag_limit").first(),
                                                  pl.col("objective").first())

R = (RE.join(outc, on=["match_key", "round"], how="left")
       .join(plant, on=["match_key", "round"], how="left")
       .join(tc, on=["match_key", "round"], how="left")
       .join(dc_r, on=["match_key", "round"], how="left")
       .join(bots_r, on=["match_key", "round"], how="left")
       .join(de_r, on=["match_key", "round"], how="left")
       .join(rst_r, on=["match_key", "round"], how="left"))
R = R.with_columns(
    pl.col("dc_rows").fill_null(0), pl.col("dc_bot_rows").fill_null(0), pl.col("dc_players").fill_null(0),
    pl.col("n_deaths").fill_null(0), pl.col("n_round_start").fill_null(0),
    pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False),
    pl.col("n_outcome_rows").fill_null(0),
)
R = R.with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
    pl.when(pl.col("defused")).then(pl.lit(3)).when(pl.col("exploded")).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"),
    (pl.col("defused") | pl.col("exploded")).alias("has_outcome"),
    (pl.col("round") == 13).alias("is_r13"),
)
R = R.with_columns(
    pl.when(pl.col("has_outcome")).then(pl.col("bomb_side") != pl.col("winner_team_code"))
      .otherwise(None).alias("mismatch"),
)

# ---- match-level aggregates from rounds
mrows = R.group_by("match_key").agg(
    pl.len().alias("n_round_end_rows"),
    pl.col("round").n_unique().alias("n_rounds"),
    (pl.col("player_count") == 20).mean().alias("frac_pc20"),
    pl.col("player_count").min().alias("pc_min"),
    pl.col("player_count").max().alias("pc_max"),
    pl.col("dc_rows").sum().alias("match_dc_rows"),
    pl.col("dc_bot_rows").sum().alias("match_dc_bot_rows"),
    pl.col("mismatch").sum().alias("match_mismatch_n"),
)
M = H.join(mrows, on="match_key", how="left")

# ---- duplicate signatures replicating bakeoff.py (round_end, player_death, player_info)
print("== duplicate signatures (bakeoff rule)")
sig_parts = collections.defaultdict(list)
for x in RE.iter_rows(named=True):
    sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
for x in DE.iter_rows(named=True):
    sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
for x in PI.iter_rows(named=True):
    sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
sig = {m: hashlib.md5("|".join(sorted(v)).encode("utf-8")).hexdigest() for m, v in sig_parts.items()}
M = M.with_columns(pl.col("match_key").replace_strict(sig, default=None, return_dtype=pl.String).alias("dup_sig"))
gs = M.group_by("dup_sig").agg(pl.len().alias("dup_group_size"),
                               pl.col("match_key").min().alias("dup_lowest_key"))
M = M.join(gs, on="dup_sig", how="left").with_columns(
    (pl.col("dup_group_size") > 1).alias("is_dup_group"),
    (pl.col("match_key") == pl.col("dup_lowest_key")).alias("is_lowest_key_in_group"),
)
n_groups = M.filter(pl.col("dup_group_size") > 1).select("dup_sig").n_unique()
print("matches:", M.height, "| dup groups (size>1):", n_groups,
      "| matches in dup groups:", M.filter(pl.col("is_dup_group")).height,
      "| non-lowest-key dup matches:", M.filter(pl.col("is_dup_group") & ~pl.col("is_lowest_key_in_group")).height)
for c in ["sharecode", "demo_id", "number_of_points"]:
    n_dupval = M.group_by(c).len().filter(pl.col("len") > 1).height
    print(f"header {c}: distinct values shared by >1 match: {n_dupval}")

R.write_parquet(SCR / "D9_round.parquet")
M.write_parquet(SCR / "D9_match.parquet")
print("wrote", SCR / "D9_round.parquet", "rows", R.height)
print("wrote", SCR / "D9_match.parquet", "rows", M.height)

# ---- reconcile with the side-check population
side = R.filter((pl.col("player_count") == 20) & pl.col("has_outcome"))
print("\nRECONCILE side-check population (pc==20 & outcome): rows", side.height,
      "mismatch", int(side["mismatch"].sum()))
r13s = side.filter(pl.col("is_r13"))
print("RECONCILE round 13 (pc==20 & outcome): rows", r13s.height, "mismatch", int(r13s["mismatch"].sum()))
allo = R.filter(pl.col("has_outcome"))
print("all rounds with outcome (any player_count): rows", allo.height, "mismatch", int(allo["mismatch"].sum()))
