"""verify D9-10 (b): diagnostics for the strongest alternative explanations.

Alternatives tested (all on build 9961, pc20 rounds, raw 2024 tome, no scratchpad parquet):
  ALT-1 roster artifact: n_T / n_CT from player_info are stale or swapped in round 13 (or the first
        round after a switch), so the death test flips while the winner label is fine.
  ALT-2 death-side artifact: player_death.player_team_code is stale for the first round after a switch.
        Checked by victim-level agreement between death team code and player_info team code in the same round.
  ALT-3 missing death side: null player_team_code on deaths (dropped from T_dead/CT_dead) concentrated in flipped rounds.
  ALT-4 side switch not at round 13 in these matches: player_info team_code does not change at the 12->13 boundary.
ASSUMPTION: player_info.team_code is the side at that round (2 = T, 3 = CT); a side switch shows as a change
  for the same player_id between consecutive rounds.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
ROUNDS_OF_INTEREST = [8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30]


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    missing = [c for c in cols if c not in have]
    if missing:
        raise SystemExit(f"[{name}] missing columns {missing}")
    return lf.select(cols).collect()


H = scan("header", ["key", "build_num"]).rename({"key": "match_key"})
B9961 = set(H.filter(pl.col("build_num") == 9961)["match_key"].to_list())
print("build 9961 matches:", len(B9961))

RE = scan("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    subset=["match_key", "round"], keep="first")
RE = RE.filter(pl.col("match_key").is_in(list(B9961)) & (pl.col("player_count") == 20))

PI = scan("player_info", ["match_key", "round", "player_id", "team_code"]).filter(
    pl.col("match_key").is_in(list(B9961)))
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
tc = (PIu.group_by(["match_key", "round"])
      .agg((pl.col("team_code") == 2).sum().alias("n_T"),
           (pl.col("team_code") == 3).sum().alias("n_CT")))

DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"]).filter(
    pl.col("match_key").is_in(list(B9961)))
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dead = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead"),
             pl.col("player_team_code").is_null().sum().alias("dead_null_team")))

BS = scan("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("match_key").is_in(list(B9961)))
bomb = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("bomb_any")))

T = (RE.join(tc, on=["match_key", "round"], how="left")
       .join(dead, on=["match_key", "round"], how="left")
       .join(bomb, on=["match_key", "round"], how="left")
       .with_columns(pl.col("bomb_any").fill_null(False), pl.col("T_dead").fill_null(0),
                     pl.col("CT_dead").fill_null(0), pl.col("dead_null_team").fill_null(0)))

S = T.filter(pl.col("n_CT").is_not_null() & pl.col("round").is_in(ROUNDS_OF_INTEREST))
S = S.with_columns(
    pl.when(pl.col("bomb_any")).then(pl.lit("-"))
      .when(pl.col("winner_team_code") == 2).then(
          pl.when(pl.col("CT_dead") >= pl.col("n_CT")).then(pl.lit("R"))
          .when(pl.col("T_dead") >= pl.col("n_T")).then(pl.lit("F")).otherwise(pl.lit("N")))
      .otherwise(
          pl.when(pl.col("T_dead") >= pl.col("n_T")).then(pl.lit("R"))
          .when(pl.col("CT_dead") >= pl.col("n_CT")).then(pl.lit("F")).otherwise(pl.lit("N")))
      .alias("c"))

# ---------- ALT-1: roster counts by round and flip status
print("\n== ALT-1 roster counts (n_T, n_CT) by round, all scored rounds (R/F/N)")
S = S.with_columns((pl.col("n_T") + pl.col("n_CT")).alias("tot"))
for rr in [9, 12, 13, 14, 25, 28]:
    sub = S.filter((pl.col("round") == rr) & pl.col("c").is_in(["R", "F"]))
    print(f"  round {rr}: scored={sub.height}, tot (n_T+n_CT) distribution:",
          sub.group_by("tot").len().sort("tot").rows())
print("  round 13 (scored), by c and (n_T, n_CT) pattern among tot==10 vs other:")
r13 = S.filter((pl.col("round") == 13) & pl.col("c").is_in(["R", "F"]))
print("   ", r13.with_columns(pl.when(pl.col("tot") == 10).then(pl.lit("tot10")).otherwise(pl.lit("other")).alias("grp"))
      .group_by(["c", "grp"]).len().sort(["c", "grp"]).rows())

# ---------- ALT-3: null death team codes by round (dropped from T_dead / CT_dead)
print("\n== ALT-3 null player_team_code on deaths, by round (scored rounds only)")
sc = S.filter(pl.col("c").is_in(["R", "F"]))
print(sc.group_by("round").agg(pl.len().alias("n"), (pl.col("dead_null_team") > 0).sum().alias("rounds_with_null_victim"),
                               pl.col("dead_null_team").sum().alias("null_victims"))
      .sort("round").rows())
print("  flipped (F) vs R rounds, share with any null-team victim:")
print(sc.group_by("c").agg(pl.len().alias("n"), (pl.col("dead_null_team") > 0).mean().alias("share_any_null"))
      .sort("c").rows())

# ---------- ALT-4: side switch as seen in player_info per boundary (player_id team change between rounds)
print("\n== ALT-4 player_info side change per round boundary (build 9961, all rounds of these matches)")
pi_w = PIu.select(["match_key", "round", "player_id", "team_code"])
nxt = pi_w.with_columns((pl.col("round") - 1).alias("round"))  # align round r+1 rows to boundary r->r+1 on key round=r
both = pi_w.join(nxt.rename({"team_code": "team_next"}), on=["match_key", "round", "player_id"], how="inner")
# both: rows with team_code at round r (left) and team at round r+1 (team_next)
both = both.with_columns((pl.col("team_code") != pl.col("team_next")).alias("changed"))
bd = (both.group_by(["match_key", "round"])
      .agg(pl.len().alias("n_common"), pl.col("changed").mean().alias("frac_changed")))
bd = bd.with_columns(pl.when(pl.col("frac_changed") >= 0.5).then(pl.lit("switch_like"))
                     .when(pl.col("frac_changed") <= 0.1).then(pl.lit("no_change"))
                     .otherwise(pl.lit("mixed")).alias("kind"))
print("  boundary r->r+1 counts by kind (matches with >=3 common players):")
print(bd.filter(pl.col("n_common") >= 3).group_by(["round", "kind"]).len().sort(["round", "kind"])
      .filter(pl.col("round").is_in([7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29]))
      .rows())

sw = bd.filter(pl.col("n_common") >= 3).select(["match_key", "round", "kind"])
sw13 = sw.filter(pl.col("round") == 12).rename({"kind": "kind_12_13"}).select(["match_key", "kind_12_13"])
sw_ = sw.filter(pl.col("round") == 8).rename({"kind": "kind_8_9"}).select(["match_key", "kind_8_9"])
sw8_9 = sw_
joined = S.filter(pl.col("round") == 13).select(["match_key", "c"]).join(sw13, on="match_key", how="left")
print("\n  round-13 flip status vs player_info switch at boundary 12->13 (scored only):")
print(joined.filter(pl.col("c").is_in(["R", "F"])).group_by(["c", "kind_12_13"]).len()
      .sort(["c", "kind_12_13"]).rows())
joined9 = S.filter(pl.col("round") == 9).select(["match_key", "c"]).join(sw8_9, on="match_key", how="left")
print("\n  round-9 flip status vs player_info change at boundary 8->9 (scored only):")
print(joined9.filter(pl.col("c").is_in(["R", "F"])).group_by(["c", "kind_8_9"]).len()
      .sort(["c", "kind_8_9"]).rows())

# ---------- ALT-2: victim-level agreement, death team code vs player_info team code in the same round
print("\n== ALT-2 victim-level agreement: death player_team_code vs player_info team_code (same match, round, player)")
dv = (DEu.select(["match_key", "round", "player_id", "player_team_code"])
      .join(PIu.select(["match_key", "round", "player_id", "team_code"]),
            on=["match_key", "round", "player_id"], how="inner"))
dv = dv.with_columns((pl.col("player_team_code") == pl.col("team_code")).alias("agree"))
print("  victims matched to player_info:", dv.height, "| agree rate:",
      round(dv["agree"].mean(), 5))
dv_round = dv.group_by("round").agg(pl.len().alias("n"), pl.col("agree").mean().round(4).alias("agree_rate")).sort("round")
print("  agree rate by round (build 9961):")
print(dv_round.filter(pl.col("round").is_in(ROUNDS_OF_INTEREST)).rows())
# agreement restricted to rounds where the flip status is known
keyed = S.filter(pl.col("c").is_in(["R", "F"])).select(["match_key", "round", "c"])
dvk = dv.join(keyed, on=["match_key", "round"], how="inner")
print("  agree rate on victims in scored rounds, by flip status (c):")
print(dvk.group_by(["round", "c"]).agg(pl.len().alias("n"), pl.col("agree").mean().round(4).alias("agree_rate"))
      .filter(pl.col("round").is_in([9, 12, 13, 14, 25, 28])).sort(["round", "c"]).rows())
