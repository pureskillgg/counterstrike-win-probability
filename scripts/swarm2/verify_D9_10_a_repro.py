"""verify D9-10 (a): independent reproduction of the claimed build-9961 flip counts from the raw 2024 tome.

Does NOT read the scratchpad parquet written by D9_build.py; rebuilds the per-round table directly.
Rule copied from D9_tests10.py (the claim's rule), so the reproduction tests the number, not the rule.
Rule (per non-bomb round, winner_team_code from round_end):
  winner == 2 (T): R if all CT dead (CT_dead >= n_CT); F if all T dead; else N
  winner != 2:     R if all T dead;                    F if all CT dead; else N
  bomb outcome present in bomb_state -> '-' (not scored)
ASSUMPTIONS (ASSUMPTION): n_T/n_CT = distinct players in player_info per round (keep last);
  victim side = player_death.player_team_code (keep first per player per round);
  build = header.build_num; pc20 = round_end.player_count == 20 on that round.
Read-only on the tome. Memory: lazy scans with projection, a few hundred MB peak.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
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


H = scan("header", ["key", "build_num", "game_directory"]).rename({"key": "match_key"})
print("header rows:", H.height, "| distinct match_key:", H.n_unique("match_key"))
print("build_num counts (top):", H.group_by("build_num").len().sort("len", descending=True).head(8).rows())

RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message", "player_count"])
print("round_end rows:", RE.height, "| unique (match,round):", RE.select(["match_key", "round"]).unique().height)
RE = RE.unique(subset=["match_key", "round"], keep="first")

PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
pi_dup = PI.group_by(["match_key", "round", "player_id"]).agg(pl.col("team_code").n_unique().alias("nteam"),
                                                              pl.len().alias("n"))
print("player_info (match,round,player) groups:", pi_dup.height,
      "| groups with >1 row:", pi_dup.filter(pl.col("n") > 1).height,
      "| groups with >1 distinct team_code:", pi_dup.filter(pl.col("nteam") > 1).height)
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
tc = (PIu.group_by(["match_key", "round"])
      .agg((pl.col("team_code") == 2).sum().alias("n_T"),
           (pl.col("team_code") == 3).sum().alias("n_CT"),
           pl.len().alias("n_info")))

DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"])
print("player_death rows:", DE.height, "| null player_id:", DE.filter(pl.col("player_id").is_null()).height,
      "| null player_team_code:", DE.filter(pl.col("player_team_code").is_null()).height)
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dead = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead")))

BS = scan("bomb_state", ["match_key", "round", "event_type"])
bomb = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("bomb_any")))

T = (RE.join(H.select(["match_key", "build_num"]), on="match_key", how="left")
       .join(tc, on=["match_key", "round"], how="left")
       .join(dead, on=["match_key", "round"], how="left")
       .join(bomb, on=["match_key", "round"], how="left")
       .with_columns(pl.col("bomb_any").fill_null(False),
                     pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0)))

print("\nbuild_num missing after join:", T.filter(pl.col("build_num").is_null()).height)
print("winner_team_code values:", T.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("player_count values (all rounds):", T.group_by("player_count").len().sort("len", descending=True).head(6).rows())
print("rounds with no player_info (n_T null):", T.filter(pl.col("n_T").is_null()).height)

# n_T + n_CT distribution on pc20 rounds (is the roster 10 vs 10?)
pc20 = T.filter((pl.col("build_num") == 9961) & (pl.col("player_count") == 20))
print("\nbuild 9961, pc20 rounds:", pc20.height,
      "| n_T+n_CT distribution:", pc20.with_columns((pl.col("n_T") + pl.col("n_CT")).alias("tot"))
      .group_by("tot").len().sort("tot").rows())

S = T.filter((pl.col("build_num") == 9961) & (pl.col("player_count") == 20) & pl.col("n_CT").is_not_null()
             & pl.col("round").is_in(ROUNDS_OF_INTEREST))
S = S.with_columns(
    pl.when(pl.col("bomb_any")).then(pl.lit("-"))
      .when(pl.col("winner_team_code") == 2).then(
          pl.when(pl.col("CT_dead") >= pl.col("n_CT")).then(pl.lit("R"))
          .when(pl.col("T_dead") >= pl.col("n_T")).then(pl.lit("F")).otherwise(pl.lit("N")))
      .otherwise(
          pl.when(pl.col("T_dead") >= pl.col("n_T")).then(pl.lit("R"))
          .when(pl.col("CT_dead") >= pl.col("n_CT")).then(pl.lit("F")).otherwise(pl.lit("N")))
      .alias("c"))

print("\n== per round label: scored (R/F/N), flipped F, share of scored")
for rr in ROUNDS_OF_INTEREST:
    sub = S.filter(pl.col("round") == rr)
    counts = sub.group_by("c").len().rows()
    d = dict(counts)
    scored = d.get("R", 0) + d.get("F", 0)
    f = d.get("F", 0)
    share = (f / scored) if scored else float("nan")
    print(f"  round {rr:>2}: rows={sub.height:>5} R={d.get('R',0):>5} F={f:>5} N={d.get('N',0):>5} "
          f"-={d.get('-',0):>5} | flipped share of scored = {share:.4f}")

pat_matches = S.group_by("match_key").len().height
print("\nmatches with any row in the round set (pc20, build 9961, n_CT present):", pat_matches)
scored_any = (S.filter(pl.col("c").is_in(["R", "F"])).select("match_key").unique().height)
print("matches with any scored round in the set:", scored_any)

Fm = S.filter(pl.col("c") == "F").select(["match_key", "round"])
flips_per_match = Fm.group_by("match_key").agg(pl.len().alias("nf"))
print("matches by number of flipped rounds:",
      flips_per_match.group_by("nf").agg(pl.len().alias("n_matches")).sort("nf").rows(),
      "| matches with >=1 flip:", flips_per_match.height,
      "| scored matches with zero flips:", scored_any - flips_per_match.height)

