"""verify_D8_5_build: independent per-round table for the 2024 tome (D8-5 verifier).

Reads (read-only) the 2024 tome channels header, round_end, bomb_state, round_state.
Writes only intermediate parquet to the session scratchpad (not under the repo, not under E:).
Per round: match_key, round, winner (round_end, first row per match/round), tb/cb (round_state
round_freeze_end scores, first tick), bside (3=CT if bomb_defused row, 2=T if only bomb_exploded row).
Per match: header map_name, tA = t_starters_score_final, tB = ct_starters_score_final, day from the key.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D8_5_build.py
"""
import glob
import re
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D8_5")
T, CT = 2, 3
DAY_RE = re.compile(r"csds/2024/02/(\d\d)/")


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    assert fs, f"no dataframe files in {fd}"
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def chan(name):
    return BASE / f"basic_{name}{SUF}"


OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ header
H = read_dir(BASE / "header", ["key", "map_name", "t_starters_score_final", "ct_starters_score_final"])
print(f"header rows {H.height}, unique keys {H['key'].n_unique()}, null map {H['map_name'].null_count()}, "
      f"null tA {H['t_starters_score_final'].null_count()}, null tB {H['ct_starters_score_final'].null_count()}")
print("header map_name counts:", H.group_by("map_name").len().sort("map_name").rows())
bad_day = H.filter(~pl.col("key").str.contains(DAY_RE.pattern)).height
print(f"header keys not matching csds/2024/02/DD/: {bad_day}")
print("example key:", H["key"][0])
M = H.select([pl.col("key").alias("match_key"), "map_name",
              pl.col("t_starters_score_final").cast(pl.Int64).alias("tA"),
              pl.col("ct_starters_score_final").cast(pl.Int64).alias("tB")])

# ------------------------------------------------------------------ round_end
RE_all = read_dir(chan("round_end"), ["match_key", "round", "winner_team_code"])
g = RE_all.group_by(["match_key", "round"]).agg(pl.len().alias("rows"),
                                                 pl.col("winner_team_code").n_unique().alias("nw"))
print(f"round_end rows {RE_all.height}; (match,round) groups {g.height}; groups with >1 row {(g['rows'] > 1).sum()}; "
      f"groups with conflicting winner {(g['nw'] > 1).sum()}; winner codes {RE_all['winner_team_code'].unique().sort().to_list()}")
RE = RE_all.unique(["match_key", "round"], keep="first").with_columns(pl.col("round").cast(pl.Int64),
                                                                       pl.col("winner_team_code").cast(pl.Int64).alias("winner"))
RE = RE.select(["match_key", "round", "winner"])

# ------------------------------------------------------------------ bomb outcomes
BS = read_dir(chan("bomb_state"), ["match_key", "round", "event_type"])
print("bomb_state event types:", BS.group_by("event_type").len().sort("event_type").rows())
BSo = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bo = BSo.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
bo = bo.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"),
    pl.col("evs").list.len().alias("n_ev_types")).select(["match_key", "round", "bside", "n_ev_types"])
print(f"rounds with a bomb outcome row: {bo.height}; with both defused and exploded: {(bo['n_ev_types'] > 1).sum()}")

# ------------------------------------------------------------------ round_state
RS = read_dir(chan("round_state"), ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
print("round_state event types:", RS.group_by("event_type").len().sort("event_type").rows())
fe_raw = RS.filter(pl.col("event_type") == "round_freeze_end")
fe_n = fe_raw.group_by(["match_key", "round"]).agg(pl.len().alias("nfe"))
print(f"round_freeze_end rows {fe_raw.height}; (match,round) groups {fe_n.height}; groups with >1 freeze_end: "
      f"{(fe_n['nfe'] > 1).sum()}; max per group {fe_n['nfe'].max()}")
fe = (fe_raw.sort("tick").group_by(["match_key", "round"])
      .agg(pl.col("t_score").first().cast(pl.Int64).alias("tb"), pl.col("ct_score").first().cast(pl.Int64).alias("cb")))

# ------------------------------------------------------------------ long table
L = (RE.join(fe, on=["match_key", "round"], how="left")
     .join(bo.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")
     .join(M, on="match_key", how="left")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day")))
print(f"long table rounds {L.height}; null map {L['map_name'].null_count()}; null day {L['day'].null_count()}; "
      f"rounds without freeze_end score {L['tb'].null_count()}; rounds with bomb outcome {L['bside'].is_not_null().sum()}")
fe_only = fe.join(RE, on=["match_key", "round"], how="anti").height
print(f"freeze_end (match,round) groups with no round_end row: {fe_only}")
L.write_parquet(OUT / "rounds.parquet")
M.write_parquet(OUT / "matches_header.parquet")
print(f"wrote {OUT / 'rounds.parquet'} and matches_header.parquet")
print("BUILD DONE")
