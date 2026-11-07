"""D5-7 verifier, probe 3 (appends to logs/swarm2/verify_D5_7.log; reads only the 2024 tome on G:).

Alternative explanation under test: the regime (round_announce_match_start present) is a server/game-build
split (header build_num, game_directory, demo_version_guid), not a per-match recording marker. Also checks
whether round-13 disagreement follows build_num or regime.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_7_build.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
LOG = ROOT / "logs" / "swarm2" / "verify_D5_7.log"
FH = open(LOG, "a", encoding="utf-8", newline="\n")

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_width_chars(220)

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]


def out(*parts):
    s = " ".join(str(p) for p in parts)
    print(s, flush=True)
    FH.write(s + "\n")
    FH.flush()


def chfiles(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no dataframe files for channel {ch}")
    return fs


def scan(ch):
    return pl.concat([pl.scan_parquet(f) for f in chfiles(ch)], how="vertical_relaxed")


out("\n\n# ===== probe 3 (verify_D5_7_build.py): build / game-directory alternative =====")
hfiles = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
H = (pl.concat([pl.read_parquet(f, columns=["key", "build_num", "game_directory", "demo_version_guid",
                                            "server_name", "match_date", "map_name"]) for f in hfiles],
               how="vertical_relaxed")
     .rename({"key": "match_key"})
     .with_columns(day=pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1)))

RS = (scan("round_state").select(["match_key", "round", "event_type", "t_score", "ct_score", "tick"])
      .filter(pl.col("event_type").is_in(["round_announce_match_start", "round_freeze_end"])).collect())
ras = (RS.filter(pl.col("event_type") == "round_announce_match_start").group_by("match_key")
       .agg(pl.len().alias("n_ras")))
MK = H.join(ras, on="match_key", how="left").with_columns(
    n_ras=pl.col("n_ras").fill_null(0)).with_columns(
    regime=pl.when(pl.col("n_ras") > 0).then(pl.lit("A")).otherwise(pl.lit("B")))

out("== 3a. build_num x regime (all 2024 matches) ==")
out(str(MK.group_by(["build_num", "regime"]).agg(pl.len().alias("matches")).sort(["build_num", "regime"])))
out("== 3b. game_directory x regime ==")
out(str(MK.group_by(["game_directory", "regime"]).agg(pl.len().alias("matches")).sort(["game_directory", "regime"])))
out("== 3c. demo_version_guid x regime (distinct guid count per regime) ==")
out(str(MK.group_by(["regime"]).agg(pl.col("demo_version_guid").n_unique().alias("distinct_guids"),
                                    pl.col("build_num").n_unique().alias("distinct_builds"))))
out("== 3d. build_num x day x regime (late and early windows) ==")
out(str(MK.group_by(["day", "build_num", "regime"]).agg(pl.len().alias("matches")).sort(["day", "build_num", "regime"])))

# round-13 outcome and freeze-end delta, joined to build
RE_raw = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
RE2 = (RE_raw.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("winner_team_code").first().alias("winner"), pl.col("player_count").first().alias("pc")))
BSc = (scan("bomb_state").select(["match_key", "round", "tick", "event_type"])
       .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).collect())
OUT_BS = (BSc.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
          .with_columns(bomb_side=pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2)))
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t_fe"),
                                           pl.col("ct_score").first().alias("ct_fe")))
FEN = FE.select(["match_key", (pl.col("round") - 1).alias("round"),
                 pl.col("t_fe").alias("t_next"), pl.col("ct_fe").alias("ct_next")])
PAIR = (FE.join(FEN, on=["match_key", "round"], how="inner")
        .join(RE2, on=["match_key", "round"], how="left")
        .join(OUT_BS.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")
        .join(MK.select(["match_key", "regime", "build_num", "day"]), on="match_key", how="left")
        .with_columns(dT=pl.col("t_next") - pl.col("t_fe"), dCT=pl.col("ct_next") - pl.col("ct_fe")))
PAIR = PAIR.with_columns(fe_side=pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(2)
                         .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(3).otherwise(pl.lit(None)))
PAIR = PAIR.with_columns(
    cat=pl.when(pl.col("winner").is_null()).then(pl.lit("winner_null"))
    .when(pl.col("fe_side").is_null()).then(pl.lit("fe_not_1_0"))
    .when(pl.col("winner") == pl.col("fe_side")).then(pl.lit("agree"))
    .otherwise(pl.lit("disagree")))

out("\n== 3e. round 13: freeze-end-delta agreement by build_num and regime ==")
out(str(PAIR.filter(pl.col("round") == 13).group_by(["build_num", "regime", "cat"]).agg(pl.len().alias("pairs"))
        .sort(["build_num", "regime", "cat"])))
out("== 3f. round 13 bomb-outcome rows: mismatch by build_num and regime (pc==20) ==")
R13b = (RE2.filter((pl.col("round") == 13) & (pl.col("pc") == 20))
        .join(OUT_BS.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="inner")
        .join(MK.select(["match_key", "regime", "build_num", "day"]), on="match_key", how="left")
        .with_columns(mismatch=pl.col("bomb_side") != pl.col("winner")))
out(str(R13b.group_by(["build_num", "regime", "mismatch"]).agg(pl.len().alias("rows")).sort(["build_num", "regime", "mismatch"])))

out("== 3g. round 9, 25, 27, 28 freeze-end disagreement in regime A by build_num ==")
out(str(PAIR.filter(pl.col("round").is_in([9, 25, 27, 28]))
        .group_by(["round", "regime", "build_num", "cat"]).agg(pl.len().alias("pairs"))
        .sort(["round", "regime", "build_num", "cat"])))

out("\n== end probe 3 ==")
FH.close()
