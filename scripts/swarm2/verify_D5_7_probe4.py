"""D5-7 verifier, probe 4 (appends to logs/swarm2/verify_D5_7.log; reads only the 2024 tome on G:).

Third in-tome source for round winners: round_officially_ended and cs_win_panel_round carry score fields.
  - ROE-FE delta: round_officially_ended(k) scores minus round_freeze_end(k) scores (side-based winner of round k).
  - Round 1 in build 9842 (no round_freeze_end for round 1, no round_end): round_officially_ended(1) scores vs 0-0 start.
  - cs_win_panel_round(1) scores in build 9842.
Compares each with the recorded round_end winner and with the bomb outcome (round 13).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_7_probe4.py
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


out("\n\n# ===== probe 4 (verify_D5_7_probe4.py): round_officially_ended / cs_win_panel_round as third source =====")
hfiles = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
H = (pl.concat([pl.read_parquet(f, columns=["key", "build_num"]) for f in hfiles], how="vertical_relaxed")
     .rename({"key": "match_key"}))

EV = ["round_announce_match_start", "round_freeze_end", "round_officially_ended", "cs_win_panel_round"]
RS = (scan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type").is_in(EV)).collect())
MK = H.join(RS.filter(pl.col("event_type") == "round_announce_match_start").select("match_key").unique()
            .with_columns(ras=pl.lit(True)), on="match_key", how="left").with_columns(
    regime=pl.when(pl.col("ras").fill_null(False)).then(pl.lit("A")).otherwise(pl.lit("B")))


def first_by_round(et, prefix):
    return (RS.filter(pl.col("event_type") == et).sort(["match_key", "round", "tick"])
            .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias(f"t_{prefix}"),
                                                 pl.col("ct_score").first().alias(f"ct_{prefix}"),
                                                 pl.len().alias(f"n_{prefix}")))


FE = first_by_round("round_freeze_end", "fe")
ROE = first_by_round("round_officially_ended", "roe")
CWP = first_by_round("cs_win_panel_round", "cwp")

RE_raw = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
RE1 = (RE_raw.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("winner_team_code").first().alias("winner"), pl.col("player_count").first().alias("pc")))

BSc = (scan("bomb_state").select(["match_key", "round", "event_type"])
       .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).collect())
OUT_BS = (BSc.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
          .with_columns(bomb_side=pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2)))


def side(dT, dCT):
    return (pl.when((dT == 1) & (dCT == 0)).then(2).when((dCT == 1) & (dT == 0)).then(3).otherwise(pl.lit(None)))


# ---- A. ROE-FE delta vs recorded round_end winner, all rounds with both events ----
D = (FE.join(ROE, on=["match_key", "round"], how="inner")
     .with_columns(dT=pl.col("t_roe") - pl.col("t_fe"), dCT=pl.col("ct_roe") - pl.col("ct_fe"))
     .with_columns(roe_side=side(pl.col("dT"), pl.col("dCT")))
     .join(RE1.select(["match_key", "round", "winner"]), on=["match_key", "round"], how="left")
     .join(MK.select(["match_key", "regime", "build_num"]), on="match_key", how="left")
     .with_columns(cat=pl.when(pl.col("winner").is_null()).then(pl.lit("winner_null"))
                   .when(pl.col("roe_side").is_null()).then(pl.lit("roe_not_1_0"))
                   .when(pl.col("winner") == pl.col("roe_side")).then(pl.lit("agree"))
                   .otherwise(pl.lit("disagree"))))
out("== 13a. ROE-FE delta winner vs round_end winner, by build and round label ==")
out(str(D.group_by(["build_num", "round", "cat"]).agg(pl.len().alias("n"))
        .pivot(on="cat", index=["build_num", "round"], values="n", aggregate_function="sum").fill_null(0)
        .sort(["build_num", "round"])))

# ---- B. round 13 with bomb: ROE side vs bomb side vs round_end winner ----
R13 = (D.filter(pl.col("round") == 13).join(OUT_BS.select(["match_key", "round", "bomb_side"]),
                                            on=["match_key", "round"], how="inner"))
out("== 13b. round 13 bomb rows: ROE-delta side vs bomb side vs round_end winner ==")
out(str(R13.group_by(["build_num", "winner", "bomb_side", "roe_side"]).agg(pl.len().alias("rows"))
        .sort(["build_num", "winner", "bomb_side", "roe_side"])))

# ---- C. round 1: build 9842 has no round_end and no round-1 freeze-end; use ROE(1) and CWP(1) directly ----
r1 = (MK.select(["match_key", "regime", "build_num"])
      .join(ROE.filter(pl.col("round") == 1).select(["match_key", "t_roe", "ct_roe"]), on="match_key", how="left")
      .join(CWP.filter(pl.col("round") == 1).select(["match_key", pl.col("t_cwp"), pl.col("ct_cwp")]),
            on="match_key", how="left")
      .join(FE.filter(pl.col("round") == 2).select(["match_key", pl.col("t_fe").alias("t2"), pl.col("ct_fe").alias("ct2")]),
            on="match_key", how="left")
      .join(RE1.filter(pl.col("round") == 1).select(["match_key", pl.col("winner").alias("w1")]), on="match_key", how="left"))
r1 = r1.with_columns(
    roe1_side=side(pl.col("t_roe"), pl.col("ct_roe")),
    cwp1_side=side(pl.col("t_cwp"), pl.col("ct_cwp")),
    fe2_side=side(pl.col("t2"), pl.col("ct2")))
out("== 13c. round-1 winner sources by regime (roe1 = ROE(1) scores vs 0-0; cwp1 = cs_win_panel_round(1) scores; fe2 = freeze-end scores at round 2) ==")
out(str(r1.group_by("regime").agg(
    pl.len().alias("matches"),
    pl.col("w1").is_not_null().sum().alias("recorded_w1"),
    pl.col("roe1_side").is_not_null().sum().alias("roe1_defined"),
    pl.col("cwp1_side").is_not_null().sum().alias("cwp1_defined"),
    pl.col("fe2_side").is_not_null().sum().alias("fe2_defined"),
    ((pl.col("roe1_side") == pl.col("w1"))).sum().alias("roe1_eq_w1"),
    ((pl.col("fe2_side") == pl.col("w1"))).sum().alias("fe2_eq_w1"),
    ((pl.col("cwp1_side") == pl.col("fe2_side"))).sum().alias("cwp1_eq_fe2"),
    ((pl.col("roe1_side") == pl.col("fe2_side"))).sum().alias("roe1_eq_fe2")).sort("regime")))
out("regime B: ROE(1) round-1 score totals (should be 1):")
out(str(r1.filter(pl.col("regime") == "B").group_by(
    (pl.col("t_roe") + pl.col("ct_roe")).alias("tot")).agg(pl.len().alias("matches")).sort("tot")))

out("\n== end probe 4 ==")
FH.close()
