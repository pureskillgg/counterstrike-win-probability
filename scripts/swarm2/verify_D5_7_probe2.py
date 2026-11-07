"""D5-7 verifier, probe 2 (appends to logs/swarm2/verify_D5_7.log; reads only the 2024 tome on G:).

Question: how wide is the regime-A round-13 disagreement between round_end winner and the
freeze-end score delta, and is it tied to bomb-ending rounds, reason code, or map?
Run after verify_D5_7_regime.py: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_7_probe2.py
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


out("\n\n# ===== probe 2 (verify_D5_7_probe2.py) =====")
hfiles = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
H = (pl.concat([pl.read_parquet(f, columns=["key", "map_name"]) for f in hfiles], how="vertical_relaxed")
     .rename({"key": "match_key"})
     .with_columns(day=pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1)))

RS = (scan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type").is_in(["round_announce_match_start", "cs_win_panel_round", "round_freeze_end"]))
      .collect())
ras = RS.filter(pl.col("event_type") == "round_announce_match_start").group_by("match_key").agg(pl.len().alias("n_ras"))
MK = H.join(ras, on="match_key", how="left").with_columns(
    n_ras=pl.col("n_ras").fill_null(0)).with_columns(
    regime=pl.when(pl.col("n_ras") > 0).then(pl.lit("A")).otherwise(pl.lit("B")))

FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t_fe"),
                                           pl.col("ct_score").first().alias("ct_fe")))
FEN = FE.select(["match_key", (pl.col("round") - 1).alias("round"),
                 pl.col("t_fe").alias("t_next"), pl.col("ct_fe").alias("ct_next")])

RE_raw = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                   "player_count"]).collect()
RE2 = (RE_raw.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("winner_team_code").first().alias("winner"),
    pl.col("win_reason_code").first().alias("reason"),
    pl.col("player_count").first().alias("pc")))

BSc = (scan("bomb_state").select(["match_key", "round", "tick", "event_type"])
       .filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).collect())
OUT_BS = (BSc.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
          .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
          .with_columns(bomb_side=pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2),
                        bomb_outcome=pl.lit(True)))
PLANT = BSc.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique().with_columns(
    planted=pl.lit(True))

PAIR = (FE.join(FEN, on=["match_key", "round"], how="inner")
        .join(RE2, on=["match_key", "round"], how="left")
        .join(OUT_BS.select(["match_key", "round", "bomb_side", "bomb_outcome"]), on=["match_key", "round"], how="left")
        .join(PLANT, on=["match_key", "round"], how="left")
        .join(MK.select(["match_key", "regime", "day", "map_name"]), on="match_key", how="left")
        .with_columns(dT=pl.col("t_next") - pl.col("t_fe"), dCT=pl.col("ct_next") - pl.col("ct_fe"),
                      bomb_outcome=pl.col("bomb_outcome").fill_null(False),
                      planted=pl.col("planted").fill_null(False)))
PAIR = PAIR.with_columns(fe_side=pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(2)
                         .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(3).otherwise(pl.lit(None)))
PAIR = PAIR.with_columns(
    cat=pl.when(pl.col("winner").is_null()).then(pl.lit("winner_null"))
    .when(pl.col("fe_side").is_null()).then(pl.lit("fe_not_1_0"))
    .when(pl.col("winner") == pl.col("fe_side")).then(pl.lit("agree"))
    .otherwise(pl.lit("disagree")))

out("\n== 9. round-by-round disagreement breakdown: reason code, bomb outcome, planted ==")
out("(cat: agree / disagree = round_end winner vs freeze-end delta, side semantics)")
sel = PAIR.filter(pl.col("round").is_in([9, 12, 13, 25, 27, 28]) & pl.col("regime").is_in(["A", "B"]))
show_tbl = (sel.group_by(["regime", "round", "reason", "bomb_outcome", "planted", "cat"])
            .agg(pl.len().alias("pairs")).sort(["regime", "round", "reason", "bomb_outcome", "planted", "cat"]))
out(str(show_tbl))

out("\n== 9b. round 13 regime A: disagree vs agree by late/early and by map ==")
r13 = PAIR.filter((pl.col("round") == 13) & (pl.col("regime") == "A"))
out(str(r13.group_by(["day", "cat"]).agg(pl.len().alias("pairs")).sort(["day", "cat"])))
out(str(r13.group_by(["map_name", "cat"]).agg(pl.len().alias("pairs")).sort(["map_name", "cat"])))

out("\n== 10. freeze-end totals: t_score + ct_score at freeze-end of round k should equal k-1 ==")
FEchk = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
         .group_by(["match_key", "round"]).agg((pl.col("t_score").first() + pl.col("ct_score").first()).alias("tot"))
         .with_columns(ok=(pl.col("tot") == pl.col("round") - 1)))
FEchk = FEchk.join(MK.select(["match_key", "regime"]), on="match_key", how="left")
out(str(FEchk.group_by(["regime", "ok"]).agg(pl.len().alias("rounds")).sort(["regime", "ok"])))
out("round-1 freeze-end rows present by regime (count of (match, round 1) rows):")
out(str(FEchk.filter(pl.col("round") == 1).group_by("regime").agg(pl.len().alias("fe1_rows"))))
out("round-2 freeze-end totals not equal to 1 (regime, count):")
out(str(FEchk.filter((pl.col("round") == 2) & (~pl.col("ok"))).group_by("regime").agg(pl.len().alias("n"))))

out("\n== 11. map check for round 13: bomb outcomes by map and regime A round-13 disagreement by map ==")
out(str(OUT_BS.join(MK.select(["match_key", "map_name", "regime"]), on="match_key", how="left")
        .group_by(["map_name", "regime"]).agg(pl.len().alias("bomb_outcome_rows"))
        .sort(["map_name", "regime"])))
out(str(MK.group_by(["map_name", "regime"]).agg(pl.len().alias("matches")).sort(["map_name", "regime"])))
out("regime A late, round 13, pairs by map and cat:")
out(str(PAIR.filter((pl.col("round") == 13) & (pl.col("regime") == "A") & pl.col("day").is_in(LATE))
        .group_by(["map_name", "cat"]).agg(pl.len().alias("pairs")).sort(["map_name", "cat"])))

out("\n== 12. match-level round-13 summary (regime A and B) ==")
m13 = (PAIR.filter(pl.col("round") == 13)
       .group_by(["regime", "cat"]).agg(pl.len().alias("matches")).sort(["regime", "cat"]))
out(str(m13))
m13b = (PAIR.filter(pl.col("round") == 13)
        .group_by(["regime", "bomb_outcome", "cat"]).agg(pl.len().alias("matches")).sort(["regime", "bomb_outcome", "cat"]))
out("split by bomb outcome present at round 13:")
out(str(m13b))

out("\n== end probe 2 ==")
FH.close()
