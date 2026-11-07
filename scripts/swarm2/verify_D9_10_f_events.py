"""verify D9-10 (f): event-level team codes on bomb rows, and which rounds the game marks as 'halftime'.

  F1: for build 9961 bomb rounds, the player_team_code on bomb_defused / bomb_exploded / bomb_planted rows,
      split by whether the round_end label agrees with the bomb outcome. A defuse must be by CT (3) and a
      planted bomb by T (2) if the event codes are side codes. If event codes say CT defused in a round the
      label calls a T win, the label is the inconsistent record.
  F2: per match, which rounds carry round_state phase == 'halftime' (the game's own break marker), grouped
      by pattern, for build 9961 and build 9842.
  F3: whether the inverted sites line up with the halftime-marker rounds (rounds r where r-1 is marked).
ASSUMPTIONS: format and build from the same rules as the other verify scripts; bomb outcome = defused -> CT,
  exploded -> T (same as e2_side_check_r13.py); a round with both outcome events uses defused.
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
HB = {r["match_key"]: r["build_num"] for r in H.iter_rows(named=True)}
B9961 = [m for m, b in HB.items() if b == 9961]
B9842 = [m for m, b in HB.items() if b == 9842]
Bkeys = list(HB.keys())

PI = scan("player_info", ["match_key", "round", "player_id", "team_code"]).filter(pl.col("match_key").is_in(B9961))
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
a = PIu.rename({"round": "r", "team_code": "tA"})
b = PIu.with_columns((pl.col("round") - 1).alias("round")).rename({"round": "r", "team_code": "tB"})
pairs = a.join(b, on=["match_key", "r", "player_id"], how="inner").with_columns((pl.col("tA") != pl.col("tB")).alias("chg"))
bd = (pairs.group_by(["match_key", "r"]).agg(pl.len().alias("nc"), pl.col("chg").mean().alias("fch"))
      .filter(pl.col("nc") >= 3).with_columns((pl.col("fch") >= 0.5).alias("sw")))
first_sw = bd.filter(pl.col("sw")).group_by("match_key").agg(pl.col("r").min().alias("first_switch"))
fmt = (pl.DataFrame({"match_key": B9961}).join(first_sw, on="match_key", how="left")
       .with_columns(pl.when(pl.col("first_switch") == 8).then(pl.lit("sw8"))
                     .when(pl.col("first_switch") == 12).then(pl.lit("sw12"))
                     .otherwise(pl.lit("other_or_none")).alias("fmt")))

BS = scan("bomb_state", ["match_key", "round", "event_type", "player_team_code", "player_id"]).filter(
    pl.col("match_key").is_in(B9961))
RE = scan("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    subset=["match_key", "round"], keep="first").filter(pl.col("match_key").is_in(B9961))

ev = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded", "bomb_planted"]))
evs = ev.group_by(["match_key", "round", "event_type"]).agg(pl.col("player_team_code").first().alias("ptc"),
                                                            pl.len().alias("n_rows"))
outc = (ev.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
outc = outc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3))
                         .otherwise(pl.lit(2)).alias("bomb_side"))
J = (outc.join(RE, on=["match_key", "round"], how="inner").join(fmt, on="match_key", how="left")
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm")))
J = J.join(evs.filter(pl.col("event_type") == "bomb_defused").select(["match_key", "round", "ptc"])
           .rename({"ptc": "defuse_team"}), on=["match_key", "round"], how="left")
J = J.join(evs.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round", "ptc"])
           .rename({"ptc": "explode_team"}), on=["match_key", "round"], how="left")
J = J.join(evs.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round", "ptc"])
           .rename({"ptc": "plant_team"}), on=["match_key", "round"], how="left")

print("== F1 event-level team codes on bomb rows, build 9961, by format and round (sw8 r9, sw12 r13/25/28, other)")
J = J.with_columns(pl.when((pl.col("fmt") == "sw8") & (pl.col("round") == 9)).then(pl.lit("sw8_r9"))
                   .when((pl.col("fmt") == "sw12") & (pl.col("round") == 13)).then(pl.lit("sw12_r13"))
                   .when((pl.col("fmt") == "sw12") & (pl.col("round") == 25)).then(pl.lit("sw12_r25"))
                   .when((pl.col("fmt") == "sw12") & (pl.col("round") == 28)).then(pl.lit("sw12_r28"))
                   .when((pl.col("fmt") == "sw12") & (pl.col("round") == 9)).then(pl.lit("sw12_r9"))
                   .otherwise(pl.lit("other")).alias("site"))
print(J.group_by(["site", "mm", "defuse_team"]).len().sort(["site", "mm", "defuse_team"]).rows())
print("   exploded-round event team codes:")
print(J.group_by(["site", "mm", "explode_team"]).len().sort(["site", "mm", "explode_team"]).rows())
print("   planted-round planter team codes:")
print(J.group_by(["site", "mm", "plant_team"]).len().sort(["site", "mm", "plant_team"]).rows())

# ---- F2: halftime marker rounds per match
RS = scan("round_state", ["match_key", "round", "phase"])
ht_rows = (RS.filter(pl.col("phase") == "halftime").select(["match_key", "round"]).unique()
           .filter(pl.col("match_key").is_in(Bkeys)))
ht_by = ht_rows.sort("round").group_by("match_key", maintain_order=True).agg(pl.col("round").alias("ht_rounds"))
ht_by = ht_by.with_columns(pl.col("ht_rounds").list.eval(pl.element().cast(pl.String)).list.join(",").alias("ht_pat"))
HT = pl.DataFrame({"match_key": Bkeys}).join(ht_by.select(["match_key", "ht_pat"]), on="match_key", how="left")
HT = HT.with_columns(pl.col("ht_pat").fill_null("none"), pl.col("match_key").replace_strict(HB, default=None).alias("build"))
print("\n== F2 halftime-marker round pattern per match (round_state phase=='halftime'), by build (top 12)")
print(HT.group_by(["build", "ht_pat"]).len().sort("len", descending=True).head(12).rows())

# ---- F3: do the inverted sites line up with the rounds after a halftime marker?
print("\n== F3 build 9961: label inverted (bomb mismatch) vs halftime marker on previous round")
HT_MAP = {r["match_key"]: r["ht_pat"] for r in HT.iter_rows(named=True)}
HT_SET = {m: (set(int(x) for x in p.split(",")) if p not in ("none", None, "") else set())
          for m, p in HT_MAP.items()}
J2 = J.with_columns(
    pl.struct(["match_key", "round"]).map_elements(
        lambda s: ((s["round"] - 1) in HT_SET.get(s["match_key"], set())), return_dtype=pl.Boolean)
    .alias("prev_is_halftime"))
print(J2.group_by(["site", "prev_is_halftime", "mm"]).len().sort(["site", "prev_is_halftime", "mm"]).rows())
