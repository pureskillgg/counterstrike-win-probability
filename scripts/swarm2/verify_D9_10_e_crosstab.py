"""verify D9-10 (e): composition, round-25 mechanism, bomb-shift alternative, and build/day scope.

Uses the same switch-format definition as verify_D9_10_c_schedule.py / verify_D9_10_d_bomb.py.
  E1: death-test flip status (R/F/N, non-bomb, pc20) cross-tabbed by switch format (sw8 vs sw12 vs none).
      Tests whether the 88.9% round-13 share is a composition effect of the 8-format matches.
  E2: round 25 mechanism: for bomb planters in round 25, was the same player on T in round 24?
      Also round_state phase by round for sw12 matches (does the game mark a phase change at 24/25 or 27/28?).
  E3: bomb-shift alternative: does the round-13 bomb outcome (sw12) match the label of round 12 or 14 better?
  E4: scope: bomb mismatch by build x format x round, and by day for sw12 round 13.
  E5: recount "F followed immediately by R" matches (claim: 1,631) and total match count with any scored round.
ASSUMPTIONS: as in verify_D9_10_c_schedule.py (switch = >=50% team change, >=3 common players).
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


def scan(name, cols, keep_matches=None):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    missing = [c for c in cols if c not in have]
    if missing:
        raise SystemExit(f"[{name}] missing columns {missing}")
    lf = lf.select(cols)
    if keep_matches is not None:
        lf = lf.filter(pl.col("match_key").is_in(keep_matches))
    return lf.collect()


H = scan("header", ["key", "build_num", "match_date", "game_directory"]).rename({"key": "match_key"})
H = H.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
ALLM = H["match_key"].to_list()

PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
a = PIu.rename({"round": "r", "team_code": "tA"})
b = PIu.with_columns((pl.col("round") - 1).alias("round")).rename({"round": "r", "team_code": "tB"})
pairs = a.join(b, on=["match_key", "r", "player_id"], how="inner").with_columns((pl.col("tA") != pl.col("tB")).alias("chg"))
bd = (pairs.group_by(["match_key", "r"]).agg(pl.len().alias("nc"), pl.col("chg").mean().alias("fch"))
      .filter(pl.col("nc") >= 3).with_columns((pl.col("fch") >= 0.5).alias("sw")))
first_sw = bd.filter(pl.col("sw")).group_by("match_key").agg(pl.col("r").min().alias("first_switch"))
fmt = (pl.DataFrame({"match_key": ALLM}).join(first_sw, on="match_key", how="left")
       .with_columns(pl.when(pl.col("first_switch") == 8).then(pl.lit("sw8"))
                     .when(pl.col("first_switch") == 12).then(pl.lit("sw12"))
                     .when(pl.col("first_switch").is_null()).then(pl.lit("no_switch_seen"))
                     .otherwise(pl.lit("sw_other")).alias("fmt")))
H = H.join(fmt.select(["match_key", "fmt"]), on="match_key", how="left")
H9 = H.filter(pl.col("build_num") == 9961)
B9 = H9["match_key"].to_list()

RE = scan("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    subset=["match_key", "round"], keep="first")
RE = RE.join(H.select(["match_key", "build_num", "fmt", "day"]), on="match_key", how="left")

DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"])
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dead = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead")))
tc = (PIu.group_by(["match_key", "round"])
      .agg((pl.col("team_code") == 2).sum().alias("n_T"), (pl.col("team_code") == 3).sum().alias("n_CT")))
BS = scan("bomb_state", ["match_key", "round", "event_type", "player_team_code", "player_id"])
bomb_rounds = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("bomb_any")))
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
outc = outc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3))
                         .otherwise(pl.lit(2)).alias("bomb_side"))

T = (RE.join(tc, on=["match_key", "round"], how="left")
       .join(dead, on=["match_key", "round"], how="left")
       .join(bomb_rounds, on=["match_key", "round"], how="left")
       .with_columns(pl.col("bomb_any").fill_null(False), pl.col("T_dead").fill_null(0),
                     pl.col("CT_dead").fill_null(0)))
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

# ---------- E1: flip status by switch format
print("== E1 death-test status by switch format, pc20 build 9961, selected rounds (R/F/N/-)")
for rr in [9, 13, 25, 28]:
    sub = S.filter(pl.col("round") == rr)
    print(f"  round {rr}:")
    print("    ", sub.group_by(["fmt", "c"]).len().sort(["fmt", "c"]).rows())
sc13 = S.filter((pl.col("round") == 13) & pl.col("c").is_in(["R", "F"]))
print("  round 13 scored, flip share within sw12 format:",
      round(sc13.filter(pl.col("fmt") == "sw12").select((pl.col("c") == "F").mean()).item(), 4),
      "| n sw12 scored:", sc13.filter(pl.col("fmt") == "sw12").height)
sc9 = S.filter((pl.col("round") == 9) & pl.col("c").is_in(["R", "F"]))
print("  round 9 scored, flip share within sw8 format:",
      round(sc9.filter(pl.col("fmt") == "sw8").select((pl.col("c") == "F").mean()).item(), 4),
      "| n sw8 scored:", sc9.filter(pl.col("fmt") == "sw8").height)
print("  round 9 scored, flip share within sw12 format:",
      round(sc9.filter(pl.col("fmt") == "sw12").select((pl.col("c") == "F").mean()).item(), 4),
      "| n sw12 scored:", sc9.filter(pl.col("fmt") == "sw12").height)

# ---------- E5: recount claimed match-level numbers
pat = (S.sort("round").group_by("match_key", maintain_order=True)
       .agg(pl.col("round").cast(pl.String).alias("rs"), pl.col("c").alias("cs")))
pat = pat.with_columns(pl.struct(["rs", "cs"]).map_elements(
    lambda s: " ".join(f"{r}{c}" for r, c in zip(s["rs"], s["cs"])), return_dtype=pl.String).alias("pattern"))
print("\n== E5 recount")
print("  matches with any row in the round set:", pat.height)
print("  matches with any scored (R/F) round:",
      pat.filter(pl.col("pattern").str.contains(r"\d[RF]")).height)
print("  matches with a flipped round followed immediately by an R round (regex '\\dF \\d+R'):",
      pat.filter(pl.col("pattern").str.contains(r"\dF \d+R")).height)
nf = pat.with_columns(pl.col("pattern").str.count_matches(r"\dF").alias("nf")).group_by("nf").agg(pl.len().alias("n"))
print("  matches by count of F tokens:", nf.sort("nf").rows())

# ---------- E2: round 25 mechanism (planter team in 24 vs 25; phase by round)
PLANT = (BS.filter((pl.col("event_type") == "bomb_planted") & pl.col("match_key").is_in(B9))
         .select(["match_key", "round", "player_id", "player_team_code"]))
print("\n== E2 round 25 mechanism (build 9961 bomb planters)")
p25 = PLANT.filter(pl.col("round") == 25).rename({"player_team_code": "plant_t25"})
p24_info = PIu.filter((pl.col("round") == 24)).select(["match_key", "player_id", "team_code"]).rename({"team_code": "info_t24"})
p25_info = PIu.filter((pl.col("round") == 25)).select(["match_key", "player_id", "team_code"]).rename({"team_code": "info_t25"})
p25x = (p25.join(p24_info, on=["match_key", "player_id"], how="left")
           .join(p25_info, on=["match_key", "player_id"], how="left"))
print("  round-25 planters: (plant team, info team r24, info team r25) counts:",
      p25x.group_by(["plant_t25", "info_t24", "info_t25"]).len().sort(["plant_t25", "info_t24", "info_t25"]).rows())
p28 = PLANT.filter(pl.col("round") == 28).rename({"player_team_code": "plant_t28"})
p27_info = PIu.filter(pl.col("round") == 27).select(["match_key", "player_id", "team_code"]).rename({"team_code": "info_t27"})
p28_info = PIu.filter(pl.col("round") == 28).select(["match_key", "player_id", "team_code"]).rename({"team_code": "info_t28"})
p28x = (p28.join(p27_info, on=["match_key", "player_id"], how="left").join(p28_info, on=["match_key", "player_id"], how="left"))
print("  round-28 planters: (plant team, info r27, info r28):",
      p28x.group_by(["plant_t28", "info_t27", "info_t28"]).len().sort(["plant_t28", "info_t27", "info_t28"]).rows())

RS = scan("round_state", ["match_key", "round", "phase"]).filter(pl.col("match_key").is_in(B9))
ph = (RS.join(H.select(["match_key", "fmt"]), on="match_key", how="left")
      .filter(pl.col("fmt") == "sw12").group_by(["round", "phase"]).len())
print("  sw12 matches: phase counts for rounds 11-14 and 23-29:")
print(ph.filter(pl.col("round").is_in([11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29])).sort(["round", "phase"]).rows())

# ---------- E3: shift alternative: bomb outcome of round r vs label of r-1 and r+1 (sw12, r=13; sw8 r=9)
LAB = RE.select(["match_key", "round", "winner_team_code"])
for tgt, fm in [(13, "sw12"), (9, "sw8"), (25, "sw12"), (28, "sw12")]:
    B_ = outc.join(H.select(["match_key", "fmt", "build_num"]), on="match_key", how="inner").filter(
        (pl.col("fmt") == fm) & (pl.col("round") == tgt) & (pl.col("build_num") == 9961))
    cur = B_.join(LAB, on=["match_key", "round"], how="left")
    prv = B_.with_columns((pl.col("round") - 1).alias("round")).join(LAB, on=["match_key", "round"], how="left")
    nxt = B_.with_columns((pl.col("round") + 1).alias("round")).join(LAB, on=["match_key", "round"], how="left")
    print(f"\n== E3 shift test {fm} round {tgt}: n bomb rounds={cur.height}")
    print("   bomb side == label(r):", int((cur["bomb_side"] == cur["winner_team_code"]).sum()),
          "| bomb side == label(r-1):", int((prv["bomb_side"] == prv["winner_team_code"]).sum()),
          "| bomb side == label(r+1):", int((nxt["bomb_side"] == nxt["winner_team_code"]).sum()))

# ---------- E4: scope by build x format x round (bomb mismatch), and by day for sw12 round 13
MM = (outc.join(RE.select(["match_key", "round", "winner_team_code", "build_num", "fmt", "day"]),
                on=["match_key", "round"], how="inner")
      .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
                    pl.when(pl.col("round").is_in([9, 13, 25, 28])).then(pl.col("round").cast(pl.String))
                    .otherwise(pl.lit("other")).alias("rbin")))
print("\n== E4 bomb mismatch by build x format x round bin")
print(MM.group_by(["build_num", "fmt", "rbin"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm_n"),
                                                   pl.col("mm").mean().round(4).alias("rate"))
      .sort(["build_num", "fmt", "rbin"]).rows())
print("\n== E4 sw12 round 13 bomb mismatch by day (build 9961 and all builds)")
r13 = MM.filter((pl.col("round") == 13) & (pl.col("fmt") == "sw12"))
print(r13.group_by(["build_num", "day"]).agg(pl.len().alias("n"), pl.col("mm").mean().round(3).alias("rate"))
      .sort(["build_num", "day"]).rows())
print("\n== E4 2024 round-13 bomb rows overall (all builds, all formats):",
      MM.filter(pl.col("round") == 13).group_by(["build_num", "fmt"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm_n"))
      .sort(["build_num", "fmt"]).rows())
