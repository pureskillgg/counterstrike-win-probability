"""verify_D9_10 final: self-contained reproduction of every number used in the verdict.

Reads the 2024-02-g tome directly (read-only, projected columns, one table at a time). Writes nothing except stdout.
Population for the claim: build 9961, player_count 20, roster present, rounds 8-14 and 24-30, non-bomb rounds.
Ground truth used for inversion checks (independent of the death test):
  - bomb_defused -> CT (3) won; bomb_exploded -> T (2) won (the bomb-outcome rule from the defect)
  - defuser player_team_code (bomb_defuse) should be 3 (a defuse is a CT action)
  - planter player_team_code (bomb_planted) should be 2 (a plant is a T action)
ASSUMPTION: a team-side change for the match is the first round r in {9, 13} where more than half of the players'
  player_info team_code differs from round r-1 (the same rule as verify_D9_10_04_alt.py).
ASSUMPTION: round_state.phase of the first row of each round is the game's own half marker.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
LABELS = [8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30]


def scan(ch, cols, header=False):
    fd = BASE / "header" if header else BASE / f"basic_{ch}{SUF}"
    lf = pl.scan_parquet(sorted(glob.glob(str(fd / "dataframe_*"))))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in cols if c in have]).collect()


print("=== D9-10 verification: reproduction log (read-only on tome) ===")
H = scan("header", ["key", "build_num"], header=True).rename({"key": "match_key"})
RE = scan("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    subset=["match_key", "round"], keep="first")
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_id", "player_team_code"])
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
          .group_by(["match_key", "round"])
          .agg((pl.col("event_type") == "bomb_defused").any().alias("defused"),
               (pl.col("event_type") == "bomb_exploded").any().alias("exploded")))
planter = (BS.filter(pl.col("event_type") == "bomb_planted").sort("tick")
             .group_by(["match_key", "round"])
             .agg(pl.col("player_id").first().alias("planter_id"),
                  pl.col("player_team_code").first().alias("planter_side")))
DF = scan("bomb_defuse", ["match_key", "round", "player_team_code"])
dsd = (DF.filter(pl.col("player_team_code").is_not_null()).group_by(["match_key", "round"])
         .agg(pl.col("player_team_code").first().alias("defuser_side")))
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last")
tc = PIu.group_by(["match_key", "round"]).agg((pl.col("team_code") == 2).sum().alias("n_T"),
                                              (pl.col("team_code") == 3).sum().alias("n_CT"))
DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"])
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dd = DEu.group_by(["match_key", "round"]).agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
                                              (pl.col("player_team_code") == 3).sum().alias("CT_dead"))
lab = DEu.join(PIu.select(["match_key", "round", "player_id", pl.col("team_code").alias("it")]),
               on=["match_key", "round", "player_id"], how="inner").join(
    H.select(["match_key", "build_num"]), on="match_key", how="left")

T = (RE.join(H, on="match_key", how="left")
       .join(outc, on=["match_key", "round"], how="left")
       .join(planter, on=["match_key", "round"], how="left")
       .join(dsd, on=["match_key", "round"], how="left")
       .join(tc, on=["match_key", "round"], how="left")
       .join(dd, on=["match_key", "round"], how="left")
       .with_columns(pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False),
                     pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0)))
T = T.with_columns((pl.col("defused") | pl.col("exploded")).alias("bomb"),
                   pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
T = T.with_columns((pl.col("bomb") & (pl.col("bomb_side") != pl.col("winner_team_code"))).alias("bomb_mismatch"))

print("\n[0] rounds (all builds):", T.height,
      "| build 9961 matches:", T.filter(pl.col("build_num") == 9961)["match_key"].n_unique(),
      "| build 9842 matches:", T.filter(pl.col("build_num") == 9842)["match_key"].n_unique())
l9961 = lab.filter(pl.col("build_num") == 9961)
print("[0] death rows vs player_info side, build 9961 (player_team_code == team_code):",
      l9961.group_by(pl.col("player_team_code") == pl.col("it")).len().sort("player_team_code").rows())

# ---- per-match side-change round for build 9961 (player_info, >50% of players change over consecutive rounds)
pr = PIu.sort(["match_key", "player_id", "round"]).with_columns(
    pl.col("team_code").shift(1).over(["match_key", "player_id"]).alias("prev_team"),
    pl.col("round").shift(1).over(["match_key", "player_id"]).alias("prev_round"))
pr = pr.filter(pl.col("prev_team").is_not_null() & (pl.col("round") - pl.col("prev_round") == 1))
fc = (pr.with_columns((pl.col("team_code") != pl.col("prev_team")).alias("chg"))
        .group_by(["match_key", "round"]).agg(pl.col("chg").mean().alias("frac")))
fc9961 = fc.join(H.select(["match_key", "build_num"]), on="match_key", how="left").filter(pl.col("build_num") == 9961)
allm = H.filter(pl.col("build_num") == 9961).select("match_key")
half = (fc9961.filter((pl.col("frac") > 0.5) & pl.col("round").is_in([9, 13]))
              .group_by("match_key").agg(pl.col("round").min().alias("halftime_r")))
half = allm.join(half, on="match_key", how="left")
print("\n[1] build 9961 matches:", allm.height,
      "| side change at 9 (halftime after round 8):", half.filter(pl.col("halftime_r") == 9).height,
      "| at 13 (halftime after round 12):", half.filter(pl.col("halftime_r") == 13).height,
      "| neither:", half.filter(pl.col("halftime_r").is_null()).height)
print("[1] build 9961 match-rounds with >50% side change at rounds 20-30:",
      fc9961.filter((pl.col("frac") > 0.5) & pl.col("round").is_between(20, 30)).group_by("round").len().sort("round").rows())

# ---- planter identity: the planter's own player_info code at r-1 and r (physical side change check)
pl_r = planter.select(["match_key", "round", "planter_id"])
pi_r = PIu.select(["match_key", "round", "player_id", "team_code"])
pl_chg = (pl_r.join(pi_r.rename({"team_code": "code_r"}), left_on=["match_key", "round", "planter_id"],
                    right_on=["match_key", "round", "player_id"], how="left")
              .join(pi_r.with_columns((pl.col("round") + 1).alias("round")).rename({"team_code": "code_prev"}),
                    left_on=["match_key", "round", "planter_id"], right_on=["match_key", "round", "player_id"], how="left"))
pl_chg = pl_chg.join(H.select(["match_key", "build_num"]), on="match_key", how="left").filter(
    (pl.col("build_num") == 9961) & pl.col("round").is_in([9, 13, 24, 25, 27, 28]))
print("\n[2] planter own side code, build 9961, (round: code at r-1 -> code at r) counts:")
print(pl_chg.group_by(["round", "code_prev", "code_r"]).len().sort(["round", "code_prev", "code_r"]).rows())

# ---- table with halftime class for build 9961
T9961 = T.filter(pl.col("build_num") == 9961).join(half, on="match_key", how="left")
T9961 = T9961.with_columns(
    pl.when(pl.col("bomb")).then(pl.lit("bomb"))
      .when((pl.col("CT_dead") >= pl.col("n_CT")) & (pl.col("winner_team_code") == 2)).then(pl.lit("R"))
      .when((pl.col("T_dead") >= pl.col("n_T")) & (pl.col("winner_team_code") == 2)).then(pl.lit("F"))
      .when((pl.col("T_dead") >= pl.col("n_T")) & (pl.col("winner_team_code") == 3)).then(pl.lit("R"))
      .when((pl.col("CT_dead") >= pl.col("n_CT")) & (pl.col("winner_team_code") == 3)).then(pl.lit("F"))
      .otherwise(pl.lit("N")).alias("c"))

# ---- A. claim recomputation
CL = T9961.filter((pl.col("player_count") == 20) & pl.col("n_CT").is_not_null() & pl.col("round").is_in(LABELS))
S = CL.filter(pl.col("c") != "bomb")
print("\n[3] CLAIM RECOMPUTATION: build 9961, pc20, roster, labels 8-14/24-30")
print("   matches in scope:", CL["match_key"].n_unique(), "| non-bomb rounds:", S.height,
      "| bomb rounds (not scored):", CL.filter(pl.col("c") == "bomb").height)
print("   label  R+F(denominator)  F    R    N    F/(R+F)")
for rr in LABELS:
    x = S.filter(pl.col("round") == rr)
    f = int((x["c"] == "F").sum())
    r_ = int((x["c"] == "R").sum())
    n_ = int((x["c"] == "N").sum())
    den = f + r_
    rate = (f / den) if den else float("nan")
    print(f"   r{rr:<3}  {den:>15}  {f:>4} {r_:>4} {n_:>4}   {rate:.4f}")
mF = S.filter(pl.col("c") == "F").group_by("match_key").agg(pl.col("round").alias("Fr"))
f9 = mF.filter(pl.col("Fr").list.contains(9)).height
f13 = mF.filter(pl.col("Fr").list.contains(13)).height
both = mF.filter(pl.col("Fr").list.contains(9) & pl.col("Fr").list.contains(13)).height
print(f"   matches with flip at 9: {f9} | flip at 13: {f13} | both: {both} | 9-flip without 13-flip: {f9 - both}")
print("   matches by number of flipped rounds:",
      mF.with_columns(pl.col("Fr").list.len().alias("n")).group_by("n").len().sort("n").rows())

# ---- B. flips split by per-match halftime
print("\n[4] non-bomb status of round 9 and round 13 split by the match's side-change round (build 9961, pc20)")
for rr in [9, 13]:
    x = S.filter(pl.col("round") == rr).with_columns(pl.col("halftime_r").fill_null(-1).alias("ht"))
    print(f"   round {rr}: (halftime_r, status, count)", x.group_by(["ht", "c"]).len().sort(["ht", "c"]).rows())

# ---- C. bomb-verified inversion by label and halftime class
B9961 = T9961.filter(pl.col("bomb"))
print("\n[5] bomb-verified winner inversion (recorded winner != bomb side), build 9961, all player counts")
bm = (B9961.with_columns(pl.col("halftime_r").fill_null(-1).alias("ht"))
           .group_by(["round", "ht"]).agg(pl.len().alias("n"), pl.col("bomb_mismatch").sum().alias("mm"))
           .sort(["round", "ht"]))
print("   (round, halftime_r) -> n bomb rounds, mismatches (mismatch rows only, plus labels 12-14 and 24-28 for context)")
for row in bm.iter_rows(named=True):
    if row["mm"] or row["round"] in (9, 12, 13, 14, 24, 25, 26, 27, 28):
        print(f"   r{row['round']:<3} ht={row['ht']:<3} n={row['n']:>4} mismatch={row['mm']:>4}")
print("   total bomb rounds build 9961:", B9961.height, "| mismatches:", int(B9961["bomb_mismatch"].sum()),
      "| rounds with any mismatch:", sorted(set(B9961.filter(pl.col("bomb_mismatch"))["round"].to_list())))
B9842 = T.filter((pl.col("build_num") == 9842) & pl.col("bomb"))
print("   control build 9842: bomb rounds", B9842.height, "| mismatches:", int(B9842["bomb_mismatch"].sum()))
dfz = B9961.filter(pl.col("defused") & pl.col("defuser_side").is_not_null())
print("   defused rounds with a defuser side (build 9961):", dfz.height,
      "| defuser side counts:", dfz.group_by("defuser_side").len().sort("defuser_side").rows())
print("   defused rounds at 25/28 with recorded winner 2 (T):",
      dfz.filter(pl.col("round").is_in([25, 28]) & (pl.col("winner_team_code") == 2)).height,
      "of", dfz.filter(pl.col("round").is_in([25, 28])).height)
print("   planter side on bomb rounds (build 9961):", B9961.group_by("planter_side").len().sort("planter_side").rows())

# ---- D. death-row label agreement (build 9961)
print("\n[6] death-row label vs player_info, build 9961: (agree, disagree) =",
      l9961.group_by(pl.col("player_team_code") == pl.col("it")).len().sort("player_team_code").rows())

# ---- E. fix evaluation on bomb rounds (build 9961)
def errors_after(flip_expr):
    x = B9961.with_columns(flip_expr.fill_null(False).alias("flip"))
    new_w = pl.when(pl.col("flip")).then(5 - pl.col("winner_team_code")).otherwise(pl.col("winner_team_code"))
    return int(x.with_columns((new_w != pl.col("bomb_side")).alias("err"))["err"].sum())


rule_index = pl.col("round").is_in([13, 25, 28])
rule_match = (pl.col("round") == pl.col("halftime_r")) | pl.col("round").is_in([25, 28])
rule_match_no_ot = pl.col("round") == pl.col("halftime_r")
print("\n[7] fix evaluation on build 9961 bomb rounds (n =", B9961.height, "); errors = bomb outcome != corrected winner")
print("   no fix:                                              ", errors_after(pl.lit(False)))
print("   round-index only (flip 13, 25, 28 in every match):   ", errors_after(rule_index))
print("   per-match (flip the halftime round, plus 25 and 28): ", errors_after(rule_match))
print("   per-match halftime only (no 25/28 flip):             ", errors_after(rule_match_no_ot))

# ---- F. date vs build
d1225 = T.filter((pl.col("build_num") == 9842) & pl.col("bomb")
                 & pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).is_between(12, 25))
print("\n[8] date vs build: build 9842 bomb rounds on days 12-25:", d1225.height,
      "| mismatches:", int(d1225["bomb_mismatch"].sum()))
print("    matches on day 12: build 9842 =",
      H.filter((pl.col("build_num") == 9842) & pl.col("match_key").str.contains("csds/2024/02/12/")).height,
      "| build 9961 =", H.filter((pl.col("build_num") == 9961) & pl.col("match_key").str.contains("csds/2024/02/12/")).height)
print("\n=== end of reproduction ===")
