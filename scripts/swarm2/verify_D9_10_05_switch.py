"""verify_D9_10 step 5: side-switch structure per match and neighbour alignment of round_end labels.

  E. team identity A = players on the side they hold in the match's first round present; side of team A per round
     = mode of team_code among A's players that round. Run-length signature over rounds 1-30 (build 9961).
  F. cross-tab: first detected switch round (player_info) vs non-bomb flip status for rounds 9 and 13.
  G. neighbour alignment on bomb rounds: bomb outcome side vs recorded winner of r, r-1, r+1.
ASSUMPTION: team identity by first-round side is stable within a match (players do not change team mid-match).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_10")
R = pl.read_parquet(SCR / "round_table.parquet")
FS = pl.read_parquet(SCR / "first_switch.parquet")
PI = pl.scan_parquet(sorted(glob.glob(str(BASE / f"basic_player_info{SUF}" / "dataframe_*")))).select(
    ["match_key", "round", "player_id", "team_code"]).collect()
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last").sort(["match_key", "player_id", "round"])
first_code = PIu.group_by(["match_key", "player_id"]).agg(pl.col("team_code").first().alias("code_r1"),
                                                           pl.col("round").min().alias("r_first"))
PIa = PIu.join(first_code, on=["match_key", "player_id"], how="left")
# team A = players whose team_code in their first round is the value held by the most players of that match in that round
mm = (PIa.filter(pl.col("round") == pl.col("r_first")).group_by(["match_key", "code_r1"]).len())
topcode = mm.sort(["match_key", "len"], descending=True).group_by("match_key").agg(pl.col("code_r1").first().alias("A_code0"))
PIa = PIa.join(topcode, on="match_key", how="left")
PIa = PIa.with_columns(pl.col("code_r1").eq(pl.col("A_code0")).alias("in_A"))
A_side = (PIa.group_by(["match_key", "round", "in_A"]).agg(pl.col("team_code").mode().first().alias("side"))
            .filter(pl.col("in_A")).select(["match_key", "round", "side"]))

pc = R.select(["match_key", "build_num"]).unique(subset=["match_key"])
seq = (A_side.filter(pl.col("round") <= 30).sort(["match_key", "round"])
          .group_by("match_key").agg(pl.col("round").alias("rs"), pl.col("side").alias("ss"))
          .join(pc, on="match_key", how="left").filter(pl.col("build_num") == 9961))


def rle(ss, rs):
    out, prev, start, last = [], None, None, None
    for r, s in zip(rs, ss):
        if s != prev:
            if prev is not None:
                out.append(f"{prev}:{start}-{last}")
            prev, start = s, r
        last = r
    if prev is not None:
        out.append(f"{prev}:{start}-{last}")
    return " ".join(out)


seq = seq.with_columns(pl.struct(["ss", "rs"]).map_elements(lambda s: rle(s["ss"], s["rs"]), return_dtype=pl.String).alias("sig"))
print("== E. team-A side run-length signature (build 9961, rounds 1-30): top 12")
print(seq.group_by("sig").len().sort("len", descending=True).head(12))
print("distinct signatures:", seq["sig"].n_unique(), "| matches:", seq.height)
# switch positions as a set: where does team-A side change?
chg = collections.Counter()
for row in seq.iter_rows(named=True):
    prev = None
    for r, s in zip(row["rs"], row["ss"]):
        if prev is not None and s != prev:
            chg[r] += 1
        prev = s
print("team-A side change positions (round index where side differs from prior round), build 9961:", sorted(chg.items()))

# ---- F. cross-tab first switch vs flip status (non-bomb, pc20, roster, build 9961)
S = R.filter((pl.col("build_num") == 9961) & (pl.col("player_count") == 20) & pl.col("n_CT").is_not_null()
             & pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30]) & ~pl.col("has_bomb_outcome"))
S = S.with_columns(
    pl.when((pl.col("CT_dead") >= pl.col("n_CT")) & (pl.col("winner_team_code") == 2)).then(pl.lit("R"))
      .when((pl.col("T_dead") >= pl.col("n_T")) & (pl.col("winner_team_code") == 2)).then(pl.lit("F"))
      .when((pl.col("T_dead") >= pl.col("n_T")) & (pl.col("winner_team_code") == 3)).then(pl.lit("R"))
      .when((pl.col("CT_dead") >= pl.col("n_CT")) & (pl.col("winner_team_code") == 3)).then(pl.lit("F"))
      .otherwise(pl.lit("N")).alias("c"))
cdict = collections.defaultdict(dict)
for row in S.iter_rows(named=True):
    cdict[row["match_key"]][row["round"]] = row["c"]
fsw = dict(zip(FS["match_key"].to_list(), FS["first_switch_round"].to_list()))
xt9 = collections.Counter()
xt13 = collections.Counter()
for mk, d in cdict.items():
    xt9[(fsw.get(mk), d.get(9, "-"))] += 1
    xt13[(fsw.get(mk), d.get(13, "-"))] += 1
print("\n== F. first switch round (player_info) x round-9 non-bomb status:")
for k in sorted(xt9, key=lambda x: (str(x[0]), x[1])):
    print("   ", k, xt9[k])
print("== F. first switch round (player_info) x round-13 non-bomb status:")
for k in sorted(xt13, key=lambda x: (str(x[0]), x[1])):
    print("   ", k, xt13[k])

# ---- G. neighbour alignment on bomb rounds
RE = R.select(["match_key", "round", "winner_team_code", "build_num"]).sort(["match_key", "round"])
RE = RE.with_columns(pl.col("winner_team_code").shift(1).over("match_key").alias("w_prev"),
                     pl.col("round").shift(1).over("match_key").alias("r_prev"),
                     pl.col("winner_team_code").shift(-1).over("match_key").alias("w_next"),
                     pl.col("round").shift(-1).over("match_key").alias("r_next"))
B = R.filter(pl.col("has_bomb_outcome")).with_columns(
    pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
B = B.join(RE.select(["match_key", "round", "w_prev", "w_next", "r_prev", "r_next"]), on=["match_key", "round"], how="left")
Bn = B.filter((pl.col("build_num") == 9961) & pl.col("round").is_in([8, 9, 10, 12, 13, 14, 24, 25, 26, 28, 29]))
print("\n== G. bomb rounds (build 9961): agreement of bomb side with recorded winner of r, r-1, r+1")
print(Bn.group_by("round").agg(
    pl.len().alias("n"),
    (pl.col("bomb_side") == pl.col("winner_team_code")).sum().alias("agree_own"),
    (pl.col("bomb_side") == pl.col("w_prev")).sum().alias("agree_prev"),
    (pl.col("bomb_side") == pl.col("w_next")).sum().alias("agree_next")).sort("round"))
