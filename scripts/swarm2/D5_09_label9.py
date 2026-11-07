"""D5 step 9: is the label-9 disagreement a second flip, and does it co-occur with the 13 flip?

Per (match, label) truth:
  death_implied   side opposite the last victim (no-bomb rounds with deaths)
  bomb_side       bomb outcome side (rounds with a bomb outcome)
  chain           freeze-end score step r -> r+1 (valid for r = 9, 13, 25, 28: no side swap in between;
                  r = 12, 24, 27 cross a swap and are not used)
Per match (late groups): flip9 and flip13 from the best available truth (bomb if present, else death).
Also prints label-9 disagreement by reason_code and by chain agreement.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_09_label9.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D5_common import read  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
B = pl.read_parquet(SCR / "bomb.parquet")

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
M = M.with_columns(grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
r1 = R.filter(pl.col("round") == 1).select(["match_key", pl.col("in_end").alias("r1_end")])

DE = read("player_death", ["match_key", "round", "tick", "player_team_code"]).filter(pl.col("player_team_code").is_in([2, 3]))
last = (DE.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("player_team_code").last().alias("last_victim")))
last = last.with_columns(death_implied=pl.when(pl.col("last_victim") == 2).then(pl.lit(3)).otherwise(pl.lit(2))).select(
    ["match_key", "round", "death_implied"])
planted = B.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique().with_columns(
    has_plant=pl.lit(True))
bside = (B.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
         .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
         .with_columns(bomb_side=pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)))
         .select(["match_key", "round", "bomb_side"]))

# chain for label r: fe(r) -> fe(r+1)
fe = R.select(["match_key", "round", "fe_t_score", "fe_ct_score"])
chain = (fe.join(fe.with_columns(round=pl.col("round") - 1).rename({"fe_t_score": "nt", "fe_ct_score": "nct"}),
                 on=["match_key", "round"], how="inner")
         .with_columns(dt=pl.col("nt") - pl.col("fe_t_score"), dct=pl.col("nct") - pl.col("fe_ct_score"))
         .with_columns(chain=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
                       .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)))
         .select(["match_key", "round", "chain"]))

T = (R.filter(pl.col("in_end") & pl.col("winner").is_not_null())
     .select(["match_key", "round", "winner", "reason_code"])
     .join(M.select(["match_key", "grp", "day"]), on="match_key", how="left")
     .join(r1, on="match_key", how="left").with_columns(pl.col("r1_end").fill_null(False))
     .join(last, on=["match_key", "round"], how="left")
     .join(planted, on=["match_key", "round"], how="left")
     .join(bside, on=["match_key", "round"], how="left")
     .join(chain, on=["match_key", "round"], how="left")
     .with_columns(has_plant=pl.col("has_plant").fill_null(False)))

# best truth: bomb side if present, else death-implied when no plant
T = T.with_columns(
    truth=pl.when(pl.col("bomb_side").is_not_null()).then(pl.col("bomb_side"))
    .when(~pl.col("has_plant") & pl.col("death_implied").is_not_null()).then(pl.col("death_implied"))
    .otherwise(pl.lit(None)),
    truth_src=pl.when(pl.col("bomb_side").is_not_null()).then(pl.lit("bomb"))
    .when(~pl.col("has_plant") & pl.col("death_implied").is_not_null()).then(pl.lit("death"))
    .otherwise(pl.lit("none")))
T = T.with_columns(flip=pl.when(pl.col("truth").is_null()).then(None)
                   .otherwise((pl.col("truth") != pl.col("winner")).cast(pl.Int8)))

print("== label 9 flip by group, r1_end, truth source (flip = truth != round_end winner) ==")
print(T.filter(pl.col("round") == 9).group_by(["grp", "r1_end", "truth_src"]).agg(
    pl.len().alias("rounds"), pl.col("flip").sum().alias("flipped"),
    pl.col("flip").mean().round(3).alias("rate")).sort(["grp", "r1_end", "truth_src"]))

print("\n== label 9 chain vs round_end (valid chain), by group x r1_end ==")
c9 = T.filter((pl.col("round") == 9) & pl.col("chain").is_not_null())
print(c9.group_by(["grp", "r1_end"]).agg(
    pl.len().alias("n"),
    (pl.col("chain") == 0).sum().alias("chain_zero"),
    ((pl.col("chain") != 0) & (pl.col("chain") != pl.col("winner"))).sum().alias("chain_disagree"),
).sort(["grp", "r1_end"]))

print("\n== label 9 no-bomb flipped rounds (ctrl_late, r1_end): reason_code x winner ==")
f9 = T.filter((pl.col("round") == 9) & (pl.col("grp") == "ctrl_late") & pl.col("r1_end") & (pl.col("flip") == 1))
print(f9.group_by(["reason_code", "winner", "truth"]).agg(pl.len().alias("rounds")).sort(["reason_code", "winner", "truth"]))

print("\n== per-match flip13 x flip9 (late r1_end matches; flip from best truth) ==")
f13 = T.filter((pl.col("round") == 13) & pl.col("flip").is_not_null()).select(["match_key", pl.col("flip").alias("f13")])
f9m = T.filter((pl.col("round") == 9) & pl.col("flip").is_not_null()).select(["match_key", pl.col("flip").alias("f9")])
X = (M.select(["match_key", "grp"]).join(r1, on="match_key", how="left").with_columns(pl.col("r1_end").fill_null(False))
     .join(f13, on="match_key", how="inner").join(f9m, on="match_key", how="inner")
     .filter(pl.col("grp").is_in(["ctrl_late", "aff_r13"]) & pl.col("r1_end")))
print(X.group_by(["grp", "f9", "f13"]).agg(pl.len().alias("matches")).sort(["grp", "f9", "f13"]))

print("\n== label 9 flip rate in ctrl_late r1_end matches split by whether round 13 flipped ==")
print(X.filter(pl.col("grp") == "ctrl_late").group_by(["f13", "f9"]).agg(pl.len().alias("matches")).sort(["f13", "f9"]))

print("\n== day-level label-9 flip rate (no-bomb, death truth), ctrl_late r1_end ==")
print(T.filter((pl.col("round") == 9) & (pl.col("grp") == "ctrl_late") & pl.col("r1_end") & (pl.col("truth_src") == "death"))
      .group_by("day").agg(pl.len().alias("rounds"), pl.col("flip").sum().alias("flipped"),
                           pl.col("flip").mean().round(3).alias("rate")).sort("day"))
