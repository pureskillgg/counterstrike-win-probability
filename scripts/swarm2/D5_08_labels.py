"""D5 step 8: which labels disagree with independent truth, by group and round-end label-1 presence.

Truth sources (independent of the round_end winner_team_code being tested):
  death  winner implied by the side opposite the last victim (player_death.player_team_code),
         for rounds with no bomb_planted row and at least one death. Noise ~5-8% in early.
  bomb   bomb outcome side (defused -> 3, exploded -> 2) for rounds with a bomb outcome.
Output: for each label 2..30, the count of rounds and disagreement rate vs round_end winner,
for death truth and bomb truth, split by group (early / ctrl_late / aff_r13) and r1_end.
Chain (freeze-end score step) is NOT used here: it is invalid across side swaps.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_08_labels.py
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
base = (R.filter(pl.col("in_end") & pl.col("winner").is_not_null())
        .select(["match_key", "round", "winner"])
        .join(M.select(["match_key", "grp"]), on="match_key", how="left")
        .join(r1, on="match_key", how="left").with_columns(pl.col("r1_end").fill_null(False)))

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

T = (base.join(last, on=["match_key", "round"], how="left")
     .join(planted, on=["match_key", "round"], how="left")
     .join(bside, on=["match_key", "round"], how="left")
     .with_columns(has_plant=pl.col("has_plant").fill_null(False)))
T = T.with_columns(
    death_dis=pl.when(pl.col("death_implied").is_null() | pl.col("has_plant")).then(None)
    .otherwise((pl.col("death_implied") != pl.col("winner")).cast(pl.Int8)),
    bomb_dis=pl.when(pl.col("bomb_side").is_null()).then(None)
    .otherwise((pl.col("bomb_side") != pl.col("winner")).cast(pl.Int8)),
)

for name, col in [("DEATH truth (no-bomb rounds): disagreement rate of round_end winner", "death_dis"),
                  ("BOMB truth (rounds with bomb outcome): disagreement rate of round_end winner", "bomb_dis")]:
    print("\n==========", name, "==========")
    g = (T.filter(pl.col(col).is_not_null() & pl.col("round").is_between(1, 30))
         .group_by(["round", "grp", "r1_end"]).agg(pl.len().alias("n"), pl.col(col).sum().alias("dis"))
         .with_columns((pl.col("dis") / pl.col("n")).round(3).alias("rate")))
    # wide view: rate by label for each group x r1_end
    keys = (g.select(["grp", "r1_end"]).unique().sort(["grp", "r1_end"]).rows())
    out = None
    for grp, r1e in keys:
        s = g.filter((pl.col("grp") == grp) & (pl.col("r1_end") == r1e)).select(
            ["round", pl.col("n").alias(f"n|{grp}|{int(r1e)}"), pl.col("rate").alias(f"rate|{grp}|{int(r1e)}")])
        out = s if out is None else out.join(s, on="round", how="full", coalesce=True)
    print(out.sort("round"))

print("\n== label-1 round_end presence by group (matches) ==")
print(M.join(r1, on="match_key", how="left").with_columns(pl.col("r1_end").fill_null(False))
      .group_by(["grp", "r1_end"]).agg(pl.len().alias("matches")).sort(["grp", "r1_end"]))
