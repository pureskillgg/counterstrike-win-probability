"""D5 step 6: independent truth for round winners from player deaths (no bomb rounds).

For each (match, round) with no bomb_planted row and at least one player_death row,
the round is decided by elimination in most cases, so the side opposite to the LAST
victim (max tick) is the implied winner: victim side 2 (T) -> implied 3 (CT); 3 -> 2.
Compares the implied winner with:
  rw     round_end winner_team_code (label r)
  chain  winner from freeze_end score step r -> r+1 (2/3, or 0 if not a single +1)
Reports agreement by group and label, with the round_end label-1 presence split.
ASSUMPTION: the last death is decisive in a no-bomb round (time-outs are rare).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_06_deaths.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D5_common import read  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
B = pl.read_parquet(SCR / "bomb.parquet")

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
M = M.with_columns(grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))

DE = read("player_death", ["match_key", "round", "tick", "player_team_code"])
print("player_death rows:", DE.height, " null victim side:", DE.filter(pl.col("player_team_code").is_null()).height)
print("victim side counts:", DE.group_by("player_team_code").len().sort("player_team_code").to_dicts())
last = (DE.filter(pl.col("player_team_code").is_in([2, 3]))
        .sort(["match_key", "round", "tick"])
        .group_by(["match_key", "round"], maintain_order=False).agg(
            pl.col("player_team_code").last().alias("last_victim"),
            pl.col("tick").max().alias("last_death_tick")))
last = last.with_columns(implied=pl.when(pl.col("last_victim") == 2).then(pl.lit(3)).otherwise(pl.lit(2)))

planted = B.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique().with_columns(
    has_plant=pl.lit(True))

# chain winner for label r (fe r -> fe r+1)
ch = (R.sort(["match_key", "round"]).with_columns(
    nt=pl.col("fe_t_score").shift(-1).over("match_key"),
    nct=pl.col("fe_ct_score").shift(-1).over("match_key"),
    nr=pl.col("round").shift(-1).over("match_key"))
    .filter((pl.col("nr") == pl.col("round") + 1) & pl.col("fe_t_score").is_not_null() & pl.col("nt").is_not_null())
    .with_columns(dt=pl.col("nt") - pl.col("fe_t_score"), dct=pl.col("nct") - pl.col("fe_ct_score"))
    .with_columns(chain=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
                  .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)))
    .select(["match_key", "round", "chain"]))

base = (R.filter(pl.col("in_end") & pl.col("winner").is_not_null())
        .select(["match_key", "round", "winner", "reason_code", "legacy_code"])
        .join(M.select(["match_key", "grp", "day"]), on="match_key", how="left")
        .join(ch, on=["match_key", "round"], how="left")
        .join(last, on=["match_key", "round"], how="left")
        .join(planted, on=["match_key", "round"], how="left")
        .with_columns(has_plant=pl.col("has_plant").fill_null(False))
        .join(R.filter(pl.col("round") == 1).select(["match_key", pl.col("in_end").alias("r1_end")]),
              on="match_key", how="left"))

nb = base.filter((~pl.col("has_plant")) & pl.col("implied").is_not_null())
print("\nno-bomb rounds with deaths, round_end present:", nb.height)
nb = nb.with_columns(
    imp_vs_rw=pl.when(pl.col("implied") == pl.col("winner")).then(pl.lit("agree")).otherwise(pl.lit("disagree")),
    imp_vs_chain=pl.when(pl.col("chain") == 0).then(pl.lit("chain_zero"))
    .when(pl.col("implied") == pl.col("chain")).then(pl.lit("agree")).otherwise(pl.lit("disagree")))

print("\n== no-bomb rounds: implied (last victim) vs round_end winner, by group ==")
print(nb.group_by(["grp", "imp_vs_rw"]).agg(pl.len().alias("rounds")).sort(["grp", "imp_vs_rw"]))

print("\n== label 13 no-bomb rounds: implied vs rw vs chain, by group and r1_end ==")
l13 = nb.filter(pl.col("round") == 13)
print(l13.group_by(["grp", "r1_end", "imp_vs_rw", "imp_vs_chain"]).agg(pl.len().alias("rounds"))
      .sort(["grp", "r1_end", "imp_vs_rw", "imp_vs_chain"]))

print("\n== label 13 disagree-with-chain rows (ctrl_late, r1_end): implied vs chain vs rw counts ==")
d13 = l13.filter((pl.col("grp") == "ctrl_late") & (pl.col("imp_vs_chain") == "disagree"))
print(d13.group_by(["implied", "chain", "winner"]).agg(pl.len().alias("rounds")).sort(["implied", "chain", "winner"]))

print("\n== per label agreement (implied vs rw, implied vs chain), grp in early/ctrl_late/aff_r13, labels 9..16 and 24..29 ==")
pl_tbl = (nb.filter(pl.col("round").is_in(list(range(9, 17)) + list(range(24, 30))))
          .group_by(["grp", "round"]).agg(
              pl.len().alias("rounds"),
              (pl.col("imp_vs_rw") == "agree").sum().alias("agree_rw"),
              (pl.col("imp_vs_chain") == "agree").sum().alias("agree_chain"),
              (pl.col("imp_vs_chain") == "chain_zero").sum().alias("chain_zero"))
          .with_columns((pl.col("agree_rw") / pl.col("rounds")).round(3).alias("rate_rw"),
                        (pl.col("agree_chain") / pl.col("rounds")).round(3).alias("rate_chain"))
          .sort(["grp", "round"]))
print(pl_tbl)

print("\n== label 13 no-bomb rounds in ctrl_late, r1_end=true: rw agreement with implied by day ==")
c13 = nb.filter((pl.col("round") == 13) & (pl.col("grp") == "ctrl_late") & pl.col("r1_end"))
print(c13.group_by(["day", "imp_vs_rw"]).agg(pl.len().alias("rounds")).sort(["day", "imp_vs_rw"]))

print("\n== no-bomb: all late rounds (ctrl_late+aff) with r1_end=true, labels 13 vs other labels: rw agreement ==")
t = nb.filter(pl.col("grp").is_in(["ctrl_late", "aff_r13"]) & pl.col("r1_end"))
t = t.with_columns(lbl=pl.when(pl.col("round") == 13).then(pl.lit("13"))
                   .when(pl.col("round").is_in([25, 28])).then(pl.lit("25/28"))
                   .otherwise(pl.lit("other")))
print(t.group_by(["lbl", "imp_vs_rw"]).agg(pl.len().alias("rounds")).sort(["lbl", "imp_vs_rw"]))
