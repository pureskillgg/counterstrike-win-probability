r"""D4 decisive check: side-level round-13 evidence (round_state score change; bomb side) by match class, 2024.

Classes come from D4_swap_schedule.py logic (complete regulation matches, n <= 24):
  affected      label-based r13 bomb mismatch
  flip_nobomb   S12 header misfit fixed by flipping r13 only, no r13 bomb row
  fits_S12      S12 header fit
  other_misfit  everything else
For each class: round-13 rounds with an unambiguous score side (freeze_end 13 -> 14) and round_end winner:
agreement of score side with round_end, and with the bomb side where a bomb row exists.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_class_score.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final"])
Hd = {r["key"]: r for r in H.iter_rows(named=True)}
re_ = read("round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "player_count"]).unique(
    ["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
outs = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
R = re_.join(outs.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")
mm13 = set(R.filter((pl.col("round") == 13) & pl.col("bside").is_not_null())
           .filter(pl.col("bside") != pl.col("winner_team_code"))["match_key"].to_list())
by = {}
for r in R.sort(["match_key", "round"]).iter_rows(named=True):
    by.setdefault(r["match_key"], []).append(r)
cls_of = {}
for m, rounds in by.items():
    h = Hd.get(m)
    if h is None:
        continue
    tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(rounds)
    if not (n == tA + tB and n <= 24 and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    has_r13_bomb = any(int(x["round"]) == 13 and x["bside"] is not None for x in rounds)
    r13 = next((x for x in rounds if int(x["round"]) == 13), None)
    wa12 = sum(1 for x in rounds if int(x["winner_team_code"]) == (T if int(x["round"]) <= 12 else CT))
    fits = (wa12, n - wa12) == (tA, tB)
    if m in mm13:
        c = "affected"
    elif fits:
        c = "fits_S12"
    elif r13 is None:
        c = "other_misfit"
    else:
        nwa = wa12 - 1 if int(r13["winner_team_code"]) == CT else wa12 + 1
        c = "flip_nobomb" if ((nwa, n - nwa) == (tA, tB) and not has_r13_bomb) else "other_misfit"
    cls_of[m] = c
C = pl.DataFrame({"match_key": list(cls_of.keys()), "cls": list(cls_of.values())})

fe = (read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "t_score", "ct_score"])
      .unique(["match_key", "round"], keep="first"))
nxt = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t_next", "ct_score": "ct_next"})
D = fe.filter(pl.col("round") == 13).join(nxt.filter(pl.col("round") == 13), on=["match_key", "round"], how="inner")
D = D.with_columns((pl.col("ct_next") - pl.col("ct_score")).alias("dC"), (pl.col("t_next") - pl.col("t_score")).alias("dT"))
D = D.with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
J = (C.join(D.select(["match_key", "score_side"]), on="match_key", how="inner")
     .join(R.filter(pl.col("round") == 13).select(["match_key", "winner_team_code", "bside", "win_reason_code"]),
           on="match_key", how="inner")
     .filter(pl.col("score_side") != 0))
J = J.with_columns(
    (pl.col("score_side") == pl.col("winner_team_code")).alias("score_eq_end"),
    (pl.col("score_side") == pl.col("bside")).alias("score_eq_bomb"))
print("round 13, complete regulation matches, unambiguous score side: class summary")
print(J.group_by("cls").agg(pl.len().alias("n"),
                            pl.col("score_eq_end").mean().alias("score_eq_round_end"),
                            pl.col("bside").is_not_null().sum().alias("n_with_bomb"),
                            pl.col("score_eq_bomb").filter(pl.col("bside").is_not_null()).mean().alias("score_eq_bomb_rate")
                            ).sort("cls"))
print("\nflip_nobomb: score side vs round_end winner and win_reason_code")
fn = J.filter(pl.col("cls") == "flip_nobomb")
print(fn.group_by(["win_reason_code", "winner_team_code", "score_side"]).len().sort(["win_reason_code", "winner_team_code", "score_side"]))

# Death check for flip_nobomb rounds without the 3 s window: last victim in the live window (any gap).
de = read("player_death", ["match_key", "round", "tick", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]) & (pl.col("round") == 13))
lv = (read("round_state", ["match_key", "round", "tick", "event_type"]).filter(pl.col("event_type") == "round_freeze_end")
      .select(["match_key", "round", pl.col("tick").alias("live")]).unique(["match_key", "round"], keep="first"))
ends = re_.filter(pl.col("round") == 13).select(["match_key", pl.col("round"), "winner_team_code"])
endt = read("round_end", ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first").rename({"tick": "end"})
W13 = (J.select(["match_key", "cls", "score_side", "winner_team_code"])
       .join(lv.filter(pl.col("round") == 13), on="match_key", how="inner")
       .join(endt.filter(pl.col("round") == 13), on="match_key", how="inner"))
DD = de.join(W13.select(["match_key", "live", "end"]), on="match_key", how="inner").filter(
    (pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
lastt = DD.group_by("match_key").agg(pl.col("tick").max().alias("lt"))
lastv = DD.join(lastt, on="match_key").filter(pl.col("tick") == pl.col("lt")).group_by("match_key").agg(
    pl.col("player_team_code").n_unique().alias("ns"), pl.col("player_team_code").first().alias("victim"), pl.col("lt").first())
K2 = W13.join(lastv, on="match_key", how="left").with_columns(
    (pl.col("end") - pl.col("lt")).alias("gap_ticks"),
    (5 - pl.col("victim")).alias("pred_victim"))
K2 = K2.filter(pl.col("cls") == "flip_nobomb").with_columns(
    (pl.col("pred_victim") == pl.col("score_side")).alias("victim_eq_score"),
    (pl.col("pred_victim") == pl.col("winner_team_code")).alias("victim_eq_round_end"))
print("\nflip_nobomb round 13: last death (any gap) predicts which side? (score side vs round_end vs victim)")
print(K2.select(pl.len().alias("n"), pl.col("victim").is_not_null().sum().alias("with_death"),
                pl.col("victim_eq_score").mean().alias("victim_eq_score"),
                pl.col("victim_eq_round_end").mean().alias("victim_eq_round_end"),
                pl.col("gap_ticks").median().alias("median_gap_ticks"),
                pl.col("gap_ticks").quantile(0.9).alias("p90_gap_ticks")))
print(K2.with_columns(pl.when(pl.col("gap_ticks") <= 192).then(pl.lit("<=3s")).when(pl.col("gap_ticks") <= 600).then(pl.lit("3-9s"))
      .otherwise(pl.lit(">9s")).alias("gapbin")).group_by("gapbin").agg(
    pl.len().alias("n"), pl.col("victim_eq_score").mean().alias("victim_eq_score"),
    pl.col("victim_eq_round_end").mean().alias("victim_eq_round_end")).sort("gapbin"))
