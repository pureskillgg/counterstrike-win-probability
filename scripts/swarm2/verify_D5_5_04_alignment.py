r"""Verifier for D5-5, stage 3: alternative explanations.

  A. Label alignment: are freeze_end(k) < round_end(k) < freeze_end(k+1) in tick order for every match?
     (a label shift would show up as window violations concentrated in r1-true matches)
  B. Offset test: in r1-true matches, does the clean naive step at k match round_end at k-1, k or k+1?
  C. Death-based elimination evidence per label for r1-true vs r1-false, by reason code, independent of
     the score step (is round_end contradicted by deaths at labels other than 13?).
  D. Bomb timing: plant->explode and explode->round_end gaps for label 13 vs other labels (ticks).
  E. Header final score check: do header finals match the step tallies or the round_end tallies?

Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_5_04_alignment.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")

W = pl.read_parquet(SCR / "wide.parquet")
WIN = pl.read_parquet(SCR / "win_elim.parquet")  # has k, fe_tick, re_tick, rw, reason, step, bomb_bs, r1, allT_dead, allCT_dead
BS = pl.read_parquet(SCR / "bomb_state_sub.parquet")
R1 = W.filter(pl.col("k") == 1).select(["match_key", pl.col("rw").is_not_null().alias("r1")])
W = W.join(R1, on="match_key", how="left").with_columns(pl.col("r1").fill_null(False))

# ---------- A. label alignment in tick order ----------
print("== A. tick-order window check: fe(k) < re(k) < fe(k+1) ==")
S = W.filter(pl.col("fe_tick").is_not_null()).select(["match_key", "k", "fe_tick", "re_tick", "r1"]).sort(["match_key", "k"])
S = S.with_columns(fe_next=pl.col("fe_tick").shift(-1).over("match_key"),
                   k_next=pl.col("k").shift(-1).over("match_key"))
S = S.filter(pl.col("k_next") == pl.col("k") + 1)  # consecutive labels only
S = S.with_columns(
    ok_in=(pl.col("re_tick").is_null()) | ((pl.col("re_tick") > pl.col("fe_tick"))),
    ok_out=(pl.col("re_tick").is_null()) | (pl.col("re_tick") < pl.col("fe_next")))
viol = S.filter(~(pl.col("ok_in") & pl.col("ok_out")))
print("consecutive-label pairs:", S.height, " window violations:", viol.height)
print(viol.group_by(["r1", "k"]).agg(pl.len().alias("n")).sort(["r1", "k"]).head(40))

# ---------- B. offset test (r1-true, clean naive step at k vs round_end at k+d) ----------
print("\n== B. offset test: clean naive step(k) vs round_end winner at k+d, by r1 ==")
C = W.filter((pl.col("step") != 0) & pl.col("rw").is_not_null()).select(["match_key", "k", "step", "r1"])
RW = W.filter(pl.col("rw").is_not_null()).select(["match_key", "k", "rw"])
out = []
for d in [-2, -1, 0, 1, 2]:
    j = C.with_columns(kk=pl.col("k") + d).join(RW.rename({"k": "kk"}), on=["match_key", "kk"], how="inner")
    g = j.group_by("r1").agg(pl.len().alias("pairs"), (pl.col("step") == pl.col("rw")).sum().alias("agree"))
    g = g.with_columns(d=pl.lit(d), rate=(pl.col("agree") / pl.col("pairs")).round(3))
    out.append(g)
print(pl.concat(out).sort(["r1", "d"]))

# ---------- C. death-based evidence by label, r1 group, reason ----------
print("\n== C. round_end reasons 8/9 contradicted by deaths (roster-complete), by label x r1 ==")
E = WIN.filter(pl.col("reason").is_in([8, 9]) & pl.col("rw").is_not_null())
E = E.with_columns(elim_ok=pl.when(pl.col("reason") == 8).then(pl.col("allT_dead")).otherwise(pl.col("allCT_dead")))
ct = (E.group_by(["k", "r1"]).agg(pl.len().alias("n_elim_rows"), (~pl.col("elim_ok")).sum().alias("contradicted"))
      .with_columns((pl.col("contradicted") / pl.col("n_elim_rows")).round(3).alias("rate")))
print(ct.filter(pl.col("contradicted") > 0).sort(["k", "r1"]).head(60))
print("\nbaseline (r1-false, all labels except 13): contradicted rate:")
b = E.filter((~pl.col("r1")) & (pl.col("k") != 13)).select((~pl.col("elim_ok")).mean().alias("rate"),
                                                          pl.len().alias("n"))
print(b)
print("\nr1-true, all labels except 13: contradicted rate:")
b2 = E.filter((pl.col("r1")) & (pl.col("k") != 13)).select((~pl.col("elim_ok")).mean().alias("rate"),
                                                            pl.len().alias("n"))
print(b2)

# ---------- D. bomb timing ----------
print("\n== D. bomb timing: plant->explode and explode->round_end (ticks), by label 13 vs others, r1 ==")
bt = (BS.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_exploded", "bomb_defused"]))
      .group_by(["match_key", "round", "event_type"]).agg(pl.col("tick").min().alias("t"))
      .pivot(on="event_type", index=["match_key", "round"], values="t")
      .rename({"round": "k"}))
X = (WIN.filter(pl.col("re_tick").is_not_null()).select(["match_key", "k", "re_tick", "r1", "rw", "reason"])
     .join(bt, on=["match_key", "k"], how="inner"))
X = X.with_columns(
    grp=pl.when(pl.col("k") == 13).then(pl.lit("k13")).otherwise(pl.lit("other")),
    gap_end=(pl.col("re_tick") - pl.col("bomb_exploded")),
    gap_def_end=(pl.col("re_tick") - pl.col("bomb_defused")),
    plant_to_boom=(pl.col("bomb_exploded") - pl.col("bomb_planted")),
)
print(X.filter(pl.col("bomb_exploded").is_not_null()).group_by(["grp", "r1"]).agg(
    pl.len().alias("n"), pl.col("gap_end").median().alias("med_explode_to_re"),
    pl.col("gap_end").min().alias("min_explode_to_re"), pl.col("gap_end").max().alias("max_explode_to_re"),
    pl.col("plant_to_boom").median().alias("med_plant_to_boom")).sort(["grp", "r1"]))
print(X.filter(pl.col("bomb_defused").is_not_null()).group_by(["grp", "r1"]).agg(
    pl.len().alias("n"), pl.col("gap_def_end").median().alias("med_defuse_to_re"),
    pl.col("gap_def_end").min().alias("min"), pl.col("gap_def_end").max().alias("max")).sort(["grp", "r1"]))

# ---------- E. header finals vs tallies ----------
print("\n== E. header final scores vs tallies from round_end winners and from clean steps ==")
H = pl.read_parquet(SCR / "header.parquet").rename({"key": "match_key"})
RWT = (W.filter(pl.col("rw").is_not_null()).group_by("match_key").agg(
    (pl.col("rw") == 2).sum().alias("rw_T"), (pl.col("rw") == 3).sum().alias("rw_CT"),
    pl.len().alias("n_rounds_re")))
STP = (W.filter(pl.col("step") != 0).group_by("match_key").agg(
    (pl.col("step") == 2).sum().alias("st_T"), (pl.col("step") == 3).sum().alias("st_CT")))
F = H.select(["match_key", "t_starters_score_final", "ct_starters_score_final", "day"]).join(
    RWT, on="match_key", how="left").join(STP, on="match_key", how="left")
F = F.with_columns(
    fin_sum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"),
    rw_sum=pl.col("rw_T") + pl.col("rw_CT"),
    st_sum=pl.col("st_T") + pl.col("st_CT"))
F = F.with_columns(
    fin_eq_rw=(pl.col("t_starters_score_final") == pl.col("rw_T")) & (pl.col("ct_starters_score_final") == pl.col("rw_CT")),
    fin_eq_rw_swapped=(pl.col("t_starters_score_final") == pl.col("rw_CT")) & (pl.col("ct_starters_score_final") == pl.col("rw_T")),
    fin_eq_st=(pl.col("t_starters_score_final") == pl.col("st_T")) & (pl.col("ct_starters_score_final") == pl.col("st_CT")),
    fin_eq_st_swapped=(pl.col("t_starters_score_final") == pl.col("st_CT")) & (pl.col("ct_starters_score_final") == pl.col("st_T")),
)
print("matches:", F.height)
print("header final == rw tally (direct):", int(F["fin_eq_rw"].sum()), " (swapped):", int(F["fin_eq_rw_swapped"].sum()))
print("header final == clean-step tally (direct):", int(F["fin_eq_st"].sum()), " (swapped):", int(F["fin_eq_st_swapped"].sum()))
print("header final_sum vs rounds with round_end (distribution of diff):")
print(F.with_columns((pl.col("fin_sum") - pl.col("rw_sum")).alias("diff")).group_by(
    pl.col("diff").clip(-3, 10)).len().sort("diff"))
F.write_parquet(SCR / "header_check.parquet")
