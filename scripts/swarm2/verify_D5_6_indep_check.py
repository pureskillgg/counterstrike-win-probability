"""verify D5-6 (independent recompute, separate from any other verify_D5_6_* file).

Reads only the 2024 tome on G: (lazy scans, column projection). Builds per (match, round) features
directly from raw channels (round_end, round_state, bomb_state, player_death) and the header.
Does NOT read any D5 scratchpad file or any other verifier's scratch file.

Sections:
  C0  reproduce the claimed label-13 numbers (no-bomb, ctrl_late, regime A = round_announce_match_start>0)
  T1  header final-score test: team totals from round_end winners vs header t_starters_score_final
      (no death or freeze-end channel in the as-is variant). Round-13 variants: flipped, chain, death.
  T2  per-player victim side code across consecutive death rounds (death-channel side coding check)
  T3  bomb_planted player_team_code (planter must be T=2) by label and regime
  T4  bomb-plant rate by label and regime (is the no-bomb set contaminated at label 13?)
  T5  reason-message vs victim-count elimination check (round_end winner vs which side died out)
  T6  misattribution test: death-implied side at label 13 vs round_end at labels 12, 13, 14

ASSUMPTIONS (marked where used):
  A1 side swap at halftime after round 12 (MR12). Matches with max round > 24 are excluded from T1.
  A2 roster = 5 players per side for the elimination check T5 (checked against per-match maxima).
  A3 chain = freeze-end score step fe(r)->fe(r+1) with side-based scores (rule from D5_06).
  A4 regime A = round_announce_match_start count > 0 (matches the D5_10 marker definition).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_indep_check.py
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
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_6_indep")
SCR.mkdir(parents=True, exist_ok=True)
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]


def scan(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    if not fs:
        raise FileNotFoundError(ch)
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ---------------- header ----------------
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))[0]
H = (pl.read_parquet(hf, columns=["key", "t_starters_score_final", "ct_starters_score_final"])
     .rename({"key": "match_key"})
     .with_columns(day=pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1)))
print("header rows:", H.height, " unique keys:", H["match_key"].n_unique(), " null final scores:",
      H.filter(pl.col("t_starters_score_final").is_null() | pl.col("ct_starters_score_final").is_null()).height)

# ---------------- round_end ----------------
RE_raw = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                   "player_count"]).collect()
RE = (RE_raw.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("winner_team_code").first().alias("rw"),
    pl.col("win_reason_code").first().alias("reason"),
    pl.col("winner_team_code").n_unique().alias("n_distinct_w"),
    pl.len().alias("n_end_rows"),
    pl.col("player_count").max().alias("pc")))
print("round_end raw rows:", RE_raw.height, " (match,round) pairs:", RE.height,
      " matches:", RE["match_key"].n_unique(), " pairs with >1 winner code:", RE.filter(pl.col("n_distinct_w") > 1).height)

# ---------------- round_state: freeze-end scores + match markers ----------------
RS = (scan("round_state")
      .filter(pl.col("event_type").is_in(["round_freeze_end", "round_announce_match_start", "cs_win_panel_round"]))
      .select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect())
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(
          pl.col("t_score").first().alias("fe_t"),
          pl.col("ct_score").first().alias("fe_ct"),
          pl.len().alias("n_fe")))
MK = RS.group_by("match_key").agg(
    (pl.col("event_type") == "round_announce_match_start").sum().alias("mstart"),
    (pl.col("event_type") == "cs_win_panel_round").sum().alias("cwp"))
print("freeze-end (match,round) pairs:", FE.height, " mstart>0 matches:", MK.filter(pl.col("mstart") > 0).height,
      " cwp>0 matches:", MK.filter(pl.col("cwp") > 0).height)

# ---------------- bomb_state ----------------
BS = (scan("bomb_state")
      .filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect())
PL = (BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(
    pl.len().alias("n_plant"),
    pl.col("player_team_code").first().alias("plant_team")))
BO = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
      .with_columns(bomb_side=pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)))
      .select(["match_key", "round", "bomb_side"]))
print("bomb_planted rows:", int(PL["n_plant"].sum()), " bomb outcome (match,round):", BO.height)

# ---------------- player_death ----------------
DE = scan("player_death").select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
DEv = DE.filter(pl.col("player_team_code").is_in([2, 3]) & pl.col("round").is_not_null())
print("player_death rows:", DE.height, " with victim side 2/3:", DEv.height)

dmax = DEv.group_by(["match_key", "round"]).agg(
    pl.col("tick").max().alias("last_tick"), pl.len().alias("n_deaths"))
dl = DEv.join(dmax, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("last_tick"))
lastv = dl.group_by(["match_key", "round"]).agg(
    pl.col("player_team_code").n_unique().alias("n_sides_last"),
    pl.col("player_team_code").max().alias("last_side_max"),
    pl.len().alias("n_at_last"))
lastv = lastv.with_columns(
    last_victim=pl.when(pl.col("n_sides_last") == 1).then(pl.col("last_side_max")).otherwise(None))
vc = DEv.drop_nulls("player_id").group_by(["match_key", "round"]).agg(
    pl.col("player_id").filter(pl.col("player_team_code") == 2).n_unique().alias("nT"),
    pl.col("player_id").filter(pl.col("player_team_code") == 3).n_unique().alias("nCT"))

# ---------------- match-level ----------------
MR = RE.group_by("match_key").agg(
    pl.col("round").max().alias("max_round"),
    pl.col("round").n_unique().alias("n_rounds"),
    (pl.col("round") == 1).any().alias("r1_end"))
MR = MR.join(H, on="match_key", how="left").join(MK, on="match_key", how="left")

# ---------------- per-round table ----------------
R = (RE.join(FE, on=["match_key", "round"], how="left")
     .join(PL, on=["match_key", "round"], how="left")
     .join(BO, on=["match_key", "round"], how="left")
     .join(dmax, on=["match_key", "round"], how="left")
     .join(lastv.select(["match_key", "round", "last_victim", "n_at_last"]), on=["match_key", "round"], how="left")
     .join(vc, on=["match_key", "round"], how="left"))
R = R.with_columns(
    has_plant=pl.col("n_plant").fill_null(0) > 0,
    implied=pl.when(pl.col("last_victim") == 2).then(pl.lit(3))
    .when(pl.col("last_victim") == 3).then(pl.lit(2)).otherwise(None))

# aff_r13 exactly as D5_02 / e2: round player_count == 20, bomb outcome present, round-13 mismatch
mm = R.filter((pl.col("pc") == 20) & pl.col("bomb_side").is_not_null()
              & (pl.col("bomb_side") != pl.col("rw")) & (pl.col("round") == 13))
AFF = mm.select("match_key").unique().with_columns(aff_r13=pl.lit(True))
print("\nmatches with round-13 bomb mismatch (aff_r13, D5_02 definition):", AFF.height)

MR = MR.join(AFF, on="match_key", how="left").with_columns(
    aff_r13=pl.col("aff_r13").fill_null(False),
    regime=pl.when(pl.col("mstart") > 0).then(pl.lit("A")).otherwise(pl.lit("B")))
MR = MR.with_columns(grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
                     .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                     .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
print("regime x grp x r1_end (matches):")
print(MR.group_by(["regime", "grp", "r1_end"]).agg(pl.len().alias("matches")).sort(["regime", "grp", "r1_end"]))
print("regime A <=> r1_end mismatches:", MR.filter((pl.col("regime") == "A") != pl.col("r1_end")).height)

# chain (A3): fe(r) -> fe(r+1), rule from D5_06
FEs = FE.sort(["match_key", "round"]).with_columns(
    nr=pl.col("round").shift(-1).over("match_key"),
    nfe_t=pl.col("fe_t").shift(-1).over("match_key"),
    nfe_ct=pl.col("fe_ct").shift(-1).over("match_key"))
CH = (FEs.filter((pl.col("nr") == pl.col("round") + 1) & pl.col("nfe_t").is_not_null())
      .with_columns(dt=pl.col("nfe_t") - pl.col("fe_t"), dct=pl.col("nfe_ct") - pl.col("fe_ct"))
      .with_columns(chain=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
                    .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)))
      .select(["match_key", "round", "chain", "dt", "dct"]))

R = (R.join(CH, on=["match_key", "round"], how="left")
     .join(MR.select(["match_key", "grp", "regime", "r1_end", "max_round", "n_rounds"]), on="match_key", how="left")
     .join(H.select(["match_key", "day"]), on="match_key", how="left"))
R.write_parquet(SCR / "rounds.parquet")

# =====================================================================================
section("C0. Reproduce the claim: label 13, no-bomb rounds with deaths, round_end present")
nb = R.filter((~pl.col("has_plant")) & pl.col("implied").is_not_null() & pl.col("rw").is_not_null())
print("no-bomb rounds with deaths (all labels):", nb.height)
nb13 = nb.filter(pl.col("round") == 13).with_columns(
    rw_dis=(pl.col("implied") != pl.col("rw")),
    ch_agree=(pl.col("chain") == pl.col("implied")),
    ch_zero=(pl.col("chain") == 0),
    ch_null=pl.col("chain").is_null())
print(nb13.group_by(["grp", "r1_end"]).agg(
    pl.len().alias("n"),
    pl.col("rw_dis").sum().alias("rw_disagree_last_victim"),
    pl.col("ch_agree").sum().alias("chain_agree_last_victim"),
    pl.col("ch_zero").sum().alias("chain_zero"),
    pl.col("ch_null").sum().alias("chain_null")).sort(["grp", "r1_end"]))
c = nb13.filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end"))
print("CLAIM CHECK ctrl_late & r1_end (regime A): n=%d  rw disagree=%d (%.3f)  chain agree=%d (%.3f)" % (
    c.height, int(c["rw_dis"].sum()), c["rw_dis"].sum() / c.height,
    int(c["ch_agree"].sum()), c["ch_agree"].sum() / c.height))

# =====================================================================================
section("T1. Header final-score test (as-is variant uses round_end only)")
print("ASSUMPTION A1: MR12 halftime swap after round 12; team A = t_starters (started on T).")
M_ok = MR.filter((pl.col("max_round") <= 24) & (pl.col("n_rounds") == pl.col("max_round"))
                 & ((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")) == pl.col("n_rounds")))
print("matches eligible (<=24 rounds, contiguous, header sum == rounds):", M_ok.height, " of", MR.height)
X = (R.filter(pl.col("match_key").is_in(M_ok["match_key"]) & pl.col("rw").is_not_null())
     .with_columns(a_as=pl.when(pl.col("round") <= 12).then((pl.col("rw") == 2).cast(pl.Int32))
                   .otherwise((pl.col("rw") == 3).cast(pl.Int32))))
per = X.group_by("match_key").agg(
    pl.col("a_as").sum().alias("a_tot"),
    pl.col("a_as").filter(pl.col("round") == 13).sum().alias("a13_as"),
    (pl.col("round") == 13).any().alias("has13"))
r13 = X.filter(pl.col("round") == 13).select(["match_key", "rw", "chain", "implied"])
per = (per.join(r13, on="match_key", how="left")
       .join(M_ok.select(["match_key", "grp", "regime", "t_starters_score_final"]), on="match_key", how="left"))
per = per.with_columns(
    a_rest=pl.col("a_tot") - pl.col("a13_as"),
    a13_chain=pl.when(pl.col("chain") == 3).then(1).when(pl.col("chain") == 2).then(0).otherwise(None),
    a13_death=pl.when(pl.col("implied") == 3).then(1).when(pl.col("implied") == 2).then(0).otherwise(None))
per = per.with_columns(
    ok_as=(pl.col("a_tot") == pl.col("t_starters_score_final")),
    ok_flip13=((pl.col("a_rest") + (1 - pl.col("a13_as"))) == pl.col("t_starters_score_final")),
    ok_chain13=pl.when(pl.col("a13_chain").is_null()).then(pl.lit(None, dtype=pl.Boolean))
    .otherwise((pl.col("a_rest") + pl.col("a13_chain")) == pl.col("t_starters_score_final")),
    ok_death13=pl.when(pl.col("a13_death").is_null()).then(pl.lit(None, dtype=pl.Boolean))
    .otherwise((pl.col("a_rest") + pl.col("a13_death")) == pl.col("t_starters_score_final")))
print("\nT1 header consistency by group x regime (matches with a round 13):")
print(per.filter(pl.col("has13")).group_by(["grp", "regime"]).agg(
    pl.len().alias("matches"),
    pl.col("ok_as").mean().round(3).alias("ok_as_roundend"),
    pl.col("ok_flip13").mean().round(3).alias("ok_if_r13_flipped"),
    pl.col("ok_chain13").drop_nulls().mean().round(3).alias("ok_r13_from_chain"),
    pl.col("ok_chain13").drop_nulls().len().alias("n_chain13"),
    pl.col("ok_death13").drop_nulls().mean().round(3).alias("ok_r13_from_death"),
    pl.col("ok_death13").drop_nulls().len().alias("n_death13")).sort(["grp", "regime"]))
print("\nT1 as-is consistency counts by group x regime x ok_as (matches with a round 13):")
print(per.filter(pl.col("has13")).group_by(["grp", "regime", "ok_as"]).agg(pl.len().alias("matches")).sort(
    ["grp", "regime", "ok_as"]))
print("\nT1 early-group sanity (should be near 1.0 if the header convention holds):")
print(per.filter(pl.col("grp") == "early").select(
    pl.col("ok_as").mean().round(3).alias("early_ok_as"), pl.len().alias("n")))

# =====================================================================================
section("T2. Per-player victim side code across consecutive death rounds (death-channel side coding)")
print("Expect: same side code within a half (pairs r->r+1 with r<=11 or r>=13), different across 12->13.")
PD = DEv.drop_nulls("player_id").select(["match_key", "round", "player_id", "player_team_code"]).unique(
    ["match_key", "round", "player_id"], keep="first")
A_ = PD.rename({"player_team_code": "c_a"})
B_ = PD.with_columns((pl.col("round") - 1).alias("round")).rename({"player_team_code": "c_b"})
PP = A_.join(B_, on=["match_key", "player_id", "round"], how="inner")
PP = PP.join(MR.select(["match_key", "grp", "regime"]), on="match_key", how="left")
PP = PP.with_columns(same=(pl.col("c_a") == pl.col("c_b")))
t2 = (PP.filter(pl.col("round").is_between(9, 15)).group_by(["grp", "regime", "round"]).agg(
    pl.len().alias("pairs"), pl.col("same").sum().alias("same_side_code"),
    (pl.col("same").sum() / pl.len()).round(3).alias("same_rate")).sort(["grp", "regime", "round"]))
print("pairs (round r -> r+1) with the same victim side code, by label r:")
print(t2)

# =====================================================================================
section("T3. bomb_planted player_team_code (planter must be T=2) by label and regime")
PLx = (BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round", "player_team_code"])
       .join(MR.select(["match_key", "regime", "grp"]), on="match_key", how="left"))
print(PLx.filter(pl.col("round").is_between(9, 16)).group_by(["regime", "round"]).agg(
    pl.len().alias("plants"), (pl.col("player_team_code") == 2).sum().alias("planter_is_T"),
    pl.col("player_team_code").is_null().sum().alias("null_code")).sort(["regime", "round"]))

# =====================================================================================
section("T4. Bomb-plant rate by label and regime (rounds with round_end winner)")
plant = (R.filter(pl.col("rw").is_not_null() & pl.col("grp").is_in(["ctrl_late", "aff_r13", "early"]))
         .group_by(["grp", "regime", "round"]).agg(
             pl.len().alias("rounds"),
             pl.col("has_plant").sum().alias("with_plant"),
             (pl.col("has_plant").sum() / pl.len()).round(3).alias("plant_rate"),
             pl.col("bomb_side").is_not_null().sum().alias("with_outcome"))
         .filter(pl.col("round").is_between(9, 16)).sort(["grp", "regime", "round"]))
print(plant)

# =====================================================================================
section("T5. Elimination check: reason message vs victim counts (A2: roster 5 per side)")
mx = (R.filter(pl.col("nT").is_not_null()).group_by("match_key").agg(
    pl.col("nT").max().alias("mT"), pl.col("nCT").max().alias("mCT")))
print("max nT per match (value counts, top):", mx["mT"].value_counts().sort("mT").to_dicts()[-8:])
print("max nCT per match (value counts, top):", mx["mCT"].value_counts().sort("mCT").to_dicts()[-8:])
el = R.filter(pl.col("reason").is_in([8, 9]) & pl.col("rw").is_not_null() & pl.col("nT").is_not_null()).with_columns(
    loser_n=pl.when(pl.col("rw") == 2).then(pl.col("nCT")).otherwise(pl.col("nT")))
el = el.with_columns(elim_ok=(pl.col("loser_n") >= 5))
print("\nelimination-consistency (loser side has >=5 distinct victims), reason 8/9 rounds, by grp, regime, label:")
print(el.filter(pl.col("round").is_between(9, 16) & pl.col("grp").is_in(["ctrl_late", "early"])).group_by(
    ["grp", "regime", "round"]).agg(
    pl.len().alias("rounds"), pl.col("elim_ok").mean().round(3).alias("elim_ok_rate")).sort(["grp", "regime", "round"]))

# =====================================================================================
section("T6. Misattribution test: death-implied side at label 13 vs round_end at labels 12, 13, 14 (ctrl_late)")
RW_prev = R.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("rw").alias("rw_prev")])
RW_next = R.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("rw").alias("rw_next")])
t6 = (nb.filter(pl.col("round") == 13).join(RW_prev, on=["match_key", "round"], how="left")
      .join(RW_next, on=["match_key", "round"], how="left"))
t6 = t6.filter(pl.col("grp") == "ctrl_late").with_columns(
    eq_rw13=(pl.col("implied") == pl.col("rw")).cast(pl.Int8),
    eq_rw12=(pl.col("implied") == pl.col("rw_prev")).cast(pl.Int8),
    eq_rw14=(pl.col("implied") == pl.col("rw_next")).cast(pl.Int8))
print(t6.group_by("r1_end").agg(
    pl.len().alias("n"),
    pl.col("eq_rw13").mean().round(3).alias("implied13_eq_rw13"),
    pl.col("eq_rw12").mean().round(3).alias("implied13_eq_rw12"),
    pl.col("eq_rw14").mean().round(3).alias("implied13_eq_rw14")).sort("r1_end"))

print("\nDONE")
