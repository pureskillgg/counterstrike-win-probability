"""D3-2 independent verifier: 2024 round-13 bomb-side vs round_end winner mismatch.

Read-only on the G: 2024 tome (round_end, bomb_state, round_state). Prints to stdout; the log is
written by the shell redirection to logs\\swarm2\\verify_D3_2.log.

Assumptions (ASSUMPTION, marked again where used):
  A1 bomb_defused => CT (3) won the round; bomb_exploded => T (2) won the round (CS2 semantics).
  A2 score side = sign of the change in freeze_end score between round r and round r+1
     (t_score / ct_score = cleaned columns; t_score_raw / ct_score_raw checked as a variant).
  A3 round_end rows used: player_count == 20, first row per (match, round) by tick.
  A4 bomb outcome tick window for round r: [round_freeze_end tick of r, round_end tick of r].
  A5 day = DD from match_key csds/2024/02/DD/...
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_2_main.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def load(ch, cols):
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    assert fs, f"no files for {ch}"
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


RE = load("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
BS = load("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
RS = load("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                          "t_score_raw", "ct_score_raw"])
print(f"rows: round_end {RE.height}  bomb_state {BS.height}  round_state {RS.height}")
print(f"matches: round_end {RE['match_key'].n_unique()}  round_state {RS['match_key'].n_unique()}")

DAY = pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64)

# ---------------------------------------------------------------- 0. raw winner/reason crosstab
print("\n== 0. round_end winner_team_code x win_reason_code (all rows, all player_count)")
print(RE.group_by(["winner_team_code", "win_reason_code"]).len().sort(["winner_team_code", "win_reason_code"]))

# ---------------------------------------------------------------- 1. multiplicity checks
print("\n== 1. multiplicity per (match, round)")
re_per = RE.group_by(["match_key", "round"]).len()
print("round_end rows per (match,round), all player_count:")
print(re_per.group_by("len").agg(pl.len().alias("n_groups")).sort("len"))
re20 = RE.filter(pl.col("player_count") == 20)
re20_per = re20.group_by(["match_key", "round"]).len()
print("round_end rows per (match,round), player_count==20:")
print(re20_per.group_by("len").agg(pl.len().alias("n_groups")).sort("len"))
print("rounds with a player_count!=20 round_end row AND a player_count==20 row:",
      RE.filter(pl.col("player_count") != 20).select(["match_key", "round"]).unique()
      .join(re20_per.select(["match_key", "round"]), on=["match_key", "round"], how="inner").height)

OUT = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
outc = (OUT.group_by(["match_key", "round"])
        .agg(pl.col("event_type").unique().alias("evs"),
             pl.col("tick").min().alias("out_tick"),
             pl.col("tick").max().alias("out_tick_max"),
             pl.len().alias("out_n"))
        .with_columns(pl.col("evs").list.contains("bomb_defused").alias("has_def"),
                      pl.col("evs").list.contains("bomb_exploded").alias("has_exp")))
print("\nbomb outcome rows per (match,round):")
print(outc.group_by("out_n").len().sort("out_n"))
print("(match,round) with both defused and exploded:", outc.filter(pl.col("has_def") & pl.col("has_exp")).height)

fe_per = (RS.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).len())
print("round_freeze_end rows per (match,round):")
print(fe_per.group_by("len").agg(pl.len().alias("n_groups")).sort("len"))

PL = (BS.filter(pl.col("event_type") == "bomb_planted").sort("tick")
      .group_by(["match_key", "round"])
      .agg(pl.col("tick").first().alias("plant_tick"), pl.col("player_team_code").first().alias("planter"),
           pl.len().alias("plant_n")))
print("bomb_planted rows per (match,round):")
print(PL.group_by("plant_n").len().sort("plant_n"))


# ---------------------------------------------------------------- 2. score change (cleaned and raw)
def score_table(tcol, ccol, tag):
    fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
          .group_by(["match_key", "round"])
          .agg(pl.col(tcol).first().alias("t_b"), pl.col(ccol).first().alias("c_b"),
               pl.col("tick").first().alias("fe_tick")))
    nxt = fe.select(["match_key", (pl.col("round") - 1).alias("round"),
                     pl.col("t_b").alias("t_a"), pl.col("c_b").alias("c_a"),
                     pl.col("fe_tick").alias("fe_next_tick")])
    s = fe.join(nxt, on=["match_key", "round"], how="left")
    s = s.with_columns((pl.col("t_a") - pl.col("t_b")).alias("dt"),
                       (pl.col("c_a") - pl.col("c_b")).alias("dc"))
    s = s.with_columns(
        pl.when((pl.col("dc") == 1) & (pl.col("dt") == 0)).then(pl.lit(CT, dtype=pl.Int64))
        .when((pl.col("dt") == 1) & (pl.col("dc") == 0)).then(pl.lit(T, dtype=pl.Int64))
        .otherwise(pl.lit(None, dtype=pl.Int64)).alias(f"score_side{tag}"))
    return s.select(["match_key", "round", "fe_tick", "fe_next_tick", f"score_side{tag}",
                     pl.col("dt").alias(f"dt{tag}"), pl.col("dc").alias(f"dc{tag}")])


SC = score_table("t_score", "ct_score", "")
SCR = score_table("t_score_raw", "ct_score_raw", "_raw")
print("\nscore_side coverage: cleaned", SC["score_side"].is_not_null().sum(), "/", SC.height,
      " raw", SCR["score_side_raw"].is_not_null().sum(), "/", SCR.height)
print("cleaned score != raw score on freeze_end rows:",
      RS.filter(pl.col("event_type") == "round_freeze_end")
      .filter((pl.col("t_score") != pl.col("t_score_raw")) | (pl.col("ct_score") != pl.col("ct_score_raw")))
      .height, "of", RS.filter(pl.col("event_type") == "round_freeze_end").height)

ROE = (RS.filter(pl.col("event_type") == "round_officially_ended")
       .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("roe_tick")))

# ---------------------------------------------------------------- 3. per-(match, round) table
REd = (re20.sort("tick").group_by(["match_key", "round"])
       .agg(pl.col("winner_team_code").first().alias("re_w"),
            pl.col("win_reason_code").first().alias("re_reason"),
            pl.col("tick").first().alias("re_tick")))
ALL = (REd.join(outc, on=["match_key", "round"], how="left")
       .join(PL, on=["match_key", "round"], how="left")
       .join(SC, on=["match_key", "round"], how="left")
       .join(SCR.select(["match_key", "round", "score_side_raw", "dt_raw", "dc_raw"]),
             on=["match_key", "round"], how="left")
       .join(ROE, on=["match_key", "round"], how="left")
       .with_columns(pl.col("has_def").fill_null(False), pl.col("has_exp").fill_null(False))
       .with_columns(DAY.alias("day"),
                     pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                     .when(pl.col("round") == 25).then(pl.lit("r25"))
                     .otherwise(pl.lit("other")).alias("rbin")))

B = (ALL.filter(pl.col("has_def") | pl.col("has_exp"))
     .with_columns(
         pl.when(pl.col("has_def")).then(pl.lit(CT, dtype=pl.Int64)).otherwise(pl.lit(T, dtype=pl.Int64))
         .alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("re_w")).alias("mm")))

print(f"\n== 3. replication of the claim (player_count==20, bomb outcome present)")
print(f"base rounds: {B.height}   mismatches: {int(B['mm'].sum())}   (claim: 15436 / 499)")
r13_late = B.filter((pl.col("rbin") == "r13") & (pl.col("day") >= 12))
print(f"round 13, days 12-25: n={r13_late.height} mm={int(r13_late['mm'].sum())}  (claim: 605 / 434)")
r13_early = B.filter((pl.col("rbin") == "r13") & (pl.col("day") < 12))
print(f"round 13, days 01-07: n={r13_early.height} mm={int(r13_early['mm'].sum())}  (claim: 180 / 0)")
print("planter side on mismatch rounds:", B.filter(pl.col("mm"))["planter"].value_counts().rows())
print("planter side on agreeing rounds:", B.filter(~pl.col("mm"))["planter"].value_counts().rows())

rr = B.filter((pl.col("rbin") == "r13") & (pl.col("day") >= 12) & pl.col("mm"))
print("\nmismatch round-13 rows: (bomb_side, re_w, re_reason) counts")
print(rr.group_by(["bomb_side", "re_w", "re_reason"]).len().sort("len", descending=True))
print("mismatch round-13 rows: score_side vs re_w vs bomb_side")
print(rr.group_by(["score_side", "re_w", "bomb_side"], maintain_order=True).len())
print("mismatch round-13 rows: score_side_raw vs re_w vs bomb_side")
print(rr.group_by(["score_side_raw", "re_w", "bomb_side"], maintain_order=True).len())

ag = B.filter((pl.col("rbin") == "r13") & (pl.col("day") >= 12) & ~pl.col("mm"))
print("\nagreeing round-13 rows (days 12-25): re_reason x bomb_side")
print(ag.group_by(["bomb_side", "re_reason"]).len().sort(["bomb_side", "re_reason"]))
print("mismatch round-13 rows: re_reason is a bomb reason (1 or 7)?",
      rr.filter(pl.col("re_reason").is_in([1, 7])).height, "of", rr.height)

# ---------------------------------------------------------------- 4. tick-window placement of the bomb event
print("\n== 4. tick-window placement (A4): outcome tick inside round window?")
B2 = B.with_columns(
    ((pl.col("out_tick") >= pl.col("fe_tick")) & (pl.col("out_tick") <= pl.col("re_tick"))).alias("out_in_window"),
    (pl.col("out_tick") <= pl.col("roe_tick")).alias("out_before_roe"),
    (pl.col("re_tick") < pl.col("fe_next_tick")).alias("re_before_next_fe"),
    (pl.col("plant_tick") <= pl.col("out_tick")).alias("plant_before_out"),
    (pl.col("out_tick") - pl.col("re_tick")).alias("out_minus_re"),
)
for rb, dayf, lab in [("r13", pl.col("day") >= 12, "r13 days12-25"), ("r13", pl.col("day") < 12, "r13 days01-07"),
                      ("r25", pl.col("day") >= 12, "r25 days12-25"), ("other", pl.col("day") >= 12, "other days12-25")]:
    sub = B2.filter((pl.col("rbin") == rb) & dayf)
    for flag in [True, False]:
        s = sub.filter(pl.col("mm") == flag)
        if s.height == 0:
            continue
        print(f"{lab:<16} mm={str(flag):<5} n={s.height:>4} out_in_window={int(s['out_in_window'].sum()):>4} "
              f"out_before_roe={int(s['out_before_roe'].sum()):>4} re_before_next_fe={int(s['re_before_next_fe'].sum()):>4} "
              f"plant_before_out={int(s['plant_before_out'].sum()):>4} "
              f"median(out-re_tick)={s['out_minus_re'].median()}")
print("\nmismatch round-13 days12-25: out_tick - re_tick distribution (ticks)")
m13 = B2.filter((pl.col("rbin") == "r13") & (pl.col("day") >= 12) & pl.col("mm"))
print(m13.select("out_minus_re").describe())
print("mismatch round-13 days12-25: re_tick - fe_tick distribution (ticks)")
print(m13.select((pl.col("re_tick") - pl.col("fe_tick")).alias("re_minus_fe")).describe())
print("mismatch round-13 days12-25: out_tick - fe_tick distribution (ticks)")
print(m13.select((pl.col("out_tick") - pl.col("fe_tick")).alias("out_minus_fe")).describe())
print("agreeing round-13 days12-25: out_tick - re_tick distribution (ticks)")
print(B2.filter((pl.col("rbin") == "r13") & (pl.col("day") >= 12) & ~pl.col("mm"))
      .select("out_minus_re").describe())

# ---------------------------------------------------------------- 5. neighbouring-round shift test (A3)
print("\n== 5. neighbouring-round shift test: does round_end(13) equal the neighbours' result?")
Wre = ALL.select(["match_key", "round", "re_w"])
Wsc = ALL.select(["match_key", "round", "score_side"])
prev_re = Wre.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("re_w").alias("re_prev")])
next_re = Wre.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("re_w").alias("re_next")])
prev_sc = Wsc.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("score_side").alias("sc_prev")])
next_sc = Wsc.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("score_side").alias("sc_next")])
S5 = (B.filter((pl.col("rbin") == "r13") & (pl.col("day") >= 12))
      .join(prev_re, on=["match_key", "round"], how="left")
      .join(next_re, on=["match_key", "round"], how="left")
      .join(prev_sc, on=["match_key", "round"], how="left")
      .join(next_sc, on=["match_key", "round"], how="left"))
for flag in [True, False]:
    s = S5.filter(pl.col("mm") == flag)
    n = s.height
    print(f"mm={flag}: n={n}  re(r-1)==bomb {int((s['re_prev'] == s['bomb_side']).sum())}  "
          f"re(r+1)==bomb {int((s['re_next'] == s['bomb_side']).sum())}  "
          f"score(r-1)==bomb {int((s['sc_prev'] == s['bomb_side']).sum())}  "
          f"score(r+1)==bomb {int((s['sc_next'] == s['bomb_side']).sum())}  "
          f"re(r-1)==re(r) {int((s['re_prev'] == s['re_w']).sum())}")

# ---------------------------------------------------------------- 6. control: does the score chain agree with
# round_end on rounds without any bomb outcome, and on round 12 (spans halftime)?
print("\n== 6. score-chain vs round_end agreement, by round number (rounds with score_side, no bomb outcome)")
ctrl = ALL.filter(~(pl.col("has_def") | pl.col("has_exp")) & pl.col("score_side").is_not_null()
                  & (pl.col("day") >= 12))
agree_by_round = (ctrl.with_columns((pl.col("score_side") == pl.col("re_w")).alias("ok"))
                  .group_by("round").agg(pl.len().alias("n"), pl.col("ok").sum().alias("ok"))
                  .sort("round"))
print(agree_by_round.rows())
print("score_side vs re_w, all non-bomb rounds days 12-25:",
      int(ctrl.filter(pl.col("score_side") == pl.col("re_w")).height), "/", ctrl.height)

# ---------------------------------------------------------------- 7. final score reconciliation (A2)
print("\n== 7. final score (round_state, last row per match) vs round_end side totals")
fin = (RS.sort(["match_key", "tick"]).group_by("match_key", maintain_order=True)
       .agg(pl.col("t_score").last().alias("t_fin"), pl.col("ct_score").last().alias("ct_fin"),
            pl.col("t_score_raw").last().alias("t_fin_raw"), pl.col("ct_score_raw").last().alias("ct_fin_raw")))
tot = (REd.group_by("match_key").agg((pl.col("re_w") == T).sum().alias("T_re"),
                                     (pl.col("re_w") == CT).sum().alias("CT_re"),
                                     pl.len().alias("n_re")))
r13mm = (B.filter((pl.col("rbin") == "r13") & (pl.col("day") >= 12) & pl.col("mm"))
         .select("match_key").unique().with_columns(pl.lit(True).alias("has_r13_mm")))
M = (tot.join(fin, on="match_key", how="left").join(r13mm, on="match_key", how="left")
     .with_columns(pl.col("has_r13_mm").fill_null(False),
                   (pl.col("T_re") - pl.col("t_fin")).alias("dT"),
                   (pl.col("CT_re") - pl.col("ct_fin")).alias("dCT"),
                   (pl.col("T_re") - pl.col("t_fin_raw")).alias("dT_raw"),
                   (pl.col("CT_re") - pl.col("ct_fin_raw")).alias("dCT_raw")))
print("matches with any round_end row:", M.height, "  with r13 mismatch (days 12-25):", int(M["has_r13_mm"].sum()))
print("cleaned final score: (dT, dCT) distribution by r13-mismatch flag")
print(M.group_by(["has_r13_mm", "dT", "dCT"]).len().sort(["has_r13_mm", "len"], descending=[False, True]))
print("raw final score: (dT_raw, dCT_raw) distribution by r13-mismatch flag")
print(M.group_by(["has_r13_mm", "dT_raw", "dCT_raw"]).len().sort(["has_r13_mm", "len"], descending=[False, True]))

# ---------------------------------------------------------------- 8. per-day rate table for round 13
print("\n== 8. round 13 mismatch rate by day (replication of claim table)")
print(B.filter(pl.col("rbin") == "r13").group_by("day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
      .sort("day"))

# ---------------------------------------------------------------- 9. scope: is the defect bomb-specific?
print("\n== 9. scope: round_end vs score chain (cleaned), WITH and WITHOUT bomb outcome, by round bin and day range")
SCALL = (ALL.filter(pl.col("score_side").is_not_null())
         .with_columns((pl.col("score_side") != pl.col("re_w")).alias("dis"),
                       (pl.col("has_def") | pl.col("has_exp")).alias("has_bomb")))
for lab, cond in [("day01-07", pl.col("day") < 12), ("day12-25", pl.col("day") >= 12)]:
    sub = SCALL.filter(cond)
    print(lab)
    print(sub.group_by(["rbin", "has_bomb"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"))
          .sort(["rbin", "has_bomb"]))
r13nb = SCALL.filter((pl.col("round") == 13) & (pl.col("day") >= 12) & ~pl.col("has_bomb"))
print("round 13, days12-25, NO bomb outcome: round_end reason x disagreement with score chain")
print(r13nb.group_by(["dis", "re_reason"]).len().sort(["dis", "re_reason"]))
print("round 13, days12-25, NO bomb outcome: (score_side, re_w) counts")
print(r13nb.group_by(["score_side", "re_w"]).len().sort(["score_side", "re_w"]))
print("round 13 disagreement rate by day (score chain vs round_end), all rows with score_side:")
print(SCALL.filter(pl.col("rbin") == "r13").group_by("day")
      .agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"), pl.col("has_bomb").sum().alias("n_bomb"))
      .sort("day"))
print("round 13 days12-25 matches: count with >=1 round-13 disagreement vs total matches with round 13 score row:")
r13m = SCALL.filter((pl.col("round") == 13) & (pl.col("day") >= 12))
per_m = r13m.group_by("match_key").agg(pl.col("dis").any().alias("any_dis"))
print(per_m.group_by("any_dis").len())
print("round 9 / 25 / 27 / 28 days12-25 disagreement with score chain:")
print(SCALL.filter((pl.col("day") >= 12) & pl.col("round").is_in([9, 25, 27, 28]))
      .group_by("round").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).sort("round"))

# ---------------------------------------------------------------- 10. final-score basis check by period
print("\n== 10. final-score basis check by period (matches with round_end rows)")
anyd = SCALL.group_by("match_key").agg(
    pl.col("dis").any().alias("any_dis"),
    (pl.col("dis") & (pl.col("round") == 13)).any().alias("r13_dis"))
M3 = (M.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
      .with_columns(pl.when(pl.col("day") < 12).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("period"))
      .join(anyd, on="match_key", how="left")
      .with_columns(pl.col("any_dis").fill_null(False), pl.col("r13_dis").fill_null(False)))
print(M3.group_by(["period", "r13_dis", "any_dis"]).agg(
    pl.len().alias("n_matches"),
    (pl.col("dT") == 0).sum().alias("dT_eq0"),
    (pl.col("dT").abs() == 1).sum().alias("abs_dT_eq1"),
    pl.col("dT").abs().max().alias("max_abs_dT")).sort(["period", "r13_dis", "any_dis"]))

print("\n== DONE")
