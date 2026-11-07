"""D5 step 3: tests of the hypothesis that round numbering shifts (pause, restart,
knife, extra round) around round 13 in the affected 2024 matches.

Input: scratchpad D5/*.parquet written by D5_02_build.py (2024 tome only).
Groups:
  aff_r13     matches with a round-13 bomb-outcome mismatch (434; the defect set).
  ctrl_late   days 12-25 with no round-13 mismatch (outcome-selected control).
  early       days 01-07 (no mismatches at all; structural baseline).
ASSUMPTION: "affected" = the e2 definition (player_count == 20, bomb outcome vs
round_end winner). Groups aff_r13 vs ctrl_late are selected on the bomb outcome, so
bomb-based metrics are circular for the mismatch itself. Structure metrics (ticks,
counts, reason codes) are not selected on the bomb channel.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_03_tests.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(80)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5")
R0 = pl.read_parquet(SCR / "rounds.parquet")
M0 = pl.read_parquet(SCR / "matches.parquet")
B = pl.read_parquet(SCR / "bomb.parquet")
DE = pl.read_parquet(SCR / "deaths.parquet")

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
GROUPS = ["aff_r13", "ctrl_late", "early", "other"]

M = M0.with_columns(
    grp=pl.when(pl.col("aff_r13")).then(pl.lit("aff_r13"))
    .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
    .when(pl.col("day").is_in(EARLY)).then(pl.lit("early"))
    .otherwise(pl.lit("other")))
GRP = M.select(["match_key", "grp", "map_name", "day", "aff_r13", "final_sum", "match_date",
                "n_end_pairs", "max_end_round"])
print("== group sizes (matches) ==")
print(GRP.group_by("grp").agg(pl.len().alias("matches")).sort("grp"))

R = R0.join(GRP.select(["match_key", "grp", "map_name", "day", "aff_r13", "final_sum"]),
            on="match_key", how="left")
print("rounds union rows:", R.height)

# ------------------------------------------------------------------ D5-1
print("\n==================== D5-1 round numbering, counts, duplicates, extras ====================")
per = (R.group_by("match_key").agg(
    pl.col("in_end").sum().alias("n_end"),
    pl.col("in_start").sum().alias("n_start"),
    pl.col("in_fe").sum().alias("n_fe"),
    pl.col("round").filter(pl.col("in_end")).max().alias("max_end_label"),
    pl.col("round").max().alias("max_any_label"),
    (pl.col("n_end_rows").fill_null(0) > 1).sum().alias("n_dup_end_labels"),
    pl.col("round").filter(pl.col("in_end") & ~pl.col("in_start")).len().alias("n_end_no_start"),
    pl.col("round").filter(pl.col("in_fe") & ~pl.col("in_start")).len().alias("n_fe_no_start"),
    pl.col("round").filter(pl.col("in_start") & ~pl.col("in_end")).len().alias("n_start_no_end"),
    pl.col("round").filter(pl.col("in_fe") & ~pl.col("in_end")).len().alias("n_fe_no_end"),
)
    .join(GRP.select(["match_key", "grp", "final_sum", "day", "map_name", "aff_r13"]), on="match_key", how="left")
    .with_columns(
        missing_end_labels=pl.col("max_end_label") - pl.col("n_end"),
        end_minus_final=pl.col("n_end") - pl.col("final_sum"),
    ))
summ1 = per.group_by("grp").agg(
    pl.len().alias("matches"),
    (pl.col("end_minus_final") != 0).sum().alias("end_rows_ne_final_score_sum"),
    (pl.col("missing_end_labels") > 0).sum().alias("missing_end_label_matches"),
    (pl.col("n_dup_end_labels") > 0).sum().alias("dup_end_label_matches"),
    (pl.col("n_end_no_start") > 0).sum().alias("matches_end_without_start"),
    (pl.col("n_fe_no_start") > 0).sum().alias("matches_fe_without_start"),
    (pl.col("n_start_no_end") > 0).sum().alias("matches_start_without_end"),
    pl.col("n_fe_no_start").sum().alias("total_fe_without_start_rounds"),
    pl.col("n_end_no_start").sum().alias("total_end_without_start_rounds"),
    pl.col("n_fe_no_end").sum().alias("total_fe_without_end_rounds"),
    pl.col("n_end").mean().round(2).alias("mean_end_rounds"),
    pl.col("n_start").mean().round(2).alias("mean_start_rounds"),
    pl.col("n_fe").mean().round(2).alias("mean_fe_rounds"),
).sort("grp")
print(summ1)
print("\nend_rows minus header final score sum, by group (value counts):")
print(per.group_by(["grp", "end_minus_final"]).agg(pl.len().alias("matches")).sort(["grp", "end_minus_final"]))
print("\nmissing end labels below max end label, by group (value counts):")
print(per.group_by(["grp", "missing_end_labels"]).agg(pl.len().alias("matches")).sort(["grp", "missing_end_labels"]))
print("\nmatches with a duplicate round_end (match, round) pair:")
print(per.filter(pl.col("n_dup_end_labels") > 0).select(["match_key", "grp", "n_end", "final_sum", "n_dup_end_labels"]))

# rounds 11..15 presence per group
rr = (R.filter(pl.col("round").is_between(11, 15))
      .group_by(["grp", "round"]).agg(
          pl.len().alias("match_rows"),
          pl.col("in_start").sum().alias("with_round_start_row"),
          pl.col("in_fe").sum().alias("with_freeze_end"),
          pl.col("in_end").sum().alias("with_round_end"),
          (pl.col("n_end_rows").fill_null(0) > 1).sum().alias("dup_round_end"),
      ).sort(["grp", "round"]))
print("\nrounds 11..15 presence by group (match_rows = matches whose label set includes the round):")
print(rr)

# ------------------------------------------------------------------ D5-2
print("\n==================== D5-2 tick ordering across consecutive rounds ====================")
RA = R.sort(["match_key", "round"]).with_columns(
    prev_round=pl.col("round").shift(1).over("match_key"),
    prev_end=pl.col("end_tick").shift(1).over("match_key"),
    prev_end_label=pl.col("round").shift(1).over("match_key"),
)
RA = RA.with_columns(
    cons=(pl.col("round") - pl.col("prev_round") == 1),
    end_not_increasing=(pl.col("in_end") & pl.col("prev_end").is_not_null()
                        & (pl.col("end_tick") <= pl.col("prev_end"))),
    start_before_prev_end=(pl.col("in_start") & pl.col("prev_end").is_not_null()
                           & (pl.col("start_tick") <= pl.col("prev_end"))),
    start_after_end=(pl.col("in_start") & pl.col("in_end") & (pl.col("start_tick") >= pl.col("end_tick"))),
    fe_outside_start_end=(pl.col("in_fe") & pl.col("in_start") & pl.col("in_end")
                          & ((pl.col("fe_tick") < pl.col("start_tick")) | (pl.col("fe_tick") > pl.col("end_tick")))),
    fe_before_start=(pl.col("in_fe") & pl.col("in_start") & (pl.col("fe_tick") < pl.col("start_tick"))),
)
flags = ["end_not_increasing", "start_before_prev_end", "start_after_end", "fe_outside_start_end"]
print("matches-level flag counts by group (any round in the match):")
mflags = RA.group_by(["grp", "match_key"]).agg([pl.col(f).fill_null(False).any().alias(f) for f in flags])
print(mflags.group_by("grp").agg([pl.len().alias("matches")] + [pl.col(f).sum().alias(f) for f in flags]).sort("grp"))
print("\nround-level flag counts by group and label for labels 11..15:")
rl = (RA.filter(pl.col("round").is_between(11, 15))
      .group_by(["grp", "round"]).agg([pl.col(f).fill_null(False).sum().alias(f) for f in flags]
                                      + [pl.col("cons").fill_null(False).sum().alias("consecutive_pairs")])
      .sort(["grp", "round"]))
print(rl)
print("\nlabel gaps (round - prev_round != 1) among rounds present, by group, labels 2..30 (value counts):")
gaps = (RA.filter(pl.col("prev_round").is_not_null())
        .with_columns((pl.col("round") - pl.col("prev_round")).alias("step"))
        .group_by(["grp", "step"]).agg(pl.len().alias("n")).sort(["grp", "step"]))
print(gaps)

# ------------------------------------------------------------------ D5-3
print("\n==================== D5-3 bomb / death tick -> round window mapping ====================")
E = (R.filter(pl.col("in_end")).select(["match_key", "round", "end_tick"])
     .rename({"round": "win_round"}).sort("end_tick"))
own = R.select(["match_key", "round", "start_tick", "fe_tick", "end_tick"]).rename({"end_tick": "own_end"})


def window_map(ev, name):
    x = ev.sort("tick").join_asof(E, left_on="tick", right_on="end_tick", by="match_key",
                                  strategy="forward")
    x = x.with_columns(shift=(pl.col("win_round") - pl.col("round")))
    x = x.join(own, on=["match_key", "round"], how="left")
    x = x.with_columns(
        in_own_window=(pl.col("tick") <= pl.col("own_end"))
        & (pl.col("tick") >= pl.coalesce(["fe_tick", "start_tick"])))
    x = x.join(GRP.select(["match_key", "grp"]), on="match_key", how="left")
    print(f"\n-- {name}: events {x.height}; window label missing (no later round_end): "
          f"{x.filter(pl.col('win_round').is_null()).height}")
    print("shift (window_round - labelled round) value counts, by group:")
    print(x.group_by(["grp", "shift"]).agg(pl.len().alias("events")).sort(["grp", "shift"]))
    print("tick inside labelled round's [freeze_end or start, round_end] window, by group:")
    print(x.group_by("grp").agg(pl.len().alias("events"),
                                pl.col("in_own_window").sum().alias("in_own_window")).sort("grp"))
    return x


BO = B.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bo_map = window_map(BO, "bomb outcome (defused/exploded) rows from bomb_state")
pl_map = window_map(B.filter(pl.col("event_type") == "bomb_planted"), "bomb_planted rows")
de_map = window_map(DE, "player_death rows (control channel)")

print("\nbomb outcome rows at labelled round 13, by group and shift:")
print(bo_map.filter(pl.col("round") == 13).group_by(["grp", "shift"]).agg(pl.len().alias("events"))
      .sort(["grp", "shift"]))
print("\nbomb outcome rows at labelled round 13 in aff_r13, shift x in_own_window:")
print(bo_map.filter((pl.col("round") == 13) & (pl.col("grp") == "aff_r13"))
      .group_by(["shift", "in_own_window"]).agg(pl.len().alias("events")).sort(["shift", "in_own_window"]))
print("\nbomb outcome rows at labelled rounds 9, 13, 25, 28 (all groups), shift value counts:")
print(bo_map.filter(pl.col("round").is_in([9, 13, 25, 28])).group_by(["round", "shift"])
      .agg(pl.len().alias("events")).sort(["round", "shift"]))
print("\nexample aff_r13 bomb outcome rows (first 12 by match):")
print(bo_map.filter((pl.col("round") == 13) & (pl.col("grp") == "aff_r13"))
      .select(["match_key", "event_type", "tick", "round", "win_round", "shift", "own_end", "in_own_window",
               "player_team_code"]).sort("match_key").head(12))

# ------------------------------------------------------------------ D5-4
print("\n==================== D5-4 score-chain consistency (round_state freeze_end scores) ====================")
SCHAIN = (R.sort(["match_key", "round"]).with_columns(
    nxt_round=pl.col("round").shift(-1).over("match_key"),
    nxt_t=pl.col("fe_t_score").shift(-1).over("match_key"),
    nxt_ct=pl.col("fe_ct_score").shift(-1).over("match_key"))
    .filter((pl.col("nxt_round") == pl.col("round") + 1)
            & pl.col("fe_t_score").is_not_null() & pl.col("nxt_t").is_not_null())
    .with_columns(dt=pl.col("nxt_t") - pl.col("fe_t_score"), dct=pl.col("nxt_ct") - pl.col("fe_ct_score"))
    .with_columns(chain_winner=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(pl.lit(2))
                  .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(pl.lit(3)).otherwise(pl.lit(0)))
    .with_columns(agree=(pl.col("chain_winner") == pl.col("winner")))
    .filter(pl.col("winner").is_not_null()))
print("chain pairs with round_end winner:", SCHAIN.height)
print("chain_winner (0 = score change not a single-team +1) value counts, all groups:")
print(SCHAIN.group_by("chain_winner").agg(pl.len().alias("pairs")).sort("chain_winner"))
print("\nagreement of score-chain winner with round_end winner, by group and label 1..30:")
sc_t = (SCHAIN.group_by(["grp", "round"]).agg(pl.len().alias("pairs"),
                                               pl.col("agree").sum().alias("agree"),
                                               (pl.col("chain_winner") == 0).sum().alias("chain_zero"))
        .with_columns((pl.col("agree") / pl.col("pairs")).round(4).alias("agree_rate"))
        .sort(["grp", "round"]))
print(sc_t.filter(pl.col("round").is_between(9, 16)))
print(sc_t.filter(pl.col("round").is_in([25, 26, 27, 28, 29])))

BSIDE = (B.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
         .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
         .with_columns(bomb_side=pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3))
                       .otherwise(pl.lit(2))))
S13 = (SCHAIN.filter(pl.col("round") == 13).join(BSIDE.select(["match_key", "round", "bomb_side"]),
                                                  on=["match_key", "round"], how="left"))
print("\nlabel 13 transitions (score 13 -> 14), aff_r13 vs others: chain_winner x round_end winner x bomb_side")
print(S13.with_columns(pl.when(pl.col("grp") == "aff_r13").then(pl.lit("aff_r13")).otherwise(pl.lit("other")).alias("g2"))
      .group_by(["g2", "chain_winner", "winner", "bomb_side"]).agg(pl.len().alias("n"))
      .sort(["g2", "chain_winner", "winner", "bomb_side"], nulls_last=True))
print("\nlabel 12 transitions (score 12 -> 13) for comparison (halftime boundary side for half 2 start)")
S12 = (SCHAIN.filter(pl.col("round") == 12).join(BSIDE.select(["match_key", "round", "bomb_side"]),
                                                  on=["match_key", "round"], how="left"))
print(S12.group_by(["grp", "chain_winner", "winner"]).agg(pl.len().alias("n")).sort(["grp", "chain_winner", "winner"]))

# ------------------------------------------------------------------ D5-5
print("\n==================== D5-5 win_reason / legacy_code consistency ====================")
T_REASON = [1, 9, 17]
CT_REASON = [7, 8, 12, 18]
RR = R.filter(pl.col("in_end")).with_columns(
    reason_side=pl.when(pl.col("reason_code").is_in(T_REASON)).then(pl.lit(2))
    .when(pl.col("reason_code").is_in(CT_REASON)).then(pl.lit(3)).otherwise(pl.lit(0)))
RR = RR.with_columns(reason_vs_winner_ok=(pl.col("reason_side") == pl.col("winner")))
print("reason_side (0 = other reason) vs round_end winner, by group:")
print(RR.group_by(["grp", "reason_side", "winner"]).agg(pl.len().alias("rounds"))
      .sort(["grp", "reason_side", "winner"]))
print("\nreason_code x message x winner, all groups (rounds):")
print(RR.group_by(["reason_code", "reason_msg", "winner"]).agg(pl.len().alias("rounds"))
      .sort(["reason_code", "winner"]))
print("\nlegacy_code value counts by group:")
print(RR.group_by(["grp", "legacy_code"]).agg(pl.len().alias("rounds")).sort(["grp", "legacy_code"]))
print("\nlegacy_code != 0 rows: round label and group (all):")
lc = RR.filter(pl.col("legacy_code") != 0)
print(lc.group_by(["grp", "round"]).agg(pl.len().alias("rounds")).sort(["grp", "round"]))
print("\nreason_code / winner at labelled rounds 11..15 by group:")
print(RR.filter(pl.col("round").is_between(11, 15)).group_by(["grp", "round", "reason_code", "winner"])
      .agg(pl.len().alias("rounds")).sort(["grp", "round", "reason_code", "winner"]))

# ------------------------------------------------------------------ D5-6
print("\n==================== D5-6 pause / restart / knife / extra-round probes ====================")
PROBE_CNT = [c for c in R.columns if c.startswith("n_") and c not in
             ("n_start_rows", "n_fe_rows", "n_end_rows", "n_start", "n_fe")]
print("probe columns:", PROBE_CNT)
pm = (R.group_by("match_key").agg([pl.col(c).fill_null(0).sum().alias(c) for c in PROBE_CNT])
      .join(GRP.select(["match_key", "grp"]), on="match_key", how="left"))
rows = []
for c in PROBE_CNT:
    g = pm.group_by("grp").agg(
        pl.col(c).sum().alias("total"),
        (pl.col(c) > 0).sum().alias("matches_with_any"),
        pl.len().alias("matches")).sort("grp")
    g = g.with_columns(pl.lit(c).alias("probe"))
    rows.append(g)
print(pl.concat(rows).select(["probe", "grp", "matches", "matches_with_any", "total"]).sort(["probe", "grp"]))

print("\nround-level: restart-type events per labelled round, labels 11..15, by group:")
RP = R.filter(pl.col("round").is_between(11, 15)).with_columns(
    prestart_not_one=(pl.col("n_round_prestart").fill_null(0) != 1),
    officially_ended_not_one=(pl.col("n_round_officially_ended").fill_null(0) != 1),
    any_pre_restart=(pl.col("n_cs_pre_restart").fill_null(0) > 0),
    any_win_panel_round=(pl.col("n_cs_win_panel_round").fill_null(0) > 0),
    no_round_start_state=(pl.col("n_round_start").fill_null(0) == 0),
)
print(RP.group_by(["grp", "round"]).agg(
    pl.len().alias("rows"),
    pl.col("prestart_not_one").sum().alias("prestart_count_not_1"),
    pl.col("officially_ended_not_one").sum().alias("officially_ended_count_not_1"),
    pl.col("any_pre_restart").sum().alias("any_cs_pre_restart"),
    pl.col("any_win_panel_round").sum().alias("any_cs_win_panel_round"),
    pl.col("no_round_start_state").sum().alias("no_round_start_state_row"),
).sort(["grp", "round"]))

print("\nphase at freeze_end for labels 12, 13, 14 by group (value counts):")
print(R.filter(pl.col("round").is_in([12, 13, 14]) & pl.col("in_fe"))
      .group_by(["grp", "round", "fe_phase"]).agg(pl.len().alias("rounds")).sort(["grp", "round", "fe_phase"]))

# ------------------------------------------------------------------ D5-7 maps
print("\n==================== D5-7 map distribution (secondary) ====================")
mp = (M.filter(pl.col("day").is_in(LATE)).group_by("map_name").agg(
    pl.len().alias("matches_days12_25"),
    pl.col("aff_r13").sum().alias("aff_r13_matches"))
    .with_columns((pl.col("aff_r13_matches") / pl.col("matches_days12_25")).round(3).alias("aff_rate"))
    .sort("matches_days12_25", descending=True))
print(mp)
mp2 = (M.group_by("map_name").agg(pl.len().alias("all_matches"), pl.col("aff_r13").sum().alias("aff")))
print(mp2.sort("all_matches", descending=True))
print("\nround-13 mismatch count by map (aff matches) and map-level late matches with no r13 bomb outcome:")
print(per.filter(pl.col("aff_r13")).group_by("map_name").agg(pl.len().alias("aff_matches")).sort("aff_matches", descending=True))

# ------------------------------------------------------------------ extra: match-level check on final score
print("\n==================== D5-1b header final score vs round_end rows in aff matches ====================")
aff_end = per.filter(pl.col("aff_r13")).select(["match_key", "n_end", "final_sum", "max_end_label", "n_dup_end_labels"])
print("aff matches: n_end == final_sum:", aff_end.filter(pl.col("n_end") == pl.col("final_sum")).height,
      "of", aff_end.height)
print("aff matches with max_end_label > 24 (overtime):", aff_end.filter(pl.col("max_end_label") > 24).height)
print(aff_end.group_by("max_end_label").agg(pl.len().alias("matches")).sort("max_end_label"))
