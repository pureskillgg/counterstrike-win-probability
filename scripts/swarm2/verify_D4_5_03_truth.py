"""verify D4-5, step 3: which source is right for the disputed rounds? (2024 only; 2026 control at the end)

Checks (each prints its own table):
  C1 time alignment: is round_end.tick inside freeze_end(r) .. freeze_end(r+1) for the same round index?
  C2 bomb timing: for bomb_exploded / bomb_defused rows in a round, is the bomb tick inside the round window,
     and how far is it from round_end.tick? Reason-code crosstab against bomb outcome.
  C3 reason-code consistency: for disputed round-13 rows, does round_end (winner, win_reason_code) agree with
     the bomb outcome? Compare with the same crosstab for all other rounds.
  C4 header final score: does round_end (with swap convention) or round_state final score reproduce the
     header t_starters_score_final / ct_starters_score_final for the match? Split by disputed vs not.
  C5 last death: among elimination-ended rounds (reason 8 or 9), does the winner equal the team opposite the
     last victim (player_team_code of the latest player_death)? Split by round and by disputed.
ASSUMPTION: swap_expected with mr = 24 for 2024 (no max_rounds column in 2024 header).
ASSUMPTION: player_death.player_team_code is a side code (2 = T, 3 = CT) at the time of death.
Run via the project interpreter. Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, day_expr, swap_expected_expr, T, CT  # noqa: E402

pl.Config.set_tbl_rows(70)
pl.Config.set_tbl_cols(16)

base = TOMES["2024"]
MR = 24

# ---------- round-level table ----------
rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
fe_all = rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
fe = fe_all.unique(["match_key", "round"], keep="first").select(
    ["match_key", "round", "tick", "t_score", "ct_score"]).rename(
    {"tick": "fe_tick", "t_score": "t0", "ct_score": "c0"})
fe_next = fe.with_columns((pl.col("round") - 1).alias("round")).rename(
    {"fe_tick": "fe1_tick", "t0": "t1", "c0": "c1"})
D = fe.join(fe_next, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t1") - pl.col("t0")).alias("dT"), (pl.col("c1") - pl.col("c0")).alias("dC"))
D = D.with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"),
    swap_expected_expr(pl.lit(MR)).alias("sw"))

re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                               "player_count"]).collect()
re_ = re_.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first").rename(
    {"tick": "re_tick"})

# bomb outcome rows (round attribute as recorded, plus tick)
bs = scan(base, "bomb_state", ["match_key", "round", "tick", "event_type"]).collect()
bo = bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
bo_first = bo.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("event_type").first().alias("bomb_ev"), pl.col("tick").first().alias("bomb_tick"),
    pl.len().alias("n_bomb_out"), pl.col("event_type").n_unique().alias("n_bomb_types"))
pl_first = bs.filter(pl.col("event_type") == "bomb_planted").sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("tick").first().alias("plant_tick"))

# round-level join
R = (D.join(re_.select(["match_key", "round", "re_tick", "winner_team_code", "win_reason_code", "player_count"]),
            on=["match_key", "round"], how="left")
     .join(bo_first, on=["match_key", "round"], how="left")
     .join(pl_first, on=["match_key", "round"], how="left")
     .with_columns(day_expr().alias("day"),
                   pl.when(pl.col("bomb_ev") == "bomb_exploded").then(pl.lit(T))
                   .when(pl.col("bomb_ev") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(0)).alias("bside")))
R = R.with_columns(
    (pl.col("score_side") != pl.col("winner_team_code")).alias("dis_re"),
    (pl.col("re_tick") > pl.col("fe_tick")).alias("re_after_fe"),
    (pl.col("re_tick") < pl.col("fe1_tick")).alias("re_before_fe1"),
    pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).when(pl.col("day").is_between(12, 25))
    .then(pl.lit("d12-25")).otherwise(pl.lit("other")).alias("grp"),
)
valid = R.filter((~pl.col("sw")) & (pl.col("score_side") != 0) & pl.col("winner_team_code").is_not_null())
print(f"valid unambiguous rounds with round_end: {valid.height}")

# ---------- C1 time alignment ----------
print("\n== C1 time alignment (round_end tick must sit between freeze_end(r) and freeze_end(r+1))")
c1 = valid.group_by(["grp", "round"]).agg(
    pl.len().alias("n"),
    (~pl.col("re_after_fe")).sum().alias("re_not_after_fe"),
    (~pl.col("re_before_fe1")).sum().alias("re_not_before_fe1"),
).filter((pl.col("round") == 13) | (pl.col("re_not_after_fe") + pl.col("re_not_before_fe1") > 0))
print(c1.sort(["grp", "round"]))
print("all groups, all rounds, violations:",
      int((~valid["re_after_fe"]).sum()), "/", int((~valid["re_before_fe1"]).sum()))

# ---------- C2 bomb timing ----------
print("\n== C2 bomb outcome rows: tick inside round window and distance to round_end (seconds at 64 ticks/s)")
Bw = R.filter(pl.col("bside") != 0).with_columns(
    ((pl.col("bomb_tick") >= pl.col("fe_tick")) & (pl.col("bomb_tick") <= pl.col("re_tick"))).alias("in_win"),
    ((pl.col("re_tick") - pl.col("bomb_tick")) / 64.0).alias("re_minus_bomb_s"),
    ((pl.col("bomb_tick") - pl.col("plant_tick")) / 64.0).alias("bomb_minus_plant_s"),
)
Bw_valid = Bw.filter((~pl.col("sw")) & (pl.col("score_side") != 0))
for label, sub in [("round 13, d12-25", Bw_valid.filter((pl.col("round") == 13) & (pl.col("grp") == "d12-25"))),
                   ("other rounds, d12-25", Bw_valid.filter((pl.col("round") != 13) & (pl.col("grp") == "d12-25"))),
                   ("all rounds, d01-07", Bw_valid.filter(pl.col("grp") == "d01-07"))]:
    print(f"\n-- {label}: n={sub.height}")
    if sub.height == 0:
        continue
    print(sub.group_by(["bomb_ev", "dis_re"]).agg(
        pl.len().alias("n"),
        pl.col("in_win").mean().alias("frac_in_window"),
        pl.col("re_minus_bomb_s").median().alias("median_re_minus_bomb_s"),
        pl.col("re_minus_bomb_s").min().alias("min_re_minus_bomb_s"),
        pl.col("re_minus_bomb_s").max().alias("max_re_minus_bomb_s"),
        pl.col("bomb_minus_plant_s").median().alias("median_bomb_minus_plant_s"),
    ).sort(["bomb_ev", "dis_re"]))

print("\n-- round 13 d12-25 bomb rows: winner_team_code by bomb outcome and reason code")
s13 = Bw_valid.filter((pl.col("round") == 13) & (pl.col("grp") == "d12-25"))
print(s13.group_by(["bomb_ev", "win_reason_code", "winner_team_code", "score_side"]).len().sort(
    ["bomb_ev", "win_reason_code", "winner_team_code", "score_side"]))

print("\n-- reason x winner crosstab for ALL bomb rows, all rounds, d12-25 (reference)")
ref = Bw.filter(pl.col("grp") == "d12-25")
print(ref.group_by(["bomb_ev", "win_reason_code", "winner_team_code"]).len().sort(
    ["bomb_ev", "win_reason_code", "winner_team_code"]))

# ---------- C3 reason-code consistency, all rounds ----------
print("\n== C3 round_end internal consistency: reason code vs winner (all 2024 rows, d12-25 split by round 13)")
rr = R.filter(pl.col("grp") == "d12-25").with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rb"))
print(rr.group_by(["rb", "win_reason_code", "winner_team_code"]).len().sort(["rb", "win_reason_code", "winner_team_code"]))

# ---------- C4 header final score ----------
print("\n== C4 header final score vs round_end-with-convention and round_state final score")
H = header(base, ["key", "t_starters_score_final", "ct_starters_score_final"])
H = H.rename({"key": "match_key", "t_starters_score_final": "hT", "ct_starters_score_final": "hC"})
# swaps before round r: s(r) = sum_{k<r} swap_expected(k, 24); team X (started T) is on T in round r iff s(r) even
rounds = R.select(["match_key", "round", "winner_team_code", "re_tick"]).filter(
    pl.col("winner_team_code").is_not_null()).sort(["match_key", "round"])
rounds = rounds.with_columns(swap_expected_expr(pl.lit(MR)).cast(pl.Int64).alias("swf"))
rounds = rounds.with_columns((pl.col("swf").cum_sum().over("match_key") - pl.col("swf")).alias("s_before"))
rounds = rounds.with_columns(
    (((pl.col("winner_team_code") == T) & (pl.col("s_before") % 2 == 0)) |
     ((pl.col("winner_team_code") == CT) & (pl.col("s_before") % 2 == 1))).cast(pl.Int64).alias("x_won"))
tot = rounds.group_by("match_key").agg(pl.len().alias("n_rounds"), pl.col("x_won").sum().alias("xRE"),
                                       pl.col("round").max().alias("last_round"))
tot = tot.with_columns((pl.col("n_rounds") - pl.col("xRE")).alias("yRE"))
# round_state final cumulative score at the last round index's freeze end (score fields are side-based;
# team X is on T in the last round iff s(last) even)
last_round = tot.select(["match_key", "last_round"])
last_s = rounds.join(last_round, on="match_key").filter(pl.col("round") == pl.col("last_round")).select(
    ["match_key", "s_before"])
fin = (rs.sort(["match_key", "tick"]).group_by("match_key").agg(
    pl.col("t_score").last().alias("t_last"), pl.col("ct_score").last().alias("c_last")))
fin = fin.join(last_s, on="match_key", how="left").with_columns(
    pl.when(pl.col("s_before") % 2 == 0).then(pl.col("t_last")).otherwise(pl.col("c_last")).alias("xSC"),
    pl.when(pl.col("s_before") % 2 == 0).then(pl.col("c_last")).otherwise(pl.col("t_last")).alias("ySC"))
M = tot.join(fin.select(["match_key", "xSC", "ySC"]), on="match_key", how="left").join(H, on="match_key", how="left")
M = M.with_columns(day_expr().alias("day"),
                   (pl.col("xRE") == pl.col("hT")).alias("RE_eq_hdr"),
                   (pl.col("xSC") == pl.col("hT")).alias("SC_eq_hdr"),
                   (pl.col("xRE") + pl.col("yRE") == pl.col("hT") + pl.col("hC")).alias("total_eq"))
# disputed matches: any round-13 (or any) disagreement between score side and round_end in valid rounds
disp = valid.filter(pl.col("dis_re")).select("match_key").unique().with_columns(pl.lit(True).alias("disputed"))
M = M.join(disp, on="match_key", how="left").with_columns(pl.col("disputed").fill_null(False))
M = M.with_columns(pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).when(pl.col("day").is_between(12, 25))
                   .then(pl.lit("d12-25")).otherwise(pl.lit("other")).alias("grp"))
print(M.group_by(["grp", "disputed"]).agg(
    pl.len().alias("matches"),
    pl.col("RE_eq_hdr").mean().alias("frac_roundend_eq_header_T"),
    pl.col("SC_eq_hdr").mean().alias("frac_statefinal_eq_header_T"),
    pl.col("total_eq").mean().alias("frac_total_rounds_eq_header_sum"),
).sort(["grp", "disputed"]))
print("\nnote: 'disputed' = match has >=1 valid round where score side != round_end (any round).")
print("matches with header final present:", int(M["hT"].is_not_null().sum()), "of", M.height)

# ---------- C5 last death ----------
print("\n== C5 elimination-ended rounds: winner vs team opposite the last victim")
pdth = scan(base, "player_death", ["match_key", "round", "tick", "player_team_code"]).collect()
last_d = (pdth.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("tick").last().alias("last_tick"), pl.col("player_team_code").last().alias("victim_team")))
E = R.join(last_d, on=["match_key", "round"], how="left").filter(
    pl.col("win_reason_code").is_in([8, 9]) & pl.col("victim_team").is_not_null() & (~pl.col("sw")))
E = E.with_columns(
    (pl.col("victim_team") == pl.col("winner_team_code")).alias("victim_is_winner"),
    (pl.col("victim_team") == pl.col("score_side")).alias("victim_is_score_side"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rb"))
print(E.group_by(["grp", "rb", "dis_re"]).agg(
    pl.len().alias("n"),
    (~pl.col("victim_is_winner")).mean().alias("frac_winner_is_not_victim_team"),
    pl.col("victim_is_score_side").mean().alias("frac_victim_eq_score_side"),
).sort(["grp", "rb", "dis_re"]))
print("\nvictim_team code distribution (elimination rounds, d12-25 round 13):")
print(E.filter((pl.col("grp") == "d12-25") & (pl.col("round") == 13)).group_by("victim_team").len().sort("victim_team"))
