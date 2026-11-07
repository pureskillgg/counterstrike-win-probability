"""verify_D4_6 main: independent recomputation of claim D4-6 (2024 days 12-25 flagged rounds).

Claim under test: on the flagged rounds (score-step side != round_end winner), the winner implied by the last death in the
live window (5 - victim side) agrees with the score-step side and not with round_end. The 3 s window of D4_elim_check is
said to be too narrow for 2024. Verdict asked for: CONFIRMED, REFUTED or PARTIAL.

Definitions (written here from scratch; the D4 scripts are not imported):
  T=2, CT=3. 5 - side gives the other side.
  live = round_state round_freeze_end tick (first row per match, round).
  end  = round_end tick (first row per match, round); winner = round_end.winner_team_code.
  sc   = score step: freeze_end(r+1) - freeze_end(r). CT if only ct_score rose by 1, T if only t_score rose by 1,
         else 0 (unusable). scr = same on t_score_raw / ct_score_raw.
  bomb truth (game rule, no team field): bomb_defused -> CT wins, bomb_exploded -> T wins (label round).
  death rule: player_death rows with team code T/CT and tick in [live, end]; keep the last tick; if one side is the
         victim side at that tick, pred = 5 - victim. gap = end - last death tick.
  ASSUMPTION (tick rate): 64 ticks per second. Checked against header.tick_rate and timer-round durations (section 8).
  ASSUMPTION (scope): full round = round_end.player_count == 20 (2024) or 10 (2026); days 12-25 = match_key day >= 12.
  ASSUMPTION (noise): agreement rates are read against a noise floor of roughly 3-9% (the unflagged rounds).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D4_6_main.py
Output is captured in logs\\swarm2\\verify_D4_6.log.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
BASE26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T, CT = 2, 3
FOCUS = [9, 13, 25, 28]
CONTROLS = [1, 2, 10, 14, 19]
ROUNDS = FOCUS + CONTROLS
TICKS_PER_S = 64  # ASSUMPTION, see section 8


def read_dir(d, cols):
    files = sorted(glob.glob(str(d / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def read(base, ch, cols):
    return read_dir(sorted(base.glob(f"basic_{ch}.*"))[0], cols)


def hdr(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def build(base, era, full_pc):
    hdr(f"BUILD {era}")
    rs = read(base, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                                    "t_score_raw", "ct_score_raw"])
    fe_rows = rs.filter(pl.col("event_type") == "round_freeze_end")
    fe = fe_rows.unique(["match_key", "round"], keep="first", maintain_order=True)
    conflicts = (fe_rows.group_by(["match_key", "round"])
                 .agg(pl.col("tick").n_unique().alias("nt"), pl.col("t_score").n_unique().alias("nts"),
                      pl.col("ct_score").n_unique().alias("ncs"))
                 .filter((pl.col("nt") > 1) | (pl.col("nts") > 1) | (pl.col("ncs") > 1)).height)
    print(f"round_freeze_end rows {fe_rows.height}; unique (match,round) {fe.height}; "
          f"duplicate rows dropped {fe_rows.height - fe.height}; (match,round) with conflicting rows {conflicts}")

    nxt = fe.select(["match_key", (pl.col("round") - 1).alias("round"),
                     pl.col("t_score").alias("t_next"), pl.col("ct_score").alias("ct_next"),
                     pl.col("t_score_raw").alias("tr_next"), pl.col("ct_score_raw").alias("ctr_next")])
    S = fe.select(["match_key", "round", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).join(
        nxt, on=["match_key", "round"], how="inner")
    S = S.with_columns(
        (pl.col("t_next") - pl.col("t_score")).alias("dT"),
        (pl.col("ct_next") - pl.col("ct_score")).alias("dC"),
        (pl.col("tr_next") - pl.col("t_score_raw")).alias("dTr"),
        (pl.col("ctr_next") - pl.col("ct_score_raw")).alias("dCr"),
    ).with_columns(
        pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
        .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("sc"),
        pl.when((pl.col("dCr") == 1) & (pl.col("dTr") == 0)).then(pl.lit(CT))
        .when((pl.col("dTr") == 1) & (pl.col("dCr") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("scr"),
    )

    re_rows = read(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                       "player_count"])
    re_ = re_rows.unique(["match_key", "round"], keep="first", maintain_order=True)
    print(f"round_end rows {re_rows.height}; unique (match,round) {re_.height}; duplicates dropped "
          f"{re_rows.height - re_.height}")

    bs = read(base, "bomb_state", ["match_key", "round", "tick", "event_type"])
    outc = bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
    bo = outc.group_by(["match_key", "round"]).agg(
        pl.col("event_type").n_unique().alias("n_types"), pl.len().alias("n_out"),
        pl.col("event_type").first().alias("ev"), pl.col("tick").min().alias("bomb_tick"))
    print(f"outcome rounds {bo.height}; with two outcome types {bo.filter(pl.col('n_types') > 1).height}; "
          f"with two outcome rows {bo.filter(pl.col('n_out') > 1).height}")
    bo = bo.with_columns(pl.when(pl.col("ev") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(T))
                         .alias("bomb_truth"))
    planted = (bs.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique()
               .with_columns(pl.lit(True).alias("planted")))

    de = read(base, "player_death", ["match_key", "round", "tick", "player_team_code"]).filter(
        pl.col("player_team_code").is_in([T, CT]))

    hd = read_dir(base / "header", ["key", "tick_rate", "t_starters_score_final", "ct_starters_score_final"])
    hd = hd.unique("key", keep="first").with_columns(
        (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr_total"))
    labs = re_.group_by("match_key").agg(pl.col("round").max().alias("max_lab"),
                                        pl.col("round").n_unique().alias("n_lab"),
                                        (pl.col("round") == 1).any().alias("has_r1"))
    labs = labs.join(hd.select(["key", "hdr_total", "tick_rate"]), left_on="match_key", right_on="key", how="left")
    labs = labs.with_columns((pl.col("max_lab") == pl.col("hdr_total")).alias("hdr_complete"))
    print(f"matches in round_end {labs.height}; with header row {labs.filter(pl.col('hdr_total').is_not_null()).height}; "
          f"last round label == header total {labs.filter(pl.col('hdr_complete')).height}; "
          f"has round-1 end (regime A marker) {labs.filter(pl.col('has_r1')).height}")

    R0 = (re_.select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
          .rename({"tick": "end"})
          .join(fe.select(["match_key", "round", "tick"]).rename({"tick": "live"}), on=["match_key", "round"],
                how="inner"))
    Dw = de.join(R0.select(["match_key", "round", "live", "end"]), on=["match_key", "round"], how="inner").filter(
        (pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
    nd = Dw.group_by(["match_key", "round"]).agg(pl.len().alias("n_deaths"))
    lt = Dw.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("last_tick"))
    LD = (Dw.join(lt, on=["match_key", "round"])
          .filter(pl.col("tick") == pl.col("last_tick"))
          .group_by(["match_key", "round"]).agg(
              pl.col("player_team_code").n_unique().alias("n_sides"),
              pl.col("player_team_code").max().alias("victim"),
              pl.col("last_tick").first().alias("last_tick")))

    R = (R0.join(S.select(["match_key", "round", "sc", "scr", "dT", "dC"]), on=["match_key", "round"], how="left")
         .join(bo.select(["match_key", "round", "bomb_truth", "bomb_tick", "n_out"]), on=["match_key", "round"],
               how="left")
         .join(planted, on=["match_key", "round"], how="left")
         .join(LD, on=["match_key", "round"], how="left")
         .join(nd, on=["match_key", "round"], how="left")
         .join(labs.select(["match_key", "has_r1", "hdr_complete", "tick_rate"]), on="match_key", how="left")
         .with_columns(
             pl.col("planted").fill_null(False),
             pl.col("n_deaths").fill_null(0),
             pl.col("sc").fill_null(0),
             pl.col("scr").fill_null(0),
             (pl.col("end") - pl.col("last_tick")).alias("gap"),
             pl.when(pl.col("n_sides") == 1).then(pl.lit(5) - pl.col("victim"))
             .otherwise(pl.lit(None).cast(pl.Int64)).alias("pred"),
             pl.col("match_key").str.extract(r"^csds/(\d{4})/(\d\d)/(\d\d)/", 3).cast(pl.Int32).alias("day"),
             pl.col("match_key").str.extract(r"^csds/(\d{4})/", 1).alias("year"),
         )
         .with_columns(
             (pl.col("sc") != 0).alias("usable"),
             ((pl.col("sc") != 0) & (pl.col("sc") != pl.col("winner_team_code"))).alias("flag"),
             (pl.col("pred") == pl.col("sc")).alias("pred_sc"),
             (pl.col("pred") == pl.col("winner_team_code")).alias("pred_end"),
             (pl.col("winner_team_code") == pl.col("bomb_truth")).alias("end_bomb"),
             (pl.col("sc") == pl.col("bomb_truth")).alias("sc_bomb"),
             (pl.col("pred") == pl.col("bomb_truth")).alias("pred_bomb"),
         ))
    print(f"round rows {R.height}; day parse null {R.filter(pl.col('day').is_null()).height}; "
          f"year values {sorted(R['year'].unique().to_list())}")
    return R, S, bo, labs


def rbin():
    return pl.when(pl.col("round") == 9).then(pl.lit("r09")).when(pl.col("round") == 13).then(pl.lit("r13")) \
        .when(pl.col("round") == 25).then(pl.lit("r25")).when(pl.col("round") == 28).then(pl.lit("r28")) \
        .otherwise(pl.lit("other"))


def metrics(df, keys):
    return (df.group_by(keys).agg(
        pl.len().alias("n"),
        pl.col("pred_sc").mean().round(4).alias("pred=score"),
        pl.col("pred_end").mean().round(4).alias("pred=round_end"),
        pl.col("gap").median().alias("med_gap"),
        pl.col("gap").quantile(0.9, interpolation="nearest").alias("p90_gap"),
    ).sort(keys))


# ----------------------------------------------------------------------------------------------------------------------
R24, S24, bo24, labs24 = build(BASE24, "2024", 20)
R24 = R24.filter(pl.col("year") == "2024")
FULL24 = R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12) & pl.col("usable")
                    & pl.col("pred").is_not_null())

hdr("SECTION 0. 2024 rounds in scope and basic counts (days 12-25, full rounds, usable score step)")
print("full rounds days 12-25 with usable score step:", R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12)
                                                                  & pl.col("usable")).height)
print("  of which with a last death in window (pred known):", FULL24.height)
print("  of which flagged (score side != round_end):", FULL24.filter(pl.col("flag")).height)
print("flagged rounds by round number (all rounds 1-30, days 12-25, full, usable, pred known):")
print(FULL24.filter(pl.col("flag")).group_by("round").len().sort("round"))

# Section 1. Reproduce the claimed table on the claimed scope (all bomb statuses, any gap, claimed round list).
hdr("SECTION 1. Claimed table recomputed on the claimed scope (all bomb statuses, any gap, claimed rounds)")
A1 = FULL24.filter(pl.col("round").is_in(ROUNDS)).with_columns(rbin().alias("rb"))
print(metrics(A1, ["round", "flag"]))

# Section 2. Remove bomb-decided rounds (bomb outcome row or bomb planted): the death rule is an elimination rule only.
hdr("SECTION 2. Elimination-candidate scope: no bomb planted and no bomb outcome in the round")
A2 = FULL24.filter(pl.col("round").is_in(ROUNDS) & pl.col("bomb_truth").is_null() & (~pl.col("planted")))
print(metrics(A2, ["round", "flag"]))
print("bomb-decided rounds dropped from the claimed scope (focus + controls):",
      FULL24.filter(pl.col("round").is_in(ROUNDS) & (pl.col("bomb_truth").is_not_null() | pl.col("planted"))).height)
print("flagged focus rounds in scope with bomb outcome or plant:",
      FULL24.filter(pl.col("round").is_in(FOCUS) & pl.col("flag") & (pl.col("bomb_truth").is_not_null() | pl.col("planted"))).height)

# Section 3. Gap distribution: does the 3 s window select the flagged rounds at all?
hdr("SECTION 3. Gap from last death to round_end, elimination-candidate scope (ticks; 64 ticks = 1 s)")
A3 = A2.with_columns(
    pl.when(pl.col("gap") <= 192).then(pl.lit("a: <=3s (192)"))
    .when(pl.col("gap") <= 448).then(pl.lit("b: 3-7s (448)"))
    .when(pl.col("gap") <= 1000).then(pl.lit("c: 7-15.6s (1000)"))
    .when(pl.col("gap") <= 2000).then(pl.lit("d: 15.6-31s (2000)"))
    .otherwise(pl.lit("e: >31s")).alias("gapbin"))
print("gap quantiles (ticks) by round and flag:")
print(A2.group_by(["round", "flag"]).agg(
    pl.len().alias("n"),
    pl.col("gap").quantile(0.1, interpolation="nearest").alias("p10"),
    pl.col("gap").quantile(0.5, interpolation="nearest").alias("p50"),
    pl.col("gap").quantile(0.75, interpolation="nearest").alias("p75"),
    pl.col("gap").quantile(0.9, interpolation="nearest").alias("p90"),
    (pl.col("gap") <= 192).mean().round(3).alias("share<=3s"),
    (pl.col("gap") <= 448).mean().round(3).alias("share<=7s"),
    (pl.col("gap") <= 1000).mean().round(3).alias("share<=15.6s"),
).sort(["round", "flag"]))
print("agreement by gap bin (elimination-candidate scope):")
print(A3.group_by(["round", "flag", "gapbin"]).agg(
    pl.len().alias("n"),
    pl.col("pred_sc").mean().round(4).alias("pred=score"),
    pl.col("pred_end").mean().round(4).alias("pred=round_end")).sort(["round", "flag", "gapbin"]))

# Section 4. Bomb rule as ground truth (no team field involved). Compare round_end, score step and death prediction.
hdr("SECTION 4. Bomb-decided rounds (game rule: exploded = T, defused = CT), days 12-25, full, usable score step")
B = R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12) & pl.col("usable")
               & pl.col("bomb_truth").is_not_null()).with_columns(rbin().alias("rb"))
print("bomb-decided usable rounds:", B.height)
print(B.group_by(["rb", "flag"]).agg(
    pl.len().alias("n"),
    pl.col("end_bomb").mean().round(4).alias("round_end=bomb"),
    pl.col("sc_bomb").mean().round(4).alias("score=bomb"),
    pl.col("pred").is_not_null().sum().alias("n_pred"),
    pl.col("pred_bomb").filter(pl.col("pred").is_not_null()).mean().round(4).alias("death_pred=bomb"),
).sort(["rb", "flag"]))
print("all bomb-decided rounds (no score filter), round_end vs bomb rule by round (2024 days 12-25):")
Ball = R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12) & pl.col("bomb_truth").is_not_null())
print(Ball.group_by("round").agg(pl.len().alias("n"),
                                 pl.col("end_bomb").mean().round(4).alias("round_end=bomb"),
                                 (pl.col("sc") != 0).sum().alias("n_usable"),
                                 pl.col("sc_bomb").filter(pl.col("sc") != 0).mean().round(4).alias("score=bomb"))
          .filter(pl.col("round").is_in([9, 13, 25, 28, 12, 14])).sort("round"))

# Section 5. Index-shift test: is round_end(r) really the winner of a neighbouring round?
hdr("SECTION 5. Index-shift test on flagged elimination-candidate rounds: round_end(r) vs neighbours")
N = R24.select(["match_key", "round", "sc", "bomb_truth", "winner_team_code"])
prev = N.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("sc").alias("sc_prev"),
                 pl.col("bomb_truth").alias("bt_prev")])
nxtn = N.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("sc").alias("sc_next"),
                 pl.col("bomb_truth").alias("bt_next")])
A5 = (A2.join(prev, on=["match_key", "round"], how="left").join(nxtn, on=["match_key", "round"], how="left"))
A5 = A5.with_columns(
    (pl.col("winner_team_code") == pl.col("sc_prev")).alias("end=sc_prev"),
    (pl.col("winner_team_code") == pl.col("sc_next")).alias("end=sc_next"),
    (pl.col("winner_team_code") == pl.col("bt_prev")).alias("end=bomb_prev"),
    (pl.col("winner_team_code") == pl.col("bt_next")).alias("end=bomb_next"),
)
A5c = (R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12) & pl.col("round").is_in(FOCUS))
       .join(prev, on=["match_key", "round"], how="left").join(nxtn, on=["match_key", "round"], how="left")
       .with_columns(pl.col("bomb_truth").is_not_null().alias("bombdec")))
g5 = A5c.group_by(["round", "flag", "bombdec"]).agg(
    pl.len().alias("n"),
    ((pl.col("sc_prev") != 0)).sum().alias("av_sc_prev"),
    (((pl.col("sc_prev") != 0)) & (pl.col("winner_team_code") == pl.col("sc_prev"))).sum().alias("eq_sc_prev"),
    ((pl.col("sc_next") != 0)).sum().alias("av_sc_next"),
    (((pl.col("sc_next") != 0)) & (pl.col("winner_team_code") == pl.col("sc_next"))).sum().alias("eq_sc_next"),
    pl.col("bt_prev").is_not_null().sum().alias("av_bt_prev"),
    ((pl.col("bt_prev").is_not_null()) & (pl.col("winner_team_code") == pl.col("bt_prev"))).sum().alias("eq_bt_prev"),
    pl.col("bt_next").is_not_null().sum().alias("av_bt_next"),
    ((pl.col("bt_next").is_not_null()) & (pl.col("winner_team_code") == pl.col("bt_next"))).sum().alias("eq_bt_next"),
).sort(["round", "flag", "bombdec"])
print("counts with available neighbour and equality counts (rates use only available neighbours):")
print(g5)
print("rates:")
print(g5.with_columns(
    (pl.col("eq_sc_prev") / pl.col("av_sc_prev")).round(3).alias("end=score(r-1)|avail"),
    (pl.col("eq_sc_next") / pl.col("av_sc_next")).round(3).alias("end=score(r+1)|avail"),
    (pl.col("eq_bt_prev") / pl.col("av_bt_prev")).round(3).alias("end=bomb(r-1)|avail"),
    (pl.col("eq_bt_next") / pl.col("av_bt_next")).round(3).alias("end=bomb(r+1)|avail"),
).select(["round", "flag", "bombdec", "n", "end=score(r-1)|avail", "end=score(r+1)|avail",
          "end=bomb(r-1)|avail", "end=bomb(r+1)|avail", "av_bt_prev", "av_bt_next"]))

# Section 6. Internal consistency of the flagged round_end rows: reason code vs winner.
hdr("SECTION 6. Win reason vs round_end winner on flagged vs unflagged elimination-candidate focus rounds")
print(A2.filter(pl.col("round").is_in([13]) & pl.col("flag")).group_by(["win_reason_code", "winner_team_code", "sc"])
      .len().sort(["win_reason_code", "winner_team_code", "sc"]))
print("unflagged round 13, same scope:")
print(A2.filter(pl.col("round").is_in([13]) & (~pl.col("flag"))).group_by(["win_reason_code", "winner_team_code"])
      .len().sort(["win_reason_code", "winner_team_code"]))

# Section 7. Regime marker: round_end label 1 present in the match (A6/D5 regime A marker).
hdr("SECTION 7. Flagged rate by regime marker (round-1 round_end row present) and header completeness, round 13 and others")
C7 = R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12) & pl.col("usable")
                & pl.col("round").is_in(ROUNDS)).with_columns(rbin().alias("rb"))
print(C7.group_by(["rb", "has_r1", "hdr_complete"]).agg(
    pl.len().alias("n"), pl.col("flag").mean().round(4).alias("flag_rate"),
    pl.col("gap").median().alias("med_gap_all")).sort(["rb", "has_r1", "hdr_complete"]))

# Section 8. Tick-rate check: header value and timer-round durations.
hdr("SECTION 8. Tick-rate check (2024 header.tick_rate; timer rounds without death or bomb)")
print("2024 header tick_rate values:", labs24.group_by("tick_rate").len().to_dict(as_series=False))
T8 = R24.filter((pl.col("player_count") == 20) & (pl.col("n_deaths") == 0) & (~pl.col("planted"))
                & pl.col("bomb_truth").is_null())
dur = (T8.with_columns((pl.col("end") - pl.col("live")).alias("dur")))
print("timer-like rounds (no death, no bomb plant), duration end-live in ticks: n =", dur.height)
print(dur.select(pl.col("dur").quantile(0.1, interpolation="nearest").alias("p10"),
                 pl.col("dur").median().alias("median"),
                 pl.col("dur").quantile(0.9, interpolation="nearest").alias("p90")))
print("most common durations (ticks, top 6):")
print(dur.group_by("dur").len().sort("len", descending=True).head(6))
print("expected for 115 s round time: 7360 ticks at 64/s, 14720 at 128/s (not a verified round-time constant)")

# Section 9. Raw score fields vs adjusted score fields.
hdr("SECTION 9. Raw score (t_score_raw / ct_score_raw) vs score fields used for the score step, 2024 days 12-25")
Z = R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12) & pl.col("usable")
               & (pl.col("scr") != 0) & pl.col("round").is_in(ROUNDS))
print("steps with both usable: n =", Z.height, " share sc == scr:", round((Z["sc"] == Z["scr"]).mean(), 4))
print(Z.group_by(["round", "flag"]).agg(
    pl.len().alias("n"),
    (pl.col("sc") == pl.col("scr")).mean().round(4).alias("sc=scr"),
    (pl.col("scr") == pl.col("winner_team_code")).mean().round(4).alias("raw_step=round_end"),
).sort(["round", "flag"]))

# Section 10. Control: 2026 tome (full rounds have player_count 10).
hdr("SECTION 10. Control: 2026 tome, full rounds (player_count 10), same definitions")
R26, S26, bo26, labs26 = build(BASE26, "2026", 10)
R26 = R26.filter(pl.col("year") == "2026")
F26 = R26.filter((pl.col("player_count") == 10) & pl.col("usable"))
print("usable full rounds 2026:", F26.height, " flagged:", F26.filter(pl.col("flag")).height,
      " flag rate:", round(F26.filter(pl.col("flag")).height / max(F26.height, 1), 5))
print("flagged 2026 by round (focus rounds only):")
print(F26.filter(pl.col("flag") & pl.col("round").is_in(ROUNDS)).group_by("round").len().sort("round"))
print("2026 bomb-decided usable rounds: round_end=bomb rate",
      round(F26.filter(pl.col("bomb_truth").is_not_null())["end_bomb"].mean(), 4) if F26.filter(
          pl.col("bomb_truth").is_not_null()).height else None)

# Section 11. Residual cases the claim does not address.
hdr("SECTION 11. Residual cases: flagged rounds where the death rule agrees with round_end, and flagged non-focus bomb rounds")
res = A2.filter(pl.col("round").is_in(FOCUS) & pl.col("flag") & (pl.col("gap") <= 192))
print("flagged focus elimination-candidate rounds with gap <= 192 ticks (3 s):", res.height,
      " pred=round_end:", res.filter(pl.col("pred_end")).height, " pred=score:", res.filter(pl.col("pred_sc")).height)
print(res.select(["match_key", "round", "gap", "sc", "winner_team_code", "pred", "has_r1"]).head(20))
nb = R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12) & pl.col("flag")
                & pl.col("bomb_truth").is_not_null() & (~pl.col("round").is_in(FOCUS)))
print("flagged bomb-decided rounds outside the focus list:", nb.height,
      " round_end = bomb rule:", nb.filter(pl.col("end_bomb")).height,
      " score = bomb rule:", nb.filter(pl.col("sc_bomb")).height)
print(nb.group_by("round").len().sort("round"))
print("2026 flagged rounds (all rounds) by round, top 10:")
print(F26.filter(pl.col("flag")).group_by("round").len().sort("len", descending=True).head(10))

# Section 12. Gap signature. The 429-tick lag is a regime-A property; check it on non-focus rounds too.
hdr("SECTION 12. Gap signature: last death to round_end (ticks), elimination-candidate scope")
E = A2.filter(~pl.col("round").is_in(FOCUS))
print("non-focus rounds by regime marker (round-1 end present):")
print(E.group_by("has_r1").agg(pl.len().alias("n"), (pl.col("gap") == 429).mean().round(3).alias("share_gap_eq_429"),
                               (pl.col("gap") == 0).mean().round(3).alias("share_gap_eq_0"),
                               pl.col("gap").median().alias("median_gap")))
G = A2.filter(pl.col("round").is_in([13]) & pl.col("flag"))
print("flagged round 13, elimination-candidate, gap value counts (top 6):")
print(G.group_by("gap").len().sort("len", descending=True).head(6))

# Section 13. Every flagged round, not only the claimed list. Round 27 is flagged in 2024 days 12-25 and in 2026.
hdr("SECTION 13. Flagged rounds by round number, all rounds (usable score step, full rounds)")
FL24 = R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12) & pl.col("usable") & pl.col("flag"))
print("2024 days 12-25 flagged rounds by round (n_flag, n with death pred, bomb-decided, plant-only, pred=score, pred=round_end):")
print(FL24.group_by("round").agg(
    pl.len().alias("n_flag"),
    pl.col("pred").is_not_null().sum().alias("n_pred"),
    pl.col("bomb_truth").is_not_null().sum().alias("n_bomb"),
    (pl.col("planted") & pl.col("bomb_truth").is_null()).sum().alias("n_plant_only"),
    pl.col("pred_sc").filter(pl.col("pred").is_not_null()).mean().round(3).alias("pred=score"),
    pl.col("pred_end").filter(pl.col("pred").is_not_null()).mean().round(3).alias("pred=round_end"),
    pl.col("end_bomb").filter(pl.col("bomb_truth").is_not_null()).mean().round(3).alias("end=bomb(bomb rows)"),
    pl.col("sc_bomb").filter(pl.col("bomb_truth").is_not_null()).mean().round(3).alias("score=bomb(bomb rows)"),
).sort("round"))
print("Round 27, flagged, elimination-candidate (no bomb outcome, no plant), 2024 days 12-25:")
r27 = FL24.filter(pl.col("round") == 27)
print(r27.group_by(["bomb_truth", "planted"]).agg(pl.len().alias("n"),
                                                 pl.col("pred").is_not_null().sum().alias("n_pred")))
r27e = FL24.filter((pl.col("round") == 27) & pl.col("bomb_truth").is_null() & (~pl.col("planted")) & pl.col("pred").is_not_null())
print("  elimination-candidate flagged r27 with death pred:", r27e.height,
      " pred=score:", r27e.filter(pl.col("pred_sc")).height,
      " pred=round_end:", r27e.filter(pl.col("pred_end")).height)
print("  gap quantiles (p10, p50, p90):",
      r27e.select(pl.col("gap").quantile(0.1, interpolation="nearest").alias("p10"),
                  pl.col("gap").median().alias("p50"),
                  pl.col("gap").quantile(0.9, interpolation="nearest").alias("p90")).row(0))
print("  win reason vs winner (flagged r27 elim-candidate):")
print(r27e.group_by(["win_reason_code", "winner_team_code", "sc"]).len().sort(["win_reason_code", "winner_team_code"]))
print("  regime marker of flagged r27 rows:", r27.group_by("has_r1").len().to_dict(as_series=False))
print("  unflagged r27, same scope, round_end vs death pred:")
U27 = R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12) & pl.col("usable") & (~pl.col("flag"))
                 & (pl.col("round") == 27) & pl.col("pred").is_not_null() & pl.col("bomb_truth").is_null()
                 & (~pl.col("planted")))
print("   n:", U27.height, " pred=score:", round(U27.filter(pl.col("pred_sc")).height / max(U27.height, 1), 3),
      " pred=round_end:", round(U27.filter(pl.col("pred_end")).height / max(U27.height, 1), 3))

hdr("SECTION 13b. 2026 flagged rounds by round (same definitions), with death check")
FL26 = R26.filter((pl.col("player_count") == 10) & pl.col("usable") & pl.col("flag"))
print(FL26.group_by("round").agg(
    pl.len().alias("n_flag"),
    pl.col("pred").is_not_null().sum().alias("n_pred"),
    pl.col("bomb_truth").is_not_null().sum().alias("n_bomb"),
    pl.col("pred_sc").filter(pl.col("pred").is_not_null()).mean().round(3).alias("pred=score"),
    pl.col("pred_end").filter(pl.col("pred").is_not_null()).mean().round(3).alias("pred=round_end"),
).sort("round"))
print("2026 round 27 flagged, bomb rows and death check:")
r27_26 = FL26.filter(pl.col("round") == 27)
print(r27_26.group_by(["bomb_truth"]).agg(pl.len().alias("n"), pl.col("end_bomb").mean().alias("end=bomb"),
                                          pl.col("sc_bomb").mean().alias("score=bomb")))
print("  gap quantiles (p10, p50, p90) for 2026 flagged r27 with pred:",
      r27_26.filter(pl.col("pred").is_not_null()).select(
          pl.col("gap").quantile(0.1, interpolation="nearest").alias("p10"),
          pl.col("gap").median().alias("p50"),
          pl.col("gap").quantile(0.9, interpolation="nearest").alias("p90")).row(0))
print("2026 round 27 flagged, pred=score / pred=round_end among rows with pred:",
      r27_26.filter(pl.col("pred").is_not_null()).select(
          pl.col("pred_sc").mean().round(3).alias("pred=score"),
          pl.col("pred_end").mean().round(3).alias("pred=round_end")).row(0))
U26_27 = R26.filter((pl.col("player_count") == 10) & pl.col("usable") & (~pl.col("flag")) & (pl.col("round") == 27)
                    & pl.col("pred").is_not_null() & pl.col("bomb_truth").is_null() & (~pl.col("planted")))
print("2026 unflagged r27 elimination-candidate baseline: n", U26_27.height,
      " pred=round_end:", round(U26_27.filter(pl.col("pred_end")).height / max(U26_27.height, 1), 3))

# Section 2b. Scope with only bomb outcomes removed (planted-but-undecided rounds kept).
hdr("SECTION 2b. Scope variant: remove only bomb-outcome rounds (keep planted-but-undecided rounds), focus + controls")
A2b = FULL24.filter(pl.col("round").is_in(ROUNDS) & pl.col("bomb_truth").is_null())
print(metrics(A2b, ["round", "flag"]))

# Section 13c. Round 27: how many usable steps exist and how many are flagged; neighbour test on the score step.
hdr("SECTION 13c. Round 27 and the other flagged-prone rounds: usable steps, flags, neighbour test")
for name, RR, pc, cond in [("2024 days 12-25", R24, 20, pl.col("day") >= 12), ("2026 all days", R26, 10, pl.col("day") >= 0)]:
    for rnd in [9, 13, 25, 27, 28]:
        U = RR.filter((pl.col("player_count") == pc) & cond & pl.col("usable") & (pl.col("round") == rnd))
        print(f"{name} round {rnd}: usable {U.height}, flagged {U.filter(pl.col('flag')).height}, "
              f"unflagged {U.filter(~pl.col('flag')).height}")
W24 = R24.select(["match_key", "round", "winner_team_code", "sc", "bomb_truth"])
wprev = W24.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("winner_team_code").alias("w_p"),
                    pl.col("sc").alias("sc_p"), pl.col("bomb_truth").alias("bt_p")])
wnext = W24.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("winner_team_code").alias("w_n"),
                    pl.col("sc").alias("sc_n"), pl.col("bomb_truth").alias("bt_n")])
X27 = (R24.filter((pl.col("player_count") == 20) & (pl.col("day") >= 12) & pl.col("usable")
                  & (pl.col("round") == 27) & pl.col("flag"))
       .join(wprev, on=["match_key", "round"], how="left").join(wnext, on=["match_key", "round"], how="left"))
print("2024 flagged r27 (usable), n =", X27.height)
print("  score step (sc27) == round_end(26):", X27.filter(pl.col("w_p") == pl.col("sc")).height,
      " == round_end(28):", X27.filter(pl.col("w_n") == pl.col("sc")).height,
      " == score step(26):", X27.filter((pl.col("sc_p") == pl.col("sc")) & (pl.col("sc_p") != 0)).height,
      " == score step(28):", X27.filter((pl.col("sc_n") == pl.col("sc")) & (pl.col("sc_n") != 0)).height)
print("  round_end(27) == round_end(26):", X27.filter(pl.col("w_p") == pl.col("winner_team_code")).height,
      " == round_end(28):", X27.filter(pl.col("w_n") == pl.col("winner_team_code")).height)
print("  bomb-decided neighbours: round 26 bomb-rule winner matches sc27:",
      X27.filter(pl.col("bt_p").is_not_null() & (pl.col("bt_p") == pl.col("sc"))).height,
      " of", X27.filter(pl.col("bt_p").is_not_null()).height,
      "; round 28 bomb-rule winner matches sc27:",
      X27.filter(pl.col("bt_n").is_not_null() & (pl.col("bt_n") == pl.col("sc"))).height,
      " of", X27.filter(pl.col("bt_n").is_not_null()).height)
print("  winner(27) equals death-implied winner (pred) among flagged r27 with pred:",
      X27.filter(pl.col("pred").is_not_null()).select(
          (pl.col("pred") == pl.col("winner_team_code")).mean().round(3).alias("pred=round_end"),
          (pl.col("pred") == pl.col("sc")).mean().round(3).alias("pred=score")).row(0))

# Section 13d. Totals for the claim scope: how much of the flagged set goes in the claimed direction?
hdr("SECTION 13d. Flagged rounds with a death prediction, by direction (2024 days 12-25, full)")
FP = FL24.filter(pl.col("pred").is_not_null())
dir24 = FP.group_by(["round"]).agg(pl.len().alias("n"),
                                  pl.col("pred_sc").sum().alias("death_agrees_score"),
                                  pl.col("pred_end").sum().alias("death_agrees_round_end")).sort("round")
print(dir24)
tot = FP.height
print("total flagged with pred:", tot,
      " in claimed direction (round 9/13/25/28):",
      FP.filter(pl.col("round").is_in(FOCUS)).height,
      " share:", round(FP.filter(pl.col("round").is_in(FOCUS)).height / max(tot, 1), 4))
print("total flagged with pred in 2026 (all rounds):", FL26.filter(pl.col("pred").is_not_null()).height,
      " death agrees round_end:", FL26.filter(pl.col("pred_end")).height)

print("\nDONE")
