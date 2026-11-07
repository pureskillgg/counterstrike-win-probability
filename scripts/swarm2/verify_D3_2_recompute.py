"""D3-2 independent verifier: recompute the 2024 round-13 round_end winner mismatch and test alternatives.

Reads only the G: 2024 tome (column-projected, one channel at a time). Writes nothing except stdout,
which the caller redirects to logs\\swarm2\\verify_D3_2.log.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_2_recompute.py
"""
import glob
import re
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(240)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3  # side codes: 2 = Terrorists, 3 = Counter-Terrorists (ASSUMPTION, consistent with claim)
REASON_SIDE = {1: T, 9: T, 13: T, 17: T, 7: CT, 8: CT, 11: CT, 12: CT, 18: CT}


def g_read(ch, cols):
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def side_from_delta(dt_col, dct_col):
    return (pl.when((pl.col(dct_col) == 1) & (pl.col(dt_col) == 0)).then(pl.lit(CT))
            .when((pl.col(dt_col) == 1) & (pl.col(dct_col) == 0)).then(pl.lit(T))
            .otherwise(pl.lit(None, dtype=pl.Int64)))


def pct(a, b):
    return f"{a}/{b} = {100.0 * a / b:.2f}%" if b else f"{a}/{b}"


# ---------------------------------------------------------------- load
RE = g_read("round_end", ["match_key", "round", "tick", "second", "winner_team_code", "win_reason_code",
                          "player_count"])
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
RS = g_read("round_state", ["match_key", "round", "tick", "second", "event_type", "t_score", "ct_score",
                            "t_score_raw", "ct_score_raw"])
print(f"rows: round_end={RE.height} bomb_state={BS.height} round_state={RS.height}")

# day parse, robust to / or \ separators
sample = RE["match_key"].head(2).to_list()
print("match_key samples:", sample)
DAY_RE = r"2024[/\\]02[/\\](\d\d)"
print("match_keys with a 2024-02 date:", RE.filter(pl.col("match_key").str.contains(DAY_RE)).height,
      "of", RE.height)

# ---------------------------------------------------------------- 0. duplicates / join safety
g_re = RE.group_by(["match_key", "round"]).len()
print(f"\n[0] round_end (match,round) groups: {g_re.height}; groups with >1 row (any player_count): "
      f"{g_re.filter(pl.col('len') > 1).height}")
RE20 = RE.filter(pl.col("player_count") == 20)
g20 = RE20.group_by(["match_key", "round"]).len()
print(f"[0] round_end player_count==20 groups: {g20.height}; with >1 row: {g20.filter(pl.col('len') > 1).height}")
g_fe = RS.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).len()
print(f"[0] round_freeze_end groups: {g_fe.height}; with >1 row: {g_fe.filter(pl.col('len') > 1).height}")
g_bo = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").n_unique().alias("nu"), pl.len().alias("n")))
print(f"[0] bomb outcome groups: {g_bo.height}; with >1 distinct outcome type: "
      f"{g_bo.filter(pl.col('nu') > 1).height}; with >1 row: {g_bo.filter(pl.col('n') > 1).height}")

# ---------------------------------------------------------------- 1. tick rate (ASSUMPTION check)
tr = RE.filter(pl.col("second").is_not_null()).select(
    ((pl.cov("tick", "second")) / pl.var("second")).alias("slope_tick_per_s")).item()
print(f"\n[1] tick rate estimate from round_end (tick vs second slope): {tr:.3f} ticks/s")
TICKS_PER_S = tr  # used only for converting tick gaps to seconds

# ---------------------------------------------------------------- 2. bomb timing + outcome per round
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(
            pl.col("event_type").unique().alias("evs"),
            pl.col("tick").min().alias("out_tick")))
plant = (BS.filter(pl.col("event_type") == "bomb_planted").sort("tick")
         .group_by(["match_key", "round"]).agg(
             pl.col("tick").first().alias("plant_tick"),
             pl.col("player_team_code").first().alias("planter")))

# ---------------------------------------------------------------- 3. engine score change r -> r+1 (claim logic)
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(
          pl.col("t_score").first().alias("t_b"), pl.col("ct_score").first().alias("ct_b"),
          pl.col("t_score_raw").first().alias("trb"), pl.col("ct_score_raw").first().alias("ctrb")))
fe_nx = fe.select(["match_key", (pl.col("round") - 1).alias("round"),
                   pl.col("t_b").alias("t_a"), pl.col("ct_b").alias("ct_a"),
                   pl.col("trb").alias("tra"), pl.col("ctrb").alias("ctra")])
sc = fe.join(fe_nx, on=["match_key", "round"], how="left").with_columns(
    (pl.col("t_a") - pl.col("t_b")).alias("dt"), (pl.col("ct_a") - pl.col("ct_b")).alias("dct"),
    (pl.col("tra") - pl.col("trb")).alias("dtr"), (pl.col("ctra") - pl.col("ctrb")).alias("dctr"))
sc = sc.with_columns(
    side_from_delta("dt", "dct").alias("score_side"),
    side_from_delta("dtr", "dctr").alias("score_side_raw"),
    (pl.col("t_a") + pl.col("ct_a")).alias("sum_next"),
    (pl.col("t_b") + pl.col("ct_b")).alias("sum_cur"))

# ---------------------------------------------------------------- 4. base table (claim logic, own code)
base = (RE20.join(outc, on=["match_key", "round"], how="inner")
        .join(plant, on=["match_key", "round"], how="left")
        .join(sc.select(["match_key", "round", "score_side", "score_side_raw", "dt", "dct", "dtr", "dctr"]),
              on=["match_key", "round"], how="left")
        .with_columns(
            pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T))
            .alias("bomb_side"),
            pl.col("win_reason_code").replace_strict(REASON_SIDE, default=None, return_dtype=pl.Int64)
            .alias("reason_side"),
            pl.col("match_key").str.extract(DAY_RE, 1).cast(pl.Int64).alias("day"),
            pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
            .when(pl.col("round") == 28).then(pl.lit("r28")).when(pl.col("round") == 9).then(pl.lit("r09"))
            .otherwise(pl.lit("other")).alias("rbin"),
            (pl.col("out_tick") - pl.col("plant_tick")).alias("plant_to_out_ticks"))
        .with_columns(
            (pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
            (pl.col("tick") - pl.col("out_tick")).alias("re_minus_out_ticks"),
            (pl.col("out_tick") - pl.col("plant_tick")).alias("plant_to_out_ticks_b")))
print(f"\n[4] base rounds (player_count==20, bomb outcome): {base.height}; mismatches: {int(base['mm'].sum())}")
print("    claim says 15,436 and 499")
print("    planter side on mismatch rounds:", base.filter(pl.col("mm"))["planter"].value_counts().rows())

# ---------------------------------------------------------------- 5. claim numbers recomputed
r13 = base.filter((pl.col("round") == 13) & pl.col("day").is_between(12, 25))
print(f"\n[5] round 13, days 12-25, bomb rows: {r13.height}; mismatches: {int(r13['mm'].sum())}")
print("    claim says 605 and 434")
r13_all_days = base.filter(pl.col("round") == 13)
print("    round 13 by day (n, mm):")
print(r13_all_days.group_by("day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("day").rows())
print("\n    mismatches by round number, all days (round: n_bomb_rows, mm):")
rr = base.group_by("round").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("round")
print([(r, n, m) for r, n, m in rr.rows() if m > 0])
print("    mismatch day range:", base.filter(pl.col("mm"))["day"].min(), "to", base.filter(pl.col("mm"))["day"].max())
print("    mismatch rounds outside 12-25 days:", base.filter(pl.col("mm") & (pl.col("day") < 12)).height)

# ---------------------------------------------------------------- 6. agreement of independent sources
def cmp(df, a, b):
    d = df.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
    return d.height, int((d[a] == d[b]).sum())


print("\n[6] source agreement, round 13 days 12-25 (n with both defined, agree):")
for flag in [True, False]:
    s = r13.filter(pl.col("mm") == flag)
    n1, a1 = cmp(s, "reason_side", "winner_team_code")
    n2, a2 = cmp(s, "reason_side", "bomb_side")
    n3, a3 = cmp(s, "score_side", "winner_team_code")
    n4, a4 = cmp(s, "score_side", "bomb_side")
    n5, a5 = cmp(s, "score_side_raw", "bomb_side")
    n6, a6 = cmp(s, "score_side_raw", "winner_team_code")
    print(f"    mismatch={flag} n={s.height}: reason~winner {a1}/{n1}; reason~bomb {a2}/{n2}; "
          f"score~winner {a3}/{n3}; score~bomb {a4}/{n4}; score_raw~bomb {a5}/{n5}; score_raw~winner {a6}/{n6}")
print("    reason_side vs winner is internal to round_end (same event), not independent evidence.")

# ---------------------------------------------------------------- 7. broad check: round_end winner vs engine score, ALL rounds
allrows = (RE20.join(sc.select(["match_key", "round", "score_side", "score_side_raw"]),
                     on=["match_key", "round"], how="left")
           .with_columns(pl.col("match_key").str.extract(DAY_RE, 1).cast(pl.Int64).alias("day")))
allrows = allrows.with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("sc_disagree"))
print("\n[7] round_end winner vs engine score (score_side), ALL player_count==20 rounds with a score change:")
d = allrows.filter(pl.col("score_side").is_not_null())
print(f"    total defined: {d.height}; disagree: {int(d['sc_disagree'].sum())}")
bydays = d.with_columns(
    pl.when(pl.col("day") >= 12).then(pl.lit("day12-25")).otherwise(pl.lit("day01-07")).alias("period"))
print("    disagreements by period and round 13 vs others:")
print(bydays.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("not13")).alias("r13flag"))
    .group_by(["period", "r13flag"]).agg(pl.len().alias("n"), pl.col("sc_disagree").sum().alias("dis"))
    .sort(["period", "r13flag"]).rows())
no_bomb_r13 = (d.filter((pl.col("round") == 13) & (pl.col("day") >= 12))
               .join(outc, on=["match_key", "round"], how="anti"))
print(f"    round 13, days 12-25, rounds WITHOUT a bomb outcome: n={no_bomb_r13.height}, "
      f"round_end disagrees with engine score: {int(no_bomb_r13['sc_disagree'].sum())}")
print("    disagreements by round, days 12-25 (round: n, dis), only rounds with any disagreement:")
dd = d.filter(pl.col("day") >= 12).group_by("round").agg(pl.len().alias("n"),
                                                          pl.col("sc_disagree").sum().alias("dis")).sort("round")
print([(r, n, x) for r, n, x in dd.rows() if x > 0])

# ---------------------------------------------------------------- 8. score_raw vs score (orientation / label check)
print("\n[8] t_score vs t_score_raw consistency at freeze_end (all rounds):")
fe_raw = RS.filter(pl.col("event_type") == "round_freeze_end").select(
    [(pl.col("t_score") != pl.col("t_score_raw")).alias("t_diff"),
     (pl.col("ct_score") != pl.col("ct_score_raw")).alias("ct_diff")])
print(f"    freeze_end rows where t_score != t_score_raw: {int(fe_raw['t_diff'].sum())}; "
      f"ct_score != ct_score_raw: {int(fe_raw['ct_diff'].sum())}; of {fe_raw.height}")
sumchk = (sc.with_columns((pl.col("sum_cur") == (pl.col("round") - 1)).alias("sum_ok"))
          .with_columns(pl.col("match_key").str.extract(DAY_RE, 1).cast(pl.Int64).alias("day")))
print("    freeze_end score sum == round-1 (rounds completed), by round for days 12-25 (round: ok/n) for 9-14:")
sk = sumchk.filter((pl.col("day") >= 12) & pl.col("round").is_between(9, 14)).group_by("round").agg(
    pl.col("sum_ok").sum().alias("ok"), pl.len().alias("n")).sort("round")
print(sk.rows())
print("    overall sum==round-1 fraction, all days:", pct(int(sumchk["sum_ok"].sum()), sumchk.height))

# ---------------------------------------------------------------- 9. neighbour-round test (row shift)
truth = sc.select(["match_key", "round", "score_side"]).join(
    outc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT))
                      .otherwise(pl.lit(T)).alias("bomb_side")).select(["match_key", "round", "bomb_side"]),
    on=["match_key", "round"], how="left")
truth = truth.with_columns(
    pl.coalesce(["bomb_side", "score_side"]).alias("truth_side"),
    pl.col("match_key").str.extract(DAY_RE, 1).cast(pl.Int64).alias("day"))
mm13 = r13.filter(pl.col("mm")).select(["match_key", "winner_team_code"])
for off in [-1, 1]:
    tgt = truth.filter(pl.col("round") == 13 + off).select(["match_key", pl.col("truth_side").alias("nb")])
    j = mm13.join(tgt, on="match_key", how="inner")
    print(f"\n[9] round-13 mismatch rows: round_end winner vs truth of round {13 + off} (same match): "
          f"agree {int((j['winner_team_code'] == j['nb']).sum())}/{j.height}")
# and the true r13 truth for the same rows
tgt13 = truth.filter(pl.col("round") == 13).select(["match_key", "truth_side"])
print("    round-13 truth (bomb else score) for the same mismatch rows vs round_end winner:",
      int((mm13.join(tgt13, on="match_key")["winner_team_code"] !=
           mm13.join(tgt13, on="match_key")["truth_side"]).sum()), "disagree")

# ---------------------------------------------------------------- 10. bomb-event timing (physical check)
print("\n[10] bomb timing in seconds (tick gaps / tick rate):")
tk = TICKS_PER_S
b = base.with_columns(
    (pl.col("plant_to_out_ticks") / tk).alias("plant_to_out_s"),
    ((pl.col("tick") - pl.col("out_tick")) / tk).alias("re_minus_out_s"),
    pl.when(pl.col("mm")).then(pl.lit("mismatch")).otherwise(pl.lit("match")).alias("grp"))
ev_only = b.with_columns(pl.col("evs").list.first().alias("ev1"))
print("    plant->outcome seconds by event and group (n, median, min, max):")
print(ev_only.group_by(["ev1", "grp"]).agg(
    pl.len().alias("n"),
    pl.col("plant_to_out_s").median().round(2).alias("median_s"),
    pl.col("plant_to_out_s").min().round(2).alias("min_s"),
    pl.col("plant_to_out_s").max().round(2).alias("max_s")).sort(["ev1", "grp"]))
print("    round_end minus outcome seconds, by event and group (n, median, min, max):")
print(ev_only.group_by(["ev1", "grp"]).agg(
    pl.col("re_minus_out_s").median().round(2).alias("median_s"),
    pl.col("re_minus_out_s").min().round(2).alias("min_s"),
    pl.col("re_minus_out_s").max().round(2).alias("max_s")).sort(["ev1", "grp"]))
print("    mismatch rows only: outcome type x round:")
print(ev_only.filter(pl.col("mm")).group_by(["ev1", "rbin"]).len().sort(["ev1", "rbin"]))

# ---------------------------------------------------------------- 11. round_state's own bomb events (third source)
rs_bomb = (RS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("rs_evs")))
bb = base.join(rs_bomb, on=["match_key", "round"], how="left").with_columns(
    pl.col("rs_evs").list.contains("bomb_defused").alias("rs_def"))
print("\n[11] round_state bomb events vs bomb_state labels, mismatch rows, round 13 days 12-25:")
x = bb.filter((pl.col("round") == 13) & pl.col("day").is_between(12, 25))
print(x.with_columns(pl.col("evs").list.first().alias("bs_ev1"))
      .group_by(["mm", "bs_ev1", "rs_def"]).len().sort(["mm", "bs_ev1"]))

print("\nDONE")
