"""verify_D2_7 step 1: independent recount of 2024 (and 2026 control) round-winner flips from raw channels.

Independent of notes/swarm2/D2.md code and of cross_period.py. Reads one channel at a time with
column projection. Nothing is written to disk.

Side truth (no winner code, no team-identity assumption):
  - swap indicator s(r) for round r -> r+1: players whose player_info team_code changes side between
    round r and r+1 (majority rule). ASSUMPTION: membership is side-attached in player_info.
  - score truth: freeze-end t_score / ct_score at r and r+1. If s=0 the side that gained one point won.
    If s=1 the team that was on CT at r is on T at r+1, so its score is t_score(r+1), and vice versa.
    ASSUMPTION: score fields are side-attached (as in D2).
Cross-checks:
  - bomb truth: bomb_defused -> CT wins, bomb_exploded -> T wins, window [freeze_end, round_end + 40].
  - elimination truth: all five membership-T players dead in [freeze_end, round_end] and no plant in
    window -> CT wins; all five membership-CT players dead -> T wins. Otherwise undecided.
Recorded label: R = round_end.winner_team_code (CT if == 3). Flag = score truth != R.
"""

import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"
TOME26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUF26 = ".2026-09-18,2026-10-08"
T, CT = 2, 3
TAIL = 40  # ASSUMPTION, same as D2: bomb events allowed up to 40 ticks past round end
PC_FILTER = {"2024": 20, "2026": 10}  # full-round filter used by D2 and e2 (total players)


def read(tome, suf, ch, cols):
    folder = tome / ("header" if ch == "header" else f"basic_{ch}{suf}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    if not files:
        raise SystemExit(f"no pages in {folder}")
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
    if "round" in df.columns:
        df = df.with_columns(pl.col("round").cast(pl.Int64))
    if "tick" in df.columns:
        df = df.with_columns(pl.col("tick").cast(pl.Int64))
    return df


def group_expr():
    r = pl.col("round")
    return (pl.when(r == 13).then(pl.lit("r13")).when(r == 25).then(pl.lit("r25"))
            .when(r == 28).then(pl.lit("r28")).when(r <= 12).then(pl.lit("r01-12"))
            .when(r <= 24).then(pl.lit("r14-24")).otherwise(pl.lit("r29+")).alias("grp"))


def build(year, tome, suf):
    print(f"\n######## {year} ########")
    hdr = read(tome, suf, "header", ["key", "tick_rate", "map_name", "match_date"])
    print(f"header matches: {hdr.height}; tick_rate counts:")
    print(hdr.group_by("tick_rate").len().sort("tick_rate"))

    RE = read(tome, suf, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    n_re = RE.height
    dup_re = n_re - RE.unique(subset=["match_key", "round"]).height
    RE = RE.unique(subset=["match_key", "round"], keep="first", maintain_order=True)
    print(f"round_end rows: {n_re}; duplicate (match, round) rows dropped: {dup_re}; "
          f"player_count counts: {RE.group_by('player_count').len().sort('player_count').rows()}")

    RS = read(tome, suf, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    frz_all = RS.filter(pl.col("event_type") == "round_freeze_end")
    n_frz = frz_all.height
    frz = (frz_all.select(["match_key", "round", "tick", "t_score", "ct_score"])
           .sort("tick")
           .unique(subset=["match_key", "round"], keep="first", maintain_order=True))
    print(f"freeze-end rows: {n_frz}; kept per (match, round): {frz.height}; null score rows: "
          f"{frz.filter(pl.col('t_score').is_null() | pl.col('ct_score').is_null()).height}")
    sum_ok = frz.filter((pl.col("t_score") + pl.col("ct_score")) == (pl.col("round") - 1)).height
    print(f"freeze rows with t_score + ct_score == round - 1: {sum_ok} of {frz.height}")
    del RS, frz_all

    PI_raw = read(tome, suf, "player_info", ["match_key", "round", "player_id", "team_code"])
    PI = (PI_raw.filter(pl.col("team_code").is_in([T, CT]))
          .unique(subset=["match_key", "round", "player_id"], keep="first", maintain_order=True))
    del PI_raw

    # swap indicator r -> r+1 from membership (majority of players that changed side)
    pi_next = PI.select(["match_key", "round", "player_id", pl.col("team_code").alias("team_next")]) \
        .with_columns((pl.col("round") - 1).alias("round"))
    sw = (PI.join(pi_next, on=["match_key", "round", "player_id"], how="inner")
          .group_by(["match_key", "round"])
          .agg((pl.col("team_code") != pl.col("team_next")).sum().alias("diff"),
               (pl.col("team_code") == pl.col("team_next")).sum().alias("same"))
          .with_columns(pl.when(pl.col("diff") > pl.col("same")).then(1)
                        .when(pl.col("same") > pl.col("diff")).then(0)
                        .otherwise(None).cast(pl.Int64).alias("swap")))
    print(f"swap pairs (round r -> r+1 with membership): {sw.height}; swap=1: "
          f"{sw.filter(pl.col('swap') == 1).height}; swap=0: {sw.filter(pl.col('swap') == 0).height}; "
          f"undecided: {sw.filter(pl.col('swap').is_null()).height}")

    mem = PI.group_by(["match_key", "round"]).agg((pl.col("team_code") == T).sum().alias("n_t"),
                                                  (pl.col("team_code") == CT).sum().alias("n_ct"))

    # score truth
    nxt = frz.select(["match_key", (pl.col("round") - 1).alias("round"),
                      pl.col("t_score").alias("t_next"), pl.col("ct_score").alias("ct_next")])
    pairs = (frz.select(["match_key", "round", "t_score", "ct_score"])
             .join(nxt, on=["match_key", "round"], how="inner")
             .join(sw.select(["match_key", "round", "swap"]), on=["match_key", "round"], how="left"))
    pairs = pairs.with_columns(
        pl.when(pl.col("swap") == 0).then(pl.col("ct_next") - pl.col("ct_score"))
        .when(pl.col("swap") == 1).then(pl.col("t_next") - pl.col("ct_score")).alias("ct_gain"),
        pl.when(pl.col("swap") == 0).then(pl.col("t_next") - pl.col("t_score"))
        .when(pl.col("swap") == 1).then(pl.col("ct_next") - pl.col("t_score")).alias("t_gain"))
    pairs = pairs.with_columns(
        pl.when((pl.col("ct_gain") == 1) & (pl.col("t_gain") == 0)).then(1)
        .when((pl.col("t_gain") == 1) & (pl.col("ct_gain") == 0)).then(0)
        .otherwise(None).cast(pl.Int64).alias("truth_ct"))
    print(f"score truth pairs (freeze r and r+1 both present): {pairs.height}; decided: "
          f"{pairs.filter(pl.col('truth_ct').is_not_null()).height}")

    base = (RE.join(frz.select(["match_key", "round", "tick", "t_score", "ct_score"]).rename({"tick": "t_frz"}),
                    on=["match_key", "round"], how="left")
            .rename({"tick": "t_end"})
            .join(pairs.select(["match_key", "round", "truth_ct", "swap"]), on=["match_key", "round"], how="left")
            .join(mem, on=["match_key", "round"], how="left")
            .with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d\d)/", 1).alias("day"),
                          (pl.col("winner_team_code") == CT).cast(pl.Int64).alias("R_ct"),
                          group_expr()))
    base = base.with_columns(((pl.col("n_t") == 5) & (pl.col("n_ct") == 5)).alias("valid_mem"))
    win = base.select(["match_key", "round", "t_frz", "t_end"])

    # bomb truth
    BS = read(tome, suf, "bomb_state", ["match_key", "round", "tick", "event_type"])
    bw = (BS.join(win, on=["match_key", "round"], how="inner"))
    plant_in = (bw.filter((pl.col("event_type") == "bomb_planted")
                          & (pl.col("tick") >= pl.col("t_frz")) & (pl.col("tick") <= pl.col("t_end")))
                .group_by(["match_key", "round"]).agg(pl.len().alias("plant_inwin")))
    bomb = (bw.filter((pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
                      & (pl.col("tick") >= pl.col("t_frz")) & (pl.col("tick") <= pl.col("t_end") + TAIL))
            .group_by(["match_key", "round"])
            .agg((pl.col("event_type") == "bomb_defused").any().alias("defused"),
                 (pl.col("event_type") == "bomb_exploded").any().alias("exploded"))
            .with_columns(pl.when(pl.col("defused") & ~pl.col("exploded")).then(1)
                          .when(pl.col("exploded") & ~pl.col("defused")).then(0)
                          .otherwise(None).cast(pl.Int64).alias("bomb_ct")))
    print(f"bomb outcome rounds (window + tail): {bomb.height}; conflicting defused+exploded: "
          f"{bomb.filter(pl.col('defused') & pl.col('exploded')).height}")

    # elimination truth and death-side consistency
    DE = read(tome, suf, "player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
    dw = DE.join(win, on=["match_key", "round"], how="inner").filter(
        (pl.col("tick") >= pl.col("t_frz")) & (pl.col("tick") <= pl.col("t_end")))
    dw = dw.join(PI.select(["match_key", "round", "player_id", "team_code"]),
                 on=["match_key", "round", "player_id"], how="left")
    side_chk = dw.filter(pl.col("player_team_code").is_not_null() & pl.col("team_code").is_not_null()).with_columns(
        (pl.col("player_team_code") != pl.col("team_code")).alias("mism"))
    death_mism = (side_chk.group_by(["match_key", "round"])
                  .agg(pl.len().alias("deaths_chk"), pl.col("mism").sum().alias("death_side_mism")))
    dead = (dw.filter(pl.col("team_code").is_not_null())
            .unique(subset=["match_key", "round", "player_id"])
            .group_by(["match_key", "round", "team_code"]).len()
            .pivot(on="team_code", index=["match_key", "round"], values="len")
            .rename({"2": "dead_t", "3": "dead_ct"}))
    for c in ("dead_t", "dead_ct"):
        if c not in dead.columns:
            dead = dead.with_columns(pl.lit(0).cast(pl.UInt32).alias(c))
    elim = (base.select(["match_key", "round", "n_t", "n_ct"])
            .join(dead, on=["match_key", "round"], how="left")
            .join(plant_in, on=["match_key", "round"], how="left")
            .with_columns(pl.col("dead_t").fill_null(0), pl.col("dead_ct").fill_null(0),
                          pl.col("plant_inwin").fill_null(0))
            .with_columns(((pl.col("n_t") == 5) & (pl.col("dead_t") == 5)).alias("allT"),
                          ((pl.col("n_ct") == 5) & (pl.col("dead_ct") == 5)).alias("allCT")))
    elim = elim.with_columns(
        pl.when(pl.col("allCT") & ~pl.col("allT")).then(0)
        .when(pl.col("allT") & ~pl.col("allCT") & (pl.col("plant_inwin") == 0)).then(1)
        .otherwise(None).cast(pl.Int64).alias("elim_ct"))

    tab = (base.join(bomb.select(["match_key", "round", "bomb_ct"]), on=["match_key", "round"], how="left")
           .join(elim.select(["match_key", "round", "elim_ct", "plant_inwin"]),
                 on=["match_key", "round"], how="left")
           .join(death_mism, on=["match_key", "round"], how="left")
           .with_columns(pl.col("death_side_mism").fill_null(0), pl.col("deaths_chk").fill_null(0),
                         (pl.col("truth_ct") != pl.col("R_ct")).cast(pl.Int64).alias("flag_raw"),
                         pl.when(pl.col("truth_ct").is_not_null())
                         .then((pl.col("truth_ct") != pl.col("R_ct")).cast(pl.Int64)).otherwise(None).alias("flag")))
    return tab, {"hdr": hdr, "death_side": side_chk, "n_re": n_re, "dup_re": dup_re}


def fmt_rate(num, den):
    return f"{num}/{den} = {100 * num / den:.2f}%" if den else "0/0"


def summarize(year, tab):
    pc = PC_FILTER[year]
    full = tab.filter(pl.col("player_count") == pc)
    print(f"\n-- {year}: rounds (player_count == {pc}): {full.height}; with score truth: "
          f"{full.filter(pl.col('truth_ct').is_not_null()).height}; with valid membership: "
          f"{full.filter(pl.col('valid_mem')).height}")
    print("   note: D2 says 2024 score truth covers 63,141 rounds of 78,595 (score only).")

    t = full.filter(pl.col("truth_ct").is_not_null())
    g = (t.group_by("grp").agg(pl.len().alias("n_truth"), pl.col("flag").sum().alias("flagged"))
         .with_columns((pl.col("flagged") / pl.col("n_truth") * 100).round(2).alias("pct"))
         .sort("grp"))
    print("   flagged (score truth != recorded winner) by round group:")
    print(g)
    print(f"   total flagged in rounds with score truth: {int(t['flag'].sum())} of {t.height}")
    off = t.filter(~pl.col("round").is_in([13, 25, 28]) & (pl.col("flag") == 1))
    print(f"   flagged outside rounds 13, 25, 28: {off.height}; by round: "
          f"{off.group_by('round').len().sort('round').rows()}")

    print("   round 13 flagged by match day (score truth):")
    r13 = t.filter(pl.col("round") == 13)
    print(r13.group_by("day").agg(pl.len().alias("n"), pl.col("flag").sum().alias("flagged"),
                                  (pl.col("flag").mean() * 100).round(1).alias("pct"))
          .sort("day"))

    # bomb and elimination cross-checks
    b = full.filter(pl.col("truth_ct").is_not_null() & pl.col("bomb_ct").is_not_null())
    print(f"   bomb cross-check: rounds with score truth and bomb outcome: {b.height}; "
          f"score == bomb: {int((b['truth_ct'] == b['bomb_ct']).sum())}; recorded R == bomb: "
          f"{int((b['R_ct'] == b['bomb_ct']).sum())}")
    print("     by group (score==bomb / R==bomb / n):")
    print(b.group_by("grp").agg((pl.col("truth_ct") == pl.col("bomb_ct")).sum().alias("score_eq_bomb"),
                                (pl.col("R_ct") == pl.col("bomb_ct")).sum().alias("R_eq_bomb"),
                                pl.len().alias("n")).sort("grp"))
    e = full.filter(pl.col("truth_ct").is_not_null() & pl.col("elim_ct").is_not_null())
    print(f"   elimination cross-check: rounds with score truth and elimination truth: {e.height}; "
          f"score == elim: {int((e['truth_ct'] == e['elim_ct']).sum())}; R == elim: "
          f"{int((e['R_ct'] == e['elim_ct']).sum())}")
    print(e.group_by("grp").agg((pl.col("truth_ct") == pl.col("elim_ct")).sum().alias("score_eq_elim"),
                                (pl.col("R_ct") == pl.col("elim_ct")).sum().alias("R_eq_elim"),
                                pl.len().alias("n")).sort("grp"))

    # death-side consistency (feature side for alive counts in cross_period.py uses player_death.player_team_code)
    ds = full.select(["match_key", "round", "grp", "flag", "deaths_chk", "death_side_mism"])
    print("   death side (player_death.player_team_code vs player_info team) mismatches, by flag:")
    print(ds.group_by(["grp", "flag"]).agg(pl.col("deaths_chk").sum().alias("deaths_checked"),
                                           pl.col("death_side_mism").sum().alias("mismatch")).sort(["grp", "flag"]))


def shift_test(tab):
    """Does R at round 13 in flagged rows match the score truth of round 12 or 14 instead?"""
    t = tab.filter(pl.col("truth_ct").is_not_null()).select(["match_key", "round", "truth_ct", "R_ct", "flag", "player_count"])
    # T12: truth of round r-1, keyed to round r; T14: truth of round r+1, keyed to round r
    t_m1 = t.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("truth_ct").alias("T_r_minus_1")])
    t_p1 = t.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("truth_ct").alias("T_r_plus_1")])
    x = (t.filter(pl.col("round") == 13).join(t_m1, on=["match_key", "round"], how="left")
         .join(t_p1, on=["match_key", "round"], how="left"))
    x = x.filter(pl.col("player_count") == 20)
    fl = x.filter(pl.col("flag") == 1)
    print(f"\n-- shift test (2024, round 13 flagged rows, player_count 20): flagged n={fl.height}")
    print(f"   R13 == T12 (score truth of round 12): {int((fl['R_ct'] == fl['T_r_minus_1']).sum())} "
          f"(n with T12: {int(fl['T_r_minus_1'].is_not_null().sum())})")
    print(f"   R13 == T14 (score truth of round 14): {int((fl['R_ct'] == fl['T_r_plus_1']).sum())} "
          f"(n with T14: {int(fl['T_r_plus_1'].is_not_null().sum())})")


def main():
    tab24, info24 = build("2024", TOME24, SUF24)
    summarize("2024", tab24)
    shift_test(tab24)
    tab26, info26 = build("2026", TOME26, SUF26)
    summarize("2026", tab26)
    # save nothing to disk; this run is printed to logs\swarm2\verify_D2_7.log by the caller


if __name__ == "__main__":
    main()
