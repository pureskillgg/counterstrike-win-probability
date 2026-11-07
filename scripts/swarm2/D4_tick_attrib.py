"""D4: tick-based round re-attribution of bomb events (hypothesis: bomb events carry an off-by-one round field).

For every bomb event in bomb_state (bomb_planted, bomb_defused, bomb_exploded), find the round r whose live
window [round_freeze_end tick, round_end tick] contains the event tick, and compare r with the event's own
round field (called lbl here). The same check runs on the bomb rows inside round_state as a second source.

Outcome rows (bomb_defused, bomb_exploded) are then re-attributed by tick, and the side-vs-winner mismatch
is recomputed with the same rule as scripts\\swarm\\e2_side_check_r13.py (defused -> side 3 CT,
exploded -> side 2 T, mismatch if winner_team_code differs; player_count == 20 only).

ASSUMPTIONS (marked in the report)
  A1. Window = [round_freeze_end.tick, round_end.tick], both ends inclusive.
  A2. An event outside every window is assigned to the nearest window (distance in ticks) and is
      reported as outside_window, never as agree.
  A3. Duplicate (match, round) rows in round_end or round_freeze_end: first row kept (counts reported).
  A4. Match date for the per-day tables comes from the match_key path (csds/YYYY/MM/DD/...).
  A5. Matches are not de-duplicated, to stay comparable with e2_side_check_r13.py (499 mismatches).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D4_tick_attrib.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}
BOMB_TYPES = ["bomb_planted", "bomb_defused", "bomb_exploded"]
OUTCOMES = ["bomb_defused", "bomb_exploded"]
# round_end.player_count of a full round: 20 in 2024 (as in e2_side_check_r13.py), 10 in 2026 (as in bakeoff.py)
PC = {"2024": 20, "2026": 10}
T, CT = 2, 3
BIG = 10**9


def read(base, ch, cols):
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def rbin_expr(col):
    return (pl.when(pl.col(col) == 13).then(pl.lit("r13"))
            .when(pl.col(col) == 25).then(pl.lit("r25"))
            .otherwise(pl.lit("other")))


def load_windows(base):
    rs = read(base, "round_state", ["match_key", "round", "tick", "event_type"])
    live_all = (rs.filter(pl.col("event_type") == "round_freeze_end")
                .select(["match_key", "round", pl.col("tick").alias("live")]))
    live = live_all.unique(["match_key", "round"], keep="first")
    re_ = read(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    conflict = (re_.group_by(["match_key", "round"])
                .agg(pl.col("winner_team_code").n_unique().alias("nw"), pl.col("tick").n_unique().alias("nt")))
    re_u = re_.unique(["match_key", "round"], keep="first").rename({"tick": "end"})
    W = re_u.join(live, on=["match_key", "round"], how="left")
    info = {
        "round_freeze_end rows": live_all.height,
        "round_freeze_end duplicate rows (dropped)": live_all.height - live.height,
        "round_end rows": re_.height,
        "round_end duplicate rows (dropped)": re_.height - re_u.height,
        "round_end (match,round) with conflicting dup tick/winner": conflict.filter(
            (pl.col("nw") > 1) | (pl.col("nt") > 1)).height,
        "windows": W.height,
        "windows without live start": W.filter(pl.col("live").is_null()).height,
        "windows with end < live": W.filter(pl.col("live").is_not_null() & (pl.col("end") < pl.col("live"))).height,
    }
    return W, info


def attribute(ev, W):
    """ev: eid, match_key, lbl, tick, event_type. Returns one row per event with the tick-based round."""
    cand = (W.filter(pl.col("live").is_not_null())
            .select(["match_key", pl.col("round").alias("tr"), "live", "end"]))
    X = ev.join(cand, on="match_key", how="left")
    X = X.with_columns(
        ((pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end"))).fill_null(False).alias("inside"),
        pl.when(pl.col("live").is_null()).then(pl.lit(BIG))
        .otherwise(pl.max_horizontal(pl.col("live") - pl.col("tick"), pl.col("tick") - pl.col("end"), pl.lit(0)))
        .alias("dist"),
    )
    n_in = X.group_by("eid").agg(pl.col("inside").sum().alias("n_inside"))
    best = X.sort(["eid", "dist", "tr"], nulls_last=True).unique(subset=["eid"], keep="first")
    best = best.join(n_in, on="eid", how="left")
    best = best.with_columns(
        pl.col("inside").fill_null(False),
        (pl.col("lbl") - pl.col("tr")).alias("delta"),
        pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
    ).with_columns(
        pl.when(pl.col("tr").is_null()).then(pl.lit("no_window"))
        .when(pl.col("inside") & (pl.col("delta") == 0)).then(pl.lit("agree"))
        .when(pl.col("inside")).then(pl.lit("disagree_inside"))
        .otherwise(pl.lit("outside_window")).alias("cls"),
    )
    return best


def grouped_mismatch(outs, col, W, pc):
    """Replicates e2_side_check_r13.py: group outcome rows by (match, round), bomb side from the set of
    outcome types (defused -> 3 CT, else 2 T), compare with winner_team_code of that round (full rounds only)."""
    g = (outs.group_by(["match_key", col])
         .agg(pl.col("event_type").unique().alias("evs"), pl.len().alias("n_out"))
         .rename({col: "rr"}))
    wj = W.select(["match_key", pl.col("round").alias("rr"), "winner_team_code", "player_count"])
    j = (g.join(wj, on=["match_key", "rr"], how="inner")
         .filter(pl.col("player_count") == pc)
         .with_columns(
             pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bomb_side"),
             pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
         )
         .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
                       rbin_expr("rr").alias("rbin")))
    return j


def section(title):
    print(f"\n-- {title}")


def run_era(era, base):
    pc = PC[era]
    print(f"\n==================== ERA {era} (full round = player_count {pc}) ====================")
    W, winfo = load_windows(base)
    print("round_end player_count distribution:",
          W.group_by("player_count").len().sort("player_count").rows())
    print("window build:")
    for k, v in winfo.items():
        print(f"  {k}: {v}")

    bs = (read(base, "bomb_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("event_type").is_in(BOMB_TYPES))
          .rename({"round": "lbl"})
          .with_row_index(name="eid"))
    rs = (read(base, "round_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("event_type").is_in(BOMB_TYPES))
          .rename({"round": "lbl"})
          .with_row_index(name="eid"))
    print(f"\nbomb_state bomb rows: {bs.height}; round_state bomb rows: {rs.height}")

    A = attribute(bs, W)
    B = attribute(rs, W)

    section("1. bomb_state events: class by event_type (A = tick-based attribution)")
    print(A.group_by(["event_type", "cls"]).len().sort(["event_type", "cls"]))
    print(f"events with n_inside > 1 (overlapping windows): {A.filter(pl.col('n_inside') > 1).height}")
    print(f"events with n_inside == 0 (no window contains tick): {A.filter(pl.col('n_inside') == 0).height}")

    section("1b. delta = event round label minus tick-based round, for disagree_inside and outside_window")
    print(A.filter(pl.col("cls").is_in(["disagree_inside", "outside_window"]))
          .group_by(["event_type", "cls", "delta"]).len().sort(["event_type", "cls", "delta"]))

    section("1c. same classification on round_state bomb rows (second source, B)")
    print(B.group_by(["event_type", "cls"]).len().sort(["event_type", "cls"]))

    section("1d. cross-source label agreement: bomb_state row vs round_state row with same (match, tick, event)")
    bk = A.select(["match_key", "tick", "event_type", "lbl"])
    rk = B.select(["match_key", "tick", "event_type", pl.col("lbl").alias("lbl_rs")]).unique(
        ["match_key", "tick", "event_type"], keep="first")
    jj = bk.join(rk, on=["match_key", "tick", "event_type"], how="left")
    print(f"bomb_state rows: {jj.height}; with a round_state twin: {jj.filter(pl.col('lbl_rs').is_not_null()).height}")
    print(jj.with_columns(
        pl.when(pl.col("lbl_rs").is_null()).then(pl.lit("no_twin"))
        .when(pl.col("lbl_rs") == pl.col("lbl")).then(pl.lit("same_label"))
        .otherwise(pl.lit("different_label")).alias("x")).group_by("x").len().sort("x"))

    section("2. disagree_inside by day (all bomb events, bomb_state)")
    d = (A.group_by("day").agg(
        pl.len().alias("events"),
        (pl.col("cls") == "agree").sum().alias("agree"),
        (pl.col("cls") == "disagree_inside").sum().alias("disagree_inside"),
        (pl.col("cls") == "outside_window").sum().alias("outside_window"),
        (pl.col("cls") == "no_window").sum().alias("no_window"),
    ).with_columns((pl.col("disagree_inside") / pl.col("events")).alias("dis_rate")).sort("day"))
    print(d)

    section("2b. disagree_inside by day, outcome events only (defused + exploded)")
    dout = (A.filter(pl.col("event_type").is_in(OUTCOMES)).group_by("day").agg(
        pl.len().alias("outcome_events"),
        (pl.col("cls") == "disagree_inside").sum().alias("disagree_inside"),
        (pl.col("cls") == "outside_window").sum().alias("outside_window"),
        (pl.col("delta") == 1).sum().alias("delta_plus1"),
        (pl.col("delta") == -1).sum().alias("delta_minus1"),
    ).sort("day"))
    print(dout)

    section("2c. disagree_inside by label round, outcome events (rounds 10-16 and 24-26 shown; all rounds below)")
    byr = (A.filter(pl.col("event_type").is_in(OUTCOMES)).group_by("lbl").agg(
        pl.len().alias("outcome_events"),
        (pl.col("cls") == "disagree_inside").sum().alias("disagree_inside"),
        (pl.col("cls") == "outside_window").sum().alias("outside_window"),
    ).sort("lbl"))
    print(byr.filter(pl.col("lbl").is_between(10, 16) | pl.col("lbl").is_between(24, 26)))
    print("all rounds, totals:", byr.select(["outcome_events", "disagree_inside", "outside_window"]).sum().row(0))

    section("3. label-based outcome mismatch (replicates e2_side_check_r13.py; grouped by (match, round))")
    outs_lbl = A.filter(pl.col("event_type").is_in(OUTCOMES)).select(["match_key", "lbl", "event_type"])
    L = grouped_mismatch(outs_lbl, "lbl", W, pc)
    print(f"rounds with outcome (player_count 20): {L.height}; mismatches: {int(L['mm'].sum())}")
    print(L.group_by("rbin").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"),
                                 pl.col("mm").mean().alias("rate")).sort("rbin"))
    print(L.group_by("day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"),
                                pl.col("mm").mean().alias("rate")).sort("day"))

    section("4. tick-based outcome mismatch (outcome events inside a window, grouped by (match, tick-round))")
    outs_tick = (A.filter(pl.col("event_type").is_in(OUTCOMES) & pl.col("inside"))
                 .select(["match_key", "tr", "event_type"]))
    Tk = grouped_mismatch(outs_tick, "tr", W, pc)
    n_out_inside = outs_tick.height
    print(f"outcome events inside a window: {n_out_inside}; dropped from tick grouping (outside window): "
          f"{A.filter(pl.col('event_type').is_in(OUTCOMES) & ~pl.col('inside')).height}")
    coll = Tk.filter(pl.col("n_out") > 1).height
    print(f"tick-round groups with >1 outcome event: {coll}")
    print(f"rounds with outcome (player_count 20): {Tk.height}; mismatches: {int(Tk['mm'].sum())}")
    print(Tk.group_by("rbin").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"),
                                  pl.col("mm").mean().alias("rate")).sort("rbin"))
    print(Tk.group_by("day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"),
                                 pl.col("mm").mean().alias("rate")).sort("day"))

    section("5. event-level 2x2 for outcome events: label-round winner vs tick-round winner")
    oe = A.filter(pl.col("event_type").is_in(OUTCOMES)).with_columns(
        pl.when(pl.col("event_type") == "bomb_defused").then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bomb_side"))
    wl = W.select(["match_key", pl.col("round").alias("lbl"), pl.col("winner_team_code").alias("w_lbl"),
                   pl.col("player_count").alias("pc_lbl")])
    wt = W.select(["match_key", pl.col("round").alias("tr"), pl.col("winner_team_code").alias("w_tr"),
                   pl.col("player_count").alias("pc_tr")])
    oe = oe.join(wl, on=["match_key", "lbl"], how="left").join(wt, on=["match_key", "tr"], how="left")
    oe = oe.with_columns(
        (pl.col("bomb_side") == pl.col("w_lbl")).alias("ok_lbl"),
        (pl.col("bomb_side") == pl.col("w_tr")).alias("ok_tr"),
    )
    cl = oe.filter((pl.col("pc_lbl") == pc))
    print(f"outcome events, label round has player_count {pc}:")
    print(cl.group_by(["cls", "ok_lbl"]).len().sort(["cls", "ok_lbl"]))
    print("...split by label round bin:")
    print(cl.with_columns(rbin_expr("lbl").alias("rbin")).group_by(["rbin", "cls", "ok_lbl"]).len()
          .sort(["rbin", "cls", "ok_lbl"]))
    ct = oe.filter(pl.col("cls") == "agree").filter(pl.col("pc_tr") == pc)
    print(f"agree events (tick round == label round, full round): {ct.height}")

    section("6. the label-mismatched round-13 groups: where do their outcome ticks sit?")
    mm_keys = L.filter(pl.col("mm") & (pl.col("rr") == 13)).select(["match_key", "rr"]).rename({"rr": "lbl"})
    mmev = oe.join(mm_keys, on=["match_key", "lbl"], how="inner")
    print(f"label-mismatched r13 groups: {mm_keys.height}; their outcome events: {mmev.height}")
    print(mmev.group_by(["cls", "delta"]).len().sort(["cls", "delta"]))
    print("tick-round winner agrees with the event's bomb side (ok_tr) for these events:")
    print(mmev.group_by(["ok_tr"]).len().sort("ok_tr"))
    print("sample rows (first 12):")
    print(mmev.select(["match_key", "lbl", "tr", "event_type", "tick", "live", "end", "delta", "cls",
                       "w_lbl", "w_tr", "ok_tr"]).head(12))

    section("6b. control: r13 groups that agree with the label winner (same day mix)")
    ok_keys = L.filter((~pl.col("mm")) & (pl.col("rr") == 13)).select(["match_key", "rr"]).rename({"rr": "lbl"})
    okev = oe.join(ok_keys, on=["match_key", "lbl"], how="inner")
    print(f"agreeing r13 groups: {ok_keys.height}; their outcome events: {okev.height}")
    print(okev.group_by(["cls", "delta"]).len().sort(["cls", "delta"]))

    section("7. matches with label-mismatched r13 outcome, top 5 by count")
    mm_per_match = mmev.group_by("match_key").agg(pl.len().alias("n_ev")).sort("n_ev", descending=True)
    print(f"matches with label-mismatched r13 outcome: {mm_per_match.height}")
    print(mm_per_match.head(5))

    section("8. outcome mismatch profile by label round (grouped, label-based), rounds 1-30")
    prof = (L.group_by("rr").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
            .with_columns((pl.col("mm") / pl.col("n")).alias("rate")).sort("rr"))
    print(prof.rename({"rr": "round"}))

    section("9. outside-window events: which side of the window, and how far (ticks; 64 ticks = 1 s)")
    ow = A.filter(pl.col("cls") == "outside_window").with_columns(
        pl.when(pl.col("tick") > pl.col("end")).then(pl.lit("after_round_end"))
        .otherwise(pl.lit("before_live")).alias("side"),
        pl.when(pl.col("tick") > pl.col("end")).then(pl.col("tick") - pl.col("end"))
        .otherwise(pl.col("live") - pl.col("tick")).alias("gap_ticks"))
    print(ow.group_by(["event_type", "side"]).agg(
        pl.len().alias("n"), pl.col("gap_ticks").median().alias("median_gap"),
        pl.col("gap_ticks").quantile(0.9).alias("p90_gap"), pl.col("gap_ticks").max().alias("max_gap"),
    ).sort(["event_type", "side"]))

    return {"A": A, "B": B, "W": W, "L": L, "Tk": Tk, "oe": oe}


if __name__ == "__main__":
    for era, base in TOMES.items():
        run_era(era, base)
    print("\nDONE")
