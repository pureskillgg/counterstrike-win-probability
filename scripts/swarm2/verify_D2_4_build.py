"""Verifier D2-4: independent rebuild of the per-round inputs from the raw tome channels.

Does NOT read any D2_* intermediate parquet. Every quantity is recomputed from basic_<channel> folders.

Per (match_key, round), in-memory only (no files written):
  R          round_end.winner_team_code (first row by tick after the player_count filter, as in the claim)
  t_fe       round_state tick of the first round_freeze_end row
  t_end      round_end tick
  t0,ct0     round_state t_score, ct_score at freeze end of round r
  t1,ct1     the same at freeze end of round r+1 (window end of round r)
  sx         side (2 = T, 3 = CT) of team X in round r, from player_info.team_code (team X = players on code 2
             in the first valid 5v5 round of the match). Valid = 10 players, 5 on code 2, 5 on code 3.
  sx_nx      sx in round r+1; sx_pv: sx in round r-1
  s_bomb     side of the first bomb_exploded (2) / bomb_defused (3) event in [t_fe, t_end + SLACK]
  s_bomb_st  same with no slack (window [t_fe, t_end])
  planter    player_team_code of the first bomb_planted row in the round
  t_dead,ct_dead  deaths by side (membership = side at round F). Also reports deaths with unknown membership.
  s_elim     side that won by elimination (CT all five dead -> T; T all five dead -> CT)
  s_score    claim method (team-tracked round_state delta across the swap)
  s_side     sx-free side-field delta (t1-t0 == 1 & ct1-ct0 == 0 -> T; ct1-ct0 == 1 & t1-t0 == 0 -> CT),
             gated on sx == sx_nx (no player-side swap between r and r+1)

ASSUMPTION (carried from the claim, same values): PC 2024 = 20, 2026 = 10; SLACK = 40 ticks.
"""
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
PC = {"2024": 20, "2026": 10}  # ASSUMPTION carried from D2_truth.py
SLACK = 40                     # ASSUMPTION carried from D2_truth.py


def folders(label, ch):
    return sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))


def channel(label, ch, cols):
    """All dataframe_* files of every folder for this channel (no folder is dropped)."""
    fs = folders(label, ch)
    if not fs:
        raise SystemExit(f"no folder for channel {ch} in {label}")
    files = []
    for f in fs:
        files += sorted(glob.glob(f + "/dataframe_*"))
    return pl.concat([pl.read_parquet(p, columns=cols) for p in files], how="vertical_relaxed")


def side_field(side, t, ct):
    """Score field for a side: T-field if side == 2 else CT-field."""
    return pl.when(side == 2).then(t).otherwise(ct)


def build(label, pc_filter=True, slack=SLACK, membership_key="player_id", window_deaths=False, valid_np=(10,)):
    """valid_np=(10,) is the claim's rule (10 players, 5 per side). Use (9, 10) for the 9-player extension."""
    pc = PC[label]

    # --- round_end: first row per (match, round) by tick, after the player_count filter (claim order)
    re_all = channel(label, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    re_ = re_all.filter(pl.col("player_count") == pc) if pc_filter else re_all
    re_ = (re_.sort(["match_key", "round", "tick"])
           .unique(["match_key", "round"], keep="first", maintain_order=True)
           .select(["match_key", "round", pl.col("tick").alias("t_end"),
                    pl.col("winner_team_code").cast(pl.Int64).alias("R"), "win_reason_code"]))

    # --- round_state freeze-end fields
    rs = channel(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
          .unique(["match_key", "round"], keep="first", maintain_order=True)
          .select(["match_key", "round", pl.col("tick").alias("t_fe"),
                   pl.col("t_score").alias("t0"), pl.col("ct_score").alias("ct0")]))
    fe_next = fe.select(["match_key", (pl.col("round") - 1).alias("round"),
                         pl.col("t0").alias("t1"), pl.col("ct0").alias("ct1")])

    base = re_.join(fe, on=["match_key", "round"], how="left").join(fe_next, on=["match_key", "round"], how="left")

    # --- player sides (no winner input)
    pi = channel(label, "player_info", ["match_key", "round", "player_id", "player_id_fixed", "team_code"])
    pi = pi.unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
    cnt = pi.group_by(["match_key", "round"]).agg(
        pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("n2"), (pl.col("team_code") == 3).sum().alias("n3"))
    if valid_np == (10,):
        valid = cnt.filter((pl.col("np") == 10) & (pl.col("n2") == 5) & (pl.col("n3") == 5)).select(["match_key", "round"])
    else:  # extension: any player count in valid_np; the side rule below still requires one consistent side per team
        valid = cnt.filter(pl.col("np").is_in(list(valid_np))).select(["match_key", "round"])
    first = valid.group_by("match_key").agg(pl.col("round").min().alias("F"))
    roster = (pi.join(first, on="match_key").filter(pl.col("round") == pl.col("F")))
    # membership: team X = players on code 2 in round F
    x_ids = roster.filter(pl.col("team_code") == 2).select(["match_key", membership_key]).unique().with_columns(pl.lit(True).alias("isX"))
    pv = pi.join(valid, on=["match_key", "round"]).join(
        x_ids.rename({membership_key: membership_key}), on=["match_key", membership_key], how="left"
    ).with_columns(pl.col("isX").fill_null(False))
    sides = pv.group_by(["match_key", "round"]).agg(
        pl.col("isX").sum().alias("nx"),
        ((pl.col("isX")) & (pl.col("team_code") == 2)).sum().alias("xon2"),
        ((pl.col("isX")) & (pl.col("team_code") == 3)).sum().alias("xon3"),
        ((~pl.col("isX")) & (pl.col("team_code") == 2)).sum().alias("yon2"),
        ((~pl.col("isX")) & (pl.col("team_code") == 3)).sum().alias("yon3"))
    sides = sides.with_columns(
        pl.when((pl.col("nx") == 5) & (pl.col("xon2") == 5) & (pl.col("yon2") == 0)).then(2)
        .when((pl.col("nx") == 5) & (pl.col("xon3") == 5) & (pl.col("yon3") == 0)).then(3)
        .otherwise(None).cast(pl.Int64).alias("sx")).select(["match_key", "round", "sx"])
    base = (base.join(sides, on=["match_key", "round"], how="left")
            .join(sides.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("sx").alias("sx_nx")]),
                  on=["match_key", "round"], how="left")
            .join(sides.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("sx").alias("sx_pv")]),
                  on=["match_key", "round"], how="left"))

    # --- bomb outcome, with and without slack
    bs = channel(label, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
    bo_ev = bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(["match_key", "round", "tick", "event_type"])
    bo_j = bo_ev.join(base.select(["match_key", "round", "t_fe", "t_end"]), on=["match_key", "round"], how="inner")

    def bomb_side(window_end_expr):
        w = bo_j.filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= window_end_expr))
        f = (w.sort("tick").group_by(["match_key", "round"]).agg(
            pl.col("event_type").first().alias("bo_first"),
            pl.col("event_type").n_unique().alias("bo_types"),
            pl.len().alias("bo_n")))
        return f.with_columns(
            pl.when(pl.col("bo_first") == "bomb_exploded").then(2).otherwise(3).cast(pl.Int64).alias("s_bomb_x"),
            pl.col("bo_types").alias("bo_types_x"), pl.col("bo_n").alias("bo_n_x"),
        ).select(["match_key", "round", "s_bomb_x", "bo_types_x", "bo_n_x"])

    b_slack = bomb_side(pl.col("t_end") + slack).rename({"s_bomb_x": "s_bomb", "bo_types_x": "bo_types", "bo_n_x": "bo_n"})
    b_strict = bomb_side(pl.col("t_end")).rename({"s_bomb_x": "s_bomb_st", "bo_types_x": "bo_types_st", "bo_n_x": "bo_n_st"})
    planter = (bs.filter(pl.col("event_type") == "bomb_planted").sort("tick")
               .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter"),
                                                      pl.len().alias("n_planted")))
    base = (base.join(b_slack, on=["match_key", "round"], how="left")
            .join(b_strict, on=["match_key", "round"], how="left")
            .join(planter, on=["match_key", "round"], how="left"))

    # --- elimination side
    dth = channel(label, "player_death", ["match_key", "round", "tick", membership_key])
    dth = dth.with_columns(pl.col(membership_key).cast(pl.Int64))  # player_id_fixed is Float64 in deaths
    dth = dth.join(
        roster.select(["match_key", membership_key, pl.col("team_code").alias("rteam")]).unique(),
        on=["match_key", membership_key], how="left")
    dth = dth.join(base.select(["match_key", "round", "sx", "t_fe", "t_end"]), on=["match_key", "round"], how="left")
    if window_deaths:
        dth = dth.filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end")))
    dth = dth.with_columns(
        pl.when(pl.col("rteam") == 2).then(pl.lit(True)).when(pl.col("rteam") == 3).then(pl.lit(False))
        .otherwise(None).alias("isXd"))
    # deaths of players not in the round-F roster (membership unknown): counted separately, not assigned to a side
    dth = dth.with_columns(
        pl.when(pl.col("isXd") & (pl.col("sx") == 2)).then(2)
        .when(pl.col("isXd") & (pl.col("sx") == 3)).then(3)
        .when((~pl.col("isXd")) & (pl.col("sx") == 2)).then(3)
        .when((~pl.col("isXd")) & (pl.col("sx") == 3)).then(2)
        .otherwise(None).cast(pl.Int64).alias("vside"))
    el = dth.group_by(["match_key", "round"]).agg(
        (pl.col("vside") == 2).sum().alias("t_dead"), (pl.col("vside") == 3).sum().alias("ct_dead"),
        pl.col("isXd").is_null().sum().alias("n_unknown_member"))
    el = el.with_columns(
        pl.when((pl.col("ct_dead") == 5) & (pl.col("t_dead") < 5)).then(2)
        .when((pl.col("t_dead") == 5) & (pl.col("ct_dead") < 5)).then(3)
        .otherwise(None).cast(pl.Int64).alias("s_elim"))
    base = base.join(el, on=["match_key", "round"], how="left")
    base = base.with_columns(pl.col("t_dead").fill_null(0), pl.col("ct_dead").fill_null(0),
                             pl.col("n_unknown_member").fill_null(0))

    # --- truths
    base = base.with_columns(
        # claim method: team X score follows the side it stood on in round r (r+1 for the end value)
        (side_field(pl.col("sx"), pl.col("t0"), pl.col("ct0")).alias("x0")),
        (side_field(pl.col("sx_nx"), pl.col("t1"), pl.col("ct1")).alias("x1")),
        (side_field(5 - pl.col("sx"), pl.col("t0"), pl.col("ct0")).alias("y0")),
        (side_field(5 - pl.col("sx_nx"), pl.col("t1"), pl.col("ct1")).alias("y1")),
    )
    base = base.with_columns(
        (pl.col("x1") - pl.col("x0")).alias("dX"), (pl.col("y1") - pl.col("y0")).alias("dY"))
    base = base.with_columns(
        pl.when((pl.col("dX") == 1) & (pl.col("dY") == 0)).then(pl.col("sx"))
        .when((pl.col("dY") == 1) & (pl.col("dX") == 0)).then(5 - pl.col("sx"))
        .otherwise(None).cast(pl.Int64).alias("s_score"))
    base = base.with_columns(
        (pl.col("t1") - pl.col("t0")).alias("dT_side"), (pl.col("ct1") - pl.col("ct0")).alias("dCT_side"),
        (pl.col("sx") != pl.col("sx_nx")).alias("swap_next"),
        (pl.col("sx") != pl.col("sx_pv")).alias("swap_now"))
    base = base.with_columns(
        pl.when((pl.col("dT_side") == 1) & (pl.col("dCT_side") == 0)).then(2)
        .when((pl.col("dCT_side") == 1) & (pl.col("dT_side") == 0)).then(3)
        .otherwise(None).cast(pl.Int64).alias("s_side_raw"))
    base = base.with_columns(
        pl.when(pl.col("sx_nx") == pl.col("sx")).then(pl.col("s_side_raw")).otherwise(None).cast(pl.Int64).alias("s_side"))
    base = base.with_columns(
        pl.coalesce(["s_score", "s_bomb", "s_elim"]).cast(pl.Int64).alias("T"),
        pl.col("R").cast(pl.Int64))
    base = base.with_columns(
        pl.when(pl.col("T").is_not_null()).then(pl.col("R") != pl.col("T")).otherwise(None).alias("flip"),
        pl.lit(label).alias("year"),
        pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
    )
    return base


def bin_expr():
    return (pl.when(pl.col("round") == 13).then(pl.lit("r13"))
            .when(pl.col("round") == 25).then(pl.lit("r25"))
            .when(pl.col("round") == 28).then(pl.lit("r28"))
            .when(pl.col("round") <= 12).then(pl.lit("r01_12"))
            .when(pl.col("round") <= 24).then(pl.lit("r14_24"))
            .otherwise(pl.lit("r29plus")).alias("bin"))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for lab in ["2024", "2026"]:
        for ch in ["round_end", "round_state", "player_info", "player_death", "bomb_state"]:
            print(lab, ch, "folders:", [Path(f).name for f in folders(lab, ch)])
