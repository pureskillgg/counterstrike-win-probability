"""verify D2-5: does win_reason_code carry the round-13 side defect, and is it an independent reference?

Independent recomputation (does not read D2 tables except as a final cross-check):
  R        = winner_team_code (round_end, player_count filter as in D2: 2024 -> 20, 2026 -> 10)
  RS_msg   = side of win_reason_code, mapped from win_reason_message TEXT (Terrorists_Win -> T, Bomb_Defused -> CT, ...)
  RS_d2    = side under D2's mapping (T = {1,9,13,17}, CT = {7,8,11,12,18})
  T_score  = side-attached winner from round_state freeze-end scores, team-tracked across the swap:
             team X = players on code 2 at first valid round of the match; side of X and Y per round from player_info;
             winner = team whose score rose by exactly 1 between freeze_end r and freeze_end r+1; its side at round r.
  B        = bomb side from bomb_state (bomb_defused -> 3 CT, bomb_exploded -> 2 T), any tick in the round.

Output: stdout only. Run:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_5_recompute.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
PC = {"2024": 20, "2026": 10}
MSG_SIDE = {  # side that WON the round, read from the message text (self-describing labels)
    "#SFUI_Notice_Target_Bombed": 2,
    "#SFUI_Notice_Terrorists_Win": 2,
    "#SFUI_Notice_Hostages_Not_Rescued": 2,
    "#SFUI_Notice_CTs_Surrender": 2,
    "#SFUI_Notice_Bomb_Defused": 3,
    "#SFUI_Notice_CTs_Win": 3,
    "#SFUI_Notice_All_Hostages_Rescued": 3,
    "#SFUI_Notice_Target_Saved": 3,
    "#SFUI_Notice_Terrorists_Surrender": 3,
}
D2_T = [1, 9, 13, 17]     # D2_final.py T_REASON
D2_CT = [7, 8, 11, 12, 18]  # D2_final.py CT_REASON
TICK_SLACK = 40           # bomb tick may exceed round_end tick by this much (same as D2_truth window note)


def files(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return sorted(glob.glob(fd + "/dataframe_*"))


def read(label, ch, cols, where=None):
    fs = files(label, ch)
    have = pl.read_parquet_schema(fs[0]).keys()
    use = [c for c in cols if c in have]
    lf = pl.scan_parquet(fs).select(use)
    if where is not None:
        lf = lf.filter(where)
    return lf.collect()


def round_table(label):
    re_ = read(label, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                    "win_reason_message", "player_count"])
    n_raw = re_.height
    re_ = re_.filter(pl.col("player_count") == PC[label])
    n_pc = re_.height
    re_ = re_.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
    print(f"[{label}] round_end raw rows {n_raw}; player_count=={PC[label]} rows {n_pc}; unique (match,round) {re_.height}")
    msg_df = pl.DataFrame({"win_reason_message": list(MSG_SIDE.keys()), "RS_msg": list(MSG_SIDE.values())})
    unmapped = re_.join(msg_df, on="win_reason_message", how="anti")
    print(f"[{label}] messages not in MSG_SIDE: {unmapped.height} rows; distinct: {unmapped['win_reason_message'].unique().to_list()}")
    re_ = re_.join(msg_df, on="win_reason_message", how="left")
    re_ = re_.with_columns(
        pl.when(pl.col("win_reason_code").is_in(D2_T)).then(2)
        .when(pl.col("win_reason_code").is_in(D2_CT)).then(3)
        .otherwise(None).cast(pl.Int64).alias("RS_d2"))
    return re_.rename({"winner_team_code": "R"})


def score_truth(label):
    pi = read(label, "player_info", ["match_key", "round", "player_id", "team_code"],
              where=pl.col("team_code").is_in([2, 3]))
    cnt = pi.group_by(["match_key", "round", "team_code"]).agg(pl.len().alias("n"))
    full = cnt.filter(pl.col("n") == 5)
    valid = full.group_by(["match_key", "round"]).agg(pl.len().alias("k")).filter(pl.col("k") == 2).select(["match_key", "round"])
    r0 = (valid.sort(["match_key", "round"]).unique("match_key", keep="first", maintain_order=True)
          .rename({"round": "r0"}))
    first = pi.join(r0, on="match_key").filter(pl.col("round") == pl.col("r0"))
    xset = (first.filter(pl.col("team_code") == 2).select(["match_key", "player_id"]).unique()
            .with_columns(pl.lit(True).alias("isX")))
    pi2 = pi.join(xset, on=["match_key", "player_id"], how="left").with_columns(pl.col("isX").fill_null(False))
    side = pi2.group_by(["match_key", "round", "isX"]).agg(
        pl.col("team_code").min().alias("side"), pl.col("team_code").max().alias("side_max"), pl.len().alias("n"))
    bad_side = side.filter(pl.col("side") != pl.col("side_max")).height
    bad_n = side.filter(pl.col("n") != 5).height
    print(f"[{label}] player_info side groups: {side.height}; groups with mixed codes: {bad_side}; groups with n!=5: {bad_n}")
    sx = side.filter(pl.col("isX")).select(["match_key", "round", pl.col("side").alias("sX")])
    sy = side.filter(~pl.col("isX")).select(["match_key", "round", pl.col("side").alias("sY")])

    rs = read(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
              where=pl.col("event_type") == "round_freeze_end")
    fe = (rs.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
          .select(["match_key", "round", "t_score", "ct_score"]))
    T = fe.join(sx, on=["match_key", "round"], how="inner").join(sy, on=["match_key", "round"], how="inner")
    T = T.with_columns(
        pl.when(pl.col("sX") == 2).then(pl.col("t_score")).otherwise(pl.col("ct_score")).alias("tX"),
        pl.when(pl.col("sY") == 2).then(pl.col("t_score")).otherwise(pl.col("ct_score")).alias("tY"))
    T = T.sort(["match_key", "round"]).with_columns(
        pl.col("round").shift(-1).over("match_key").alias("nr"),
        pl.col("tX").shift(-1).over("match_key").alias("ntX"),
        pl.col("tY").shift(-1).over("match_key").alias("ntY"))
    T = T.with_columns(
        pl.when(pl.col("nr") == pl.col("round") + 1).then(pl.col("ntX") - pl.col("tX")).otherwise(None).alias("dX"),
        pl.when(pl.col("nr") == pl.col("round") + 1).then(pl.col("ntY") - pl.col("tY")).otherwise(None).alias("dY"))
    T = T.with_columns(
        pl.when((pl.col("dX") == 1) & (pl.col("dY") == 0)).then(pl.lit("X"))
        .when((pl.col("dY") == 1) & (pl.col("dX") == 0)).then(pl.lit("Y"))
        .otherwise(None).alias("wteam"))
    T = T.with_columns(
        pl.when(pl.col("wteam") == "X").then(pl.col("sX"))
        .when(pl.col("wteam") == "Y").then(pl.col("sY"))
        .otherwise(None).cast(pl.Int64).alias("T_score"))
    print(f"[{label}] freeze_end rounds with team-tracked sides: {T.height}; rounds with a clean +1 winner: "
          f"{T.filter(pl.col('T_score').is_not_null()).height}; swap rounds (sX changes): "
          f"{T.sort(['match_key','round']).with_columns(pl.col('sX').shift(1).over('match_key').alias('psX')).filter(pl.col('sX') != pl.col('psX')).filter(pl.col('psX').is_not_null()).height}")
    return T.select(["match_key", "round", "sX", "sY", "T_score"])


def bomb_table(label):
    bs = read(label, "bomb_state", ["match_key", "round", "tick", "event_type"],
              where=pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
    bs = bs.with_columns(pl.when(pl.col("event_type") == "bomb_defused").then(3).otherwise(2).cast(pl.Int64).alias("bside"))
    g = bs.group_by(["match_key", "round"]).agg(
        pl.col("bside").n_unique().alias("nb_side"),
        pl.col("bside").min().alias("B"),
        pl.len().alias("n_bomb_ev"),
        pl.col("tick").min().alias("bomb_tick"))
    print(f"[{label}] bomb outcome rounds {g.height}; rounds with conflicting defused+exploded: {g.filter(pl.col('nb_side') > 1).height}")
    return g


def pct(a, b):
    return f"{a:,} of {b:,} ({(100.0 * a / b if b else float('nan')):.4f}%)"


def analyse(label):
    print(f"\n==================== {label} ====================")
    rt = round_table(label)
    tt = score_truth(label)
    bt = bomb_table(label)
    J = rt.join(tt, on=["match_key", "round"], how="left").join(bt, on=["match_key", "round"], how="left")
    J = J.with_columns(
        (pl.col("T_score") != pl.col("R")).alias("flip_T"),
        (pl.col("B") != pl.col("R")).alias("flip_B"),
    )
    N = J.height
    print(f"\n-- reason side vs R, ALL rounds (no truth needed): N={N}")
    print("   RS_msg == R:", pct(J.filter(pl.col("RS_msg") == pl.col("R")).height, N),
          "| RS_msg null:", J.filter(pl.col("RS_msg").is_null()).height)
    print("   RS_d2  == R:", pct(J.filter(pl.col("RS_d2") == pl.col("R")).height, N),
          "| RS_d2 null:", J.filter(pl.col("RS_d2").is_null()).height)
    print("   D2 mapping disagreements with R, by reason code (code: n_all, n_with_T):")
    dis = J.filter(pl.col("RS_d2") != pl.col("R"))
    print(dis.group_by("win_reason_code").agg(pl.len().alias("n_all"), pl.col("T_score").is_not_null().sum().alias("n_with_T"),
                                              pl.col("R").first().alias("R_value"), pl.col("win_reason_message").first().alias("msg"))
          .sort("win_reason_code"))

    print("\n-- determinism: distinct R per win_reason_code (1 everywhere => R is a function of the code)")
    det = J.group_by("win_reason_code").agg(pl.col("R").n_unique().alias("nR"), pl.len().alias("n")).sort("win_reason_code")
    print(det)

    JT = J.filter(pl.col("T_score").is_not_null())
    NT = JT.height
    print(f"\n-- with score-based truth T: N_T={NT}")
    print("   RS_msg == R:", pct(JT.filter(pl.col("RS_msg") == pl.col("R")).height, NT))
    print("   RS_d2  == R:", pct(JT.filter(pl.col("RS_d2") == pl.col("R")).height, NT))
    F = JT.filter(pl.col("flip_T"))
    print(f"   flipped rows (R != T): {F.height}")
    print("   flipped by round group:", F.with_columns(
        pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
        .when(pl.col("round") == 28).then(pl.lit("r28")).when(pl.col("round") <= 12).then(pl.lit("r01_12"))
        .when(pl.col("round") <= 24).then(pl.lit("r14_24")).otherwise(pl.lit("r29+")).alias("grp")
    ).group_by("grp").len().sort("grp").rows())
    print("   flipped: RS_msg == R:", pct(F.filter(pl.col("RS_msg") == pl.col("R")).height, F.height),
          "| RS_msg == T:", F.filter(pl.col("RS_msg") == pl.col("T_score")).height)
    print("   flipped: RS_d2  == R:", pct(F.filter(pl.col("RS_d2") == pl.col("R")).height, F.height),
          "| RS_d2 == T:", F.filter(pl.col("RS_d2") == pl.col("T_score")).height)
    print("   flipped rows, D2-mapping reason == T (listing):")
    print(F.filter(pl.col("RS_d2") == pl.col("T_score")).select(
        ["match_key", "round", "win_reason_code", "win_reason_message", "R", "T_score", "RS_d2", "RS_msg", "B"]))

    print("\n-- bomb outcome (independent of team identity) vs R and vs T_score")
    JB = J.filter(pl.col("B").is_not_null())
    print(f"   rounds with bomb outcome: {JB.height}; B == R: {JB.filter(pl.col('B') == pl.col('R')).height}")
    JBT = JB.filter(pl.col("T_score").is_not_null())
    print(f"   with T too: {JBT.height}; B == T: {JBT.filter(pl.col('B') == pl.col('T_score')).height}; "
          f"B != R (flip_B): {JBT.filter(pl.col('flip_B')).height}")
    print("   bomb-flipped rounds: RS_msg == B:", JB.filter(pl.col("flip_B") & (pl.col("RS_msg") == pl.col("B"))).height,
          "of", JB.filter(pl.col("flip_B")).height, "| RS_d2 == B:", JB.filter(pl.col("flip_B") & (pl.col("RS_d2") == pl.col("B"))).height)
    print("   bomb-flipped rounds by round group:", JB.filter(pl.col("flip_B")).group_by(
        pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
        .when(pl.col("round") == 28).then(pl.lit("r28")).otherwise(pl.lit("other")).alias("grp")).len().sort("grp").rows())

    print("\n-- what the reason code says on bomb-flipped rounds (crosstab of reason code, message, R, bomb)")
    print(JB.filter(pl.col("flip_B")).group_by(["win_reason_code", "win_reason_message", "R", "B"]).len()
          .sort(["win_reason_code", "R", "B"]))
    print("   2026 control: bomb rounds where B == R:", JB.filter(pl.col("B") == pl.col("R")).height, "of", JB.height,
          "| where RS_msg == B:", JB.filter(pl.col("RS_msg") == pl.col("B")).height)

    print("\n-- alternative explanation A: is each bomb event inside its own round window? (tick-based attribution)")
    rw = rt.sort(["match_key", "round"]).with_columns(
        pl.col("tick").shift(1).over("match_key").alias("prev_tick"),
        pl.col("round").shift(1).over("match_key").alias("prev_round"))
    bw = bt.join(rw.select(["match_key", "round", "tick", "prev_tick", "prev_round"]), on=["match_key", "round"], how="inner")
    inwin = bw.filter((pl.col("bomb_tick") > pl.col("prev_tick")) & (pl.col("bomb_tick") <= pl.col("tick") + TICK_SLACK))
    print(f"   bomb rounds with a round_end row: {bw.height}; bomb tick inside (prev round_end, round_end+{TICK_SLACK}]: {inwin.height}")
    fb = JB.filter(pl.col("flip_B")).select(["match_key", "round"])
    fbw = bw.join(fb, on=["match_key", "round"], how="inner")
    fin = fbw.filter((pl.col("bomb_tick") > pl.col("prev_tick")) & (pl.col("bomb_tick") <= pl.col("tick") + TICK_SLACK))
    print(f"   bomb-flipped rounds: {fbw.height}; inside window: {fin.height}")

    print("\n-- alternative explanation B: is the bomb outcome better explained by a neighbouring round's R (index shift)?")
    nb = (rt.sort(["match_key", "round"]).with_columns(
        pl.col("R").shift(1).over("match_key").alias("R_prev"),
        pl.col("R").shift(-1).over("match_key").alias("R_next"),
        pl.col("round").shift(1).over("match_key").alias("p_round"),
        pl.col("round").shift(-1).over("match_key").alias("n_round"))
        .select(["match_key", "round", "R_prev", "R_next", "p_round", "n_round"]))
    fbn = fb.join(nb, on=["match_key", "round"], how="left").join(JB.select(["match_key", "round", "B"]), on=["match_key", "round"], how="left")
    print(f"   bomb-flipped rounds: {fbn.height}; B == R(prev round): {fbn.filter(pl.col('B') == pl.col('R_prev')).height}; "
          f"B == R(next round): {fbn.filter(pl.col('B') == pl.col('R_next')).height}")

    print("\n-- bomb-only flips outside rounds 13/25/28 (truth from bomb only): listing")
    oth = JB.filter(pl.col("flip_B") & ~pl.col("round").is_in([13, 25, 28]))
    print(oth.select(["match_key", "round", "R", "B", "T_score", "win_reason_code", "win_reason_message", "bomb_tick", "tick"])
          .sort(["match_key", "round"]))

    print("\n-- neighbour base rate (is B == R of neighbour more often than chance, for ALL bomb rounds?)")
    allb = JB.join(nb, on=["match_key", "round"], how="left")
    print(f"   all bomb rounds: {allb.height}; B == R_prev: {allb.filter(pl.col('B') == pl.col('R_prev')).height}; "
          f"B == R_next: {allb.filter(pl.col('B') == pl.col('R_next')).height}")
    nf = allb.filter(~pl.col("flip_B"))
    print(f"   unflipped bomb rounds: {nf.height}; B == R_prev: {nf.filter(pl.col('B') == pl.col('R_prev')).height}; "
          f"B == R_next: {nf.filter(pl.col('B') == pl.col('R_next')).height}")

    print("\n-- D2 truth table cross-check (reads logs\\swarm2\\D2_truth_<label>.parquet, not used as an input)")
    p = ROOT / "logs" / "swarm2" / f"D2_truth_{label}.parquet"
    D = pl.read_parquet(p)
    keep = [c for c in ["match_key", "round", "T", "R", "flip", "win_reason_code", "s_reason", "mm_reason"] if c in D.columns]
    D = D.select(keep)
    jj = J.select(["match_key", "round", "R", "win_reason_code", "T_score", "RS_d2", "RS_msg"]).join(
        D, on=["match_key", "round"], how="inner", suffix="_D2")
    print(f"   matched rounds: {jj.height}; D2 columns used: {keep}")
    both = jj.filter(pl.col("T").is_not_null() & pl.col("T_score").is_not_null())
    print(f"   T agreement (both known): {both.filter(pl.col('T') == pl.col('T_score')).height} of {both.height}")
    print("   D2 T known, mine not:", jj.filter(pl.col("T").is_not_null() & pl.col("T_score").is_null()).height,
          "| mine known, D2 not:", jj.filter(pl.col("T").is_null() & pl.col("T_score").is_not_null()).height)
    print("   D2 R == my R:", jj.filter(pl.col("R") == pl.col("R_D2")).height, "of", jj.height)
    dt = jj.filter(pl.col("T").is_not_null())
    if "s_reason" in jj.columns:
        print(f"   D2 own table, D2 truth rows {dt.height}: s_reason == R_D2 {dt.filter(pl.col('s_reason') == pl.col('R_D2')).height}; "
              f"D2 mismatch rows (s_reason != R_D2): {dt.filter(pl.col('s_reason') != pl.col('R_D2')).height}")
        print("   D2 mismatch rows listing:")
        print(dt.filter(pl.col("s_reason") != pl.col("R_D2")).select(["match_key", "round", "R_D2", "win_reason_code", "s_reason", "T"]))
    dflip = jj.filter(pl.col("T").is_not_null() & (pl.col("R_D2") != pl.col("T")))
    print(f"   D2 flipped rows (R != T, D2 T): {dflip.height}")
    if "s_reason" in jj.columns:
        print(f"   D2 flipped rows with s_reason == D2 T: {dflip.filter(pl.col('s_reason') == pl.col('T')).height}")
        print("   D2 flipped rows, s_reason == D2 T, listing:")
        print(dflip.filter(pl.col("s_reason") == pl.col("T")).select(
            ["match_key", "round", "R", "T", "win_reason_code", "s_reason", "RS_d2", "RS_msg", "T_score"]))
    ov = jj.select(["match_key", "round", "R_D2", "T", "T_score"]).with_columns(
        (pl.col("T_score") != pl.col("R_D2")).alias("mine_flip"),
        (pl.col("T") != pl.col("R_D2")).alias("d2_flip"))
    print("\n   flip-set overlap (mine vs D2), over all matched rounds:")
    print(ov.group_by(["mine_flip", "d2_flip"]).len().sort(["mine_flip", "d2_flip"]))
    print("   mine flipped but D2 T null:", ov.filter(pl.col("mine_flip") & pl.col("T").is_null()).height,
          "| D2 flipped but my T null:", ov.filter(pl.col("d2_flip") & pl.col("T_score").is_null()).height)
    fx = ov.filter((pl.col("mine_flip") & ~pl.col("d2_flip")).fill_null(False))
    print("   rounds flipped in mine, not flipped in D2:", fx.height)
    if fx.height:
        print(fx.select(["match_key", "round", "R_D2", "T", "T_score"]).head(20))
    dx = ov.filter((~pl.col("mine_flip") & pl.col("d2_flip")).fill_null(False))
    print("   rounds flipped in D2, not flipped in mine (T_score == R):", dx.height)
    if dx.height:
        print(dx.join(J.select(["match_key", "round", "B", "win_reason_message"]), on=["match_key", "round"], how="left")
              .select(["match_key", "round", "R_D2", "T", "T_score", "B", "win_reason_message"]))

    print("\n-- bomb check on MY score-based flips (bomb is physical, independent of team identity and of R)")
    mf = J.filter(pl.col("flip_T"))
    mfb = mf.filter(pl.col("B").is_not_null())
    print(f"   my flipped rows: {mf.height}; with bomb outcome: {mfb.height}; B == T_score: {mfb.filter(pl.col('B') == pl.col('T_score')).height}; "
          f"B == R (bomb contradicts my flip): {mfb.filter(pl.col('B') == pl.col('R')).height}")
    print("   my flipped rows WITHOUT bomb outcome (rely on score alone):", mf.filter(pl.col("B").is_null()).height)
    nb_ok = J.filter(pl.col("T_score").is_not_null() & ~pl.col("flip_T") & pl.col("B").is_not_null())
    print(f"   my NOT-flipped rows with bomb outcome: {nb_ok.height}; B == T_score: {nb_ok.filter(pl.col('B') == pl.col('T_score')).height}")
    if mfb.filter(pl.col("B") == pl.col("R")).height:
        print("   my flips contradicted by bomb (listing):")
        print(mfb.filter(pl.col("B") == pl.col("R")).select(["match_key", "round", "R", "T_score", "B", "win_reason_message"]))
    print("   my flipped rows, by round, with R, T, bomb (2026 only: listing):")
    if label == "2026":
        print(mf.select(["match_key", "round", "R", "T_score", "B", "win_reason_message"]).sort(["match_key", "round"]))
    return J


if __name__ == "__main__":
    print("polars", pl.__version__)
    J24 = analyse("2024")
    J26 = analyse("2026")
