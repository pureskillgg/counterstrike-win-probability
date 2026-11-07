"""Verifier for D2-4 (independent recompute). Written from the raw tome tables, not from D2_truth/D2_final outputs.

Claimed by D2 (to verify):
  2024 in-window bomb rows with T: 14,525; bomb==T 14,525; bomb==R 14,032; 493 flipped bomb rows, all bomb==T.
  2024 flipped rows with elimination: 1,914; elim==T 1,891; elim==R 23.
  2024 s_score vs s_bomb: 12,920 of 12,920 equal. 2026: 6,512 of 6,512; 2026 bomb==T 6,478 of 6,478.

Definitions (independent code, same intent as D2):
  R        = round_end.winner_team_code (first row per match,round by tick)
  t_fe     = round_freeze_end tick (first per match,round); scores at freeze end
  t_end    = round_end tick (first per match,round)
  sx       = side (2=T, 3=CT) of team X in round r. X = players on code 2 at the first valid 5v5 round.
  s_score  = team-tracked freeze-end score delta from round r to r+1 (side-read scores, as D2 assumes)
  s_bomb   = first bomb_exploded (->2) or bomb_defused (->3) with tick in [t_fe, t_end + W], W = 40 ASSUMPTION
  s_elim   = side with five distinct dead members of the 10-player roster (ct_dead==5 -> 2, t_dead==5 -> 3)
  T        = coalesce(s_score, s_bomb, s_elim)
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D2_4_check.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(25)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
PC = {"2024": 20, "2026": 10}   # full 5v5 round marker in round_end.player_count (matches D2; ASSUMPTION)
W_MAIN = 40                     # ticks after round_end allowed for bomb outcome (ASSUMPTION, as D2)


def rd(label, ch, cols, where=None):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    lf = pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*")))
    if where is not None:
        lf = lf.filter(where)
    return lf.select(cols).collect()


def bomb_side(label, base, W):
    """First outcome event in [t_fe, t_end + W] per match,round. Returns s_bomb, n_out, n_types, off_first."""
    bs = rd(label, "bomb_state", ["match_key", "round", "tick", "event_type"],
            pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
    bw = bs.join(base.select(["match_key", "round", "t_fe", "t_end"]), on=["match_key", "round"], how="inner")
    bw = bw.filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + W))
    agg = (bw.sort("tick").group_by(["match_key", "round"]).agg(
        pl.col("event_type").first().alias("bo_first"),
        pl.len().alias("n_out"),
        pl.col("event_type").n_unique().alias("n_types"),
        (pl.col("tick").first() - pl.col("t_end").first()).alias("off_first")))
    agg = agg.with_columns(
        pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
        .cast(pl.Int64).alias("s_bomb"))
    return agg.select(["match_key", "round", "s_bomb", "n_out", "n_types", "off_first"])


def build(label):
    re_ = rd(label, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    base = (re_.sort("tick").unique(["match_key", "round"], keep="first")
            .select(["match_key", "round", pl.col("tick").alias("t_end"),
                     pl.col("winner_team_code").alias("R"), "player_count"]))
    rs = rd(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
            pl.col("event_type") == "round_freeze_end")
    fe = (rs.sort("tick").unique(["match_key", "round"], keep="first")
          .select(["match_key", "round", pl.col("tick").alias("t_fe"), "t_score", "ct_score"]))
    base = base.join(fe, on=["match_key", "round"], how="left")

    # roster and side of team X
    pi = (rd(label, "player_info", ["match_key", "round", "player_id", "team_code"])
          .filter(pl.col("team_code").is_in([2, 3]))
          .unique(["match_key", "round", "player_id"], keep="first"))
    cnt = pi.group_by(["match_key", "round"]).agg(
        pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("n2"), (pl.col("team_code") == 3).sum().alias("n3"))
    valid = cnt.filter((pl.col("np") == 10) & (pl.col("n2") == 5) & (pl.col("n3") == 5))
    Fm = valid.group_by("match_key").agg(pl.col("round").min().alias("F"))
    mem = (pi.join(Fm, on="match_key", how="inner").filter(pl.col("round") == pl.col("F"))
           .select(["match_key", "player_id", (pl.col("team_code") == 2).alias("isX")])
           .unique(["match_key", "player_id"]))
    pim = pi.join(mem, on=["match_key", "player_id"], how="inner")
    sv = pim.group_by(["match_key", "round"]).agg(
        ((pl.col("isX") & (pl.col("team_code") == 2)) | (~pl.col("isX") & (pl.col("team_code") == 3))).sum().alias("v2"),
        ((pl.col("isX") & (pl.col("team_code") == 3)) | (~pl.col("isX") & (pl.col("team_code") == 2))).sum().alias("v3"),
        pl.len().alias("np_m"))
    sv = sv.with_columns(
        pl.when((pl.col("v2") > 0) & (pl.col("v3") == 0)).then(pl.lit(2))
        .when((pl.col("v3") > 0) & (pl.col("v2") == 0)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("sx"))
    sxr = sv.select(["match_key", "round", "sx", "v2", "v3"])
    base = base.join(sxr, on=["match_key", "round"], how="left")
    nx = base.select(["match_key", (pl.col("round") - 1).alias("round"),
                      pl.col("t_score").alias("tn"), pl.col("ct_score").alias("ctn")])
    nxs = sxr.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("sx").alias("sx_n")])
    pvs = sxr.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("sx").alias("sx_p")])
    base = (base.join(nx, on=["match_key", "round"], how="left")
            .join(nxs, on=["match_key", "round"], how="left")
            .join(pvs, on=["match_key", "round"], how="left"))

    # side-read score (D2 assumption), team-tracked across the swap
    base = base.with_columns(
        pl.when(pl.col("sx") == 2).then(pl.col("t_score")).when(pl.col("sx") == 3).then(pl.col("ct_score"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("xs0"),
        pl.when(pl.col("sx") == 3).then(pl.col("t_score")).when(pl.col("sx") == 2).then(pl.col("ct_score"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("ys0"),
        pl.when(pl.col("sx_n") == 2).then(pl.col("tn")).when(pl.col("sx_n") == 3).then(pl.col("ctn"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("xs1"),
        pl.when(pl.col("sx_n") == 3).then(pl.col("tn")).when(pl.col("sx_n") == 2).then(pl.col("ctn"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("ys1"))
    base = base.with_columns(
        (pl.col("xs1") - pl.col("xs0")).alias("dX"), (pl.col("ys1") - pl.col("ys0")).alias("dY"))
    base = base.with_columns(
        pl.when((pl.col("dX") == 1) & (pl.col("dY") == 0)).then(pl.col("sx"))
        .when((pl.col("dY") == 1) & (pl.col("dX") == 0)).then(5 - pl.col("sx"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_score"),
        pl.when(((pl.col("dX") == 1) & (pl.col("dY") == 0)) | ((pl.col("dX") == 0) & (pl.col("dY") == 1)))
        .then(pl.lit(1)).otherwise(pl.lit(0)).alias("clean"))

    # identity-fixed alternative: t_score always belongs to team X, ct_score always to team Y
    base = base.with_columns(
        (pl.col("tn") - pl.col("t_score")).alias("dXa"), (pl.col("ctn") - pl.col("ct_score")).alias("dYa"))
    base = base.with_columns(
        pl.when((pl.col("dXa") == 1) & (pl.col("dYa") == 0)).then(pl.col("sx"))
        .when((pl.col("dYa") == 1) & (pl.col("dXa") == 0)).then(5 - pl.col("sx"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_score_idfix"),
        pl.when(((pl.col("dXa") == 1) & (pl.col("dYa") == 0)) | ((pl.col("dXa") == 0) & (pl.col("dYa") == 1)))
        .then(pl.lit(1)).otherwise(pl.lit(0)).alias("clean_idfix"))

    # elimination (identity membership at F; victim side at round r = sx or 5-sx)
    dth = (rd(label, "player_death", ["match_key", "round", "player_id"])
           .unique(["match_key", "round", "player_id"]))
    dth = dth.join(mem, on=["match_key", "player_id"], how="inner")
    dth = (dth.join(sxr.select(["match_key", "round", "sx"]), on=["match_key", "round"], how="inner")
           .filter(pl.col("sx").is_not_null()))
    dth = dth.with_columns(pl.when(pl.col("isX")).then(pl.col("sx")).otherwise(5 - pl.col("sx")).alias("vside"))
    el = dth.group_by(["match_key", "round"]).agg(
        (pl.col("vside") == 2).sum().alias("t_dead"), (pl.col("vside") == 3).sum().alias("ct_dead"))
    el = el.with_columns(
        pl.when((pl.col("ct_dead") == 5) & (pl.col("t_dead") < 5)).then(pl.lit(2))
        .when((pl.col("t_dead") == 5) & (pl.col("ct_dead") < 5)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("s_elim"))
    base = base.join(el.select(["match_key", "round", "t_dead", "ct_dead", "s_elim"]),
                     on=["match_key", "round"], how="left")

    # bomb outcome (main window)
    bm = bomb_side(label, base, W_MAIN)
    base = base.join(bm, on=["match_key", "round"], how="left")
    base = base.with_columns(
        pl.coalesce(["s_score", "s_bomb", "s_elim"]).alias("T"),
        pl.when(pl.col("s_score").is_not_null()).then(pl.lit("score"))
        .when(pl.col("s_bomb").is_not_null()).then(pl.lit("bomb"))
        .when(pl.col("s_elim").is_not_null()).then(pl.lit("elim"))
        .otherwise(pl.lit("none")).alias("T_src"),
        pl.col("match_key").str.extract(r"/(\d{2})/[^/]+$", 1).alias("day"),
        pl.lit(label).alias("year"))
    base = base.with_columns(
        (pl.col("R") != pl.col("T")).alias("flip"),
        (pl.col("R") != pl.col("s_score")).alias("flip_score"))
    return base


def main():
    out = {}
    for label in ["2024", "2026"]:
        out[label] = build(label)
        print(f"built {label}: full rounds {out[label].height}")
    for label in ["2024", "2026"]:
        full = out[label]
        pop = full.filter(pl.col("player_count") == PC[label])
        print(f"\n=========== {label}: population rows {pop.height}; with sx {pop.filter(pl.col('sx').is_not_null()).height}; "
              f"T known {pop.filter(pl.col('T').is_not_null()).height} "
              f"({pop.filter(pl.col('T').is_not_null()).height / pop.height:.2%}); "
              f"T_src {pop.group_by('T_src').len().sort('T_src').rows()}")

        # ---- A. bomb claim
        a1 = pop.filter(pl.col("s_bomb").is_not_null() & pl.col("T").is_not_null())
        a_f = a1.filter(pl.col("s_bomb") != pl.col("R"))
        print(f"[A] bomb rows with T known: {a1.height}; bomb==T {a1.filter(pl.col('s_bomb') == pl.col('T')).height}; "
              f"bomb==R {a1.filter(pl.col('s_bomb') == pl.col('R')).height}; bomb!=R (flipped by bomb) {a_f.height}")
        print(f"    flipped-by-bomb rows by T_src: {a_f.group_by('T_src').len().sort('T_src').rows()}")
        print(f"    flipped-by-bomb rows with s_score known: {a_f.filter(pl.col('s_score').is_not_null()).height}; "
              f"of which s_score==s_bomb: {a_f.filter(pl.col('s_score').is_not_null() & (pl.col('s_score') == pl.col('s_bomb'))).height}")
        ns = pop.filter(pl.col("s_score").is_not_null() & pl.col("s_bomb").is_not_null())
        print(f"[A2] NON-CIRCULAR: rows with s_score AND s_bomb: {ns.height}; equal {ns.filter(pl.col('s_score') == pl.col('s_bomb')).height}; "
              f"differ {ns.filter(pl.col('s_score') != pl.col('s_bomb')).height}")
        print(f"     of which R != s_score (flipped by score): {ns.filter(pl.col('flip_score')).height}; "
              f"of those s_bomb==s_score: {ns.filter(pl.col('flip_score') & (pl.col('s_bomb') == pl.col('s_score'))).height}")

        # ---- B. elimination claim
        b_f = pop.filter(pl.col("flip_score") & pl.col("s_elim").is_not_null() & pl.col("s_score").is_not_null())
        b1 = pop.filter(pl.col("s_elim").is_not_null() & pl.col("s_score").is_not_null())
        print(f"[B] elim & score known: {b1.height}; elim==score {b1.filter(pl.col('s_elim') == pl.col('s_score')).height}")
        print(f"    flipped-by-score rows with elim: {b_f.height}; elim==score {b_f.filter(pl.col('s_elim') == pl.col('s_score')).height}; "
              f"elim==R {b_f.filter(pl.col('s_elim') == pl.col('R')).height}")
        bad = b_f.filter(pl.col("s_elim") != pl.col("s_score"))
        if bad.height:
            print("    elim!=score flipped rows (match suffix, round, R, s_score, s_bomb, n_out, t_dead, ct_dead):")
            for r in bad.select(["match_key", "round", "R", "s_score", "s_bomb", "n_out", "t_dead", "ct_dead"]).iter_rows():
                print("     ", r[0][-24:], r[1:])
        bfe = pop.filter(pl.col("flip_score") & pl.col("s_elim").is_not_null() & pl.col("s_bomb").is_not_null())
        print(f"    flipped-by-score rows with elim AND bomb: {bfe.height}; elim==bomb {bfe.filter(pl.col('s_elim') == pl.col('s_bomb')).height}")

        # ---- C. R vs score-based truth, by round group and day
        c1 = pop.filter(pl.col("s_score").is_not_null())
        print(f"[C] rows with s_score: {c1.height}; R==s_score {c1.filter(~pl.col('flip_score')).height}; flipped {c1.filter(pl.col('flip_score')).height}")
        c_grp = c1.with_columns(
            pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
            .when(pl.col("round") == 28).then(pl.lit("r28")).when(pl.col("round") <= 12).then(pl.lit("r01_12"))
            .otherwise(pl.lit("r14_24_29p")).alias("grp"))
        print(c_grp.group_by("grp").agg(pl.len().alias("n"), pl.col("flip_score").sum().alias("flipped")).sort("grp"))
        r13 = c1.filter(pl.col("round") == 13)
        print("    round 13 flipped-by-score by day [day, n, flipped]:",
              r13.group_by("day").agg(pl.len().alias("n"), pl.col("flip_score").sum().alias("fl")).sort("day").rows())
        print("    flipped rows in pop: all", pop.filter(pl.col("flip")).height,
              " with score unknown (T from bomb/elim only):", pop.filter(pl.col("flip") & pl.col("s_score").is_null()).height)

        # ---- D. clean delta test: exactly one team rises by 1 (side-read vs identity-fixed), by swap
        dd = pop.filter(pl.col("sx").is_not_null() & pl.col("sx_n").is_not_null() & pl.col("xs1").is_not_null())
        sw = dd.with_columns((pl.col("sx") != pl.col("sx_n")).alias("swap"))
        print("[D] clean rate (exactly one team +1):")
        print(sw.group_by("swap").agg(pl.len().alias("n"), pl.col("clean").mean().round(5).alias("side_read_clean"),
                                      pl.col("clean_idfix").mean().round(5).alias("idfix_clean")).sort("swap"))
        sw_b = sw.filter(pl.col("swap") & pl.col("s_bomb").is_not_null())
        print(f"    SWAP rounds with bomb: {sw_b.height}; side-read==bomb {sw_b.filter(pl.col('s_score') == pl.col('s_bomb')).height}; "
              f"idfix==bomb {sw_b.filter(pl.col('s_score_idfix') == pl.col('s_bomb')).height}; R==bomb {sw_b.filter(pl.col('R') == pl.col('s_bomb')).height}")

        # ---- E. monotonicity of freeze-end scores (identity-fixed scores would never fall)
        mono = pop.filter(pl.col("tn").is_not_null() & pl.col("t_score").is_not_null())
        dec = mono.filter((pl.col("tn") < pl.col("t_score")) | (pl.col("ctn") < pl.col("ct_score")))
        known_sx = mono.filter(pl.col("sx").is_not_null() & pl.col("sx_n").is_not_null())
        sx_chg = known_sx.filter(pl.col("sx") != pl.col("sx_n"))
        dec_sw = dec.filter(pl.col("sx").is_not_null() & pl.col("sx_n").is_not_null() & (pl.col("sx") != pl.col("sx_n")))
        print(f"[E] consecutive freeze-end pairs: {mono.height}; pairs where a score FELL: {dec.height}; "
              f"of which sx changed: {dec_sw.height}; pairs with sx change: {sx_chg.height}")

        # ---- F. neighbour test: is R equal to the neighbouring round's truth?
        fl = pop.filter(pl.col("flip_score") & pl.col("s_score").is_not_null()).select(["match_key", "round", "R", "s_score"])
        tprev = full.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("T").alias("T_prev")])
        tnext = full.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("T").alias("T_next")])
        fl2 = fl.join(tprev, on=["match_key", "round"], how="left").join(tnext, on=["match_key", "round"], how="left")
        print(f"[F] flipped-by-score rows: {fl2.height}; R==T(r-1) {fl2.filter(pl.col('R') == pl.col('T_prev')).height}; "
              f"R==T(r+1) {fl2.filter(pl.col('R') == pl.col('T_next')).height}; R==5-score {fl2.filter(pl.col('R') == 5 - pl.col('s_score')).height}")

        # ---- G. match-level pattern on swap rounds with score truth
        sr = pop.filter(pl.col("sx").is_not_null() & pl.col("sx_p").is_not_null()
                        & (pl.col("sx") != pl.col("sx_p")) & pl.col("s_score").is_not_null())
        pm = sr.group_by("match_key").agg(pl.len().alias("k"), pl.col("flip_score").sum().alias("f"))
        print(f"[G] matches with swap rounds and score truth: {pm.height}; all swaps flipped "
              f"{pm.filter((pl.col('f') == pl.col('k')) & (pl.col('k') > 0)).height}; no swap flipped "
              f"{pm.filter(pl.col('f') == 0).height}; mixed {pm.filter((pl.col('f') > 0) & (pl.col('f') < pl.col('k'))).height}")

        # ---- H. bomb window sensitivity (rows with score truth and a bomb event)
        for W in [0, 40, 200, 1000]:
            bw = bomb_side(label, full, W).select(["match_key", "round", pl.col("s_bomb").alias("sb_w")])
            tmp = pop.join(bw, on=["match_key", "round"], how="left").filter(
                pl.col("s_score").is_not_null() & pl.col("sb_w").is_not_null())
            print(f"[H] window end +{W:>4}: rows with score & bomb {tmp.height}; score==bomb {tmp.filter(pl.col('s_score') == pl.col('sb_w')).height}; "
                  f"bomb==R {tmp.filter(pl.col('sb_w') == pl.col('R')).height}")

        # ---- I. bomb first-event offset from round_end tick (ticks)
        bm = bomb_side(label, full, W_MAIN)
        bm2 = (bm.join(full.select(["match_key", "round", "day"]), on=["match_key", "round"], how="left")
               .with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("g")))
        print("[I] bomb first-event offset from round_end tick (ticks):")
        print(bm2.group_by("g").agg(pl.len().alias("n"),
                                    pl.col("off_first").quantile(0.05).alias("q05"),
                                    pl.col("off_first").median().alias("med"),
                                    pl.col("off_first").quantile(0.95).alias("q95"),
                                    pl.col("n_types").max().alias("max_types")).sort("g"))


if __name__ == "__main__":
    main()
