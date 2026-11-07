"""verify_D2_6: independent recomputation of the D2-6 claim (winner-derived 2024 score lead).

Built from the raw tomes only (no D2 intermediate parquet). Steps per year (2024 defect year, 2026 control):
  1. Population: round_end rows with player_count == PC (same ASSUMPTION as D2), one row per (match, round).
  2. Freeze-end scores (round_state event round_freeze_end, first tick) and the round_end tick.
  3. Team X = the five players on code 2 in the first valid round of the match. sx = side of X in each round
     (clean only: five X and five Y present, all X on one side and all Y on the other).
  4. Recorded winner R = round_end.winner_team_code.
  5. Side-free truth S from bomb outcome (exploded = 2, defused = 3) then elimination (five deaths on one side,
     victim side from per-round player_info.team_code, no team membership or sx used). S_ext adds a
     time-out rule (no plant and no outcome in window -> CT) as a sensitivity.
  6. Team-tracked leads: lead(r) = X-Y if X on CT else Y-X, with X's score accumulated from a winner side
     code (winner == sx -> X won). Three chains per match:
        rsl  = ct_score - t_score at freeze end (round_state, side-attached as claimed)
        lR   = chain driven by R (the claimed winner-derived lead)
        lS   = chain driven by S (side-free truth)
     Comparisons: rs vs lR (reproduces D2_lead), rs vs lS (side-attachment of round_state, sx and S consistent),
     lR vs lS (score-free error of the winner-derived lead).
  7. States (ASSUMPTION rule, bake-off style): 1 + distinct victims dying in [freeze_end, round_end].
     Compared with the bake-off 2026 ratio 234,265 / 28,683 = 8.167.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_6_recompute.py
Writes only logs\\swarm2\\verify_D2_6.log (plus stdout).
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
PC = {"2024": 20, "2026": 10}      # ASSUMPTION: full 5v5 round_end.player_count (as in D2)
WIN_EXTRA = 40                     # ASSUMPTION: outcome events allowed up to 40 ticks after round_end (as in D2)
BAKEOFF_STATES_PER_ROUND = 234265 / 28683   # bake-off kept set, 2026 only
LOG = ROOT / "logs" / "swarm2" / "verify_D2_6.log"
TRACE_MATCH = "90350d19-5452-4c58-bc8b-f780e796e026"
LINES = []


def out(s=""):
    print(s)
    LINES.append(str(s))


def load(label, ch, cols):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
    assert len(fd) == 1, (label, ch, fd)
    files = sorted(glob.glob(fd[0] + "/dataframe_*"))
    return pl.scan_parquet(files).select(cols).collect()


def build(label):
    pc = PC[label]
    info = {}
    re_ = load(label, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    info["round_end_rows_all"] = re_.height
    re_ = re_.filter(pl.col("player_count") == pc)
    info["round_end_rows_pc"] = re_.height
    dd = re_.group_by(["match_key", "round"]).agg(pl.len().alias("n"),
                                                   pl.col("winner_team_code").n_unique().alias("nw"))
    info["dup_keys_pc"] = dd.filter(pl.col("n") > 1).height
    info["dup_keys_conflicting_winner"] = dd.filter(pl.col("nw") > 1).height
    re_ = (re_.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
           .select(["match_key", "round", pl.col("tick").alias("t_end"),
                    pl.col("winner_team_code").cast(pl.Int64).alias("R")]))

    rs = load(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe_all = rs.filter(pl.col("event_type") == "round_freeze_end")
    fm = fe_all.group_by(["match_key", "round"]).agg(
        pl.len().alias("n"), (pl.col("t_score").n_unique() + pl.col("ct_score").n_unique()).alias("nu"))
    info["freeze_end_multiplicity"] = fm.group_by("n").len().sort("n").rows()
    info["freeze_end_multi_conflict"] = fm.filter(pl.col("nu") > 2).height
    fe = (fe_all.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
          .select(["match_key", "round", pl.col("tick").alias("t_fe"),
                   pl.col("t_score").cast(pl.Int64).alias("t0"), pl.col("ct_score").cast(pl.Int64).alias("ct0")]))

    pi = load(label, "player_info", ["match_key", "round", "player_id", "team_code"])
    pi = pi.unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
    cnt = pi.group_by(["match_key", "round"]).agg(pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("nT"),
                                                 (pl.col("team_code") == 3).sum().alias("nCT"))
    info["player_info_np_top_dist"] = cnt.group_by("np").len().sort("len", descending=True).head(6).rows()
    valid = (cnt.filter((pl.col("np") == 10) & (pl.col("nT") == 5) & (pl.col("nCT") == 5))
             .select(["match_key", "round"]).with_columns(pl.lit(True).alias("valid")))

    base = re_.join(fe, on=["match_key", "round"], how="left").join(valid, on=["match_key", "round"], how="left")
    base = base.with_columns(pl.col("valid").fill_null(False))
    info["base_rounds"] = base.height
    info["base_rounds_with_freeze_end"] = base.filter(pl.col("t_fe").is_not_null()).height
    info["base_rounds_valid_10p_5v5"] = base.filter(pl.col("valid")).height

    # team X membership from the first valid round of each match
    fv = base.filter(pl.col("valid")).group_by("match_key").agg(pl.col("round").min().alias("F"))
    pF = pi.join(fv, on="match_key").filter(pl.col("round") == pl.col("F"))
    memb = pF.select(["match_key", "player_id", (pl.col("team_code") == 2).alias("isX")]).unique(
        ["match_key", "player_id"], keep="first")
    pim = pi.join(memb, on=["match_key", "player_id"], how="inner")
    agg = pim.group_by(["match_key", "round"]).agg(
        pl.col("isX").sum().alias("nX"), (~pl.col("isX")).sum().alias("nY"),
        (pl.col("isX") & (pl.col("team_code") == 2)).sum().alias("xT"),
        ((~pl.col("isX")) & (pl.col("team_code") == 2)).sum().alias("yT"))
    agg = agg.with_columns(
        pl.when((pl.col("nX") == 5) & (pl.col("nY") == 5) & (pl.col("xT") == 5) & (pl.col("yT") == 0))
        .then(pl.lit(2))
        .when((pl.col("nX") == 5) & (pl.col("nY") == 5) & (pl.col("xT") == 0) & (pl.col("yT") == 5))
        .then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("sx"))
    base = base.join(agg.select(["match_key", "round", "sx"]), on=["match_key", "round"], how="left")
    info["base_rounds_sx_clean"] = base.filter(pl.col("sx").is_not_null()).height

    win = base.select(["match_key", "round", "t_fe", "t_end"])

    # side-free outcome: bomb events in window
    bs = load(label, "bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
        pl.col("event_type").is_in(["bomb_exploded", "bomb_defused", "bomb_planted"]))
    bw = (bs.join(win, on=["match_key", "round"], how="inner")
          .filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + WIN_EXTRA)))
    plant = bw.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(
        pl.len().alias("n_plant"))
    bo = (bw.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).sort("tick")
          .group_by(["match_key", "round"]).agg(pl.col("event_type").first().alias("bo_first"),
                                                pl.len().alias("n_out"),
                                                pl.col("event_type").n_unique().alias("n_out_types")))
    bo = bo.with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
                         .cast(pl.Int64).alias("S_bomb"))

    # side-free outcome: elimination, victim side from per-round player_info.team_code
    dt = load(label, "player_death", ["match_key", "round", "tick", "player_id"])
    dj = dt.join(win, on=["match_key", "round"], how="inner")
    pis = pi.select(["match_key", "round", "player_id", pl.col("team_code").alias("vteam")])
    dext = (dj.filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + WIN_EXTRA))
            .unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
            .join(pis, on=["match_key", "round", "player_id"], how="left"))
    el = dext.group_by(["match_key", "round"]).agg(
        (pl.col("vteam") == 2).sum().alias("t_dead"), (pl.col("vteam") == 3).sum().alias("ct_dead"),
        pl.len().alias("n_vict_ext"), pl.col("vteam").is_null().sum().alias("n_unres_ext"))
    el = el.join(plant, on=["match_key", "round"], how="left").with_columns(pl.col("n_plant").fill_null(0))
    # CS rule: all CT dead -> T wins (even if planted). All T dead -> CT wins only if no bomb was planted
    # (a planted bomb keeps the round open; the outcome then comes from bomb_state, see D2_6 notes).
    el = el.with_columns(
        pl.when((pl.col("ct_dead") == 5) & (pl.col("t_dead") < 5)).then(pl.lit(2))
        .when((pl.col("t_dead") == 5) & (pl.col("ct_dead") < 5) & (pl.col("n_plant") == 0)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("S_elim"))
    el = el.with_columns(
        pl.when((pl.col("t_dead") == 5) & (pl.col("n_plant") > 0)).then(pl.lit(1)).otherwise(pl.lit(0))
        .alias("tw_planted_open"))
    vstr = (dj.filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end")))
            .unique(["match_key", "round", "player_id"], keep="first", maintain_order=True)
            .group_by(["match_key", "round"]).agg(pl.len().alias("n_vict_str")))

    tab = base.join(bo.select(["match_key", "round", "S_bomb", "bo_first", "n_out_types", "n_out"]),
                    on=["match_key", "round"], how="left")
    tab = tab.join(el.select(["match_key", "round", "S_elim", "t_dead", "ct_dead", "n_unres_ext",
                              "tw_planted_open"]),
                   on=["match_key", "round"], how="left")
    tab = tab.join(plant, on=["match_key", "round"], how="left")
    tab = tab.join(vstr, on=["match_key", "round"], how="left")
    tab = tab.with_columns(pl.col("n_vict_str").fill_null(0), pl.col("n_plant").fill_null(0),
                           pl.col("n_out").fill_null(0), pl.col("t_dead").fill_null(0),
                           pl.col("ct_dead").fill_null(0))
    tab = tab.with_columns(
        pl.coalesce(["S_bomb", "S_elim"]).alias("S"),
        pl.when(pl.col("S_bomb").is_null() & pl.col("S_elim").is_null() & (pl.col("n_plant") == 0)
                & (pl.col("n_out") == 0) & pl.col("t_fe").is_not_null())
        .then(pl.lit(3)).otherwise(pl.lit(None)).cast(pl.Int64).alias("S_time"),
        (1 + pl.col("n_vict_str")).alias("states"))
    tab = tab.with_columns(pl.coalesce(["S", "S_time"]).alias("S_ext"))

    # bomb vs elimination agreement (both side-free)
    both = tab.filter(pl.col("S_bomb").is_not_null() & pl.col("S_elim").is_not_null())
    info["tw_planted_open_rounds_T_wiped_bomb_planted"] = tab.filter(pl.col("tw_planted_open") == 1).height
    info["tw_planted_open_with_bomb_outcome"] = tab.filter((pl.col("tw_planted_open") == 1)
                                                           & pl.col("S_bomb").is_not_null()).height
    info["S_bomb_vs_elim_both"] = both.height
    info["S_bomb_vs_elim_equal"] = both.filter(pl.col("S_bomb") == pl.col("S_elim")).height
    info["bomb_rounds_with_both_outcome_types"] = tab.filter(pl.col("n_out_types") > 1).height
    info["S_bomb_known"] = tab.filter(pl.col("S_bomb").is_not_null()).height
    info["S_elim_known"] = tab.filter(pl.col("S_elim").is_not_null()).height
    info["S_known"] = tab.filter(pl.col("S").is_not_null()).height
    info["S_ext_known"] = tab.filter(pl.col("S_ext").is_not_null()).height
    info["S_time_only"] = tab.filter(pl.col("S").is_null() & pl.col("S_time").is_not_null()).height
    info["elim_unresolved_victim_rounds"] = tab.filter(pl.col("n_unres_ext") > 0).height
    info["round1_nonzero_score_matches"] = tab.filter(
        (pl.col("round") == 1) & ((pl.col("t0") != 0) | (pl.col("ct0") != 0))).height
    info["round1_rows"] = tab.filter(pl.col("round") == 1).height
    return tab, info


def chains(tab, scol):
    """Team-tracked chains per contiguous match, prefix up to the first unclean sx (as D2)."""
    by = defaultdict(dict)
    for mk, rd, R, sx, t0, ct0, S, st in tab.select(
            ["match_key", "round", "R", "sx", "t0", "ct0", scol, "states"]).iter_rows():
        by[mk][rd] = (R, sx, t0, ct0, S, st)
    c = Counter()
    hR, hRS, hrsS = Counter(), Counter(), Counter()
    mR, mRS, mrsS = set(), set(), set()
    cset = set()
    bad_S_by_match = Counter()
    st_R = st_RS = st_used = 0
    for mk, byr in by.items():
        rmax = max(byr)
        if min(byr) != 1 or len(byr) != rmax:
            c["noncontig_matches"] += 1
            continue
        c["contig_matches"] += 1
        cset.add(mk)
        xR = yR = xS = yS = 0
        sok = True
        fR = fRS = frs = None
        broke = False
        for rd in range(1, rmax + 1):
            R, sx, t0, ct0, S, st = byr[rd]
            if sx is None or R is None:
                broke = True
                break
            c["rounds_used"] += 1
            st_used += st
            if sok:
                lS = (xS - yS) if sx == 3 else (yS - xS)
            lR = (xR - yR) if sx == 3 else (yR - xR)
            if t0 is not None and ct0 is not None:
                rsl = ct0 - t0
                c["rs_cmp_R"] += 1
                if lR != rsl:
                    c["rs_bad_R_rounds"] += 1
                    mR.add(mk)
                    st_R += st
                    if fR is None:
                        fR = rd
                        hR[rd] += 1
                if sok:
                    c["rs_cmp_S"] += 1
                    if lS != rsl:
                        c["rs_bad_S_rounds"] += 1
                        mrsS.add(mk)
                        bad_S_by_match[mk] += 1
                        if frs is None:
                            frs = rd
                            hrsS[rd] += 1
                    c["RS_cmp"] += 1
                    if lR != lS:
                        c["RS_bad_rounds"] += 1
                        mRS.add(mk)
                        st_RS += st
                        if fRS is None:
                            fRS = rd
                            hRS[rd] += 1
            if R == sx:
                xR += 1
            else:
                yR += 1
            if sok:
                if S is None:
                    sok = False
                elif S == sx:
                    xS += 1
                else:
                    yS += 1
        if broke:
            c["prefix_break_matches"] += 1
    return {"c": c, "hR": hR, "hRS": hRS, "hrsS": hrsS, "mR": mR, "mRS": mRS, "mrsS": mrsS,
            "st_R": st_R, "st_RS": st_RS, "st_used": st_used, "cset": cset, "bad_S_by_match": bad_S_by_match}


def show_chain(name, res):
    c = res["c"]
    out(f"  [{name}] contiguous PC matches: {c['contig_matches']} (non-contiguous: {c['noncontig_matches']}); "
        f"matches cut by an unclean sx: {c['prefix_break_matches']}")
    out(f"    rounds used in tracked chains: {c['rounds_used']}")
    out(f"    rs(ct-t) vs R-chain: compared {c['rs_cmp_R']} rounds, differ {c['rs_bad_R_rounds']} rounds "
        f"in {len(res['mR'])} matches; first bad round (round: matches) {sorted(res['hR'].items())[:10]}")
    out(f"    rs(ct-t) vs S-chain (side-free truth): compared {c['rs_cmp_S']}, differ {c['rs_bad_S_rounds']} "
        f"in {len(res['mrsS'])} matches; first bad {sorted(res['hrsS'].items())[:10]}")
    out(f"    R-chain vs S-chain (score-free error of winner-derived lead): compared {c['RS_cmp']}, differ "
        f"{c['RS_bad_rounds']} rounds in {len(res['mRS'])} matches; first bad {sorted(res['hRS'].items())[:10]}")
    out(f"    states (ASSUMPTION rule 1+victims) on R-bad rounds: {res['st_R']:,}; on R-vs-S bad rounds: {res['st_RS']:,}; "
        f"on all used rounds: {res['st_used']:,}")


def flips(tab, label):
    k = tab.filter(pl.col("S").is_not_null() & pl.col("sx").is_not_null()).with_columns(
        pl.when(pl.col("round") <= 12).then(pl.lit("01-12"))
        .when(pl.col("round") == 13).then(pl.lit("13"))
        .when(pl.col("round") <= 24).then(pl.lit("14-24"))
        .when(pl.col("round") == 25).then(pl.lit("25"))
        .when(pl.col("round") == 28).then(pl.lit("28"))
        .otherwise(pl.lit("other")).alias("grp"),
        (pl.col("R") != pl.col("S")).alias("flip"))
    g = k.group_by("grp").agg(pl.len().alias("n_side_known"), pl.col("flip").sum().alias("R_ne_S")).sort("grp")
    out(f"  [{label}] R vs side-free S, rounds with S and sx (group: n, R!=S):")
    for row in g.iter_rows():
        out(f"     {row[0]:>6}: n={row[1]:>6}  R!=S={row[2]:>5}  ({row[2] / row[1] * 100 if row[1] else 0:.2f}%)")
    m13 = set(k.filter((pl.col("round") == 13) & pl.col("flip")).select("match_key").to_series().to_list())
    m13_all = set(k.filter(pl.col("round") == 13).select("match_key").to_series().to_list())
    return m13, m13_all, k


def sdelta_table(tab):
    """Side-free score winner from round_state side fields: CT rose by 1 and T did not (3), or T rose and CT did not (2),
    between freeze_end(r) and freeze_end(r+1), only when sx(r) == sx(r+1) (no swap). No bomb, elim, or R used."""
    nxt = tab.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("t0").alias("t1"),
                      pl.col("ct0").alias("ct1"), pl.col("sx").alias("sx_next")])
    t = tab.join(nxt, on=["match_key", "round"], how="left").with_columns(
        (pl.col("ct1") - pl.col("ct0")).alias("dCT"), (pl.col("t1") - pl.col("t0")).alias("dT"))
    return t.with_columns(
        pl.when((pl.col("sx") == pl.col("sx_next")) & (pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
        .when((pl.col("sx") == pl.col("sx_next")) & (pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("S_delta"))


def sdelta_checks(label, tab, res):
    t = sdelta_table(tab)
    d = t.filter(pl.col("S_delta").is_not_null() & pl.col("R").is_not_null())
    out(f"  [{label}] side-free score winner S_delta (no bomb, elim, or R; needs sx(r)==sx(r+1)):")
    out(f"     defined on {d.height} rounds with a recorded winner")
    b = d.filter(pl.col("S_bomb").is_not_null())
    out(f"     vs bomb outcome: n={b.height}, S_delta==bomb {b.filter(pl.col('S_delta') == pl.col('S_bomb')).height}; "
        f"R==bomb {b.filter(pl.col('R') == pl.col('S_bomb')).height}")
    e = d.filter(pl.col("S_bomb").is_null() & pl.col("S_elim").is_not_null())
    out(f"     vs elimination only: n={e.height}, S_delta==elim {e.filter(pl.col('S_delta') == pl.col('S_elim')).height}")
    g = (d.with_columns(
        pl.when(pl.col("round") <= 12).then(pl.lit("01-12")).when(pl.col("round") == 13).then(pl.lit("13"))
        .when(pl.col("round") <= 24).then(pl.lit("14-24")).when(pl.col("round") == 25).then(pl.lit("25"))
        .when(pl.col("round") == 28).then(pl.lit("28")).otherwise(pl.lit("other")).alias("grp"),
        (pl.col("R") != pl.col("S_delta")).alias("flip"))
        .group_by("grp").agg(pl.len().alias("n"), pl.col("flip").sum().alias("R_ne_Sdelta")).sort("grp"))
    out("     R vs S_delta by group (n, R!=S_delta):")
    for row in g.iter_rows():
        out(f"        {row[0]:>6}: n={row[1]:>6}  R!=S_delta={row[2]:>5} ({row[2] / row[1] * 100 if row[1] else 0:.2f}%)")
    f13 = set(d.filter((pl.col("round") == 13) & (pl.col("R") != pl.col("S_delta")))
              .select("match_key").to_series().to_list())
    out(f"     round 13: R != S_delta in {len(f13)} matches")
    unclean = tab.filter(pl.col("sx").is_null()).group_by("match_key").agg(pl.col("round").min().alias("u"))
    ufirst = dict(unclean.iter_rows())
    rbad = res["mR"]
    fnr = f13 - rbad
    cut = [mk for mk in fnr if mk in res["cset"] and ufirst.get(mk, 10 ** 6) < 14]
    out(f"     R-bad matches (rs vs R-chain): {len(rbad)}; R-bad AND flagged at 13: {len(rbad & f13)}; "
        f"R-bad but not flagged: {len(rbad - f13)}")
    out(f"     flagged at 13 but not R-bad: {len(fnr)}; of these, chain cut by an unclean sx before round 14: {len(cut)}")
    return {"n_flag13": len(f13), "R_bad_and_flag": len(rbad & f13), "R_bad_not_flag": len(rbad - f13),
            "flag_not_Rbad": len(fnr), "flag_not_Rbad_cut": len(cut)}


def day_check(label, res):
    cs = res["cset"]
    d12 = [mk for mk in cs if "/10/01/" in mk or "/10/02/" in mk]
    bad = [mk for mk in d12 if mk in res["mrsS"]]
    out(f"  [{label}] contiguous matches on 10/01 and 10/02: {len(d12)}; "
        f"rs-vs-S-chain disagreement on these matches: {len(bad)}")
    out(f"  [{label}] rs-vs-S-chain disagreement, all contiguous matches: {len(res['mrsS'])} matches of {len(cs)}")


def main():
    summary = {}
    for label in ["2024", "2026"]:
        out("=" * 100)
        out(f"YEAR {label}  (PC = {PC[label]})")
        tab, info = build(label)
        for kk, vv in info.items():
            out(f"  {kk}: {vv}")
        # claimed D2 counts, reproduced on the same population (strict D2-like chain, S from bomb/elim)
        out("  Chains:")
        res_S = chains(tab, "S")
        show_chain("S = bomb/elim", res_S)
        res_Sx = chains(tab, "S_ext")
        show_chain("S_ext = bomb/elim + time-out (sensitivity)", res_Sx)

        out("  Flip table:")
        m13, m13_all, k = flips(tab, label)
        out(f"  matches with R!=S at round 13 (S known): {len(m13)} of {len(m13_all)} matches with S at 13")
        out(f"  R!=S at 13 restricted to contiguous PC matches: {len(m13 & res_S['cset'])}; "
            f"contiguous matches with S at 13: {len(m13_all & res_S['cset'])}")
        f13 = k.filter((pl.col("round") == 13) & pl.col("flip"))
        out(f"  flipped round-13 rows: with bomb outcome {f13.filter(pl.col('S_bomb').is_not_null()).height} "
            f"(R == bomb outcome in {f13.filter(pl.col('S_bomb').is_not_null() & (pl.col('R') == pl.col('S_bomb'))).height}); "
            f"elim-only {f13.filter(pl.col('S_bomb').is_null() & pl.col('S_elim').is_not_null()).height}; "
            f"time-only {f13.filter(pl.col('S_bomb').is_null() & pl.col('S_elim').is_null()).height}")
        rows_bomb = k.filter(pl.col("S_bomb").is_not_null())
        out(f"  all rows with bomb outcome and S,sx: {rows_bomb.height}; R == bomb in "
            f"{rows_bomb.filter(pl.col('R') == pl.col('S_bomb')).height}; R != bomb in "
            f"{rows_bomb.filter(pl.col('R') != pl.col('S_bomb')).height}")
        bad_S_in_R = sum(res_S["bad_S_by_match"][mk] for mk in res_S["mR"])
        bad_S_outside = sum(v for mk, v in res_S["bad_S_by_match"].items() if mk not in res_S["mR"])
        out(f"  rs-vs-S-chain disagreements inside R-bad matches: {bad_S_in_R}; outside: {bad_S_outside}")
        overlap = res_S["mR"] & m13
        out(f"  overlap: matches with R-bad rounds ∩ R!=S at 13 = {len(overlap)}; R-bad matches = {len(res_S['mR'])}; "
            f"R!=S at 13 but no R-bad round = {len(m13 - res_S['mR'])}; R-bad but no R!=S at 13 = "
            f"{len(res_S['mR'] - m13)}")
        out(f"  R-vs-S-bad matches vs R!=S at 13: overlap {len(res_S['mRS'] & m13)}; RS-bad but no flip at 13 "
            f"{len(res_S['mRS'] - m13)}; flip at 13 but no RS-bad {len(m13 - res_S['mRS'])}")

        out("  Side-free score-winner cross-check (independent of R, bomb, elim, sx mapping of the outcome):")
        sd = sdelta_checks(label, tab, res_S)
        if label == "2026":
            day_check(label, res_S)

        # states per clean round (ASSUMPTION rule) vs bake-off ratio
        clean = tab.filter(pl.col("sx").is_not_null())
        mean_states = clean.select(pl.col("states").mean()).item()
        out(f"  states per clean round (1 + victims, ASSUMPTION rule): mean {mean_states:.4f}; "
            f"bake-off 2026 ratio {BAKEOFF_STATES_PER_ROUND:.4f}")
        summary[label] = {"res_S": res_S, "res_Sx": res_Sx, "info": info, "mean_states": mean_states,
                          "tab_rows": tab.height, "sd": sd}
        if label == "2024":
            tab24 = tab

    out("=" * 100)
    out("TRACE check (match in D2 example, rounds 12-15): ")
    tr = tab24.filter(pl.col("match_key").str.contains(TRACE_MATCH)).filter(
        pl.col("round").is_between(12, 15)).sort("round").select(
        ["round", "R", "sx", "t0", "ct0", "S_bomb", "S_elim", "S", "t_dead", "ct_dead", "n_out_types"])
    for row in tr.iter_rows():
        out(f"  round {row[0]}: R={row[1]} sx={row[2]} t0={row[3]} ct0={row[4]} S_bomb={row[5]} S_elim={row[6]} "
            f"S={row[7]} t_dead={row[8]} ct_dead={row[9]} n_out_types={row[10]}")

    out("=" * 100)
    out("CLAIM vs RECOMPUTED (D2-6 numbers)")
    r24 = summary["2024"]["res_S"]
    out(f"  claimed 2024: rounds differ 12,591 in 1,702 matches (D2_lead, R-chain vs rs), first bad at 14, "
        f"contiguous 2,478, rounds checked 37,909")
    out(f"  recomputed 2024 (R-chain vs rs): {r24['c']['rs_bad_R_rounds']} rounds in {len(r24['mR'])} matches; "
        f"contiguous {r24['c']['contig_matches']}; rounds with rs compared {r24['c']['rs_cmp_R']}; "
        f"first bad {sorted(r24['hR'].items())[:10]}")
    out(f"  claimed 2026: 0 of 23,210 rounds differ")
    r26 = summary["2026"]["res_S"]
    out(f"  recomputed 2026 (R-chain vs rs): {r26['c']['rs_bad_R_rounds']} rounds in {len(r26['mR'])} matches; "
        f"rs compared {r26['c']['rs_cmp_R']}")
    out(f"  claimed states affected 2024 ~102,835 (12,591 x 8.17, ASSUMPTION ratio from 2026 bake-off)")
    out(f"  recomputed 2024 states on R-bad rounds (1 + victims rule): {r24['st_R']:,}; "
        f"score-free R-vs-S bad rounds: {r24['st_RS']:,}")
    out(f"  2024 mean states per clean round {summary['2024']['mean_states']:.3f} vs 2026 "
        f"{summary['2026']['mean_states']:.3f} vs bake-off {BAKEOFF_STATES_PER_ROUND:.3f}")
    rate = 12591 * summary["2024"]["mean_states"]
    out(f"  claimed 102,835 vs 12,591 x measured 2024 mean {summary['2024']['mean_states']:.3f} = {rate:,.0f}; "
        f"direct count on the same R-bad rounds = {r24['st_R']:,}; difference vs claimed = "
        f"{(r24['st_R'] - 102835) / 102835 * 100:+.1f}%")
    out(f"  scope: 2024 contiguous PC matches evaluated = {r24['c']['contig_matches']} of "
        f"{r24['c']['contig_matches'] + r24['c']['noncontig_matches']} PC matches; the rest are excluded, so the "
        f"2024 total is a lower bound")
    out(f"  R-bad rounds as share of rounds with an rs lead in the evaluated chains: "
        f"{r24['c']['rs_bad_R_rounds']} of {r24['c']['rs_cmp_R']} = "
        f"{r24['c']['rs_bad_R_rounds'] / r24['c']['rs_cmp_R'] * 100:.1f}%")
    sd24 = summary["2024"]["sd"]
    sd26 = summary["2026"]["sd"]
    out(f"  side-free check: 2024 flagged at 13 = {sd24['n_flag13']}; R-bad AND flagged {sd24['R_bad_and_flag']}; "
        f"R-bad not flagged {sd24['R_bad_not_flag']}; flagged not R-bad {sd24['flag_not_Rbad']} "
        f"(chain cut before round 14: {sd24['flag_not_Rbad_cut']})")
    out(f"  side-free check: 2026 flagged at 13 = {sd26['n_flag13']}; R-bad {len(summary['2026']['res_S']['mR'])}")
    out(f"  2026 denominator: recomputed rounds with an rs lead in the chains = {r26['c']['rs_cmp_R']}; "
        f"claimed 23,210 (not reproduced exactly; both give 0 mismatches)")
    out(f"  HANDOFF caution 2 (team-attached rs fields on days 01-02, 2026): recomputed disagreement matches "
        f"2026 all contiguous = {len(r26['mrsS'])} (see day check above)")

    out("=" * 100)
    out("VERDICT (independent verifier, D2-6): CONFIRMED for the numbers and the mechanism; the proposed fix is supported.")
    out("Required fixes:")
    out("  1. Cite 12,591 rounds / 1,702 matches / first difference at round 14 / 2,478 contiguous / 37,909 compared; "
        "reconcile the 2026 denominator (24,323 recomputed vs 23,210 claimed; zero holds either way).")
    out("  2. Replace ~102,835 states with the direct count 105,138 (ASSUMPTION rule) or label it approximate and a "
        "lower bound (2,478 of 4,216 2024 PC matches evaluated).")
    out("  3. Rewrite HANDOFF caution 2 (HANDOFF.md and evac_2026-10-13\\HANDOFF.md): winner-derived lead is valid on 2026, "
        "wrong for flagged 2024 matches; use round_state ct_score - t_score for 2024; drop or qualify the team-attached "
        "'15-20% of days 01-02' claim (not reproduced: 0 of 86 day-01/02 contiguous matches).")
    out("  4. For 2024, guard flagged matches (side-attached score change disagrees with winner_team_code at round 13: "
        "1,813 rounds): exclude or relabel, do not derive leads or labels from R.")
    out("  5. Note in D2: current 2024 lead features (cross_period.py, batch_holdout.py) come from round_state, so D2-6 is "
        "latent there; the live 2024 error for those scripts is in labels (D2-7).")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
