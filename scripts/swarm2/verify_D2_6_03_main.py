"""verify D2-6 (independent verifier): recompute the winner-derived vs round_state score-lead gap from raw
tome parquet, and test the strongest alternative explanations with sources that do not use winner_team_code.

Claim under test (D2-6, notes\\swarm2\\D2.md): 2024 winner-derived cumulative score lead differs from the
round_state freeze-end lead in 12,591 rounds across 1,702 matches, first difference at round 14 in every
affected match; 2026 has 0 differences in 23,210 rounds; about 102,835 states affected (ASSUMPTION).

Sources used (all raw tome channels, column projection, no D2_* intermediates):
  round_end (winner_team_code = R, the field under test; tick; player_count)
  round_state (round_freeze_end: t_score/ct_score; cs_win_panel_match: end-of-match t/ct scores)
  bomb_state (bomb_exploded / bomb_defused): side-level, independent of R and of score fields
  player_death (player_team_code): side-level death record
  player_info (team_code): roster side per round, used only for team identity X
  header (t_starters_score_final, ct_starters_score_final)

Side convention: 2 = T, 3 = CT. X = the five players on code 2 in the first valid 5v5 round of the match.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_6_03_main.py
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
MK = "match_key"
BOMB_TAIL = 40                      # ASSUMPTION: carried from D2_truth.py (bomb events up to 40 ticks past round_end)
PC_D2 = {"2024": 20, "2026": 10}    # ASSUMPTION: D2 player_count filter (full 5v5 round)
STATES_PER_ROUND = 234265 / 28683   # ASSUMPTION: bake-off kept states per kept round (2026 only)


def scan(label, ch):
    if ch == "header":
        d = TOME[label] / "header"
    else:
        d = Path(sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0])
    return pl.scan_parquet(sorted(glob.glob(str(d / "dataframe_*"))))


def build(label):
    info = {}
    ro = scan(label, "round_end").select([MK, "round", "tick", "winner_team_code", "player_count"]).collect()
    info["round_end_rows"] = ro.height
    info["round_end_dup_pairs"] = ro.group_by([MK, "round"]).len().filter(pl.col("len") > 1).height
    ro = ro.sort("tick").unique([MK, "round"], keep="first", maintain_order=True)
    ro = ro.rename({"tick": "t_end", "winner_team_code": "R", "player_count": "pc"})

    fe_raw = (scan(label, "round_state").filter(pl.col("event_type") == "round_freeze_end")
              .select([MK, "round", "tick", "t_score", "ct_score"]).collect())
    info["freeze_end_rows"] = fe_raw.height
    info["freeze_end_rounds_multi"] = fe_raw.group_by([MK, "round"]).len().filter(pl.col("len") > 1).height
    fe = (fe_raw.sort("tick").unique([MK, "round"], keep="first", maintain_order=True)
          .rename({"tick": "t_fe", "t_score": "t0", "ct_score": "ct0"}))
    fe_next = fe.select([MK, (pl.col("round") - 1).alias("round"),
                         pl.col("t0").alias("t1"), pl.col("ct0").alias("ct1")])

    rt = ro.join(fe, on=[MK, "round"], how="left").join(fe_next, on=[MK, "round"], how="left")

    pi = scan(label, "player_info").select([MK, "round", "player_id", "team_code"]).collect()
    info["player_info_rows"] = pi.height
    info["player_info_conflict_keys"] = (pi.group_by([MK, "round", "player_id"])
                                         .agg(pl.col("team_code").n_unique().alias("nu"))
                                         .filter(pl.col("nu") > 1).height)
    pi = pi.unique([MK, "round", "player_id"], keep="first", maintain_order=True)
    cnt = pi.group_by([MK, "round"]).agg((pl.col("team_code") == 2).sum().alias("n2"),
                                         (pl.col("team_code") == 3).sum().alias("n3"))
    valid = (cnt.filter((pl.col("n2") == 5) & (pl.col("n3") == 5)).sort([MK, "round"])
             .unique([MK], keep="first", maintain_order=True).select([MK, pl.col("round").alias("F")]))
    xm = (pi.join(valid, on=MK).filter(pl.col("round") == pl.col("F")).filter(pl.col("team_code") == 2)
          .select([MK, "player_id"]).unique().with_columns(pl.lit(1).alias("isX")))
    pix = pi.join(xm, on=[MK, "player_id"], how="left")
    sxg = (pix.filter(pl.col("isX") == 1).group_by([MK, "round"])
           .agg(pl.col("team_code").min().alias("cmin"), pl.col("team_code").max().alias("cmax"),
                pl.len().alias("nX")))
    sxg = sxg.with_columns(pl.when(pl.col("cmin") == pl.col("cmax")).then(pl.col("cmin"))
                           .otherwise(pl.lit(None, dtype=pl.Int64)).cast(pl.Int64).alias("sx")).select([MK, "round", "sx", "nX"])
    yg = (pix.filter(pl.col("isX").is_null()).group_by([MK, "round"])
          .agg(pl.col("team_code").min().alias("ymin"), pl.col("team_code").max().alias("ymax"), pl.len().alias("nY")))
    sxg = sxg.join(yg, on=[MK, "round"], how="left")
    info["sx_rounds"] = sxg.height
    info["sx_null_rounds"] = sxg.filter(pl.col("sx").is_null()).height
    info["sx_X_count_not5_rounds"] = sxg.filter(pl.col("nX") != 5).height
    info["sx_Y_inconsistent_rounds"] = sxg.filter(
        pl.col("sx").is_not_null() & ~((pl.col("ymin") == pl.col("ymax")) & (pl.col("ymin") == 5 - pl.col("sx")))).height
    sxg_next = sxg.select([MK, (pl.col("round") - 1).alias("round"), pl.col("sx").alias("sx_next")])
    rt = rt.join(sxg.select([MK, "round", "sx", "nX"]), on=[MK, "round"], how="left").join(sxg_next, on=[MK, "round"], how="left")

    # bomb outcome, side-level (exploded -> T = 2, defused -> CT = 3), window freeze_end..round_end+tail
    bo = (scan(label, "bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
          .select([MK, "round", "tick", "event_type"]).collect())
    rw = rt.select([MK, "round", "t_fe", "t_end"])
    bw = (bo.join(rw, on=[MK, "round"]).filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + BOMB_TAIL))
          .sort("tick").group_by([MK, "round"]).agg(pl.col("event_type").first().alias("bo_first"),
                                                    pl.col("event_type").n_unique().alias("bo_types")))
    rt = rt.join(bw, on=[MK, "round"], how="left").with_columns(
        pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2))
        .when(pl.col("bo_first") == "bomb_defused").then(pl.lit(3))
        .otherwise(pl.lit(None, dtype=pl.Int64)).cast(pl.Int64).alias("bomb_side"))

    # elimination, side-level: the side with five distinct deaths loses (player_team_code of the victim)
    dth = scan(label, "player_death").select([MK, "round", "tick", "player_id", "player_team_code"]).collect()
    dw = (dth.join(rw, on=[MK, "round"]).filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + BOMB_TAIL))
          .filter(pl.col("player_team_code").is_in([2, 3])).unique([MK, "round", "player_id"], keep="first"))
    ed = dw.group_by([MK, "round"]).agg((pl.col("player_team_code") == 2).sum().alias("t_dead"),
                                        (pl.col("player_team_code") == 3).sum().alias("ct_dead"))
    ed = ed.with_columns(
        pl.when((pl.col("ct_dead") == 5) & (pl.col("t_dead") < 5)).then(pl.lit(2))
        .when((pl.col("t_dead") == 5) & (pl.col("ct_dead") < 5)).then(pl.lit(3))
        .otherwise(pl.lit(None, dtype=pl.Int64)).cast(pl.Int64).alias("elim_side"))
    rt = rt.join(ed.select([MK, "round", "elim_side"]), on=[MK, "round"], how="left")

    # panel: end-of-match round_state row (engine), t/ct as side-attached scores at the end
    panel = (scan(label, "round_state").filter(pl.col("event_type") == "cs_win_panel_match")
             .select([MK, "round", "t_score", "ct_score"]).collect()
             .unique([MK], keep="first", maintain_order=True)
             .rename({"round": "round_panel", "t_score": "t_p", "ct_score": "ct_p"}))
    hd = (scan(label, "header").select(["key", "t_starters_score_final", "ct_starters_score_final"]).collect()
          .rename({"key": MK}))
    return rt, panel, hd, info


def analyze(label):
    rt, panel, hd, info = build(label)
    print(f"\n######## {label} ########")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  round table rows: {rt.height}, matches: {rt[MK].n_unique()}")

    rows = rt.sort([MK, "round"]).to_dicts()
    M = defaultdict(dict)
    for r in rows:
        M[r[MK]][r["round"]] = r

    # score-chain winner per round, side-attached fields, team X tracked through the swap via sx(r), sx(r+1)
    for mk, byr in M.items():
        for rd, r in byr.items():
            r["W_sc"] = None
            sx, sxn = r["sx"], r["sx_next"]
            if sx is None or sxn is None or None in (r["t0"], r["ct0"], r["t1"], r["ct1"]):
                continue
            x0 = r["t0"] if sx == 2 else r["ct0"]
            y0 = r["ct0"] if sx == 2 else r["t0"]
            x1 = r["t1"] if sxn == 2 else r["ct1"]
            y1 = r["ct1"] if sxn == 2 else r["t1"]
            dX, dY = x1 - x0, y1 - y0
            if dX == 1 and dY == 0:
                r["W_sc"] = sx
            elif dY == 1 and dX == 0:
                r["W_sc"] = 5 - sx
            r["dX"], r["dY"] = dX, dY
            r["swap"] = (sxn != sx)

    # independent side-level winner: bomb first, else elimination
    for mk, byr in M.items():
        for rd, r in byr.items():
            r["W_ind"] = r["bomb_side"] if r["bomb_side"] is not None else r["elim_side"]
            r["W_ind_src"] = "bomb" if r["bomb_side"] is not None else ("elim" if r["elim_side"] is not None else None)

    # ---- per-round tests (sx-free where possible)
    def tally(filt, name):
        n_ind = n_flip = n_sc = n_sc_eq_ind = n_sc_known_both = n_sc_eq_R = 0
        for mk, byr in M.items():
            for rd, r in byr.items():
                if not filt(rd, r):
                    continue
                if r["W_ind"] is not None:
                    n_ind += 1
                    if r["R"] != r["W_ind"]:
                        n_flip += 1
                if r["W_sc"] is not None and r["W_ind"] is not None:
                    n_sc_known_both += 1
                    if r["W_sc"] == r["W_ind"]:
                        n_sc_eq_ind += 1
                if r["W_sc"] is not None:
                    n_sc += 1
                    if r["W_sc"] == r["R"]:
                        n_sc_eq_R += 1
        rate = (n_flip / n_ind) if n_ind else float("nan")
        print(f"  {name:<34} R vs W_ind: n={n_ind} R!=W_ind={n_flip} ({rate:.4f}) | "
              f"W_sc known={n_sc} W_sc==R {n_sc_eq_R} | W_sc vs W_ind: n={n_sc_known_both} equal={n_sc_eq_ind}")

    print("\n  Per-round tests (R = winner_team_code under test; W_ind = bomb else elimination; W_sc = score-chain):")
    tally(lambda rd, r: True, "all rounds")
    tally(lambda rd, r: rd == 13, "round 13")
    tally(lambda rd, r: rd <= 12, "rounds 1-12")
    tally(lambda rd, r: 14 <= rd <= 24, "rounds 14-24")
    tally(lambda rd, r: rd == 25, "round 25")
    tally(lambda rd, r: rd == 28, "round 28")
    tally(lambda rd, r: r.get("swap") is True, "swap rounds (sx changes)")
    tally(lambda rd, r: r.get("swap") is False, "non-swap rounds")
    tally(lambda rd, r: r.get("swap") is True and rd == 13, "round 13 swap rows")

    # bomb vs elimination disagreement where both known
    both = [(r["bomb_side"], r["elim_side"]) for byr in M.values() for r in byr.values()
            if r["bomb_side"] is not None and r["elim_side"] is not None]
    print(f"  bomb vs elimination where both known: n={len(both)} equal={sum(1 for a, b in both if a == b)}")

    # score-internal: t0 + ct0 == round - 1 at each freeze end (fields are cumulative team scores)
    ok_sum = bad_sum = 0
    bad_examples = []
    for mk, byr in M.items():
        for rd, r in byr.items():
            if r["t0"] is None:
                continue
            if r["t0"] + r["ct0"] == rd - 1:
                ok_sum += 1
            else:
                bad_sum += 1
                if len(bad_examples) < 3:
                    bad_examples.append((mk, rd, r["t0"], r["ct0"]))
    print(f"  t_score + ct_score == round-1 at freeze_end: ok={ok_sum} bad={bad_sum} examples={bad_examples}")

    # ---- match-level selections and the D2_lead chain (replicated exactly)
    def contiguous(byr, selected):
        rs = sorted(selected)
        return bool(rs) and rs[0] == 1 and rs[-1] == len(rs)

    def chain(mk, byr, rmax):
        scoreX = scoreY = 0
        checked = bad = 0
        first_bad = None
        bad_rounds = []
        for rd in range(1, rmax + 1):
            r = byr[rd]
            sx, R = r["sx"], r["R"]
            if sx is None or R is None:
                return None
            if r["t0"] is not None and r["ct0"] is not None:
                rs_lead = r["ct0"] - r["t0"]
                tracked = (scoreX - scoreY) if sx == 3 else (scoreY - scoreX)
                checked += 1
                if tracked != rs_lead:
                    bad += 1
                    bad_rounds.append(rd)
                    if first_bad is None:
                        first_bad = rd
            if R == sx:
                scoreX += 1
            else:
                scoreY += 1
        return {"checked": checked, "bad": bad, "first_bad": first_bad, "bad_rounds": bad_rounds}

    for sel_name in ["D2 filter (player_count==%d)" % PC_D2[label], "all rounds (no player_count filter)"]:
        n_contig = 0
        checked = bad_total = 0
        bad_matches = []
        first_hist = Counter()
        match_res = {}
        for mk, byr in M.items():
            if sel_name.startswith("D2"):
                sel = [rd for rd, r in byr.items() if r["pc"] == PC_D2[label]]
            else:
                sel = list(byr.keys())
            if not contiguous(byr, sel):
                continue
            rmax = max(sel)
            res = chain(mk, byr, rmax)
            if res is None:
                continue
            n_contig += 1
            checked += res["checked"]
            match_res[mk] = res
            if res["bad"]:
                bad_total += res["bad"]
                bad_matches.append(mk)
                first_hist[res["first_bad"]] += 1
        print(f"\n  [{sel_name}] contiguous & chain-complete matches: {n_contig}")
        print(f"    rounds with round_state lead and a full R-chain: {checked}")
        print(f"    R-tracked lead != round_state lead: {bad_total} rounds in {len(bad_matches)} matches")
        print(f"    first bad round per affected match (round: matches): {sorted(first_hist.items())[:12]}")
        if sel_name.startswith("D2"):
            d2_res = match_res
            d2_bad = set(bad_matches)

    # ---- do the R-chain divergent matches coincide with R flips vs bomb/elim at round 13 (D2 selection)?
    print("\n  Link between R flips and R-chain divergence (D2 selection):")
    confusion = Counter()
    first_flip_check = Counter()
    for mk, res in d2_res.items():
        byr = M[mk]
        f13 = byr[13]["W_ind"] is not None and byr[13]["R"] != byr[13]["W_ind"] if 13 in byr else None
        confusion[("flip13" if f13 else ("no_flip13" if f13 is not None else "13_unknown"),
                   "diverge" if res["bad"] else "clean")] += 1
        # first R flip against W_ind (any round) vs first divergence
        flips = [rd for rd in sorted(byr) if byr[rd]["W_ind"] is not None and byr[rd]["R"] != byr[rd]["W_ind"]]
        if flips and res["bad"]:
            first_flip_check["first_bad == first_flip+1" if res["first_bad"] == flips[0] + 1 else "first_bad != first_flip+1"] += 1
        elif flips and not res["bad"]:
            first_flip_check["flip but no divergence (flip at last round?)"] += 1
        elif not flips and res["bad"]:
            first_flip_check["divergence but no W_ind flip"] += 1
    for k, v in sorted(confusion.items()):
        print(f"    round13 {k[0]:<14} x chain {k[1]:<8}: {v}")
    for k, v in sorted(first_flip_check.items()):
        print(f"    {k}: {v}")

    # ---- final-score checks: panel (engine end-of-match row) vs R-tally vs score-chain tally
    pm = {r[MK]: r for r in panel.to_dicts()}
    hdm = {r[MK]: r for r in hd.to_dicts()}
    fin = Counter()
    fin_flagged = Counter()
    panel_sum_ok = Counter()
    header_rel = Counter()
    for mk, byr in M.items():
        if mk not in pm:
            fin["no panel row"] += 1
            continue
        rmax = max(byr)
        rr = byr[rmax]
        if rr["sx"] is None or any(byr.get(k, {}).get("sx") is None for k in range(1, rmax + 1)):
            fin["sx gap (skip)"] += 1
            continue
        p = pm[mk]
        panel_sum_ok[(p["t_p"] + p["ct_p"] == rmax)] += 1
        sx_last = rr["sx"]
        X_p = p["ct_p"] if sx_last == 3 else p["t_p"]
        Y_p = p["t_p"] if sx_last == 3 else p["ct_p"]
        X_R = sum(1 for k in range(1, rmax + 1) if byr[k]["R"] == byr[k]["sx"])
        Y_R = rmax - X_R
        # score chain over rounds 1..rmax-1 (no R used), last round from panel
        X_sc = 0
        known = True
        for k in range(1, rmax):
            w = byr[k]["W_sc"]
            if w is None:
                known = False
                break
            X_sc += 1 if w == byr[k]["sx"] else 0
        # flagged: any round where R disagrees with W_ind (W_ind known)
        flagged = any(byr[k]["W_ind"] is not None and byr[k]["R"] != byr[k]["W_ind"] for k in range(1, rmax + 1))
        ok_R = (X_R == X_p) and (Y_R == Y_p)
        fin_flagged[("flagged" if flagged else "unflagged", "R-tally==panel" if ok_R else "R-tally!=panel")] += 1
        if known:
            last_x = X_p - X_sc
            fin[("score-chain consistent with panel" if last_x in (0, 1) else "score-chain NOT consistent with panel",
                 "flagged" if flagged else "unflagged")] += 1
        # header relation: X = team that started on T (code 2 in round 1 of valid rounds)
        h = hdm.get(mk)
        if h is not None:
            t_st, ct_st = h["t_starters_score_final"], h["ct_starters_score_final"]
            if X_p == t_st and Y_p == ct_st:
                header_rel["X=t_starters, Y=ct_starters"] += 1
            elif X_p == ct_st and Y_p == t_st:
                header_rel["X=ct_starters, Y=t_starters"] += 1
            else:
                header_rel["neither (panel vs header)"] += 1
    print("\n  Final-score checks (panel = engine end-of-match row; no winner_team_code used on the panel side):")
    print(f"    panel t+ct == rounds played (rmax): {dict(panel_sum_ok)}")
    for k, v in sorted(fin_flagged.items()):
        print(f"    {k[0]:<10} {k[1]:<18}: {v}")
    for k, v in sorted(fin.items(), key=lambda kv: str(kv[0])):
        print(f"    {k}: {v}")
    print(f"    header starters vs panel: {dict(header_rel)}")
    return rt, M, info


def main():
    res = {}
    for label in ["2024", "2026"]:
        res[label] = analyze(label)
    n_states = 12591 * STATES_PER_ROUND
    print(f"\nASSUMPTION states-affected arithmetic (not a count in any current artifact): "
          f"12,591 x {STATES_PER_ROUND:.4f} = {n_states:,.0f}")


if __name__ == "__main__":
    main()
