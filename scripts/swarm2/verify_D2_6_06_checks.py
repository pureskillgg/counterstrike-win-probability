"""verify D2-6: second-pass checks (alternative explanations and loose ends from the first pass).

  A. Truth T = score-chain winner where known, else bomb/elimination winner. Flip = R != T.
  B. D2-rule chains (pc filter, contiguity on filtered rounds, break at first null sx, D2 sx):
     first divergence vs first flip under T. Divergence with no flip under T.
  C. 2026 match that diverges when the pc filter is dropped (round 2 first difference): details.
  D. Header final scores vs panel and vs R-tally (is the header independent of winner_team_code?).
  E. Rounds where D2 sx is null but raw sx is non-null: what the raw sx rests on.
  F. Round 13 split by side swap at its start (sx(13) != sx(12)).

Reuses build() from verify_D2_6_03_main.py. Run with the project interpreter, -I -X utf8.
"""
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
MK = "match_key"
PC = {"2024": 20, "2026": 10}

spec = importlib.util.spec_from_file_location("v03", ROOT / "scripts" / "swarm2" / "verify_D2_6_03_main.py")
v03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v03)


def load_M(label):
    rt, panel, hd, info = v03.build(label)
    rows = rt.sort([MK, "round"]).to_dicts()
    M = defaultdict(dict)
    for r in rows:
        M[r[MK]][r["round"]] = r
    # score-chain winner (side), as in verify_D2_6_03_main.analyze
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
        for rd, r in byr.items():
            r["W_ind"] = r["bomb_side"] if r["bomb_side"] is not None else r["elim_side"]
            r["T"] = r["W_sc"] if r["W_sc"] is not None else r["W_ind"]
            r["T_src"] = "score" if r["W_sc"] is not None else ("bomb/elim" if r["W_ind"] is not None else None)
            prev = byr.get(rd - 1)
            r["swap_in"] = (prev is not None and r["sx"] is not None and prev["sx"] is not None and r["sx"] != prev["sx"])
    return M, panel, hd, info


def d2_chain(byr, label, sx_key, rs_selected):
    """D2_lead semantics for one match. Returns dict or None when not contiguous on the pc-filtered rounds."""
    if not rs_selected or rs_selected[0] != 1 or rs_selected[-1] != len(rs_selected):
        return None
    rmax = rs_selected[-1]
    scoreX = scoreY = 0
    checked = bad = 0
    first_bad = None
    stop_round = None
    for rd in range(1, rmax + 1):
        r = byr[rd]
        sx, R = r[sx_key], r["R"]
        if sx is None or R is None:
            stop_round = rd
            break
        if r["t0"] is not None and r["ct0"] is not None:
            rs_lead = r["ct0"] - r["t0"]
            tracked = (scoreX - scoreY) if sx == 3 else (scoreY - scoreX)
            checked += 1
            if tracked != rs_lead:
                bad += 1
                if first_bad is None:
                    first_bad = rd
        scoreX += 1 if R == sx else 0
        scoreY += 0 if R == sx else 1
    return {"checked": checked, "bad": bad, "first_bad": first_bad, "stop": stop_round, "rmax": rmax}


def section_B(label, M):
    print(f"\n--- [{label}] B. D2-rule chains: first divergence vs first flip under T (T = score-chain else bomb/elim)")
    stats = Counter()
    examples = []
    flip_only_examples = []
    for mk, byr in M.items():
        rs_f = sorted(rd for rd, r in byr.items() if r["pc"] == PC[label])
        res = d2_chain(byr, label, "sx_d2" if "sx_d2" in next(iter(byr.values())) else "sx", rs_f)
        if res is None:
            continue
        stats["contig"] += 1
        flips = [rd for rd in sorted(byr) if byr[rd]["T"] is not None and byr[rd]["R"] != byr[rd]["T"]]
        f_first = flips[0] if flips else None
        if res["first_bad"] is not None:
            stats["diverge"] += 1
            if f_first is None:
                stats["diverge_no_flip_under_T"] += 1
                if len(examples) < 4:
                    examples.append((mk, res["first_bad"], res["stop"]))
            elif res["first_bad"] == f_first + 1:
                stats["diverge_first_bad_eq_flip_plus1"] += 1
            else:
                stats["diverge_first_bad_ne_flip_plus1"] += 1
                if len(flip_only_examples) < 4:
                    flip_only_examples.append((mk, f_first, res["first_bad"], flips[:6]))
        else:
            stats["clean"] += 1
            if f_first is not None:
                stats["clean_but_flip_under_T"] += 1
                if len(flip_only_examples) < 8:
                    flip_only_examples.append(("clean", mk, f_first, res["rmax"], res["stop"], flips[:6]))
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}")
    print("    examples diverge with no flip under T (match, first_bad, stop):", examples)
    print("    examples flip/divergence mismatch:")
    for e in flip_only_examples[:8]:
        print("      ", e)


def section_A(label, M):
    print(f"\n--- [{label}] A. R vs T (T = score-chain where known, else bomb/elim)")
    tab = Counter()
    for mk, byr in M.items():
        for rd, r in byr.items():
            if r["T"] is None:
                tab[("T unknown", rd if rd in (13, 25, 28) else ("1-12" if rd <= 12 else ("14-24" if rd <= 24 else "other")))] += 1
                continue
            key = ("13" if rd == 13 else ("25" if rd == 25 else ("28" if rd == 28 else ("1-12" if rd <= 12 else ("14-24" if rd <= 24 else "other")))))
            tab[(key, "n")] += 1
            if r["R"] != r["T"]:
                tab[(key, "flip")] += 1
    for grp in ["1-12", "13", "14-24", "25", "28", "other"]:
        n = tab[(grp, "n")]
        f = tab[(grp, "flip")]
        print(f"    rounds {grp:<7} T known n={n:<6} R!=T={f:<6} rate={(f / n if n else float('nan')):.4f}   T-unknown rounds={tab[('T unknown', grp if grp not in ('1-12','14-24','other') else grp)] if False else ''}")
    unk = Counter()
    for mk, byr in M.items():
        for rd, r in byr.items():
            if r["T"] is None:
                unk[("13" if rd == 13 else ("1-12" if rd <= 12 else "other"))] += 1
    print("    rounds with no truth (T unknown) for 13 / 1-12 / other:", unk["13"], unk["1-12"], unk["other"])


def section_C(label, M):
    print(f"\n--- [{label}] C. 2026 match that diverges only when the pc filter is dropped")
    for mk, byr in M.items():
        rs_all = sorted(byr)
        if not rs_all or rs_all[0] != 1 or rs_all[-1] != len(rs_all):
            continue
        rmax = rs_all[-1]
        res_all = d2_chain(byr, label, "sx", rs_all)
        if res_all is None or res_all["first_bad"] is None:
            continue
        pcs = Counter(byr[rd]["pc"] for rd in rs_all)
        print(f"    match {mk}  rmax={rmax}  first_bad={res_all['first_bad']}  bad={res_all['bad']}  pc values={dict(pcs)}")
        for rd in range(1, min(rmax, 4) + 1):
            r = byr[rd]
            print(f"      r{rd}: R={r['R']} sx={r['sx']} pc={r['pc']} t0={r['t0']} ct0={r['ct0']} t1={r['t1']} ct1={r['ct1']} "
                  f"sx_next={r['sx_next']} W_sc={r['W_sc']} bomb={r['bomb_side']} elim={r['elim_side']} T={r['T']} nX={r['nX']}")
        print(f"      panel: {panel_for(mk, label)}")


PANEL_CACHE = {}


def panel_for(mk, label):
    p = PANEL_CACHE[label].get(mk)
    return p


def section_D(label, M, panel, hd):
    print(f"\n--- [{label}] D. header starters vs panel vs R-tally (matches with complete sx chains)")
    pm = {r[MK]: r for r in panel.to_dicts()}
    hdm = {r[MK]: r for r in hd.to_dicts()}
    cnt = Counter()
    for mk, byr in M.items():
        if mk not in pm or mk not in hdm:
            continue
        rmax = max(byr)
        if any(byr.get(k, {}).get("sx") is None for k in range(1, rmax + 1)):
            continue
        p = pm[mk]
        h = hdm[mk]
        sx_last = byr[rmax]["sx"]
        X_p = p["ct_p"] if sx_last == 3 else p["t_p"]
        Y_p = p["t_p"] if sx_last == 3 else p["ct_p"]
        X_R = sum(1 for k in range(1, rmax + 1) if byr[k]["R"] == byr[k]["sx"])
        Y_R = rmax - X_R
        R_ok = (X_R == X_p and Y_R == Y_p)
        if h["t_starters_score_final"] == X_p and h["ct_starters_score_final"] == Y_p:
            hrel = "header=panel(X=starters T)"
        elif h["ct_starters_score_final"] == X_p and h["t_starters_score_final"] == Y_p:
            hrel = "header=panel(X=starters CT)"
        else:
            hrel = "header!=panel"
        hR = "header=R-tally" if (h["t_starters_score_final"] == X_R and h["ct_starters_score_final"] == Y_R) else (
            "header=R-tally(swapped)" if (h["ct_starters_score_final"] == X_R and h["t_starters_score_final"] == Y_R) else "header!=R-tally")
        cnt[(hrel, "R-tally=panel" if R_ok else "R-tally!=panel", hR)] += 1
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"    {v:>5}  {k[0]:<28} {k[1]:<18} {k[2]}")


def section_E(label, M):
    print(f"\n--- [{label}] E. rounds where D2 sx is null but raw sx is set: player_info support")
    cnt = Counter()
    for mk, byr in M.items():
        for rd, r in byr.items():
            if r.get("sx_d2") is None and r["sx"] is not None:
                cnt[("nX=%s" % r["nX"],)] += 1
    print("   ", dict(cnt))


def section_F(label, M):
    print(f"\n--- [{label}] F. round 13: side swap at its start (sx(13) != sx(12)) vs no swap")
    tab = Counter()
    for mk, byr in M.items():
        r = byr.get(13)
        if r is None or r["T"] is None:
            continue
        key = "swap_in" if r["swap_in"] else "no_swap_in"
        tab[(key, "n")] += 1
        if r["R"] != r["T"]:
            tab[(key, "flip")] += 1
    for key in ["swap_in", "no_swap_in"]:
        n, f = tab[(key, "n")], tab[(key, "flip")]
        print(f"    round 13 {key:<10} T known n={n:<6} R!=T={f:<6} rate={(f / n if n else float('nan')):.4f}")


def main():
    global PANEL_CACHE
    for label in ["2024", "2026"]:
        M, panel, hd, info = load_M(label)
        PANEL_CACHE[label] = {r[MK]: r for r in panel.to_dicts()}
        # D2 sx (for the D2-rule chain) where it exists; otherwise raw sx
        d2 = pl.read_parquet(ROOT / "logs" / "swarm2" / f"D2_rounds_{label}.parquet").select(
            [MK, "round", "sx"]).rename({"sx": "sx_d2_tab"})
        d2map = {(r[MK], r["round"]): r["sx_d2_tab"] for r in d2.to_dicts()}
        for mk, byr in M.items():
            for rd, r in byr.items():
                r["sx_d2"] = d2map.get((mk, rd))
        print(f"\n################ {label} ################")
        section_A(label, M)
        section_B(label, M)
        section_D(label, M, panel, hd)
        section_E(label, M)
        section_F(label, M)
        if label == "2026":
            section_C(label, M)


if __name__ == "__main__":
    main()
