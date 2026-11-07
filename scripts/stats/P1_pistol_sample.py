"""P1 part 2: pistol-round weapons, armor/helmets and money from the 1 Hz freeze-end sample.

Universe: data/state_sample/players_1hz.parquet (150 matches). Freeze-end state = first snapshot
(minimum tick) per (match_key, round). Pistol rounds = round 1 and round 13.
Side winner from data/state_sample/rounds.parquet ct_win (1 = CT won), cross-checked with
basic_round_end winner_team_code. Team codes: 2 = T, 3 = CT.

Pistol names are from basic_weapon_fire player_weapon_code -> weapon_name (see P1_verify_mapping log).
Default secondary: T = glock (4), CT = usp_silencer (61). inv_secondary 0 = no secondary.
Match-cluster bootstrap: 1000 draws over matches, seed 0.
"""
import glob
import os
from collections import Counter

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
SAMPLE = os.path.join(ROOT, "data", "state_sample", "players_1hz.parquet")
ROUNDS = os.path.join(ROOT, "data", "state_sample", "rounds.parquet")
BUYS = os.path.join(ROOT, "data", "state_sample", "round_buy_types.parquet")
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")

B = 1000
SEED = 0
T, CT = 2, 3
DEFAULT = {T: 4, CT: 61}
NAMES = {0: "none", 1: "deagle", 2: "elite (dual berettas)", 3: "fiveseven", 4: "glock",
         30: "tec9", 32: "hkp2000 (P2000)", 36: "p250", 61: "usp_silencer", 63: "cz75a",
         64: "revolver (R8)"}
PISTOL_ROUNDS = (1, 13)


def pname(code):
    if code == "tie":
        return "tie (no single modal)"
    return f"{code} {NAMES.get(code, 'UNMAPPED')}"


def load_first_snapshots():
    cols = ["match_key", "round", "tick", "seconds_since_live", "player_id", "team_code",
            "money", "equipment_value_calc", "armor", "has_helmet", "inv_secondary"]
    lf = pl.scan_parquet(SAMPLE).select(cols)
    t0 = lf.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("t0"))
    fs = lf.join(t0, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("t0")).collect()
    return fs


def load_round_end():
    folder = glob.glob(os.path.join(TOME, "basic_round_end.*"))[0]
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    return (pl.scan_parquet(files).select(["match_key", "round", "winner_team_code"]).collect())


def main():
    print("== Sample checks ==")
    lf = pl.scan_parquet(SAMPLE)
    print("sample rows:", lf.select(pl.len()).collect().item(),
          " matches:", lf.select(pl.col("match_key").n_unique()).collect().item())
    fs = load_first_snapshots()
    print("first-snapshot rows:", fs.height)
    grp = fs.group_by(["match_key", "round"]).agg(pl.len().alias("n"))
    print("(match,round) groups in first snapshot:", grp.height)
    print("players per first snapshot (n: count):", sorted(Counter(grp["n"].to_list()).items()))
    dup = fs.group_by(["match_key", "round", "player_id"]).len().filter(pl.col("len") > 1).height
    print("duplicate player rows within first snapshot:", dup)
    sl = fs["seconds_since_live"].to_numpy()
    print("seconds_since_live at first snapshot: min %s p50 %s max %s; distinct values %d" % (
        sl.min(), np.median(sl), sl.max(), len(np.unique(sl))))
    print("first-snapshot team_code counts:", sorted(Counter(fs["team_code"].to_list()).items()))
    fsr = fs.to_dicts()
    per_side = Counter()
    tmp = {}
    for x in fsr:
        tmp.setdefault((x["match_key"], x["round"]), Counter())[x["team_code"]] += 1
    for k, c in tmp.items():
        per_side[(c.get(T, 0), c.get(CT, 0))] += 1
    print("(n_T, n_CT) per (match,round), top 8:", per_side.most_common(8))

    rd = pl.read_parquet(ROUNDS).select(["match_key", "round", "ct_win"]).to_dicts()
    ct_win = {(x["match_key"], x["round"]): x["ct_win"] for x in rd}
    re_df = load_round_end().to_dicts()
    tome_win = {}
    for x in re_df:
        tome_win[(x["match_key"], x["round"])] = x["winner_team_code"]
    both = [k for k in ct_win if k in tome_win]
    agree = sum(1 for k in both if ((ct_win[k] == 1) == (tome_win[k] == CT)))
    print("rounds.parquet vs basic_round_end winner agreement: %d / %d" % (agree, len(both)))

    buys = pl.read_parquet(BUYS).to_dicts()
    bm = {(x["match_key"], x["round"]): x for x in buys}
    print("round_buy_types rows:", len(buys), " is_pistol rounds:",
          sorted(Counter(x["round"] for x in buys if x["is_pistol"]).items()))
    # money cross-check: side money sums from the sample vs round_buy_types
    money_sum = {}
    for x in fsr:
        money_sum[(x["match_key"], x["round"], x["team_code"])] = money_sum.get(
            (x["match_key"], x["round"], x["team_code"]), 0) + x["money"]
    ok_ct = ok_t = n_cmp = 0
    for (m, r), x in bm.items():
        ct_key, t_key = (m, r, CT), (m, r, T)
        if ct_key in money_sum and t_key in money_sum:
            n_cmp += 1
            ok_ct += int(money_sum[ct_key] == x["ct_money"])
            ok_t += int(money_sum[t_key] == x["t_money"])
    print("round_buy_types money-sum agreement (sample first snapshot): CT %d/%d, T %d/%d" % (
        ok_ct, n_cmp, ok_t, n_cmp))

    # ------------------------------------------------------------------
    matches = sorted(set(x["match_key"] for x in fsr))
    M = len(matches)
    midx_of = {m: i for i, m in enumerate(matches)}
    rng = np.random.default_rng(SEED)
    Wb = np.empty((B, M), dtype=np.float64)
    for b in range(B):
        Wb[b] = np.bincount(rng.integers(0, M, M), minlength=M)

    def ci_mean(label, midx, x):
        midx = np.asarray(midx)
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            print(f"{label:<60} n=0")
            return
        pt = float(x.mean())
        samp = np.empty(B)
        for b in range(B):
            w = Wb[b][midx]
            den = w.sum()
            samp[b] = (w * x).sum() / den if den > 0 else np.nan
        lo, hi = np.nanpercentile(samp, [2.5, 97.5])
        print(f"{label:<60} n_rows={len(x):>5} n_matches={len(np.unique(midx)):>4} "
              f"rate={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # ------------------------------------------------------------------
    # player-level pistol rows and side-level aggregates
    sides = {}
    for x in fsr:
        if x["round"] not in PISTOL_ROUNDS:
            continue
        key = (x["match_key"], x["round"], x["team_code"])
        sides.setdefault(key, []).append(x)

    side_rows = []
    for key, pl_list in sides.items():
        m, r, t = key
        n = len(pl_list)
        sec_codes = [p["inv_secondary"] for p in pl_list]
        dcode = DEFAULT[t]
        n_nondef = sum(1 for c in sec_codes if c != 0 and c != dcode)
        n_none = sum(1 for c in sec_codes if c == 0)
        cnt = Counter(sec_codes).most_common()
        if len(cnt) == 1 or cnt[0][1] > cnt[1][1]:
            modal = cnt[0][0]
        else:
            modal = "tie"
        winner = CT if ct_win[(m, r)] == 1 else T
        side_rows.append(dict(
            match=m, midx=midx_of[m], round=r, team=t, n=n, win=int(t == winner),
            sum_money=sum(p["money"] for p in pl_list),
            sum_equip=sum(p["equipment_value_calc"] for p in pl_list),
            n_armor=sum(1 for p in pl_list if p["armor"] > 0),
            n_helmet=sum(1 for p in pl_list if p["has_helmet"]),
            n_aoh=sum(1 for p in pl_list if (p["armor"] > 0) or p["has_helmet"]),
            n_nondef=n_nondef, n_none=n_none, modal=modal, codes=sec_codes,
        ))
    print("\n== Pistol side-rounds in sample:", len(side_rows),
          " (expected 2 per pistol round per match) ==")
    print("players per side-round:", sorted(Counter(s["n"] for s in side_rows).items()))
    for r in PISTOL_ROUNDS:
        rows_r = [s for s in side_rows if s["round"] == r]
        print(f"round {r}: side-rounds={len(rows_r)} matches={len(set(s['match'] for s in rows_r))}")

    # ------------------------------------------------------------------
    print("\n== 1. Most common secondary weapon per side (pistol rounds) ==")
    for r in PISTOL_ROUNDS:
        for t, tl in ((T, "T"), (CT, "CT")):
            rows_rt = [s for s in side_rows if s["round"] == r and s["team"] == t]
            cnt = Counter(s["modal"] for s in rows_rt)
            tot = len(rows_rt)
            print(f"round {r} {tl} (n sides={tot}): " + "; ".join(
                f"{pname(k)} {v} ({v / tot:.3f})" for k, v in cnt.most_common(6)))
            pcnt = Counter()
            for s in rows_rt:
                pcnt.update(s["codes"])
            ptot = sum(pcnt.values())
            print(f"   player-level secondary (n players={ptot}): " + "; ".join(
                f"{pname(k)} {v / ptot:.3f}" for k, v in pcnt.most_common(8)))

    print("\n== 1b. Non-default pistol purchases ==")
    for r in PISTOL_ROUNDS + ("pooled",):
        for t, tl in ((T, "T"), (CT, "CT")):
            rows_rt = [s for s in side_rows if (r == "pooled" or s["round"] == r) and s["team"] == t]
            n_pl = sum(s["n"] for s in rows_rt)
            nd_pl = sum(s["n_nondef"] for s in rows_rt)
            maj = [s for s in rows_rt if s["n_nondef"] >= 3]
            anyn = [s for s in rows_rt if s["n_nondef"] >= 1]
            modal_nd = [s for s in rows_rt if s["modal"] not in (DEFAULT[t], "tie")]
            print(f"round {str(r):>6} {tl}: player non-default share {nd_pl}/{n_pl}={nd_pl / n_pl:.3f}; "
                  f"sides with >=3 non-default {len(maj)}/{len(rows_rt)}={len(maj) / len(rows_rt):.3f}; "
                  f"sides with >=1 non-default {len(anyn)}/{len(rows_rt)}={len(anyn) / len(rows_rt):.3f}; "
                  f"modal non-default {len(modal_nd)}/{len(rows_rt)}={len(modal_nd) / len(rows_rt):.3f}")
    n_none_tot = sum(s["n_none"] for s in side_rows)
    print("players with no secondary (code 0) in pistol first snapshots:", n_none_tot)

    print("\n== 2. Win rate by side's modal secondary pistol (pooled rounds 1 and 13) ==")
    modal_types = sorted(set(s["modal"] for s in side_rows), key=lambda k: (-sum(1 for s in side_rows if s["modal"] == k)))
    for mt in modal_types:
        rows_mt = [s for s in side_rows if s["modal"] == mt]
        midx = [s["midx"] for s in rows_mt]
        x = [s["win"] for s in rows_mt]
        ci_mean(f"win rate, modal {pname(mt)} (pooled)", midx, x)
    print("  by team (modal type, T and CT, pooled rounds):")
    for t, tl in ((T, "T"), (CT, "CT")):
        for mt in modal_types:
            rows_mt = [s for s in side_rows if s["modal"] == mt and s["team"] == t]
            if not rows_mt:
                continue
            ci_mean(f"   {tl}: modal {pname(mt)}", [s["midx"] for s in rows_mt], [s["win"] for s in rows_mt])
    print("  by round (modal type, pooled sides):")
    for r in PISTOL_ROUNDS:
        for mt in modal_types:
            rows_mt = [s for s in side_rows if s["modal"] == mt and s["round"] == r]
            if rows_mt:
                ci_mean(f"   R{r}: modal {pname(mt)}", [s["midx"] for s in rows_mt], [s["win"] for s in rows_mt])

    # ------------------------------------------------------------------
    print("\n== 3. Armor and helmets at freeze end (pistol rounds) ==")
    for r in PISTOL_ROUNDS + ("pooled",):
        pl_rows = [p for p in fsr if p["round"] in PISTOL_ROUNDS and (r == "pooled" or p["round"] == r)]
        n = len(pl_rows)
        a = sum(1 for p in pl_rows if p["armor"] > 0) / n
        h = sum(1 for p in pl_rows if p["has_helmet"]) / n
        ah = sum(1 for p in pl_rows if (p["armor"] > 0) or p["has_helmet"]) / n
        full = sum(1 for p in pl_rows if p["armor"] >= 100) / n
        print(f"round {str(r):>6}: players={n}  armor>0 {a:.4f}  helmet {h:.4f}  armor-or-helmet {ah:.4f}  armor=100 {full:.4f}")

    print("side with more players holding armor-or-helmet (ties excluded), win rate of that side:")
    for r in PISTOL_ROUNDS + ("pooled",):
        rows_pair = {}
        for s in side_rows:
            if r != "pooled" and s["round"] != r:
                continue
            rows_pair.setdefault((s["match"], s["round"]), {})[s["team"]] = s
        for label, field in (("armor-or-helmet", "n_aoh"), ("armor only", "n_armor"), ("helmet only", "n_helmet")):
            midx, win, ties = [], [], 0
            for (m, rr), d in rows_pair.items():
                if T not in d or CT not in d:
                    continue
                a_, b_ = d[T][field], d[CT][field]
                if a_ == b_:
                    ties += 1
                    continue
                more = T if a_ > b_ else CT
                midx.append(midx_of[m])
                win.append(d[more]["win"])
            ci_mean(f"R{str(r):>6} more {label:<16} (ties {ties})", midx, win)

    print("halftime check: share with armor>0 / helmet / mean money by round index (first snapshot, all rounds 1-24):")
    for r in range(1, 25):
        pl_rows = [p for p in fsr if p["round"] == r]
        if not pl_rows:
            continue
        n = len(pl_rows)
        a = sum(1 for p in pl_rows if p["armor"] > 0) / n
        h = sum(1 for p in pl_rows if p["has_helmet"]) / n
        mm = sum(p["money"] for p in pl_rows) / n
        print(f"  R{r:>2}: n={n:>4} armor>0 {a:.3f} helmet {h:.3f} mean money {mm:8.1f}")

    # ------------------------------------------------------------------
    print("\n== 4. Money at freeze end (side total, pistol rounds) ==")
    qs = [10, 25, 50, 75, 90]
    for r in PISTOL_ROUNDS + ("pooled",):
        win_m = [s["sum_money"] for s in side_rows if (r == "pooled" or s["round"] == r) and s["win"] == 1]
        lose_m = [s["sum_money"] for s in side_rows if (r == "pooled" or s["round"] == r) and s["win"] == 0]
        all_m = win_m + lose_m
        def q(v):
            return " ".join(f"p{p}={np.percentile(v, p):.0f}" for p in qs)
        print(f"round {str(r):>6} all sides (n={len(all_m)}): {q(all_m)} mean={np.mean(all_m):.0f}")
        print(f"            winners   (n={len(win_m)}): {q(win_m)} mean={np.mean(win_m):.0f}")
        print(f"            losers    (n={len(lose_m)}): {q(lose_m)} mean={np.mean(lose_m):.0f}")
        ew = [s["sum_equip"] for s in side_rows if (r == "pooled" or s["round"] == r) and s["win"] == 1]
        el = [s["sum_equip"] for s in side_rows if (r == "pooled" or s["round"] == r) and s["win"] == 0]
        print(f"            equipment_value_calc side sum: winners mean={np.mean(ew):.0f} losers mean={np.mean(el):.0f}")

    print("richer side (higher total money) wins, ties excluded:")
    for r in PISTOL_ROUNDS + ("pooled",):
        for field, label in (("sum_money", "total money"), ("sum_equip", "equipment_value_calc sum")):
            rows_pair = {}
            for s in side_rows:
                if r != "pooled" and s["round"] != r:
                    continue
                rows_pair.setdefault((s["match"], s["round"]), {})[s["team"]] = s
            midx, win, ties = [], [], 0
            diffs = []
            for (m, rr), d in rows_pair.items():
                if T not in d or CT not in d:
                    continue
                a_, b_ = d[T][field], d[CT][field]
                if a_ == b_:
                    ties += 1
                    continue
                more = T if a_ > b_ else CT
                midx.append(midx_of[m])
                win.append(d[more]["win"])
                diffs.append(abs(a_ - b_))
            ci_mean(f"R{str(r):>6} richer side wins by {label:<26} (ties {ties})", midx, win)
            if field == "sum_money" and r == "pooled":
                print("   |money gap| quantiles (pooled, decisive pairs): " + " ".join(
                    f"p{p}={np.percentile(diffs, p):.0f}" for p in (25, 50, 75, 90)))

    print("win rate by own-minus-opponent money gap (pooled pistol side-rounds, quintile bins):")
    pair_money = {}
    for s in side_rows:
        pair_money.setdefault((s["match"], s["round"]), {})[s["team"]] = s
    gap_rows = []
    for (m, rr), d in pair_money.items():
        if T not in d or CT not in d:
            continue
        for t in (T, CT):
            own, opp = d[t], d[CT if t == T else T]
            gap_rows.append((own["sum_money"] - opp["sum_money"], own["win"], midx_of[m]))
    gaps = np.array([g[0] for g in gap_rows], dtype=float)
    wins_g = np.array([g[1] for g in gap_rows], dtype=float)
    mids_g = np.array([g[2] for g in gap_rows])
    edges = np.quantile(gaps, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(gaps, edges, right=False)
    print("  quintile edges (own minus opp money):", [round(float(e), 0) for e in edges])
    for k in range(5):
        sel = bins == k
        lo_e = "-inf" if k == 0 else f"{edges[k - 1]:.0f}"
        hi_e = "+inf" if k == 4 else f"{edges[k]:.0f}"
        ci_mean(f"  bin {k + 1} gap in [{lo_e}, {hi_e})", mids_g[sel], wins_g[sel])

    print("\n== Done ==")


if __name__ == "__main__":
    main()
