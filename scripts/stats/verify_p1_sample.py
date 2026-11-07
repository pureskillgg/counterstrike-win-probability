"""Independent verifier for P1 sections 2-4 (pistol secondaries, armor, money)
using the 1 Hz freeze-end sample, plus a code-to-name check from item_equip.

Reads only the columns it needs from players_1hz.parquet (lazy projection).
Uncertainty: match-cluster bootstrap over all sample matches, 1000 draws,
numpy default_rng(0), resampling matches (not rounds).

Appends to G:\\pureskill\\csds-cs2\\logs\\stats\\verify_p1.log (created by verify_p1_outcomes.py).
"""
import glob
import os
from collections import defaultdict

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SAMPLE = os.path.join(ROOT, "data", "state_sample")
LOG = os.path.join(ROOT, "logs", "stats", "verify_p1.log")
N_BOOT = 1000
SEED = 0
T_CODE, CT_CODE = 2, 3
GLOCK, USP = 4, 61
PISTOL_ROUNDS = (1, 13)


def log(msg=""):
    print(msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def boot_ratio(num_m, den_m, draws):
    return num_m[draws].sum(axis=1) / den_m[draws].sum(axis=1)


def report_ratio(label, num_m, den_m, draws):
    est = num_m.sum() / den_m.sum()
    boot = boot_ratio(num_m, den_m, draws)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    log(f"{label:<60} n={int(den_m.sum()):>5}  events={int(num_m.sum()):>5}  "
        f"rate={100*est:6.2f}%  CI {100*lo:6.2f} to {100*hi:6.2f}")
    return est


def main():
    log("")
    log("== verify_p1_sample.py ==")
    pq = os.path.join(SAMPLE, "players_1hz.parquet")
    cols = ["match_key", "round", "tick", "seconds_since_live", "split", "player_id",
            "team_code", "money", "equipment_value_calc", "armor", "has_helmet",
            "inv_secondary", "weapon_code"]
    players = pl.scan_parquet(pq).select(cols).collect()
    matches = sorted(players["match_key"].unique().to_list())
    M = len(matches)
    midx = {k: i for i, k in enumerate(matches)}
    log(f"players_1hz rows={players.height}  matches={M}  splits={players['split'].unique().to_list()}")
    log(f"rounds present (distinct) min={int(players['round'].min())} max={int(players['round'].max())}")
    log(f"team_code counts: {players['team_code'].value_counts().sort('team_code').rows()}")
    log(f"armor values (top): {players['armor'].value_counts().sort('count', descending=True).head(6).rows()}")

    # First snapshot per (match, round) = min tick
    tmin = players.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("tmin"))
    first = players.join(tmin, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("tmin"))
    per_round_players = first.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("player_id").n_unique().alias("nu"))
    log(f"first-snapshot rows={first.height}; rounds={per_round_players.height}; "
        f"rounds with 10 players={int((per_round_players['n']==10).sum())}; "
        f"rounds with 10 distinct players={int((per_round_players['nu']==10).sum())}")
    log(f"first-snapshot seconds_since_live: min={first['seconds_since_live'].min()} max={first['seconds_since_live'].max()}")
    odd = per_round_players.filter(pl.col("n") != 10)
    log(f"rounds whose first snapshot is not 10 players: {odd.rows()}")

    # Rounds metadata and outcomes
    rounds = pl.read_parquet(os.path.join(SAMPLE, "rounds.parquet"),
                             columns=["match_key", "round", "ct_win", "round_number"])
    log(f"rounds.parquet rows={rounds.height}; round==round_number in "
        f"{int((rounds['round']==rounds['round_number']).sum())} rows")
    ctw = {(r[0], r[1]): r[2] for r in rounds.select(["match_key", "round", "ct_win"]).iter_rows()}

    # Cross-check ct_win against tome round_end
    re_files = sorted(glob.glob(os.path.join(TOME, "basic_round_end.*", "dataframe_*")))
    re_df = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code"]) for f in re_files]).collect()
    re_map = {(r[0], r[1]): (1 if r[2] == CT_CODE else 0) for r in re_df.iter_rows()}
    agree = 0
    disagree = 0
    missing = 0
    for key, cw in ctw.items():
        if key not in re_map:
            missing += 1
        elif re_map[key] == cw:
            agree += 1
        else:
            disagree += 1
    log(f"ct_win (rounds.parquet) vs tome round_end winner: agree={agree} disagree={disagree} missing_in_tome={missing}")

    # Cross-check first snapshot tick vs round_freeze_end in tome round_state
    rs_files = sorted(glob.glob(os.path.join(TOME, "basic_round_state.*", "dataframe_*")))
    rs = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type"]) for f in rs_files]).collect()
    log(f"round_state event types: {rs['event_type'].unique().to_list()}")
    fe = rs.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("fe_tick"))
    chk = tmin.join(fe, on=["match_key", "round"], how="inner")
    eq = int((chk["tmin"] == chk["fe_tick"]).sum())
    log(f"first snapshot tick == round_freeze_end tick: {eq} of {chk.height} matched rounds (sample rounds={tmin.height})")

    # Code-to-name check: player_weapon_code vs weapon_name in weapon_fire, and vs item in item_equip
    pistol_codes = [0, 1, 2, 3, 4, 30, 32, 36, 61, 63, 64]
    for chan, namecol, folder in [("weapon_fire", "weapon_name", "basic_weapon_fire.*"),
                                  ("item_equip", "item", "basic_item_equip.*")]:
        fs = sorted(glob.glob(os.path.join(TOME, folder, "dataframe_*")))
        tab = (pl.concat([pl.scan_parquet(f).select(["player_weapon_code", namecol]) for f in fs])
               .group_by(["player_weapon_code", namecol]).agg(pl.len().alias("n")).collect())
        log(f"{chan}: player_weapon_code -> most common {namecol} (share of that code):")
        for dcode in pistol_codes:
            sub = tab.filter(pl.col("player_weapon_code") == dcode).sort("n", descending=True)
            if sub.height == 0:
                log(f"  {dcode:>3}: no rows")
                continue
            tot = int(sub["n"].sum())
            top = sub.row(0)
            log(f"  {dcode:>3}: {top[1]}  share={100*top[2]/tot:.2f}%  (rows={tot})")

    sec_vals = players["inv_secondary"].value_counts().sort("count", descending=True)
    log(f"inv_secondary values (top 12): {sec_vals.head(12).rows()}")

    # Build side table for pistol rounds and all rounds from first snapshot
    rows = first.select(["match_key", "round", "team_code", "player_id", "armor",
                         "has_helmet", "money", "equipment_value_calc", "inv_secondary"]).rows()
    sides = defaultdict(list)
    for mk, rd, tc, pid, arm, hel, mon, eq_v, sec in rows:
        sides[(mk, rd, tc)].append((pid, arm, hel, mon, eq_v, sec))

    side_rec = {}
    for (mk, rd, tc), plist in sides.items():
        secs = [p[5] for p in plist]
        counts = defaultdict(int)
        for s in secs:
            counts[s] += 1
        top_n = max(counts.values())
        modes = [s for s, c in counts.items() if c == top_n]
        modal = modes[0] if len(modes) == 1 else None
        side_rec[(mk, rd, tc)] = dict(
            n=len(plist),
            n_arm=sum(1 for p in plist if p[1] > 0),
            n_hel=sum(1 for p in plist if p[2]),
            money=sum(p[3] for p in plist),
            equip=sum(p[4] for p in plist),
            modal=modal,
            modal_share=top_n,
            secs=secs,
            n_nondefault=sum(1 for s in secs if s != (GLOCK if tc == T_CODE else USP)),
            has_tec9=any(s == 30 for s in secs),
            has_deagle=any(s == 1 for s in secs),
        )

    round_keys = sorted({(mk, rd) for (mk, rd, _) in side_rec})
    log("")
    log(f"sample rounds with both sides: {len(round_keys)}")

    # Section 3 headline: armor
    def arm_block(rd_filter, label):
        num = np.zeros(M)
        den = np.zeros(M)
        ties = 0
        n_rounds = 0
        arm_share_players = [0, 0]
        for mk, rd in round_keys:
            if not rd_filter(rd):
                continue
            n_rounds += 1
            c = side_rec.get((mk, rd, CT_CODE))
            t = side_rec.get((mk, rd, T_CODE))
            if c is None or t is None:
                continue
            if c["n_arm"] == t["n_arm"]:
                ties += 1
                continue
            ct_more = c["n_arm"] > t["n_arm"]
            win = (ctw[(mk, rd)] == 1) if ct_more else (ctw[(mk, rd)] == 0)
            num[midx[mk]] += 1.0 if win else 0.0
            den[midx[mk]] += 1.0
        return num, den, ties, n_rounds

    draws = np.random.default_rng(SEED).integers(0, M, size=(N_BOOT, M))
    log("")
    log("-- Section 3: armor (first snapshot, side with more armored players wins) --")
    for label, filt in [("R1", lambda r: r == 1), ("R13", lambda r: r == 13), ("R1 and R13 pooled", lambda r: r in PISTOL_ROUNDS)]:
        num, den, ties, n_rounds = arm_block(filt, label)
        est = report_ratio(f"armored-side wins, {label}", num, den, draws)
        log(f"   rounds={n_rounds}  decisive={int(den.sum())}  ties={ties}")

    # Armored share and helmets by round
    log("")
    log("-- Armor and helmet share among players at freeze end --")
    for label, filt in [("R1", lambda r: r == 1), ("R13", lambda r: r == 13), ("R12", lambda r: r == 12), ("R14", lambda r: r == 14), ("R1 and R13", lambda r: r in PISTOL_ROUNDS)]:
        n_p = 0
        n_a = 0
        n_h = 0
        for (mk, rd, tc), rec in side_rec.items():
            if filt(rd):
                n_p += rec["n"]
                n_a += rec["n_arm"]
                n_h += rec["n_hel"]
        log(f"  {label:<12} players={n_p:>5}  armor>0={100*n_a/n_p:6.2f}%  helmet={100*n_h/n_p:6.2f}%")

    # Section 2: pistol secondaries
    log("")
    log("-- Section 2: pistol secondaries (modal per side, 1 Hz sample) --")
    for rd in PISTOL_ROUNDS:
        for tc, label, default in [(T_CODE, "T", GLOCK), (CT_CODE, "CT", USP)]:
            n_side = 0
            n_default = 0
            n_tied = 0
            for (mk, r, t), rec in side_rec.items():
                if r == rd and t == tc:
                    n_side += 1
                    if rec["modal"] is None:
                        n_tied += 1
                    elif rec["modal"] == default:
                        n_default += 1
            log(f"  R{rd} {label} modal default ({default}): {n_default}/{n_side} = {100*n_default/n_side:.1f}%  (tied modes={n_tied})")

    # Non-default purchases
    log("")
    log("-- Non-default shares, pistol rounds (R1 + R13) --")
    for tc, label, default in [(T_CODE, "T", GLOCK), (CT_CODE, "CT", USP)]:
        n_pl = 0
        n_nd = 0
        n_sides = 0
        n_any = 0
        n_three = 0
        for (mk, r, t), rec in side_rec.items():
            if r in PISTOL_ROUNDS and t == tc:
                n_sides += 1
                n_pl += rec["n"]
                nd = sum(1 for s in rec["secs"] if s != default)
                n_nd += nd
                if nd >= 1:
                    n_any += 1
                if nd >= 3:
                    n_three += 1
        log(f"  {label}: non-default players {n_nd}/{n_pl} = {100*n_nd/n_pl:.1f}%; "
            f"sides with >=1 non-default {n_any}/{n_sides} = {100*n_any/n_sides:.1f}%; "
            f">=3 non-default {n_three}/{n_sides} = {100*n_three/n_sides:.1f}%")

    # Sample CT win rates in pistol rounds (rounds.parquet ct_win)
    for rd in PISTOL_ROUNDS:
        vals = [v for (mk, r), v in ctw.items() if r == rd]
        log(f"sample CT win R{rd}: {sum(vals)}/{len(vals)} = {100*sum(vals)/len(vals):.1f}%")

    # Handling of players with no secondary (inv_secondary == 0) and armor values at freeze end
    log("")
    log("-- Checks: players with inv_secondary==0 and freeze-end armor values (R1+R13) --")
    zero_by_team = {T_CODE: 0, CT_CODE: 0}
    armor_vals = defaultdict(int)
    for (mk, r, t), rec in side_rec.items():
        if r not in PISTOL_ROUNDS:
            continue
        zero_by_team[t] += sum(1 for s in rec["secs"] if s == 0)
        for p in sides[(mk, r, t)]:
            if p[1] > 0:
                armor_vals[p[1]] += 1
    log(f"  players with inv_secondary==0: T={zero_by_team[T_CODE]}, CT={zero_by_team[CT_CODE]}")
    log(f"  freeze-end armor>0 values (R1+R13): {sorted(armor_vals.items())}")
    for tc, label, default in [(T_CODE, "T", GLOCK), (CT_CODE, "CT", USP)]:
        n_nd_ex0 = 0
        sides_any = 0
        sides_three = 0
        n_sides = 0
        for (mk, r, t), rec in side_rec.items():
            if r in PISTOL_ROUNDS and t == tc:
                n_sides += 1
                nd = sum(1 for s in rec["secs"] if s != default and s != 0)
                n_nd_ex0 += nd
                if nd >= 1:
                    sides_any += 1
                if nd >= 3:
                    sides_three += 1
        log(f"  {label} excluding zeros: non-default players {n_nd_ex0}/1350 = {100*n_nd_ex0/1350:.2f}%; "
            f"sides >=1 {sides_any}/{n_sides} = {100*sides_any/n_sides:.1f}%; "
            f"sides >=3 {sides_three}/{n_sides} = {100*sides_three/n_sides:.1f}%")

    # Win rates by modal type (side win rate, bootstrap by match)
    log("")
    log("-- Pistol side win rates by modal type --")
    for rd_set, label in [((1,), "R1"), ((13,), "R13"), (PISTOL_ROUNDS, "R1+R13")]:
        num_g = np.zeros(M)
        den_g = np.zeros(M)
        num_u = np.zeros(M)
        den_u = np.zeros(M)
        for (mk, rd, t), rec in side_rec.items():
            if rd not in rd_set:
                continue
            if t == T_CODE and rec["modal"] == GLOCK:
                den_g[midx[mk]] += 1
                num_g[midx[mk]] += 1 if ctw[(mk, rd)] == 0 else 0
            if t == CT_CODE and rec["modal"] == USP:
                den_u[midx[mk]] += 1
                num_u[midx[mk]] += 1 if ctw[(mk, rd)] == 1 else 0
        report_ratio(f"T glock-modal side wins, {label}", num_g, den_g, draws)
        report_ratio(f"CT usp-modal side wins, {label}", num_u, den_u, draws)

    # Within-team type comparisons (any player with the type vs without), R1+R13
    log("")
    log("-- Within-team comparisons, any player with type vs none (R1+R13) --")
    for tc, label, typ_key, typ_name in [(T_CODE, "T", "has_tec9", "tec9"), (CT_CODE, "CT", "has_deagle", "deagle")]:
        num_y = np.zeros(M)
        den_y = np.zeros(M)
        num_n = np.zeros(M)
        den_n = np.zeros(M)
        for (mk, rd, t), rec in side_rec.items():
            if rd not in PISTOL_ROUNDS or t != tc:
                continue
            won = (ctw[(mk, rd)] == 1) if tc == CT_CODE else (ctw[(mk, rd)] == 0)
            if rec[typ_key]:
                den_y[midx[mk]] += 1
                num_y[midx[mk]] += 1 if won else 0
            else:
                den_n[midx[mk]] += 1
                num_n[midx[mk]] += 1 if won else 0
        n_y = int(den_y.sum())
        rate_y = num_y.sum() / den_y.sum() if den_y.sum() else float("nan")
        rate_n = num_n.sum() / den_n.sum() if den_n.sum() else float("nan")
        diff = 100 * (rate_y - rate_n)
        bdiff = 100 * (boot_ratio(num_y, den_y, draws) - boot_ratio(num_n, den_n, draws))
        lo, hi = np.nanpercentile(bdiff, [2.5, 97.5])
        log(f"  {label} {typ_name}: sides with={n_y}, win={100*rate_y:.1f}%; without={int(den_n.sum())}, win={100*rate_n:.1f}%; "
            f"diff={diff:+.1f} CI {lo:+.1f} to {hi:+.1f}")

    # Section 4: money and equipment at freeze end (side totals)
    log("")
    log("-- Section 4: money at freeze end (side totals, R1 and R13) --")
    for rd in PISTOL_ROUNDS:
        win_m, lose_m = [], []
        for (mk, r, t), rec in side_rec.items():
            if r != rd:
                continue
            won = (ctw[(mk, rd)] == 1) if t == CT_CODE else (ctw[(mk, rd)] == 0)
            (win_m if won else lose_m).append(rec["money"])
        for label, arr in [("winners", win_m), ("losers", lose_m)]:
            a = np.array(arr, dtype=float)
            pcts = np.percentile(a, [10, 25, 50, 75, 90])
            log(f"  R{rd} {label:<8} n={len(a)}  p10={pcts[0]:.0f} p25={pcts[1]:.0f} p50={pcts[2]:.0f} "
                f"p75={pcts[3]:.0f} p90={pcts[4]:.0f} mean={a.mean():.0f}")

    money_num = np.zeros(M)
    money_den = np.zeros(M)
    equip_num = np.zeros(M)
    equip_den = np.zeros(M)
    money_gap = []
    for mk, rd in round_keys:
        if rd not in PISTOL_ROUNDS:
            continue
        c = side_rec.get((mk, rd, CT_CODE))
        t = side_rec.get((mk, rd, T_CODE))
        if c is None or t is None:
            continue
        mi = midx[mk]
        if c["money"] != t["money"]:
            ct_rich = c["money"] > t["money"]
            win = (ctw[(mk, rd)] == 1) if ct_rich else (ctw[(mk, rd)] == 0)
            money_num[mi] += 1 if win else 0
            money_den[mi] += 1
            money_gap.append((abs(c["money"] - t["money"]), 1 if win else 0, ct_rich))
        if c["equip"] != t["equip"]:
            ct_rich = c["equip"] > t["equip"]
            win = (ctw[(mk, rd)] == 1) if ct_rich else (ctw[(mk, rd)] == 0)
            equip_num[mi] += 1 if win else 0
            equip_den[mi] += 1
    n_ties_money = sum(1 for mk, rd in round_keys if rd in PISTOL_ROUNDS and side_rec.get((mk, rd, CT_CODE)) and side_rec.get((mk, rd, T_CODE)) and side_rec[(mk, rd, CT_CODE)]["money"] == side_rec[(mk, rd, T_CODE)]["money"])
    report_ratio("richer-by-remaining-cash side wins (R1+R13)", money_num, money_den, draws)
    log(f"   money ties={n_ties_money}")
    report_ratio("richer-by-equipment side wins (R1+R13)", equip_num, equip_den, draws)

    # Gap quintiles for cash: rate the poorer side wins
    if money_gap:
        gaps = np.array([g[0] for g in money_gap], dtype=float)
        # g[1] is 1 when the richer side won, so the poorer side wins when g[1] is 0.
        poorer_won = np.array([1.0 - g[1] for g in money_gap], dtype=float)
        qs = np.quantile(gaps, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(gaps, qs)
        log("   gap quintile (poorer-side win rate, cash):")
        for q in range(5):
            sel = bins == q
            if sel.sum() == 0:
                continue
            log(f"     Q{q+1}: n={int(sel.sum())}  poorer side wins {100*poorer_won[sel].mean():.1f}%")


if __name__ == "__main__":
    main()
