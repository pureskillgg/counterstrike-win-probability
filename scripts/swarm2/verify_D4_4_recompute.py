r"""verify_D4_4 independent recompute of claim D4-4 (2024 round-13 side defect, header reconciliation).

Sources read directly from the tomes (no claim script is imported):
  header        t_starters_score_final, ct_starters_score_final  (match final totals)
  round_end     winner_team_code, win_reason_code                 (engine round-end record)
  bomb_state    bomb_planted / bomb_defused / bomb_exploded       (bomb outcome side: defused->CT, exploded->T)
  round_state   t_score / ct_score at round_freeze_end and round_officially_ended (running engine score)
  player_spawn  roster per round (player_id, side)
  player_death  deaths per round (player_id, tick, victim side)  (elimination proxy, no bomb rows needed)

ASSUMPTIONS (marked in the log):
  A1 regulation = n <= 24 rounds; halftime swap after round (ht - 1), where ht = first round whose
     round_freeze_end phase label is 'second' (phase-derived, NOT assumed). Rounds with no 'second'
     phase (match ended before halftime) use HALF = 12 (irrelevant when n <= 12).
  A2 complete = round_end rounds are exactly 1..n (unique per round) and n == header total.
  A3 team A = header t_starters starts on T; A is on CT after the halftime swap.
  A4 state winner: within one round, the side whose running score rose by exactly 1 won; if the phase
     label changes inside the round (halftime), the displayed scores are swapped.
  A5 round_end duplicates: earliest row by tick is used.
  A6 elimination proxy: if every spawned player of one side died (tick <= round_end tick), the other
     side is the proxy winner; if no side is eliminated the proxy is undefined (time-out or bomb).
  A7 'affected' (claim definition): complete regulation match with a round-13 bomb outcome whose side
     differs from round_end winner. Claim day grouping uses the match_key path day (csds/2024/02/DD/).

Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_4_recompute.py
"""
import glob
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}
T, CT = 2, 3


def chan_dir(base, ch):
    if ch == "header":
        return base / "header"
    ds = sorted(base.glob(f"basic_{ch}.*"))
    assert len(ds) == 1, (ch, ds)
    return ds[0]


def read_ch(base, ch, cols, keep=None):
    parts = []
    for f in sorted(glob.glob(str(chan_dir(base, ch) / "dataframe_*"))):
        have = set(pl.read_parquet_schema(f).keys())
        df = pl.read_parquet(f, columns=[c for c in cols if c in have])
        if keep is not None:
            df = df.filter(keep)
        parts.append(df)
    return pl.concat(parts, how="vertical_relaxed")


def day_of(key):
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", key)
    return m.group(3) if m else "??"


def state_winner(pre, post):
    if pre is None or post is None:
        return None
    pt, pc, pp, _ = pre
    qt, qc, qp, _ = post
    if None in (pt, pc, qt, qc):
        return None
    if pp == qp:
        if qt == pt + 1 and qc == pc:
            return T
        if qc == pc + 1 and qt == pt:
            return CT
        return None
    if qc == pt + 1 and qt == pc:  # halftime inside round: displayed scores swapped
        return T
    if qt == pc + 1 and qc == pt:
        return CT
    return None


def reconcile(winners, n, tA, tB, half):
    wa = wb = 0
    side_A = T
    for r in range(1, n + 1):
        w = winners[r]
        if w == side_A:
            wa += 1
        else:
            wb += 1
        if r == half:
            side_A = CT if side_A == T else T
    if (wa, wb) == (tA, tB):
        return "ok"
    if (wa, wb) == (tB, tA):
        return "swapped"
    return "bad"


def load_season(season):
    base = TOMES[season]
    H = read_ch(base, "header", ["key", "t_starters_score_final", "ct_starters_score_final", "max_rounds",
                                 "match_type", "match_date"])
    hdr = {r["key"]: r for r in H.iter_rows(named=True)}
    RE = read_ch(base, "round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "tick"])
    BS = read_ch(base, "bomb_state", ["match_key", "round", "event_type", "tick"],
                 keep=pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
    RS = read_ch(base, "round_state", ["match_key", "round", "event_type", "phase", "t_score", "ct_score", "tick"],
                 keep=pl.col("event_type").is_in(["round_freeze_end", "round_officially_ended"]))
    PS = read_ch(base, "player_spawn", ["match_key", "round", "player_id", "player_team_code"],
                 keep=pl.col("player_team_code").is_in([T, CT]))
    PD = read_ch(base, "player_death", ["match_key", "round", "player_id", "player_team_code", "tick"],
                 keep=pl.col("player_team_code").is_in([T, CT]))
    return hdr, RE, BS, RS, PS, PD


def build(season):
    hdr, RE, BS, RS, PS, PD = load_season(season)
    re_rows = defaultdict(list)
    for r in RE.sort("tick").iter_rows(named=True):
        re_rows[(r["match_key"], r["round"])].append((r["winner_team_code"], r["win_reason_code"], r["tick"]))
    bomb = defaultdict(set)
    planted = set()
    for r in BS.iter_rows(named=True):
        k = (r["match_key"], r["round"])
        if r["event_type"] == "bomb_planted":
            planted.add(k)
        else:
            bomb[k].add(r["event_type"])
    fz, oe = {}, {}
    for r in RS.sort("tick").iter_rows(named=True):
        k = (r["match_key"], r["round"])
        val = (r["t_score"], r["ct_score"], r["phase"], r["tick"])
        if r["event_type"] == "round_freeze_end":
            fz.setdefault(k, val)
        else:
            oe.setdefault(k, val)
    roster = defaultdict(lambda: {T: set(), CT: set()})
    for r in PS.iter_rows(named=True):
        roster[(r["match_key"], r["round"])][r["player_team_code"]].add(r["player_id"])
    deaths = defaultdict(dict)
    for r in PD.sort("tick").iter_rows(named=True):
        deaths[(r["match_key"], r["round"])].setdefault(r["player_id"], (r["tick"], r["player_team_code"]))
    M = defaultdict(dict)
    for (m, rnd), lst in re_rows.items():
        M[m][rnd] = lst
    # halftime from phase labels
    ht = {}
    for (m, rnd), (t, c, ph, tk) in fz.items():
        if ph == "second":
            ht[m] = min(ht.get(m, 10**6), rnd)
    return hdr, M, bomb, planted, fz, oe, roster, deaths, ht


def elim_proxy(roster_sides, deaths_round, end_tick):
    dead_ok = {}
    for side in (T, CT):
        players = roster_sides.get(side, set())
        if not players:
            dead_ok[side] = None
            continue
        all_dead = all((pid in deaths_round and deaths_round[pid][0] <= end_tick) for pid in players)
        dead_ok[side] = all_dead
    t_dead, ct_dead = dead_ok.get(T), dead_ok.get(CT)
    if ct_dead and not t_dead:
        return T
    if t_dead and not ct_dead:
        return CT
    return None


def analyse(season):
    hdr, M, bomb, planted, fz, oe, roster, deaths, ht_map = build(season)
    print(f"\n==================== SEASON {season} ====================")
    print(f"header matches={len(hdr)}; matches with round_end={len(M)}; keys not in header="
          f"{sum(1 for m in M if m not in hdr)}; (match,round) with >1 round_end row="
          f"{sum(1 for m in M for r in M[m] if len(M[m][r]) > 1)}")

    recs = {}
    for m, rnds in M.items():
        h = hdr.get(m)
        n = len(rnds)
        rec = {"n": n, "day": day_of(m), "ht": ht_map.get(m), "rounds": {}}
        rec["seq_ok"] = sorted(rnds) == list(range(1, n + 1))
        if h is not None:
            rec["tA"] = int(h["t_starters_score_final"])
            rec["tB"] = int(h["ct_starters_score_final"])
            rec["complete"] = rec["seq_ok"] and n == rec["tA"] + rec["tB"]
        else:
            rec["complete"] = False
        rec["half"] = (rec["ht"] - 1) if rec["ht"] is not None else 12
        for r in range(1, n + 1):
            lst = rnds.get(r, [])
            end_tick = lst[0][2] if lst else None
            ev = bomb.get((m, r), set())
            bs = None
            conflict = False
            if ev:
                bs = CT if "bomb_defused" in ev else T
                conflict = len({"bomb_defused", "bomb_exploded"} & ev) == 2
            rs = roster.get((m, r))
            ep = elim_proxy(rs, deaths.get((m, r), {}), end_tick) if (rs and end_tick is not None) else None
            rec["rounds"][r] = {
                "re": lst[0][0] if lst else None,
                "reason": lst[0][1] if lst else None,
                "bomb": bs, "bomb_conflict": conflict, "planted": (m, r) in planted,
                "state": state_winner(fz.get((m, r)), oe.get((m, r))),
                "elim": ep,
            }
        recs[m] = rec

    comp = [m for m, x in recs.items() if x["complete"]]
    reg = [m for m in comp if recs[m]["n"] <= 24]
    print(f"complete (rounds 1..n, n==header total): {len(comp)}; regulation n<=24: {len(reg)}; "
          f"overtime n>24: {len(comp) - len(reg)}")
    print("halftime format of regulation complete matches (ht-1 = swap after round):")
    print("  ", Counter((recs[m]["ht"] - 1) if recs[m]["ht"] else "no second phase" for m in reg))

    aff = set()
    for m in reg:
        d13 = recs[m]["rounds"].get(13)
        if d13 and d13["bomb"] is not None and d13["bomb"] != d13["re"]:
            aff.add(m)
    unaff = set(reg) - aff
    print(f"claim-style affected (r13 bomb side != round_end): {len(aff)}; unaffected: {len(unaff)}")

    def win_for(m, src):
        x = recs[m]
        out = {}
        for r in range(1, x["n"] + 1):
            d = x["rounds"][r]
            if src == "RE":
                out[r] = d["re"]
            elif src == "BOMB":
                out[r] = d["bomb"] if d["bomb"] is not None else d["re"]
            elif src == "STATE":
                out[r] = d["state"] if d["state"] is not None else d["re"]
        return out

    print("\nheader reconciliation with per-match halftime (A1), regulation complete:")
    print(f"  {'group':<11}{'source':<8}{'n':>6}{'ok':>7}{'swapped':>9}{'bad':>7}{'ok_rate':>9}")
    for grp, ms in (("affected", aff), ("unaffected", unaff), ("all_reg", set(reg))):
        for src in ("RE", "BOMB", "STATE"):
            c = Counter(reconcile(win_for(m, src), recs[m]["n"], recs[m]["tA"], recs[m]["tB"], recs[m]["half"])
                        for m in ms)
            tot = sum(c.values())
            print(f"  {grp:<11}{src:<8}{tot:>6}{c['ok']:>7}{c['swapped']:>9}{c['bad']:>7}"
                  f"{c['ok'] / max(tot, 1):>9.3f}")

    # round-level agreement by position relative to halftime
    print("\nround_end vs round_state winner, regulation complete, rounds split by position:")
    pos = defaultdict(lambda: [0, 0])
    for m in reg:
        x = recs[m]
        for r in range(1, x["n"] + 1):
            d = x["rounds"][r]
            if d["re"] is None or d["state"] is None:
                continue
            if x["ht"] is None:
                key = "no_ht"
            elif r == x["ht"]:
                key = "first_post_halftime_round"
            elif r < x["ht"]:
                key = "pre_halftime_rounds"
            else:
                key = "later_post_halftime_rounds"
            pos[(key, x["ht"])][0] += 1
            pos[(key, x["ht"])][1] += int(d["re"] == d["state"])
    for k in sorted(pos, key=lambda z: (z[0], z[1] or 0)):
        n_, ag = pos[k]
        print(f"  {k[0]:<28} ht={k[1]!s:<5} agree {ag}/{n_}  disagree {n_ - ag}")

    print("\nper-round RE vs state disagreement (regulation complete), rounds with disagreement only:")
    per_r = defaultdict(lambda: [0, 0])
    for m in reg:
        x = recs[m]
        for r in range(1, x["n"] + 1):
            d = x["rounds"][r]
            if d["re"] is None or d["state"] is None:
                continue
            per_r[(x["ht"], r)][0] += 1
            per_r[(x["ht"], r)][1] += int(d["re"] != d["state"])
    print("  (ht, round): disagree/total  -- only cells with disagreement shown")
    shown = {k: v for k, v in per_r.items() if v[1] > 0}
    for k in sorted(shown, key=lambda z: (z[0] or 0, z[1])):
        print(f"   ht={k[0]} r{k[1]}: {shown[k][1]}/{shown[k][0]}")

    # round 13 (and first post-halftime round) cross tab with bomb / elimination proxy
    print("\nfirst post-halftime round, bomb outcome and elimination proxy vs round_end and state:")
    xt = Counter()
    for m in reg:
        x = recs[m]
        if x["ht"] is None:
            continue
        r = x["ht"]
        if r not in x["rounds"]:
            continue
        d = x["rounds"][r]
        if d["re"] is None or d["state"] is None:
            continue
        b = "bomb_outcome_row" if d["bomb"] is not None else ("planted_no_outcome" if d["planted"] else "no_plant")
        e = "elim_none" if d["elim"] is None else ("elim==RE" if d["elim"] == d["re"] else "elim!=RE")
        es = "" if d["elim"] is None else ("elim==state" if d["elim"] == d["state"] else "elim!=state")
        xt[(f"ht={x['ht']}", b, "RE==state" if d["re"] == d["state"] else "RE!=state", e, es)] += 1
    for k, v in sorted(xt.items()):
        print(f"  {k}: {v}")

    # bomb-row evidence: among any round with a bomb outcome, which source agrees with the bomb side
    print("\nbomb-outcome rounds (regulation complete, all rounds): bomb side vs round_end and vs state")
    bc = Counter()
    for m in reg:
        x = recs[m]
        for r in range(1, x["n"] + 1):
            d = x["rounds"][r]
            if d["bomb"] is None or d["re"] is None:
                continue
            pos_lab = "ht_round" if r == x["ht"] else ("r13" if r == 13 else "other")
            bc[(pos_lab, f"RE==bomb:{d['re'] == d['bomb']}",
                f"state==bomb:{(d['state'] == d['bomb']) if d['state'] is not None else 'state_missing'}")] += 1
    for k, v in sorted(bc.items(), key=lambda z: str(z[0])):
        print(f"  {k}: {v}")

    # elimination proxy validity on rounds where RE == state (ground truth proxy)
    print("\nelimination proxy accuracy where round_end == state (regulation complete, rounds with proxy):")
    acc = Counter()
    for m in reg:
        x = recs[m]
        for r in range(1, x["n"] + 1):
            d = x["rounds"][r]
            if d["re"] is None or d["state"] is None or d["elim"] is None or d["bomb"] is not None:
                continue
            if d["re"] == d["state"]:
                acc["RE==state_and_elim==RE" if d["elim"] == d["re"] else "RE==state_but_elim!=RE"] += 1
    print("  ", dict(acc))
    # now the proxy against RE vs state on first-post-halftime rounds with no bomb outcome
    acc2 = Counter()
    for m in reg:
        x = recs[m]
        if x["ht"] is None or x["ht"] not in x["rounds"]:
            continue
        d = x["rounds"][x["ht"]]
        if d["re"] is None or d["state"] is None or d["elim"] is None or d["bomb"] is not None:
            continue
        acc2[f"ht_round_no_bomb: elim==RE:{d['elim'] == d['re']} elim==state:{d['elim'] == d['state']}"] += 1
    print("  first post-halftime round, no bomb outcome, proxy defined:")
    for k, v in sorted(acc2.items()):
        print(f"    {k}: {v}")

    # day breakdown of first-post-halftime disagreement (regulation complete)
    print("\nday breakdown, regulation complete: first post-halftime round RE vs state disagreement")
    byday = defaultdict(lambda: [0, 0, 0])
    for m in reg:
        x = recs[m]
        if x["ht"] is None or x["ht"] not in x["rounds"]:
            continue
        d = x["rounds"][x["ht"]]
        if d["re"] is None or d["state"] is None:
            continue
        byday[x["day"]][0] += 1
        byday[x["day"]][1] += int(d["re"] != d["state"])
        byday[x["day"]][2] += int(x["ht"] == 13)
    for dday in sorted(byday):
        n_, dis, mr12 = byday[dday]
        print(f"  day {dday}: n={n_} disagree={dis} (MR12 among these: {mr12})")

    return {"recs": recs, "aff": aff, "unaff": unaff, "reg": reg, "comp": comp}


def control(season):
    hdr, M, bomb, planted, fz, oe, roster, deaths, ht_map = build(season)
    print(f"\n==================== CONTROL {season} (n<=24, per-match halftime) ====================")
    c = {"RE": Counter(), "BOMB": Counter(), "STATE": Counter()}
    fmt = Counter()
    for m, rnds in M.items():
        h = hdr.get(m)
        if h is None:
            continue
        n = len(rnds)
        tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
        if sorted(rnds) != list(range(1, n + 1)) or n != tA + tB or n > 24:
            continue
        half = (ht_map[m] - 1) if m in ht_map else 12
        fmt[half] += 1
        w = {"RE": {}, "BOMB": {}, "STATE": {}}
        for r in range(1, n + 1):
            re_w = rnds[r][0][0]
            ev = bomb.get((m, r), set())
            bs = (CT if "bomb_defused" in ev else T) if ev else None
            st = state_winner(fz.get((m, r)), oe.get((m, r)))
            w["RE"][r] = re_w
            w["BOMB"][r] = bs if bs is not None else re_w
            w["STATE"][r] = st if st is not None else re_w
        for name in c:
            c[name][reconcile(w[name], n, tA, tB, half)] += 1
    print("  halftime after round (ASSUMPTION-free, phase-derived):", dict(sorted(fmt.items())))
    for name in c:
        print(f"  {name:<6} {dict(c[name])}")


def extra(season):
    """Checks: flip only the first post-halftime round; reason codes; all-match day flip rate; no-second-phase group."""
    hdr, M, bomb, planted, fz, oe, roster, deaths, ht_map = build(season)
    print(f"\n==================== EXTRA CHECKS {season} ====================")
    flip_fix = Counter()
    reason_xt = Counter()
    nosecond_n = Counter()
    swapped_by_ht = Counter()
    daily_all = defaultdict(lambda: [0, 0, 0])  # day -> [matches with ht & RE & state at ht, disagree, complete]
    for m, rnds in M.items():
        n = len(rnds)
        seq_ok = sorted(rnds) == list(range(1, n + 1))
        h = hdr.get(m)
        ht = ht_map.get(m)
        day = day_of(m)
        if ht is not None and ht in rnds:
            d = (fz.get((m, ht)), oe.get((m, ht)))
            st = state_winner(*d)
            re_w = rnds[ht][0][0]
            if st is not None:
                daily_all[day][0] += 1
                daily_all[day][1] += int(st != re_w)
                reason_xt[(f"ht={ht}", "RE winner", re_w, "RE reason", rnds[ht][0][1], "state winner", st)] += 1
            if h is not None and seq_ok and n == int(h["t_starters_score_final"]) + int(h["ct_starters_score_final"]):
                daily_all[day][2] += 1
        if h is None or not seq_ok or n != int(h["t_starters_score_final"]) + int(h["ct_starters_score_final"]) or n > 24:
            continue
        tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
        half = (ht - 1) if ht is not None else 12
        if ht is None:
            nosecond_n[n] += 1
        # flip only the first post-halftime round in round_end, reconcile header
        w = {}
        for r in range(1, n + 1):
            w[r] = rnds[r][0][0]
        if ht is not None and ht in w:
            w[ht] = CT if w[ht] == T else T
        flip_fix[reconcile(w, n, tA, tB, half)] += 1
        st_w = {}
        for r in range(1, n + 1):
            s = state_winner(fz.get((m, r)), oe.get((m, r)))
            st_w[r] = s if s is not None else rnds[r][0][0]
        if reconcile(st_w, n, tA, tB, half) == "swapped":
            swapped_by_ht[ht] += 1
    print("complete regulation matches, round_end with ONLY the first post-halftime round flipped, header reconcile:")
    print("  ", dict(flip_fix))
    print("state-based swapped (header labels reversed) by halftime round:", dict(swapped_by_ht))
    print("regulation complete matches with no second-phase round, by n (match ended before halftime?):",
          dict(sorted(nosecond_n.items())))
    print("\nround_end reason vs state winner at first post-halftime round (all matches with state at ht):")
    for k, v in sorted(reason_xt.items(), key=lambda z: -z[1])[:12]:
        print(f"  {k}: {v}")
    print("\nday flip rate at first post-halftime round, ALL matches with ht and state (path day):")
    print(f"  {'day':<5}{'with_ht_state':>15}{'RE!=state':>12}{'rate':>8}{'complete_reg':>14}")
    for day in sorted(daily_all):
        a, dis, comp_ = daily_all[day]
        print(f"  {day:<5}{a:>15}{dis:>12}{(dis / a if a else 0):>8.3f}{comp_:>14}")


def date_check(season):
    """Claim says 'matches dated 2024-02-12..25' but groups by folder path day. Compare with header match_date."""
    hdr, M, bomb, planted, fz, oe, roster, deaths, ht_map = build(season)
    print(f"\n==================== DATE CHECK {season} ====================")
    agree = Counter()
    flip_by_md = defaultdict(lambda: [0, 0])
    for m, rnds in M.items():
        h = hdr.get(m)
        if h is None:
            continue
        n = len(rnds)
        if sorted(rnds) != list(range(1, n + 1)) or n != int(h["t_starters_score_final"]) + int(h["ct_starters_score_final"]) or n > 24:
            continue
        md = str(h["match_date"])[:10]
        path_day = day_of(m)
        agree["header date day == path day" if md[-2:] == path_day else "header date day != path day"] += 1
        ht = ht_map.get(m)
        if ht is not None and ht in rnds:
            st = state_winner(fz.get((m, ht)), oe.get((m, ht)))
            if st is not None:
                flip_by_md[md[:7]][0] += 1
                flip_by_md[md[:7]][1] += int(st != rnds[ht][0][0])
    print("complete regulation matches, header match_date vs folder day:", dict(agree))
    print("complete regulation halftime-round flips by header match_date month:",
          {k: f"{v[1]}/{v[0]}" for k, v in sorted(flip_by_md.items())})


def bomb_at_ht(season):
    hdr, M, bomb, planted, fz, oe, roster, deaths, ht_map = build(season)
    c = Counter()
    for m, rnds in M.items():
        h = hdr.get(m)
        if h is None:
            continue
        n = len(rnds)
        if sorted(rnds) != list(range(1, n + 1)) or n != int(h["t_starters_score_final"]) + int(h["ct_starters_score_final"]) or n > 24:
            continue
        ht = ht_map.get(m)
        if ht is None or ht not in rnds:
            continue
        ev = bomb.get((m, ht), set())
        if not ev:
            continue
        bs = CT if "bomb_defused" in ev else T
        st = state_winner(fz.get((m, ht)), oe.get((m, ht)))
        c["bomb outcome rows at first post-halftime round"] += 1
        c[f"  round_end == bomb side: {rnds[ht][0][0] == bs}"] += 1
        c[f"  running state == bomb side: {st == bs}"] += 1
    print(f"\n==================== BOMB AT FIRST POST-HALFTIME ROUND {season} ====================")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    analyse("2024")
    control("2026")
    control("2024")
    extra("2024")
    extra("2026")
    date_check("2024")
    bomb_at_ht("2024")
    bomb_at_ht("2026")
