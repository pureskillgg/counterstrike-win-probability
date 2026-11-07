"""Task 2: batch effect of the 2026-10-08 batch vs 2026-10-01..07 (read-only).

Batch = header match_date day. Channels: round_end, round_state, bomb_state, player_death, player_info.
Run: pixi run python -I scripts\\eda\\batch_effect.py
Prints tables and tests; the report is written by hand from this output.
"""
import collections
import glob
import math
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
TICK_RATE = 64
T, CT = 2, 3
NEW = "2026-10-08"
WIN_REASON = {1: "Target_Bombed", 7: "Bomb_Defused", 8: "CTs_Win", 9: "Terrorists_Win",
              11: "All_Hostages_Rescued", 12: "Target_Saved", 13: "Hostages_Not_Rescued",
              17: "Terrorists_Surrender", 18: "CTs_Surrender"}


def read(ch):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def two_prop(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    pval = math.erfc(abs(z) / math.sqrt(2))
    return p1 - p2, z, pval


def welch(a, b):
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    t = (ma - mb) / se if se > 0 else 0.0
    pval = math.erfc(abs(t) / math.sqrt(2))  # normal approx; n is large in every use here
    return ma - mb, t, pval


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else float("nan")


def main():
    hdr = read("header")
    day_of = {}
    for k, md in zip(hdr["key"].to_list(), hdr["match_date"].to_list()):
        day_of[k] = md[:10]
    batch_of = {}
    for k, d in day_of.items():
        if d == NEW:
            batch_of[k] = "new_1008"
        elif "2026-10-01" <= d <= "2026-10-07":
            batch_of[k] = "other_1001_1007"
        else:
            batch_of[k] = "outside"
    print("## Header day distribution")
    print(f"- match_date day counts: {dict(sorted(collections.Counter(day_of.values()).items()))}")
    print(f"- batch counts: {dict(collections.Counter(batch_of.values()))}")
    key_day = {k: "/".join(k.split("/")[1:4]) for k in day_of}
    agree = sum(1 for k in day_of if day_of[k].replace("-", "/") == key_day[k])
    print(f"- match_date day equals key-path day for {agree} of {len(day_of)} matches")
    print(f"- key-path day counts: {dict(sorted(collections.Counter(key_day.values()).items()))}")

    # header categorical checks per batch (match level)
    print("\n## Header match-level: match_type, platform, max_rounds per batch")
    mt = collections.defaultdict(collections.Counter)
    pf = collections.defaultdict(collections.Counter)
    mr = collections.defaultdict(collections.Counter)
    wg = collections.defaultdict(collections.Counter)
    for k, t, p, m, w in zip(hdr["key"].to_list(), hdr["match_type"].to_list(), hdr["platform"].to_list(),
                             hdr["max_rounds"].to_list(), hdr["is_wingman"].to_list()):
        b = batch_of[k]
        mt[b][t] += 1
        pf[b][p] += 1
        mr[b][m] += 1
        wg[b][bool(w)] += 1
    for b in ["new_1008", "other_1001_1007", "outside"]:
        if b not in mt:
            continue
        n = sum(mt[b].values())
        print(f"- {b} (n={n}): match_type {dict(mt[b])}; platform {dict(pf[b])}; max_rounds {dict(mr[b])}; "
              f"is_wingman {dict(wg[b])}")

    # channel loads
    RE = read("round_end")
    RS = read("round_state")
    BS = read("bomb_state")
    DE = read("player_death")
    PI = read("player_info")

    def batch(df):
        return [batch_of.get(m, "missing") for m in df["match_key"].to_list()]

    # per-round table from round_end
    rounds = []  # (batch, match, round, winner, reason, player_count, tick)
    for m, r, w, wr, pc, t in zip(RE["match_key"].to_list(), RE["round"].to_list(),
                                  RE["winner_team_code"].to_list(), RE["win_reason_code"].to_list(),
                                  RE["player_count"].to_list(), RE["tick"].to_list()):
        rounds.append((batch_of.get(m, "missing"), m, r, w, wr, pc, t))
    live = {}
    for m, r, t, et in zip(RS["match_key"].to_list(), RS["round"].to_list(), RS["tick"].to_list(),
                           RS["event_type"].to_list()):
        if et == "round_freeze_end":
            live.setdefault((m, r), t)
    plant_tick = {}  # first bomb_planted row per round, any time (naive)
    for m, r, t, et in zip(BS["match_key"].to_list(), BS["round"].to_list(), BS["tick"].to_list(),
                           BS["event_type"].to_list()):
        if et == "bomb_planted":
            plant_tick.setdefault((m, r), t)
    end_of = {(m, r): t for m, r, t in zip(RE["match_key"].to_list(), RE["round"].to_list(),
                                           RE["tick"].to_list())}
    plant_in = {}  # first bomb_planted row inside [live start, round_end] (the model's definition)
    for m, r, t, et in zip(BS["match_key"].to_list(), BS["round"].to_list(), BS["tick"].to_list(),
                           BS["event_type"].to_list()):
        if et == "bomb_planted" and (m, r) in live and (m, r) in end_of and live[(m, r)] <= t <= end_of[(m, r)]:
            plant_in.setdefault((m, r), t)
    # kills: attacker valid and not the victim
    print("\n## player_death attacker/victim sentinel check")
    att = DE["attacker_id_fixed"].to_list()
    vic = DE["player_id_fixed"].to_list()
    print(f"- attacker_id_fixed null: {sum(1 for x in att if x is None)}, <=0: "
          f"{sum(1 for x in att if x is not None and x <= 0)}, distinct {len(set(att))}")
    print(f"- victim player_id_fixed null: {sum(1 for x in vic if x is None)}")
    kills = collections.Counter()
    suicides = collections.Counter()
    for m, r, a, v in zip(DE["match_key"].to_list(), DE["round"].to_list(), att, vic):
        if a is None or a <= 0:
            continue
        if a == v:
            suicides[(m, r)] += 1
        else:
            kills[(m, r)] += 1
    print(f"- deaths with valid attacker != victim: {sum(kills.values())}; suicides/self: {sum(suicides.values())}")

    # plant rate and time, per round
    print("\n## Per-batch round metrics")
    per_b = collections.defaultdict(list)
    for row in rounds:
        per_b[row[0]].append(row)
    stats = {}
    for b in ["new_1008", "other_1001_1007"]:
        rr = per_b[b]
        n_r = len(rr)
        matches = {x[1] for x in rr}
        per_match = collections.Counter(x[1] for x in rr)
        t_wins = sum(1 for x in rr if x[3] == T)
        ct_wins = sum(1 for x in rr if x[3] == CT)
        reasons = collections.Counter(x[4] for x in rr)
        planted_rounds = [x for x in rr if (x[1], x[2]) in plant_in]
        planted_any = sum(1 for x in rr if (x[1], x[2]) in plant_tick)
        plant_secs = []
        for x in planted_rounds:
            key = (x[1], x[2])
            if key in live:
                plant_secs.append((plant_in[key] - live[key]) / TICK_RATE)
        kpr = [kills.get((x[1], x[2]), 0) for x in rr]
        pc = collections.Counter(x[5] for x in rr)
        stats[b] = dict(rounds=rr, n_r=n_r, matches=len(matches), per_match=per_match,
                        t_wins=t_wins, ct_wins=ct_wins, reasons=reasons,
                        planted=len(planted_rounds), plant_secs=plant_secs, kpr=kpr, pc=pc,
                        planted_any=planted_any)
        print(f"\n### {b}")
        print(f"- matches: {len(matches)}; rounds: {n_r}; rounds per match mean {mean(list(per_match.values())):.2f}, "
              f"median {median(list(per_match.values()))}")
        print(f"- winner T (2): {t_wins} = {t_wins / n_r:.4f}; winner CT (3): {ct_wins} = {ct_wins / n_r:.4f}")
        print("- win_reason shares: " + "; ".join(
            f"{k} {WIN_REASON.get(k, 'UNMAPPED')}={v} ({v / n_r:.4f})" for k, v in sorted(reasons.items())))
        print(f"- plant rate per round (plant inside live..round_end): {len(planted_rounds)}/{n_r} = "
              f"{len(planted_rounds) / n_r:.4f}; any bomb_planted row in round (naive): {planted_any} = "
              f"{planted_any / n_r:.4f}")
        print(f"- plant time (s from live start): n={len(plant_secs)} mean {mean(plant_secs):.2f}, "
              f"median {median(plant_secs):.2f}; negative values: {sum(1 for s in plant_secs if s < 0)}")
        print(f"- kills per round: mean {mean(kpr):.3f}; rounds with 0 kills: {sum(1 for k in kpr if k == 0)}")
        print("- round_end player_count shares: " + "; ".join(
            f"{k}={v} ({v / n_r:.4f})" for k, v in sorted(pc.items())))
        print(f"- rounds with player_count == 10: {pc.get(10, 0)} ({pc.get(10, 0) / n_r:.4f})")

    # player_info: players per round and 5v5 at start
    print("\n## player_info per round")
    pi_round = collections.defaultdict(lambda: {"n": 0, "T": 0, "CT": 0})
    seen = set()
    for m, r, pid, tc in zip(PI["match_key"].to_list(), PI["round"].to_list(),
                             PI["player_id"].to_list(), PI["team_code"].to_list()):
        key = (m, r, pid)
        if key in seen:
            continue
        seen.add(key)
        d = pi_round[(m, r)]
        d["n"] += 1
        if tc == T:
            d["T"] += 1
        elif tc == CT:
            d["CT"] += 1
    for b in ["new_1008", "other_1001_1007"]:
        keys = [k for k in pi_round if batch_of.get(k[0]) == b]
        n_keys = len(keys)
        five = sum(1 for k in keys if pi_round[k]["T"] == 5 and pi_round[k]["CT"] == 5)
        ns = [pi_round[k]["n"] for k in keys]
        print(f"- {b}: rounds with player_info {n_keys}; mean players {mean(ns):.3f}; "
              f"exactly 5 T and 5 CT: {five} ({five / n_keys:.4f})")

    # two-sample tests
    new = stats["new_1008"]
    old = stats["other_1001_1007"]
    print("\n## Tests (new 2026-10-08 vs other 2026-10-01..07)")
    d, z, p = two_prop(new["t_wins"], new["n_r"], old["t_wins"], old["n_r"])
    print(f"- T win share diff (new - old): {d:+.4f}; z {z:.2f}; p {p:.2e}")
    d, z, p = two_prop(new["ct_wins"], new["n_r"], old["ct_wins"], old["n_r"])
    print(f"- CT win share diff: {d:+.4f}; z {z:.2f}; p {p:.2e}")
    for code in sorted(set(new["reasons"]) | set(old["reasons"])):
        d, z, p = two_prop(new["reasons"].get(code, 0), new["n_r"], old["reasons"].get(code, 0), old["n_r"])
        print(f"- win_reason {code} ({WIN_REASON.get(code, 'UNMAPPED')}) share diff: {d:+.4f}; z {z:.2f}; p {p:.2e}")
    d, z, p = two_prop(new["planted"], new["n_r"], old["planted"], old["n_r"])
    print(f"- plant rate diff: {d:+.4f}; z {z:.2f}; p {p:.2e}")
    d, z, p = two_prop(new["pc"].get(10, 0), new["n_r"], old["pc"].get(10, 0), old["n_r"])
    print(f"- player_count==10 share diff: {d:+.4f}; z {z:.2f}; p {p:.2e}")
    dm, t, p = welch(new["plant_secs"], old["plant_secs"])
    print(f"- plant time diff (s): {dm:+.3f}; Welch t {t:.2f}; p {p:.2e}")
    dm, t, p = welch(new["kpr"], old["kpr"])
    print(f"- kills per round diff: {dm:+.4f}; Welch t (rounds) {t:.2f}; p {p:.2e}")
    dm, t, p = welch(list(new["per_match"].values()), list(old["per_match"].values()))
    print(f"- rounds per match diff: {dm:+.3f}; Welch t (matches) {t:.2f}; p {p:.2e}")
    # match-level robustness for shares
    nm_plant = {}
    for b, st in (("new", new), ("old", old)):
        pass
    def per_match_share(rr, pred):
        hits = collections.Counter(x[1] for x in rr if pred(x))
        return [hits.get(m, 0) / n for m, n in collections.Counter(x[1] for x in rr).items()]

    mt_new = per_match_share(new["rounds"], lambda x: (x[1], x[2]) in plant_in)
    mt_old = per_match_share(old["rounds"], lambda x: (x[1], x[2]) in plant_in)
    dm, t, p = welch(mt_new, mt_old)
    print(f"- plant rate per match (match-level Welch): diff {dm:+.4f}; t {t:.2f}; p {p:.2e}")
    tw_new = per_match_share(new["rounds"], lambda x: x[3] == T)
    tw_old = per_match_share(old["rounds"], lambda x: x[3] == T)
    dm, t, p = welch(tw_new, tw_old)
    print(f"- T win share per match (match-level Welch): diff {dm:+.4f}; t {t:.2f}; p {p:.2e}")

    # match-level unknown checks
    n_new_m = sum(1 for k in day_of if batch_of[k] == "new_1008")
    n_old_m = sum(1 for k in day_of if batch_of[k] == "other_1001_1007")
    unk = {}
    for b in ["new_1008", "other_1001_1007"]:
        ks = [k for k in day_of if batch_of[k] == b]
        rows_b = [(t, p_) for k, t, p_ in zip(hdr["key"].to_list(), hdr["match_type"].to_list(),
                                              hdr["platform"].to_list()) if batch_of[k] == b]
        mt_u = sum(1 for t, _ in rows_b if t == "unknown")
        pf_u = sum(1 for _, p_ in rows_b if p_ == "unknown")
        either = sum(1 for t, p_ in rows_b if t == "unknown" or p_ == "unknown")
        unk[b] = (len(ks), mt_u, pf_u, either)
        print(f"- {b}: matches {len(ks)}; match_type unknown {mt_u} ({mt_u / len(ks):.4f}); "
              f"platform unknown {pf_u} ({pf_u / len(ks):.4f}); either {either} ({either / len(ks):.4f})")
    a = unk["new_1008"]
    b_ = unk["other_1001_1007"]
    d, z, p = two_prop(a[3], a[0], b_[3], b_[0])
    print(f"- either-unknown share diff: {d:+.4f}; z {z:.2f}; p {p:.2e}")

    # per-day breakdown for context
    print("\n## Per-day breakdown (round_end rows by match_date day)")
    by_day = collections.defaultdict(list)
    for row in rounds:
        by_day[day_of.get(row[1], "?")].append(row)
    plant_by_day = collections.Counter()
    for row in rounds:
        if (row[1], row[2]) in plant_in:
            plant_by_day[day_of.get(row[1], "?")] += 1
    print("| day | matches | rounds | rounds/match | T share | plant rate | player_count==10 share |")
    print("|---|---|---|---|---|---|---|")
    for dday in sorted(by_day):
        rr = by_day[dday]
        nmat = len({x[1] for x in rr})
        tw = sum(1 for x in rr if x[3] == T)
        pcn = sum(1 for x in rr if x[5] == 10)
        print(f"| {dday} | {nmat} | {len(rr)} | {len(rr) / nmat:.2f} | {tw / len(rr):.4f} | "
              f"{plant_by_day[dday] / len(rr):.4f} | {pcn / len(rr):.4f} |")

    # batch identifiability: can round-level features known at live start predict the batch?
    print("\n## Batch identifiability (round-level, label-free features)")
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        X, y, g = [], [], []
        mo = {}
        for k, mr_v in zip(hdr["key"].to_list(), hdr["max_rounds"].to_list()):
            mo[k] = mr_v
        for b, m, r, w, wr, pc, t in rounds:
            if b not in ("new_1008", "other_1001_1007"):
                continue
            X.append([r, mo.get(m, 0), pc])
            y.append(1 if b == "new_1008" else 0)
            g.append(m)
        X = np.array(X, dtype=float)
        y = np.array(y)
        g = np.array(g)
        aucs = []
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0)
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        print(f"- features [round_number, max_rounds, player_count] -> batch, 5-fold grouped-by-match AUC: "
              f"{[round(a_, 3) for a_ in aucs]} mean {np.mean(aucs):.3f}")
    except Exception as e:  # keep the descriptive output even if sklearn is unavailable
        print(f"- identifiability check skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
