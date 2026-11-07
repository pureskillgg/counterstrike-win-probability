"""Task 2 follow-up: batch comparison restricted to 5v5 rounds, and batch identifiability (read-only).

5v5 subset = matches with header is_wingman False, rounds with round_end.player_count == 10.
Identifiability uses only round-start features (no outcome, no death counts):
  round_number, max_rounds, t_score, ct_score at live start, is_pistol, is_overtime.
OMP_NUM_THREADS is pinned to 1 before sklearn is imported (an earlier run stalled in HistGradientBoosting
under CPU contention).

Run: pixi run python -I scripts\\eda\\batch_5v5.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
              12: "Target_Saved", 17: "Terrorists_Surrender", 18: "CTs_Surrender"}


def read(ch, cols=None):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def two_prop(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return p1 - p2, z, math.erfc(abs(z) / math.sqrt(2))


def welch(a, b):
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    t = (ma - mb) / se if se > 0 else 0.0
    return ma - mb, t, math.erfc(abs(t) / math.sqrt(2))


def mean(xs):
    return sum(xs) / len(xs)


def main():
    hdr = read("header", ["key", "match_date", "match_type", "platform", "is_wingman", "max_rounds"])
    day = {k: md[:10] for k, md in zip(hdr["key"].to_list(), hdr["match_date"].to_list())}
    wing = {k: bool(w) for k, w in zip(hdr["key"].to_list(), hdr["is_wingman"].to_list())}
    mr = dict(zip(hdr["key"].to_list(), hdr["max_rounds"].to_list()))
    mtype = dict(zip(hdr["key"].to_list(), hdr["match_type"].to_list()))
    plat = dict(zip(hdr["key"].to_list(), hdr["platform"].to_list()))

    def batch(k):
        d = day[k]
        if d == NEW:
            return "new_1008"
        if "2026-10-01" <= d <= "2026-10-07":
            return "other_1001_1007"
        return "outside"

    RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    BS = read("bomb_state", ["match_key", "round", "tick", "event_type"])
    DE = read("player_death", ["match_key", "round", "tick", "attacker_id_fixed", "player_id_fixed"])

    live = {}
    live_scores = {}
    for m, r, t, et, ts, cs in zip(RS["match_key"].to_list(), RS["round"].to_list(), RS["tick"].to_list(),
                                   RS["event_type"].to_list(), RS["t_score"].to_list(), RS["ct_score"].to_list()):
        if et == "round_freeze_end" and (m, r) not in live:
            live[(m, r)] = t
            live_scores[(m, r)] = (ts, cs)
    end = {}
    plant_in = {}
    for m, r, t in zip(RE["match_key"].to_list(), RE["round"].to_list(), RE["tick"].to_list()):
        end.setdefault((m, r), t)
    for m, r, t, et in zip(BS["match_key"].to_list(), BS["round"].to_list(), BS["tick"].to_list(),
                           BS["event_type"].to_list()):
        if et == "bomb_planted" and (m, r) in live and (m, r) in end and live[(m, r)] <= t <= end[(m, r)]:
            plant_in.setdefault((m, r), t)
    kills = collections.Counter()
    for m, r, a, v in zip(DE["match_key"].to_list(), DE["round"].to_list(), DE["attacker_id_fixed"].to_list(),
                          DE["player_id_fixed"].to_list()):
        if a is not None and a > 0 and a != v:
            kills[(m, r)] += 1

    # 5v5 rounds
    rows = []  # batch, match, round, winner, reason, plant(bool), plant_sec, kills, features
    for m, r, w, wr, pc in zip(RE["match_key"].to_list(), RE["round"].to_list(), RE["winner_team_code"].to_list(),
                               RE["win_reason_code"].to_list(), RE["player_count"].to_list()):
        if wing.get(m, True) or pc != 10:
            continue
        b = batch(m)
        if b == "outside":
            continue
        pl_ = (m, r) in plant_in
        psec = (plant_in[(m, r)] - live[(m, r)]) / TICK_RATE if pl_ and (m, r) in live else None
        ts, cs = live_scores.get((m, r), (None, None))
        rows.append(dict(b=b, m=m, r=r, w=w, wr=wr, plant=pl_, psec=psec, kills=kills.get((m, r), 0),
                         ts=ts, cs=cs, mr=mr[m]))

    print("## 5v5 subset (is_wingman False, player_count == 10)")
    print("| batch | matches | rounds | rounds/match | T share | plant rate (in window) | plant time mean (s) | kills/round | Target_Bombed | Bomb_Defused | CTs_Win | Terrorists_Win |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    st = {}
    for b in ["new_1008", "other_1001_1007"]:
        rr = [x for x in rows if x["b"] == b]
        ms = collections.Counter(x["m"] for x in rr)
        n = len(rr)
        tw = sum(1 for x in rr if x["w"] == T)
        pl_n = sum(1 for x in rr if x["plant"])
        secs = [x["psec"] for x in rr if x["psec"] is not None]
        k = [x["kills"] for x in rr]
        wrc = collections.Counter(x["wr"] for x in rr)
        st[b] = dict(rr=rr, n=n, tw=tw, pl=pl_n, secs=secs, k=k, wrc=wrc, ms=ms)
        print(f"| {b} | {len(ms)} | {n} | {n / len(ms):.2f} | {tw / n:.4f} | {pl_n / n:.4f} | {mean(secs):.2f} | "
              f"{mean(k):.3f} | {wrc.get(1, 0) / n:.4f} | {wrc.get(7, 0) / n:.4f} | {wrc.get(8, 0) / n:.4f} | "
              f"{wrc.get(9, 0) / n:.4f} |")
    a, o = st["new_1008"], st["other_1001_1007"]
    print("\n### Tests on 5v5 rounds (new - other)")
    d, z, p = two_prop(a["tw"], a["n"], o["tw"], o["n"])
    print(f"- T share diff {d:+.4f}; z {z:.2f}; p {p:.2e}")
    d, z, p = two_prop(a["pl"], a["n"], o["pl"], o["n"])
    print(f"- plant rate (in window) diff {d:+.4f}; z {z:.2f}; p {p:.2e}")
    d, z, p = two_prop(a["wrc"].get(1, 0), a["n"], o["wrc"].get(1, 0), o["n"])
    print(f"- Target_Bombed share diff {d:+.4f}; z {z:.2f}; p {p:.2e}")
    d, z, p = two_prop(a["wrc"].get(7, 0), a["n"], o["wrc"].get(7, 0), o["n"])
    print(f"- Bomb_Defused share diff {d:+.4f}; z {z:.2f}; p {p:.2e}")
    d, z, p = welch(a["secs"], o["secs"])
    print(f"- plant time diff {d:+.3f} s; Welch t {t_fmt(z)}; p {p:.2e}")
    d, z, p = welch(a["k"], o["k"])
    print(f"- kills per round diff {d:+.4f}; Welch t {t_fmt(z)}; p {p:.2e}")
    d, z, p = welch(list(a["ms"].values()), list(o["ms"].values()))
    print(f"- rounds per match diff {d:+.3f}; Welch t {t_fmt(z)}; p {p:.2e}")

    # unknown metadata within 5v5 matches
    print("\n### Header metadata for 5v5 matches (match level)")
    for b in ["new_1008", "other_1001_1007"]:
        ks = [k for k in day if batch(k) == b and not wing[k]]
        u = sum(1 for k in ks if mtype[k] == "unknown")
        print(f"- {b}: 5v5 matches {len(ks)}; match_type unknown {u} ({u / len(ks):.4f})")

    # per-day, 5v5 only
    print("\n### Per-day, 5v5 only")
    print("| day | matches | rounds | T share | plant rate (in window) |")
    print("|---|---|---|---|---|")
    byday = collections.defaultdict(list)
    for x in rows:
        byday[day[x["m"]]].append(x)
    for dd in sorted(byday):
        rr = byday[dd]
        nm = len({x["m"] for x in rr})
        print(f"| {dd} | {nm} | {len(rr)} | {sum(1 for x in rr if x['w'] == T) / len(rr):.4f} | "
              f"{sum(1 for x in rr if x['plant']) / len(rr):.4f} |")

    # identifiability from round-start features (5v5 rounds)
    print("\n### Batch identifiability (5v5 rounds, round-start features only)")
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        feats, y, g = [], [], []
        for x in rows:
            if x["ts"] is None:
                continue
            ispistol = int(x["r"] in (1, x["mr"] // 2 + 1))
            isot = int(x["r"] > x["mr"])
            feats.append([x["r"], x["mr"], x["ts"], x["cs"], ispistol, isot])
            y.append(1 if x["b"] == "new_1008" else 0)
            g.append(x["m"])
        X = np.array(feats, dtype=float)
        y = np.array(y)
        g = np.array(g)
        aucs = []
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0)
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        print(f"- rounds {len(y)}; batch AUC by 5-fold match-grouped CV: {[round(v, 3) for v in aucs]}; "
              f"mean {float(np.mean(aucs)):.3f}")
        print(f"- base rate new batch among these rounds: {float(y.mean()):.4f}")
    except Exception as e:  # keep the descriptive output
        print(f"- identifiability skipped: {type(e).__name__}: {e}")


def t_fmt(z):
    return f"{z:.2f}"


if __name__ == "__main__":
    main()
