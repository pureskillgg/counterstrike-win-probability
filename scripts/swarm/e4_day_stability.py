"""E4: per-day and batch stability of CT win rate, plant rate, rounds per match, kills per round.

Read-only. Reads parquet pages from G:\\pureskill\\csds-cs2\\tomes only:
  2026-10-01_to_08 (key-path days 2026/10/01 .. 2026/10/08) and 2024-02-g (key-path days 2024/02/01 .. 2024/02/25).

Definitions (kept identical to scripts\\eda\\batch_effect.py where they overlap):
  day        = folder day in match_key (csds/YYYY/MM/DD/<uuid>/csds)
  CT win     = round_end.winner_team_code == 3 (2 = T), share over all round_end rows
  plant      = bomb_planted row with live start (round_freeze_end, min tick) <= tick <= round_end tick
  kill       = player_death with attacker_id_fixed > 0 and attacker != victim (player_id_fixed)
  rounds/match = count of round_end rows per match (distinct round numbers)
  5v5        = round_end.player_count == 10 (is_wingman is absent from the 2024 header, so the same rule is used in both tomes)
  batch      = the largest key-path day in each tome (2026/10/08, 2024/02/12) against the other days of that tome
  dedupe     = signature groups (round_end + player_death + player_info, same repr scheme as dedupe_replicate.py),
               over matches with >= 1 round; lowest match_key kept. Run with and without dedupe.

Intervals: 95% percentile bootstrap that resamples MATCHES within a day (B = 2000, seed 20261013),
           plus the analytic cluster-robust SE (linearised ratio, V = sum_i (x_i - r n_i)^2 / R^2).
Chi-square: naive (rounds independent: binomial / Poisson homogeneity) and cluster-robust
            (per-day estimate and its cluster-robust V; df = days - 1). rounds/match uses V = s^2 / M.

Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I -X utf8 scripts\\swarm\\e4_day_stability.py
"""
import collections
import glob
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2026": {"base": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
             "suffix": ".2026-09-18,2026-10-08", "batch_day": "2026/10/08"},
    "2024": {"base": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
             "suffix": ".2023-12-28,2024-02-25", "batch_day": "2024/02/12"},
}
T, CT = 2, 3
B = 2000
SEED = 20261013
Z95 = 1.959963984540054

try:
    from scipy import stats as _st

    def chi2_sf(x, df):
        return float(_st.chi2.sf(x, df))

    CHI2_BACKEND = "scipy"
except Exception:  # regularised upper incomplete gamma (Numerical Recipes gammq), no scipy needed
    def _gammaincc(a, x):
        if x <= 0:
            return 1.0
        gln = math.lgamma(a)
        if x < a + 1.0:
            ap = a
            s = 1.0 / a
            d = s
            for _ in range(10000):
                ap += 1.0
                d *= x / ap
                s += d
                if abs(d) < abs(s) * 1e-15:
                    break
            return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
        fpmin = 1e-300
        b = x + 1.0 - a
        c = 1.0 / fpmin
        d = 1.0 / b
        h = d
        for i in range(1, 10000):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < fpmin:
                d = fpmin
            c = b + an / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-15:
                break
        return math.exp(-x + a * math.log(x) - gln) * h

    def chi2_sf(x, df):
        return _gammaincc(df / 2.0, x / 2.0)

    CHI2_BACKEND = "fallback(gammaincc)"


def read(tk, ch, cols):
    base, suf = TOMES[tk]["base"], TOMES[tk]["suffix"]
    folder = base / ("header" if ch == "header" else f"basic_{ch}{suf}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def load(tk):
    hdr = read(tk, "header", ["key", "match_date"])
    key_day = {}
    agree = 0
    for k, md in hdr.select(["key", "match_date"]).iter_rows():
        d = "/".join(k.split("/")[1:4])
        key_day[k] = d
        if md and str(md)[:10].replace("-", "/") == d:
            agree += 1
    D = {
        "key_day": key_day,
        "header_n": hdr.height,
        "md_agree": agree,
        "re": read(tk, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]),
        "rs": read(tk, "round_state", ["match_key", "round", "tick", "event_type"]),
        "bs": read(tk, "bomb_state", ["match_key", "round", "tick", "event_type"]),
        "de": read(tk, "player_death", ["match_key", "round", "tick", "player_id", "player_id_fixed",
                                        "attacker_id_fixed"]),
        "pi": read(tk, "player_info", ["match_key", "round", "player_id", "team_code"]),
    }
    return D


def signatures(D):
    parts = collections.defaultdict(list)
    for m, r, t, w in D["re"].select(["match_key", "round", "tick", "winner_team_code"]).iter_rows():
        parts[m].append(repr(("R", str(r), str(t), str(w))))
    for m, r, t, pid in D["de"].select(["match_key", "round", "tick", "player_id"]).iter_rows():
        parts[m].append(repr(("D", str(r), str(t), str(pid))))
    for m, r, pid, tc in D["pi"].select(["match_key", "round", "player_id", "team_code"]).iter_rows():
        parts[m].append(repr(("P", str(r), str(pid), str(tc))))
    return {m: tuple(sorted(v)) for m, v in parts.items()}


def dedupe(sigs, matches):
    groups = collections.defaultdict(list)
    for m in matches:
        s = sigs.get(m)
        if s:
            groups[s].append(m)
    dup_groups = [sorted(g) for g in groups.values() if len(g) > 1]
    drop = set()
    for g in dup_groups:
        drop.update(g[1:])
    return drop, dup_groups


def round_table(D):
    re_ = D["re"]
    rds_dupe_rows = re_.height - re_.select(["match_key", "round"]).unique().height
    rds = re_.unique(subset=["match_key", "round"], keep="first", maintain_order=True)
    live = (D["rs"].filter(pl.col("event_type") == "round_freeze_end")
            .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("live_tick")))
    plant_rows = (D["bs"].filter(pl.col("event_type") == "bomb_planted")
                  .select(["match_key", "round", "tick"]).rename({"tick": "plant_tick"}))
    plants = (plant_rows
              .join(rds.select(["match_key", "round", "tick"]).rename({"tick": "end_tick"}),
                    on=["match_key", "round"], how="inner")
              .join(live, on=["match_key", "round"], how="inner")
              .filter((pl.col("plant_tick") >= pl.col("live_tick")) & (pl.col("plant_tick") <= pl.col("end_tick")))
              .select(["match_key", "round"]).unique()
              .with_columns(pl.lit(1, dtype=pl.Int64).alias("planted")))
    kills = (D["de"].filter(pl.col("attacker_id_fixed").is_not_null() & (pl.col("attacker_id_fixed") > 0)
                            & (pl.col("player_id_fixed").is_null()
                               | (pl.col("attacker_id_fixed") != pl.col("player_id_fixed"))))
             .group_by(["match_key", "round"]).agg(pl.col("round").count().alias("kills")))
    pi_u = D["pi"].unique(subset=["match_key", "round", "player_id"], keep="first")
    pi_cnt = pi_u.group_by(["match_key", "round"]).agg([
        (pl.col("team_code") == T).cast(pl.Int64).sum().alias("nT"),
        (pl.col("team_code") == CT).cast(pl.Int64).sum().alias("nCT"),
    ])
    rounds = (rds.select(["match_key", "round", "winner_team_code", "player_count"])
              .join(plants, on=["match_key", "round"], how="left")
              .join(kills, on=["match_key", "round"], how="left")
              .join(pi_cnt, on=["match_key", "round"], how="left")
              .with_columns([
                  pl.col("planted").fill_null(0).cast(pl.Int64),
                  pl.col("kills").fill_null(0).cast(pl.Int64),
                  pl.col("nT").fill_null(0).cast(pl.Int64),
                  pl.col("nCT").fill_null(0).cast(pl.Int64),
                  (pl.col("winner_team_code") == CT).cast(pl.Int64).alias("ct"),
                  (pl.col("winner_team_code") == T).cast(pl.Int64).alias("t"),
              ])
              .with_columns([
                  ((pl.col("nT") == 5) & (pl.col("nCT") == 5)).cast(pl.Int64).alias("is5"),
              ]))
    live_keys = set(zip(live["match_key"].to_list(), live["round"].to_list()))
    no_live = sum(1 for k in zip(rds["match_key"].to_list(), rds["round"].to_list()) if k not in live_keys)
    info = {
        "re_rows": re_.height,
        "re_dupe_rows": rds_dupe_rows,
        "rounds_unique": rds.height,
        "rounds_no_live_start": no_live,
        "rounds_no_live_start_share": no_live / rds.height,
        "freeze_end_rows": D["rs"].filter(pl.col("event_type") == "round_freeze_end").height,
        "bomb_planted_rows": plant_rows.height,
        "planted_rounds_in_window": plants.height,
        "kill_rows": kills["kills"].sum(),
        "death_rows": D["de"].height,
        "attacker_null_or_le0": D["de"].filter(pl.col("attacker_id_fixed").is_null()
                                               | (pl.col("attacker_id_fixed") <= 0)).height,
        "winner_codes": dict(sorted(collections.Counter(rds["winner_team_code"].to_list()).items(),
                                    key=lambda kv: str(kv[0]))),
        "pi5_share": float(rounds["is5"].sum()) / rounds.height,
        "pc_top": collections.Counter(rds["player_count"].to_list()).most_common(4),
        "rs_event_types": sorted(set(D["rs"]["event_type"].unique().to_list()))[:40],
        "bs_event_types": sorted(set(D["bs"]["event_type"].unique().to_list())),
    }
    return rounds, info


def match_table(rounds, key_day):
    agg = rounds.group_by("match_key").agg([
        pl.col("round").count().alias("n"),
        pl.col("ct").sum().alias("ct"),
        pl.col("planted").sum().alias("pl"),
        pl.col("kills").sum().alias("k"),
        pl.col("is5").sum().alias("n5"),
        (pl.col("ct") * pl.col("is5")).sum().alias("ct5"),
        (pl.col("planted") * pl.col("is5")).sum().alias("pl5"),
        (pl.col("kills") * pl.col("is5")).sum().alias("k5"),
    ])
    dayf = pl.DataFrame({"match_key": list(key_day.keys()), "day": list(key_day.values())})
    return agg.join(dayf, on="match_key", how="left")


def build(D, do_dedupe):
    matches = sorted(set(D["re"]["match_key"].unique().to_list()))
    sigs = signatures(D)
    drop, groups = dedupe(sigs, matches) if do_dedupe else (set(), [])
    keep = [m for m in matches if m not in drop]
    rounds, info = round_table(D)
    rounds = rounds.filter(pl.col("match_key").is_in(keep))
    m = match_table(rounds, D["key_day"])
    info.update({
        "header_matches": D["header_n"],
        "matches_with_rounds": len(matches),
        "header_without_rounds": D["header_n"] - len(matches),
        "dup_groups": len(groups),
        "dup_group_sizes": dict(collections.Counter(len(g) for g in groups)),
        "dropped": len(drop),
        "kept_matches": len(keep),
        "dropped_by_day": dict(sorted(collections.Counter(D["key_day"][x] for x in drop).items())),
    })
    return m, info, groups


def arr(s, col):
    return s[col].to_numpy().astype(float)


def ratio(x, n, rng=None, boot=False):
    R = float(n.sum())
    X = float(x.sum())
    if R == 0:
        nan2 = (float("nan"), float("nan"))
        return {"est": float("nan"), "V": float("nan"), "R": 0.0, "X": X, "M": len(n), "an": nan2, "boot": nan2}
    r = X / R
    V = float(((x - r * n) ** 2).sum()) / (R * R)
    out = {"est": r, "V": V, "R": R, "X": X, "M": len(n)}
    se = math.sqrt(V)
    out["an"] = (r - Z95 * se, r + Z95 * se)
    if boot:
        idx = rng.integers(0, len(n), size=(B, len(n)))
        rb = x[idx].sum(axis=1) / n[idx].sum(axis=1)
        out["boot"] = tuple(float(v) for v in np.percentile(rb, [2.5, 97.5]))
    return out


def meanM(n, rng=None, boot=False):
    M = len(n)
    mu = float(n.mean())
    s2 = float(n.var(ddof=1)) if M > 1 else float("nan")
    V = s2 / M if M > 1 else float("nan")
    out = {"est": mu, "V": V, "R": float(n.sum()), "X": float(n.sum()), "M": M}
    se = math.sqrt(V) if V == V else float("nan")
    out["an"] = (mu - Z95 * se, mu + Z95 * se)
    if boot:
        idx = rng.integers(0, M, size=(B, M))
        rb = n[idx].mean(axis=1)
        out["boot"] = tuple(float(v) for v in np.percentile(rb, [2.5, 97.5]))
    return out


def het_cluster(stats):
    est = np.array([s["est"] for s in stats])
    V = np.array([s["V"] for s in stats])
    ok = np.isfinite(V) & (V > 0)
    if ok.sum() < 2:
        return float("nan"), 0, float("nan")
    w = 1.0 / V[ok]
    pbar = float((w * est[ok]).sum() / w.sum())
    Q = float((w * (est[ok] - pbar) ** 2).sum())
    df = int(ok.sum() - 1)
    return Q, df, chi2_sf(Q, df)


def het_naive_binom(x, R):
    p = x.sum() / R.sum()
    E = R * p
    Q = float(((x - E) ** 2 / (E * (1 - p))).sum())
    df = len(x) - 1
    return Q, df, chi2_sf(Q, df)


def het_naive_poisson(k, R):
    lam = k.sum() / R.sum()
    E = R * lam
    Q = float(((k - E) ** 2 / E).sum())
    df = len(k) - 1
    return Q, df, chi2_sf(Q, df)


def metric_arrays(s, five):
    if five:
        s = s.filter(pl.col("n5") > 0)
        return arr(s, "n5"), arr(s, "ct5"), arr(s, "pl5"), arr(s, "k5")
    return arr(s, "n"), arr(s, "ct"), arr(s, "pl"), arr(s, "k")


def per_day(m, days, five):
    out = {}
    for d in days:
        s = m.filter(pl.col("day") == d)
        n, ct, pl_, k = metric_arrays(s, five)
        if len(n):
            out[d] = (n, ct, pl_, k)
    return out


def fmt_ci(c, scale=100.0, digits=2):
    return f"{scale * c['est']:.{digits}f} [{scale * c['boot'][0]:.{digits}f}, {scale * c['boot'][1]:.{digits}f}]"


def chi_block(per, label):
    rows = []
    days = list(per)
    ns = [per[d][0] for d in days]
    cts = [per[d][1] for d in days]
    pls = [per[d][2] for d in days]
    ks = [per[d][3] for d in days]
    R = np.array([x.sum() for x in ns])
    CTs = np.array([x.sum() for x in cts])
    PLs = np.array([x.sum() for x in pls])
    KS = np.array([x.sum() for x in ks])
    Mn = np.array([len(x) for x in ns])
    # naive: rounds independent
    qn = het_naive_binom(CTs, R)
    qp = het_naive_binom(PLs, R)
    qk = het_naive_poisson(KS, R)
    # cluster-robust per day
    st_ct = [ratio(cts[i], ns[i]) for i in range(len(days))]
    st_pl = [ratio(pls[i], ns[i]) for i in range(len(days))]
    st_k = [ratio(ks[i], ns[i]) for i in range(len(days))]
    st_r = [meanM(ns[i]) for i in range(len(days))]
    qc = het_cluster(st_ct)
    qpc = het_cluster(st_pl)
    qkc = het_cluster(st_k)
    qrc = het_cluster(st_r)
    rows.append((f"{label} CT win rate", qn, qc))
    rows.append((f"{label} plant rate", qp, qpc))
    rows.append((f"{label} kills per round", qk, qkc))
    rows.append((f"{label} rounds per match", (float("nan"), 0, float("nan")), qrc))
    return rows, days


def fmt_p(p):
    if p != p:
        return "n/a"
    return f"{p:.3g}"


def print_chi(rows, days, title):
    print(f"\n### {title} (days = {len(days)})")
    print("| metric | naive chi2 (df) | naive p | cluster-robust Q (df) | robust p | dispersion Q/df (robust) |")
    print("|---|---|---|---|---|---|")
    for name, qn, qc in rows:
        naive = "n/a" if qn[0] != qn[0] else f"{qn[0]:.1f} ({qn[1]})"
        disp = (qc[0] / qc[1]) if qc[1] else float("nan")
        print(f"| {name} | {naive} | {fmt_p(qn[2])} | {qc[0]:.1f} ({qc[1]}) | {fmt_p(qc[2])} | {disp:.2f} |")


def batch_rows(m, batch_day, five):
    days_all = sorted(set(m["day"].to_list()))
    other = [d for d in days_all if d != batch_day]
    sa = m.filter(pl.col("day") == batch_day)
    sb = m.filter(pl.col("day") != batch_day)
    na, cta, pla, ka = metric_arrays(sa, five)
    nb, ctb, plb, kb = metric_arrays(sb, five)
    out = []
    for name, xa, xb, kind in (("CT win rate", cta, ctb, "ratio"), ("plant rate", pla, plb, "ratio"),
                               ("kills per round", ka, kb, "ratio"), ("rounds per match", na, nb, "mean")):
        if kind == "ratio":
            A = ratio(xa, na)
            Bq = ratio(xb, nb)
        else:
            A = meanM(na)
            Bq = meanM(nb)
        diff = A["est"] - Bq["est"]
        se = math.sqrt(A["V"] + Bq["V"])
        z = diff / se
        p = math.erfc(abs(z) / math.sqrt(2))
        out.append((name, A["est"], Bq["est"], diff, z, p, A["M"], Bq["M"]))
    return out, other


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print(f"polars {pl.__version__}; chi2 backend {CHI2_BACKEND}; bootstrap B={B}; seed={SEED}")
    for tk in ["2026", "2024"]:
        D = load(tk)
        bday = TOMES[tk]["batch_day"]
        print(f"\n\n==================== TOME {tk} ====================")
        print(f"header matches {D['header_n']}; key-path day == match_date day for {D['md_agree']} "
              f"({D['md_agree'] / D['header_n']:.4f})")
        for do_dedupe in (True, False):
            m, info, groups = build(D, do_dedupe)
            tag = "dedupe ON (primary)" if do_dedupe else "dedupe OFF (sensitivity)"
            print(f"\n## Tome {tk}: {tag}")
            print(f"- matches with >=1 round_end row: {info['matches_with_rounds']}; header matches with no round_end: "
                  f"{info['header_without_rounds']}")
            print(f"- duplicate groups {info['dup_groups']} sizes {info['dup_group_sizes']}; dropped {info['dropped']}; "
                  f"kept {info['kept_matches']}; dropped by day {info['dropped_by_day']}")
            print(f"- round_end rows {info['re_rows']}; duplicate (match, round) rows {info['re_dupe_rows']}; "
                  f"unique rounds kept (before dedupe) {info['rounds_unique']}")
            if do_dedupe:
                print(f"- rounds without round_freeze_end (live start): {info['rounds_no_live_start']} "
                      f"({info['rounds_no_live_start_share']:.4f}); freeze_end rows {info['freeze_end_rows']}")
                print(f"- bomb_planted rows {info['bomb_planted_rows']}; planted rounds in window {info['planted_rounds_in_window']}")
                print(f"- death rows {info['death_rows']}; kill rows {info['kill_rows']}; attacker null or <=0: "
                      f"{info['attacker_null_or_le0']}")
                print(f"- winner_team_code counts {info['winner_codes']}; 5v5 share by player_info (exactly 5 T and 5 CT) "
                      f"{info['pi5_share']:.4f}; most common round_end.player_count {info['pc_top']}")
                print(f"- round_state event types (first 40): {info['rs_event_types']}")
                print(f"- bomb_state event types: {info['bs_event_types']}")

            days = sorted(set(m["day"].to_list()))
            if do_dedupe:
                print(f"\n### Per key-day, all rounds (95% CI: match-cluster bootstrap)")
                print("| day | header matches | matches (kept) | rounds | CT win % [CI] | plant % [CI] | rounds/match [CI] | kills/round [CI] |")
                print("|---|---|---|---|---|---|---|---|")
                header_day = collections.Counter(D["key_day"].values())
                for d in days:
                    s = m.filter(pl.col("day") == d)
                    n, ct, pl_, k = metric_arrays(s, False)
                    c_ct = ratio(ct, n, rng, True)
                    c_pl = ratio(pl_, n, rng, True)
                    c_rm = meanM(n, rng, True)
                    c_k = ratio(k, n, rng, True)
                    tagb = " (batch)" if d == bday else ""
                    print(f"| {d}{tagb} | {header_day[d]} | {len(n)} | {int(n.sum())} | {fmt_ci(c_ct)} | "
                          f"{fmt_ci(c_pl)} | {fmt_ci(c_rm, 1.0)} | {fmt_ci(c_k, 1.0, 3)} |")
                # pooled
                n, ct, pl_, k = metric_arrays(m, False)
                c_ct = ratio(ct, n, rng, True)
                c_pl = ratio(pl_, n, rng, True)
                c_rm = meanM(n, rng, True)
                c_k = ratio(k, n, rng, True)
                print(f"| POOLED ({tk}) | {D['header_n']} | {len(n)} | {int(n.sum())} | {fmt_ci(c_ct)} | "
                      f"{fmt_ci(c_pl)} | {fmt_ci(c_rm, 1.0)} | {fmt_ci(c_k, 1.0, 3)} |")

                print(f"\n### Per key-day, 5v5 rounds (exactly 5 T and 5 CT in player_info; analytic cluster-robust 95% CI)")
                print("| day | 5v5 matches | 5v5 rounds | CT win % [CI] | plant % [CI] | rounds/match [CI] | kills/round [CI] |")
                print("|---|---|---|---|---|---|---|")
                for d in days:
                    s = m.filter(pl.col("day") == d)
                    n, ct, pl_, k = metric_arrays(s, True)
                    if len(n) == 0:
                        print(f"| {d} | 0 | 0 | n/a | n/a | n/a | n/a |")
                        continue
                    a = ratio(ct, n)
                    b = ratio(pl_, n)
                    c = meanM(n)
                    e = ratio(k, n)
                    print(f"| {d} | {len(n)} | {int(n.sum())} | {100*a['est']:.2f} [{100*a['an'][0]:.2f}, {100*a['an'][1]:.2f}] | "
                          f"{100*b['est']:.2f} [{100*b['an'][0]:.2f}, {100*b['an'][1]:.2f}] | "
                          f"{c['est']:.2f} [{c['an'][0]:.2f}, {c['an'][1]:.2f}] | "
                          f"{e['est']:.3f} [{e['an'][0]:.3f}, {e['an'][1]:.3f}] |")

            per_all = per_day(m, days, False)
            rows, dd = chi_block(per_all, "all rounds")
            print_chi(rows, dd, f"Tome {tk} chi-square, all rounds, {tag}")
            per_5 = per_day(m, days, True)
            rows5, dd5 = chi_block(per_5, "5v5")
            print_chi(rows5, dd5, f"Tome {tk} chi-square, 5v5 rounds, {tag}")

            ex = [d for d in days if d != bday]
            rowsx, ddx = chi_block(per_day(m, ex, False), "all rounds excl. batch")
            print_chi(rowsx, ddx, f"Tome {tk} chi-square, all rounds, EXCLUDING batch {bday}, {tag}")
            rowsx5, ddx5 = chi_block(per_day(m, ex, True), "5v5 excl. batch")
            print_chi(rowsx5, ddx5, f"Tome {tk} chi-square, 5v5 rounds, EXCLUDING batch {bday}, {tag}")

            if do_dedupe:
                for five in (False, True):
                    out, other = batch_rows(m, bday, five)
                    lab = "5v5" if five else "all rounds"
                    print(f"\n### Batch {bday} vs other {tk} key-days ({lab}; cluster-robust Welch-type z)")
                    print("| metric | batch | rest | diff | z | p | batch matches | rest matches |")
                    print("|---|---|---|---|---|---|---|---|")
                    for name, a_, b_, diff, z, p, ma, mb in out:
                        scale = 1.0 if name in ("rounds per match", "kills per round") else 100.0
                        dd_ = diff * scale
                        print(f"| {name} | {a_*scale:.3f} | {b_*scale:.3f} | {dd_:+.3f} | {z:+.2f} | {p:.3g} | {ma} | {mb} |")
                print(f"\n- other {tk} days used for the batch comparison: {other}")
                print(f"- elapsed {time.time() - t0:.1f}s")
    print(f"\nTOTAL elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
