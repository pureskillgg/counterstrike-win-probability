"""A3 duplicate check on the 2024-02-g tome, and across the 2024 and 2026 tomes (read-only).

The bake-off dedupe covers only the 2026 tome. The cross-period result trains on 2026 and tests
on 2024, and the round-13 analysis reads 2024 rows, so duplicates in 2024 would matter there.
Same method as A3_dedupe_audit.py: exact event signature (round_end, player_death, player_info),
then a tick-tolerant pair test (same round, same winner, round_end within 64 ticks).
"""
import collections
import glob
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOMES = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}


def files(tag, ch):
    tome, suffix = TOMES[tag]
    folder = tome / ("header" if ch == "header" else f"basic_{ch}{suffix}")
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no pages in {folder}")
    return fs


def read(tag, ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(tag, ch)], how="vertical_relaxed")


def md5_of(rows):
    return hashlib.md5("\n".join(sorted(repr(r) for r in rows)).encode("utf-8")).hexdigest()[:12]


def load(tag):
    hdr = read(tag, "header", ["key", "match_id", "match_date", "tick_rate",
                               "t_starters_score_final", "ct_starters_score_final"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    RE = read(tag, "round_end", ["match_key", "round", "tick", "winner_team_code"])
    DE = read(tag, "player_death", ["match_key", "round", "tick", "player_id"])
    PI = read(tag, "player_info", ["match_key", "round", "player_id", "team_code"])
    RS = read(tag, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    BS = read(tag, "bomb_state", ["match_key", "round", "tick", "event_type"])
    return H, RE, DE, PI, RS, BS


def main():
    print(f"polars {pl.__version__}")
    data = {}
    for tag in ("2024", "2026"):
        H, RE, DE, PI, RS, BS = load(tag)
        data[tag] = (H, RE, DE, PI, RS, BS)
        years = collections.Counter(k.split("/")[1] for k in H)
        dates = sorted(h["match_date"][:10] for h in H.values())
        print(f"\n== tome {tag} ==")
        print(f"header matches: {len(H)}; unique keys: {len({k for k in H})}; key years: {dict(years)}")
        print(f"match_date range: {dates[0]} .. {dates[-1]}; tick_rate: "
              f"{dict(collections.Counter(h['tick_rate'] for h in H.values()))}")
        print(f"round_end rows: {RE.height}; player_death rows: {DE.height}; player_info rows: {PI.height}")
        print(f"keys in round_end: {RE['match_key'].n_unique()}; in header: {len(H)}; "
              f"round_end keys not in header: {len(set(RE['match_key'].unique().to_list()) - set(H))}")

    # signatures per tome (all matches)
    sigs = {}
    sig_rows = {}
    rec = {}
    for tag in ("2024", "2026"):
        H, RE, DE, PI, RS, BS = data[tag]
        parts = collections.defaultdict(list)
        for x in RE.iter_rows(named=True):
            parts[x["match_key"]].append(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"])))
        for x in DE.iter_rows(named=True):
            parts[x["match_key"]].append(("D", int(x["round"]), int(x["tick"]), str(x["player_id"])))
        for x in PI.iter_rows(named=True):
            parts[x["match_key"]].append(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"])))
        for m, v in parts.items():
            sigs[(tag, m)] = tuple(sorted(repr(p) for p in v))
            sig_rows[(tag, m)] = len(v)
        rec[tag] = {m: 0 for m in H}

    # exact signature groups within each tome and across tomes
    grp = collections.defaultdict(list)
    for (tag, m), s in sigs.items():
        grp[s].append((tag, m))
    print("\n== exact-signature groups (round_end, player_death, player_info) ==")
    for tag in ("2024", "2026", "cross"):
        if tag == "cross":
            gs = [g for g in grp.values() if len({t for t, _ in g}) > 1]
        else:
            gs = [g for g in grp.values() if len(g) > 1 and all(t == tag for t, _ in g)]
        n_m = sum(len(g) for g in gs)
        n_extra = sum(len(g) - 1 for g in gs)
        sizes = collections.Counter(len(g) for g in gs)
        print(f"  {tag}: groups {len(gs)}; matches in groups {n_m}; redundant copies {n_extra}; "
              f"sizes {dict(sorted(sizes.items()))}")
    # which tome each group sits in, and whether the copies agree on other channels
    for tag in ("2024", "2026"):
        H, RE, DE, PI, RS, BS = data[tag]
        gs = [g for g in grp.values() if len(g) > 1 and all(t == tag for t, _ in g)]
        diff_ch = collections.Counter()
        date_gap = []
        hdr_diff = 0
        for g in gs:
            ms = [m for _, m in g]
            for name, df, cols in (("round_state", RS, ["round", "tick", "event_type", "t_score", "ct_score"]),
                                   ("bomb_state", BS, ["round", "tick", "event_type"])):
                per = collections.defaultdict(list)
                for row in df.select(["match_key"] + cols).iter_rows():
                    per[row[0]].append(tuple(row[1:]))
                hs = {md5_of(per.get(m, [])) for m in ms}
                if len(hs) > 1:
                    diff_ch[name] += 1
            if len({(H[m]["t_starters_score_final"], H[m]["ct_starters_score_final"])
                    for m in ms}) > 1:
                hdr_diff += 1
            ds = [m for m in ms]
            from datetime import datetime
            times = [datetime.fromisoformat(H[m]["match_date"].replace("Z", "+00:00")) for m in ds]
            date_gap.append((max(times) - min(times)).total_seconds())
        date_gap.sort()
        print(f"  tome {tag}: groups where round_state or bomb_state copies differ: {dict(diff_ch) or 0}; "
              f"header final score disagrees in {hdr_diff} groups; match_date spread "
              f"(s) min {date_gap[0] if date_gap else 'n/a'}, max {date_gap[-1] if date_gap else 'n/a'}")

    # tick-tolerant pair test across both tomes combined
    print("\n== tick-tolerant pair test (same round, same winner, round_end within 64 ticks), both tomes ==")
    mlist = [(tag, m) for tag in ("2024", "2026") for m in data[tag][0]]
    midx = {k: i for i, k in enumerate(mlist)}
    n_m = len(mlist)
    cnt = np.zeros((n_m, n_m), dtype=np.int16)
    exact = np.zeros((n_m, n_m), dtype=np.int16)
    max_r = 0
    for tag in ("2024", "2026"):
        max_r = max(max_r, int(data[tag][1]["round"].max()))
    for r in range(1, max_r + 1):
        tk = np.full(n_m, np.nan, dtype=np.float32)
        wn = np.full(n_m, -1, dtype=np.int64)
        for tag in ("2024", "2026"):
            sub = data[tag][1].filter(pl.col("round") == r)
            for x in sub.iter_rows(named=True):
                i = midx[(tag, x["match_key"])]
                tk[i] = x["tick"]
                wn[i] = x["winner_team_code"]
        have = ~np.isnan(tk)
        if have.sum() < 2:
            continue
        d = np.abs(tk[:, None] - tk[None, :])
        same_w = (wn[:, None] == wn[None, :]) & have[:, None] & have[None, :]
        cnt += ((d <= 64) & same_w).astype(np.int16)
        exact += ((d == 0) & same_w).astype(np.int16)
        del d, same_w
    iu_i, iu_j = np.where(np.triu(cnt >= 3, k=1))
    print(f"pairs with >=3 tolerant rounds: {len(iu_i)}")
    sig_pairs = []
    non_sig = []
    for i, j in zip(iu_i, iu_j):
        a, b = mlist[i], mlist[j]
        c = int(cnt[i, j])
        if sigs.get(a, ()) == sigs.get(b, ()):
            sig_pairs.append((a, b, c))
        else:
            non_sig.append((a, b, c))
    print(f"  sharing an exact signature: {len(sig_pairs)}; min tolerant rounds among them: "
          f"{min((c for _, _, c in sig_pairs), default=0)}")
    print(f"  not sharing a signature: {len(non_sig)}; max tolerant rounds among them: "
          f"{max((c for _, _, c in non_sig), default=0)}")
    big = [t for t in non_sig if t[2] >= 8]
    print(f"  NON-GROUPED pairs with >=8 tolerant rounds: {len(big)}")
    for a, b, c in big[:20]:
        print(f"     tol_rounds={c} {a} {b}")
    cross_sig = [t for t in sig_pairs if t[0][0] != t[1][0]]
    print(f"  exact-signature pairs across tomes: {len(cross_sig)}")
    for a, b, c in cross_sig[:10]:
        print(f"     {a} {b} rounds={c}")
    # overlap of 2024 duplicate copies with the 2026 tome's rounds is covered by the pair test above


if __name__ == "__main__":
    main()
