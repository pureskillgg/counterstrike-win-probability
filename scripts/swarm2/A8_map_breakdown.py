"""A8 check 8: per-map log loss, non-round-end states, for the in-period 2026 test set (model B,
primary split seed 0) and the cross-period 2024 set (model A, as cross_period.py). Map from the
header map_name. Diagnostic only; the model has no map feature.
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_map_breakdown.py
"""

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ll(y, p):
    return float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))


def per_map(df, y, p, label):
    maps = df["map"].to_list()
    groups = defaultdict(list)
    for i, m in enumerate(maps):
        groups[m].append(i)
    print(f"\n-- {label} --")
    print(f"{'map':<14}{'matches':>8}{'states':>8}{'LL':>9}{'CT-win':>9}{'mean p':>9}")
    rows = []
    for m, idx in groups.items():
        idx = np.array(idx)
        mm = df["match_key"].to_list()
        n_match = len({mm[i] for i in idx})
        rows.append((m, n_match, len(idx), ll(y[idx], p[idx]), float(y[idx].mean()), float(p[idx].mean())))
    for m, nm, ns, l, cw, mp in sorted(rows, key=lambda r: -r[2]):
        print(f"{m:<14}{nm:>8}{ns:>8}{l:>9.4f}{cw:>9.3f}{mp:>9.3f}")
    print(f"{'ALL':<14}{len(set(df['match_key'].to_list())):>8}{len(y):>8}{ll(y, p):>9.4f}"
          f"{float(y.mean()):>9.3f}{float(p.mean()):>9.3f}")


def main():
    cp = load("cross_period", ROOT / "scripts" / "model" / "sufficiency" / "cross_period.py")
    bk, fv = cp.bk, cp.fv
    A = cp.run_2026()
    # 2026 header map
    H26 = fv.read("header", ["key", "map_name"])
    map26 = dict(H26.select(["key", "map_name"]).iter_rows())
    # model B on the 2026 primary test set (as in cross_period.py)
    Xtr_b, ytr_b = bk.xy(A["S_trB"])
    Xva_b, yva_b = bk.xy(A["S_vaB"])
    Xte_b, yte_b = bk.xy(A["S_teB"])
    mB, _, biB, _ = cp.fit_xgb(Xtr_b, ytr_b, Xva_b, yva_b)
    pte = cp.predict(mB, Xte_b, biB)
    te = A["S_teB"]
    not_end_te = np.ones(len(yte_b), dtype=bool)
    ends26 = fv.read("round_end", ["match_key", "round", "tick"])
    e26 = {(m, int(r)): int(t) for m, r, t in ends26.iter_rows()}
    keys = list(zip(te["match_key"].to_list(), [int(r) for r in te["round"].to_list()], te["tick"].to_list()))
    not_end_te = np.array([t != e26.get((m, r), -1) for m, r, t in keys])
    d26 = pl.DataFrame({"match_key": te["match_key"].to_list(), "map": [map26.get(m, "?") for m in te["match_key"].to_list()]})
    per_map(d26.filter(pl.Series(not_end_te)), yte_b[not_end_te], pte[not_end_te],
            "2026 primary test, model B, non-round-end states")

    # 2024 with model A, as cross_period.py
    Xtr_a, ytr_a = bk.xy(A["S_trA"])
    Xva_a, yva_a = bk.xy(A["S_vaA"])
    mA, _, biA, _ = cp.fit_xgb(Xtr_a, ytr_a, Xva_a, yva_a)
    B = cp.run_2024()
    S24 = B["S"]
    X24, y24 = bk.xy(S24)
    p24 = cp.predict(mA, X24, biA)
    old = (fv.TOME, fv.SUFFIX)
    fv.TOME, fv.SUFFIX = cp.TOME_2024, cp.SUFFIX_2024
    try:
        H24 = fv.read("header", ["key", "map_name"])
        E24 = fv.read("round_end", ["match_key", "round", "tick"])
    finally:
        fv.TOME, fv.SUFFIX = old
    map24 = dict(H24.select(["key", "map_name"]).iter_rows())
    e24 = {(m, int(r)): int(t) for m, r, t in E24.iter_rows()}
    k24 = list(zip(S24["match_key"].to_list(), [int(r) for r in S24["round"].to_list()], S24["tick"].to_list()))
    ne24 = np.array([t != e24.get((m, r), -1) for m, r, t in k24])
    d24 = pl.DataFrame({"match_key": S24["match_key"].to_list(),
                        "map": [map24.get(m, "?") for m in S24["match_key"].to_list()]})
    per_map(d24.filter(pl.Series(ne24)), y24[ne24], p24[ne24], "2024-02 kept set, model A, non-round-end states")
    print("\nReading: per-map LL is diagnostic only. Matches per map in 2026 test are small for rare maps.")


if __name__ == "__main__":
    main()
