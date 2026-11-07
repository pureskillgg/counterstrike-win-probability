"""A9 audit probes (read-only):
  P1  XGBoost build and device: USE_CUDA flag; a tiny synthetic fit with device="cuda" (no project data);
      which device the fitted model's config reports, and whether the fallback warning is captured.
  P2  2024-02 max_rounds inference (cross_period.py): for matches with and without the halftime
      announcement event, the round count, final score sum, and the completion rule; the 2026 check
      of the same rule on its header; player_count values in 2024 round_end.

Reads only: tomes/2024-02-g and tomes/2026-10-01_to_08 (parquet, column-projected).
Run (project root): python.exe -B -I -X utf8 scripts\\swarm2\\A9_probe_device_halftime.py
"""

import collections
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
HALF_EVENT = "round_announce_last_round_half"
TOME_2024 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUFFIX_2024 = ".2023-12-28,2024-02-25"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe_device():
    print("== P1 device ==")
    print(f"xgboost {xgb.__version__}; build_info USE_CUDA = {xgb.build_info().get('USE_CUDA')}")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=2000) > 0).astype(int)
    for dev in ("cuda", "cpu"):
        m = xgb.XGBClassifier(n_estimators=20, tree_method="hist", device=dev, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.fit(X, y, verbose=False)
        msgs = [str(w.message)[:140] for w in caught]
        cfg = json.loads(m.get_booster().save_config())
        ran = cfg["learner"]["generic_param"]["device"]
        print(f"  requested {dev}: config device after fit = {ran}; captured Python warnings: {len(msgs)} "
              f"{msgs[:1]}")


def probe_2024(fv):
    print("== P2 2024 halftime rule and completion ==")
    fv.TOME, fv.SUFFIX = TOME_2024, SUFFIX_2024
    hdr = fv.read("header", ["key", "match_id", "match_date", "t_starters_score_final",
                             "ct_starters_score_final"])
    H0 = {r["key"]: r for r in hdr.iter_rows(named=True)}
    rs = fv.read("round_state", ["match_key", "round", "event_type"])
    ev = (rs.filter(pl.col("event_type") == HALF_EVENT).group_by("match_key")
          .agg(pl.col("round").min().alias("r")))
    half24 = {m: int(r) for m, r in ev.iter_rows()}
    re = fv.read("round_end", ["match_key", "round", "player_count", "winner_team_code"])
    n_rounds = re.group_by("match_key").agg(pl.len().alias("n"), pl.col("round").max().alias("mx"))
    nr = {m: (int(n), int(mx)) for m, n, mx in n_rounds.iter_rows()}
    pc = re["player_count"].value_counts().sort("count", descending=True)
    print(f"  2024 header matches {len(H0)}; with halftime event {len(half24)}; "
          f"without {len(H0) - len(half24)}")
    print(f"  2024 round_end player_count values (top): {pc.head(6).rows()}")
    inferred = collections.Counter(2 * half24[m] + 2 for m in half24)
    print(f"  inferred max_rounds from event (2 x r + 2): {dict(sorted(inferred.items()))}")
    over = sum(1 for m in half24 if m in nr and nr[m][1] > 2 * half24[m] + 2)
    print(f"  with-event matches whose max round exceeds the inferred max: {over}")

    no_ev = [m for m in H0 if m not in half24]
    buckets = collections.Counter()
    ex = []
    for m in no_ev:
        n, mx = nr.get(m, (0, 0))
        h = H0[m]
        t, c = h["t_starters_score_final"], h["ct_starters_score_final"]
        top = max(t or 0, c or 0)
        if n == 0:
            buckets["no round_end rows"] += 1
        elif n <= 16:
            buckets["<=16 round_end rows"] += 1
        else:
            buckets[">16 round_end rows"] += 1
        buckets["final top side >= 13"] += int(top >= 13)
        buckets["final top side in 9..12 (16-round win threshold)"] += int(9 <= top <= 12)
        if len(ex) < 6:
            ex.append((m[:40], n, mx, t, c))
    print(f"  no-halftime matches: {len(no_ev)}; buckets {dict(buckets)}")
    print(f"  examples (match, round_end rows, max round, T final, CT final): {ex}")
    short_no_ev = sum(1 for m in no_ev if nr.get(m, (0, 0))[0] <= 16)
    print(f"  no-halftime matches with <=16 round_end rows: {short_no_ev} of {len(no_ev)} "
          f"(hard-coded text in cross_period.py says 'all short')")

    return H0, half24, nr, no_ev


def probe_2026_no_event(fv):
    print("== P2b 2026 matches without the halftime event (header max_rounds) ==")
    fv.TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
    fv.SUFFIX = ".2026-09-18,2026-10-08"
    hdr = fv.read("header", ["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    rs = fv.read("round_state", ["match_key", "round", "event_type"])
    half26 = {m: int(r) for m, r in (rs.filter(pl.col("event_type") == HALF_EVENT)
                                      .group_by("match_key").agg(pl.col("round").min().alias("r"))
                                      .iter_rows())}
    re = fv.read("round_end", ["match_key", "round"])
    nr = dict(re.group_by("match_key").agg(pl.len().alias("n")).iter_rows())
    no_ev = [m for m in H if m not in half26]
    print(f"  2026 header matches {len(H)}; without event {len(no_ev)}")
    print(f"  header max_rounds of those: {dict(collections.Counter(H[m]['max_rounds'] for m in no_ev))}")
    print(f"  round_end rows of those (match, rows, max_rounds): "
          f"{sorted((H[m]['max_rounds'], nr.get(m, 0)) for m in no_ev)}")


def main():
    probe_device()
    spec_fv = load_module("feature_validation", ROOT / "scripts" / "model" / "feature_validation.py")
    probe_2024(spec_fv)
    probe_2026_no_event(spec_fv)


if __name__ == "__main__":
    main()
