"""Train/serve parity of the loadout features: the polars training path (features_status.loadout_features), the pure-Python serving path
(snapshot_features.loadout_from_players) and an independent reference written from a pinned weapon-name table must give the same 46 numbers.

  python -I -X utf8 scripts\\model\\test_snapshot_parity.py [--n 4000] [--days 5] [--seed 0] [--status-dir DIR] [--model-meta PATH] [--out PATH]

Two parts (PASS / FAIL lines like e2e_checks.py, exit code 1 on any FAIL):
  U  synthetic known answers, seconds, no data needed: item-code pins (a wrong SMG list), hand-computed loadouts incl. zero-armor players and
     integer / text side codes, the key set of the serving dict against the pinned list of 46 names (a dropped DIFF field), polars vs python on a
     synthetic player table, and features_status.inconsistent_rounds on rounds where only ONE side's table count disagrees with the grid.
  R  real rows: --days day files spread over the whole 1 Hz history (not only the middle one); in each, --n/--days random (match, round, tick)
     snapshots. The polars features are computed by the production code from exactly the rows the Python side receives (a one-file temp folder,
     so both sides see the same input), and every value of every feature is compared with the Python result and the reference; a NaN on any side
     is a failure. Also: no duplicated (match, round, tick, player) row, and the key sets of all three paths and the model meta agree.
"""
import argparse
import glob
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checks import core  # noqa: E402

core.setup_env(threads=2)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from checks import unit  # noqa: E402
from checks.core import Reporter, fail, notrun, ok, verdict  # noqa: E402

KEYS = ["match_key", "round", "tick"]
PLAYER_COLS = {"side": "side", "health": "health", "armor": "armor", "has_helmet": "has_helmet", "has_defuser": "has_defuser", "is_defusing": "is_defusing",
               "primary": "inv_primary", "flashbang": "inv_flashbang", "smokegrenade": "inv_smokegrenade", "hegrenade": "inv_hegrenade",
               "molotov": "inv_molotov", "incgrenade": "inv_incgrenade", "decoy": "inv_decoy", "equipment_value": "equipment_value_calc"}


def real_rows(args, R):
    import features_status as FS
    from snapshot_features import loadout_from_players
    root = os.path.abspath(args.root or core.project_root())
    status_dir = args.status_dir or os.path.join(root, "data", "status_1hz")
    files = core.list_parquets(status_dir, args.min_age)  # not a file the extractor is still writing
    if not files:
        return notrun(f"no status files in {status_dir}")
    pick = sorted(set(np.linspace(0, len(files) - 1, min(args.days, len(files))).round().astype(int).tolist()))
    per_day = max(1, args.n // len(pick))
    bad, n_rows, worst, n_dup, per_day_n = [], 0, 0.0, 0, {}
    names_pol = names_py = None
    cols = list(dict.fromkeys(FS.STATUS_COLS + ["player_id"]))
    for i in pick:
        S = pl.read_parquet(files[i], columns=cols)
        keys_all = S.select(KEYS).unique()
        keys = keys_all.sample(n=min(per_day, keys_all.height), seed=args.seed) if keys_all.height > per_day else keys_all
        rows = S.join(keys, on=KEYS, how="semi")
        del S
        n_dup += rows.height - rows.unique(KEYS + ["player_id"]).height
        with tempfile.TemporaryDirectory(prefix="wp_sparity_") as d:
            rows.write_parquet(os.path.join(d, "day.parquet"))
            G = keys.with_columns(pl.col("round").cast(pl.Int64))
            F = FS.loadout_features(G, d)
        by = {}
        src = list(PLAYER_COLS.values())
        for tup in rows.select(KEYS + src).iter_rows():
            by.setdefault(tup[:3], []).append(dict(zip(PLAYER_COLS, tup[3:])))
        for k, (mk, rd, tk) in enumerate(keys.select(KEYS).iter_rows()):
            players = by.get((mk, rd, tk), [])
            py = loadout_from_players(players)
            ref = unit.reference_loadout(players)
            if names_pol is None:
                names_pol, names_py = set(F.columns), set(py)
            for c, v in py.items():
                if c not in F.columns:
                    continue
                dp = abs(float(F[c][k]) - v)
                dr = abs(ref[c] - v)
                if not dp <= args.tol:  # written so that NaN (from either side) is a failure, not a silent pass
                    bad.append(f"{os.path.basename(files[i])} {c}: polars {F[c][k]} vs python {v}")
                if not dr <= args.tol:
                    bad.append(f"{os.path.basename(files[i])} {c}: python {v} vs reference {ref[c]}")
                if dp == dp:
                    worst = max(worst, dp)
            n_rows += 1
        per_day_n[os.path.basename(files[i])] = keys.height
    notes = []
    if names_pol != names_py:
        bad.append(f"feature key sets differ: only polars {sorted(names_pol - names_py)}, only python {sorted(names_py - names_pol)}")
    if names_py != unit.ST_NAMES:
        bad.append(f"serving keys differ from the pinned list of 46: missing {sorted(unit.ST_NAMES - names_py)}, extra {sorted(names_py - unit.ST_NAMES)}")
    meta_path = args.model_meta or os.path.join(root, "models", "winprob_v1.meta.json")
    if os.path.exists(meta_path):
        import json
        st = {f for f in json.load(open(meta_path, encoding="utf-8")).get("features", []) if f.startswith("st_")}
        if st - names_py:
            bad.append(f"the model {os.path.basename(meta_path)} needs st_ features the serving path does not produce: {sorted(st - names_py)}")
        notes.append(f"model meta lists {len(st)} st_ features")
    if n_dup:
        bad.append(f"{n_dup} duplicated (match, round, tick, player) rows in the sampled snapshots")
    return verdict(not bad, f"{n_rows} snapshots from {len(pick)} day files {list(per_day_n.values())}, {len(names_py or [])} features each, max |polars - python| = {worst:.3g}; "
                   + "; ".join(notes) + ("; " + "; ".join(bad[:6]) if bad else "; all three paths identical"), n=n_rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train/serve parity of the loadout features (see the module docstring).")
    ap.add_argument("--n", type=int, default=4000, help="snapshots compared in total, spread over the day files")
    ap.add_argument("--days", type=int, default=5, help="number of 1 Hz day files, evenly spread over the history")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--min-age", type=float, default=120.0, help="ignore day files written in the last N seconds")
    ap.add_argument("--root", default=None)
    ap.add_argument("--status-dir", default=None)
    ap.add_argument("--model-meta", default=None)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--skip-real", action="store_true", help="synthetic part only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    R = Reporter(allow_missing=a.allow_missing)
    for name, res in unit.run_suite(unit.SUITE):
        R.add("U-" + name, name.replace("_", " "), res)
    if not a.skip_real:
        R.run("R1", "real rows: polars training features == python serving features == independent reference, on snapshots from day files across the history", lambda: real_rows(a, R))
    print("\n" + R.summary_line(), flush=True)
    print("PASS: identical" if R.all_passed() else "FAIL: see the lines above", flush=True)
    if a.out:
        core.dump_json_atomic({"passed": R.all_passed(), "checks": R.results, "elapsed_s": round(time.time() - t0, 1), "argv": sys.argv[1:]}, a.out)
    sys.exit(0 if R.all_passed() else 1)


if __name__ == "__main__":
    main()
