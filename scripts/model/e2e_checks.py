"""End-to-end invariant checks over the finished pipeline outputs (data/<store>, tomes/<store>, models/<name>). CPU, polars + numpy + xgboost.

  python -I -X utf8 scripts\\model\\e2e_checks.py --store combined --model winprob_v1 [--fast] [--allow-missing] [--report PATH]
         [--baseline FILE] [--write-baseline FILE] [--root DIR] [--maps de_a,de_b] [--param name=value ...] [--only S,C,D]

Prints PASS / WARN / FAIL / NOT RUN / SKIP per check with the evidence and the size n of the population it looked at. Exit code 1 if any check
FAILs (a NOT RUN counts as FAIL unless --allow-missing; WARN never fails). The JSON report (default data/<store>/e2e_<model>.json) is written
atomically, also when a check raises, and starts as a "running" report with all_passed=false, so an interrupted run cannot leave an old
all_passed=true file in place. It records an input manifest (row counts, mtimes, sha256 of the small files, tome folder listing).

Nothing is hard-coded to a store, era, build number, map or model: the map list comes from the model meta (scope.maps / map_is_*) or --maps,
eras and builds are read from the data, every threshold is a parameter (--param, defaults in checks/ctx.py). --fast skips the raw-tome causal
canary (D12) and scans only a few 1 Hz day files for the player-table canary (G10).

  S  store         header keys vs each channel's keys, tome rounds vs prepared rounds, matches/rounds/deaths/grid agree, tomes not newer than tables
  A  scope         maps (from the model), wingman / hostage, complete 5v5, disconnects, inferred sides, header-vs-roster wingman, excluded maps
  K  completeness  rounds present vs the header score total and numbering 1..n by era|build (WARN for the documented limitation, FAIL past a bound)
  B  duplicates    exact and timeline-based duplicate clusters re-derived independently of prepare: one used match per cluster; null guard
  C  labels        vote rule, bomb / elimination / score signals recomputed, relabels only on first rounds of a half, header-score reconciliation
                   of complete matches (orientation-free), per-signal agreement with the bomb outcome
  D  grid          schema, contiguity, 64-tick steps, anchor to the round table (clock, plant, span, points per round), label, recompute of alive
                   counts from deaths (as-of join) and, in the full run, from the raw tome events with only tick <= row tick
  W  orientation   T-wipe only after the plant; CT wins more often when ahead, per era and map
  E  split         disjoint / covering / documented sizes (legacy rule), frozen manifest (sha256, coverage, model counts), cross-artifact leakage
  F  features      allowlist, key-set equality serving vs polars vs model, item-code pins, single-feature leakage screen
  G  model         booster reproduces the pipeline, calibration (pooled and per subgroup), discrimination floor, determinism, monotone constraints,
                   snapshot physical bounds, player table vs grid at every tick, player-table domain canary
A check that filters for violations treats a NULL in a checked column as a violation (a null never passes).
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checks import core  # noqa: E402

core.setup_env(threads=2)
from checks import ctx as C  # noqa: E402
from checks import grid_checks, label_checks, model_checks, split_checks, store_checks  # noqa: E402
from checks.core import Reporter, fail  # noqa: E402

PROJECT_ROOT = core.project_root()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="End-to-end invariant checks (see the module docstring).")
    ap.add_argument("--store", default="combined")
    ap.add_argument("--model", default="winprob_v1")
    ap.add_argument("--root", default=PROJECT_ROOT, help="project root (default: the root of this checkout)")
    ap.add_argument("--report", default=None, help="JSON report path (default data/<store>/e2e_<model>.json)")
    ap.add_argument("--fast", action="store_true", help="skip the raw-tome causal canary; scan few 1 Hz day files")
    ap.add_argument("--allow-missing", action="store_true", help="a check that needs a missing file is NOT RUN and does not fail the run")
    ap.add_argument("--baseline", default=None, help="JSON of accepted shares (see --write-baseline): a share worse than baseline + tolerance FAILs")
    ap.add_argument("--write-baseline", default=None, help="write the measured shares of this run as a baseline file")
    ap.add_argument("--maps", default=None, help="comma list of in-scope maps (default: from the model meta)")
    ap.add_argument("--tomes", default=None, help="tome folder (default tomes/<store>/tome/csds)")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--status-dir", default=None)
    ap.add_argument("--sample", default=None, help="held-out sample parquet (default data/<store>/export_<model>_test_sample.parquet)")
    ap.add_argument("--split-manifest", default=None, help="frozen split manifest path, or 'none' (default data/<store>/split_manifest.json if present)")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="override a threshold (repeatable)")
    ap.add_argument("--only", default=None, help="comma list of check-id prefixes to run (S, A, K, B, C, D, W, E, F, G, or e.g. C9)")
    return ap.parse_args(argv)


def write_report(path, R, ctx, P, args, status, t0):
    rep = {"status": status, "all_passed": status == "complete" and R.all_passed(), "store": ctx.store, "model": ctx.model_name, "root": ctx.root,
           "argv": sys.argv[1:], "fast": ctx.fast, "allow_missing": R.allow_missing, "counts": R.counts(), "checks": R.results,
           "failed": [r["id"] for r in R.failed()], "warnings": [r["id"] for r in R.warned()],
           "params": P.as_dict(), "inputs": getattr(ctx, "manifest", None), "measured": ctx.measured,
           "elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": core.peak_rss_mb(), "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "python": sys.version.split()[0]}
    try:
        import polars as pl
        import numpy as np
        rep["polars"], rep["numpy"] = pl.__version__, np.__version__
    except Exception:  # noqa: BLE001
        pass
    core.dump_json_atomic(rep, path)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    P = core.Params(C.DEFAULTS, args.param)
    R = Reporter(allow_missing=args.allow_missing)
    baseline = {}
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            baseline = json.load(f)
    ctx = C.Ctx(args.root, args.store, args.model, P, R, tomes=args.tomes, models_dir=args.models_dir, status_dir=args.status_dir, sample=args.sample,
                maps=args.maps, split_manifest=args.split_manifest, baseline=baseline, fast=args.fast, only=args.only)
    report = args.report or os.path.join(ctx.data_dir, f"e2e_{args.model}.json")
    write_report(report, R, ctx, P, args, "running", t0)  # a crash from here on leaves an explicit non-passing report, never an old all_passed=true
    status = "crashed"
    try:
        def group(name, fn, needs=(), sel=None):
            if not any(ctx.want(s) for s in (sel or (name,))):
                return
            miss = ctx.missing(*needs) if needs else []
            if miss:
                R.add(f"{name}*", f"{name} checks", core.notrun(f"missing {miss} in {ctx.data_dir}"))
                return
            try:
                fn(ctx)
            except Exception as exc:  # noqa: BLE001 - the contract: a broken group is a recorded FAIL, the run continues
                R.add(f"{name}*", f"{name} checks", fail(f"group raised {type(exc).__name__}: {str(exc)[:300]}"), tb=traceback.format_exc())

        if ctx.want("S"):
            R.run("S0", "input manifest: prepared tables exist and their row counts, sizes, mtimes and sha256 are recorded", lambda: store_checks.check_manifest(ctx))
        group("S", store_checks.store_consistency)
        group("A", store_checks.scope_checks, ("matches.parquet", "rounds.parquet"))
        group("K", store_checks.completeness_checks, ("matches.parquet", "rounds.parquet"))
        group("B", store_checks.duplicate_checks, ("matches.parquet", "rounds.parquet"))
        group("C", label_checks.label_checks, ("matches.parquet", "rounds.parquet"))
        group("D", grid_checks.grid_checks, ("matches.parquet", "rounds.parquet", "grid.parquet"))
        group("E", split_checks.split_checks, ("matches.parquet", "rounds.parquet", "grid.parquet"))
        group("FG", model_checks.model_checks, ("matches.parquet", "rounds.parquet", "grid.parquet"), sel=("F", "G"))
        group("G10", model_checks.status_domain, sel=("G",))
        status = "complete"
    except BaseException as exc:  # noqa: BLE001 - KeyboardInterrupt included: the report must still say what happened
        R.add("RUN", "run aborted", fail(f"{type(exc).__name__}: {str(exc)[:300]}"), tb=traceback.format_exc())
        status = "aborted"
        raise
    finally:
        if not R.results:
            R.add("RUN", "at least one check ran", fail("no check ran (--only matched nothing, or the run aborted before the first check)"))
        write_report(report, R, ctx, P, args, status, t0)
        if args.write_baseline:
            core.dump_json_atomic(ctx.measured, args.write_baseline)
        print("\n" + R.summary_line(), flush=True)
        print(f"report -> {report}   ({time.time() - t0:.0f}s, peak {core.peak_rss_mb()} MiB)", flush=True)
    sys.exit(0 if R.all_passed() else 1)


if __name__ == "__main__":
    main()
