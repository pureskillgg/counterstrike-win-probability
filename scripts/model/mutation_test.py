"""Mutation test of the safety net: deliberately break the code, one fault at a time, and require the checks to notice.

  python -I -X utf8 scripts\\model\\mutation_test.py [--only W1,G4] [--all-tiers] [--keep] [--report data\\combined\\mutation_report.json]

A check suite that has never failed proves nothing. Each of the 24 mutants in scripts/model/mutants.json is a realistic one-line fault
(a flipped comparison, a swapped side code, a label inverted, a dropped feature, ...) in the serving code, the feature code, the grid
builder, the data preparation or the shared metrics. For each mutant this script
  1. copies scripts/model and scripts/data into a private scratch root (junctions give it READ-ONLY access to the real models/,
     the mini tome store and the 1 Hz player tables; every path constant in the copy is rewritten to the scratch root, so nothing the
     mutant does can touch real data, models or reports),
  2. applies the fault (the 'find' text must occur exactly once, otherwise the mutant is STALE and must be updated),
  3. runs the check tiers that are supposed to catch it (--all-tiers: every tier) and records which one failed.
Before any mutant runs, every needed tier is run on the UNMUTATED copy: if a tier is red there, its mutants are not scored (a check
that fails on good code says nothing about a fault).
Outcome per mutant: CAUGHT (a tier exited non-zero, crashed or timed out), SURVIVED (every tier passed: the safety net has a hole),
STALE (the code no longer contains the fault's anchor text), SKIPPED (baseline red or tier unavailable), ERROR (could not be run).
Exit code 0 only if every mutant is CAUGHT.
"""
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
REAL_ROOT = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
REPARSE = 0x400


# ---------------------------------------------------------------- tiers
def tiers(real):
    """name -> list of steps; a tier fails at the first step that exits non-zero. Paths are relative to the scratch root unless absolute."""
    base = [PY, "-I", "-X", "utf8"]
    sample = os.path.join(real, "data", "combined", "export_winprob_v1_test_sample.parquet")
    return {
        "stress": [base + ["scripts/model/stress_test_winprob.py", "--model", "models/winprob_v1.json", "--fallback", "models/winprob_v1_basic.json",
                                   "--report", "data/combined/stress_mut.json"],
                   base + ["scripts/model/stress_test_winprob.py", "--model", "models/winprob_v1_basic.json", "--report", "data/combined/stress_mut_basic.json"]],
        "parity": [base + ["scripts/model/parity_check.py", "--model", "models/winprob_v1.json", "--fallback", "models/winprob_v1_basic.json",
                           "--sample", sample, "--n", "3000"]],
        "unit_parity": [base + ["scripts/model/test_snapshot_parity.py", "--n", "2000"]],
        "unit": "selftests",  # every scripts/model/selftests/test_*.py, one process each
        "e2e": [base + ["scripts/data/prepare.py", "--store", "mini"],
                base + ["scripts/data/make_grid.py", "--store", "mini"],
                base + ["scripts/model/e2e_checks.py", "--store", "mini", "--model", "winprob_v1", "--fast", "--allow-missing",
                        "--report", "data/mini/e2e_mut.json"]],
    }


CATCHERS = {"stress": ["stress"], "parity": ["parity", "unit_parity"], "unit_parity": ["unit_parity", "unit"], "e2e": ["e2e", "unit"], "unit": ["unit"]}


# ---------------------------------------------------------------- scratch root
def is_reparse(path):
    try:
        return bool(getattr(os.lstat(path), "st_file_attributes", 0) & REPARSE)
    except OSError:
        return False


def junction(link, target):
    if not os.path.exists(target):
        return False
    os.makedirs(os.path.dirname(link), exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", link, target], check=True, capture_output=True)
    return True


def safe_rmtree(path):
    """Remove a scratch root WITHOUT ever following a junction into the real data: junctions are unlinked first, and the tree is walked to
    prove none is left before shutil.rmtree runs."""
    if not os.path.isdir(path):
        return
    stack = [path]
    while stack:
        cur = stack.pop()
        with os.scandir(cur) as it:
            entries = list(it)
        for e in entries:
            if is_reparse(e.path):
                try:
                    os.rmdir(e.path)  # removes only the link of a junction / directory symlink
                except OSError:
                    os.unlink(e.path)
            elif e.is_dir(follow_symlinks=False):
                stack.append(e.path)
    left = [os.path.join(dp, n) for dp, dn, fn in os.walk(path) for n in dn + fn if is_reparse(os.path.join(dp, n))]
    if left:
        raise RuntimeError(f"refusing to delete {path}: reparse points left {left[:3]}")
    shutil.rmtree(path, ignore_errors=True)


def build_root(work, name, real, mutant=None):
    """Scratch root with the (possibly mutated) scripts. Returns (root, error)."""
    root = os.path.join(work, name)
    safe_rmtree(root)
    for sub in ("model", "data"):
        shutil.copytree(os.path.join(real, "scripts", sub), os.path.join(root, "scripts", sub),
                        ignore=shutil.ignore_patterns("__pycache__", "*.stackdump", "*.pyc"))
    if mutant is not None:
        path = os.path.join(root, mutant["file"].replace("/", os.sep))
        if not os.path.isfile(path):
            return root, f"STALE: file {mutant['file']} not found"
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        n = text.count(mutant["find"])
        if n != 1:
            return root, f"STALE: anchor text occurs {n} times in {mutant['file']} (must be exactly once)"
        text = text.replace(mutant["find"], mutant["replace"], 1)
        if path.endswith(".py"):
            try:
                ast.parse(text)
            except SyntaxError as e:
                return root, f"ERROR: mutated file does not parse: {e}"
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    # every path constant in the copy points at the scratch root, never at the real project
    variants = [(real, root), (real.replace("\\", "\\\\"), root.replace("\\", "\\\\")), (real.replace("\\", "/"), root.replace("\\", "/"))]
    for dp, _, fn in os.walk(os.path.join(root, "scripts")):
        for n in fn:
            if n.endswith(".py"):
                p = os.path.join(dp, n)
                with open(p, encoding="utf-8", newline="") as f:
                    t = f.read()
                t2 = t
                for a, b in variants:
                    t2 = t2.replace(a, b)
                if t2 != t:
                    with open(p, "w", encoding="utf-8", newline="") as f:
                        f.write(t2)
    os.makedirs(os.path.join(root, "data", "combined"), exist_ok=True)  # reports of the mutated run land here
    for f in ("export_winprob_v1_test_sample.parquet", "export_winprob_v1_basic_test_sample.parquet"):  # fixtures the stress test's end-to-end check reads (copies)
        src = os.path.join(real, "data", "combined", f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(root, "data", "combined", f))
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    for rel in (("models",), ("tomes", "mini"), ("data", "status_1hz")):  # read-only views of the real inputs
        junction(os.path.join(root, *rel), os.path.join(real, *rel))
    return root, None


# ---------------------------------------------------------------- running
def run_steps(root, name, steps, timeout):
    """-> (status, seconds, detail) with status PASS / FAIL / TIMEOUT / NA."""
    t0 = time.time()
    env = dict(os.environ, MKL_THREADING_LAYER="SEQUENTIAL", POLARS_MAX_THREADS="2", PYTHONIOENCODING="utf-8")
    if steps == "selftests":
        d = os.path.join(root, "scripts", "model", "selftests")
        tests = sorted(f for f in os.listdir(d) if f.startswith("test_") and f.endswith(".py")) if os.path.isdir(d) else []
        if not tests:
            return "NA", 0.0, "no selftests/test_*.py"
        steps = [[PY, "-I", "-X", "utf8", os.path.join("scripts", "model", "selftests", t)] for t in tests]
    log = os.path.join(root, "logs", f"{name}.log")
    for cmd in steps:
        c = [str(x) for x in cmd]
        label = " ".join(os.path.basename(x) for x in c[4:7]) if len(c) > 5 else c[-1]
        try:
            with open(log, "ab") as lf:
                lf.write(f"\n$ {' '.join(c)}\n".encode())
                r = subprocess.run(c, cwd=root, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "TIMEOUT", time.time() - t0, f"timeout after {timeout}s in {label}"
        if r.returncode != 0:
            tail = open(log, "rb").read().decode("utf-8", "replace").strip().splitlines()
            fails = [x for x in tail if x.startswith("FAIL")]
            return "FAIL", time.time() - t0, f"exit {r.returncode} in {label}: " + (fails[0][:160] if fails else (tail[-1][:160] if tail else ""))
    return "PASS", time.time() - t0, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutants", default=os.path.join(HERE, "mutants.json"))
    ap.add_argument("--only", default="", help="comma list of mutant ids or id prefixes (W1, G4, ...)")
    ap.add_argument("--all-tiers", action="store_true", help="run every tier on every mutant (also finds extra catches)")
    ap.add_argument("--tiers", default="", help="comma list: only these tiers exist (others count as unavailable)")
    ap.add_argument("--timeout", type=int, default=1500, help="seconds per step")
    ap.add_argument("--work", default=None, help="scratch directory (default: a fresh folder in the system temp dir)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--report", default=os.path.join(REAL_ROOT, "data", "combined", "mutation_report.json"))
    ap.add_argument("--real-root", default=REAL_ROOT)
    a = ap.parse_args()
    real = os.path.abspath(a.real_root)
    mutants = json.load(open(a.mutants, encoding="utf-8"))
    sel = [x.strip() for x in a.only.split(",") if x.strip()]
    if sel:
        mutants = [m for m in mutants if any(m["id"] == s or m["id"].startswith(s) for s in sel)]
    work = a.work or tempfile.mkdtemp(prefix="wp_mut_")
    os.makedirs(work, exist_ok=True)
    T = tiers(real)
    if a.tiers:
        T = {k: v for k, v in T.items() if k in a.tiers.split(",")}
    print(f"{len(mutants)} mutants; scratch {work}; tiers {list(T)}", flush=True)

    needed = set(T) if a.all_tiers else {t for m in mutants for k in m["should_be_caught_by"] for t in CATCHERS.get(k, [k]) if t in T}
    baseline = {}
    root, err = build_root(work, "baseline", real)
    try:
        for t in sorted(needed):
            st, secs, det = run_steps(root, t, T[t], a.timeout)
            baseline[t] = {"status": st, "seconds": round(secs, 1), "detail": det}
            print(f"  baseline {t:12s} {st:7s} {secs:6.1f}s {det}", flush=True)
    finally:
        if not a.keep:
            safe_rmtree(root)
    usable = {t for t, b in baseline.items() if b["status"] == "PASS"}

    results = []
    for m in mutants:
        want = [t for k in m["should_be_caught_by"] for t in CATCHERS.get(k, [k])] if not a.all_tiers else list(T)
        want = [t for i, t in enumerate(want) if t in usable and t not in want[:i]]
        rec = {"id": m["id"], "file": m["file"], "description": m["description"], "expected_catchers": m["should_be_caught_by"], "tiers_run": {}}
        if not m.get("valid", True):
            rec["status"] = "SKIPPED"; rec["detail"] = "mutant marked invalid in mutants.json"
        elif not want:
            rec["status"] = "SKIPPED"; rec["detail"] = f"no usable tier (baseline red or tier missing) for {m['should_be_caught_by']}"
        else:
            root, err = build_root(work, m["id"], real, m)
            try:
                if err:
                    rec["status"] = err.split(":")[0]; rec["detail"] = err
                else:
                    caught_by = None
                    for t in want:
                        st, secs, det = run_steps(root, t, T[t], a.timeout)
                        rec["tiers_run"][t] = {"status": st, "seconds": round(secs, 1), "detail": det}
                        if st in ("FAIL", "TIMEOUT") and caught_by is None:
                            caught_by = t
                            if not a.all_tiers:
                                break
                    rec["status"] = "CAUGHT" if caught_by else "SURVIVED"
                    rec["caught_by"] = caught_by
            except Exception as e:  # never let one mutant stop the others
                rec["status"] = "ERROR"; rec["detail"] = f"{type(e).__name__}: {e}"
            finally:
                if not a.keep:
                    safe_rmtree(root)
        results.append(rec)
        print(f"{rec['status']:9s} {m['id']:48s} {rec.get('caught_by') or rec.get('detail', '')[:90]}", flush=True)

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nSUMMARY:", counts, flush=True)
    for r in results:
        if r["status"] != "CAUGHT":
            print(f"  {r['status']}: {r['id']}  {r.get('detail', '')}")
    out = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "counts": counts, "all_caught": bool(results) and all(r["status"] == "CAUGHT" for r in results),
           "baseline": baseline, "mutants_file_sha256": hashlib.sha256(open(a.mutants, "rb").read()).hexdigest(), "results": results}
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    tmp = a.report + ".tmp"
    json.dump(out, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, a.report)
    if not a.keep and not a.work:
        safe_rmtree(work)
    sys.exit(0 if out["all_caught"] else 1)


if __name__ == "__main__":
    main()
