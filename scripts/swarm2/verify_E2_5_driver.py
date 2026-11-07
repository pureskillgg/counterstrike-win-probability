"""Verifier driver for E2-5 (CatBoost vs LightGBM CPU speed claim).

Run with the cs2gpu_alt interpreter (the project env has no catboost; it is not modified):
  python -B -I -X utf8 verify_E2_5_driver.py
Each fit runs in its own child process (verify_E2_5_child.py), interleaved so that drift affects
both engines alike. Prints a report to stdout (the caller appends it to logs\\swarm2\\verify_E2_5.log).

Protocols:
  claim    the E2 config as written (LightGBM num_leaves 63 / max_bin 255 / 1 thread; CatBoost depth 6 /
           border_count 128 / 1 thread). Median of 3 reps, fit time as E2 timed it.
  matched  the closest settings found (see verify_E2_5_child.py params_for). Median of 3 reps.
  decomp   claim config at 1, 50 and 100 iterations, 2 reps. Splits fit time into a fixed part
           (intercept: CatBoost quantization, LightGBM setup) and a per-tree part (slope).
Also recomputes the E2 medians from the raw E2_BENCH_JSON records in logs\\swarm2\\E2_bench_*.log.
"""
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from ctypes import wintypes

PY = r"C:\Users\13144\.pixi\workspace-envs\cs2gpu_alt-16300978804608365045\envs\default\python.exe"
PROJECT_PY = r"G:\pureskill\csds-cs2\.pixi\envs\default\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "verify_E2_5_child.py")
REPO = r"G:\pureskill\csds-cs2"
SCRATCH = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad"


def say(msg=""):
    print(msg, flush=True)


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


def _ft(ft):
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def sys_times():
    idle, kern, user = FILETIME(), FILETIME(), FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    return _ft(idle), _ft(kern) + _ft(user)


def load_pct(seconds=2.0):
    a = sys_times()
    time.sleep(seconds)
    b = sys_times()
    d_idle, d_total = b[0] - a[0], b[1] - a[1]
    return round(100.0 * (1.0 - d_idle / d_total), 1) if d_total > 0 else None


def python_proc_count():
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
        return sum(1 for line in out.splitlines() if line.lower().startswith('"python.exe"'))
    except Exception:
        return None


def run_child(spec):
    t = time.perf_counter()
    p = subprocess.run([PY, "-B", "-I", "-X", "utf8", CHILD, json.dumps(spec)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=SCRATCH, timeout=1500)
    wall = round(time.perf_counter() - t, 2)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("VERIFY_E2_5_JSON ")]
    if not line:
        return {"status": "no_result", "exit": p.returncode, "stderr_tail": p.stderr[-600:], "spec": spec}
    rec = json.loads(line[-1][len("VERIFY_E2_5_JSON "):])
    rec["child_wall_s"] = wall
    rec["exit"] = p.returncode
    return rec


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def check_project_env():
    say("## Project interpreter check (read-only, no install)")
    code = ("import importlib.util as u, sys; print(sys.version.split()[0]); "
            "print('catboost spec:', bool(u.find_spec('catboost'))); "
            "print('lightgbm spec:', bool(u.find_spec('lightgbm')))")
    p = subprocess.run([PROJECT_PY, "-B", "-I", "-X", "utf8", "-c", code],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    say(p.stdout.strip())
    say("(The project env has no catboost, so CatBoost runs use the cs2gpu_alt interpreter that made the E2 numbers.)")


def check_e2_raw():
    say("## E2 raw records recomputed (logs\\swarm2\\E2_bench_lgb_cpu.log, E2_bench_cb_CPU.log)")
    raw = {}
    for key, name in (("lgb", "E2_bench_lgb_cpu.log"), ("cb", "E2_bench_cb_CPU.log")):
        path = os.path.join(REPO, "logs", "swarm2", name)
        rec = None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if ln.startswith("E2_BENCH_JSON "):
                    rec = json.loads(ln[len("E2_BENCH_JSON "):])
        raw[key] = rec
        fits = [r["fit_seconds"] for r in rec["reps"]]
        builds = [r.get("build_seconds") for r in rec["reps"]]
        say(f"  {key}: fit reps {fits}  median {statistics.median(fits):.4f} s  "
            f"(record median_fit_seconds {rec.get('median_fit_seconds')})  "
            f"build reps {builds}  logloss {[r['valid_logloss'] for r in rec['reps']]}  "
            f"system CPU busy % {[r.get('system_cpu_busy_mean_pct') for r in rec['reps']]}")
    lgb_fit = [r["fit_seconds"] for r in raw["lgb"]["reps"]]
    cb_fit = [r["fit_seconds"] for r in raw["cb"]["reps"]]
    lgb_build = [r["build_seconds"] for r in raw["lgb"]["reps"]]
    m_l, m_c = statistics.median(lgb_fit), statistics.median(cb_fit)
    say(f"  claim ratio, CB fit / LGB fit (medians): {m_c / m_l:.4f}  (claim says 'about 2x')")
    tot = [f + b for f, b in zip(lgb_fit, lgb_build)]
    say(f"  LGB fit+build per rep {[round(t, 4) for t in tot]}, median {statistics.median(tot):.4f} s")
    say(f"  ratio CB fit / (LGB fit+build) using raw E2 numbers: {m_c / statistics.median(tot):.4f}")
    say(f"  mean LGB build share of its timed-window total excluded from the claim: "
        f"{statistics.mean(lgb_build) / statistics.mean(tot):.4f}")
    say()


def summarize(tag, recs):
    say(f"## Protocol: {tag}")
    for eng in ("lgb", "cb"):
        rs = [r for r in recs if r["engine"] == eng and r.get("status") == "ok"]
        if not rs:
            say(f"  {eng}: no ok records")
            continue
        fits = [r["fit_seconds"] for r in rs]
        say(f"  {eng} ({rs[0]['lib_version']}) fit reps {fits} median {fmt(med(fits), 4)} s")
        if eng == "lgb":
            builds = [r["build_seconds"] for r in rs]
            say(f"  lgb build reps {builds} median {fmt(med(builds), 4)} s (excluded from fit)")
            leaves = [x for r in rs for x in r["leaves_per_tree"]]
            say(f"  lgb leaves per tree: mean {statistics.mean(leaves):.2f}, min {min(leaves)}, max {max(leaves)}")
        else:
            slots = [x for r in rs for x in r["leaf_slots_per_tree"]]
            ne = [x for r in rs for x in r["nonempty_leaves_per_tree"] if x is not None]
            say(f"  cb leaf slots per tree: {statistics.mean(slots):.2f}; "
                f"non-empty leaves per tree: mean {statistics.mean(ne):.2f}, min {min(ne)}, max {max(ne)}")
            say(f"  cb effective params: {rs[0]['effective_params']}")
        say(f"  {eng} valid logloss {[r['valid_logloss'] for r in rs]}; "
            f"system CPU busy % during fit {[r.get('system_busy_pct_fit') for r in rs]}")
    # Pair by rep index for per-rep ratios.
    by = {}
    for r in recs:
        if r.get("status") == "ok":
            by.setdefault((r["engine"], r["rep"]), r)
    reps = sorted({r["rep"] for r in recs})
    fit_ratios, total_ratios, lgb_totals, cb_fits = [], [], [], []
    for rep in reps:
        a, b = by.get(("lgb", rep)), by.get(("cb", rep))
        if a and b:
            fit_ratios.append(b["fit_seconds"] / a["fit_seconds"])
            total_ratios.append(b["fit_seconds"] / (a["fit_seconds"] + a["build_seconds"]))
            lgb_totals.append(a["fit_seconds"] + a["build_seconds"])
            cb_fits.append(b["fit_seconds"])
    if fit_ratios:
        lgb_fit_med = med([by[("lgb", r)]["fit_seconds"] for r in reps if ("lgb", r) in by])
        cb_fit_med = med([by[("cb", r)]["fit_seconds"] for r in reps if ("cb", r) in by])
        say(f"  RATIO CB fit / LGB fit (medians, claim-style window): {cb_fit_med / lgb_fit_med:.4f}; "
            f"per-rep {[round(x, 4) for x in fit_ratios]}")
        say(f"  RATIO CB fit / (LGB fit + LGB build) (same window for both): "
            f"{cb_fit_med / med(lgb_totals):.4f}; per-rep {[round(x, 4) for x in total_ratios]}")
    say()


def decomp(recs):
    say("## Protocol: decomp (claim config; fit time vs iterations)")
    table = {}
    for eng in ("lgb", "cb"):
        for it in (1, 50, 100):
            rs = [r for r in recs if r["engine"] == eng and r["iters"] == it and r.get("status") == "ok"]
            if rs:
                table[(eng, it)] = {"fit": med([r["fit_seconds"] for r in rs]),
                                    "build": med([r.get("build_seconds") for r in rs])
                                    if eng == "lgb" else None,
                                    "reps": [r["fit_seconds"] for r in rs]}
    for key in sorted(table):
        say(f"  {key[0]} iters {key[1]:>3}: fit reps {table[key]['reps']} median {fmt(table[key]['fit'], 4)} s"
            + (f"; build median {fmt(table[key]['build'], 4)} s" if key[0] == 'lgb' else ""))
    for eng in ("lgb", "cb"):
        t1, t50, t100 = (table.get((eng, i), {}).get("fit") for i in (1, 50, 100))
        if None in (t1, t50, t100):
            say(f"  {eng}: incomplete decomposition")
            continue
        slope = (t100 - t1) / 99.0
        slope50 = (t50 - t1) / 49.0
        intercept = t1 - slope
        say(f"  {eng}: per-tree slope {slope * 1000:.2f} ms/tree (from 1->100), "
            f"from 1->50 {slope50 * 1000:.2f} ms/tree; intercept (fixed) {intercept:.4f} s; "
            f"predicted t50 {intercept + 50 * slope:.4f} vs measured {t50:.4f}")
    lg, cbm = table.get(("lgb", 100), {}), table.get(("cb", 100), {})
    if lg and cbm and table.get(("lgb", 1)) and table.get(("cb", 1)):
        ls = (table[("lgb", 100)]["fit"] - table[("lgb", 1)]["fit"]) / 99.0
        cs = (table[("cb", 100)]["fit"] - table[("cb", 1)]["fit"]) / 99.0
        li = table[("lgb", 1)]["fit"] - ls
        ci = table[("cb", 1)]["fit"] - cs
        say(f"  per-tree cost ratio CB/LGB: {cs / ls:.4f}; fixed-cost ratio CB/(LGB fit intercept + build): "
            f"{ci / (li + table[('lgb', 1)]['build']):.4f}")
    say()


def main():
    t_start = time.strftime("%H:%M:%S")
    say("=" * 78)
    say(f"VERIFY E2-5 run started {t_start} (independent verifier)")
    say(f"child interpreter: {PY}")
    say("=" * 78)
    check_project_env()
    check_e2_raw()

    runs = []
    for r in range(3):
        order = ("lgb", "cb") if r % 2 == 0 else ("cb", "lgb")
        for proto in ("claim", "matched"):
            for eng in order:
                runs.append({"tag": proto, "engine": eng, "protocol": proto, "iters": 100, "rep": r})
    for r in range(2):
        order = ("lgb", "cb") if r % 2 == 0 else ("cb", "lgb")
        for it in (1, 50, 100):
            for eng in order:
                runs.append({"tag": "decomp", "engine": eng, "protocol": "claim", "iters": it, "rep": r})

    say(f"## Run plan: {len(runs)} child fits, one process each, interleaved")
    results = []
    for i, spec in enumerate(runs):
        child_spec = {"engine": spec["engine"], "protocol": spec["protocol"],
                      "iters": spec["iters"], "scratch": SCRATCH}
        pre_procs = python_proc_count()
        t0 = time.strftime("%H:%M:%S")
        rec = run_child(child_spec)
        rec.update(tag=spec["tag"], rep=spec["rep"])
        results.append(rec)
        if rec.get("status") == "ok":
            extra = (f" build {rec['build_seconds']} s" if rec["engine"] == "lgb" else "")
            say(f"[{i + 1:02d}/{len(runs)}] {t0} {spec['tag']:<8} {spec['engine']} iters={spec['iters']:<3} "
                f"rep={spec['rep']} fit {rec['fit_seconds']} s{extra} logloss {rec['valid_logloss']} "
                f"busy {rec.get('system_busy_pct_fit')} % python.exe procs before {pre_procs} "
                f"wall {rec['child_wall_s']} s")
        else:
            say(f"[{i + 1:02d}/{len(runs)}] {t0} {spec['tag']} {spec['engine']} FAILED: {json.dumps(rec, default=str)[:700]}")
    say()
    say(f"System load (whole machine, 2 s sample) at end: {load_pct()} % busy")
    say()

    for tag in ("claim", "matched"):
        summarize(tag, [r for r in results if r["tag"] == tag])
    decomp([r for r in results if r["tag"] == "decomp"])

    say("## Summary")
    say("  Compare the claim-window ratio with the matched-window ratio above; the claim asserts about 2x.")
    say(f"VERIFY E2-5 run finished {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
