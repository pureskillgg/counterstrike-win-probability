"""verify_E3_2_recompute.py: independent recompute for claim E3-2.

Claim under test: the CUDA XGBoost build (libxgboost / py-xgboost 3.4.2, cuda129) exists only in the
C: cs2gpu env, which lacks duckdb and lightgbm (3 of 3 runs); that env reports USE_CUDA=True, while the
G: project env and the C: csds env report USE_CUDA=False. The claim also proposes a GPU training smoke test.

Driver (stdlib only):
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -B -X utf8 verify_E3_2_recompute.py
Probe (one fresh process per probe, run inside each target env):
  <env>\\python.exe -I -B -X utf8 verify_E3_2_recompute.py --probe <label> [--smoke]

Writes only to logs\\swarm2\\verify_E3_2.log, appended under its own header. The file is shared with
an earlier, separate run (see the header block in that log), so this script never truncates it.
Probes import xgboost/duckdb/lightgbm only and never write to disk. Nothing is written under E:.

ASSUMPTIONS (printed in the log as well):
  A1 smoke data is synthetic: n_train=200000, n_valid=50000, p=30, float32, numpy seed 20261013
  A2 smoke params: binary:logistic, max_depth=6, eta=0.1, 100 rounds, tree_method=hist, max_bin=256, seed=0.
     Config "subsample0.8" adds subsample=0.8. Config "deterministic" uses subsample=1.0 so that
     sampling RNG is removed as a cause of cuda-vs-cpu differences. No tuning. NOT the bake-off settings
     (bakeoff.py pins device=cpu).
  A3 silent CPU fallback = saved config device does not start with the requested device, or a warning
     says "changed from GPU to CPU".
  A4 nvidia-smi memory.used is GPU-wide, so it is a supporting signal only (other GPU users can change it).
  A5 an empty nvidia-smi compute-apps listing means "not visible", not "unused". A PID row listed while the
     probe runs is treated as evidence that the process holds a GPU context.
  A6 "3 of 3 runs" = 3 separate import-only probes per env, not a re-run of E3_import_timing.py.
"""
import importlib
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime

ROOT = r"G:\pureskill\csds-cs2"
LOG = os.path.join(ROOT, "logs", "swarm2", "verify_E3_2.log")
WS = r"C:\Users\13144\.pixi\workspace-envs"
ENVS = [
    ("G_project_default", os.path.join(ROOT, ".pixi", "envs", "default")),
    ("C_csds_detached", os.path.join(WS, "csds-cs2-15611918389529754563", "envs", "default")),
    ("C_cs2gpu", os.path.join(WS, "cs2gpu-15282324412032665829", "envs", "default")),
]
SELF = os.path.abspath(__file__)
RUN_NOTE = ("run 2 of verify_E3_2_recompute (summary code fixed; in run 1 the smoke summary lines printed "
            "None because the nested smoke dict was read one level too shallow; run 1 JSON was correct)")
ASSUME = {
    "A1": "synthetic n_tr=200000 n_va=50000 p=30 float32 seed=20261013",
    "A2": "binary:logistic depth6 eta0.1 100 rounds hist max_bin256 seed0; configs subsample0.8 and deterministic(subsample1.0)",
    "A3": "silent fallback = config device not starting with requested, or 'changed from GPU to CPU' warning",
    "A4": "nvidia-smi memory.used is GPU-wide, supporting signal only",
    "A5": "empty compute-apps listing = not visible; a PID row during the probe = GPU context held",
    "A6": "3-of-3 = 3 import-only probes per env",
}


def nvidia_mem():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return "rc=%s" % r.returncode
        return r.stdout.strip().splitlines()[0].strip()
    except Exception as exc:  # record, do not hide
        return "ERROR %s: %s" % (type(exc).__name__, exc)


def nvidia_apps():
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        return {"rc": r.returncode, "stdout": r.stdout.strip()[:400], "own_pid": os.getpid()}
    except Exception as exc:
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def cfg_get(cfg, *keys):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return "MISSING:" + "/".join(keys)
        cur = cur[k]
    return cur


def train_on(xgb, np, params, dtr, dva, yva, dev, res, tag):
    rec = {"requested_device": dev}
    preds = None
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        try:
            bst = xgb.train(dict(params, device=dev), dtr, num_boost_round=100)
            rec["fit_ok"] = True
            rec["fit_s"] = round(time.perf_counter() - t0, 3)
            cfg = json.loads(bst.save_config())
            rec["config_device_after_fit"] = str(cfg_get(cfg, "learner", "generic_param", "device"))
            rec["n_trees"] = int(bst.num_boosted_rounds())
            if dev == "cuda" and tag == "subsample0.8":
                res["gpu_mem_MiB_after_cuda_fit"] = nvidia_mem()
                res["nvidia_smi_compute_apps_during_cuda_fit"] = nvidia_apps()
            preds = np.asarray(bst.predict(dva), dtype=np.float64)
            eps = 1e-15
            ll = -np.mean(yva * np.log(np.clip(preds, eps, 1.0))
                          + (1.0 - yva) * np.log(np.clip(1.0 - preds, eps, 1.0)))
            rec["valid_logloss"] = round(float(ll), 6)
        except Exception as exc:
            rec["fit_ok"] = False
            rec["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
    rec["warnings"] = [str(w.message)[:220] for w in wl][:6]
    cfg_dev = str(rec.get("config_device_after_fit", ""))
    rec["silent_fallback_flag"] = bool(
        any("changed from GPU to CPU" in m for m in rec["warnings"])
        or (rec.get("fit_ok") and not cfg_dev.startswith(dev)))
    return rec, preds


def smoke(xgb, np):
    res = {"assumptions": "A1 A2 A3 A4 A5"}
    rng = np.random.default_rng(20261013)
    n_tr, n_va, p = 200000, 50000, 30
    X = rng.standard_normal((n_tr + n_va, p), dtype=np.float32)
    logit = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + 0.3 * np.sin(3.0 * X[:, 4])
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_tr + n_va) < prob).astype(np.float32)
    Xtr, ytr, Xva, yva = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    base = dict(objective="binary:logistic", max_depth=6, eta=0.1, tree_method="hist",
                max_bin=256, seed=0)
    configs = {"subsample0.8": dict(base, subsample=0.8),
               "deterministic": dict(base, subsample=1.0)}
    res["gpu_mem_MiB_before_cuda"] = nvidia_mem()
    for tag, params in configs.items():
        sub = {}
        preds = {}
        for dev in ("cuda", "cpu"):
            sub[dev], preds[dev] = train_on(xgb, np, params, dtr, dva, yva, dev, res, tag)
        if preds.get("cuda") is not None and preds.get("cpu") is not None:
            d = np.abs(preds["cuda"] - preds["cpu"])
            sub["valid_pred_max_abs_diff_cuda_vs_cpu"] = float(np.max(d))
            sub["valid_pred_mean_abs_diff_cuda_vs_cpu"] = float(np.mean(d))
        res[tag] = sub
    return res


def probe(label, do_smoke):
    out = {"label": label, "exe": sys.executable, "prefix": sys.prefix,
           "sys_flags_isolated": sys.flags.isolated}
    for m in ("duckdb", "lightgbm", "xgboost"):
        t0 = time.perf_counter()
        try:
            importlib.import_module(m)
            out["import_" + m] = "OK"
        except Exception as exc:
            out["import_" + m] = "ERROR: %s: %s" % (type(exc).__name__, str(exc)[:160])
        out["import_s_" + m] = round(time.perf_counter() - t0, 3)
    hits = {}
    for p in sys.path:
        if not p or not os.path.isdir(p):
            continue
        try:
            names = os.listdir(p)
        except OSError:
            continue
        for pkg in ("duckdb", "lightgbm", "xgboost"):
            for n in names:
                nl = n.lower()
                if nl == pkg or (nl.startswith(pkg + "-") and nl.endswith(".dist-info")):
                    hits.setdefault(pkg, []).append(os.path.join(p, n))
    out["on_disk_hits_on_sys_path"] = hits
    try:
        import xgboost as xgb
    except Exception as exc:
        out["xgb_import_error"] = str(exc)[:200]
        return out
    bi = xgb.build_info()
    out["xgboost_version"] = xgb.__version__
    out["xgboost_file"] = xgb.__file__
    out["build_USE_CUDA"] = bi.get("USE_CUDA")
    out["build_USE_NCCL"] = bi.get("USE_NCCL")
    out["build_CUDA_VERSION"] = bi.get("CUDA_VERSION")
    out["cudart64_12_dll_in_env"] = [
        p for p in (os.path.join(sys.prefix, "Library", "bin", "cudart64_12.dll"),
                    os.path.join(sys.prefix, "Library", "mingw-w64", "bin", "cudart64_12.dll"))
        if os.path.isfile(p)]
    out["cudart64_12_dll_on_PATH"] = [
        d for d in os.environ.get("PATH", "").split(os.pathsep)
        if d and os.path.isfile(os.path.join(d, "cudart64_12.dll"))][:3]
    if do_smoke:
        import numpy as np
        out["smoke"] = smoke(xgb, np)
    return out


def run_probe(exe, label, do_smoke):
    cmd = [exe, "-I", "-B", "-X", "utf8", SELF, "--probe", label]
    if do_smoke:
        cmd.append("--smoke")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=1500)
    except subprocess.TimeoutExpired:
        return {"probe_error": "timeout"}
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("PROBE_JSON ")]
    if not lines:
        return {"probe_rc": r.returncode, "probe_error": "no PROBE_JSON line", "stderr_tail": r.stderr[-800:]}
    return json.loads(lines[-1][len("PROBE_JSON "):])


def nonisolated_find(exe):
    code = ("import importlib.util as u, sys; "
            "print('NONISO isolated=%d missing=%s' % (sys.flags.isolated, "
            "[m for m in ('duckdb', 'lightgbm') if u.find_spec(m) is None]))")
    r = subprocess.run([exe, "-B", "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=300)
    return r.stdout.strip() or r.stderr.strip()[-300:]


def conda_meta_key(prefix):
    cm = os.path.join(prefix, "conda-meta")
    if not os.path.isdir(cm):
        return {"error": "no conda-meta"}
    names = sorted(os.listdir(cm))
    keys = ("xgboost", "duckdb", "lightgbm", "cuda-version", "cudart", "cublas", "nccl")
    return {"n_json": len([n for n in names if n.endswith(".json")]),
            "key": [n for n in names if n.endswith(".json") and any(k in n.lower() for k in keys)]}


def driver():
    lines = []

    def out(s=""):
        lines.append(s)

    out("=" * 72)
    out("verify_E3_2_recompute started %s" % datetime.now().isoformat(timespec="seconds"))
    out("NOTE: " + RUN_NOTE)
    out("driver interpreter: %s  isolated=%s" % (sys.executable, bool(sys.flags.isolated)))
    out("ASSUMPTIONS:")
    for k in sorted(ASSUME):
        out("  %s %s" % (k, ASSUME[k]))

    out("")
    out("== 1. conda-meta sweep (xgboost / duckdb / lightgbm / cuda records)")
    sweep = {}
    if os.path.isdir(WS):
        for d in sorted(os.scandir(WS), key=lambda x: x.name):
            if not d.is_dir():
                continue
            ed = os.path.join(d.path, "envs")
            if os.path.isdir(ed):
                for e in sorted(os.scandir(ed), key=lambda x: x.name):
                    if e.is_dir():
                        sweep[d.name + "/" + e.name] = conda_meta_key(e.path)
    sweep["G_project/default"] = conda_meta_key(ENVS[0][1])
    for name, rec in sweep.items():
        out("  %s: n_json=%s key=%s" % (name, rec.get("n_json"), rec.get("key", rec.get("error"))))

    out("")
    out("== 2. probes (fresh interpreter each, -I -B -X utf8)")
    envs_out = {}
    for label, prefix in ENVS:
        exe = os.path.join(prefix, "python.exe")
        e = {"exe": exe, "exists": os.path.isfile(exe)}
        if e["exists"]:
            e["nonisolated_find_spec"] = nonisolated_find(exe)
            e["import_runs"] = [run_probe(exe, label + "_import_r%d" % (i + 1), False) for i in range(3)]
            e["smoke_probe"] = run_probe(exe, label + "_smoke", True)
        envs_out[label] = e
        out("  done %s" % label)

    out("")
    out("== 3. summary (numbers recomputed in this run)")
    for label, e in envs_out.items():
        if not e.get("exists"):
            out("%s: python.exe MISSING" % label)
            continue
        runs = e["import_runs"]
        r1 = runs[0]
        out("%s: prefix=%s USE_CUDA=%s xgboost=%s" % (
            label, r1.get("prefix"), r1.get("build_USE_CUDA"), r1.get("xgboost_version")))
        for i, r in enumerate(runs):
            out("   import run %d: duckdb=%s lightgbm=%s xgboost=%s %s" % (
                i + 1, r.get("import_duckdb", r.get("probe_error")), r.get("import_lightgbm"),
                r.get("import_xgboost"), ("" if "probe_error" not in r else "(" + r["probe_error"] + ")")))
        out("   nonisolated (no -I) find_spec: %s" % e["nonisolated_find_spec"])
        out("   cudart64_12 in env: %s ; on PATH: %s" % (
            r1.get("cudart64_12_dll_in_env"), r1.get("cudart64_12_dll_on_PATH")))
        sp = e.get("smoke_probe", {})
        if "probe_error" in sp:
            out("   smoke probe failed: %s %s" % (sp.get("probe_error"), sp.get("stderr_tail", "")[-300:]))
            continue
        sm = sp.get("smoke", {})
        for tag in ("subsample0.8", "deterministic"):
            t = sm.get(tag, {})
            for dev in ("cuda", "cpu"):
                r = t.get(dev, {})
                out("   smoke[%s] %s: fit_ok=%s cfg_device=%s silent_fallback=%s valid_logloss=%s fit_s=%s warnings=%s" % (
                    tag, dev, r.get("fit_ok"), r.get("config_device_after_fit"), r.get("silent_fallback_flag"),
                    r.get("valid_logloss"), r.get("fit_s"), r.get("warnings")))
            out("   smoke[%s] cuda vs cpu validation pred: max_abs_diff=%s mean_abs_diff=%s" % (
                tag, t.get("valid_pred_max_abs_diff_cuda_vs_cpu"), t.get("valid_pred_mean_abs_diff_cuda_vs_cpu")))
        out("   GPU mem MiB before cuda / after cuda fit: %s / %s" % (
            sm.get("gpu_mem_MiB_before_cuda"), sm.get("gpu_mem_MiB_after_cuda_fit")))
        out("   compute-apps during cuda fit: %s" % sm.get("nvidia_smi_compute_apps_during_cuda_fit"))
        out("")

    out("== 4. full probe JSON")
    out(json.dumps({"envs": envs_out, "conda_meta_sweep": sweep}, indent=1, default=str))
    out("verify_E3_2_recompute finished %s" % datetime.now().isoformat(timespec="seconds"))

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write("\n" + text)
    print(text)


def main():
    argv = sys.argv[1:]
    if "--probe" in argv:
        label = argv[argv.index("--probe") + 1]
        result = probe(label, "--smoke" in argv)
        print("PROBE_JSON " + json.dumps(result, default=str))
        sys.stdout.flush()
        return
    driver()


if __name__ == "__main__":
    main()
