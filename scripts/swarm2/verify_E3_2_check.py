"""verify_E3_2 driver: independent check of claim E3-2 (CUDA XGBoost build lives only in the
C: cs2gpu env, which lacks duckdb and lightgbm).

Run with the project interpreter:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -B -X utf8 verify_E3_2_check.py

Writes ONLY logs\\swarm2\\verify_E3_2.log. Everything it does to the envs is read-only
(probe subprocesses use -I -B; the GPU smoke test is a 5000 x 20, 50-round XGBoost fit).
"""
import datetime
import json
import os
import re
import subprocess
import sys
import time

ROOT = r"G:\pureskill\csds-cs2"
LOG = os.path.join(ROOT, "logs", "swarm2", "verify_E3_2.log")
PROBE = os.path.join(ROOT, "scripts", "swarm2", "verify_E3_2_probe.py")
WS = r"C:\Users\<user>\.pixi\workspace-envs"
GLOBAL_ENVS = r"C:\Users\<user>\.pixi\envs"
PREFIX = {
    "G_project_default": r"G:\pureskill\csds-cs2\.pixi\envs\default",
    "C_detached_csds": os.path.join(WS, "csds-cs2-15611918389529754563", "envs", "default"),
    "C_cs2gpu": os.path.join(WS, "cs2gpu-15282324412032665829", "envs", "default"),
}
TOKENS = [b"cudart64", b"cublas64", b"cublasLt64", b"nvrtc64", b"nvJitLink", b"nvcuda.dll",
          b"cudnn", b"nccl", b"cufft64", b"curand64", b"vcomp140", b"MSVCP140", b"VCRUNTIME140"]

os.makedirs(os.path.dirname(LOG), exist_ok=True)
logf = open(LOG, "w", encoding="utf-8")


def log(s=""):
    print(s)
    logf.write(s + "\n")
    logf.flush()


log("verify_E3_2 log")
log("started: " + datetime.datetime.now().isoformat(timespec="seconds"))
log("driver interpreter: " + sys.executable + "  (isolated flag: %s)" % bool(sys.flags.isolated))
log("claim under test: xgboost USE_CUDA true only in C: cs2gpu env; duckdb and lightgbm missing there (3 of 3 runs)")
log("")

# ---------------------------------------------------------------- 1. conda-meta sweep
log("== 1. conda-meta sweep (all workspace envs, G: project default, global pixi envs)")
PATS = {
    "xgboost": re.compile(r"^(py-xgboost|libxgboost|xgboost)-"),
    "duckdb": re.compile(r"^duckdb-"),
    "lightgbm": re.compile(r"^lightgbm-"),
    "catboost": re.compile(r"^(py-)?catboost-"),
    "cuda_runtime": re.compile(r"^(cuda-cudart|libcublas|cudnn)-"),
    "cuda_version": re.compile(r"^cuda-version-"),
}
roots = []
for d in os.scandir(WS):
    if not d.is_dir():
        continue
    envs = os.path.join(d.path, "envs")
    if os.path.isdir(envs):
        for e in os.scandir(envs):
            if e.is_dir():
                roots.append((d.name + "/envs/" + e.name, os.path.join(e.path, "conda-meta")))
roots.append(("G_project/default", os.path.join(PREFIX["G_project_default"], "conda-meta")))
if os.path.isdir(GLOBAL_ENVS):
    for e in os.scandir(GLOBAL_ENVS):
        if e.is_dir():
            roots.append(("global/" + e.name, os.path.join(e.path, "conda-meta")))

hits = {k: [] for k in PATS}
n_meta = 0
for name, cm in roots:
    if not os.path.isdir(cm):
        continue
    n_meta += 1
    for f in os.listdir(cm):
        if not f.endswith(".json"):
            continue
        for k, p in PATS.items():
            if p.match(f):
                hits[k].append((name, f))
log("env conda-meta dirs scanned: %d" % n_meta)
for k in PATS:
    envs_k = sorted(set(n for n, _ in hits[k]))
    log("  %-13s envs=%d  %s" % (k, len(envs_k), "; ".join(envs_k)[:600]))
cuda_xgb = sorted(set((n, f) for n, f in hits["xgboost"] if "cuda" in f))
log("  CUDA xgboost packages (name contains 'cuda'): %s" % (cuda_xgb if cuda_xgb else "none"))
log("")

# ---------------------------------------------------------------- 2. DLL static scan
log("== 2. static scan of xgboost.dll for CUDA-runtime and MSVC import strings")


def scan_tokens(path):
    counts = {t.decode(): 0 for t in TOKENS}
    maxlen = max(len(t) for t in TOKENS)
    chunk = 16 * 1024 * 1024
    tail = b""
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            data = tail + buf
            for t in TOKENS:
                i = data.find(t)
                while i != -1:
                    if i + len(t) > len(tail):  # count each occurrence once
                        counts[t.decode()] += 1
                    i = data.find(t, i + 1)
            tail = data[-(maxlen - 1):]
    return counts


DLLS = {
    "cs2gpu": os.path.join(PREFIX["C_cs2gpu"], "Library", "mingw-w64", "bin", "xgboost.dll"),
    "G_default": os.path.join(PREFIX["G_project_default"], "Library", "mingw-w64", "bin", "xgboost.dll"),
}
dll_scan = {}
for k, p in DLLS.items():
    if os.path.isfile(p):
        size = os.path.getsize(p)
        dll_scan[k] = scan_tokens(p)
        log("  %s: %s (%d bytes)" % (k, p, size))
        log("    " + json.dumps({a: b for a, b in dll_scan[k].items() if b}, sort_keys=True))
    else:
        dll_scan[k] = None
        log("  %s: MISSING %s" % (k, p))

cs_prefix = PREFIX["C_cs2gpu"]
cudart_in_env = []
for dp, _dn, fn in os.walk(cs_prefix):
    for f in fn:
        if re.match(r"^(cudart64_\d+|cublas64_\d+|cublasLt64_\d+|nvrtc64_\w+|nvJitLink_\w+)\.dll$", f, re.I):
            cudart_in_env.append(os.path.join(dp, f))
log("  CUDA runtime DLLs physically inside cs2gpu env prefix: %s" % (cudart_in_env if cudart_in_env else "none"))
TK = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
tk = []
if os.path.isdir(TK):
    tk = sorted(f for f in os.listdir(TK) if re.match(r"^(cudart64_\d+|cublas64_\d+|nvrtc64_\w+)\.dll$", f, re.I))
log("  system CUDA 12.9 toolkit bin (on PATH) CUDA DLLs: %s" % (tk if tk else "none / dir missing"))
log("")

# ---------------------------------------------------------------- 3. driver GPU state
log("== 3. nvidia-smi (read-only query)")
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    log("  " + (smi.stdout.strip() or smi.stderr.strip())[:400])
except Exception as exc:
    log("  nvidia-smi failed: %s: %s" % (type(exc).__name__, exc))
log("")

# ---------------------------------------------------------------- 4. probe runs
log("== 4. probe runs (each is a fresh interpreter; -I -B -X utf8)")


def run_probe(label, env_key, gpu, path_mode=None):
    py = os.path.join(PREFIX[env_key], "python.exe")
    args = [py, "-I", "-B", "-X", "utf8", PROBE] + (["--gpu"] if gpu else [])
    env = dict(os.environ)
    if path_mode == "strip_cuda_toolkit":
        keep = [p for p in env.get("PATH", "").split(";") if not re.search(r"CUDA\\v\d|NVIDIA GPU Computing Toolkit", p, re.I)]
        env["PATH"] = ";".join(keep)
    t0 = time.perf_counter()
    rec = {"label": label, "env": env_key, "gpu": gpu, "path_mode": path_mode or "as_is"}
    try:
        cp = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=900, env=env, cwd=os.path.dirname(PROBE))
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        rec["rc"] = cp.returncode
        last = (cp.stdout.strip().splitlines() or [""])[-1]
        try:
            rec["probe"] = json.loads(last)
        except Exception:
            rec["probe"] = None
            rec["raw_stdout_tail"] = cp.stdout[-600:]
        if cp.stderr.strip():
            rec["stderr_tail"] = cp.stderr.strip()[-600:]
    except Exception as exc:
        rec["probe"] = None
        rec["exception"] = "%s: %s" % (type(exc).__name__, exc)
    log("  RUN " + label + " :: " + json.dumps(rec, ensure_ascii=False, sort_keys=True))
    return rec


runs = []
runs.append(run_probe("G_default_run1", "G_project_default", False))
runs.append(run_probe("C_csds_run1", "C_detached_csds", False))
runs.append(run_probe("C_cs2gpu_run1_gpu", "C_cs2gpu", True))
runs.append(run_probe("C_cs2gpu_run2", "C_cs2gpu", False))
runs.append(run_probe("C_cs2gpu_run3", "C_cs2gpu", False))
runs.append(run_probe("C_cs2gpu_run4_gpu_noCUDAtoolkitPATH", "C_cs2gpu", True, "strip_cuda_toolkit"))
runs.append(run_probe("G_default_gpu_control", "G_project_default", True))
log("")

# ---------------------------------------------------------------- 5. summary of key numbers
log("== 5. key numbers recomputed from this run")


def probe_of(rec):
    return rec.get("probe") or {}


for r in runs:
    p = probe_of(r)
    imp = p.get("imports", {})

    def st(m):
        x = imp.get(m, {})
        return "ok" if x.get("ok") else ("ERR " + x.get("err", "?")[:60] if x else "no-data")
    gpu = p.get("gpu_smoke")
    gtxt = ""
    if isinstance(gpu, dict):
        gtxt = " gpu_smoke cuda=%s cpu=%s" % (
            (gpu.get("cuda") or {}).get("ok"), (gpu.get("cpu") or {}).get("ok"))
    log("  %-42s USE_CUDA=%s xgb=%s duckdb=[%s] lightgbm=[%s] catboost=[%s]%s" % (
        r["label"], p.get("USE_CUDA"), p.get("xgboost_version"), st("duckdb"), st("lightgbm"), st("catboost"), gtxt))
    if p.get("xgboost_init_err"):
        log("      xgboost_init_err: " + p["xgboost_init_err"])
    if p.get("xgboost_file"):
        log("      xgboost imported from: " + p["xgboost_file"])
    mods = p.get("modules_after_gpu_smoke") or p.get("modules_after_xgboost_import") or []
    if mods:
        log("      loaded CUDA/xgboost modules: " + json.dumps(mods))
    if isinstance(gpu, dict):
        log("      gpu_smoke detail: " + json.dumps(gpu, sort_keys=True))
        if "gpu_vs_cpu_max_abs_pred_diff" in p:
            log("      gpu_vs_cpu_max_abs_pred_diff=%.3g train_logloss cuda=%.6f cpu=%.6f" % (
                p["gpu_vs_cpu_max_abs_pred_diff"], p.get("train_logloss_cuda", float("nan")),
                p.get("train_logloss_cpu", float("nan"))))
    if p.get("user_site_exists") is not None:
        log("      user_site=%s exists=%s" % (p.get("user_site"), p.get("user_site_exists")))
log("")
log("finished: " + datetime.datetime.now().isoformat(timespec="seconds"))
logf.close()
