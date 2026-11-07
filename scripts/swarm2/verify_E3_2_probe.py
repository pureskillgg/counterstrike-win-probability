"""verify_E3_2 probe: runs INSIDE one interpreter, prints one JSON line.

Usage (driver calls this; read-only with respect to the env, -B means no .pyc writes):
  <env>\\python.exe -I -B -X utf8 verify_E3_2_probe.py [--gpu]

Reports: which xgboost is imported and from where, xgboost build_info USE_CUDA,
importability of duckdb / lightgbm / catboost, and (only with --gpu and USE_CUDA true)
a tiny 5000 x 20, 50-round binary:logistic smoke test on device=cuda vs device=cpu.
It also lists which CUDA-related DLLs are loaded in this process, so the driver can
see whether the CUDA runtime came from the env or from the system CUDA toolkit on PATH.
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import site
import sys
import time

GPU = "--gpu" in sys.argv
PAT = re.compile(r"(cudart|cublas|nvrtc|nvcuda|cudnn|xgboost|nccl|cufft|curand|nvjitlink)", re.I)

out = {
    "exe": sys.executable,
    "prefix": sys.prefix,
    "python": sys.version.split()[0],
    "isolated_flag_active": bool(sys.flags.isolated),
    "user_site": site.getusersitepackages(),
    "user_site_exists": os.path.isdir(site.getusersitepackages()),
    "sys_path_head": sys.path[:6],
    "gpu_requested": GPU,
}


def loaded_modules():
    """Names + full paths of loaded modules in this process whose basename matches PAT."""
    psapi = ctypes.WinDLL("psapi")
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wt.HANDLE
    hproc = k32.GetCurrentProcess()
    psapi.EnumProcessModules.argtypes = [wt.HANDLE, ctypes.POINTER(wt.HMODULE), wt.DWORD, ctypes.POINTER(wt.DWORD)]
    psapi.EnumProcessModules.restype = wt.BOOL
    psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, wt.HMODULE, wt.LPWSTR, wt.DWORD]
    psapi.GetModuleFileNameExW.restype = wt.DWORD
    mods = (wt.HMODULE * 4096)()
    needed = wt.DWORD(0)
    if not psapi.EnumProcessModules(hproc, mods, ctypes.sizeof(mods), ctypes.byref(needed)):
        return ["EnumProcessModules failed"]
    n = min(needed.value // ctypes.sizeof(wt.HMODULE), 4096)
    buf = ctypes.create_unicode_buffer(2048)
    res = []
    for i in range(n):
        if psapi.GetModuleFileNameExW(hproc, mods[i], buf, 2048):
            p = buf.value
            if PAT.search(os.path.basename(p)):
                res.append(p)
    return res


def try_import(name):
    t0 = time.perf_counter()
    try:
        mod = __import__(name)
        return {"ok": True, "version": getattr(mod, "__version__", None),
                "file": getattr(mod, "__file__", None), "s": round(time.perf_counter() - t0, 3)}
    except Exception as exc:  # record, do not hide
        return {"ok": False, "err": "%s: %s" % (type(exc).__name__, str(exc)[:240]),
                "s": round(time.perf_counter() - t0, 3)}


out["imports"] = {m: try_import(m) for m in ["xgboost", "duckdb", "lightgbm", "catboost"]}

xgb = None
try:
    import xgboost as xgb  # noqa: F811
    bi = xgb.build_info()
    out["xgboost_version"] = xgb.__version__
    out["USE_CUDA"] = bi.get("USE_CUDA")
    out["USE_NCCL"] = bi.get("USE_NCCL")
    out["xgboost_file"] = xgb.__file__
    out["modules_after_xgboost_import"] = loaded_modules()
except Exception as exc:
    out["xgboost_init_err"] = "%s: %s" % (type(exc).__name__, str(exc)[:400])

if GPU:
    if xgb is None or not out.get("USE_CUDA"):
        out["gpu_smoke"] = "SKIPPED: xgboost not importable or USE_CUDA is not true"
    else:
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5000, 20)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=5000) > 0).astype(np.float32)
        dtr = xgb.DMatrix(X, label=y)
        base = {"objective": "binary:logistic", "max_depth": 4, "eta": 0.1,
                "tree_method": "hist", "seed": 0}
        res, preds = {}, {}
        for dev in ["cuda", "cpu"]:
            t0 = time.perf_counter()
            try:
                b = xgb.train(dict(base, device=dev), dtr, num_boost_round=50)
                preds[dev] = b.predict(dtr)
                cfg = json.loads(b.save_config())
                dev_cfg = cfg.get("learner", {}).get("generic_param", {}).get("device")
                res[dev] = {"ok": True, "s": round(time.perf_counter() - t0, 3), "config_device": dev_cfg}
            except Exception as exc:
                res[dev] = {"ok": False, "err": "%s: %s" % (type(exc).__name__, str(exc)[:400])}
        out["gpu_smoke"] = res
        if "cuda" in preds and "cpu" in preds:
            pg, pc = preds["cuda"], preds["cpu"]
            out["gpu_vs_cpu_max_abs_pred_diff"] = float(np.max(np.abs(pg - pc)))
            d = np.abs(pg.astype(np.float64) - pc.astype(np.float64))
            out["gpu_vs_cpu_mean_abs_pred_diff"] = float(d.mean())
            out["gpu_vs_cpu_p99_abs_pred_diff"] = float(np.quantile(d, 0.99))
            out["gpu_vs_cpu_frac_rows_gt_1e3"] = float((d > 1e-3).mean())
            out["gpu_vs_cpu_pred_mean_cuda"] = float(pg.mean())
            out["gpu_vs_cpu_pred_mean_cpu"] = float(pc.mean())
            eps = 1e-7

            def ll(p):
                p = np.clip(p.astype(np.float64), eps, 1 - eps)
                return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            out["train_logloss_cuda"] = ll(pg)
            out["train_logloss_cpu"] = ll(pc)
        out["modules_after_gpu_smoke"] = loaded_modules()

print(json.dumps(out))
sys.stdout.flush()
