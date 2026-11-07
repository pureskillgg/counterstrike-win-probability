"""E1-1 verifier, part 2: independent GPU check of the C: cs2gpu xgboost build.

Parent (stdlib only) runs three children with the same C: interpreter:
  normal          device="cuda", the same check the E1 claim makes
  control_no_gpu  CUDA_VISIBLE_DEVICES=-1; device="cuda" must FAIL (negative control)
  no_cuda_path    CUDA Toolkit entries removed from PATH; does the env alone supply the runtime?

Each child reports: build_info, a smoke train, a PID-level check that this process holds a GPU
compute context (nvidia-smi --query-compute-apps) before and during training, the timed CUDA run,
a second CUDA run (run-to-run determinism), a 1-thread CPU run, holdout logloss for each, and the
loaded CUDA-related DLLs plus the PE import names of the xgboost DLL.

Run:
  "C:\\Users\\13144\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_E1_1_gpu.py

ASSUMPTIONS (not from the project): 300,000 train / 50,000 holdout rows (smaller than E1_bench to keep
the job under 3 GB with about 4-6 GB free RAM), 100 trees, max_depth 4, eta 0.1, 50 float32 N(0,1)
features, seed 12345. Timings are contended (other GPU and CPU jobs run on this host).
"""
import json
import mmap
import os
import re
import struct
import subprocess
import sys
import threading
import time

SELF = os.path.abspath(__file__)
N_TRAIN = 300_000
N_HOLD = 50_000
N_FEAT = 50
N_TREES = 100
MAX_DEPTH = 4
ETA = 0.1
SEED = 12345


def pe_import_names(path):
    """Return DLL names from the import and delay-import tables of a PE file (no DLL is loaded)."""
    with open(path, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
            if data[e_lfanew:e_lfanew + 4] != b"PE\0\0":
                raise ValueError("not a PE file")
            fh = e_lfanew + 4
            nsec = struct.unpack_from("<H", data, fh + 2)[0]
            opt_size = struct.unpack_from("<H", data, fh + 16)[0]
            opt = fh + 20
            magic = struct.unpack_from("<H", data, opt)[0]
            dd = opt + (112 if magic == 0x20B else 96)
            secs = []
            for i in range(nsec):
                o = opt + opt_size + i * 40
                vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, o + 8)
                secs.append((vaddr, max(vsize, rawsize), rawptr))

            def off(rva):
                for va, sz, rp in secs:
                    if va <= rva < va + sz:
                        return rva - va + rp
                raise ValueError(f"rva {rva:#x} not mapped")

            def cstr(rva):
                start = off(rva)
                end = data.find(b"\0", start)
                return data[start:end].decode("ascii", "replace")

            names = []
            imp_rva, _ = struct.unpack_from("<II", data, dd + 1 * 8)
            if imp_rva:
                o = off(imp_rva)
                while True:
                    _, _, _, name_rva, _ = struct.unpack_from("<IIIII", data, o)
                    if name_rva == 0:
                        break
                    names.append(cstr(name_rva))
                    o += 20
            d_rva, _ = struct.unpack_from("<II", data, dd + 13 * 8)
            if d_rva:
                o = off(d_rva)
                while True:
                    _, name_rva = struct.unpack_from("<II", data, o)
                    if name_rva == 0:
                        break
                    names.append("(delay) " + cstr(name_rva))
                    o += 32
            return names
        finally:
            data.close()


def loaded_modules():
    import ctypes
    from ctypes import wintypes

    psapi = ctypes.WinDLL("psapi")
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
    ]
    psapi.EnumProcessModulesEx.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD
    h = k32.GetCurrentProcess()
    arr = (wintypes.HMODULE * 2048)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcessModulesEx(h, arr, ctypes.sizeof(arr), ctypes.byref(needed), 0x03):
        return []
    n = min(needed.value // ctypes.sizeof(wintypes.HMODULE), 2048)
    buf = ctypes.create_unicode_buffer(2048)
    out = []
    for i in range(n):
        if psapi.GetModuleFileNameExW(h, arr[i], buf, 2048):
            out.append(buf.value)
    return out


def gpu_compute_pids():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    ).stdout
    return set(int(x) for x in re.findall(r"\d+", out))


def make_data(n_rows, seed):
    import numpy as np

    r = np.random.default_rng(seed)
    X = r.standard_normal((n_rows, N_FEAT), dtype=np.float32)
    logit = (
        0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3]
        + 0.4 * np.sin(2.0 * X[:, 4]) + 0.3 * (X[:, 5] > 0)
        + 0.5 * r.standard_normal(n_rows).astype(np.float32)
    ).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (r.random(n_rows) < p).astype(np.float32)
    return X, y


def logloss(p, y):
    import numpy as np

    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    yy = np.asarray(y, dtype=np.float64)
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


def child_main(variant):
    import re as _re

    import numpy as np

    pid = os.getpid()
    print(f"[{variant}] pid={pid} python={sys.executable}", flush=True)
    print(f"[{variant}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
    cuda_entries = [p for p in os.environ.get("PATH", "").split(";") if "cuda" in p.lower()]
    print(f"[{variant}] PATH entries containing 'cuda': {len(cuda_entries)} {cuda_entries}", flush=True)
    try:
        import xgboost as xgb
    except Exception as e:  # noqa: BLE001 - report verbatim
        print(f"[{variant}] IMPORT FAILED: {type(e).__name__}: {str(e)[:500]}", flush=True)
        return
    bi = xgb.build_info()
    print(f"[{variant}] xgboost {xgb.__version__} {xgb.__file__}", flush=True)
    print(f"[{variant}] build_info USE_CUDA={bi.get('USE_CUDA')} CUDA_VERSION={bi.get('CUDA_VERSION')} "
          f"USE_OPENMP={bi.get('USE_OPENMP')}", flush=True)
    print(f"[{variant}] compute pids before any CUDA use: pid present = {pid in gpu_compute_pids()}", flush=True)

    smoke_params = {"tree_method": "hist", "device": "cuda", "max_depth": 4, "objective": "binary:logistic"}
    Xs = np.random.default_rng(0).random((2000, 5), dtype=np.float32)
    ys = (Xs[:, 0] + Xs[:, 1] > 1).astype(np.float32)
    try:
        xgb.train(smoke_params, xgb.DMatrix(Xs, label=ys), num_boost_round=5)
        print(f"[{variant}] smoke device=cuda: OK", flush=True)
        smoke_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"[{variant}] smoke device=cuda: FAILED {type(e).__name__}: {str(e)[:500]}", flush=True)
        smoke_ok = False

    mods = loaded_modules()
    keep = [m for m in mods if _re.search(r"cudart|nvcuda|cublas|nvrtc|xgboost|cudnn|nccl", m, _re.I)]
    print(f"[{variant}] loaded CUDA-related modules after smoke: {len(keep)}", flush=True)
    for m in keep:
        print(f"[{variant}]   {m}", flush=True)
    dll = [m for m in keep if "xgboost" in m.lower() and m.lower().endswith(".dll")]
    for d in dll[:1]:
        try:
            names = pe_import_names(d)
            print(f"[{variant}] PE imports of {os.path.basename(d)}: {len(names)} DLL names", flush=True)
            cu = [n for n in names if _re.search(r"cuda|cublas|nvrtc|nvcuda|cudart", n, _re.I)]
            print(f"[{variant}] CUDA-related PE imports: {cu}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{variant}] PE parse failed: {e}", flush=True)

    if not smoke_ok:
        return

    Xtr, ytr = make_data(N_TRAIN, SEED)
    Xho, yho = make_data(N_HOLD, SEED + 1)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dho = xgb.DMatrix(Xho)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
              "max_depth": MAX_DEPTH, "eta": ETA, "seed": 0}
    cuda_p = dict(params, device="cuda")
    cpu_p = dict(params, device="cpu", nthread=1)

    xgb.train(cuda_p, dtr, num_boost_round=5)  # untimed warm-up (context and kernel init)
    seen, polls, stop = set(), [0], threading.Event()

    def poll():
        while not stop.is_set():
            try:
                seen.update(gpu_compute_pids())
                polls[0] += 1
            except Exception:  # noqa: BLE001
                pass
            stop.wait(0.25)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.perf_counter()
    b1 = xgb.train(cuda_p, dtr, num_boost_round=N_TREES)
    t_cuda1 = time.perf_counter() - t0
    stop.set()
    th.join(timeout=30)
    print(f"[{variant}] timed cuda run 1: {t_cuda1:.3f} s for {N_TREES} trees on {N_TRAIN} rows", flush=True)
    print(f"[{variant}] our pid seen in GPU compute apps during timed run: {pid in seen} "
          f"(polls={polls[0]}, pids seen={sorted(seen)})", flush=True)
    cfg = json.loads(b1.save_config())
    print(f"[{variant}] config device = {cfg['learner']['generic_param'].get('device')}", flush=True)

    t0 = time.perf_counter()
    b2 = xgb.train(cuda_p, dtr, num_boost_round=N_TREES)
    t_cuda2 = time.perf_counter() - t0
    print(f"[{variant}] timed cuda run 2: {t_cuda2:.3f} s", flush=True)

    t0 = time.perf_counter()
    b3 = xgb.train(cpu_p, dtr, num_boost_round=N_TREES)
    t_cpu = time.perf_counter() - t0
    print(f"[{variant}] timed cpu nthread=1 run: {t_cpu:.3f} s", flush=True)

    p1, p2, p3 = b1.predict(dho), b2.predict(dho), b3.predict(dho)
    print(f"[{variant}] holdout logloss cuda run1={logloss(p1, yho):.6f} cuda run2={logloss(p2, yho):.6f} "
          f"cpu1thread={logloss(p3, yho):.6f}", flush=True)
    print(f"[{variant}] cuda run1 vs cuda run2 max_abs={float(np.max(np.abs(p1 - p2))):.6g} "
          f"mean_abs={float(np.mean(np.abs(p1 - p2))):.6g}", flush=True)
    print(f"[{variant}] cuda run1 vs cpu1thread max_abs={float(np.max(np.abs(p1 - p3))):.6f} "
          f"mean_abs={float(np.mean(np.abs(p1 - p3))):.6f}", flush=True)
    print(f"[{variant}] speedup cpu1thread/cuda(median of 2) = "
          f"{t_cpu / ((t_cuda1 + t_cuda2) / 2):.2f}x", flush=True)


def parent_main():
    print("parent python", sys.version.split()[0], sys.executable, flush=True)
    base_path = os.environ.get("PATH", "")
    no_cuda_path = ";".join(p for p in base_path.split(";") if "cuda" not in p.lower())
    variants = ["normal", "control_no_gpu", "no_cuda_path"]
    for name in variants:
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        if name == "control_no_gpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        if name == "no_cuda_path":
            env["PATH"] = no_cuda_path
        r = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", SELF, "--child", name],
            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500,
        )
        print(f"===== variant {name} exit={r.returncode} =====", flush=True)
        print(r.stdout, end="", flush=True)
        if r.returncode != 0 and r.stderr.strip():
            print("stderr tail:", flush=True)
            print("\n".join(r.stderr.splitlines()[-20:]), flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child_main(sys.argv[2])
    else:
        parent_main()
