"""E2 env check: one case per process, because a native CUDA/LightGBM abort kills the whole interpreter.

Usage (cs2gpu_alt interpreter only, never the PATH python):
  C:\\Users\\13144\\.pixi\\workspace-envs\\cs2gpu_alt-16300978804608365045\\envs\\default\\python.exe -I -X utf8 E2_env_check.py <case>

Cases:
  meta                      versions, nvidia-smi totals, CatBoost GPU device count
  cb_cpu_tiny               CatBoost CPU, 5 trees, 2000 rows
  cb_gpu_tiny_default       CatBoost GPU, 5 trees, library default gpu_ram_part (0.95)
  cb_gpu_tiny_ram02         CatBoost GPU, 5 trees, gpu_ram_part=0.2
  lgb_cpu_tiny              LightGBM device_type=cpu, 5 trees
  lgb_gpu_tiny              LightGBM device_type=gpu (OpenCL path), 5 trees
  lgb_cuda_tiny             LightGBM device_type=cuda, 5 trees
Writes nothing to disk. The caller redirects stdout/stderr to logs\\swarm2\\E2_env_<case>.log.
"""
import json
import platform
import subprocess
import sys
import traceback

import numpy as np

case = sys.argv[1] if len(sys.argv) > 1 else "meta"
print(f"E2_CASE_START {case}", flush=True)

REPORT = {"case": case, "python": sys.version, "executable": sys.executable,
          "platform": platform.platform()}


def smi_query():
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return smi.stdout.strip()
    except Exception as exc:  # pragma: no cover
        return repr(exc)


rng = np.random.default_rng(0)
X = rng.standard_normal((2000, 10)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(2000) > 0).astype(np.int32)

try:
    if case == "meta":
        import lightgbm as lgb
        import catboost as cb
        from catboost.utils import get_gpu_device_count
        REPORT["lightgbm_version"] = lgb.__version__
        REPORT["catboost_version"] = cb.__version__
        REPORT["numpy_version"] = np.__version__
        REPORT["catboost_gpu_device_count"] = int(get_gpu_device_count())
        REPORT["nvidia_smi_before"] = smi_query()

    elif case.startswith("cb_"):
        import catboost as cb
        kwargs = dict(iterations=5, depth=4, learning_rate=0.1, verbose=0,
                      random_seed=0, allow_writing_files=False, thread_count=4)
        if "gpu" in case:
            kwargs.update(task_type="GPU", devices="0")
        if case.endswith("ram02"):
            kwargs.update(gpu_ram_part=0.2)
        REPORT["nvidia_smi_before"] = smi_query()
        model = cb.CatBoostClassifier(**kwargs)
        model.fit(X, y)
        REPORT["status"] = "ok"
        REPORT["tree_count"] = int(model.tree_count_)
        REPORT["task_type_param"] = str(model.get_all_params().get("task_type"))
        REPORT["nvidia_smi_after"] = smi_query()

    elif case.startswith("lgb_"):
        import lightgbm as lgb
        device = {"lgb_cpu_tiny": "cpu", "lgb_gpu_tiny": "gpu", "lgb_cuda_tiny": "cuda"}[case]
        params = {"objective": "binary", "device_type": device, "verbose": 0,
                  "num_leaves": 15, "seed": 0, "num_threads": 4}
        booster = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=5)
        REPORT["status"] = "ok"
        REPORT["num_trees"] = int(booster.num_trees())
    else:
        raise SystemExit(f"unknown case {case}")
except SystemExit:
    raise
except Exception as exc:
    REPORT["status"] = "error"
    REPORT["error_repr"] = repr(exc)
    traceback.print_exc()

print("E2_ENV_JSON " + json.dumps(REPORT, default=str), flush=True)
print(f"E2_CASE_END {case}", flush=True)
