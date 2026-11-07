"""E1-1 verifier, part 3: does xgboost device="cuda" fall back to CPU silently when no GPU is visible?

Parent (stdlib only) runs two children with the C: cs2gpu interpreter:
  visible  default CUDA_VISIBLE_DEVICES; device="cuda"; expect config device "cuda:0"
  hidden   CUDA_VISIBLE_DEVICES=-1;     device="cuda"; what does xgboost report, and what does it write?

Each child trains 5 rounds on 2,000 rows (the same smoke test as E1_check_build.py). The parent prints the
child's stdout and stderr, because xgboost warnings go to stderr and the E1 logs did not capture stderr.

Run:
  "C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_E1_1_fallback.py
"""
import json
import os
import subprocess
import sys

SELF = os.path.abspath(__file__)


def child(variant):
    import numpy as np
    import xgboost as xgb

    print(f"[{variant}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
    X = np.random.default_rng(0).random((2000, 5), dtype=np.float32)
    y = (X[:, 0] + X[:, 1] > 1).astype(np.float32)
    b = xgb.train({"tree_method": "hist", "device": "cuda", "max_depth": 4,
                   "objective": "binary:logistic"}, xgb.DMatrix(X, label=y), num_boost_round=5)
    cfg = json.loads(b.save_config())
    print(f"[{variant}] train with device=cuda returned normally; config device = "
          f"{cfg['learner']['generic_param'].get('device')}", flush=True)


def parent():
    print("parent python", sys.version.split()[0], sys.executable, flush=True)
    for name, extra in (("visible", {}), ("hidden", {"CUDA_VISIBLE_DEVICES": "-1"})):
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(extra)
        r = subprocess.run([sys.executable, "-I", "-X", "utf8", SELF, "--child", name],
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=600)
        print(f"===== {name} exit={r.returncode} =====", flush=True)
        print(r.stdout, end="", flush=True)
        print(f"--- {name} stderr ({len(r.stderr.splitlines())} lines) ---", flush=True)
        print(r.stderr, end="", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        parent()
