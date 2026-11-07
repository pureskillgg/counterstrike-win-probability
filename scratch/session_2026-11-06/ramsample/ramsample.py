"""Launch N concurrent sequence-trainer processes (scratch output only), sample their peak memory and GPU use, and parse step times.

    python ramsample.py --nprocs 1 --steps 200 --out <scratch dir>
Each process runs scripts/model/train_phase1_arms.py with the pinned trainer, --no_test_load, one seed each, GRU width 128.
"""
import argparse, json, os, re, subprocess, sys, time
import psutil

PY = "G:/pureskill/envs/tabicl/Scripts/python.exe"
ROOT = "G:/pureskill/csds-cs2"

ap = argparse.ArgumentParser()
ap.add_argument("--nprocs", type=int, required=True)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

procs, logs = [], []
for i in range(a.nprocs):
    out_dir = os.path.join(a.out, f"proc{i}")
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(a.out, f"proc{i}.log"), "w")
    cmd = [PY, "-I", "-X", "utf8", "-u", os.path.join(ROOT, "scripts", "model", "train_phase1_arms.py"),
           "--arms", "gru", "--seeds", str(i), "--steps", str(a.steps), "--eval_every", "1000", "--lr", "0.001",
           "--batch", "2048", "--no_test_load", "--width", "128", "--out_dir", out_dir]
    procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
    logs.append(log)

t0 = time.time()
peak_rss = 0.0
peak_gpu_util = 0
samples = []
last_gpu = 0.0
while any(p.poll() is None for p in procs):
    rss = 0.0
    for p in procs:
        if p.poll() is None:
            try:
                ps = psutil.Process(p.pid)
                rss += sum(c.memory_info().rss for c in [ps] + ps.children(recursive=True)) / 2**20
            except psutil.Error:
                pass
    peak_rss = max(peak_rss, rss)
    if time.time() - last_gpu > 10:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            util = int(out.splitlines()[0])
            peak_gpu_util = max(peak_gpu_util, util)
            samples.append({"t": round(time.time() - t0), "rss_mb": round(rss), "gpu_util": util})
        except Exception:
            pass
        last_gpu = time.time()
    time.sleep(2)
for l in logs:
    l.close()

step_times = []
for i in range(a.nprocs):
    text = open(os.path.join(a.out, f"proc{i}.log"), encoding="utf-8", errors="replace").read()
    m = re.findall(r"step (\d+): ([\d.]+) s elapsed", text)
    step_times.append({"proc": i, "elapsed_at_step": {s: float(t) for s, t in m}})

summary = {"nprocs": a.nprocs, "steps": a.steps, "wall_seconds": round(time.time() - t0, 1),
           "peak_total_rss_mb": round(peak_rss), "peak_gpu_util": peak_gpu_util,
           "return_codes": [p.returncode for p in procs], "step_times": step_times, "samples": samples}
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("nprocs", "wall_seconds", "peak_total_rss_mb", "peak_gpu_util", "return_codes")}))
