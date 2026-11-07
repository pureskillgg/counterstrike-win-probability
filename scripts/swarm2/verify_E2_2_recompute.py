"""verify_E2_2: independent recompute of the E2-2 claim (CatBoost GPU on a shared RTX 3080).

Stdlib only. Reads logs under logs/swarm2 and logs/exp, one read-only nvidia-smi query,
and one read-only GlobalMemoryStatusEx call. Writes no files; the caller redirects stdout to
logs/swarm2/verify_E2_2.log.

Run:
  G:/pureskill/csds-cs2/.pixi/envs/default/python.exe -I -X utf8 verify_E2_2_recompute.py

Every timestamp is the file's last-write time (local). ASSUMPTION markers flag inferences.
"""
import ctypes
import datetime
import json
import math
import os
import re
import statistics
import subprocess
import sys

ROOT = "G:/pureskill/csds-cs2"
LOGS = os.path.join(ROOT, "logs", "swarm2")
EXP = os.path.join(ROOT, "logs", "exp")
NOTES = os.path.join(ROOT, "notes", "swarm2")

OUT = []


def emit(line=""):
    print(line, flush=True)


def read(name, base=LOGS):
    with open(os.path.join(base, name), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def mtime(name, base=LOGS):
    return datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(base, name)))


def hms(dt):
    return dt.strftime("%H:%M:%S")


def tagged_json(tag, text):
    m = re.search(re.escape(tag) + r" (\{.*\})", text)
    return json.loads(m.group(1)) if m else None


def verdict(ok):
    return "MATCH" if ok else "MISMATCH"


# ---------------------------------------------------------------- binomial helpers
def hyper_sf(k_min, N, K, n):
    """P(X >= k_min) for X ~ Hypergeometric(N total, K marked, n drawn). One-sided Fisher exact."""
    total = math.comb(N, n)
    s = 0
    for k in range(k_min, min(K, n) + 1):
        if n - k <= N - K:
            s += math.comb(K, k) * math.comb(N - K, n - k)
    return s / total


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def bisect(f, lo, hi):
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    lower = 0.0 if k == 0 else bisect(lambda p: (1 - binom_cdf(k - 1, n, p)) - alpha / 2, 0.0, 1.0)
    upper = 1.0 if k == n else bisect(lambda p: alpha / 2 - binom_cdf(k, n, p), 0.0, 1.0)
    return lower, upper


# ---------------------------------------------------------------- A. device count
emit("=== verify_E2_2 recompute ===")
emit(f"python: {sys.executable}")
emit(f"run at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local)")
emit()

meta = tagged_json("E2_ENV_JSON", read("E2_env_meta.log"))
dev_count = meta["catboost_gpu_device_count"]
emit("A. Device visibility")
emit(f"   claim: catboost_gpu_device_count 1 | log (E2_env_meta.log): {dev_count} -> {verdict(dev_count == 1)}")
emit("   note: this shows CatBoost sees one GPU. It does not show that a fit ran on it.")
emit()

# ---------------------------------------------------------------- B. retry numbers
rtxt = read("E2_retry_cb_GPU.log")
rj = tagged_json("E2_BENCH_JSON", rtxt)
reps = rj["reps"]
fits = [r["fit_seconds"] for r in reps]
lls = [r["valid_logloss"] for r in reps]
util_mean = [r["gpu_util_mean_pct"] for r in reps]
util_max = [r["gpu_util_max_pct"] for r in reps]
med_fit = statistics.median(fits)
med_ll = statistics.median(lls)
drv = read("E2_retry_driver.log")
warm_s = rj["gpu_warmup_seconds"]

emit("B. Retry run (E2_retry_cb_GPU.log)")
emit(f"   driver line: {[l for l in drv.splitlines() if l.startswith('cb GPU')][0]} "
     f"-> {verdict('cb GPU EXIT=0 17:10:38' in drv)} for exit 0 at 17:10:38")
emit(f"   gpu_ram_part recorded in JSON: {rj['gpu_ram_part']} -> {verdict(rj['gpu_ram_part'] == 0.2)} (claim 0.2)")
emit(f"   warm-up: {warm_s} s -> {verdict(abs(warm_s - 4.561) < 1e-9)} (claim 4.561)")
emit(f"   fit s per rep: {fits} -> {verdict(fits == [25.869, 37.707])} (claim 25.869, 37.707)")
emit(f"   median fit (n=2, so the mean): {med_fit:.3f} -> {verdict(abs(med_fit - 31.788) < 1e-9)} (claim 31.788); "
     f"JSON median_fit_seconds: {rj.get('median_fit_seconds')}")
emit(f"   valid logloss per rep: {lls} -> {verdict(lls == [0.548867, 0.545308])} (claim 0.548867, 0.545308)")
emit(f"   same seed, different logloss between reps: spread {max(lls) - min(lls):.6f} "
     "(claim does not mention this; CPU repeats were identical in E2.md)")
emit(f"   fit spread between reps: {(max(fits) - min(fits)) / min(fits) * 100:.1f}% of the faster rep")
emit(f"   GPU util mean per rep: {util_mean} -> claim '40 to 44%' {verdict(min(util_mean) >= 40 and max(util_mean) <= 44)}")
emit(f"   GPU util max per rep: {util_max} -> claim 100% {verdict(max(util_max) == 100)}")
emit(f"   samples per rep: {[r['gpu_samples'] for r in reps]} (about 0.9 s apart, nvidia-smi is slow under load)")
emit("   task_type_effective: " + str(reps[0]["task_type_effective"]) +
     " (this is model.get_all_params()['task_type'], a parameter readback, not a device check)")
emit()

# ---------------------------------------------------------------- C. every CatBoost GPU attempt
emit("C. Every CatBoost GPU attempt in logs/swarm2 (outcome from log content, not from notes)")
emit(f"   {'file':34s} {'last write':10s} {'gpu_ram_part':34s} {'outcome':18s} baseline_MiB(rep0)")

ATTEMPTS = [
    # file, config as recorded or inferred, note
    ("E2_env_check.log", "unknown (case not logged)", "16:29, default-case error text, no CASE line"),
    ("E2_env_cb_gpu_tiny_default.log", "0.95 (library default, case name)", "tiny 2000x10, 5 trees"),
    ("E2_env_cb_gpu_tiny_ram02.log", "0.2 (case name)", "tiny 2000x10, 5 trees"),
    ("E2_smoke_cb_GPU.log", "0.2 (JSON)", "20k x 20, 20 trees"),
    ("E2_smoke2_cb_GPU.log", "unknown", "0 bytes, no outcome recorded"),
    ("E2_smoke3_cb_GPU.log", "0.2 (JSON)", "20k warm-up, 20k x 20 fit"),
    ("E2_bench_cb_GPU.log", "0.2 (script default; no argv logged)", "first bench attempt, warm-up crash"),
    ("E2_retry_cb_GPU.log", "0.2 (JSON)", "100k x 50, 100 trees, 2 reps"),
]

attempt_rows = []
for fname, cfg, note in ATTEMPTS:
    txt = read(fname)
    size = os.path.getsize(os.path.join(LOGS, fname))
    if "CUDA error 2: out of memory" in txt:
        outcome = "FAIL (CUDA OOM)"
    elif size == 0:
        outcome = "UNKNOWN (0 bytes)"
    elif "status" in txt and '"status": "ok"' in txt:
        outcome = "OK"
    elif re.search(r"rep \d: fit", txt):
        outcome = "OK"
    else:
        outcome = "UNCLASSIFIED"
    js = tagged_json("E2_BENCH_JSON", txt) or tagged_json("E2_ENV_JSON", txt)
    base = None
    if js and js.get("reps"):
        base = js["reps"][0].get("gpu_mem_baseline_mib")
    if js and "gpu_ram_part" in js and js["gpu_ram_part"] is not None:
        cfg = f"{js['gpu_ram_part']} (JSON)"
    if js and js.get("gpu_ram_part") is None and "cb_gpu_tiny_default" in fname:
        cfg = "0.95 (case name)"
    attempt_rows.append({"file": fname, "cfg": cfg, "outcome": outcome, "time": hms(mtime(fname)),
                         "base": base, "note": note, "src": "E2"})
    base_s = f"{base:.0f}" if base is not None else "n/a"
    emit(f"   {fname:34s} {hms(mtime(fname)):10s} {cfg:34s} {outcome:18s} {base_s}")

# X5: a separate CatBoost GPU experiment (logs/exp), run with gpu_ram_part=0.2
x5 = read("X5_catboost_gpu.log", EXP)
x5_lines = x5.splitlines()
x5_fit_lines = [l for l in x5_lines if l.startswith("depth ")]
x5_oom = [l for l in x5_fit_lines if "out of memory" in l.lower()]
x5_other = [l for l in x5_fit_lines if "FAILED" in l and "out of memory" not in l.lower()]
x5_ok = [l for l in x5_fit_lines if " OK " in l]
x5_json = next(json.loads(l) for l in x5_lines if l.startswith('{"name": "X5"'))
x5_end = mtime("X5_catboost_gpu.log", EXP)
x5_start = x5_end - datetime.timedelta(seconds=x5_json["fit_total_s"])
for l in x5_ok:
    attempt_rows.append({"file": "X5 " + l.split()[1] + " (logs/exp)", "cfg": f"{x5_json['gpu_ram_part']} (JSON)",
                         "outcome": "OK", "time": hms(x5_end), "base": None,
                         "note": "1.4M rows; OOM retry wrapper present but never triggered", "src": "X5"})
    emit(f"   {'X5 ' + l.split()[1] + ' (logs/exp)':34s} {hms(x5_end):10s} {str(x5_json['gpu_ram_part']) + ' (JSON)':34s} {'OK':18s} n/a")
emit(f"   X5 GPU fits that failed with CUDA OOM: {len(x5_oom)} | GPU fits that failed for another reason: {len(x5_other)}")
emit("     (X5's other GPU failures: 'change of option monotone_constraints is unimplemented for task type GPU')")
emit(f"   X5 fit run window (ASSUMPTION from file end minus fit_total_s): {hms(x5_start)} to {hms(x5_end)}")
emit()

# ---------------------------------------------------------------- D. occupancy at each attempt
emit("D. Whole-card VRAM occupancy near each attempt (nvidia-smi, WDDM, per-process hidden)")
e1 = read("E1_bench.log")


def e1_val(label):
    m = re.search(r"\[" + re.escape(label) + r"\] gpu\(mem,util\)=(\d+) MiB", e1)
    return int(m.group(1)) if m else None


def e1_dur(pattern):
    return [float(x) for x in re.findall(pattern, e1)]


cd = e1_dur(r"\[cpu_default\] run \d/\d: ([\d.]+) s")
c1 = e1_dur(r"\[cpu_1thread\] run \d/\d: ([\d.]+) s")
cu = e1_dur(r"\[cuda\] run \d/\d: ([\d.]+) s")
e1_end = mtime("E1_bench.log")
t_cuda_after = e1_end
t_cuda_before = t_cuda_after - datetime.timedelta(seconds=sum(cu))
t_c1_after = t_cuda_before
t_c1_before = t_c1_after - datetime.timedelta(seconds=sum(c1))
t_cd_after = t_c1_before
t_cd_before = t_cd_after - datetime.timedelta(seconds=sum(cd))
cd_before_mib = e1_val("cpu_default before timed runs")
cd_after_mib = e1_val("cpu_default after timed runs")
c1_before_mib = e1_val("cpu_1thread before timed runs")
emit(f"   E1 bench last write {hms(e1_end)}; derived bracket of its cpu_default phase (ASSUMPTION: no gaps between prints):")
emit(f"     cpu_default window {hms(t_cd_before)} to {hms(t_cd_after)}; cpu_1thread starts {hms(t_c1_before)}")
emit(f"     whole-card MiB at cpu_default before/after and cpu_1thread before: "
     f"{cd_before_mib} / {cd_after_mib} / {c1_before_mib}")
fail_time = mtime("E2_bench_cb_GPU.log")
inside = t_cd_before <= fail_time <= t_cd_after
emit(f"   CatBoost failure (E2_bench_cb_GPU.log last write {hms(fail_time)}) inside that window: {inside}")
emit(f"   -> whole-card occupancy at the failure, ASSUMPTION: {cd_before_mib} MiB (both bracket ends read the same value)")
emit()
emit("   Successful 0.2 runs with a logged pre-fit baseline (sampler, rep 0):")
occ_rows = []
for r in attempt_rows:
    if r["base"] is not None:
        occ_rows.append(r)
        emit(f"     {r['file']:34s} {r['time']}  baseline {r['base']:.0f} MiB  outcome {r['outcome']}")
emit("   Same-minute pair (E2_env, 16:31): default FAIL, ram02 OK. ram02 nvidia_smi_before = 11214 MiB (E2_env_cb_gpu_tiny_ram02.log).")
emit()
ok_02 = [r for r in attempt_rows if r["cfg"].startswith("0.2") and r["outcome"] == "OK" and r["base"]]
fail_02_occ = cd_before_mib
emit("   Test: does whole-card occupancy separate the 0.2 outcomes?")
emit(f"     0.2 FAIL at 17:00: occupancy {fail_02_occ} MiB (ASSUMPTION, from the E1 bracket)")
emit("     0.2 OK: 12438 MiB (17:10), 16123 MiB (16:44), 17161 MiB (16:34)")
emit("     -> No threshold separates them. The 17:00 failure came at lower occupancy than the 16:44 and 16:34 successes.")
emit("     -> NOT SUPPORTED as the sole mechanism: other jobs holding VRAM does not by itself predict the 0.2 OOM.")
emit()

# ---------------------------------------------------------------- E. counts and tests
emit("E. Default (0.95) vs 0.2 outcome counts and tests (small n, one-sided Fisher exact)")
default_rows = [r for r in attempt_rows if r["cfg"].startswith("0.95") or r["file"] == "E2_env_check.log"]
conf_default = [r for r in attempt_rows if r["file"] in ("E2_env_check.log", "E2_env_cb_gpu_tiny_default.log")]
e2_02 = [r for r in attempt_rows if r["src"] == "E2" and r["cfg"].startswith("0.2")]
x5_02 = [r for r in attempt_rows if r["src"] == "X5"]
all_02 = e2_02 + x5_02
d_fail = sum(1 for r in conf_default if r["outcome"].startswith("FAIL"))
d_n = len(conf_default)
a_fail = sum(1 for r in all_02 if r["outcome"].startswith("FAIL"))
a_n = len(all_02)
emit(f"   default 0.95: {d_fail} FAIL of {d_n} (E2_env_check.log is assumed default; tiny data)")
emit(f"   0.2, E2 only: {sum(1 for r in e2_02 if r['outcome'].startswith('FAIL'))} FAIL of {len(e2_02)}")
emit(f"   0.2, E2 + X5: {a_fail} FAIL of {a_n}")
N = d_n + a_n
K = d_fail + a_fail
p_main = hyper_sf(d_fail, N, K, d_n)
emit(f"   main test (default vs 0.2, E2+X5, n_default={d_n}): P(>= {d_fail} default fails | N={N}, K={K}) = {p_main:.3f}")
N2 = len(conf_default[1:]) + len(e2_02)
K2 = sum(1 for r in conf_default[1:] if r["outcome"].startswith("FAIL")) + sum(1 for r in e2_02 if r["outcome"].startswith("FAIL"))
p_e2 = hyper_sf(1, N2, K2, 1)
emit(f"   sensitivity (E2 only, excluding the unlabeled 16:29 case): default 1 of 1 fail; "
     f"P = {p_e2:.3f} (N={N2}, K={K2})")
lo, hi = clopper_pearson(a_fail, a_n)
emit(f"   0.2 failure rate {a_fail}/{a_n}: 95% exact CI [{lo:.3f}, {hi:.3f}]")
emit("   reading: the default-vs-0.2 difference is not significant at these counts.")
emit("   0.2 is not shown to be safe: the same bench script, data and parameters failed at 17:00 and passed at 17:10.")
emit()

# ---------------------------------------------------------------- F. reservation semantics
emit("F. What gpu_ram_part means (CatBoost 1.2.10 docstring, read-only)")
cbcore = "C:/Users/13144/.pixi/workspace-envs/cs2gpu_alt-16300978804608365045/envs/default/Lib/site-packages/catboost/core.py"
try:
    with open(cbcore, encoding="utf-8", errors="replace") as fh:
        src_lines = fh.read().splitlines()
    idx = next(i for i, l in enumerate(src_lines) if l.strip().startswith("gpu_ram_part : float"))
    emit(f"   {src_lines[idx].strip()}")
    emit(f"   {src_lines[idx + 1].strip()}")
except Exception as exc:  # pragma: no cover
    emit(f"   could not read docstring: {exc!r}")
emit("   The docstring says 'fraction of the GPU RAM'. It does not say 'free'. The claim's 'free VRAM' is an interpretation.")
emit("   Model check: 16:34 succeeded at 0.2 with 3319 MiB free (17161 of 20480 MiB used), so a hard 0.2 x total (4096 MiB)")
emit("   reservation that must fit in free memory is not consistent with that run. The reservation model is not established.")
emit()

# ---------------------------------------------------------------- G. other swarm GPU and host load
emit("G. Other load at the failure window (not logged by E2 at failure time)")
emit(f"   E1 CPU phase ran during the window. Its whole-card readings at the bracket ends were both {cd_before_mib} MiB; none in between.")
emit("   E1.md line 126 (another agent's note, treated as data): commit 70.7 GB of 81.2 GB and free RAM 6.3-7.5 GB at about 17:06.")
emit("   The E2 failure has no RAM or commit reading. Host pinned or commit exhaustion would also yield CUDA error 2 (not tested).")
emit()

# ---------------------------------------------------------------- H. current state, read-only
emit("H. Current state (read-only, no CatBoost run)")
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout.strip()
    emit(f"   nvidia-smi used MiB, total MiB, util %: {smi}")
except Exception as exc:
    emit(f"   nvidia-smi failed: {exc!r}")


class MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


ms = MEMSTAT()
ms.dwLength = ctypes.sizeof(MEMSTAT)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
commit_used = (ms.ullTotalPageFile - ms.ullAvailPageFile) / 2 ** 30
commit_limit = ms.ullTotalPageFile / 2 ** 30
avail_phys = ms.ullAvailPhys / 2 ** 30
emit(f"   host commit charge {commit_used:.2f} GiB of {commit_limit:.2f} GiB limit; free physical RAM {avail_phys:.2f} GiB")
emit("   Live CatBoost GPU run: skipped. Reasons: two other GPU processes are active on the card; a default-0.95 reservation")
emit("   would take most of the free VRAM from them; host commit is near its limit; and one run cannot separate the causes.")
emit()

# ---------------------------------------------------------------- I. claim-by-claim
emit("I. Claim-by-claim check")
rows = [
    ("RTX 3080 visible, catboost_gpu_device_count 1", "MATCH", "E2_env_meta.log"),
    ("GPU runs report task_type_effective GPU, device '0'", "PARTIAL", "parameter readback only; no device-execution proof"),
    ("Retry 17:10:38, gpu_ram_part=0.2, exit 0", "MATCH", "E2_retry_driver.log, E2_retry_cb_GPU.log"),
    ("warm-up 4.561 s; fits 25.869 and 37.707 s; median 31.788 s", "MATCH", "E2_retry_cb_GPU.log"),
    ("valid logloss 0.548867 and 0.545308", "MATCH", "E2_retry_cb_GPU.log (reps differ with same seed, not noted)"),
    ("GPU util mean 40 to 44%, max 100%", "PARTIAL", "whole-card value; other GPU jobs are active"),
    ("default 0.95 aborted with CUDA OOM (env check)", "MATCH", "E2_env_check.log, E2_env_cb_gpu_tiny_default.log"),
    ("0.2 aborted once with the same OOM (17:01)", "PARTIAL", "log has OOM; config inferred from script default, not logged"),
    ("other processes held 8.4 to 16 GB during these runs", "MISMATCH", "8.4 GB is 16:23 (before runs); in-run logged readings 10206 to 17161 MiB"),
    ("0.2 worked, 0.95 failed (the fix)", "PARTIAL", "same-minute pair at 16:31 (tiny data) only; bench 0.2 failed 17:00, passed 17:10"),
    ("shared VRAM occupancy explains the OOMs", "NOT SUPPORTED", "0.2 failure at 13929 MiB (ASSUMPTION); 0.2 successes at 12438, 16123, 17161 MiB"),
    ("revisit once the card is not shared", "UNTESTED", "no idle window; default requirement unknown under the docstring reading"),
]
for claim, v, why in rows:
    emit(f"   [{v:13s}] {claim}")
    emit(f"                  evidence: {why}")
emit()

# ---------------------------------------------------------------- J. verdict input
emit("J. Summary for the verdict")
emit("   Confirmed: the OOM failures happened (default 0.95 twice, 0.2 once). The device count and retry numbers reproduce.")
emit("   Not confirmed: that 0.2 prevents OOM; that VRAM held by other jobs is the cause; that GPU util is CatBoost's.")
emit("   Alternative explanations still open: transient spikes between samples; host commit or RAM pressure (pinned or commit");
emit("   memory, CUDA error 2 can come from the host); WDDM shared-memory behavior; the warm-up model's CUDA context");
emit("   still counted in the rep-0 baseline (bench script keeps 'warm' in scope; size not measured).")
emit("   Verdict: PARTIAL. The OOM events are real. The 0.2 fix and the VRAM-sharing mechanism are not established.")
