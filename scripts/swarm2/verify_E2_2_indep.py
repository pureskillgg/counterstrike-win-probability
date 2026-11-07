"""verify_E2_2_indep: independent recomputation of the E2-2 claim (CatBoost GPU CUDA OOM on the shared RTX 3080).

Read-only, stdlib only. Project interpreter, explicitly:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_E2_2_indep.py
Output goes to stdout; the caller saves it as logs\\swarm2\\verify_E2_2_indep.log.
No CatBoost is installed in the project env, so no CatBoost fit is run here.
Inputs (read only): logs\\swarm2\\E2_*.log, logs\\swarm2\\E1_bench.log, logs\\exp\\X5_catboost_gpu.log,
the DNABERT run logs (other GPU tenant), read-only nvidia-smi, read-only Windows event and commit queries.

ASSUMPTIONS (not measured):
  A1. CatBoost asks for about gpu_ram_part x free VRAM at init (the 0.95 default is 'fraction of GPU RAM').
  A2. The E2_bench baseline window [t0 - 4 s, t0) lies inside CatBoost warm-up when warm-up >= 4 s.
  A3. E1_bench.log's last write (17:04:28) is the end of its last run, and its phases run back to back.
"""
import base64
import datetime as dt
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys

ROOT = r"G:\pureskill\csds-cs2"
LOGS = os.path.join(ROOT, "logs", "swarm2")
EXP = os.path.join(ROOT, "logs", "exp")
DNABERT = r"C:\Users\<user>\Documents\dnabert-error-correction\reference\megasprint_stage5_finetuning"
RUN_LOGS = {
    "moderngena/linear": os.path.join(DNABERT, "moderngena", "linear", "seed_1", "run.log"),
    "dnabert2/linear": os.path.join(DNABERT, "dnabert2", "linear", "seed_1", "run.log"),
}
TOTAL_MIB = 20480
WARMUP_OVERLAP_S = 4.0
DAY = dt.date(2026, 10, 13)
RUN_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} INFO step (\d+)/\d+ .*peak_reserved_mib=(\d+)")


def T(hhmmss):
    parts = [int(p) for p in hhmmss.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return dt.datetime(DAY.year, DAY.month, DAY.day, parts[0], parts[1], parts[2])


def out(*args):
    print(*args, flush=True)


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def mtime(path):
    return dt.datetime.fromtimestamp(os.path.getmtime(path))


def tagged_json(text, tag):
    for line in text.splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1:])
    return None


def smi_total_used(field):
    # example: "NVIDIA GeForce RTX 3080, 591.86, 20480 MiB, 11214 MiB, 22 %"
    parts = [p.strip() for p in field.split(",")]
    return int(parts[2].split()[0]), int(parts[3].split()[0])


def guess_device(name, bench, case, oom):
    if bench:
        return str(bench.get("device")).upper()
    if oom:
        return "GPU"
    if case and "gpu" in case:
        return "GPU"
    if case and "cpu" in case:
        return "CPU"
    up = name.upper()
    if "GPU" in up:
        return "GPU"
    if "CPU" in up:
        return "CPU"
    return "?"


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def hyper_le(k, n_group, succ_total, n_total):
    """P(X <= k): X = successes among n_group draws without replacement from n_total draws
    that contain succ_total successes (one-sided exact)."""
    denom = math.comb(n_total, n_group)
    num = 0
    for x in range(0, min(k, n_group, succ_total) + 1):
        if n_group - x <= n_total - succ_total:
            num += math.comb(succ_total, x) * math.comb(n_total - succ_total, n_group - x)
    return num / denom


def load_run(path):
    pts = []
    for line in read_text(path).splitlines():
        m = RUN_RE.match(line)
        if m:
            pts.append((dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2)), int(m.group(3))))
    return pts


def reserved_at(pts, when):
    last = None
    for ts, step, res in pts:
        if ts <= when:
            last = (ts, step, res)
    return last


def part1_retry():
    out("== 1. Retry numbers, recomputed from logs/swarm2/E2_retry_cb_GPU.log ==")
    text = read_text(os.path.join(LOGS, "E2_retry_cb_GPU.log"))
    rec = tagged_json(text, "E2_BENCH_JSON")
    reps = rec["reps"]
    fits = [r["fit_seconds"] for r in reps]
    lls = [r["valid_logloss"] for r in reps]
    med_fit = statistics.median(fits)
    med_ll = statistics.median(lls)
    printed = re.findall(r"rep (\d): fit ([\d.]+) s, predict [\d.]+ s, logloss ([\d.]+), trees \d+", text)
    printed_ok = [(int(a), float(b), float(c)) for a, b, c in printed] == \
        [(r["rep"], r["fit_seconds"], r["valid_logloss"]) for r in reps]
    out(f"gpu_ram_part (JSON)        : {rec['gpu_ram_part']}")
    out(f"warm-up seconds            : {rec['gpu_warmup_seconds']}   claim 4.561")
    out(f"fit seconds per rep        : {fits}   claim 25.869, 37.707")
    out(f"median fit (n=2 => mean)   : {med_fit:.3f}   JSON {rec['median_fit_seconds']}   claim 31.788")
    out(f"valid logloss per rep      : {lls}   claim 0.548867, 0.545308")
    out(f"median logloss             : {med_ll:.7f}   JSON {rec['median_valid_logloss']}   claim 0.547087")
    out(f"rep-to-rep fit ratio       : {max(fits) / min(fits):.3f}")
    out(f"rep-to-rep logloss gap     : {abs(lls[0] - lls[1]):.6f}   (same process, same seed, same params)")
    out(f"gpu util mean per rep      : {[r['gpu_util_mean_pct'] for r in reps]}   claim 40 to 44")
    out(f"gpu util max per rep       : {[r['gpu_util_max_pct'] for r in reps]}   claim 100")
    out(f"printed rep lines match JSON: {printed_ok}")
    cpu = tagged_json(read_text(os.path.join(LOGS, "E2_bench_cb_CPU.log")), "E2_BENCH_JSON")
    out(f"CatBoost CPU median fit    : {cpu['median_fit_seconds']}  (thread_count 1)")
    out(f"GPU / CPU median fit ratio : {med_fit / cpu['median_fit_seconds']:.3f}")
    out("")


def part2_runs():
    out("== 2. Every E2 log: outcome, gpu_ram_part, whole-card memory fields ==")
    cb_attempts, clean, contaminated, cpu_util, gpu_util = [], [], [], [], []
    for path in sorted(glob.glob(os.path.join(LOGS, "E2_*.log")), key=os.path.getmtime):
        name = os.path.basename(path)
        if "driver" in name or "diag" in name:
            continue
        text = read_text(path)
        size = os.path.getsize(path)
        when = mtime(path)
        bench = tagged_json(text, "E2_BENCH_JSON")
        env = tagged_json(text, "E2_ENV_JSON")
        case_m = re.search(r"E2_CASE_START (\S+)", text)
        case = case_m.group(1) if case_m else (env or {}).get("case")
        oom = "CUDA error 2: out of memory" in text
        if case == "meta":
            out(f"  {when:%H:%M:%S}  {name}: catboost_gpu_device_count={env.get('catboost_gpu_device_count')} "
                f"catboost={env.get('catboost_version')} before={env.get('nvidia_smi_before')}")
            if env.get("nvidia_smi_before"):
                clean.append((when, smi_total_used(env["nvidia_smi_before"])[1], f"{name} (pre-init)"))
            continue
        is_lgb = "lgb" in name
        is_cb = (not is_lgb) and ("cb" in name or "catboost" in text.lower() or oom)
        device = guess_device(name, bench, case, oom)
        if bench and bench.get("gpu_ram_part") is not None:
            cfg, cfg_src = bench["gpu_ram_part"], "JSON field"
        elif case == "cb_gpu_tiny_default":
            cfg, cfg_src = 0.95, "case name (library default)"
        elif case == "cb_gpu_tiny_ram02":
            cfg, cfg_src = 0.2, "case name"
        elif name == "E2_bench_cb_GPU.log":
            cfg, cfg_src = 0.2, "INFERRED from E2_bench.py default; driver args not logged"
        elif is_cb:
            cfg, cfg_src = None, "NOT RECORDED"
        else:
            cfg, cfg_src = None, "n/a"
        if oom:
            outcome = "OOM abort (CUDA error 2)"
        elif size == 0:
            outcome = "no output (0 bytes)"
        elif (bench and bench.get("status") == "ok") or (env and env.get("status") == "ok"):
            outcome = "ok"
        elif bench:
            outcome = f"status={bench.get('status')}"
        else:
            outcome = "other"
        reps = (bench or {}).get("reps", [])
        base = reps[0].get("gpu_mem_baseline_mib") if reps else None
        peak = max((r.get("gpu_mem_peak_mib") or 0) for r in reps) if reps else None
        um = [r.get("gpu_util_mean_pct") for r in reps]
        ux = [r.get("gpu_util_max_pct") for r in reps]
        warm = (bench or {}).get("gpu_warmup_seconds")
        before = smi_total_used(env["nvidia_smi_before"])[1] if env and env.get("nvidia_smi_before") else None
        kind = "CB" if is_cb else ("LGB" if is_lgb else "?")
        out(f"  {when:%H:%M:%S}  {name:<34} {kind:<4} device={device:<5} gpu_ram_part={str(cfg):<5} src={cfg_src}")
        out(f"              outcome={outcome}; bytes={size}; baseline_MiB={base}; peak_MiB={peak}; "
            f"util_mean={um}; util_max={ux}; warmup_s={warm}; pre_init_used_MiB={before}")
        if before is not None:
            clean.append((when, before, f"{name} (nvidia-smi before CatBoost init)"))
        if bench and device == "CPU" and reps and base is not None:
            clean.append((when, base, f"{name} (CPU-only process)"))
        if bench and device == "GPU" and is_cb and warm is not None and base is not None:
            contaminated.append((when, base, name, warm, warm >= WARMUP_OVERLAP_S))
        if is_cb and bench and device == "CPU" and um:
            cpu_util.append((name, um, ux))
        if is_cb and bench and device == "GPU" and um:
            gpu_util.append((name, um, ux))
        if is_cb and device == "GPU":
            cb_attempts.append({"when": when, "name": name, "cfg": cfg, "src": cfg_src, "outcome": outcome})
    out("")
    return cb_attempts, clean, contaminated, cpu_util, gpu_util


def part2b_driver():
    out("== 2b. E2 driver lines (each is stamped at the END of that step) ==")
    for name in ("E2_bench_driver.log", "E2_retry_driver.log"):
        for line in read_text(os.path.join(LOGS, name)).splitlines():
            out(f"  {name}: {line}")
    out("")


def part2c_e1_and_x5(clean):
    out("== 2c. Other GPU work on the same card: E1 bench (logs/swarm2/E1_bench.log) and X5 (logs/exp) ==")
    e1_path = os.path.join(LOGS, "E1_bench.log")
    e1 = read_text(e1_path)
    last_write = mtime(e1_path)
    secs = {}
    for phase, s in re.findall(r"\[(\w+)\] run \d+/\d+: ([\d.]+) s", e1):
        secs[phase] = secs.get(phase, 0.0) + float(s)
    end_1t = last_write
    start_1t = end_1t - dt.timedelta(seconds=secs["cpu_1thread"])
    end_cd = start_1t
    start_cd = end_cd - dt.timedelta(seconds=secs["cpu_default"])
    out(f"  E1 phase seconds: cpu_default={secs['cpu_default']:.3f}  cpu_1thread={secs['cpu_1thread']:.3f} (A3)")
    out(f"  E1 cpu_default window ~ {start_cd:%H:%M:%S} to {end_cd:%H:%M:%S};  cpu_1thread ~ {start_1t:%H:%M:%S} to {end_1t:%H:%M:%S}")
    e1_map = {"cpu_default before timed runs": start_cd, "cpu_default after timed runs": end_cd,
              "cpu_1thread before timed runs": start_1t}
    for label, mib, util in re.findall(r"\[([^\]]+)\] gpu\(mem,util\)=(\d+) MiB, (\d+) %", e1):
        t = e1_map.get(label)
        out(f"  E1 reading  [{label}] whole-card {mib} MiB, util {util}%  time~{t:%H:%M:%S}" if t else
            f"  E1 reading  [{label}] whole-card {mib} MiB, util {util}%  time not logged")
        if t is not None:
            clean.append((t, int(mib), f"E1_bench.log [{label}] (time is A3 estimate)"))
    # drop the NOTES: line, which repeats every per-depth result of the same run
    x5 = "\n".join(line for line in read_text(os.path.join(EXP, "X5_catboost_gpu.log")).splitlines()
                   if not line.startswith("NOTES:"))
    x5_ok = re.findall(r"depth (\d+) OK device=(\w+) monotone=(\w+) best_iter=\d+ val_logloss_rowmean=([\d.]+) fit_s=([\d.]+)", x5)
    x5_fail = re.findall(r"depth (\d+) FAILED device=(\w+) monotone=(\w+): CatBoostError: (.{0,90})", x5)
    x5_cfg = re.findall(r'"gpu_ram_part": ([\d.]+)', x5)
    out(f"  X5 (logs/exp/X5_catboost_gpu.log, written {mtime(os.path.join(EXP, 'X5_catboost_gpu.log')):%H:%M:%S}):")
    for depth, dev, mono, ll, fit in x5_ok:
        out(f"    depth {depth} OK device={dev} monotone={mono} val_logloss_rowmean={ll} fit_s={fit}")
    for depth, dev, mono, msg in x5_fail:
        out(f"    depth {depth} FAILED device={dev} monotone={mono}: {msg.strip()}")
    out(f"    summary gpu_ram_part values: {x5_cfg}; OOM text present: {bool(re.search(r'out of memory|CUDA error 2', x5))}")
    out("    X5 has no nvidia-smi readings, so its load is unknown. Its failures are a monotone-constraint error, not OOM.")
    out("")
    return x5_ok, x5_fail, (start_cd, end_cd)


def part3_util(cpu_util, gpu_util):
    out("== 3. Control: whole-card GPU utilization in CPU-only CatBoost runs vs CatBoost GPU runs ==")
    for name, um, ux in cpu_util:
        out(f"  CPU-only  {name:<26} util_mean={um} util_max={ux}")
    for name, um, ux in gpu_util:
        out(f"  GPU       {name:<26} util_mean={um} util_max={ux}")
    cm = [v for _, um, _ in cpu_util for v in um]
    gm = [v for _, um, _ in gpu_util for v in um]
    cx = [v for _, _, ux in cpu_util for v in ux]
    gx = [v for _, _, ux in gpu_util for v in ux]
    overlap = min(gm) <= max(cm) and min(cm) <= max(gm)
    out(f"  util_mean: CPU-only {min(cm)}..{max(cm)}   GPU {min(gm)}..{max(gm)}   overlap={overlap}")
    out(f"  util_max : CPU-only {min(cx)}..{max(cx)}   GPU {min(gx)}..{max(gx)}")
    out("  Reading: CPU-only CatBoost fits show utilization in the same range as the GPU fits,")
    out("  so whole-card utilization cannot show that CatBoost ran on the GPU.")
    out("")


def part4_tally(cb_attempts, x5_ok):
    out("== 4. CatBoost GPU outcomes by gpu_ram_part and the default-vs-0.2 contrast ==")
    for a in sorted(cb_attempts, key=lambda x: x["when"]):
        out(f"  {a['when']:%H:%M:%S}  {a['name']:<34} cfg={str(a['cfg']):<5} outcome={a['outcome']:<28} src={a['src']}")
    ok02 = sum(a["outcome"] == "ok" for a in cb_attempts if a["cfg"] == 0.2)
    oom02 = sum(a["outcome"].startswith("OOM") for a in cb_attempts if a["cfg"] == 0.2)
    n02 = ok02 + oom02
    ok95 = sum(a["outcome"] == "ok" for a in cb_attempts if a["cfg"] == 0.95)
    oom95 = sum(a["outcome"].startswith("OOM") for a in cb_attempts if a["cfg"] == 0.95)
    n95 = ok95 + oom95
    unlabelled_fail = sum(1 for a in cb_attempts if a["cfg"] is None and a["outcome"].startswith("OOM"))
    x5_n = len(x5_ok)
    out(f"  E2 only : gpu_ram_part=0.2 ok={ok02} OOM={oom02} (n={n02}; the OOM config is inferred)")
    out(f"            gpu_ram_part=0.95 ok={ok95} OOM={oom95} (n={n95}, verified)  unlabelled OOM={unlabelled_fail}")
    out(f"  E2 + X5 : gpu_ram_part=0.2 ok={ok02 + x5_n} OOM={oom02} (n={n02 + x5_n}; X5 is a different dataset, 1.4M rows)")
    out(f"  Wilson 95% CI, 0.2 success, E2 only ({ok02}/{n02}): {wilson(ok02, n02)}")
    out(f"  Wilson 95% CI, 0.2 success, E2+X5 ({ok02 + x5_n}/{n02 + x5_n}): {wilson(ok02 + x5_n, n02 + x5_n)}")
    s_e2 = ok02 + ok95
    s_all = ok02 + x5_n + ok95
    cases = [
        ("A  E2 only, verified default only      ", n95, s_e2, n95 + n02),
        ("B  E2 only, unlabelled OOM as default  ", n95 + unlabelled_fail, s_e2, n95 + unlabelled_fail + n02),
        ("A' E2+X5, verified default only        ", n95, s_all, n95 + n02 + x5_n),
        ("B' E2+X5, unlabelled OOM as default    ", n95 + unlabelled_fail, s_all, n95 + unlabelled_fail + n02 + x5_n),
    ]
    for label, n_g, s_tot, n_tot in cases:
        p = hyper_le(0, n_g, s_tot, n_tot)
        out(f"  one-sided exact P(all {n_g} default attempt(s) fail | totals) {label}: {p:.3f}")
    out("  Reading: 1-2 default attempts against 5-7 attempts at 0.2. The contrast is not statistically established,")
    out("  and the attempts were not matched for card load, so pooling them is only a rough guide.")
    out("")


def part5_budget(clean, runs_pts, cb_fail_time, e1_window):
    out("== 5. Free VRAM at logged readings, and what each gpu_ram_part would ask for ==")
    out("  A1: ask = gpu_ram_part x free (not measured). residual = nvidia-smi used minus the two DNABERT")
    out("  torch peak_reserved_mib at that time (residual = CUDA contexts + unidentified GPU users).")
    rows = [(T("16:23:00"), 8367, "notes/swarm2/E2.md section 1 (typed in the note, not a log)")] + sorted(clean)
    for when, used, label in rows:
        used = int(used)
        free = TOTAL_MIB - used
        res = [reserved_at(pts, when) for pts in runs_pts.values()]
        res_sum = sum(r[2] for r in res if r) if all(res) else None
        residual = (used - res_sum) if res_sum is not None else None
        out(f"  {when:%H:%M:%S} used={used:6d} free={free:6d} ask_0.95={0.95 * free:8.0f} ask_0.2={0.2 * free:8.0f} "
            f"dnabert_reserved={res_sum} residual={residual}  {label}")
    bracket = [r for r in clean if e1_window[0] <= r[0] <= e1_window[1] and "E1_bench" in r[2]]
    out(f"  CatBoost 0.2 failure (last write {cb_fail_time:%H:%M:%S}) inside E1 cpu_default window: "
        f"{e1_window[0] <= cb_fail_time <= e1_window[1]}")
    if bracket:
        used = int(bracket[0][1])
        free = TOTAL_MIB - used
        res = [reserved_at(pts, cb_fail_time) for pts in runs_pts.values()]
        res_sum = sum(r[2] for r in res if r) if all(res) else None
        out(f"  Whole-card at the failure, from the E1 readings bracketing it: {used} MiB (A3; steady across those readings)")
        out(f"    free={free} MiB  ask_0.2={0.2 * free:.0f} MiB  ask_0.95={0.95 * free:.0f} MiB  "
            f"DNABERT reserved={res_sum}  residual={used - res_sum if res_sum else 'n/a'}")
    out("")


def part6_dnabert(runs_pts):
    out("== 6. DNABERT jobs (the other tenant): torch peak_reserved_mib in run.log, 15:30 to 18:00 ==")
    for label, pts in runs_pts.items():
        window = [p for p in pts if T("15:30:00") <= p[0] <= T("18:00:00")]
        vals = sorted({p[2] for p in window})
        first = window[0][0] if window else None
        last = window[-1][0] if window else None
        out(f"  {label}: lines={len(window)} distinct peak_reserved_mib={vals} first={first} last={last}")
    out("  Caveat: a line is written about every 200 steps (about 2 min), so shorter spikes are not visible.")
    out("")


PS_EVENTS = r"""
$ev = Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=[datetime]'2026-10-13 15:50:00'; EndTime=[datetime]'2026-10-13 17:40:00'} -ErrorAction SilentlyContinue
$gpu = @($ev | Where-Object { $_.ProviderName -match 'nvlddmkm|Display|dxgkrnl|nvidia|WHEA' })
'system_gpu_driver_events_15:50_to_17:40 = ' + $gpu.Count
$app = Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=[datetime]'2026-10-13 16:55:00'; EndTime=[datetime]'2026-10-13 17:10:00'} -ErrorAction SilentlyContinue
$apperr = @($app | Where-Object { $_.Id -eq 1000 })
'application_error_1000_16:55_to_17:10 = ' + $apperr.Count
foreach ($e in ($apperr | Sort-Object TimeCreated)) {
  $p = ([regex]::Match($e.Message, 'Faulting application path:\s*([^\r\n]+)')).Groups[1].Value
  $c = ([regex]::Match($e.Message, 'Exception code:\s*(\S+)')).Groups[1].Value
  $e.TimeCreated.ToString('HH:mm:ss') + ' code=' + $c + ' path=' + $p
}
'wer_1001_16:55_to_17:10 = ' + @($app | Where-Object { $_.Id -eq 1001 }).Count
$pm = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
'now commit_GiB=' + [math]::Round($pm.CommittedBytes/1GB,2) + ' commit_limit_GiB=' + [math]::Round($pm.CommitLimit/1GB,2)
"""


def part7_events():
    out("== 7. Windows event logs and commit charge (read-only) ==")
    enc = base64.b64encode(PS_EVENTS.encode("utf-16-le")).decode("ascii")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-EncodedCommand", enc],
                           capture_output=True, text=True, timeout=180)
        for line in (r.stdout or "").splitlines():
            out("  " + line)
        if r.returncode:
            out(f"  powershell exit={r.returncode} stderr={r.stderr.strip()[:300]}")
    except Exception as exc:
        out(f"  event query failed: {exc!r}")
    out("  CatBoost's own abort may not write an Application Error record, so absence is weak evidence.")
    out("")


def part8_now():
    out("== 8. nvidia-smi snapshot now (read-only) ==")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        out("  " + r.stdout.strip())
    except Exception as exc:
        out(f"  nvidia-smi failed: {exc!r}")
    out("")


def summary(clean, contaminated):
    out("== 9. Whole-card readings: clean (no CatBoost GPU work possible yet) vs contaminated ==")
    cvals = sorted(clean, key=lambda x: x[1])
    out(f"  clean n={len(cvals)}: used_MiB min={cvals[0][1]} ({cvals[0][2]}); max={cvals[-1][1]} ({cvals[-1][2]})")
    for when, base, name, warm, flag in sorted(contaminated):
        out(f"  contaminated {when:%H:%M:%S} {name:<26} baseline_MiB={base} warmup_s={warm} inside_warmup={flag}")
    if contaminated:
        cmax = max(c[1] for c in contaminated)
        out(f"  clean max {cvals[-1][1] / 1024:.2f} GiB; contaminated max {cmax / 1024:.2f} GiB (E2 note claims 8.4 to 16 GB)")
    out("")


def main():
    out(f"verify_E2_2_indep  run {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    out(f"python {sys.version.split()[0]}  executable {sys.executable}")
    out("")
    part1_retry()
    cb_attempts, clean, contaminated, cpu_util, gpu_util = part2_runs()
    part2b_driver()
    x5_ok, x5_fail, e1_window = part2c_e1_and_x5(clean)
    part3_util(cpu_util, gpu_util)
    part4_tally(cb_attempts, x5_ok)
    runs_pts = {k: load_run(v) for k, v in RUN_LOGS.items()}
    cb_fail_time = mtime(os.path.join(LOGS, "E2_bench_cb_GPU.log"))
    part5_budget(clean, runs_pts, cb_fail_time, e1_window)
    part6_dnabert(runs_pts)
    part7_events()
    part8_now()
    summary(clean, contaminated)


if __name__ == "__main__":
    main()
