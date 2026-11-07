import subprocess, time, sys
AWS = ["aws", "--profile", "runpod", "--endpoint-url", "https://s3api-eu-ro-1.runpod.io", "--region", "eu-ro-1"]
S = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
JOBS = {
    "pureskill-run-01": dict(id="ua1hn3sbve809p", ip="213.173.102.201", dir="pos24_sig_s12", marker="RUN_DONE_sig_s12",
                             files=["xlstm_sig_seed1_metrics.json", "xlstm_sig_seed2_metrics.json", "xlstm_sig_seed1_final_val_preds.parquet", "xlstm_sig_seed2_final_val_preds.parquet"]),
    "pureskill-run-04": dict(id="7jame2alggd85a", ip="213.173.99.50", dir="pos24_sig_s3", marker="RUN_DONE_sig_s3",
                             files=["xlstm_sig_seed3_metrics.json", "xlstm_sig_seed3_final_val_preds.parquet"]),
}
def ls_has(path):
    r = subprocess.run(AWS + ["s3", "ls", "s3://z7yvc40ykx/results/" + path], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""
def run_done(d, marker):
    r = subprocess.run(AWS + ["s3", "cp", f"s3://z7yvc40ykx/results/{d}/run.log", "-"], capture_output=True, text=True)
    return r.returncode == 0 and marker in r.stdout, ("Traceback" in r.stdout)
pending = dict(JOBS)
for poll in range(120):
    for name in list(pending):
        j = pending[name]
        done, tb = run_done(j["dir"], j["marker"])
        if not done:
            continue
        missing = [f for f in j["files"] if not ls_has(f"{j['dir']}/{f}")]
        if missing or tb:
            print(f"{time.strftime('%T')} {name}: run finished but NOT closed. missing={missing} traceback={tb}", flush=True)
            del pending[name]
            continue
        rc = subprocess.run([sys.executable, f"{S}/pod_terminate.py", j["id"], name, j["ip"]], capture_output=True, text=True)
        print(f"{time.strftime('%T')} {name}: outputs on volume, closing. {rc.stdout.strip()}", flush=True)
        del pending[name]
    if not pending:
        print("ALL_PODS_HANDLED", flush=True)
        sys.exit(0)
    time.sleep(120)
print("TIMEOUT pending=" + ",".join(pending), flush=True)
