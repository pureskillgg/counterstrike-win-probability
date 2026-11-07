"""Build intern-grader rows for the sequence arms (pin c807884d, validation only).

Every arm-seed in the pin is passed to evaluation/build_validation_gdt_rows.py in ONE call, so all arms share the same rows.
Column names are <arm>_s<seed>. The builder itself asserts that no holdout B or frozen-test key is in the validation keys.
"""
import glob
import os
import re
import subprocess
import sys

PY = "G:/pureskill/envs/tabicl/Scripts/python.exe"
BUILDER = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/evaluation/build_validation_gdt_rows.py"
LOC = "G:/pureskill/csds-cs2/data/diagnostics"
VOL = ("C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/"
       "scratchpad/final_results/vol_preds")
OUT = "G:/pureskill/csds-cs2/data/diagnostics/gdt_seq"

# arm-seed -> prediction file (final-step validation predictions; the same files the log-loss results used)
found = {}
for root in [LOC + "/pos24_*", VOL]:
    for f in glob.glob(root + "/**/*final_val_preds.parquet", recursive=True):
        rel = f.replace("\\", "/")
        if "/local_laptop/" in rel or "pos24_smoke" in rel or "pos24_promo" in rel:
            continue
        m = re.match(r"(.+)_seed(\d)_final_val_preds\.parquet$", os.path.basename(rel))
        if m:
            found.setdefault((m.group(1), int(m.group(2))), rel)

SEED0_ARMS = sorted(a for (a, s) in found if s == 0)
FIVE_SEED_ARMS = ["gru", "xlstm_fast", "xlstm_blk", "xlstm_sig", "rwkv7", "mingru", "tcn"]
WANT = [(a, 0) for a in SEED0_ARMS] + [(a, s) for a in FIVE_SEED_ARMS for s in (1, 2, 3, 4)]
missing = [k for k in WANT if k not in found]
if missing:
    sys.exit(f"missing prediction files: {missing}")

os.makedirs(OUT, exist_ok=True)
cmd = [PY, "-I", BUILDER, "--out", OUT + "/rows_all_arms.parquet"]
for arm, seed in WANT:
    cmd += ["--model", f"{arm}_s{seed}={found[(arm, seed)]}"]
print(f"arm-seeds: {len(WANT)} (seed 0: {len(SEED0_ARMS)} arms; seeds 1 to 4: {len(FIVE_SEED_ARMS)} arms)", flush=True)
print("running builder", flush=True)
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-4000:], flush=True)
print(r.stderr[-4000:], flush=True)
print("EXIT", r.returncode, flush=True)
