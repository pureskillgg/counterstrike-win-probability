"""Pack what the heavy GPU steps need into ONE archive for a Linux pod (data is small: the cache, grid and tables are ~260 MB).

  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\runpod\\make_pod_bundle.py [--out-dir G:\\pureskill\\runpod_out] [--dry-run]

Inside the archive (extract to /workspace so it becomes /workspace/csds-cs2):
  scripts/model/*.py      every path constant rewritten from the Windows project root to /workspace/csds-cs2 (backslash, doubled backslash and
                          forward-slash spellings); each file byte-compiled to prove it parses; the build REFUSES if the Windows root remains
  data/combined/*         grid, matches, rounds, deaths, cache_snapshot_2026 (parquet), split_manifest.json, prep_report.json
  pod_run.sh              setup + the chain (copied from scripts/runpod/pod_run.sh)
  MANIFEST.json           file list with sizes and sha256, git HEAD, creation time
A .sha256 file of the archive is written beside it. Nothing is uploaded by this script.
"""
import argparse
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

sys.stdout.reconfigure(encoding="utf-8")
REAL = r"G:\pureskill\csds-cs2"
POD = "/workspace/csds-cs2"
DATA = ["grid.parquet", "matches.parquet", "rounds.parquet", "deaths.parquet", "cache_snapshot_2026.parquet", "split_manifest.json", "prep_report.json"]
KEEP = ["wp_common.py", "wp_items.py", "features_status.py", "features_events.py", "snapshot_features.py", "export_model.py", "winprob.py", "stress_test_winprob.py",
        "inspect_rounds.py", "run_baselines.py", "exp_robustness.py", "exp_temporal.py", "exp_model_families.py", "exp_deep_distill.py", "exp_full_comparison.py",
        "exp_pistol.py", "exp_full_positions.py", "exp_positions_groups.py", "exp_tick_sampling.py", "exp_pistol_val.py", "merge_comparison.py"]
VARIANTS = [(REAL, POD), (REAL.replace("\\", "\\\\"), POD), (REAL.replace("\\", "/"), POD)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=r"G:\pureskill\runpod_out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chain", default="pod_ckpt_run.sh", help="the run script in scripts/runpod/ that becomes pod_run.sh in the archive")
    a = ap.parse_args()
    stage = tempfile.mkdtemp(prefix="pod_stage_", dir=a.out_dir if os.path.isdir(a.out_dir) else None)
    root = os.path.join(stage, "csds-cs2")
    os.makedirs(os.path.join(root, "scripts"), exist_ok=True)
    # only what the pod steps import or run (an allowlist: a script that is not part of the chain is not shipped, so it cannot break the rewrite check)
    os.makedirs(os.path.join(root, "scripts", "model"), exist_ok=True)
    for f in KEEP:
        shutil.copy2(os.path.join(REAL, "scripts", "model", f), os.path.join(root, "scripts", "model", f))
    n_rewritten, problems = 0, []
    for dp, _, fn in os.walk(os.path.join(root, "scripts")):
        for n in fn:
            if not n.endswith(".py"):
                continue
            p = os.path.join(dp, n)
            with open(p, encoding="utf-8", newline="") as f:
                t = f.read()
            t2 = t
            for x, y in VARIANTS:
                t2 = t2.replace(x, y)
            if t2 != t:
                n_rewritten += 1
                with open(p, "w", encoding="utf-8", newline="") as f:
                    f.write(t2)
            if "pureskill" in t2.lower().replace("pureskillgg", "") and ("G:\\" in t2 or "G:/" in t2):
                problems.append(f"{n}: still mentions a G: path")
            try:
                compile(t2, p, "exec")  # parses and compiles in memory, writes nothing
            except SyntaxError as e:
                problems.append(f"{n}: does not compile: {str(e)[:120]}")
    shutil.copy2(os.path.join(REAL, "scripts", "runpod", a.chain), os.path.join(root, "pod_run.sh"))  # the chain (checkpointed by default)
    with open(os.path.join(root, "pod_run.sh"), "rb") as f:
        raw = f.read()
    with open(os.path.join(root, "pod_run.sh"), "wb") as f:
        f.write(raw.replace(b"\r\n", b"\n"))  # LF only
    os.makedirs(os.path.join(root, "data", "combined"), exist_ok=True)
    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    for m in ["winprob_v3.meta.json", "winprob_v4.json", "winprob_v4.meta.json"]:  # feature list; the saved XGBoost fit the comparison reloads
        shutil.copy2(os.path.join(REAL, "models", m), os.path.join(root, "models", m))
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "pod_root": POD, "rewritten_files": n_rewritten, "data": {}, "git_head": ""}
    try:
        manifest["git_head"] = subprocess.run(["git", "-C", REAL, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    total = 0
    for f in DATA:
        src = os.path.join(REAL, "data", "combined", f)
        if not os.path.exists(src):
            problems.append(f"missing data file {f}")
            continue
        size = os.path.getsize(src)
        total += size
        manifest["data"][f] = {"bytes": size, "sha256": sha256(src)}
        if not a.dry_run:
            shutil.copy2(src, os.path.join(root, "data", "combined", f))
    bad = [p for p in problems]
    print(f"scripts rewritten: {n_rewritten}; data files {len(manifest['data'])}, {total / 1e6:,.0f} MB; problems: {bad or 'none'}")
    if bad:
        shutil.rmtree(stage, ignore_errors=True)
        sys.exit("bundle refused: " + "; ".join(bad))
    if a.dry_run:
        shutil.rmtree(stage, ignore_errors=True)
        print("dry run: nothing written")
        return
    with open(os.path.join(root, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    os.makedirs(a.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    arc = os.path.join(a.out_dir, f"pod_bundle_{stamp}.tar")
    with tarfile.open(arc, "w") as tf:  # parquet is already compressed: a plain tar is as small and much faster
        tf.add(root, arcname="csds-cs2")
    digest = sha256(arc)
    with open(arc + ".sha256", "w", encoding="utf-8", newline="\n") as f:  # LF only: the pod's sha256sum -c must not see a CR
        f.write(f"{digest}  {os.path.basename(arc)}\n")
    shutil.rmtree(stage, ignore_errors=True)
    print(f"-> {arc}  {os.path.getsize(arc) / 1e6:,.0f} MB  sha256 {digest[:16]}")


if __name__ == "__main__":
    main()
