"""Selftests for the 2024 February path. Every 2026 default must behave exactly as before; the 2024 options must resolve to the new folders.

  python -I -X utf8 scripts\\model\\selftests\\test_2024_pipeline.py
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))

import polars as pl  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"  {detail}"), flush=True)
    if not cond:
        failures.append(name)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(args):
    return subprocess.run([sys.executable, "-I", "-X", "utf8"] + args, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=ROOT)


# 1. positions extractor: raw root and paths
pos = load_module("extract_positions_1hz", os.path.join(ROOT, "scripts", "data", "extract_positions_1hz.py"))
check("positions default raw root is the E: 2026 tree", pos.RAW_ROOT == pos.E_CSDS == r"E:\csds_final\csds")
check("positions raw_dir for a 2026 key uses the E: tree",
      pos.raw_dir("csds/2026/10/08/abc/csds") == os.path.join(r"E:\csds_final\csds", "2026", "10", "08", "abc"))
pos.RAW_ROOT = os.path.join(ROOT, "staging_2024_full", "csds")
check("positions raw_dir for a 2024 key uses the staged root when --raw-root is set",
      pos.raw_dir("csds/2024/02/01/abc/csds") == os.path.join(ROOT, "staging_2024_full", "csds", "2024", "02", "01", "abc"))
pos.RAW_ROOT = pos.E_CSDS

# 2. status extractor: raw path
stat = load_module("extract_status_1hz", os.path.join(ROOT, "scripts", "data", "extract_status_1hz.py"))
check("status default raw root is the E: 2026 tree", stat.RAW_ROOT == stat.E_CSDS)
check("status raw_path for a 2026 key is unchanged",
      stat.raw_path("csds/2026/10/08/abc/csds") == os.path.join(r"E:\csds_final\csds", "2026", "10", "08", "abc", "player_status"))

# 3. list-only modes: write nothing, resolve the right days and folders
scratch = tempfile.mkdtemp(prefix="pos2024_")
shutil.rmtree(scratch)  # the list-only run must not create it
r = run([os.path.join(ROOT, "scripts", "data", "extract_positions_1hz.py"), "--era", "2024", "--raw-root",
         os.path.join(ROOT, "staging_2024_full", "csds"), "--out-dir", scratch, "--list-only"])
lines = [ln for ln in r.stdout.splitlines() if "->" in ln]
check("positions --list-only --era 2024 lists 21 days", r.returncode == 0 and len(lines) == 21, r.stdout[-300:] + r.stderr[-300:])
check("positions --list-only writes under the given out-dir only", all(scratch in ln for ln in lines))
check("positions --list-only creates no folder", not os.path.exists(scratch))

r = run([os.path.join(ROOT, "scripts", "data", "extract_positions_1hz.py"), "--list-only"])
lines26 = [ln for ln in r.stdout.splitlines() if "->" in ln]
check("positions --list-only with defaults lists the 2026 days", r.returncode == 0 and len(lines26) > 40 and all("data" in ln for ln in lines26),
      r.stdout[-300:] + r.stderr[-300:])

r = run([os.path.join(ROOT, "scripts", "data", "extract_status_1hz.py"), "--era", "2024", "--raw-root",
         os.path.join(ROOT, "staging_2024_full", "csds"), "--out-dir", scratch, "--list-only"])
lines_s = [ln for ln in r.stdout.splitlines() if "->" in ln]
check("status --list-only --era 2024 lists 21 days", r.returncode == 0 and len(lines_s) == 21, r.stdout[-300:] + r.stderr[-300:])
check("status --list-only creates no folder", not os.path.exists(scratch))

# 4. the cache builder refuses to write the control file name for 2024
r = run([os.path.join(ROOT, "scripts", "model", "make_snapshot_cache.py"), "--era", "2024"])
check("cache builder refuses 2024 into cache_snapshot_2026.parquet",
      r.returncode != 0 and "refusing" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-300:])

# 5. the roster-defect rule applies to 2024 rounds
import wp_common as W  # noqa: E402

R = pl.read_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet"), columns=["match_key", "round", "missing_side"])
r24 = R.filter(pl.col("match_key").str.contains("csds/2024/") & pl.col("missing_side").is_not_null()).select("match_key", "round")
kept = W.drop_roster_defect_rounds(r24, "combined")
check("roster-defect rule drops every flagged 2024 round", kept.height == 0 and r24.height > 0, f"kept {kept.height} of {r24.height}")

# 6. alive-count gate on synthetic data: one death, a clean run passes, one planted mismatch fails
gate = load_module("alive_gate", os.path.join(ROOT, "scripts", "model", "alive_gate.py"))
snaps = pl.DataFrame({"match_key": ["m"] * 4, "round": [1] * 4, "second": [0, 1, 2, 3]})
deaths = pl.DataFrame({"match_key": ["m"], "round": [1], "second": [2], "victim_side": [3]})
rows = []
for s, ct in zip([0, 1, 2, 3], [5, 5, 4, 4]):
    rows += [{"match_key": "m", "round": 1, "second": s, "side": 3, "player_id": 100 + i} for i in range(ct)]
    rows += [{"match_key": "m", "round": 1, "second": s, "side": 2, "player_id": 200 + i} for i in range(5)]
positions = pl.DataFrame(rows)
good = gate.alive_gate(positions, deaths, snaps)
check("alive gate passes on consistent synthetic data", good["pass"] and good["share"] == 1.0, good)
bad_rows = [r_ for r_ in rows if not (r_["side"] == 3 and r_["second"] == 2 and r_["player_id"] == 103)]  # one CT row missing at second 2
bad = gate.alive_gate(pl.DataFrame(bad_rows), deaths, snaps)
check("alive gate fails when a snapshot disagrees", (not bad["pass"]) and bad["share"] < 1.0 and len(bad["mismatch_sample"]) == 1, bad)

# 7. window check
wc = load_module("window_check", os.path.join(ROOT, "scripts", "model", "window_check.py"))
win = {"xy_min": -10.0, "xy_max": 10.0, "z_min": -5.0, "z_max": 5.0}
check("window check is zero for inside points", wc.out_of_window_share([0, 5], [0, -5], [0, 1], win) == 0.0)
check("window check counts an outside point", abs(wc.out_of_window_share([0, 50, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0], win) - 0.25) < 1e-12)
check("window check uses the z bounds", abs(wc.out_of_window_share([0], [0], [9], win) - 1.0) < 1e-12)

if failures:
    print("RESULT: FAILED", failures)
    sys.exit(1)
print("RESULT: ALL PASS")
