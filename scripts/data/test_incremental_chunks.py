"""Test of build_tomes_incremental.py --publish-chunks on real matches: extend a collection twice and check that the chunks it publishes are
exactly the new data and add up to a full build.

  G:\\pureskill\\envs\\dsdk-fork\\Scripts\\python.exe -I -X utf8 scripts\\data\\test_incremental_chunks.py

Fixtures (made earlier from E:, 40 real matches): scratch\\dsdk_raw_a (25 of them), scratch\\dsdk_raw (all 40), scratch\\tomes_full (a full build of the 40).
Steps: fresh build of the 25 -> chunk x001; run again on the 40 (extend, 15 new) -> chunk x002; a third run with nothing new publishes nothing.
Checks: CHUNKS_DONE lists x001, x002; the chunks' match sets are disjoint (25 + 15) and their union is the 40 of the full build; for every channel the rows of
x001 + x002 equal the rows of the full build; a skip list removes matches from the build. Exit code 1 on any failure.
"""
import glob
import os
import shutil
import subprocess
import sys

import pyarrow.parquet as pq

S = r"G:\pureskill\scratch"
DRV = r"G:\pureskill\csds-cs2\scripts\data\build_tomes_incremental.py"
fails = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name, detail, flush=True)
    if not ok:
        fails.append(name)


def run(*a):
    r = subprocess.run([sys.executable, "-I", "-X", "utf8", DRV, *a], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def rows_by_channel(root):
    out = {}
    for d in glob.glob(os.path.join(root, "tome", "csds", "*")):
        ch = os.path.basename(d)
        ch = "header" if ch == "header" else ch[len("basic_"):].split(".")[0]
        if ch == "header":
            continue
        out[ch] = sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(os.path.join(d, "dataframe_*")))
    return out


def keys_of(root):
    ks = set()
    for f in glob.glob(os.path.join(root, "tome", "csds", "header", "keyset_*")):
        ks |= set(pq.read_table(f).column(0).to_pylist())
    return ks


work = os.path.join(S, "chunktest")
for d in (work, work + "_x001", work + "_x002"):
    shutil.rmtree(d, ignore_errors=True)
for f in (work + ".CHUNKS_DONE",):
    if os.path.exists(f):
        os.remove(f)
live = os.path.join(S, "chunktest")
# the 15 later matches are held back with a skip list (their keys are csds/2026/10/08/<id>/csds), so run 1 builds 25 and run 2 sees 15 "new" ones appear
later = [ln.strip() for ln in open(os.path.join(S, "later15.txt"), encoding="utf-8") if ln.strip()]
skip_file = os.path.join(S, "skip15.txt")
open(skip_file, "w", encoding="utf-8").write("\n".join(f"csds/2026/10/08/{i}/csds" for i in later) + "\n")
code, out = run("--raw", os.path.join(S, "dsdk_raw"), "--out", live, "--skip-keys", skip_file, "--publish-chunks")
check("run 1 (fresh, 25 matches, 15 held back by the skip list) publishes x001", code == 0 and "chunktest_x001" in out and "25 matches" in out, out.strip().splitlines()[-2][:140] if out.strip() else "")
code, out = run("--raw", os.path.join(S, "dsdk_raw"), "--out", live, "--publish-chunks")
check("run 2 (extend by the 15 new matches) publishes x002", code == 0 and "chunktest_x002" in out and "mode=extend" in out, out.strip().splitlines()[-2][:140] if out.strip() else "")
code, out = run("--raw", os.path.join(S, "dsdk_raw"), "--out", live, "--publish-chunks")
check("run 3 (nothing new) reads and publishes nothing", code == 0 and "nothing to do" in out and not os.path.exists(live + "_x003"), out.strip().splitlines()[-1][:140] if out.strip() else "")
done = open(live + ".CHUNKS_DONE", encoding="utf-8").read().split() if os.path.exists(live + ".CHUNKS_DONE") else []
check("CHUNKS_DONE lists x001 and x002", done == ["chunktest_x001", "chunktest_x002"], str(done))
k1, k2, kf = keys_of(live + "_x001"), keys_of(live + "_x002"), keys_of(os.path.join(S, "tomes_full"))
check("chunk match sets are disjoint and add up to the full build", not (k1 & k2) and (k1 | k2) == kf and len(k1) == 25 and len(k2) == 15, f"{len(k1)} + {len(k2)} = {len(k1 | k2)}, full {len(kf)}")
r1, r2, rf = rows_by_channel(live + "_x001"), rows_by_channel(live + "_x002"), rows_by_channel(os.path.join(S, "tomes_full"))
bad = {c: (r1.get(c, 0), r2.get(c, 0), rf[c]) for c in rf if r1.get(c, 0) + r2.get(c, 0) != rf[c]}
check("rows per channel: x001 + x002 == full build", not bad and len(rf) >= 30, f"{len(rf)} channels; mismatches {bad}")
check("each chunk has its BUILD_OK marker", all(os.path.isfile(os.path.join(live + f"_x00{i}", "BUILD_OK")) for i in (1, 2)))
check("--skip-keys left exactly the held-back matches out of chunk x001", not (k1 & {f"csds/2026/10/08/{i}/csds" for i in later}), f"x001 holds {len(k1)} matches")
print("RESULT:", "ALL PASS" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
