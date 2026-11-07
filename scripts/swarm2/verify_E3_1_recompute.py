"""E3-1 independent verifier (read-only with respect to the project .pixi).

Recomputes, with its own code, the claimed numbers and mechanisms:
  - G: project env: file count, apparent bytes, hard-link counts, reparse points,
    pycache files newer than the local config, newest mtime.
  - C: detached env for the same project (for comparison): hard-link counts and
    the hard-link names of sampled linked files (fsutil hardlink list).
  - Volume ids (st_dev) for G: and C: paths (cross-volume check).
  - Python and xgboost builds recorded in each env's conda-meta.
  - pixi 0.79.0 config precedence: the real project config bytes, and a scratch copy
    of the project manifest with and without the project .pixi/config.toml
    (pixi config list / pixi info --json). Only config is read in the real project;
    .pixi is snapshot before and after to show it is unchanged.

Writes: the log path given in LOG (logs\\swarm2\\verify_E3_1.log) and scratch test
folders under the session scratchpad. Does not write under .pixi, E:, or .env/meridian.toml.

Run:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 -B scripts\\swarm2\\verify_E3_1_recompute.py
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

PROJECT = r"G:\pureskill\csds-cs2"
PROJ_PIXI = os.path.join(PROJECT, ".pixi")
G_ENV = os.path.join(PROJ_PIXI, "envs", "default")
G_CONFIG = os.path.join(PROJ_PIXI, "config.toml")
C_GLOBAL_CONFIG = r"C:\Users\13144\.pixi\config.toml"
C_RATTLER_CACHE = r"C:\Users\13144\AppData\Local\rattler\cache"
C_CSDS_ENV = r"C:\Users\13144\.pixi\workspace-envs\csds-cs2-15611918389529754563\envs\default"
C_GPU_ENV = r"C:\Users\13144\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default"
PIXI = r"C:\Users\13144\.pixi\bin\pixi.exe"
SCRATCH = (r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\e31_verify")
LOG = os.path.join(PROJECT, "logs", "swarm2", "verify_E3_1.log")
PROBE_LOG = os.path.join(PROJECT, "logs", "swarm2", "E3_size_probe.log")
FILE_ATTRIBUTE_REPARSE_POINT = 0x400

CLAIM_FILES = 20965
CLAIM_GIB = 1.872
CLAIM_PIXI_VERSION = "0.79.0"
CLAIM_PY = "3.12.15"

lines = []
results = {}


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def gib(n):
    return n / 2 ** 30


def ts(epoch):
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def walk(root):
    s = {"dirs": 0, "files": 0, "bytes": 0, "alloc_est": 0, "nlink_gt1": 0,
         "nlink_gt1_bytes": 0, "nlink1_bytes": 0, "hist": {}, "reparse": [], "errors": [],
         "pycache_files": 0, "pycache_newer_than_cfg": 0, "newest": 0.0, "newest_path": "",
         "devices": set(), "linked_examples": []}
    cfg_mtime = os.stat(G_CONFIG).st_mtime
    probe_mtime = os.stat(PROBE_LOG).st_mtime
    s["newer_than_probe"] = 0
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError as ex:
            s["errors"].append("scandir %s: %s" % (d, ex))
            continue
        with it:
            for e in it:
                try:
                    st = os.stat(e.path, follow_symlinks=False)
                except OSError as ex:
                    s["errors"].append("stat %s: %s" % (e.path, ex))
                    continue
                s["devices"].add(st.st_dev)
                is_reparse = bool(getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT)
                if is_reparse or e.is_symlink():
                    s["reparse"].append(e.path)
                    continue
                if e.is_dir(follow_symlinks=False):
                    s["dirs"] += 1
                    stack.append(e.path)
                    continue
                if not e.is_file(follow_symlinks=False):
                    s["errors"].append("other entry %s" % e.path)
                    continue
                s["files"] += 1
                s["bytes"] += st.st_size
                s["alloc_est"] += -(-st.st_size // 4096) * 4096
                n = st.st_nlink
                s["hist"][n] = s["hist"].get(n, 0) + 1
                if n > 1:
                    s["nlink_gt1"] += 1
                    s["nlink_gt1_bytes"] += st.st_size
                    if len(s["linked_examples"]) < 3:
                        s["linked_examples"].append(e.path)
                else:
                    s["nlink1_bytes"] += st.st_size
                if st.st_mtime > probe_mtime:
                    s["newer_than_probe"] += 1
                if st.st_mtime > s["newest"]:
                    s["newest"] = st.st_mtime
                    s["newest_path"] = e.path
                if e.name.endswith(".pyc") and os.sep + "__pycache__" + os.sep in e.path:
                    s["pycache_files"] += 1
                    if st.st_mtime > cfg_mtime:
                        s["pycache_newer_than_cfg"] += 1
    return s


def summarize(tag, s):
    say("  %s" % tag)
    say("    files=%d dirs=%d apparent_bytes=%d apparent_GiB=%.4f alloc_est_GiB=%.4f"
        % (s["files"], s["dirs"], s["bytes"], gib(s["bytes"]), gib(s["alloc_est"])))
    say("    nlink>1 files=%d (%.4f GiB)  nlink==1 files=%d (%.4f GiB)"
        % (s["nlink_gt1"], gib(s["nlink_gt1_bytes"]), s["files"] - s["nlink_gt1"], gib(s["nlink1_bytes"])))
    top = sorted(s["hist"].items())[:8]
    say("    nlink histogram (first 8 values): %s" % top)
    say("    reparse points (symlink/junction) = %d; walk errors = %d" % (len(s["reparse"]), len(s["errors"])))
    say("    pycache .pyc files = %d ; of which newer than G config mtime = %d"
        % (s["pycache_files"], s["pycache_newer_than_cfg"]))
    say("    newest file mtime = %s  (%s)" % (ts(s["newest"]), s["newest_path"]))
    say("    files with mtime after E3_size_probe.log was written (%s) = %d"
        % (ts(os.stat(PROBE_LOG).st_mtime), s["newer_than_probe"]))
    say("    st_dev volume ids seen = %s" % ["0x%X" % d for d in sorted(s["devices"])])


def hardlink_names(path):
    try:
        r = subprocess.run(["fsutil", "hardlink", "list", path], capture_output=True, timeout=60)
        txt = r.stdout.decode("cp437", errors="replace")
        return [l.strip() for l in txt.splitlines() if l.strip()]
    except Exception as ex:  # noqa: BLE001
        return ["error: %s" % ex]


def conda_meta(env):
    cm = os.path.join(env, "conda-meta")
    total = 0
    picks = []
    pureskill_refs = 0
    for fn in sorted(os.listdir(cm)):
        if not fn.endswith(".json"):
            continue
        total += 1
        p = os.path.join(cm, fn)
        with open(p, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        if "pureskill" in raw.lower():
            pureskill_refs += 1
        if re.match(r"^(python|xgboost|libxgboost|py-xgboost)-\d", fn):
            try:
                j = json.loads(raw)
                picks.append((j.get("name"), j.get("version"), j.get("build"), j.get("channel")))
            except ValueError:
                picks.append((fn, "parse-error", "", ""))
    return total, picks, pureskill_refs


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=300)
    return r.returncode, r.stdout.decode("utf-8", errors="replace"), r.stderr.decode("utf-8", errors="replace")


def snapshot(path):
    out = {}
    for e in os.scandir(path):
        st = os.stat(e.path, follow_symlinks=False)
        out[e.name] = (st.st_size, st.st_mtime_ns, e.is_dir(follow_symlinks=False))
    return out


def pixi_precedence(tag, with_local):
    base = os.path.join(SCRATCH, tag)
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base)
    shutil.copyfile(os.path.join(PROJECT, "pixi.toml"), os.path.join(base, "pixi.toml"))
    if with_local:
        os.makedirs(os.path.join(base, ".pixi"))
        shutil.copyfile(G_CONFIG, os.path.join(base, ".pixi", "config.toml"))
    rc1, out1, err1 = run([PIXI, "config", "list", "--no-progress"], cwd=base)
    rc2, out2, err2 = run([PIXI, "info", "--json", "--no-progress"], cwd=base)
    info = {}
    try:
        info = json.loads(out2)
    except ValueError:
        pass
    envs = info.get("environments_info", [])
    prefix = envs[0].get("prefix") if envs else None
    return {"config_list": out1.strip(), "rc_config": rc1, "prefix": prefix,
            "config_locations": info.get("config_locations"), "rc_info": rc2,
            "pixi_version": info.get("version"), "err": (err1 + err2).strip()[-400:]}


def main():
    say("E3-1 independent recompute  (generated %s)" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("interpreter: %s" % sys.executable)
    say("")

    # 1. Pixi binary, config bytes, system/global config, env vars
    say("== 1. pixi binary and config files ==")
    rc, out, err = run([PIXI, "--version"], cwd=SCRATCH if os.path.isdir(SCRATCH) else None)
    results["pixi_version"] = out.strip()
    say("  pixi --version -> %s (claim %s)" % (out.strip(), CLAIM_PIXI_VERSION))
    for label, p in (("G project .pixi/config.toml", G_CONFIG), ("C global .pixi/config.toml", C_GLOBAL_CONFIG)):
        b = open(p, "rb").read()
        mt = ts(os.stat(p).st_mtime)
        say("  %s: bytes=%d mtime=%s" % (label, len(b), mt))
        say("    raw: %r" % b)
    results["g_config_bytes"] = len(open(G_CONFIG, "rb").read())
    results["g_config_mtime"] = ts(os.stat(G_CONFIG).st_mtime)
    say("  PIXI_HOME / RATTLER_CACHE_DIR / PIXI_CACHE_DIR set? %s"
        % [k for k in ("PIXI_HOME", "RATTLER_CACHE_DIR", "PIXI_CACHE_DIR", "PIXI_PROJECT_MANIFEST") if os.environ.get(k)])
    say("  system config C:\\ProgramData\\pixi\\config.toml exists? %s" % os.path.exists(r"C:\ProgramData\pixi\config.toml"))
    rc, out, err = run(["where.exe", "python"])
    say("  where.exe python -> %s" % (out.strip().replace("\r", "").replace("\n", " | ") or err.strip()))
    say("")

    # 2. Project interpreter and metadata
    say("== 2. project env (G:) identity ==")
    say("  this interpreter: %s  version=%s" % (sys.executable, sys.version.split()[0]))
    say("  G env python sys.version: %s" % sys.version.replace("\n", " "))
    g_total, g_picks, g_pure = conda_meta(G_ENV)
    say("  G env conda-meta json files = %d ; references to 'pureskill' = %d" % (g_total, g_pure))
    for name, ver, build, chan in g_picks:
        say("    %s %s build=%s channel=%s" % (name, ver, build, chan))
    results["g_python"] = [p for p in g_picks if p[0] == "python"]
    say("")

    # 3. G env walk
    say("== 3. G: project env size and hard links (own walk) ==")
    sg = walk(G_ENV)
    summarize("G: %s" % G_ENV, sg)
    results["g_files"] = sg["files"]
    results["g_gib"] = round(gib(sg["bytes"]), 3)
    results["g_nlink_gt1"] = sg["nlink_gt1"]
    say("  claim: files=%d GiB=%.3f hard-linked=0 -> recomputed files=%d GiB=%.3f hard-linked=%d"
        % (CLAIM_FILES, CLAIM_GIB, sg["files"], round(gib(sg["bytes"]), 3), sg["nlink_gt1"]))
    say("")

    # 4. C detached env for the same project, and its hard-link targets
    say("== 4. C: detached env for this project (comparison) ==")
    sc = walk(C_CSDS_ENV)
    summarize("C: %s" % C_CSDS_ENV, sc)
    results["c_nlink_gt1"] = sc["nlink_gt1"]
    c_total, c_picks, c_pure = conda_meta(C_CSDS_ENV)
    say("  C csds conda-meta json = %d ; references to 'pureskill' = %d" % (c_total, c_pure))
    for name, ver, build, chan in c_picks:
        say("    %s %s build=%s channel=%s" % (name, ver, build, chan))
    results["c_python"] = [p for p in c_picks if p[0] == "python"]
    say("  hard-link names of 3 sampled linked files in the C env:")
    for ex in sc["linked_examples"]:
        names = hardlink_names(ex)
        say("    file: %s" % ex)
        for nm in names[:6]:
            say("      link: %s" % nm)
        results.setdefault("c_link_names", []).append(names)
    say("")

    # 5. Volume ids: cross-volume explanation
    say("== 5. volume ids (st_dev) ==")
    vols = {}
    for label, p in (("G repo root", PROJECT), ("G project .pixi", PROJ_PIXI), ("G default env", G_ENV),
                     ("C global .pixi", r"C:\Users\13144\.pixi"), ("C rattler cache", C_RATTLER_CACHE),
                     ("C detached csds env", C_CSDS_ENV)):
        vols[label] = os.stat(p).st_dev
        say("  %-22s %-60s 0x%X" % (label, p, vols[label]))
    results["g_dev"] = vols["G default env"]
    results["cache_dev"] = vols["C rattler cache"]
    say("  G and C differ: %s" % (vols["G default env"] != vols["C rattler cache"]))
    say("  cache holds package files that the C env links to: %s"
        % (sc["nlink_gt1"] > 0 and all("rattler" in " ".join(n).lower() for n in results.get("c_link_names", []) if n)))
    say("")

    # 6. Other envs on C: (GPU) for context
    say("== 6. other C: detached envs (context only, metadata only) ==")
    gpu_total, gpu_picks, _ = conda_meta(C_GPU_ENV)
    say("  cs2gpu detached env conda-meta json = %d" % gpu_total)
    for name, ver, build, chan in gpu_picks:
        say("    %s %s build=%s channel=%s" % (name, ver, build, chan))
    say("")

    # 7. Real project config read and .pixi snapshot before/after
    say("== 7. real project: pixi config list (read-only), .pixi snapshot before/after ==")
    snap_before = {"top": snapshot(PROJ_PIXI), "envs": snapshot(os.path.join(PROJ_PIXI, "envs")),
                   "default": snapshot(G_ENV)}
    rc, out, err = run([PIXI, "config", "list", "--no-progress"], cwd=PROJECT)
    say("  pixi config list (cwd = project) -> rc=%d :: %s" % (rc, out.strip().replace("\n", " | ")))
    results["project_config_list"] = out.strip()
    snap_after = {"top": snapshot(PROJ_PIXI), "envs": snapshot(os.path.join(PROJ_PIXI, "envs")),
                  "default": snapshot(G_ENV)}
    unchanged = snap_before == snap_after
    say("  .pixi top-level, envs/ and envs/default/ top-level entries unchanged: %s" % unchanged)
    say("")

    # 8. Scratch precedence test (identical config bytes and manifest, only the path differs)
    say("== 8. pixi precedence in scratch copies (same manifest; with/without project config) ==")
    w = pixi_precedence("with_local", True)
    n = pixi_precedence("no_local", False)
    for tag, r in (("with_local", w), ("no_local", n)):
        say("  %s:" % tag)
        say("    pixi config list -> %s" % r["config_list"].replace("\n", " | "))
        say("    pixi version     -> %s" % r["pixi_version"])
        say("    env prefix       -> %s" % r["prefix"])
        say("    config_locations -> %s" % r["config_locations"])
        if r["err"]:
            say("    stderr tail      -> %s" % r["err"].replace("\n", " | "))
    results["precedence_with_local_prefix"] = w["prefix"]
    results["precedence_no_local_prefix"] = n["prefix"]
    results["precedence_with_local_list"] = w["config_list"]
    results["precedence_no_local_list"] = n["config_list"]
    say("")

    # 9. Verdicts on the claim's checkable points
    say("== 9. checks against the claim ==")
    checks = [
        ("pixi is 0.79.0", results["pixi_version"].split()[-1] == CLAIM_PIXI_VERSION),
        ("G config is 34 bytes", results["g_config_bytes"] == 34),
        ("G config mtime is 2026-10-12 16:27", results["g_config_mtime"].startswith("2026-10-12 16:27")),
        ("G config sets detached-environments = false", "detached-environments = false" in results["precedence_with_local_list"]),
        ("global config value is workspace-envs path", "workspace-envs" in results["precedence_no_local_list"]),
        ("G project env python is 3.12.15 conda-forge",
         any(p[1] == CLAIM_PY and p[3] and "conda-forge" in str(p[3]) for p in results["g_python"])),
        ("G env files == claimed %d (from probe log; recomputed value reported above)" % CLAIM_FILES,
         results["g_files"] == CLAIM_FILES),
        ("G env GiB rounds to %.3f" % CLAIM_GIB, round(results["g_gib"], 3) == CLAIM_GIB),
        ("G env hard-linked files == 0", results["g_nlink_gt1"] == 0),
        ("with local config, pixi places env under project .pixi",
         bool(results["precedence_with_local_prefix"]) and "\\.pixi\\envs\\default" in results["precedence_with_local_prefix"]),
        ("without local config, pixi places env under C: workspace-envs",
         bool(results["precedence_no_local_prefix"]) and "workspace-envs" in results["precedence_no_local_prefix"]),
        ("G and cache are on different volumes", results["g_dev"] != results["cache_dev"]),
        ("C detached env (same project) is hard-linked to cache", results["c_nlink_gt1"] > 0),
        ("real project .pixi unchanged by pixi config list", unchanged),
    ]
    all_ok = True
    for label, ok in checks:
        all_ok = all_ok and ok
        say("  [%s] %s" % ("PASS" if ok else "FAIL", label))
    say("")
    say("ALL CHECKS PASS: %s" % all_ok)

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s" % LOG)


if __name__ == "__main__":
    main()
