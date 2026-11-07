"""Provenance snapshot of the deliverable: what exactly was produced, from what, by which code. Repeatable (run it after every milestone).

  python -I -X utf8 scripts\\provenance.py [--tag final] [--out notes\\PROVENANCE_<tag>_<time>.json]

Records, as one JSON file plus a short text summary beside it:
  * git: HEAD, branch, dirty files (so a result can be tied to a commit, or flagged as produced by uncommitted code)
  * sha256 + size + mtime of: models/*.json, the data/<store> result JSONs and prediction samples, the split manifest, the match / round / grid
    tables, the ledger of the combined tome store (COMBINED_BUILD.json) and its page counts, selected notes, and every script under scripts/
  * the counts the deck quotes (matches in the store, usable matches, rows, split sizes) read from the files, not typed
  * python, polars, numpy, xgboost versions of the CPU environment
Large files (parquet tables) are hashed in 1 MiB blocks; a hash of a file that is still being written is flagged by comparing mtime before and after.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = r"G:\pureskill\csds-cs2"
sys.stdout.reconfigure(encoding="utf-8")


def sha256(path):
    m1 = os.path.getmtime(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest(), os.path.getmtime(path) == m1


def entry(path):
    try:
        d, stable = sha256(path)
        st = os.stat(path)
        return {"sha256": d, "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)), **({} if stable else {"unstable": True})}
    except OSError as e:
        return {"error": str(e)[:100]}


def git(*a):
    try:
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"git failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    ap.add_argument("--store", default="combined")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-tables", action="store_true", help="skip hashing the big parquet tables")
    a = ap.parse_args()
    stamp = time.strftime("%Y-%m-%d_%H%M")
    out = a.out or os.path.join(ROOT, "notes", f"PROVENANCE_{a.tag}_{stamp}.json")
    d = os.path.join(ROOT, "data", a.store)
    rec = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": a.tag, "store": a.store}
    rec["git"] = {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "subject": git("log", "-1", "--format=%s"),
                  "dirty": [ln for ln in git("status", "--short").splitlines() if not ln.startswith("??")][:200], "untracked_count": sum(1 for ln in git("status", "--short").splitlines() if ln.startswith("??"))}
    files = {}
    mdir = os.path.join(ROOT, "models")
    for f in sorted(os.listdir(mdir)):
        files[f"models/{f}"] = entry(os.path.join(mdir, f))
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if os.path.isfile(p) and (f.endswith(".json") or (f.startswith("export_") and f.endswith(".parquet")) or f == "split_manifest.json"):
            files[f"data/{a.store}/{f}"] = entry(p)
    if not a.no_tables:
        for f in ("matches.parquet", "rounds.parquet", "grid.parquet", "deaths.parquet", "cache_snapshot_2026.parquet"):
            if os.path.exists(os.path.join(d, f)):
                files[f"data/{a.store}/{f}"] = entry(os.path.join(d, f))
    led = os.path.join(ROOT, "tomes", a.store, "COMBINED_BUILD.json")
    if os.path.exists(led):
        files[f"tomes/{a.store}/COMBINED_BUILD.json"] = entry(led)
        L = json.load(open(led, encoding="utf-8"))
        rec["tome_ledger"] = {"sources": len(L.get("sources", [])), "channels": len(L.get("channels", {})), "keys_in_header": L.get("channels", {}).get("header", {}).get("keys")}
    for dp, _, fn in os.walk(os.path.join(ROOT, "scripts")):
        if any(x in dp for x in ("__pycache__", os.sep + "swarm", os.sep + "research", os.sep + "eda")):
            continue
        for f in fn:
            if f.endswith((".py", ".ps1", ".json")) and "mini_selection" not in f:
                files["scripts/" + os.path.relpath(os.path.join(dp, f), os.path.join(ROOT, "scripts")).replace("\\", "/")] = entry(os.path.join(dp, f))
    for f in ("STATUS_2026-10-14.md", "TOME_INCREMENTAL_BUILD_DESIGN.md", "PRESENTATION_NOTES.md"):
        p = os.path.join(ROOT, "notes", f)
        if os.path.exists(p):
            files["notes/" + f] = entry(p)
    rec["files"] = files
    # the numbers the deck quotes, read from the files
    nums = {}
    try:
        import polars as pl
        M = pl.read_parquet(os.path.join(d, "matches.parquet"))
        nums["matches_in_store"] = M.height
        nums["matches_by_era"] = dict(M.group_by("era").len().iter_rows())
        R = pl.read_parquet(os.path.join(d, "rounds.parquet"), columns=["match_key", "use", "era"]).filter(pl.col("use"))
        nums["usable_rounds"] = R.height
        nums["usable_matches"] = R["match_key"].n_unique()
        nums["grid_rows"] = pl.scan_parquet(os.path.join(d, "grid.parquet")).select(pl.len()).collect().item()
        man = json.load(open(os.path.join(d, "split_manifest.json"), encoding="utf-8"))
        nums["split_manifest"] = {"sha256": man["sha256"], "history": man["history"], "sizes": {s: sum(1 for v in man["assign"].values() if v == s) for s in ("train", "val", "test")}}
        import numpy
        import xgboost
        nums["versions"] = {"python": sys.version.split()[0], "polars": pl.__version__, "numpy": numpy.__version__, "xgboost": xgboost.__version__}
    except Exception as e:  # noqa: BLE001
        nums["error"] = f"{type(e).__name__}: {e}"[:200]
    rec["numbers"] = nums
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    json.dump(rec, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, out)
    txt = out[:-5] + ".txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"Provenance {a.tag} {rec['created']}\ngit {rec['git']['head'][:12]} on {rec['git']['branch']}: {rec['git']['subject']}\n"
                f"dirty tracked files: {len(rec['git']['dirty'])}, untracked: {rec['git']['untracked_count']}\n{json.dumps(nums, indent=1)}\n\n")
        for k, v in files.items():
            if k.startswith(("models/", f"data/{a.store}/export_", f"data/{a.store}/split", f"data/{a.store}/stress", f"data/{a.store}/e2e", f"data/{a.store}/parity", f"data/{a.store}/mutation")) or k.endswith(".parquet"):
                f.write(f"{v.get('sha256', '?')[:16]}  {v.get('bytes', 0):>13,}  {v.get('mtime', '')}  {k}\n")
    print(f"-> {out}\n-> {txt}\nfiles hashed: {len(files)}")


if __name__ == "__main__":
    main()
