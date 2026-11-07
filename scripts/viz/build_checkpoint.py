"""Progress checkpoint: a hard, hash-verified archive of the project as it stands, to go back to later.

  ...python.exe -I -X utf8 scripts\\viz\\build_checkpoint.py

Writes deliverables/checkpoints/CHECKPOINT_<stamp>.zip, CHECKPOINT_<stamp>_manifest.json (sha256 of every file in the zip, plus the git
state) and CHECKPOINT_<stamp>.zip.sha256. Nothing is committed, pushed or deleted.

Left out on purpose (rebuildable, and too large): tomes/, .pixi/ (the Python environment), every *.parquet and *.npy cache, __pycache__,
the Meridian output cache, the hand-review copy of the email package, and older .bak files.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STAMP = time.strftime("%Y-%m-%d_%H%M")
OUT_DIR = os.path.join(ROOT, "deliverables", "checkpoints")
ZIP = os.path.join(OUT_DIR, f"CHECKPOINT_{STAMP}.zip")
MANIFEST = os.path.join(OUT_DIR, f"CHECKPOINT_{STAMP}_manifest.json")
SHA_FILE = ZIP + ".sha256"

INCLUDE_TREES = ["deliverables", "notes", "scripts", "models", "logs", "data/combined"]
ROOT_FILES = ["pixi.toml", "pixi.lock", "pyproject.toml", "README.md"]
EXCLUDE_DIRS = {"__pycache__", ".meridian-outputs-cache", "checkpoints", "email_package_2026-10-19_review"}
EXCLUDE_SUFFIXES = (".parquet", ".npy", ".pyc", ".bak", ".edit_tmp.docx")
DATA_ONLY_SUFFIXES = (".json", ".md", ".txt", ".log", ".csv")  # data/combined: small results only, no caches


def git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as exc:  # the checkpoint still works without git
        return f"(git unavailable: {exc})"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    files = []
    for tree in INCLUDE_TREES:
        base = os.path.join(ROOT, tree.replace("/", os.sep))
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for name in filenames:
                if name.startswith("~$") or name.endswith(EXCLUDE_SUFFIXES):
                    continue
                if tree == "data/combined" and not name.endswith(DATA_ONLY_SUFFIXES):
                    continue
                full = os.path.join(dirpath, name)
                files.append(os.path.relpath(full, ROOT).replace(os.sep, "/"))
    for name in ROOT_FILES:
        if os.path.isfile(os.path.join(ROOT, name)):
            files.append(name)
    return sorted(set(files))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = collect()
    git_head = git("rev-parse", "HEAD")
    git_branch = git("rev-parse", "--abbrev-ref", "HEAD")
    git_status = git("status", "--short").splitlines()
    rows = []
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            full = os.path.join(ROOT, rel.replace("/", os.sep))
            z.write(full, arcname=rel)
            rows.append({"path": rel, "bytes": os.path.getsize(full), "sha256": sha256(full)})
        readme = (
            f"CS2 round win probability: progress checkpoint {STAMP}\n"
            f"git base: {git_head} on branch {git_branch}\n"
            f"Working-tree changes at checkpoint time: {len(git_status)} (listed in the manifest, git_status).\n\n"
            "Restore: unzip into an empty folder. The files keep their project paths. For the git base, check out the commit\n"
            "above, then copy the files from this archive over it.\n\n"
            "Not in this archive (rebuild or restore separately): tomes/ (raw tome data), data/combined/*.parquet and *.npy caches,\n"
            "the Python environment (.pixi, restore with pixi install), __pycache__, and the Meridian output cache.\n"
            "Every file's sha256 is in the manifest; the archive's own sha256 is in the .sha256 file beside it.\n"
        )
        z.writestr("CHECKPOINT_README.txt", readme)
    manifest = {
        "stamp": STAMP,
        "project_root": ROOT,
        "git_head": git_head,
        "git_branch": git_branch,
        "git_status": git_status,
        "file_count": len(rows),
        "total_bytes": sum(r["bytes"] for r in rows),
        "files": rows,
    }
    json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), indent=1)
    zip_hash = sha256(ZIP)
    open(SHA_FILE, "w", encoding="utf-8").write(f"{zip_hash}  {os.path.basename(ZIP)}\n")
    print("wrote", ZIP, f"{os.path.getsize(ZIP) // 1024} KB")
    print("wrote", MANIFEST, "|", len(rows), "files |", f"{manifest['total_bytes'] // (1 << 20)} MB uncompressed")
    print("zip sha256:", zip_hash)
    print("git:", git_head[:12], git_branch, "| working-tree changes:", len(git_status))


if __name__ == "__main__":
    main()
