"""Compare code/doc files in the G: workspace (and the session scratchpad) with the counterstrike repo.

Writes sync_inventory.json: for each candidate file, its status vs the repo (same / differs / missing / moved-same / moved-differs).
Reads file bytes only to hash them. Skips data folders entirely.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(r"C:/Users/<user>/Documents/pureskill/counterstrike-win-probability")
WS = Path(r"G:/pureskill/csds-cs2")
SCR = Path(r"C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad")
OUT = SCR / "sync_inventory.json"

CODE_EXT = {".py", ".md", ".sh", ".toml", ".yaml", ".yml", ".txt", ".ps1", ".js", ".ipynb", ".sql", ".r", ".bat", ".cfg", ".ini"}
SMALL_EXT = {".json", ".csv", ".tsv"}  # case by case: only if small
SMALL_LIMIT = 200_000
SKIP_DIRS_WS = {"data", "tomes", "logs", "catboost_info", ".git", ".pixi", "source_2024", "merged_2024", "models",
                "staging_cs2data_gz", "__pycache__", ".meridian-outputs-cache", "node_modules"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def candidates(root, skip_dirs, skip_prefixes=()):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        parts = rel_dir.parts
        if parts and (parts[0] in skip_dirs or any(parts[0].startswith(p) for p in skip_prefixes)):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if ext in CODE_EXT or (ext in SMALL_EXT and size <= SMALL_LIMIT):
                if fn.endswith(".stackdump") or ".env" in fn.lower():
                    continue
                yield p, size


repo_by_hash = {}
repo_by_name = {}
for p in REPO.rglob("*"):
    if p.is_file() and ".git" not in p.parts:
        try:
            h = sha(p)
        except OSError:
            continue
        repo_by_hash.setdefault(h, []).append(str(p.relative_to(REPO)).replace("\\", "/"))
        repo_by_name.setdefault(p.name, []).append(str(p.relative_to(REPO)).replace("\\", "/"))

rows = []
for src, label, skip, pref in [(WS, "workspace", SKIP_DIRS_WS, ("staging", "evac_")), (SCR, "scratchpad", {"render", "handover", "keys_check", "reg_results", "dl_models_gdt_objective", "dl_results_pt", "dl_runlogs_all", "dl_metrics_tcnr", "dl_metrics_probe", "__pycache__"}, ())]:
    for p, size in candidates(src, skip, pref):
        rel = str(p.relative_to(src)).replace("\\", "/")
        h = sha(p)
        same_path = REPO / rel
        if same_path.exists():
            status = "same" if sha(same_path) == h else "differs"
        elif h in repo_by_hash:
            status = "moved-same"
        elif p.name in repo_by_name:
            status = "moved-differs"
        else:
            status = "missing"
        rows.append({"origin": label, "rel": rel, "size": size, "status": status,
                     "repo_matches": repo_by_hash.get(h, [])[:3] if status.startswith("moved") or status == "same" else repo_by_name.get(p.name, [])[:3]})

OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
from collections import Counter
for origin in ("workspace", "scratchpad"):
    c = Counter((r["status"]) for r in rows if r["origin"] == origin)
    sz = sum(r["size"] for r in rows if r["origin"] == origin and r["status"] in ("missing", "differs", "moved-differs"))
    print(origin, dict(c), f"bytes to consider: {sz/1e6:.1f} MB")
    top = Counter(r["rel"].split("/")[0] for r in rows if r["origin"] == origin and r["status"] in ("missing", "differs", "moved-differs"))
    print("  by top folder:", dict(top.most_common(25)))
print("wrote", OUT)
