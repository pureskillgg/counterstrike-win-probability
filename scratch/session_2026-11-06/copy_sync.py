"""Copy workspace code/docs and small models into the counterstrike repo (local working tree only; no git commands here)."""
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SCR = Path(r"C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad")
WS = Path(r"G:/pureskill/csds-cs2")
REPO = Path(r"C:/Users/<user>/Documents/pureskill/counterstrike-win-probability")
rows = json.loads((SCR / "sync_inventory.json").read_text(encoding="utf-8"))

# 1. workspace code and docs: missing, differs and moved-differs, same relative path
n = 0
size = 0
for r in rows:
    if r["origin"] != "workspace" or r["status"] not in ("missing", "differs", "moved-differs"):
        continue
    src = WS / r["rel"]
    dst = REPO / r["rel"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    n += 1
    size += r["size"]
print(f"workspace code/docs copied: {n} files, {size/1e6:.1f} MB")

# 2. models: XGBoost JSON boosters and metadata (flat, as in the workspace), minus tuning variants and killed runs
SKIP_SUFFIX = ("_es50", "_es200", "_nomono", "_s1", "_s2")
m = 0
msize = 0
skipped = []
for p in sorted((WS / "models").glob("*.json")):
    stem = p.name.replace(".meta.json", "").replace(".json", "").replace("_meta", "")
    if any(stem.endswith(s) for s in SKIP_SUFFIX):
        skipped.append(p.name)
        continue
    dst = REPO / "models" / p.name
    if not dst.exists() or dst.stat().st_size != p.stat().st_size:
        shutil.copy2(p, dst)
        m += 1
        msize += p.stat().st_size
# sequence GDT weights (+ their metrics json), temporal and round_state weights, no prediction files
for sub in ("seq_gdt_weights", "temporal", "round_state"):
    for p in sorted((WS / "models" / sub).rglob("*")):
        if not p.is_file() or "preds" in p.name.lower():
            if p.is_file():
                skipped.append(f"{sub}/{p.name}")
            continue
        if p.suffix.lower() not in (".pt", ".json"):
            continue
        dst = REPO / "models" / sub / p.relative_to(WS / "models" / sub)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        m += 1
        msize += p.stat().st_size
print(f"model files copied: {m}, {msize/1e6:.1f} MB")
print(f"model files skipped ({len(skipped)}):", skipped)
(SCR / "models_skipped.json").write_text(json.dumps(skipped, indent=1), encoding="utf-8")
