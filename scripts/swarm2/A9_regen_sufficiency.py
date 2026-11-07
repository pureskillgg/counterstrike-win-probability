"""A9 audit: regenerate the sufficiency JSON and Markdown into log-named files, then compare.

Each mode runs one sufficiency script's main() unchanged, with its output paths redirected under
logs/swarm2 (no file under notes/ or models/ is written). The original scripts are not edited.

Modes (one process each, so OMP/thread environment does not leak between scripts):
  lc   learning_curve.py main() -> logs/swarm2/A9_regen_lc.json.log; then learning_curve_md.py -> A9_regen_lc.md.log
  bh   batch_holdout.py main()  -> logs/swarm2/A9_regen_bh.json.log and A9_regen_bh.md.log
  cp   cross_period.py main()   -> logs/swarm2/A9_regen_cp.json.log and A9_regen_cp.md.log
  cmp  compare each regenerated file with its notes/ original (run after the others)

Usage (project root): python.exe -B -I -X utf8 scripts\\swarm2\\A9_regen_sufficiency.py <mode>
"""

import difflib
import importlib.util
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SUF = ROOT / "scripts" / "model" / "sufficiency"
ORIG = ROOT / "notes" / "model" / "sufficiency"
LOG = ROOT / "logs" / "swarm2"
IGNORE_KEYS = {"fit_s", "wall_s", "runtime_s", "run_at", "label"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def regen_lc():
    mod = load("learning_curve", SUF / "learning_curve.py")
    mod.OUT_DIR = LOG
    mod.OUT_JSON = LOG / "A9_regen_lc.json.log"
    mod.main()
    md = load("learning_curve_md", SUF / "learning_curve_md.py")
    md.OUT_DIR = LOG
    md.IN_JSON = LOG / "A9_regen_lc.json.log"
    md.OUT_MD = LOG / "A9_regen_lc.md.log"
    md.main()


def regen_bh():
    mod = load("batch_holdout", SUF / "batch_holdout.py")
    mod.OUT_DIR = LOG
    mod.OUT_JSON = LOG / "A9_regen_bh.json.log"
    mod.OUT_MD = LOG / "A9_regen_bh.md.log"
    mod.main()


def regen_cp():
    mod = load("cross_period", SUF / "cross_period.py")
    mod.OUT_DIR = LOG
    mod.OUT_JSON = LOG / "A9_regen_cp.json.log"
    mod.OUT_MD = LOG / "A9_regen_cp.md.log"
    mod.main()


def diff_json(a, b, path="", out=None):
    if out is None:
        out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE_KEYS:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'orig' if k in a else 'regen'}")
                continue
            diff_json(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: list length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_json(x, y, f"{path}[{i}]", out)
    else:
        if isinstance(a, float) and isinstance(b, float):
            if a != b:
                out.append(f"{path}: orig {a!r} regen {b!r} (diff {b - a:+.3e})")
        elif a != b:
            out.append(f"{path}: orig {a!r} regen {b!r}")
    return out


def compare(name, orig_json, regen_json, orig_md=None, regen_md=None):
    o = json.loads(orig_json.read_text(encoding="utf-8"))
    r = json.loads(regen_json.read_text(encoding="utf-8"))
    d = diff_json(o, r)
    print(f"== {name}: JSON differences (ignoring {sorted(IGNORE_KEYS)}): {len(d)}", flush=True)
    for line in d[:60]:
        print("  " + line, flush=True)
    if orig_md is not None and regen_md is not None:
        a = orig_md.read_text(encoding="utf-8").splitlines()
        b = regen_md.read_text(encoding="utf-8").splitlines()
        ud = list(difflib.unified_diff(a, b, "orig", "regen", lineterm="", n=0))
        print(f"== {name}: MD unified-diff lines: {len(ud)}", flush=True)
        for line in ud[:80]:
            print("  " + line, flush=True)


def cmp_all():
    compare("learning_curve", ORIG / "learning_curve.json", LOG / "A9_regen_lc.json.log",
            ORIG / "learning_curve.md", LOG / "A9_regen_lc.md.log")
    compare("batch_holdout", ORIG / "batch_holdout.json", LOG / "A9_regen_bh.json.log",
            ORIG / "batch_holdout.md", LOG / "A9_regen_bh.md.log")
    compare("cross_period", ORIG / "cross_period.json", LOG / "A9_regen_cp.json.log",
            ORIG / "cross_period.md", LOG / "A9_regen_cp.md.log")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "lc":
        regen_lc()
    elif mode == "bh":
        regen_bh()
    elif mode == "cp":
        regen_cp()
    elif mode == "cmp":
        cmp_all()
    else:
        raise SystemExit("mode must be lc, bh, cp or cmp")


if __name__ == "__main__":
    main()
