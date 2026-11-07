"""Day manifests for every training input table: matches per upload day and split.

Reads only the match_key column (and a split column when the table has one). Never reads per-second rows.
Writes CSVs and a summary JSON into the repo under data/manifests/.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
WS = Path("G:/pureskill/csds-cs2")
D = WS / "data"
REPO = Path("C:/Users/13144/Documents/pureskill/counterstrike-win-probability")
OUT = REPO / "data" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)
SCR = Path("C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad")
DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")
SPLIT_NAME = {0: "train", 1: "validation", 2: "test", "0": "train", "1": "validation", "2": "test"}

# export split (57,811 keys), used only when a table has no split column of its own
exp_keys = [str(k) for k in json.load(open(SCR / "keys_check/export_clean_v4_match_keys.json", encoding="utf-8"))]
exp_split = np.load(SCR / "keys_check/match_split_v4.npy")
EXPORT = dict(zip(exp_keys, (SPLIT_NAME[int(v)] for v in exp_split)))


def day_of(k):
    m = DATE.search(k)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "unknown"


def table_keys(path):
    lf = pl.scan_parquet(path)
    cols = lf.collect_schema().names()
    split_col = next((c for c in ("split", "split_code", "split_name") if c in cols), None)
    sel = ["match_key"] + ([split_col] if split_col else [])
    df = lf.select(sel).unique(subset=["match_key"], keep="first").collect()
    keys = df["match_key"].cast(pl.Utf8).to_list()
    if split_col:
        sp = [SPLIT_NAME.get(v, str(v)) for v in df[split_col].to_list()]
        return dict(zip(keys, sp)), f"table column '{split_col}'"
    return {k: EXPORT.get(k, "not_in_export_split") for k in keys}, "export_clean split (match_split.npy)"


def write(name, mapping, source, note):
    per_day = defaultdict(lambda: defaultdict(int))
    for k, s in mapping.items():
        per_day[day_of(k)][s] += 1
    splits = sorted({s for d in per_day.values() for s in d})
    lines = ["day," + ",".join(splits) + ",total"]
    for day in sorted(per_day):
        row = [per_day[day].get(s, 0) for s in splits]
        lines.append(f"{day}," + ",".join(map(str, row)) + f",{sum(row)}")
    (OUT / f"{name}_days.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    days = [d for d in per_day if d != "unknown"]
    totals = defaultdict(int)
    for d in per_day.values():
        for s, c in d.items():
            totals[s] += c
    return {"table": name, "split_source": source, "note": note, "matches": len(mapping), "days": len(days),
            "first_day": min(days) if days else None, "last_day": max(days) if days else None, "by_split": dict(totals)}


summary = []
TABLES = [
    ("cache_snapshot_2026", D / "combined/cache_snapshot_2026.parquet", "XGBoost base inputs (57/58/59/83-input models)"),
    ("cache_snapshot_2026_ext", D / "combined/cache_snapshot_2026_ext.parquet", "sequence-model state inputs (train_phase1_pos24)"),
    ("cache_pos_2026_trainctr", D / "combined/cache_pos_2026_trainctr.parquet", "XGBoost 83-input position inputs"),
    ("matches", D / "combined/matches.parquet", "all matches in the combined store"),
]
for name, path, note in TABLES:
    m, src = table_keys(path)
    summary.append(write(name, m, src, note))
    print(summary[-1])

grid = {}
src_g = None
for f in sorted((D / "diagnostics/allmaps_pos24_trainsites_grid").glob("*.parquet")):
    m, src_g = table_keys(f)
    grid.update(m)
summary.append(write("allmaps_pos24_trainsites_grid", grid, src_g, "sequence-model position inputs (per-map site grid)"))
print(summary[-1])

# round_state arrays: per-row split; take the split of each match's first row
rs = {}
for f in sorted((D / "round_state").glob("*.npz")):
    with np.load(f, allow_pickle=True) as z:
        if not {"match_keys", "match_idx", "split"} <= set(z.files):
            continue
        keys = [str(k) for k in z["match_keys"].tolist()]
        mi = z["match_idx"]
        sp = z["split"]
        first = {}
        for i, s in zip(mi.tolist(), sp.tolist()):
            if i not in first:
                first[i] = s
        for i, s in first.items():
            rs.setdefault(keys[i], SPLIT_NAME.get(int(s), str(s)))
summary.append(write("round_state", rs, "npz 'split' array (first row of each match)", "earlier temporal round models"))
print(summary[-1])

summary.append(write("export_clean_v4", EXPORT, "export_clean match_split.npy", "57,811-match training export (not used by the reported runs)"))
print(summary[-1])

tomes = sorted(p.name for p in (WS / "tomes").iterdir() if p.is_dir())
(OUT / "tome_folders.txt").write_text("\n".join(tomes) + "\n", encoding="utf-8")
(OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("tome folders:", len(tomes), tomes[:12])
