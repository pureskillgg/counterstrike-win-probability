"""A9 audit probe (read-only): do XGBoost hist quantile cut points depend on thread count?

Same 100%-row training set as learning_curve.py. Builds a QuantileDMatrix with max_bin 256 (the
XGBoost hist default) for several nthread values and compares the cut values.

Usage (project root): python.exe -B -I -X utf8 scripts\\swarm2\\A9_probe_cuts.py
"""

import collections
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = load_module("extract_state_1hz", ROOT / "scripts" / "eda" / "extract_state_1hz.py")
bk = ex.bk


def main():
    kept = ex.bakeoff_kept()
    H = kept["H"]
    rows = []
    for k in kept["rounds_ok"]:
        rr = {**kept["rd"][k], "freeze": kept["live"][k]["tick"], "t_score": kept["live"][k]["t"],
              "ct_score": kept["live"][k]["ct"]}
        rows.extend(bk.build_states(k, rr, kept["deaths"].get(k, []), kept["plants"].get(k, []),
                                    kept["players"].get(k, {}), H[k[0]]["max_rounds"], collections.Counter()))
    S = pl.DataFrame(rows)
    sp = bk.make_splits(kept["kept_final"], H)[bk.PRIMARY]
    tr = S.filter(pl.col("match_key").is_in(sorted(sp["train"])))
    X = tr.select(bk.FEATS).to_numpy().astype(np.float32)
    y = tr["label_ct_win"].to_numpy().astype(np.float32)
    base = None
    for nt in (1, 2, 4, 16):
        dm = xgb.QuantileDMatrix(X, y, max_bin=256, nthread=nt)
        ptrs, vals = dm.get_quantile_cut()
        vals = np.asarray(vals)
        ptrs = np.asarray(ptrs)
        if base is None:
            base = (ptrs, vals)
            print(f"nthread {nt}: cut values {len(vals)}, feature offsets {ptrs.tolist()}", flush=True)
            continue
        same = np.array_equal(ptrs, base[0]) and np.array_equal(vals, base[1])
        dif = np.max(np.abs(vals - base[1])) if vals.shape == base[1].shape else float("nan")
        print(f"nthread {nt}: identical cuts to nthread 1: {same}; max |diff| {dif:.3e}; "
              f"n cuts {len(vals)} vs {len(base[1])}", flush=True)


if __name__ == "__main__":
    main()
