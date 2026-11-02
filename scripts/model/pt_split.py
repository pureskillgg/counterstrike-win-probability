"""Cache the per-second grid as numpy files, split by match (bake-off primary split, seed 0).

Output (data/viz/pt_cache): train.npz, val.npz, test.npz with
  X (float32, columns = feat), y (int8), mid (match index), rid (round index), wr (round weight 1/points in round),
  feat (feature names, bake-off order); test.npz also has p_base (the CPU XGBoost grid model's predictions).
Anything can load these with numpy alone, in any environment.
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
VIZ = ROOT / "data" / "viz"
CACHE = VIZ / "pt_cache"

spec = importlib.util.spec_from_file_location("extract_state_1hz", ROOT / "scripts" / "eda" / "extract_state_1hz.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
bk = ex.bk
kept = ex.bakeoff_kept()
H = kept["H"]
sp = bk.make_splits(kept["kept_final"], H)[bk.PRIMARY]
FE = list(bk.FEATS)

G = pl.read_parquet(VIZ / "per_tick_states.parquet")
preds = pl.read_parquet(VIZ / "per_tick_test_preds.parquet")
CACHE.mkdir(parents=True, exist_ok=True)
info = {}
for name in ("train", "val", "test"):
    d = G.filter(pl.col("match_key").is_in(sorted(sp[name])))
    mk = d["match_key"].to_list()
    rs = d["round"].to_list()
    um = {m: i for i, m in enumerate(sorted(set(mk)))}
    mid = np.array([um[m] for m in mk], dtype=np.int32)
    ur = {}
    rid = np.empty(len(mk), dtype=np.int32)
    for i, key in enumerate(zip(mk, rs)):
        rid[i] = ur.setdefault(key, len(ur))
    arrs = dict(X=d.select(FE).to_numpy().astype(np.float32), y=d["label_ct_win"].to_numpy().astype(np.int8),
                mid=mid, rid=rid, wr=d["w_round"].to_numpy().astype(np.float32), feat=np.array(FE))
    if name == "test":
        assert d.height == preds.height
        assert (d["tick"].to_numpy() == preds["tick"].to_numpy()).all()
        assert d["match_key"].to_list() == preds["match_key"].to_list()
        arrs["p_base"] = preds["p_xgb_grid"].to_numpy().astype(np.float64)
    np.savez_compressed(CACHE / f"{name}.npz", **arrs)
    info[name] = {"rows": d.height, "matches": len(um), "rounds": len(ur)}
    print(name, info[name], flush=True)
(VIZ / "split.json").write_text(json.dumps({"features": FE, "sets": info,
                                            "test_matches": sorted(sp["test"])[:3], "note": "full lists derivable via bk.make_splits"}, indent=1))
print("done")
