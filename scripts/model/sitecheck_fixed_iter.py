"""Fixed-iteration version of the distance tree (SN), as a sensitivity check on the validation-tuned comparator.

SN in sitecheck_xgb.py stops early on the validation split, and its validation predictions use the best iteration found there.
This script fixes the number of trees before any validation row is seen:
  1. split the TRAINING matches (only) into 80% fit and 20% inner check, by a fixed hash of match_key;
  2. fit the SN features with early stopping on the inner check: best iteration B;
  3. refit on all training matches with exactly B trees (no early stopping);
  4. predict the validation seconds with that model. No validation row is used for any choice.
Same features as SN (57 frozen inputs and the four train-only distances), same recipe (W.fit_xgb settings), CPU.

Reports, per seed, on validation: SN_fixed minus SN (how much the validation-tuned tree is flattered), and SN_fixed minus the
sweep GRU at 1e-3 (EMA final), when the sweep files exist. Test-split rows are not read: the cache scan keeps only train and
validation matches, and the manifest is used for the split only.

  ...tabicl python.exe -I -X utf8 -u scripts/model/sitecheck_fixed_iter.py --seed 0
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEY = ["match_key", "round", "second"]
ST = [f"st{i}" for i in range(4)]
NST = [f"nst{i}" for i in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DIAG, "allmaps_sn_fixed_iter_seed{seed}.json"))
    a = ap.parse_args()
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    cols = feats + NST
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    tr_keys = [k for k, v in man.items() if v == "train"]
    va_keys = [k for k, v in man.items() if v == "val"]
    inner_val = {k for k in tr_keys if zlib.crc32(k.encode()) % 5 == 0}
    geo_old = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo", f"{m}.parquet")) for m in MAPS])
    geo_new = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo_trainsites", f"{m}.parquet")) for m in MAPS])
    geo_new = geo_new.rename({c: "n" + c for c in ST})
    keep = pl.DataFrame({"match_key": tr_keys + va_keys})
    C = (pl.scan_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"))
         .select(KEY + ["label_ct_win", "map_name"] + feats)
         .join(keep.lazy(), on="match_key", how="inner")
         .collect())
    C = W.drop_roster_defect_rounds(C)
    C = C.filter(pl.col("label_ct_win").is_not_null())
    C = C.with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    G = C.join(geo_old, on=KEY, how="inner").join(geo_new, on=KEY, how="inner").join(A, on="match_key", how="inner")
    G = G.unique(KEY, keep="first", maintain_order=True).sort(KEY)
    print(f"rows {G.height:,} ({time.time() - t0:.0f} s)", flush=True)

    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    in_fit = tr.filter(~pl.col("match_key").is_in(list(inner_val)))
    in_chk = tr.filter(pl.col("match_key").is_in(list(inner_val)))
    Xf, yf = in_fit.select(cols).to_numpy().astype(np.float32), in_fit["label_ct_win"].to_numpy().astype(np.int64)
    Xc, yc = in_chk.select(cols).to_numpy().astype(np.float32), in_chk["label_ct_win"].to_numpy().astype(np.int64)
    mono = W.mono_for(cols)
    _, B = W.fit_xgb(Xf, yf, Xc, yc, mono=mono, seed=a.seed, device="cpu")
    print(f"inner early stopping on training matches only: best iteration B = {B} ({time.time() - t0:.0f} s)", flush=True)

    params = dict(n_estimators=int(B), learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
                  tree_method="hist", device="cpu", random_state=a.seed, eval_metric="logloss")
    if mono is not None:
        params["monotone_constraints"] = tuple(mono)
    model = xgb.XGBClassifier(**params)
    X_tr = tr.select(cols).to_numpy().astype(np.float32)
    y_tr = tr["label_ct_win"].to_numpy().astype(np.int64)
    model.fit(X_tr, y_tr, verbose=False)
    X_va = va.select(cols).to_numpy().astype(np.float32)
    y_va = va["label_ct_win"].to_numpy().astype(np.int64)
    p_fix = model.predict_proba(X_va)[:, 1]
    print(f"refit on all training matches with {B} trees; validation log loss {float(W.ll_vec(p_fix, y_va).mean()):.6f} "
          f"({time.time() - t0:.0f} s)", flush=True)

    match_va = va["match_key"].to_numpy()
    sn = (pl.scan_parquet(os.path.join(DIAG, "preds", f"allmaps_xgb_SN_seed{a.seed}.parquet"))
          .filter(pl.col("split") == "val").select(KEY + ["label", "p"]).collect().rename({"p": "p_sn"}))
    j = va.select(KEY + ["match_key"]).with_columns(pl.Series("p_fix", p_fix)).join(sn, on=KEY, how="inner")
    assert j.height == va.height, (j.height, va.height)
    yj = j["label"].to_numpy().astype(np.int64)
    res = {"seed": a.seed, "best_iteration_inner": int(B), "validation_rows": int(j.height),
           "SN_fixed_minus_SN": paired(j["match_key"].to_numpy(), yj, j["p_fix"].to_numpy(), j["p_sn"].to_numpy())}
    gru = os.path.join(DIAG, "lr_sweep_screen" if a.seed == 0 else f"lr_sweep_all_seed{a.seed}",
                       f"gru_lr0.001_seed{a.seed}_ema_final_val_preds.parquet")
    if os.path.exists(gru):
        g = pl.read_parquet(gru).select(KEY + ["label", "p"]).rename({"p": "p_gru"})
        jg = j.join(g, on=KEY, how="inner")
        res["SN_fixed_minus_GRU_cosine"] = paired(jg["match_key"].to_numpy(), jg["label"].to_numpy().astype(np.int64),
                                                  jg["p_fix"].to_numpy(), jg["p_gru"].to_numpy())
    out = a.out.format(seed=a.seed)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print(f"saved {out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
