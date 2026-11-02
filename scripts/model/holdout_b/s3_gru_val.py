"""S3: reproduce the three all-maps GRU (loadout, no distance features) seeds on VALIDATION rows from their saved weights.

Inference only (no training). Per map, only the rows of validation matches (split == 1 in the frozen round-state file) are
featurised, exactly as train_temporal_allmaps.load_all does (float16 features: round-state X and loadout; float32 state: bomb
channels plus map one-hot). Windows are built per map: they never cross a match, and every row of a validation match is a
validation row, so the per-map windows equal the global ones. Predictions are scored on usable validation rows, as in run_seed().
Holdout keys are asserted absent. Writes data/holdout_b/s3_gru_val_reproduction.json only.
Run in the torch environment:  G:/pureskill/csds-cs2/.pixi/envs/torch/python.exe -I -X utf8 -u scripts/model/holdout_b/s3_gru_val.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch

ROOT = r"G:\pureskill\csds-cs2"
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
import train_temporal_round_model as ttr  # noqa: E402
import train_temporal_allmaps as T  # noqa: E402

t0 = time.time()
VAL = ttr.VAL
K = 20
SEEDS = {0: "allmaps_gru_loadout_valpreds", 1: "allmaps_gru_loadout_seed1_valpreds", 2: "allmaps_gru_loadout_seed2_valpreds"}
HOLD = np.array(sorted(open(os.path.join(ROOT, "data", "holdout_b", "holdout_keys.txt"), encoding="utf-8").read().split()))
torch.set_num_threads(max(1, os.cpu_count() or 1))
nets = {}
for s, tag in SEEDS.items():
    sd = torch.load(os.path.join(ROOT, "models", "temporal", tag + ".pt"), map_location="cpu")
    net = ttr.TemporalRoundModel(in_dim=T.N_FEAT, state_dim=3 + len(T.MAPS), gat=False)
    net.load_state_dict(sd)
    net.eval()
    nets[s] = net
print("weights loaded", flush=True)
acc = {s: {"match_key": [], "round": [], "second": [], "map_id": [], "label": [], "p": []} for s in SEEDS}
n_rows_val = 0
for mi, m in enumerate(T.MAPS):
    arr, X, table, lab, st, use = ttr.build_table(m, f"{m}_ext", T.CACHE)
    rows = np.flatnonzero(arr["split"] == VAL)
    mkeys = arr["match_keys"][arr["match_idx"][rows]]
    assert not np.isin(mkeys, HOLD).any(), "holdout key in validation rows"
    lo = np.load(T.DATA / f"{m}_ext_loadout.npy", mmap_mode="r")
    feat = np.empty((rows.size, ttr.N_SLOTS, T.N_FEAT), dtype=np.float16)
    for c in range(0, rows.size, 500_000):
        e = min(rows.size, c + 500_000)
        r = rows[c:e]
        feat[c:e, :, :6] = X[r]
        feat[c:e, :, 6:] = lo[r]
    state = np.zeros((rows.size, 3 + len(T.MAPS)), dtype=np.float32)
    state[:, :3] = st[rows]
    state[:, 3 + mi] = 1.0
    prev = ttr.previous_rows(arr["match_idx"][rows].astype(np.int64), arr["round"][rows].astype(np.int16),
                             arr["second"][rows].astype(np.int16), K).astype(np.int64)
    feat_t = torch.as_tensor(feat)
    state_t = torch.as_tensor(state)
    use_pos = np.flatnonzero(use[rows])
    labels = lab[rows][use_pos].astype(np.int64)

    def gather(r):
        idx = prev[r]
        mask = torch.as_tensor(idx >= 0, dtype=torch.float32)
        safe = torch.as_tensor(np.where(idx >= 0, idx, 0), dtype=torch.long)
        Xw = feat_t[safe].float() * mask[..., None, None]
        Sw = state_t[safe]
        return Xw.flip(1), Sw.flip(1), mask.flip(1)

    for s, net in nets.items():
        out = np.empty(use_pos.size, dtype=np.float32)
        with torch.no_grad():
            for b in range(0, use_pos.size, 8192):
                r = use_pos[b:b + 8192]
                Xw, Sw, mk = gather(r)
                out[b:b + 8192] = torch.sigmoid(net(Xw, Sw, mk)).numpy()
        a = acc[s]
        a["match_key"].append(mkeys[use_pos])
        a["round"].append(arr["round"][rows][use_pos].astype(np.int64))
        a["second"].append(arr["second"][rows][use_pos].astype(np.int64))
        a["map_id"].append(np.full(use_pos.size, mi, dtype=np.int64))
        a["label"].append(labels)
        a["p"].append(out.astype(np.float64))
    n_rows_val += use_pos.size
    print(f"  {m}: validation rows {rows.size:,}, usable {use_pos.size:,} ({time.time() - t0:.0f} s)", flush=True)
    del X, lo, feat, feat_t, state_t, prev, arr, table, st, lab, use

res = {"validation_usable_rows": int(n_rows_val), "seeds": {}}
rec = {0: "data/temporal/allmaps_gru_loadout_valpreds_metrics.json", 1: "data/temporal/allmaps_gru_loadout_seed1_valpreds_metrics.json",
       2: "data/temporal/allmaps_gru_loadout_seed2_valpreds_metrics.json"}
for s, tag in SEEDS.items():
    a = {k: np.concatenate(v) for k, v in acc[s].items()}
    ll = float(ttr.log_loss(a["label"], a["p"]))
    met = json.load(open(os.path.join(ROOT, rec[s]), encoding="utf-8"))
    sel = met["selected_epoch"]
    rec_ll = met["history"][sel - 1]["val_log_loss"]
    saved = pl.read_parquet(os.path.join(ROOT, "data", "temporal", tag + "_val_preds.parquet"))
    mine = pl.DataFrame({"match_key": a["match_key"], "round": a["round"], "second": a["second"], "p_new": a["p"]})
    j = mine.join(saved.select("match_key", "round", "second", pl.col("p").alias("p_saved")), on=["match_key", "round", "second"], how="full", coalesce=True)
    both = j.filter(pl.col("p_new").is_not_null() & pl.col("p_saved").is_not_null())
    d = np.abs(both["p_new"].to_numpy() - both["p_saved"].to_numpy())
    res["seeds"][str(s)] = {
        "tag": tag, "selected_epoch": sel, "recorded_val_log_loss": rec_ll, "reproduced_val_log_loss": ll,
        "abs_diff_log_loss": abs(ll - rec_ll), "rows_reproduced": int(a["p"].size), "rows_saved": int(saved.height),
        "rows_matched_on_key": int(both.height), "rows_only_in_one": int(j.height - both.height),
        "max_abs_diff_p": float(d.max()) if d.size else None, "mean_abs_diff_p": float(d.mean()) if d.size else None,
    }
    print(json.dumps({"seed": s, **res["seeds"][str(s)]}), flush=True)
res["seconds"] = round(time.time() - t0)
json.dump(res, open(os.path.join(ROOT, "data", "holdout_b", "s3_gru_val_reproduction.json"), "w", encoding="utf-8"), indent=1)
print("done", round(time.time() - t0), "s", flush=True)
