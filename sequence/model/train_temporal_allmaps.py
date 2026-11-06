"""GRU (+loadout) trained on all seven maps, with a map indicator; scored on the frozen de_mirage test matches.

Same model, window and loss as train_temporal_round_model.py, extended to every map so that it can be compared with the
all-maps XGBoost reference on the same test seconds:
  training    the train split of every map (the extended round-state files, <map>_ext.npz, and the extended snapshot cache)
  validation  the validation split of every map, used only to keep the best epoch
  test        the frozen de_mirage test seconds (the rows the 0.4512 reference is scored on)
A one-hot map indicator (7 columns) is appended to each second's state. Windows never cross a match, because the window keys
include the match index. Features are stored as float16 on the GPU to fit in memory; the arithmetic is float32.

--dist adds four straight-line distance features to each second's state: the nearest living CT and the nearest living T to each of the
map's two site centres (make_pos_features.site_centres), in game units divided by 1000 and capped at 10 (10 when the side has no living
player). These are the straight-line half of the distance-to-site features tested on de_mirage; no navigation mesh is used.

--walk adds the same four features measured along the navigation mesh instead of in a straight line (build_nav_walk_allmaps.py, one
file per map: <map>_ext_navwalk.npy). --dist and --walk are exclusive.

Pre-registered (written before any run): the distance model beats the baseline if, over seeds 0, 1 and 2, the mean paired log-loss
difference on the de_mirage test seconds (distance minus baseline, same seed) is <= -0.0005 and every seed's 95% match-cluster CI upper
bound is < 0. The baseline is the all-maps GRU with loadout, same seed (allmaps_gru_loadout[_seedN]_preds.parquet). The walking version
is also compared with the straight-line version (same seeds), with the same rule.

--seeds 0,1,2 runs several seeds in one process: the data is built once, and each seed trains from its own initialisation.
Writes data/temporal/allmaps_gru_loadout[_dist|_walk][_seedN]_preds.parquet (test seconds: match_key, round, second, label, p) and
data/temporal/allmaps_gru_loadout[_dist|_walk][_seedN]_metrics.json.

  ...torch env python.exe -I -X utf8 -u scripts\\model\\train_temporal_allmaps.py --seeds 0,1,2 [--dist | --walk]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
import train_temporal_round_model as ttr  # noqa: E402

MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
CACHE = "cache_snapshot_2026_ext.parquet"
TRAIN, VAL, TEST = ttr.TRAIN, ttr.VAL, ttr.TEST
DATA, OUT, MODELS = ttr.DATA, ttr.OUT, ttr.MODELS
N_FEAT = 18  # 6 state channels + 12 loadout features per slot
N_DIST = 4


def straight_distances(m, X):
    """Per second: nearest living CT and nearest living T to each of the map's two site centres, game units / 1000, capped at 10.
    Columns: site A CT, site A T, site B CT, site B T. X holds the round-state coordinates, min-max scaled with the map's window."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    import make_pos_features as MPF  # noqa: E402  (imported here: its module prints its own site-centre table)
    win = json.load(open(DATA / f"{m}_ext.json", encoding="utf-8"))["window"]
    lo = win["xy_min"]
    span = win["xy_max"] - win["xy_min"]
    centres = MPF.site_centres("combined")[m]
    sites = [((c[0] - lo) / span, (c[1] - lo) / span) for c in centres]
    n = X.shape[0]
    out = np.empty((n, N_DIST), dtype=np.float32)
    for c in range(0, n, 500_000):
        e = min(n, c + 500_000)
        blk = np.asarray(X[c:e])
        alive = blk[:, :, 0] > 0.5
        xy = blk[:, :, 1:3].astype(np.float64)
        for k, (sx, sy) in enumerate(sites):
            d = np.sqrt((xy[:, :, 0] - sx) ** 2 + (xy[:, :, 1] - sy) ** 2) * span
            for side, sl in enumerate([slice(0, 5), slice(5, 10)]):
                dd = np.where(alive[:, sl], d[:, sl], np.inf).min(axis=1)
                dd = np.minimum(np.where(np.isfinite(dd), dd, 10000.0), 10000.0)
                out[c:e, 2 * k + side] = dd / 1000.0
    return out


def load_all(lookback, dist=False, walk=False, keep_test=True, with_prev=True):
    lens = [np.load(DATA / f"{m}_ext_loadout.npy", mmap_mode="r").shape[0] for m in MAPS]
    N = int(sum(lens))
    feat = np.empty((N, ttr.N_SLOTS, N_FEAT), dtype=np.float16)
    use_dist = dist or walk
    state = np.zeros((N, 3 + len(MAPS) + (N_DIST if use_dist else 0)), dtype=np.float32)
    label = np.zeros(N, dtype=np.float32)
    split = np.zeros(N, dtype=np.int8)
    usable = np.zeros(N, dtype=bool)
    map_id = np.zeros(N, dtype=np.int8)
    round_id = np.zeros(N, dtype=np.int16)
    second = np.zeros(N, dtype=np.int16)
    match_idx = np.zeros(N, dtype=np.int64)
    match_keys = []
    key_offset = 0
    s = 0
    for mi, m in enumerate(MAPS):
        arr, X, table, lab, st, use = ttr.build_table(m, f"{m}_ext", CACHE)
        keep = np.ones(X.shape[0], dtype=bool) if keep_test else arr["split"] != TEST
        if not keep.all():
            X, lab, st, use = X[keep], lab[keep], st[keep], use[keep]
        n = X.shape[0]
        lo = np.load(DATA / f"{m}_ext_loadout.npy", mmap_mode="r")
        if not keep.all():
            lo = lo[keep]
        for c in range(0, n, 500_000):
            e = min(n, c + 500_000)
            feat[s + c:s + e, :, :6] = X[c:e]
            feat[s + c:s + e, :, 6:] = lo[c:e]
        state[s:s + n, :3] = st
        state[s:s + n, 3 + mi] = 1.0
        if dist:
            state[s:s + n, 3 + len(MAPS):] = straight_distances(m, X)
        if walk:
            wk = np.load(DATA / f"{m}_ext_navwalk.npy", mmap_mode="r")
            if not keep.all():
                wk = wk[keep]
            if wk.shape != (n, N_DIST):
                raise SystemExit(f"{m}: navwalk shape {wk.shape} does not match the round-state rows {n}")
            state[s:s + n, 3 + len(MAPS):] = np.asarray(wk, dtype=np.float32)
        label[s:s + n] = lab
        split[s:s + n] = arr["split"][keep].astype(np.int8)
        usable[s:s + n] = use
        map_id[s:s + n] = mi
        round_id[s:s + n] = arr["round"][keep].astype(np.int16)
        second[s:s + n] = arr["second"][keep].astype(np.int16)
        match_idx[s:s + n] = arr["match_idx"][keep].astype(np.int64) + key_offset
        match_keys.append(arr["match_keys"])
        key_offset += arr["match_keys"].size
        print(f"  {m}: {n:,} seconds, {arr['match_keys'].size:,} matches, usable {int(use.sum()):,}", flush=True)
        s += n
        del X, lo, arr, table, st
    keys = np.concatenate(match_keys)
    if not keep_test:
        feat, state, label, split, usable, map_id, round_id, second, match_idx = (
            x[:s] for x in (feat, state, label, split, usable, map_id, round_id, second, match_idx))
        N = s
    # the window table is N x lookback; with_prev=False skips it for trainers that build windows on the fly
    prev = ttr.previous_rows(match_idx, round_id, second, lookback).astype(np.int32) if with_prev else None
    return {"feat": feat, "state": state, "label": label, "split": split, "usable": usable, "map_id": map_id,
            "keys": keys, "match_idx": match_idx, "round": round_id, "second": second, "prev": prev, "N": N}


def run_seed(seed, a, D, feat_t, state_t, label_t, device, kind, train_rows, val_rows, test_rows, n_train_matches):
    """One seed: train, select the best validation epoch, score the de_mirage test seconds, write the outputs."""
    tag = "allmaps_gru_loadout" + kind + ("" if seed == 0 else f"_seed{seed}") + a.suffix
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prev = D["prev"]
    print(f"== seed {seed} ({tag})", flush=True)

    def gather(rows):
        idx = prev[rows]
        mask = torch.as_tensor(idx >= 0, dtype=torch.float32, device=device)
        safe = torch.as_tensor(np.where(idx >= 0, idx, 0), dtype=torch.long, device=device)
        Xw = feat_t[safe].float() * mask[..., None, None]
        Sw = state_t[safe]
        return Xw.flip(1), Sw.flip(1), mask.flip(1)

    model = ttr.TemporalRoundModel(in_dim=N_FEAT, state_dim=state_t.shape[1], gat=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def predict(rows):
        model.eval()
        out = np.empty(rows.size, dtype=np.float32)
        for s in range(0, rows.size, 8192):
            r = rows[s:s + 8192]
            Xw, Sw, m = gather(r)
            out[s:s + 8192] = torch.sigmoid(model(Xw, Sw, m)).cpu().numpy()
        model.train()
        return out

    best = (np.inf, None, 0)
    history = []
    t_train = time.time()
    if a.no_test:
        print("no test scoring in this run (learning-curve runs read validation only)", flush=True)
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(train_rows)
        running, count = 0.0, 0
        steps = -(-order.size // a.batch)
        for k, s in enumerate(range(0, order.size, a.batch)):
            r = order[s:s + a.batch]
            Xw, Sw, m = gather(r)
            logit = model(Xw, Sw, m)
            loss = loss_fn(logit, label_t[torch.as_tensor(r, dtype=torch.long, device=device)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * r.size
            count += r.size
            if k % 1000 == 0:
                print(f"  epoch {epoch}: step {k + 1}/{steps}, train loss {running / count:.4f}, "
                      f"{time.time() - t_train:.0f} s", flush=True)
        v = ttr.log_loss(D["label"][val_rows], predict(val_rows))
        history.append({"epoch": epoch, "train_loss": running / count, "val_log_loss": v,
                        "seconds": round(time.time() - t_train, 1)})
        print(json.dumps(history[-1]), flush=True)
        if v < best[0]:
            best = (v, {k: t.detach().clone() for k, t in model.state_dict().items()}, epoch)

    model.load_state_dict(best[1])
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    test_ll = None
    if not a.no_test:
        p_test = predict(test_rows)
        test_ll = ttr.log_loss(D["label"][test_rows], p_test)
        print(f"selected epoch {best[2]}; de_mirage test log loss {test_ll:.6f}", flush=True)
        pl.DataFrame({"match_key": D["keys"][D["match_idx"][test_rows]], "round": D["round"][test_rows].astype(np.int64),
                      "second": D["second"][test_rows].astype(np.int64), "label": D["label"][test_rows].astype(np.int64),
                      "p": p_test.astype(np.float64)}).write_parquet(OUT / f"{tag}_preds.parquet")
    # validation predictions of the selected checkpoint (all maps): the decision set for the per-seed comparisons
    p_val = predict(val_rows)
    pl.DataFrame({"match_key": D["keys"][D["match_idx"][val_rows]], "round": D["round"][val_rows].astype(np.int64),
                  "second": D["second"][val_rows].astype(np.int64), "map_name": D["map_id"][val_rows].astype(np.int64),
                  "label": D["label"][val_rows].astype(np.int64), "p": p_val.astype(np.float64)}).write_parquet(OUT / f"{tag}_val_preds.parquet")
    torch.save(best[1], MODELS / f"{tag}.pt")
    metrics = {"tag": tag, "maps": MAPS, "scope": "all maps (train, validation); de_mirage test", "seed": seed,
               "lookback_s": a.lookback, "epochs": a.epochs, "selected_epoch": best[2], "history": history,
               "test_log_loss": test_ll, "test_rows": None if a.no_test else int(test_rows.size),
               "test_matches": None if a.no_test else int(np.unique(D["match_idx"][test_rows]).size), "device": str(device),
               "train_rows": int(train_rows.size), "val_rows": int(val_rows.size),
               "match_fraction": a.match_fraction, "subset_seed": a.subset_seed, "train_matches": int(n_train_matches),
               "no_test": bool(a.no_test)}
    (OUT / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("test_log_loss", "test_rows", "test_matches", "selected_epoch")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds run in one process (overrides --seed), e.g. 0,1,2")
    ap.add_argument("--dist", action="store_true", help="add the four straight-line distance features to the state")
    ap.add_argument("--walk", action="store_true", help="add the four walking (navigation-mesh) distance features to the state")
    ap.add_argument("--suffix", default="", help="appended to the output tag, so a re-run does not overwrite earlier outputs")
    ap.add_argument("--match-fraction", type=float, default=1.0,
                    help="learning curve: train on this fraction of the train matches (nested draws by match key, see --subset-seed)")
    ap.add_argument("--subset-seed", type=int, default=0, help="seed for the match-key permutation behind --match-fraction")
    ap.add_argument("--no-test", action="store_true", help="skip scoring the de_mirage test seconds (learning-curve runs read validation only)")
    a = ap.parse_args()
    if a.dist and a.walk:
        raise SystemExit("--dist and --walk are exclusive")
    if not 0.0 < a.match_fraction <= 1.0:
        raise SystemExit("--match-fraction must be in (0, 1]")
    kind = "_dist" if a.dist else ("_walk" if a.walk else "")
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; torch {torch.__version__}; seeds {seeds}", flush=True)

    t0 = time.time()
    D = load_all(a.lookback, a.dist, a.walk)
    N = D["N"]
    print(f"rows {N:,} in {time.time() - t0:.0f} s", flush=True)
    feat_t = torch.as_tensor(D["feat"], device=device)
    state_t = torch.as_tensor(D["state"], device=device)
    label_t = torch.as_tensor(D["label"], device=device)

    train_rows = np.flatnonzero((D["split"] == TRAIN) & D["usable"])
    val_rows = np.flatnonzero((D["split"] == VAL) & D["usable"])
    test_rows = np.flatnonzero((D["split"] == TEST) & D["usable"] & (D["map_id"] == 0))  # de_mirage test only
    print(f"train {train_rows.size:,}, val {val_rows.size:,} (all maps), test {test_rows.size:,} (de_mirage)", flush=True)

    # learning curve: nested draws of train matches by match key (a permutation of the sorted train keys; the first k are kept)
    train_keys = D["keys"][D["match_idx"][train_rows]]
    ukeys = np.unique(train_keys)
    n_train_matches = ukeys.size
    if a.match_fraction < 1.0:
        perm = np.random.default_rng(a.subset_seed).permutation(ukeys)
        keep = perm[:max(1, int(round(a.match_fraction * ukeys.size)))]
        train_rows = train_rows[np.isin(train_keys, keep)]
        print(f"match fraction {a.match_fraction}: {keep.size:,} of {ukeys.size:,} train matches, {train_rows.size:,} rows", flush=True)

    for seed in seeds:
        run_seed(seed, a, D, feat_t, state_t, label_t, device, kind, train_rows, val_rows, test_rows, n_train_matches)
    print(f"done {len(seeds)} seed(s) in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
