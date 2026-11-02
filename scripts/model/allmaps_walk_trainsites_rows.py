"""Walking-distance features (train-only site centres) keyed by (match_key, round, second), for W2 (pin f5eec3a3).

For each map: the rows of <map>_ext.npz (via train_temporal_round_model.build_table, the same row order the walking arrays use) joined with
<map>_ext_navwalk_trainsites.npy. Writes data/diagnostics/allmaps_walk_trainsites/<map>.parquet (match_key, round, second, wk0..wk3).
Run in the torch environment (the round-state reader imports torch):
  G:/pureskill/csds-cs2/.pixi/envs/torch/python.exe -I -X utf8 -u scripts/model/allmaps_walk_trainsites_rows.py
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_temporal_round_model as ttr  # noqa: E402
import train_temporal_allmaps as T  # noqa: E402

OUT = os.path.join(ttr.ROOT, "data", "diagnostics", "allmaps_walk_trainsites")


def source_paths(m):
    return os.path.join(str(T.DATA), f"{m}_ext.npz"), os.path.join(str(T.DATA), f"{m}_ext_navwalk_trainsites.npy")


def main():
    # every input is checked before anything is written, so a missing or stale array never leaves a partial output folder
    missing = [p for m in T.MAPS for p in source_paths(m) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing inputs, nothing written: {missing}")
    stale = [m for m in T.MAPS if os.path.getmtime(source_paths(m)[1]) < os.path.getmtime(source_paths(m)[0])]
    if stale:
        raise SystemExit(f"walking arrays are older than their round-state npz; rebuild them first: {stale}")
    os.makedirs(OUT, exist_ok=True)
    manifest, pending = {}, []
    for m in T.MAPS:
        npz, npy = source_paths(m)
        arr, X, table, lab, st, use = ttr.build_table(m, f"{m}_ext", T.CACHE)
        keys = arr["match_keys"][arr["match_idx"]]
        wk = np.load(npy, mmap_mode="r")
        if wk.shape != (X.shape[0], T.N_DIST):
            raise SystemExit(f"{m}: navwalk shape {wk.shape} does not match the round-state rows {X.shape[0]}")
        wk = np.asarray(wk, dtype=np.float32)
        df = pl.DataFrame({"match_key": keys, "round": arr["round"].astype(np.int64), "second": arr["second"].astype(np.int64)})
        df = df.with_columns([pl.Series(f"wk{i}", wk[:, i]) for i in range(T.N_DIST)])
        final = os.path.join(OUT, f"{m}.parquet")
        df.write_parquet(final + ".tmp")
        pending.append((final + ".tmp", final))
        manifest[m] = {"rows": df.height, "npz_mtime": os.path.getmtime(npz), "npy_mtime": os.path.getmtime(npy)}
        print(f"{m}: {df.height:,} rows written to a temp file", flush=True)
        del X, arr, table, st, wk, df
    for tmp, final in pending:
        os.replace(tmp, final)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("all seven maps written; manifest.json records the source mtimes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
