"""Per-second distance features for every map's round-state rows, keyed by (match_key, round, second).

For each map: the straight-line distances (train_temporal_allmaps.straight_distances) and the walking distances
(<map>_ext_navwalk.npy, from build_nav_walk_allmaps.py), both in the same column layout the GRU runs use. Written per map to
data/diagnostics/allmaps_geo/<map>.parquet, so allmaps_xgb_geo.py can join them onto the snapshot cache.

Run in the torch environment (the round-state reader imports torch):
  G:/pureskill/csds-cs2/.pixi/envs/torch/python.exe scripts/model/allmaps_geo_rows.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import train_temporal_round_model as ttr  # noqa: E402
import train_temporal_allmaps as T  # noqa: E402

OUT = os.path.join(ttr.ROOT, "data", "diagnostics", "allmaps_geo")


def main():
    os.makedirs(OUT, exist_ok=True)
    for m in T.MAPS:
        arr, X, table, lab, st, use = ttr.build_table(m, f"{m}_ext", T.CACHE)
        keys = arr["match_keys"][arr["match_idx"]]
        st4 = T.straight_distances(m, X)
        wk = np.load(T.DATA / f"{m}_ext_navwalk.npy", mmap_mode="r")
        if wk.shape != (X.shape[0], T.N_DIST):
            raise SystemExit(f"{m}: navwalk shape {wk.shape} does not match the round-state rows {X.shape[0]}")
        wk = np.asarray(wk, dtype=np.float32)
        df = pl.DataFrame({"match_key": keys, "round": arr["round"].astype(np.int64), "second": arr["second"].astype(np.int64)})
        df = df.with_columns([pl.Series(f"st{i}", st4[:, i]) for i in range(4)] + [pl.Series(f"wk{i}", wk[:, i]) for i in range(4)])
        df.write_parquet(os.path.join(OUT, f"{m}.parquet"))
        print(f"{m}: {df.height:,} rows written", flush=True)
        del X, arr, table, st, st4, wk, df
    return 0


if __name__ == "__main__":
    sys.exit(main())
