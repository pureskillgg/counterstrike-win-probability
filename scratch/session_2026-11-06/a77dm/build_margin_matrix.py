"""Write the A77+DM matrix: the 83 sprint matrix columns, then margin as a new last column (84 columns), as a float32 .npy memmap
under the scratch folder. The sprint matrix is only read; the original folder is not written to.

Row chunks keep RAM use small (one chunk of about 330 MB). Afterwards the file is re-opened and checked: the first 83 columns
and the margin column match the original and a recomputation on a random sample of rows, and the NaN and finite counts match
check_margin_inputs.py.

    python.exe -I -X utf8 -B build_margin_matrix.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from margin_dm import margin_from_cols  # noqa: E402

MATRIX = "G:/pureskill/csds-cs2/data/sprint/matrix_trainctr"
OUT_DIR = os.path.join(HERE, "matrix_dm")
SUMMARY = os.path.join(HERE, "out", "build_margin_matrix.json")
CHUNK = 1_000_000
SAMPLE = 200_000


def main():
    meta = json.load(open(os.path.join(MATRIX, "meta.json")))
    feats = meta["feats_all"]
    F = len(feats)
    rows = meta["rows"]
    X = np.load(os.path.join(MATRIX, "X.npy"), mmap_mode="r")
    assert X.shape == (rows, F) and X.dtype == np.float32, (X.shape, X.dtype)
    c = {k: feats.index(k) for k in ["seconds_since_plant", "bomb_planted", "c_min_bomb_dist", "st_kit_n_ct"]}
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "X.npy")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; remove it before rebuilding")

    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(rows, F + 1))
    sentinel_mismatch = 0
    for i in range(0, rows, CHUNK):
        j = min(i + CHUNK, rows)
        blk = np.asarray(X[i:j])
        out[i:j, :F] = blk
        m = margin_from_cols(blk[:, c["seconds_since_plant"]], blk[:, c["c_min_bomb_dist"]], blk[:, c["st_kit_n_ct"]])
        out[i:j, F] = m
        sp, bp = blk[:, c["seconds_since_plant"]], blk[:, c["bomb_planted"]]
        sentinel_mismatch += int(((sp < 0) != (bp < 0.5)).sum())
    out.flush()
    del out
    print(f"wrote {out_path}: {rows:,} x {F + 1}; rows where pre-plant sentinel and bomb_planted disagree: {sentinel_mismatch}",
          flush=True)

    # re-open and verify
    A = np.load(out_path, mmap_mode="r")
    assert A.shape == (rows, F + 1) and A.dtype == np.float32, (A.shape, A.dtype)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(rows, size=SAMPLE, replace=False))
    Xs, As = np.asarray(X[idx]), np.asarray(A[idx])
    base_equal = bool(np.array_equal(As[:, :F], Xs, equal_nan=True))
    recomputed = margin_from_cols(Xs[:, c["seconds_since_plant"]], Xs[:, c["c_min_bomb_dist"]], Xs[:, c["st_kit_n_ct"]])
    margin_equal = bool(np.array_equal(As[:, F], recomputed, equal_nan=True))
    nan_total, fin_total, mn, mx = 0, 0, np.inf, -np.inf
    for i in range(0, rows, CHUNK):
        j = min(i + CHUNK, rows)
        col = np.asarray(A[i:j, F])
        nn = np.isnan(col)
        nan_total += int(nn.sum())
        fin = col[~nn]
        fin_total += int(fin.size)
        if fin.size:
            mn, mx = min(mn, float(fin.min())), max(mx, float(fin.max()))
    summary = {"file": out_path, "shape": list(A.shape), "dtype": str(A.dtype), "bytes": os.path.getsize(out_path),
               "columns_0_82_equal_source_on_sample": base_equal, "margin_equal_recomputed_on_sample": margin_equal,
               "sample_rows": SAMPLE, "margin_nan_rows": nan_total, "margin_finite_rows": fin_total,
               "margin_finite_min": mn, "margin_finite_max": mx, "sentinel_vs_bomb_planted_mismatch": sentinel_mismatch}
    assert base_equal and margin_equal and sentinel_mismatch == 0, summary

    meta_dm = dict(meta)
    meta_dm["feats_all"] = feats + ["margin"]
    meta_dm["margin_definition"] = ("(40 - seconds_since_plant) - (c_min_bomb_dist / 250 + defuse_seconds); defuse_seconds = 5 if "
                                    "st_kit_n_ct >= 1 else 10; NaN where seconds_since_plant < 0 (pre-plant sentinel)")
    meta_dm["source_matrix"] = MATRIX
    json.dump(meta_dm, open(os.path.join(OUT_DIR, "meta.json"), "w"), indent=2)
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
