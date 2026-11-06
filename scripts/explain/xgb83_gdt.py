"""GDT for the 83-input model and its drop-group ablations, using the team's scorer (evaluation/intern_gdt/score_gdt.py).

Population, from evaluation/intern_gdt/README.md and docs/final/TRAINING_METRIC.md:
  - matches dated 2026-09-01 to 2026-10-08 (the README says not to score Oct 9 to 13, the days the table was built on)
  - rounds that start 5 against 5 (5 alive on each side at the first second of the live round)
Rows: one per second of those rounds, with the team's field names (alive_ct, alive_t, equip_ct, equip_t, planted,
since_plant, second, y) and one probability column per model. The scorer pairs the models on the same datapoints.

Run after the predictions exist (xgb83_drop_ablation.py writes preds/<set>_s<seed>.npy over the validation rows, in order):
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe scripts\\explain\\xgb83_gdt.py --sets full --seeds 0 1 2
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgb83_explain import MDIR  # noqa: E402

CACHE = "G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026.parquet"
PRED_DIR = "C:/Users/13144/sprint/explain83_ablation/preds"
WORK = "C:/Users/13144/sprint/explain83_ablation/gdt"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCORER = os.path.join(REPO, "evaluation", "intern_gdt", "score_gdt.py")
PY = "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe"
FIRST, LAST = "2026-09-01", "2026-10-08"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    meta = json.load(open(os.path.join(MDIR, "meta.json")))
    feats = meta["feats_all"]
    a2 = meta["n_fit"] + meta["n_es"]
    X = np.load(os.path.join(MDIR, "X.npy"), mmap_mode="r")
    src = {"alive_ct": "alive_ct", "alive_t": "alive_t", "equip_ct": "st_equip_ct", "equip_t": "st_equip_t",
           "planted": "bomb_planted", "since_plant": "seconds_since_plant", "sec": "seconds_since_round_live"}
    cols = {k: np.asarray(X[a2:, feats.index(v)], dtype=np.float64) for k, v in src.items()}

    keys = pl.read_parquet(os.path.join(MDIR, "val_keys.parquet"))
    n = keys.height
    assert n == meta["n_val"], "validation rows differ from meta.json"
    dates = (pl.scan_parquet(CACHE).select(["match_key", "match_date"]).unique(subset=["match_key"]).collect()
             .with_columns(pl.col("match_date").cast(pl.Utf8).str.slice(0, 10).alias("day")).select(["match_key", "day"]))
    base = (keys.with_row_index("i").join(dates, on="match_key", how="left")
            .filter((pl.col("day") >= FIRST) & (pl.col("day") <= LAST)))
    bi = base["i"].to_numpy()  # validation-row positions of the window's seconds
    base = base.with_columns([pl.Series(k, cols[k][bi]) for k in src if k != "sec"]
                             + [pl.Series("sec", np.round(cols["sec"][bi]))])

    # 5 against 5 at the first second of the live round
    first = base.filter(pl.col("sec") == pl.col("sec").min().over(["match_key", "round"]))
    ok = first.filter((pl.col("alive_ct") == 5) & (pl.col("alive_t") == 5)).select(["match_key", "round"]).unique()
    pop = base.join(ok, on=["match_key", "round"], how="semi").sort(["match_key", "round", "sec"])
    print(f"population: {pop['match_key'].n_unique()} matches, {pop.select(['match_key', 'round']).unique().height} rounds, "
          f"{pop.height} seconds ({FIRST} to {LAST}, 5 against 5 starts)", flush=True)

    labels = np.asarray(keys["label_ct_win"].to_numpy(), dtype=np.float64)
    idx = pop["i"].to_numpy()
    out = pl.DataFrame({"match": pop["match_key"].to_list(), "round": pop["round"].to_numpy(),
                        "second": pop["sec"].to_numpy(), "alive_ct": pop["alive_ct"].to_numpy(),
                        "alive_t": pop["alive_t"].to_numpy(), "equip_ct": pop["equip_ct"].to_numpy(),
                        "equip_t": pop["equip_t"].to_numpy(), "planted": pop["planted"].to_numpy(),
                        "since_plant": pop["since_plant"].to_numpy(), "y": labels[idx]})
    names = []
    for s in a.sets:
        for seed in a.seeds:
            name = f"{s}_s{seed}"
            p = np.load(os.path.join(PRED_DIR, f"{name}.npy")).astype(np.float64)
            assert p.shape[0] == n, f"{name} has {p.shape[0]} predictions, expected {n}"
            out = out.with_columns(pl.Series(name, p[idx]))
            names.append(name)
    rows = os.path.join(WORK, "rows_population.parquet")
    out.write_parquet(rows)
    print(f"wrote {rows}: {out.height} rows, models {names}", flush=True)

    for seed in a.seeds:
        models = [f"{s}_s{seed}" for s in a.sets]
        outdir = os.path.join(WORK, f"score_seed{seed}")
        cmd = [PY, SCORER, "--rows", rows, "--models", ",".join(models), "--out", outdir]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        print(r.stdout[-1500:], flush=True)
        if r.returncode:
            print(r.stderr[-1500:], flush=True)
            raise SystemExit(f"scorer failed for seed {seed}")
    print("scorer outputs in", WORK, flush=True)


if __name__ == "__main__":
    main()
