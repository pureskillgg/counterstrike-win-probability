"""Why does XGBoost look better than the temporal models? Like-for-like on the frozen de_mirage test rows.

Every model is scored on the SAME rows (the usable de_mirage test seconds of the frozen 687 matches). Each comparison is a pair
on that common row set, with a match-cluster bootstrap (1000 resamples, seed 0). diff = mean(a) - mean(b); a is better when < 0.

The table also records what each model was allowed to use, because the comparisons differ in more than the architecture:
  scope       which maps the TRAINING rows come from (de_mirage only, or all maps)
  inputs      what each model reads
  budget      how many training passes or steps it had, and how its checkpoint was chosen

Predictions come from data/diagnostics/ext_baseline_{de_mirage,all}_preds.parquet (XGBoost, both arms) and from the temporal
prediction files, which are read through ext_temporal_paired.test_rows so the row set is rebuilt with the trainer's own mask.

  ...torch env python.exe -I -X utf8 -u scripts\\model\\investigate_xgb_vs_temporal.py
Writes data/diagnostics/xgb_vs_temporal_investigation.json.
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402
import ext_temporal_paired as etp  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "xgb_vs_temporal_investigation.json")
KEY = ["match_key", "round", "second"]

# name -> (kind, source, scope, inputs, budget)
MODELS = {
    "xgb_all_O": ("xgb", ("all", "O"), "all maps (train)", "57 snapshot features",
                  "XGBoost, early stopping on all-maps validation"),
    "xgb_all_E": ("xgb", ("all", "E"), "all maps (train) + 2026-10 additions", "57 snapshot features",
                  "XGBoost, early stopping on all-maps validation"),
    "xgb_de_O": ("xgb", ("de_mirage", "O"), "de_mirage only", "57 snapshot features",
                 "XGBoost, early stopping on de_mirage validation"),
    "xgb_de_E": ("xgb", ("de_mirage", "E"), "de_mirage only + additions", "57 snapshot features",
                 "XGBoost, early stopping on de_mirage validation"),
    "gru_O": ("temporal", ("de_mirage_loadout_gru", "de_mirage", "cache_snapshot_2026.parquet"), "de_mirage only",
              "per-second state + loadout (30 channels/slot)", "GRU, 3 epochs, best-validation epoch (epoch 1)"),
    "gru_E": ("temporal", ("de_mirage_ext_loadout_gru", "de_mirage_ext", "cache_snapshot_2026_ext.parquet"),
              "de_mirage only + additions", "per-second state + loadout (30 channels/slot)",
              "GRU, 3 epochs, best-validation epoch (epoch 3)"),
    "gru_O_ms8544": ("temporal", ("de_mirage_loadout_gru_ms8544", "de_mirage", "cache_snapshot_2026.parquet"), "de_mirage only",
                     "per-second state + loadout", "GRU, 8,544 steps, best-validation check every 500 steps"),
    "gru_E_ms8544": ("temporal", ("de_mirage_ext_loadout_gru_ms8544", "de_mirage_ext", "cache_snapshot_2026_ext.parquet"),
                     "de_mirage only + additions", "per-second state + loadout",
                     "GRU, 8,544 steps, best-validation check every 500 steps"),
    "thgnn_O": ("temporal", ("de_mirage_thgnn_lite_rerun", "de_mirage", "cache_snapshot_2026.parquet"), "de_mirage only",
                "per-second state + loadout + map graph (48 nodes)", "THGNN-lite, 3 epochs, best-validation epoch (epoch 2)"),
    "thgnn_E": ("temporal", ("de_mirage_ext_thgnn_lite_rerun", "de_mirage_ext", "cache_snapshot_2026_ext.parquet"),
                "de_mirage only + additions", "per-second state + loadout + map graph (48 nodes)",
                "THGNN-lite, 3 epochs, best-validation epoch (epoch 1)"),
    "thgnn_O_ms8544": ("temporal", ("de_mirage_thgnn_lite_ms8544", "de_mirage", "cache_snapshot_2026.parquet"), "de_mirage only",
                       "per-second state + loadout + map graph", "THGNN-lite, 8,544 steps, best-validation check every 500 steps"),
    "thgnn_E_ms8544": ("temporal", ("de_mirage_ext_thgnn_lite_ms8544", "de_mirage_ext", "cache_snapshot_2026_ext.parquet"),
                       "de_mirage only + additions", "per-second state + loadout + map graph",
                       "THGNN-lite, 8,544 steps, best-validation check every 500 steps"),
}

# (a, b, question)
PAIRS = [
    ("xgb_all_O", "gru_O", "reference XGBoost vs GRU, original data (scopes differ: all maps vs de_mirage)"),
    ("xgb_all_O", "thgnn_O", "reference XGBoost vs THGNN, original data (scopes differ)"),
    ("xgb_de_O", "gru_O", "same scope: de_mirage-only XGBoost vs GRU, original data"),
    ("xgb_de_O", "thgnn_O", "same scope: de_mirage-only XGBoost vs THGNN, original data"),
    ("xgb_all_O", "xgb_de_O", "what the other maps add to XGBoost, original data"),
    ("xgb_all_E", "gru_E", "reference XGBoost vs GRU, extended data (scopes differ)"),
    ("xgb_all_E", "thgnn_E", "reference XGBoost vs THGNN, extended data (scopes differ)"),
    ("xgb_de_E", "gru_E", "same scope: de_mirage-only XGBoost vs GRU, extended data"),
    ("xgb_de_E", "thgnn_E", "same scope: de_mirage-only XGBoost vs THGNN, extended data"),
    ("gru_O_ms8544", "gru_E_ms8544", "GRU, matched steps: extended minus original"),
    ("thgnn_O_ms8544", "thgnn_E_ms8544", "THGNN, matched steps: extended minus original"),
    ("gru_O", "gru_E", "GRU, checkpoint comparison: extended minus original"),
    ("thgnn_O", "thgnn_E", "THGNN, checkpoint comparison: extended minus original"),
]


def load(name):
    kind, src, *_ = MODELS[name]
    if kind == "xgb":
        scope, arm = src
        df = pl.read_parquet(os.path.join(DIAG, f"ext_baseline_{scope}_preds.parquet"))
        return df.select(KEY + ["label", pl.col(f"p_{arm}").cast(pl.Float64).alias("p")])
    pred_stem, round_stem, cache = src
    return etp.test_rows(round_stem, cache, pred_stem).select(KEY + ["label", "p"])


def main():
    frames = {name: load(name) for name in MODELS}
    common = None
    for name, df in frames.items():
        keys = df.select(KEY)
        common = keys if common is None else common.join(keys, on=KEY, how="inner")
    rows = {}
    for name, df in frames.items():
        rows[name] = df.height
    print("rows per model:", rows, flush=True)
    print("rows on the common key set:", common.height, flush=True)

    aligned = {}
    for name, df in frames.items():
        a = common.join(df, on=KEY, how="left").sort(KEY)
        aligned[name] = a
    y = aligned["xgb_all_O"]["label"].to_numpy().astype(np.int64)
    for name, a in aligned.items():
        if not (a["label"].to_numpy().astype(np.int64) == y).all():
            raise SystemExit(f"labels disagree for {name}")
    match = aligned["xgb_all_O"]["match_key"].to_numpy()

    table = {}
    for name, a in aligned.items():
        ll = W.ll_vec(a["p"].to_numpy(), y)
        table[name] = {"test_log_loss": float(ll.mean()), "scope": MODELS[name][2], "inputs": MODELS[name][3],
                       "budget": MODELS[name][4], "rows": int(a.height), "matches": int(np.unique(match).size)}
    comparisons = []
    for a_name, b_name, question in PAIRS:
        r = etp.paired(match, y, aligned[a_name]["p"].to_numpy(), aligned[b_name]["p"].to_numpy())
        comparisons.append({"a": a_name, "b": b_name, "question": question, **r,
                            "a_better": bool(r["diff"] < 0 and r["ci95"][1] < 0),
                            "b_better": bool(r["diff"] > 0 and r["ci95"][0] > 0)})
    res = {"common_rows": int(common.height), "models": table, "comparisons": comparisons}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)

    print(f"\n{'model':<16}{'test log loss':>15}  scope", flush=True)
    for name, t in table.items():
        print(f"{name:<16}{t['test_log_loss']:>15.6f}  {t['scope']}", flush=True)
    print("\ncomparisons (diff = a - b; negative means a is better):", flush=True)
    for c in comparisons:
        verdict = "a better" if c["a_better"] else ("b better" if c["b_better"] else "no clear difference")
        print(f"  {c['a']} vs {c['b']}: {c['diff']:+.6f} [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] -> {verdict}", flush=True)
        print(f"      {c['question']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
