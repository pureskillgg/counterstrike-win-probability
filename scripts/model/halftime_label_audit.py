"""Halftime and label audit (descriptive; no decision). Runs on the staging pod: it needs only the combined round table and the snapshot cache.

Checks:
  1. For each round (first row per match and round, as build_table keeps it): the cache label_ct_win is constant within the round, and it equals (final == 3), the rule build_table uses for the label.
  2. The round labelling sources and the final-versus-recorded flags, split by after_swap (the halftime, and mid-overtime, side swap).
Reports counts only. Writes data/diagnostics/halftime_label_audit.json under the root given by CS2_ROOT (default /workspace/csds on the pod).
  python3 -I -X utf8 -u scripts/model/halftime_label_audit.py
"""
import json
import os

import polars as pl

ROOT = os.environ.get("CS2_ROOT", "/workspace/csds")
OUT = os.path.join(ROOT, "data", "diagnostics", "halftime_label_audit.json")


def main():
    R = (pl.scan_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet"))
         .select(["match_key", "round", "final", "label_source", "final_differs_from_recorded", "after_swap", "rec_suspect", "use"])
         .with_columns(pl.col("round").cast(pl.Int64))
         .unique(["match_key", "round"], keep="first")
         .collect())
    out = {"rounds_unique": R.height}
    out["by_after_swap"] = R.group_by("after_swap").agg(
        pl.len().alias("rounds"),
        pl.col("final").is_null().sum().alias("no_final"),
        pl.col("final_differs_from_recorded").fill_null(False).mean().alias("final_differs_share"),
        pl.col("rec_suspect").fill_null(False).mean().alias("rec_suspect_share"),
        pl.col("use").fill_null(False).mean().alias("use_share"),
    ).sort("after_swap").to_dicts()
    out["label_source_by_after_swap"] = R.group_by(["after_swap", "label_source"]).len().sort(["after_swap", "label_source"]).to_dicts()

    L = (pl.scan_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"))
         .select(["match_key", "round", "label_ct_win"])
         .with_columns(pl.col("round").cast(pl.Int64), pl.col("label_ct_win").cast(pl.Int8))
         .group_by(["match_key", "round"])
         .agg(pl.col("label_ct_win").n_unique().alias("labels"), pl.col("label_ct_win").first().alias("label"), pl.len().alias("rows"))
         .collect(engine="streaming"))
    J = L.join(R.select(["match_key", "round", "final", "after_swap"]), on=["match_key", "round"], how="left")
    J = J.with_columns(((pl.col("final") == 3).cast(pl.Int8)).alias("expected"))
    out["cache_rounds"] = J.height
    out["cache_rounds_with_one_label"] = int((J["labels"] == 1).sum())
    out["cache_rounds_without_rounds_row"] = int(J["final"].is_null().sum())
    mism = J.filter(pl.col("final").is_not_null() & (pl.col("label") != pl.col("expected")))
    out["label_vs_final_mismatch_rounds"] = mism.height
    out["label_vs_final_mismatch_by_after_swap"] = mism.group_by("after_swap").len().sort("after_swap").to_dicts()
    out["label_vs_final_mismatch_examples"] = mism.select(["match_key", "round", "final", "label", "after_swap"]).head(10).to_dicts()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ["rounds_unique", "cache_rounds", "cache_rounds_with_one_label",
                                          "cache_rounds_without_rounds_row", "label_vs_final_mismatch_rounds"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
