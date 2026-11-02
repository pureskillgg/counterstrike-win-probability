"""Per-group GDT differences against the reference model, on the grader's own datapoints (same draws, same seed).

A group is a set of situations taken together: the time left and whether the bomb is planted, how many players are alive on each
side, or which side has the better kit. Each difference is the reference's log loss minus the model's on the group's datapoints,
so a positive number means the model is better. Intervals resample test matches, as the grader does.

    python analysis/group_gdt.py --rows <rows.parquet> --models p_v6,p_xgb83,... --out docs/gdt_results --tag sep01_oct08
"""
import argparse
import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
GRADER = os.path.join(HERE, "..", "evaluation", "intern_gdt")
sys.path.insert(0, GRADER)
import score_gdt as S  # noqa: E402

KIT = {"low": 0, "mid": 1, "high": 2}
FAMILIES = [
    # (family name, group column, filter)
    ("overall", None, None),
    ("phase and time left", "time_group", None),
    ("players alive", "alive_group", None),
    ("kit level", "kit_group", None),
    ("pre-plant: players alive", "alive_group", "pre"),
    ("pre-plant: kit level", "kit_group", "pre"),
    ("pre-plant, lot: kit level", "kit_group", "pre_lot"),
]


def build_points(rows, models, table_path, seed):
    df = S.read_rows(rows).select(S.NEED + models)
    df = df.with_columns(pl.when(pl.col("planted") == 0).then(pl.col("since_plant").fill_null(0))
                         .otherwise(pl.col("since_plant")).alias("since_plant")).drop_nulls()
    df = S.situations(df)
    table = pl.read_csv(table_path, schema_overrides={"tb": pl.Utf8, "econ_ct": pl.Utf8, "econ_t": pl.Utf8})
    table = table.select(S.CELL + [pl.col("p_ct").alias("p_table")])
    return S.draw(df, models, seed).join(table, on=S.CELL, how="left").filter(pl.col("p_table").is_not_null())


def add_groups(pts):
    gap = pl.col("alive_ct") - pl.col("alive_t")
    kit_gap = pl.col("econ_ct").replace_strict(KIT, return_dtype=pl.Int64) - pl.col("econ_t").replace_strict(KIT, return_dtype=pl.Int64)
    return pts.with_columns(
        pl.when(pl.col("planted") == 1).then(pl.lit("post-plant")).otherwise(pl.lit("pre-plant")).alias("phase"),
        pl.concat_str([pl.when(pl.col("planted") == 1).then(pl.lit("post-plant")).otherwise(pl.lit("pre-plant")),
                       pl.lit(" "), pl.col("tb"), pl.lit(" left")]).alias("time_group"),
        pl.when(gap == 0).then(pl.lit("even"))
        .when(gap == 1).then(pl.lit("CT +1"))
        .when(gap == -1).then(pl.lit("T +1"))
        .when(gap >= 2).then(pl.lit("CT +2 or more"))
        .otherwise(pl.lit("T +2 or more")).alias("alive_group"),
        pl.when(kit_gap > 0).then(pl.lit("CT better kit"))
        .when(kit_gap == 0).then(pl.lit("same kit level"))
        .otherwise(pl.lit("T better kit")).alias("kit_group"),
    )


def family_mask(pts, filt):
    if filt == "pre":
        return (pts["planted"] == 0).to_numpy()
    if filt == "pre_lot":
        return ((pts["planted"] == 0) & (pts["tb"] == "lot")).to_numpy()
    return np.ones(pts.height, dtype=bool)


def group_rows(pts, models, ref, reps, seed):
    y = pts["y"].to_numpy()
    mids = pts["match"].to_numpy()
    ids, inv = np.unique(mids, return_inverse=True)
    m = len(ids)
    lref = S.logloss(pts[ref].to_numpy(), y)
    diffs = {mod: lref - S.logloss(pts[mod].to_numpy(), y) for mod in models[1:]}
    out = []
    for fam, col, filt in FAMILIES:
        fmask = family_mask(pts, filt)
        labels = np.array(["all"] * pts.height, dtype=object) if col is None else pts[col].to_numpy()
        for g in sorted(set(labels[fmask])):
            mask = fmask & (labels == g)
            idx = np.where(mask)[0]
            cnt = np.bincount(inv[idx], minlength=m).astype("float64")
            rng = np.random.default_rng(seed)
            picks = rng.integers(0, m, (reps, m))
            n_matches = int((cnt > 0).sum())
            for mod in models[1:]:
                sums = np.bincount(inv[idx], weights=diffs[mod][idx], minlength=m)
                est = sums.sum() / cnt.sum()
                boot = sums[picks].sum(axis=1) / np.maximum(cnt[picks].sum(axis=1), 1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                out.append({"family": fam, "group": g, "datapoints": int(idx.size), "matches": n_matches, "model": mod,
                            "diff": float(est), "ci_low": float(lo), "ci_high": float(hi)})
    return out


def main():
    ap = argparse.ArgumentParser(description="Per-group GDT differences against the reference model")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--models", default="p_v6,p_v4,p_xgb83,p_gnn_hyb1,p_gnn_set,p_gru,p_lag")
    ap.add_argument("--table", default=os.path.join(GRADER, "table.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "gdt_results"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    models = [x.strip() for x in a.models.split(",") if x.strip()]
    pts = add_groups(build_points(a.rows, models, a.table, a.seed))
    rows = group_rows(pts, models, models[0], a.reps, a.seed)
    os.makedirs(a.out, exist_ok=True)
    res = pl.DataFrame(rows)
    path = os.path.join(a.out, f"group_gdt_{a.tag}.csv")
    res.write_csv(path)
    print(f"reference {models[0]}; positive = model better than the reference; written {path}")
    for fam, col, filt in FAMILIES:
        sub = res.filter(pl.col("family") == fam)
        if sub.height == 0:
            continue
        print(f"\n[{fam}]")
        for g in sub["group"].unique(maintain_order=True).to_list():
            s = sub.filter(pl.col("group") == g)
            head = f"  {g:<28} datapoints {s['datapoints'][0]:>7,}  matches {s['matches'][0]:>4}"
            cells = "  ".join(f"{r['model']} {r['diff']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]" for r in s.iter_rows(named=True))
            print(head + "  " + cells)


if __name__ == "__main__":
    main()
