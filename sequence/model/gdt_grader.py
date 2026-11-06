"""Score round win-probability models by GDT, the gain over the dumb table.

Reads your per-second rows (README.md, "The rows"), puts every second in its situation, draws one datapoint per round per
situation (seeded), looks the table's P(CT) up in table.csv, and scores every model column you name against it on the same
datapoints. With two or more models, also the difference to the first one, paired on the same datapoints.

    pip install polars numpy        (or: uv run --with polars --with numpy python score_gdt.py ...)
    python score_gdt.py --rows rows/ --models p_v6,p_new --out results

--rows is a .parquet or .csv file, a folder of them, or a glob. Writes <out>/summary.json and <out>/by_situation.csv.
"""
import argparse
import glob
import json
import os

import numpy as np
import polars as pl

CELL = ["alive_ct", "alive_t", "planted", "tb", "econ_ct", "econ_t"]
ROUND_S, BOMB_S = 115.0, 40.0                        # time left in thirds: of the 115 s round before the plant, of the 40 s bomb after it
CUTS = {"ct": (2837.5, 4200.0), "t": (2420.0, 3900.0)}  # equipment per alive player: low below the first cut, high from the second
EPS = 1e-6
SIT_MIN = 500                                         # GDT_sit: every situation with this many test datapoints or more counts once
NEED = ["match", "round", "second", "alive_ct", "alive_t", "equip_ct", "equip_t", "planted", "since_plant", "y"]


def read_rows(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")) + glob.glob(os.path.join(path, "*.csv")))
    elif any(ch in path for ch in "*?["):
        files = sorted(glob.glob(path))
    else:
        files = [path]
    if not files:
        raise SystemExit(f"no .parquet or .csv files at {path}")
    frames = [pl.read_parquet(f) if f.endswith(".parquet") else pl.read_csv(f) for f in files]
    return pl.concat(frames, how="diagonal_relaxed")


def situations(df):
    """The situation of every second: alive counts, the bomb, the time third and each side's economy level."""
    df = df.with_columns(pl.col("match").cast(pl.Utf8), pl.col("round").cast(pl.Int64), pl.col("alive_ct").cast(pl.Int64),
                         pl.col("alive_t").cast(pl.Int64), pl.col("planted").cast(pl.Int64), pl.col("y").cast(pl.Int64),
                         pl.col("second").cast(pl.Float64).round(0), pl.col("since_plant").cast(pl.Float64))

    def third(col, length):
        return (pl.when(pl.col(col) < length / 3).then(pl.lit("lot"))
                .when(pl.col(col) < 2 * length / 3).then(pl.lit("medium")).otherwise(pl.lit("little")))

    df = df.with_columns(pl.when(pl.col("planted") == 1).then(third("since_plant", BOMB_S)).otherwise(third("second", ROUND_S)).alias("tb"))
    for s in ("ct", "t"):
        lo, hi = CUTS[s]
        ev = pl.when(pl.col(f"alive_{s}") > 0).then(pl.col(f"equip_{s}").cast(pl.Float64) / pl.col(f"alive_{s}")).otherwise(pl.lit(0.0))
        df = df.with_columns(ev.alias(f"ev_{s}"))
        df = df.with_columns(pl.when(pl.col(f"ev_{s}") < lo).then(pl.lit("low")).when(pl.col(f"ev_{s}") < hi).then(pl.lit("mid"))
                             .otherwise(pl.lit("high")).alias(f"econ_{s}"))
    return df


def draw(df, models, seed):
    """One datapoint per round per situation: of the seconds a round spent in a situation, one drawn at random. The rows are sorted
    first, so the same rows give the same datapoints whatever order they came in, and every model is scored on the same ones."""
    df = df.sort(["match", "round", "second", "since_plant"])
    rng = np.random.default_rng(seed)
    df = df.with_columns(pl.Series("r", rng.random(df.height)))
    keep = ["y", "second", "since_plant"] + models
    return (df.sort("r").group_by(["match", "round"] + CELL, maintain_order=True)
            .agg([pl.col(c).first() for c in keep] + [pl.len().alias("seconds_in_situation")]))


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    y = np.asarray(y, dtype="float64")
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def wboot(match_ids, la, lb, w, reps, seed):
    """Weighted mean of la, of lb and of la - lb, with 95% intervals from resampling matches (the same resamples for every call)."""
    ids, inv = np.unique(match_ids, return_inverse=True)
    sw, sa, sb = np.bincount(inv, weights=w), np.bincount(inv, weights=w * la), np.bincount(inv, weights=w * lb)
    m = len(ids)
    rng = np.random.default_rng(seed)
    reps_d = []
    for _ in range(reps):
        pick = rng.integers(0, m, m)
        reps_d.append((sa[pick].sum() - sb[pick].sum()) / sw[pick].sum())
    lo, hi = np.percentile(reps_d, [2.5, 97.5])
    return {"a": float(sa.sum() / sw.sum()), "b": float(sb.sum() / sw.sum()), "diff": float((sa.sum() - sb.sum()) / sw.sum()),
            "ci": [float(lo), float(hi)]}


def score(pts, models, reps, seed):
    y = pts["y"].to_numpy()
    mids = pts["match"].to_numpy()
    lt = logloss(pts["p_table"].to_numpy(), y)
    n_s = pts["n_s"].to_numpy().astype("float64")
    weights = {"rounds": np.ones(len(y)), "situations": np.where(n_s >= SIT_MIN, 1.0 / n_s, 0.0)}
    losses = {m: logloss(pts[m].to_numpy(), y) for m in models}
    has_sit = weights["situations"].sum() > 0  # a small sample may have no situation with SIT_MIN datapoints

    def sit_boot(la, lb):
        return wboot(mids, la, lb, weights["situations"], reps, seed) if has_sit else {"diff": None, "ci": None}

    out = {"models": {}, "vs_first": {}}
    for m in models:
        g = wboot(mids, lt, losses[m], weights["rounds"], reps, seed)
        gs = sit_boot(lt, losses[m])
        ids, inv = np.unique(mids, return_inverse=True)
        per_match = np.bincount(inv, weights=lt - losses[m]) / np.bincount(inv)
        out["models"][m] = {"table_logloss": g["a"], "model_logloss": g["b"], "GDT": g["diff"], "GDT_ci": g["ci"], "rGDT": 1 - g["b"] / g["a"],
                            "GDT_sit": gs["diff"], "GDT_sit_ci": gs["ci"], "share_of_matches_model_better": float((per_match > 0).mean())}
    ref = models[0]
    for m in models[1:]:
        d = wboot(mids, losses[ref], losses[m], weights["rounds"], reps, seed)
        ds = sit_boot(losses[ref], losses[m])
        out["vs_first"][m] = {"against": ref, "GDT_difference": d["diff"], "GDT_difference_ci": d["ci"],
                              "GDT_sit_difference": ds["diff"], "GDT_sit_difference_ci": ds["ci"]}
    return out, lt, losses


def fmt(v, ci, sign=""):
    return "none (no situation has enough datapoints)" if v is None else f"{v:{sign}.4f} ({ci[0]:{sign}.4f} to {ci[1]:{sign}.4f})"


def main():
    ap = argparse.ArgumentParser(description="Score round win-probability models by GDT, the gain over the dumb table.")
    ap.add_argument("--rows", required=True, help="a .parquet or .csv file, a folder of them, or a glob")
    ap.add_argument("--models", default="p_ct", help="comma-separated columns, each a model's P(CT wins the round); the first is the reference")
    ap.add_argument("--table", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "table.csv"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    df = read_rows(a.rows)
    missing = [c for c in NEED + models if c not in df.columns]
    if missing:
        raise SystemExit(f"the rows have no column {', '.join(missing)} (README.md, 'The rows')")
    n0 = df.height
    df = df.select(NEED + models)
    df = df.with_columns(pl.when(pl.col("planted") == 0).then(pl.col("since_plant").fill_null(0)).otherwise(pl.col("since_plant")).alias("since_plant"))
    df = df.drop_nulls()
    if df.height < n0:
        print(f"dropped {n0 - df.height:,} rows with an empty field")
    df = situations(df)

    table = pl.read_csv(a.table, schema_overrides={"tb": pl.Utf8, "econ_ct": pl.Utf8, "econ_t": pl.Utf8})
    table = table.select(CELL + [pl.col("p_ct").alias("p_table")])
    pts = draw(df, models, a.seed).join(table, on=CELL, how="left")
    lost = pts.filter(pl.col("p_table").is_null())
    if lost.height:
        print(f"dropped {lost.height:,} datapoints in a situation the table doesn't have (alive 0 against 0?)")
        pts = pts.filter(pl.col("p_table").is_not_null())
    pts = pts.join(pts.group_by(CELL).agg(pl.len().alias("n_s")), on=CELL, how="left")

    res, lt, losses = score(pts, models, a.reps, a.seed)
    res["counts"] = {"rows": df.height, "matches": int(pts["match"].n_unique()), "rounds": int(pts.select("match", "round").n_unique()),
                     "datapoints": pts.height, "situations": int(pts.select(CELL).n_unique()),
                     f"situations_with_{SIT_MIN}_or_more": int(pts.filter(pl.col("n_s") >= SIT_MIN).select(CELL).n_unique())}
    res["settings"] = {"seed": a.seed, "reps": a.reps, "table": os.path.abspath(a.table)}

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    sit = pts.with_columns([pl.Series("ll_table", lt)] + [pl.Series(f"ll_{m}", losses[m]) for m in models])
    aggs = [pl.len().alias("datapoints"), pl.col("match").n_unique().alias("matches"), pl.col("y").mean().alias("ct_win_share"),
            pl.col("p_table").first(), pl.col("ll_table").mean()]
    for m in models:
        aggs += [pl.col(m).mean().alias(f"mean_{m}"), pl.col(f"ll_{m}").mean(), (pl.col("ll_table") - pl.col(f"ll_{m}")).mean().alias(f"gain_{m}")]
    sit.group_by(CELL).agg(aggs).sort("datapoints", descending=True).write_csv(os.path.join(a.out, "by_situation.csv"))

    c = res["counts"]
    print(f"{c['matches']:,} matches, {c['rounds']:,} rounds, {c['datapoints']:,} datapoints in {c['situations']:,} situations "
          f"({c[f'situations_with_{SIT_MIN}_or_more']:,} with {SIT_MIN} or more)")
    for m, r in res["models"].items():
        print(f"{m}: table {r['table_logloss']:.4f}, model {r['model_logloss']:.4f}, GDT {fmt(r['GDT'], r['GDT_ci'])}, rGDT {r['rGDT']:.1%}, "
              f"GDT_sit {fmt(r['GDT_sit'], r['GDT_sit_ci'])}, better than the table in {r['share_of_matches_model_better']:.1%} of matches")
    for m, r in res["vs_first"].items():
        print(f"{m} minus {r['against']}: GDT {fmt(r['GDT_difference'], r['GDT_difference_ci'], '+')}, "
              f"GDT_sit {fmt(r['GDT_sit_difference'], r['GDT_sit_difference_ci'], '+')}")
    print(f"written: {os.path.join(a.out, 'summary.json')}, {os.path.join(a.out, 'by_situation.csv')}")


if __name__ == "__main__":
    main()
