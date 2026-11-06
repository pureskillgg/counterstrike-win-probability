"""Where do the deep models fail? Per-second log loss against the GRU control, by scenario.

Uses the grader's own situations() and table.csv. Two sources:
  gdt   = rows_gdt.parquet: GDT-trained predictions (47 arm-seeds) and XGBoost seed 0, on the 297-match proxy population
  ll    = rows_all_arms.parquet: log-loss-trained predictions (47 arm-seeds), same rows
Only the seconds of the proxy population are used. No holdout or test rows.
"""
import importlib.util
import sys
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
SG = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/evaluation/intern_gdt/score_gdt.py"
TABLE = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/evaluation/intern_gdt/table.csv"
GDT_ROWS = "G:/pureskill/csds-cs2/data/diagnostics/gdt_final_2026-11-06/rows_gdt.parquet"
LL_ROWS = "G:/pureskill/csds-cs2/data/diagnostics/gdt_seq/rows_all_arms.parquet"
EPS = 1e-6

spec = importlib.util.spec_from_file_location("sg", SG)
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)

FIVE = ["gru", "xlstm_fast", "xlstm_blk", "xlstm_sig", "rwkv7", "mingru", "tcn"]
SEEDS = [0, 1, 2, 3, 4]
ALL_ARMS = ["gru", "xlstm", "xlstm_fast", "xlstm_blk", "xlstm_sig", "rwkv7", "mingru", "tcn", "tf", "tf2",
            "gdn", "gdn2", "lru", "lstm", "minlstm", "mamba2", "griffin", "gru2", "xlstm_mh"]


def load(path):
    df = pl.read_parquet(path)
    df = df.with_columns(
        pl.col("second").cast(pl.Float64).round(0),
        pl.col("since_plant").cast(pl.Float64),
        pl.col("y").cast(pl.Int64), pl.col("planted").cast(pl.Int64),
        pl.col("alive_ct").cast(pl.Int64), pl.col("alive_t").cast(pl.Int64),
        pl.col("round").cast(pl.Int64),
    )
    df = df.with_columns(pl.when(pl.col("planted") == 0).then(pl.lit(0.0)).otherwise(pl.col("since_plant")).alias("since_plant"))
    sit = sg.situations(df)
    table = pl.read_csv(TABLE, schema_overrides={"tb": pl.Utf8, "econ_ct": pl.Utf8, "econ_t": pl.Utf8})
    table = table.select(sg.CELL + [pl.col("p_ct").alias("p_table")])
    sit = sit.join(table, on=sg.CELL, how="left")
    return sit


def ll(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run(label, path, prefix_fn):
    sit = load(path)
    y = sit["y"].to_numpy().astype(np.float64)
    ll_table = ll(sit["p_table"].to_numpy(), y)
    seg_cols = {"planted": sit["planted"].to_numpy(), "tb": sit["tb"].to_numpy(),
                "alive": (sit["alive_ct"].to_numpy() * 10 + sit["alive_t"].to_numpy()),
                "econ_ct": sit["econ_ct"].to_numpy(), "econ_t": sit["econ_t"].to_numpy()}
    # per-second loss, averaged over seeds for each arm (five-seed arms) or the one seed
    arm_ll = {}
    for arm in FIVE + [a for a in ALL_ARMS if a not in FIVE]:
        cols = [prefix_fn(arm, s) for s in SEEDS]
        cols = [c for c in cols if c in sit.columns]
        if not cols:
            continue
        arm_ll[arm] = np.mean([ll(sit[c].to_numpy(), y) for c in cols], axis=0)
    print(f"\n===== {label}: {len(y):,} seconds, {sit['match'].n_unique()} matches")
    print(f"table per-second log loss: {ll_table.mean():.5f}")
    print("overall (mean over seeds): " + ", ".join(f"{a} {v.mean():.5f}" for a, v in arm_ll.items() if a in FIVE))
    gru = arm_ll["gru"]
    for seg_name, seg in seg_cols.items():
        print(f"\n-- by {seg_name} (gain vs GRU = GRU loss minus arm loss; positive = arm better; seconds)")
        keys = np.unique(seg)
        for k in keys:
            m = seg == k
            n = int(m.sum())
            if n < 20000:
                continue
            parts = []
            for a in FIVE[1:]:
                if a in arm_ll:
                    parts.append(f"{a} {float((gru[m] - arm_ll[a][m]).mean()):+.4f}")
            tab_gain = float((ll_table[m] - gru[m]).mean())
            print(f"  {str(k):>8} n={n:>9,} table-vs-GRU {tab_gain:+.4f} | " + " | ".join(parts))


if __name__ == "__main__":
    run("GDT-trained (proxy population)", GDT_ROWS, lambda arm, s: f"{arm}_s{s}")
    run("log-loss-trained (proxy population)", LL_ROWS, lambda arm, s: f"{arm}_s{s}")
