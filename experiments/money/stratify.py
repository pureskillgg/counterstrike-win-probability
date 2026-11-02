import json
import numpy as np, polars as pl

S = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
KEYS = ["match_key", "round", "second"]

lag = pl.read_parquet(S + "/lag/lag_test_preds.parquet").with_columns(pl.col("second").cast(pl.Int64))
gru = pl.read_parquet(S + "/gru/gru_full_test_preds.parquet").with_columns(pl.col("second").cast(pl.Int64))
cache = pl.read_parquet("G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026.parquet",
                        columns=KEYS + ["seconds_since_round_live", "bomb_planted", "alive_ct", "alive_t"]).with_columns(pl.col("second").cast(pl.Int64))

j = lag.join(gru.select(KEYS + ["p_gru"]), on=KEYS, how="inner").join(cache, on=KEYS, how="left")
print("joined rows", j.height, "matches", j["match_key"].n_unique(), flush=True)

y = j["label_ct_win"].cast(pl.Float64).to_numpy()

def ll(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

lb = ll(j["p_base59"].to_numpy())
lg = ll(j["p_lag"].to_numpy())
lr = ll(j["p_gru"].to_numpy())
j = j.with_columns([pl.Series("l_base", lb), pl.Series("d_lag", lg - lb), pl.Series("d_gru", lr - lb)])

def boot_ci(sub, col, n_draw=1000):
    codes, inv = np.unique(sub["match_key"].to_numpy(), return_inverse=True)
    s = np.bincount(inv, weights=sub[col].to_numpy())
    c = np.bincount(inv)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(codes), size=(n_draw, len(codes)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def table(group_col, expr_rows):
    print("\n== by " + group_col + " ==", flush=True)
    out = []
    for label, cond in expr_rows:
        sub = j.filter(cond)
        if sub.height == 0:
            continue
        lo, hi = boot_ci(sub, "d_lag")
        glo, ghi = boot_ci(sub, "d_gru")
        row = {"bucket": label, "rows": sub.height, "base_LL": round(float(sub["l_base"].mean()), 4),
               "d_lag": round(float(sub["d_lag"].mean()), 5), "d_lag_CI": [round(lo, 5), round(hi, 5)],
               "d_gru": round(float(sub["d_gru"].mean()), 5), "d_gru_CI": [round(glo, 5), round(ghi, 5)]}
        out.append(row)
        print(json.dumps(row), flush=True)
    return out

res = {}
res["secs_in_round"] = table("seconds into round", [
    ("<5s", pl.col("seconds_since_round_live") < 5),
    ("5-15s", (pl.col("seconds_since_round_live") >= 5) & (pl.col("seconds_since_round_live") < 15)),
    ("15-30s", (pl.col("seconds_since_round_live") >= 15) & (pl.col("seconds_since_round_live") < 30)),
    ("30-60s", (pl.col("seconds_since_round_live") >= 30) & (pl.col("seconds_since_round_live") < 60)),
    ("60-90s", (pl.col("seconds_since_round_live") >= 60) & (pl.col("seconds_since_round_live") < 90)),
    ("90s+", pl.col("seconds_since_round_live") >= 90),
])
res["phase"] = table("bomb phase", [
    ("pre-plant", pl.col("bomb_planted") == 0),
    ("post-plant", pl.col("bomb_planted") == 1),
])
res["pre_plant_5v5"] = table("pre-plant alive config", [
    ("pre-plant 5v5", (pl.col("bomb_planted") == 0) & (pl.col("alive_ct") == 5) & (pl.col("alive_t") == 5)),
    ("pre-plant other", (pl.col("bomb_planted") == 0) & ~((pl.col("alive_ct") == 5) & (pl.col("alive_t") == 5))),
])
json.dump(res, open(S + "/money/stratify_results.json", "w"), indent=1)
print("saved", flush=True)
