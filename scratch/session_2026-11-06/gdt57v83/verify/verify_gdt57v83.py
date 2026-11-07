"""Independent re-check of the 57-vs-83 GDT validation run.

Reads only identifier columns from the holdout and frozen key sets (for the overlap assertion).
Recomputes the scorer's draw, table lookup, GDT, GDT_sit and match-bootstrap intervals from the
saved validation rows, then cross-checks against score_ref_57 / score_ref_base summary.json and
by_situation.csv. Writes only under the gdt57v83 scratchpad.
"""
import csv
import json
import os
from collections import Counter

import numpy as np
import polars as pl

S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/gdt57v83"
PB = "C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/positions_validation"
G = "G:/pureskill/csds-cs2"
REPO = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability"
OUT = f"{S}/verify_out"
os.makedirs(OUT, exist_ok=True)
R = {}
ARMS = ["p_57", "p_full_trainctr", "p_sprint83", "p_base"]


def date_of(k):
    return "/".join(k.split("/")[1:4])


# ---------------------------------------------------------------- key sets (identifiers only)
hb = set(json.load(open(f"{G}/data/combined/holdout_22_25_test_keys.json", encoding="utf-8"))["match_keys"])
fz = set(json.load(open(f"{G}/data/benchmark/frozen_test_keys.json", encoding="utf-8")))
oc = set(pl.read_parquet("C:/Users/<user>/sprint/holdout_oct26_31_keys.parquet", columns=["match_key"])["match_key"].to_list())
ocf = pl.read_parquet("C:/Users/<user>/sprint/holdout_oct26_31/keys.parquet", columns=["match_key", "round", "second"])
ocf_set = set(ocf["match_key"].unique().to_list())
HOLD = {"holdoutB": hb, "frozen": fz, "oct26_31_matchkeys": oc, "oct26_31_fullkeys": ocf_set}
R["key_set_sizes"] = {k: len(v) for k, v in HOLD.items()}
R["oct_match_keysets_equal"] = oc == ocf_set

# ---------------------------------------------------------------- the rows used
rows = pl.read_parquet(f"{S}/validation_rows_297_five_arms.parquet")
used = set(rows["match"].unique().to_list())
R["rows"] = {"rows": rows.height, "matches": len(used), "rounds": rows.select(["match", "round"]).n_unique(),
             "dup_match_round_second": int(rows.select(["match", "round", "second"]).is_duplicated().sum()),
             "nulls_in_arms": {a: rows[a].null_count() for a in ARMS}}
R["overlap_rows_matches_vs_keysets"] = {k: len(used & v) for k, v in HOLD.items()}
rk = rows.select([pl.col("match"), pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)])
ock = ocf.select([pl.col("match_key").alias("match"), pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64)]).unique()
R["overlap_exact_rows_oct26_31_full"] = rk.join(ock, on=["match", "round", "second"], how="inner").height

# ---------------------------------------------------------------- each arm's source key set
srcs = {
    "p_57": (f"{S}/val_rescored_v6_drop_heavy_57.parquet", "match_key"),
    "p_sprint83": (f"{G}/data/sprint/run/gpu_fast/preds/full_d4_e0.03_m20_s0.parquet", "match_key"),
    "p_base": (f"{PB}/base_seed0.parquet", "match_key"),
    "p_full_trainctr": (f"{PB}/full_trainctr_seed0.parquet", "match_key"),
}
src_sets = {}
R["source_matches"] = {}
for arm, (path, col) in srcs.items():
    ks = set(pl.read_parquet(path, columns=[col])[col].unique().to_list())
    src_sets[arm] = ks
    R["source_matches"][arm] = {
        "n_matches_in_source_file": len(ks),
        "used_rows_present_in_source": len(used & ks),
        "used_overlap_with_keysets": {k: len(used & v) for k, v in HOLD.items()},
        "whole_source_overlap_with_keysets": {k: len(ks & v) for k, v in HOLD.items()},
    }
R["source_schemas"] = {a: list(pl.read_parquet_schema(p).keys()) for a, (p, _) in srcs.items()}

# validation split key file (schema first, then overlap if a match column exists)
vk_path = f"{G}/data/sprint/matrix_trainctr/val_keys.parquet"
vsch = pl.read_parquet_schema(vk_path)
R["val_keys_schema"] = {k: str(v) for k, v in vsch.items()}
vcol = "match_key" if "match_key" in vsch else ("match" if "match" in vsch else None)
if vcol:
    vk = set(pl.read_parquet(vk_path, columns=[vcol])[vcol].unique().to_list())
    R["val_keys"] = {"n_matches": len(vk), "equals_57_file_matches": vk == src_sets["p_57"],
                     "overlap_with_keysets": {k: len(vk & v) for k, v in HOLD.items()}}

# ---------------------------------------------------------------- population (check 2)
d297 = sorted({date_of(k) for k in used})
v57 = src_sets["p_57"]
R["population"] = {
    "matches_used": len(used), "date_min": d297[0], "date_max": d297[-1], "distinct_dates": len(d297),
    "used_on_oct9_to_13_table_days": sum(1 for k in used if "2026/10/09" <= date_of(k) <= "2026/10/13"),
    "used_on_or_after_oct14": sum(1 for k in used if date_of(k) >= "2026/10/14"),
    "used_all_csds2026_prefix": all(k.startswith("csds/2026/") for k in used),
    "used_subset_of_57_validation_file": used <= v57,
    "validation_file_matches": len(v57),
    "validation_file_on_oct9_to_13": sum(1 for k in v57 if "2026/10/09" <= date_of(k) <= "2026/10/13"),
    "validation_file_date_min": min(date_of(k) for k in v57), "validation_file_date_max": max(date_of(k) for k in v57),
}

# p_57 and p_sprint83 columns in the rows equal their source predictions at (match, round, second); labels too
for arm in ["p_57", "p_sprint83"]:
    path = srcs[arm][0]
    f = pl.read_parquet(path, columns=["match_key", "round", "second", "p", "label_ct_win"])
    f = f.filter(pl.col("match_key").is_in(list(used))).select([
        pl.col("match_key").alias("match"), pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64),
        pl.col("p").cast(pl.Float64).alias("p_file"), pl.col("label_ct_win").cast(pl.Int64).alias("y_file")])
    j = rows.select([pl.col("match"), pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64),
                     pl.col(arm).cast(pl.Float64).alias("p_rows"), pl.col("y").cast(pl.Int64).alias("y_rows")]) \
        .join(f, on=["match", "round", "second"], how="inner")
    R[f"join_{arm}"] = {"rows_in": rows.height, "joined": j.height,
                        "max_abs_p_diff": float((j["p_rows"] - j["p_file"]).abs().max()),
                        "label_mismatch": int((j["y_rows"] != j["y_file"]).sum())}

# ---------------------------------------------------------------- independent recomputation (check 3 and 4)
df = rows.sort(["match", "round", "second", "since_plant"])
n = df.height
rng = np.random.default_rng(0)
r = rng.random(n)                      # same draw as score_gdt.draw(): one uniform per row, in sorted order
match = np.array(df["match"].to_list(), dtype=object)
rnd = df["round"].to_numpy().astype(np.int64)
sec = np.round(df["second"].to_numpy().astype(np.float64))
plant = df["planted"].to_numpy().astype(np.int64)
sp = df["since_plant"].to_numpy().astype(np.float64)
ac = df["alive_ct"].to_numpy().astype(np.int64)
at = df["alive_t"].to_numpy().astype(np.int64)
ec = df["equip_ct"].to_numpy().astype(np.float64)
et = df["equip_t"].to_numpy().astype(np.float64)
yy = df["y"].to_numpy().astype(np.float64)


def third(x, length):
    return np.where(x < length / 3, 0, np.where(x < 2 * length / 3, 1, 2))   # 0 lot, 1 medium, 2 little


def econ(eq, al, lo, hi):
    ev = np.where(al > 0, eq / np.maximum(al, 1), 0.0)
    return np.where(ev < lo, 0, np.where(ev < hi, 1, 2))                       # 0 low, 1 mid, 2 high


tbc = np.where(plant == 1, third(sp, 40.0), third(sec, 115.0))
ecc = econ(ec, ac, 2837.5, 4200.0)
ect = econ(et, at, 2420.0, 3900.0)

TB = {"lot": 0, "medium": 1, "little": 2}
EC = {"low": 0, "mid": 1, "high": 2}
tab = {}
with open(f"{REPO}/evaluation/intern_gdt/table.csv", newline="", encoding="utf-8") as fh:
    trows = list(csv.DictReader(fh))
for x in trows:
    tab[(int(x["alive_ct"]), int(x["alive_t"]), int(x["planted"]), TB[x["tb"]], EC[x["econ_ct"]], EC[x["econ_t"]])] = float(x["p_ct"])
R["table"] = {"rows": len(trows), "unique_keys": len(tab)}

mi = np.unique(match, return_inverse=True)[1].reshape(-1)
keyarr = np.column_stack([mi, rnd, ac, at, plant, tbc, ecc, ect]).astype(np.int64)
_, Gid = np.unique(keyarr, axis=0, return_inverse=True)
Gid = Gid.reshape(-1)
order = np.lexsort((r, Gid))           # within each (match, round, situation) group, smallest r first
Gs = Gid[order]
first = np.concatenate([[True], Gs[1:] != Gs[:-1]])
chosen = order[first]                  # one datapoint per round per situation

dp_match = match[chosen]
dp_y = yy[chosen]
dp_cell = keyarr[chosen][:, 2:]        # alive_ct, alive_t, planted, tb, econ_ct, econ_t
p_table = np.array([tab.get(tuple(int(v) for v in c), np.nan) for c in dp_cell])
dp_n = len(chosen)
cell_counter = Counter(tuple(int(v) for v in c) for c in dp_cell)
n_s = np.array([cell_counter[tuple(int(v) for v in c)] for c in dp_cell], dtype=np.float64)
rounds_dp = len(np.unique(np.column_stack([mi[chosen], rnd[chosen]]), axis=0))
R["recount"] = {
    "rows": n, "matches": len(np.unique(dp_match)), "rounds": rounds_dp, "datapoints": int(dp_n),
    "situations": len(cell_counter), "situations_500_plus": sum(1 for c in cell_counter.values() if c >= 500),
    "datapoints_missing_table": int(np.isnan(p_table).sum()),
}


def ll(p, yv):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return -(yv * np.log(p) + (1 - yv) * np.log(1 - p))


def wboot(mids, la, lb, w, reps=1000, seed=0):
    ids, inv = np.unique(mids, return_inverse=True)
    inv = inv.reshape(-1)
    sw = np.bincount(inv, weights=w)
    sa = np.bincount(inv, weights=w * la)
    sb = np.bincount(inv, weights=w * lb)
    m = len(ids)
    gen = np.random.default_rng(seed)
    dd = np.empty(reps)
    for i in range(reps):
        pick = gen.integers(0, m, m)
        dd[i] = (sa[pick].sum() - sb[pick].sum()) / sw[pick].sum()
    lo, hi = np.percentile(dd, [2.5, 97.5])
    return {"a": sa.sum() / sw.sum(), "b": sb.sum() / sw.sum(), "diff": (sa.sum() - sb.sum()) / sw.sum(),
            "ci": [float(lo), float(hi)]}


ones = np.ones(dp_n)
w_sit = np.where(n_s >= 500, 1.0 / n_s, 0.0)
has_sit = w_sit.sum() > 0
lt = ll(p_table, dp_y)
arm_vals = {a: df[a].to_numpy().astype(np.float64)[chosen] for a in ARMS}


def recompute(models, ref):
    lm = {m: ll(arm_vals[m], dp_y) for m in models}
    res = {"models": {}, "vs_first": {}}
    ids, inv = np.unique(dp_match, return_inverse=True)
    inv = inv.reshape(-1)
    for m in models:
        g = wboot(dp_match, lt, lm[m], ones)
        gs = wboot(dp_match, lt, lm[m], w_sit) if has_sit else None
        pm = np.bincount(inv, weights=lt - lm[m]) / np.bincount(inv)
        res["models"][m] = {"table_logloss": g["a"], "model_logloss": g["b"], "GDT": g["diff"], "GDT_ci": g["ci"],
                            "rGDT": 1 - g["b"] / g["a"], "GDT_sit": gs["diff"] if gs else None,
                            "GDT_sit_ci": gs["ci"] if gs else None,
                            "share_of_matches_model_better": float((pm > 0).mean())}
    for m in models:
        if m == ref:
            continue
        d = wboot(dp_match, lm[ref], lm[m], ones)
        ds = wboot(dp_match, lm[ref], lm[m], w_sit) if has_sit else None
        res["vs_first"][m] = {"against": ref, "GDT_difference": d["diff"], "GDT_difference_ci": d["ci"],
                              "GDT_sit_difference": ds["diff"] if ds else None,
                              "GDT_sit_difference_ci": ds["ci"] if ds else None}
    return res


def compare(mine, summ):
    out = {}
    for m, rr in mine["models"].items():
        s = summ["models"][m]
        for k in ["table_logloss", "model_logloss", "GDT", "rGDT", "GDT_sit", "share_of_matches_model_better"]:
            out[f"{m}.{k}"] = abs(rr[k] - s[k])
        for k in ["GDT_ci", "GDT_sit_ci"]:
            out[f"{m}.{k}"] = max(abs(a - b) for a, b in zip(rr[k], s[k]))
    for m, rr in mine["vs_first"].items():
        s = summ["vs_first"][m]
        for k in ["GDT_difference", "GDT_sit_difference"]:
            out[f"diff:{m}.{k}"] = abs(rr[k] - s[k])
        for k in ["GDT_difference_ci", "GDT_sit_difference_ci"]:
            out[f"diff:{m}.{k}"] = max(abs(a - b) for a, b in zip(rr[k], s[k]))
    return {k: float(v) for k, v in out.items()}


runA = recompute(["p_57", "p_full_trainctr", "p_sprint83"], "p_57")
runB = recompute(["p_base", "p_57", "p_full_trainctr", "p_sprint83"], "p_base")
sumA = json.load(open(f"{S}/score_ref_57/summary.json", encoding="utf-8"))
sumB = json.load(open(f"{S}/score_ref_base/summary.json", encoding="utf-8"))
R["recomputed_runA"] = runA
R["recomputed_runB"] = runB
R["max_abs_diff_vs_summary_runA"] = max(compare(runA, sumA).values())
R["max_abs_diff_vs_summary_runB"] = max(compare(runB, sumB).values())
R["summary_counts_runA"] = sumA["counts"]
R["summary_counts_runB"] = sumB["counts"]
R["headline_83_minus_57_recomputed"] = {
    "full_trainctr": {"GDT": runA["vs_first"]["p_full_trainctr"]["GDT_difference"], "ci": runA["vs_first"]["p_full_trainctr"]["GDT_difference_ci"]},
    "sprint83": {"GDT": runA["vs_first"]["p_sprint83"]["GDT_difference"], "ci": runA["vs_first"]["p_sprint83"]["GDT_difference_ci"]},
}

# ---------------------------------------------------------------- cross-check from saved by_situation.csv
def from_sit(path, models):
    with open(path, newline="", encoding="utf-8") as fh:
        rr = list(csv.DictReader(fh))
    dp = np.array([float(x["datapoints"]) for x in rr])
    llt = np.array([float(x["ll_table"]) for x in rr])
    out = {"situation_rows": len(rr), "sum_datapoints": int(dp.sum()), "rows_with_500_plus": int((dp >= 500).sum()),
           "table_logloss": float((dp * llt).sum() / dp.sum())}
    for m in models:
        gain = np.array([float(x[f"gain_{m}"]) for x in rr])
        llm = np.array([float(x[f"ll_{m}"]) for x in rr])
        out[m] = {"model_logloss": float((dp * llm).sum() / dp.sum()), "GDT": float((dp * gain).sum() / dp.sum()),
                  "GDT_sit": float(gain[dp >= 500].mean())}
    return out, rr


sitA, rrA = from_sit(f"{S}/score_ref_57/by_situation.csv", ["p_57", "p_full_trainctr", "p_sprint83"])
sitB, rrB = from_sit(f"{S}/score_ref_base/by_situation.csv", ["p_base", "p_57", "p_full_trainctr", "p_sprint83"])
R["by_situation_runA"] = sitA
R["by_situation_runB"] = sitB
R["by_situation_gdt_diff_83_minus_57"] = {
    "full_trainctr": sitA["p_full_trainctr"]["GDT"] - sitA["p_57"]["GDT"],
    "sprint83": sitA["p_sprint83"]["GDT"] - sitA["p_57"]["GDT"],
}
# per-situation datapoint counts: my recount vs the scorer's by_situation.csv
my_cells = {k: v for k, v in cell_counter.items()}
csv_cells = {}
for x in rrA:
    k = (int(x["alive_ct"]), int(x["alive_t"]), int(x["planted"]), TB[x["tb"]], EC[x["econ_ct"]], EC[x["econ_t"]])
    csv_cells[k] = int(x["datapoints"])
R["situation_count_match"] = {"cells_mine": len(my_cells), "cells_csv": len(csv_cells),
                              "cells_equal_set": set(my_cells) == set(csv_cells),
                              "cells_count_mismatch": sum(1 for k in my_cells if csv_cells.get(k) != my_cells[k])}

with open(f"{OUT}/check_results.json", "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
