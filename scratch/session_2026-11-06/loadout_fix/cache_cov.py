import json, os, sys, time
from collections import Counter
import numpy as np
import polars as pl

S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix/"
name = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else "G:/pureskill/csds-cs2/data/"
folder = root + name
maxf = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
t0 = time.time()

keys = json.load(open(S + "match_keys.json"))
K = set(keys)
rf = np.load(S + "round_freeze.npy").astype(np.int64)
rm = np.load(S + "round_match.npy")
rn = np.load(S + "round_num.npy").astype(np.int32)
exp = pl.DataFrame({"match_key": [keys[i] for i in rm.tolist()], "round": rn, "freeze": rf})
exp_dup_pairs = exp.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1).height
expk = exp.unique(subset=["match_key", "round"], keep="first", maintain_order=False)
export_month = Counter(k[5:12] for k in keys)

files = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))[:maxf]
seen = set(); covered = set()
per_file = []
dt_c = Counter(); ds_c = Counter(); np_c = Counter(); ns_c = Counter(); nps_c = Counter()
rem_c = Counter(); diff_c = Counter()
tot_rows = 0; matched_rows = 0; rounds_total = 0; rounds_matched = 0; date_mis_keys = 0
cols_seen = None; dtypes_seen = None
for f in files:
    p = folder + "/" + f
    fd = f[:-8].replace("-", "/")
    df = pl.read_parquet(p, columns=["match_key", "round", "tick", "second", "player_id", "side"])
    if cols_seen is None:
        cols_seen = df.columns; dtypes_seen = [str(t) for t in df.dtypes]
    n = df.height; tot_rows += n
    fkeys = df.get_column("match_key").unique().to_list()
    seen.update(fkeys)
    inK = [k for k in fkeys if k in K]
    covered.update(inK)
    dmis = sum(1 for k in fkeys if k[5:15] != fd)
    date_mis_keys += dmis
    d = df.sort(["match_key", "round", "player_id", "tick"]).with_columns(
        dt=pl.col("tick").diff().over(["match_key", "round", "player_id"]),
        ds=pl.col("second").diff().over(["match_key", "round", "player_id"]))
    for v, c in d.group_by("dt").agg(pl.len().alias("n")).iter_rows():
        dt_c[str(v)] += c
    for v, c in d.group_by("ds").agg(pl.len().alias("n")).iter_rows():
        ds_c[str(v)] += c
    r = df.group_by(["match_key", "round"]).agg(
        pl.col("player_id").n_unique().alias("np"), pl.col("side").n_unique().alias("ns"))
    rounds_total += r.height
    for v, c in r.group_by("np").agg(pl.len().alias("n")).iter_rows():
        np_c[str(v)] += c
    for v, c in r.group_by("ns").agg(pl.len().alias("n")).iter_rows():
        ns_c[str(v)] += c
    ps = df.group_by(["match_key", "round", "side"]).agg(pl.col("player_id").n_unique().alias("npp"))
    for v, c in ps.group_by("npp").agg(pl.len().alias("n")).iter_rows():
        nps_c[str(v)] += c
    j = df.select(["match_key", "round", "tick", "second"]).join(expk, on=["match_key", "round"], how="left")
    jm = j.filter(pl.col("freeze").is_not_null())
    mrows = jm.height; matched_rows += mrows
    rm_pairs = jm.select(["match_key", "round"]).unique().height
    rounds_matched += rm_pairs
    if mrows > 0:
        rel = jm.with_columns(
            rem=((pl.col("tick") - pl.col("freeze")) % 64 + 64) % 64,
            diff=(pl.col("tick") - pl.col("freeze") - 64 * pl.col("second")))
        for v, c in rel.group_by("rem").agg(pl.len().alias("n")).iter_rows():
            rem_c[str(v)] += c
        for v, c in rel.group_by("diff").agg(pl.len().alias("n")).iter_rows():
            diff_c[str(v)] += c
    per_file.append({"file": f, "date_from_name": fd, "rows": n, "match_keys": len(fkeys),
                     "keys_in_export": len(inK), "keys_not_in_export": len(fkeys) - len(inK),
                     "keys_date_mismatch": dmis, "rounds": r.height, "rounds_matched_to_export": rm_pairs,
                     "rows_matched_to_export_freeze": mrows})
    print(f"{name} {f} rows={n} keys={len(fkeys)} inK={len(inK)} rounds={r.height} matched_rounds={rm_pairs} t={time.time()-t0:.1f}s", flush=True)

exp_rounds_per_key = exp.group_by("match_key").agg(pl.len().alias("n"))
erk = dict(zip(exp_rounds_per_key.get_column("match_key").to_list(), exp_rounds_per_key.get_column("n").to_list()))
export_rounds_for_covered = sum(erk.get(k, 0) for k in covered)
keys_in_more_than_one_file = sum(p["keys_in_export"] for p in per_file) - len(covered)
cov_month = Counter(k[5:12] for k in covered)
def top(c, k=8):
    tot = sum(c.values())
    return [(a, b, round(b / tot, 5)) for a, b in sorted(c.items(), key=lambda x: -x[1])[:k]]
summary = {
    "cache": name, "folder": folder, "files": len(files), "columns": cols_seen, "dtypes": dtypes_seen,
    "rows": tot_rows, "distinct_match_keys_seen": len(seen), "seen_not_in_export": len(seen - K),
    "covered_export_matches": len(covered), "export_total": len(K),
    "covered_by_month": {m: [cov_month.get(m, 0), export_month[m]] for m in sorted(export_month)},
    "months_with_any_cover": sorted(m for m in cov_month),
    "rounds_in_cache": rounds_total, "rounds_matched_export": rounds_matched,
    "export_rounds_total": int(exp.height), "export_rounds_for_covered_keys": int(export_rounds_for_covered),
    "keys_in_more_than_one_file": int(keys_in_more_than_one_file),
    "export_dup_match_round_pairs": exp_dup_pairs,
    "keys_date_mismatch_vs_filename": date_mis_keys,
    "rows_matched_export_freeze": matched_rows,
    "dt_tick_diff_top": top(dt_c), "ds_second_diff_top": top(ds_c),
    "players_per_round_top": top(np_c), "sides_per_round_top": top(ns_c), "players_per_side_top": top(nps_c),
    "rem_tick_minus_freeze_mod64_top": top(rem_c), "tick_minus_freeze_minus_64sec_top": top(diff_c),
    "per_file": per_file, "seconds": round(time.time() - t0, 1),
}
out = S + f"cov_{name}" + ("" if root.startswith("G:") else "_E") + (f"_first{maxf}" if maxf < 10**9 else "") + ".json"
json.dump(summary, open(out, "w"), indent=1, default=str)
print("WROTE", out, "covered", len(covered), "rows", tot_rows, "secs", summary["seconds"])
for k in ["covered_export_matches", "distinct_match_keys_seen", "seen_not_in_export", "rounds_in_cache", "rounds_matched_export", "keys_date_mismatch_vs_filename", "rows_matched_export_freeze", "dt_tick_diff_top", "ds_second_diff_top", "players_per_round_top", "sides_per_round_top", "players_per_side_top", "rem_tick_minus_freeze_mod64_top", "tick_minus_freeze_minus_64sec_top"]:
    print(k, summary[k])
