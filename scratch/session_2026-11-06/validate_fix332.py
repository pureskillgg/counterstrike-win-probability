"""fix332: validate the recount against earlier artifacts and reproduce the CHANGES counts (read-only on inputs)."""
import csv
import json
import os

import polars as pl

FIX = "/home/sagemaker-user/adam/builder/fixes/"
W = FIX + "fix332/"
out = {}

RF = pl.read_parquet(W + "round_flags.parquet")
bucket = RF.filter(pl.col("src") == "bucket")

# 1. reproduce the bucket scope scan (freeze_scope/scope_rounds.parquet), which used the same tome pages
sc = pl.read_parquet(FIX + "freeze_scope/scope_rounds.parquet")
j = bucket.select(["match_key", "round", "freeze", "alive_n", "unro_n", "dup_alive_n"]).join(
    sc.select(["match_key", "round", "freeze", "alive_n", "dup_alive", "unro_alive", "affected"]).rename({"freeze": "freeze_s", "alive_n": "alive_n_s"}),
    on=["match_key", "round"], how="full", coalesce=True)
out["scope_rounds_bucket"] = sc.height
out["recount_bucket_rounds"] = bucket.height
out["scope_join_missing_in_recount"] = j.filter(pl.col("freeze").is_null()).height
out["recount_missing_in_scope"] = j.filter(pl.col("freeze_s").is_null()).height
jj = j.filter(pl.col("freeze").is_not_null() & pl.col("freeze_s").is_not_null())
out["scope_freeze_mismatch"] = jj.filter(pl.col("freeze") != pl.col("freeze_s")).height
out["scope_alive_mismatch"] = jj.filter(pl.col("alive_n") != pl.col("alive_n_s")).height
out["scope_dup_mismatch"] = jj.filter(pl.col("dup_alive_n") != pl.col("dup_alive")).height
out["scope_unro_mismatch"] = jj.filter(pl.col("unro_n") != pl.col("unro_alive")).height
out["scope_affected_recount"] = jj.filter((pl.col("dup_alive_n") > 0) | (pl.col("unro_n") > 0)).height
out["scope_affected_scan"] = jj.filter(pl.col("affected")).height

# 2. per-match raw audit (fixes/f7_audit.json, 150 rounds from per-match player files)
f7 = json.load(open(FIX + "f7_audit.json"))
ours = {(r["match_key"], r["round"]): r for r in RF.select(["match_key", "round", "freeze", "alive_n", "unro_n", "dup_alive_n", "dup_roster_n"]).iter_rows(named=True)}
f7_ok = f7_bad = f7_missing = 0
f7_detail = []
for r in f7:
    k = (r["match_key"], int(r["round"]))
    o = ours.get(k)
    if o is None:
        f7_missing += 1
        continue
    same = (int(o["freeze"]) == int(r["freeze"]) and int(o["alive_n"]) == int(r["alive_at_freeze"])
            and int(o["unro_n"]) == len(r["unrostered_alive"]) and int(o["dup_alive_n"]) == len(r["dup_pids"]))
    if same:
        f7_ok += 1
    else:
        f7_bad += 1
        f7_detail.append({"match_key": k[0], "round": k[1], "ours": {x: o[x] for x in o}, "f7_alive": r["alive_at_freeze"],
                          "f7_unro": len(r["unrostered_alive"]), "f7_dup": len(r["dup_pids"]), "f7_freeze": r["freeze"]})
out["f7_rounds"] = len(f7)
out["f7_agree"] = f7_ok
out["f7_disagree"] = f7_bad
out["f7_not_in_recount"] = f7_missing
out["f7_disagree_examples"] = f7_detail[:5]

# 3. override_rounds.csv (the 283 fixed rounds and 1 open round from the investigated matches)
ov = list(csv.DictReader(open(FIX + "freeze_fix/override_rounds.csv")))
fixed = {(r["match_key"], int(r["round"])) for r in ov if r["status"] == "fixed"}
notfixed = {(r["match_key"], int(r["round"])) for r in ov if r["status"] != "fixed"}
out["override_rows"] = len(ov)
out["override_fixed"] = len(fixed)
out["override_not_fixed"] = len(notfixed)
cand = pl.read_csv(W + "candidates_tome.csv", schema_overrides={"dup_ids": pl.String, "unro_ids": pl.String})
cset = {(r["match_key"], int(r["round"])): r for r in cand.iter_rows(named=True)}
out["fixed_in_recount"] = sum(1 for k in fixed if k in cset)
out["fixed_not_in_recount"] = sum(1 for k in fixed if k not in cset)
out["fixed_kind_counts"] = {kd: sum(1 for k in fixed if k in cset and cset[k]["kind"] == kd) for kd in ["dup+unro", "dup", "unro"]}
out["notfixed_kind"] = [cset[k]["kind"] if k in cset else None for k in notfixed]
ov_map = {(r["match_key"], int(r["round"])): r for r in ov}
dup_pid_agree = unro_pid_agree = 0
for k in fixed | notfixed:
    if k not in cset or k not in ov_map:
        continue
    o = ov_map[k]
    od = sorted(int(x) for x in o["dups"].strip("[]").split(",") if x.strip())
    ou = sorted(int(x) for x in o["unrostered"].strip("[]").split(",") if x.strip())
    rd = sorted(int(x) for x in cset[k]["dup_ids"].split(";") if x) if cset[k]["dup_ids"] else []
    ru = sorted(int(x) for x in cset[k]["unro_ids"].split(";") if x) if cset[k]["unro_ids"] else []
    dup_pid_agree += int(od == rd)
    unro_pid_agree += int(ou == ru)
out["override_dup_pid_agree"] = dup_pid_agree
out["override_unro_pid_agree"] = unro_pid_agree

# 4. CHANGES reproduction: open = recount candidates minus the 283 fixed rounds
open_rows = [v for k, v in cset.items() if k not in fixed]
out["open_total"] = len(open_rows)
out["open_unro_only"] = sum(1 for v in open_rows if v["kind"] == "unro")
out["open_dup_related"] = sum(1 for v in open_rows if v["kind"] in ("dup", "dup+unro"))
out["open_dup_only"] = sum(1 for v in open_rows if v["kind"] == "dup")
out["open_dup_unro"] = sum(1 for v in open_rows if v["kind"] == "dup+unro")
out["changes_expected"] = {"unro_only_not_fixed": 302, "dup_related_not_fixed": 30, "total": 332}
out["changes_match"] = (out["open_unro_only"] == 302 and out["open_dup_related"] == 30)
out["unro_rounds_total_recount"] = sum(1 for v in cset.values() if v["kind"] in ("unro", "dup+unro"))

# 5. alive side sanity for the fixed set: 10 alive = dup + unro + 8 single-code (expect alive_ct + alive_t = 8 for fixed rounds)
side = [(cset[k]["alive_ct"], cset[k]["alive_t"]) for k in fixed if k in cset]
out["fixed_alive_ct_t_sum_eq_8"] = sum(1 for a, b in side if a + b == 8)
out["fixed_alive_ct_hist"] = {str(c): sum(1 for a, _ in side if a == c) for c in range(0, 11)}

# 6. definition counts across the whole export (tome coverage)
out["def1_dup_roster_rounds_all"] = RF.filter(pl.col("dup_roster_n") > 0).height
out["def2_unro_alive_rounds_all"] = RF.filter(pl.col("unro_n") > 0).height
out["candidates_tome_rows"] = cand.height
json.dump(out, open(W + "validation_tome.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str), flush=True)
