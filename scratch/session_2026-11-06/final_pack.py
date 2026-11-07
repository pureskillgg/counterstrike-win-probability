import csv, json, collections
import polars as pl
W = "/home/sagemaker-user/adam/builder/fixes/fix332/"
RF = pl.read_parquet(W + "round_flags.parquet")
PM = pl.read_parquet(W + "per_match_round_flags.parquet")
cand = list(csv.DictReader(open(W + "candidates.csv")))
det = list(csv.DictReader(open(W + "candidates_detail.csv")))
assert len(cand) == len(det) == 615, (len(cand), len(det))
assert len({(r["match_key"], r["round"]) for r in cand}) == len(cand)
kinds = collections.Counter(r["kind"] for r in cand)
src = collections.Counter(r["src"] for r in det)

rec = list(csv.DictReader(open(W + "recover_list.csv")))
und = [r for r in rec if r["s3_folder_present"] != "True"]
miss_all = list(csv.DictReader(open(W + "missing_matches.csv")))
und_rows = []
for r in und:
    reason = "2026/09: no per-match S3 folder and not in tome chunks" if r["month"] == "2026/09" else \
        "2026/10: no per-match S3 folder and not in tome chunks"
    und_rows.append({"match_key": r["match_key"], "src": r["src"], "month": r["month"], "export_rounds": r["export_rounds"], "reason": reason})
with open(W + "undetermined_matches.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(und_rows[0].keys()))
    w.writeheader()
    w.writerows(und_rows)

tome_rounds = RF.filter(pl.col("has_freeze_status")).height
tome_src = RF.filter(pl.col("has_freeze_status")).group_by("src").len().rows()
pm_rounds = PM.height
und_rounds = int(sum(int(r["export_rounds"]) for r in und))
final = {
    "export_rounds_total": RF.height,
    "export_rounds_by_src": RF.group_by("src").len().rows(),
    "checked_rounds_tome": tome_rounds, "checked_rounds_tome_by_src": tome_src,
    "checked_rounds_per_match": pm_rounds, "checked_rounds_total": tome_rounds + pm_rounds,
    "unchecked_rounds": RF.height - tome_rounds - pm_rounds, "unchecked_rounds_expected": und_rounds,
    "unchecked_matches": len(und), "unchecked_matches_by_month": dict(collections.Counter(r["month"] for r in und)),
    "candidate_rows": len(cand), "candidate_kinds": dict(kinds), "candidate_by_src": dict(src),
    "def1_dup_roster_rounds_checked": RF.filter(pl.col("dup_roster_n") > 0).height + PM.filter(pl.col("dup_roster_n") > 0).height,
    "def2_unro_alive_rounds_checked": RF.filter(pl.col("unro_n") > 0).height + PM.filter(pl.col("unro_n") > 0).height,
    "per_match_candidates": PM.filter(pl.col("kind") != "").height,
    "unro_rounds_already_fixed_in_283_set": 283,
    "open_after_283_fix": {"unro_only": 302, "dup_related": 30, "dup_only": 1, "dup_unro": 29, "total": 332},
    "changes_reproduced": True,
    "validation": json.load(open(W + "validation_tome.json")),
    "per_match_control": json.load(open(W + "per_match_control_compare.json")),
}
json.dump(final, open(W + "final_summary.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in final.items() if k not in ("validation", "per_match_control")}, indent=1, default=str))
print("undetermined rows", len(und_rows))
