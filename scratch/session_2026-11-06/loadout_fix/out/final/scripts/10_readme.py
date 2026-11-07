import json, os, hashlib, shutil, re
import numpy as np
import polars as pl
S = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/loadout_fix"
FIN = S + "/out/final"
os.makedirs(FIN + "/scripts", exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def chunk_stats(logpath, tag):
    out = []
    for line in open(logpath, encoding="utf-8", errors="replace"):
        if '"chunk"' in line:
            out.append(json.loads(line[line.index("{"):]))
    for r in out:
        r["job"] = tag
    return out


cs = chunk_stats(S + "/out/s3_job.log", "job1_bucket") + chunk_stats(S + "/out/s3_job2.log", "job2_local_covered_in_channel")
json.dump(cs, open(FIN + "/s3_chunk_stats.json", "w"), indent=1)
tot = {}
for k in ["status_rows_raw", "rows_wanted_matches", "rows_no_export_round", "rows_unrostered", "rows_bad_side",
          "rows_pre_freeze", "rows_off_grid", "rows_beyond_round_steps", "rows_kept_before_dedupe", "duplicates_dropped",
          "agg_rows", "target_agg_rows_written"]:
    for tag in ("job1_bucket", "job2_local_covered_in_channel"):
        tot[tag + "." + k] = int(sum(r.get(k, 0) for r in cs if r["job"] == tag))
chunks_read = {tag: sorted(set(r["chunk"] for r in cs if r["job"] == tag)) for tag in ("job1_bucket", "job2_local_covered_in_channel")}
chunks_read = {k: [chunks_read[k][0], chunks_read[k][-1], len(chunks_read[k])] for k in chunks_read}
local_stats = json.load(open(S + "/out/space_local_file_stats.json"))
local_tot = {k: int(sum(x[k] for x in local_stats)) for k in ["raw_rows", "rows_in_scope_matches", "rows_no_export_round", "rows_pre_freeze",
                                                               "rows_off_grid", "rows_beyond_round_steps", "rows_bad_side",
                                                               "duplicates_dropped", "agg_rows_written"]}
local_files = [{"file": x["file"], "src": x["src"], "raw_rows_in_file_or_reduced": x["raw_rows"], "rows_kept": x["rows_kept_before_dedupe"], "matches_written": x["matches"]} for x in local_stats]
local_full_stats = json.load(open(S + "/out/local_file_stats.json"))
full_rows_by_file = {x["file"]: x["raw_rows"] for x in local_full_stats}

# built steps and matches
mi = pl.read_parquet(S + "/out/final/match_status.parquet")
off = np.load(S + "/offsets.npy").astype(np.int64)
rm = np.load(S + "/round_match.npy").astype(np.int64)
nsteps = np.diff(off)
steps_per_match = np.bincount(rm, weights=nsteps, minlength=len(mi)).astype(np.int64)
mi = mi.with_columns(pl.Series("steps_total", steps_per_match))
built = mi.filter(pl.col("build_status").is_in(["s3_channel", "local_1hz"]))
steps_built = int(built["steps_total"].sum())
steps_total = int(off[-1])
status_counts = mi.group_by(["build_status", "reason"]).agg(pl.len().alias("matches"), pl.col("steps_total").sum().alias("steps")).sort(["build_status", "reason"]).to_dicts()
split_counts = mi.group_by("split").agg(pl.len().alias("matches")).sort("split").to_dicts()
prot = mi.with_columns((pl.col("date") >= "2026-10-21").alias("protected_date")).group_by(["split", "protected_date"]).agg(pl.len().alias("matches")).sort(["split", "protected_date"]).to_dicts()
verify = json.load(open(S + "/out/verify_result.json"))
parity = json.load(open(S + "/out/parity_result.json"))
hashes_space = verify["hashes"]
hashes_local = json.load(open(S + "/out/local_hashes_sel.json"))
n_hash_ok = sum(1 for k in hashes_local if hashes_space.get(k) == hashes_local[k])
hash_summary = {"sampled_local_required_matches": len(hashes_local), "equal": n_hash_ok, "not_equal": [k for k in hashes_local if hashes_space.get(k) != hashes_local[k]][:10]}

shared_fields = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle", "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy"]
arrays = {}
for f in ["loadout_step_i16.npy", "loadout_equip_i32.npy"]:
    p = FIN + "/" + f
    arrays[f] = {"bytes": os.path.getsize(p) if os.path.exists(p) else None}

readme = {
    "name": "export_clean_v3_loadout",
    "status": "PARTIAL-COMPLETE: every in-scope match is built; test split and protected-date matches are named and not built",
    "built_on": "2026-11-05",
    "aligned_to": "export_clean_v2 (same match index, round index, offsets.npy, steps_alive.npy). Step s of round r is offsets[r] + k, with tick = round_freeze[r] + 64*k, k < offsets[r+1]-offsets[r].",
    "shape_and_dtype": {
        "loadout_step_i16.npy": "int16, shape (72567700, 2, 19). Axis 1: 0 = CT, 1 = T. Axis 2: " + ", ".join(shared_fields) + ". Fields are the serving per-side aggregates (win_probability.py lines 257-277) except equip; the six CT-minus-T diffs are not stored (derive them).",
        "loadout_equip_i32.npy": "int32, shape (72567700, 2). Sum of equipment_value_calc per side (int32 because one side can exceed 32767).",
    },
    "rows": {
        "total_steps_in_export": steps_total,
        "steps_of_built_matches": steps_built,
        "per_second_rows_with_any_data_CT_T": verify.get("rows_with_data_CT_T"),
        "steps_with_any_data": verify.get("steps_with_any_data"),
        "note": "a row is one (step, side). A step or side with no living player has all-zero values, which is the serving rule for a tick with no rows.",
    },
    "rule_serving_quoted": {
        "source": "G:/pureskill/csds-cs2/scripts/serving/win_probability.py (HEAD 4320dce)",
        "grid_L249": "live = (ps.join(fdf, on=\"round\", how=\"inner\").filter(((pl.col(\"tick\") - pl.col(\"frz\")) % TICK == 0) & (pl.col(\"tick\") >= pl.col(\"frz\"))))",
        "freeze_L212_213": "freeze.setdefault(int(r[0]), int(r[1])) over round_state rows with event_type round_freeze_end",
        "loop_bound_L141_146": "last = end if planted_at is not None else min(end, frz + UNPLANTED_CAP_SECONDS * TICK); tick = frz + TICK * k; if tick >= last: break",
        "side_L281": "side_key = \"ct\" if int(r[\"side\"]) == CT else \"t\"  (CT=3, T=2)",
        "missing_rows_L170_and_L287_289": "load.get(tick, ZERO_LOADOUT); a side with no rows is set to 0.0 (no carry-forward)",
        "null_rules_L259_272": "health, armor, equipment, inv_* null -> 0; flags null -> false; inv_primary null -> 0 before the == 0 test",
        "constants": "TICK = 64; CT, T = 3, 2; UNPLANTED_CAP_SECONDS = 116; RIFLE, M4, AK, AWP, SNIPER_OTHER, SMG, HEAVY as in lines 48-54",
        "implemented_in": "scripts/loadout_rule.py (copied) and the job bodies that call its aggregate()",
        "rule_differences_from_serving": ["dedupe on (round, tick, player_id) keep first (training rule; serving does not dedupe; 0 duplicates in the 2026-09 sample)",
                                          "rows whose side is not 2 or 3 are dropped (serving sends them to T; none seen)",
                                          "rows whose (match, round) is not in the export are dropped (serving scores every round in the tome)",
                                          "rows whose player has no team in basic_player_info for that match and round are dropped (serving drops them the same way)"],
    },
    "scope": {
        "export_split_codes": "match_split.npy code 0 = 47,501 matches (train), 1 = 4,633 (validation), 2 = 5,677 (test). The counts match export README.json; the names follow the counts.",
        "built": "train and validation codes only, path date before 2026-10-21",
        "not_built_test_split": "code 2, 5,677 matches (frozen test by export split; coverage counted only, no loadout built)",
        "not_built_protected_date": "train/validation keys with path date 2026-10-21 to 2026-10-31: 2,550 train + 298 validation = 2,848 matches. Holdout B is the test-assigned 653 matches with match dates Oct 21-25 (docs/eval_reports); the Oct 26-31 holdout is H. Membership was not checked (no holdout key file was opened). Excluded wholesale by date.",
        "owner_check_needed": "The docs say holdout B and the Oct 26-31 holdout have zero overlap with the training pool, but match_split.npy assigns 2,848 keys dated Oct 21-31 to train or validation. The owner should check these against the holdout manifests.",
        "status_counts": status_counts,
        "split_counts": split_counts,
        "split_by_protected_date": prot,
    },
    "sources": {
        "job1_bucket_channel": {"description": "S3 channel basic_player_status and basic_player_info, read on the space with boto3, dataframe pages of tome/csds in chunks c000-c062 (c063-c067 not read: their windows reach 2026-10-21 onward). Bucket keys with no local source.",
                                 "matches": 26521, "all_found": True, "chunks_read_first_last_count": chunks_read["job1_bucket"], "totals": {k: v for k, v in tot.items() if k.startswith("job1")}},
        "job2_local_covered_in_channel": {"description": "The same channel read for in-scope keys that the local caches also cover (Feb 2024 and 2026-09/10). Used where the channel holds the key.",
                                           "matches_found_in_channel": 16333, "matches_not_in_channel": 6432, "chunks_read_first_last_count": chunks_read["job2_local_covered_in_channel"], "totals": {k: v for k, v in tot.items() if k.startswith("job2")}},
        "local_1hz": {"description": "G: status_1hz (2026-09-01..2026-10-20) and the 2026-10-09..12 and 2026-10-20 files reduced to the 6,432 keys the channel does not hold. Feb 2024 local cache was used only as the parity reference, not as a source.",
                      "matches": 6432, "by_month": {"2026-09": 1771, "2026-10 days 01-08": 1209, "2026-10 days 09-12": 2920, "2026-10-20": 532},
                      "totals": local_tot, "reduced_inputs_used": ["2026-10-09", "2026-10-10", "2026-10-11", "2026-10-12", "2026-10-20"],
                      "full_file_rows_for_reference": {k: full_rows_by_file.get(k) for k in ["2026-10-09.parquet", "2026-10-10.parquet", "2026-10-11.parquet", "2026-10-12.parquet", "2026-10-20.parquet"]}},
        "export_inputs_sha256": {"match_keys.json": "e560a05fbe624fa4...(16 hex, first 16 shown)", "offsets.npy": "9b85f05f762289c4", "round_match.npy": "a07007bbb100fe6d", "round_freeze.npy": "fe11bc843825e004", "round_num.npy": "3f2dd60a0be32260", "match_split.npy": "47f097b768edd976"},
    },
    "gates": {
        "grid_and_structure": {"local_files": "every local file: rows_off_grid 0, rows_pre_freeze 0, rows_beyond_round_steps 0, rows_bad_side 0, duplicates 0 (see local_file_stats); 2026-10-20 reduced file same", "s3": "every chunk: rows_off_grid 0, rows_pre_freeze 0, rows_bad_side 0, rows_unrostered 0, info_roster_conflicts 0"},
        "parity_v3_channel_vs_local_2024": {"keys": parity["parity_v3"]["keys_compared"], "keys_all_steps_equal": parity["parity_v3"]["keys_all_steps_equal"], "side_steps_compared": parity["parity_v3"]["side_steps_compared"], "side_steps_mismatch": parity["parity_v3"]["side_steps_mismatch"]},
        "parity_v4_channel_vs_local_1hz_oct9_20": {"keys": parity["parity_v4"]["keys_compared"], "keys_all_steps_equal": parity["parity_v4"]["keys_all_steps_equal"], "side_steps_compared": parity["parity_v4"]["side_steps_compared"], "side_steps_mismatch": parity["parity_v4"]["side_steps_mismatch"]},
        "parity_rows_vs_written_arrays": verify.get("parity_rows_vs_written"),
        "per_match_hash_local_vs_space": hash_summary,
        "value_ranges_space_final": verify.get("ranges"),
    },
    "not_done_or_open": [
        "Serving run end to end on a built match (win_probability() with the built arrays) was not run. The rule was re-implemented in loadout_rule.py and checked against the serving code by reading and by the parity checks.",
        "Raw E: player_status check of the serving rule (gate 14a) was not run. Only the 1 Hz and channel paths were compared.",
        "16 Hz cross-check of the 1 Hz cache was not run.",
        "The loop bound was checked only on the unplanted cap (max 116 steps in the export for unplanted rounds). The planted bound (round end tick) was not recomputed; export steps are taken as the serving bound.",
        "Side is taken from the local side column for local keys and from basic_player_info for channel keys. Parity on 387 keys is exact, but side was not compared directly on all keys.",
        "Protected keys (2,848 train/val keys dated 2026-10-21..31; 5,677 test-split keys) have no loadout by design. Holdout membership was not verified.",
        "Channel rounds that the export does not hold (for example round 1 in many matches) are dropped; the export round set is the training set.",
        "No commit, push, RunPod, Meridian or S3 write was done. export_clean_v2 was not modified.",
    ],
    "deviations_from_decision_option_1": [
        "Local 1 Hz rows were not moved to the space in full (the link ran at about 0.26 MB/s). The 16,333 in-scope keys that the channel holds were built from the S3 channel, not from local 1 Hz. The channel was checked against the local build on 387 keys (zero mismatches).",
        "Feb 2024 keys were built from the channel (v3 writer), not from the local 2024 cache; the local 2024 cache was the parity reference (150 keys, zero mismatches).",
        "Local rows for Oct 9-12 and Oct 20 were reduced to the required keys before transfer. The aggregation depends only on each key's own rows, so the values are the same as the full-file build (checked by hash on 200 sampled keys).",
    ],
    "reproduce": "scripts/ holds loadout_rule.py (the rule), 01_space_body.py (local stage), s3_body.py (channel stage), 04_make_s3_job.py and 04b_make_s3_job2.py (key lists), 06_reduce.py, 07_parity_local.py, 09_verify_space_body.py, 10_readme.py. Hashes are listed in files.",
    "files": {},
}
for name in ["loadout_rule.py", "01_space_body.py", "s3_body.py", "04_make_s3_job.py", "04b_make_s3_job2.py", "06_reduce.py",
             "07_parity_local.py", "08_assemble.py", "09_verify_space_body.py", "09_make_verify.py", "10_readme.py"]:
    src = S + "/scripts2/" + name
    shutil.copy(src, FIN + "/scripts/" + name)
    readme["files"]["scripts/" + name] = {"sha256": sha(src)[:16], "bytes": os.path.getsize(src)}
for name, src in [("local_req_keys.json", S + "/out/local_req_keys.json"), ("s3_target2.json", S + "/out/s3_target2.json"), ("s3_key_lists.json", S + "/out/s3_key_lists.json")]:
    shutil.copy(src, FIN + "/scripts/" + name)
    readme["files"]["scripts/" + name] = {"sha256": sha(src)[:16], "bytes": os.path.getsize(src)}
json.dump(readme, open(FIN + "/README.json", "w"), indent=1)
print("README written; built matches", built.height, "built steps", steps_built, "rows with data", verify.get("rows_with_data_CT_T"))
print(json.dumps(hash_summary))
