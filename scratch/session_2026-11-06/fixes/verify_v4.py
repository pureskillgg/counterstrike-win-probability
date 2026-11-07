# Verifies v4 against v3: built rows unchanged, every changed step inside a new match, plus coverage and freeze-state sanity counts.
import json, re
import numpy as np
import polars as pl

V3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
V4S = V4 + "scripts/"
EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
LOGS = "/home/sagemaker-user/adam/builder/logs/"

off = np.load(EXP + "offsets.npy").astype(np.int64)
rm = np.load(EXP + "round_match.npy").astype(np.int64)
NS = int(off[-1])
st = pl.read_csv(V4 + "match_loadout_status.csv", schema_overrides={"match_key": pl.String})
new = st.filter(pl.col("v4_batch").is_not_null())
new_m = new["m"].to_numpy().astype(np.int64)
round_new = np.isin(rm, new_m)
old_built_m = st.filter(pl.col("v4_batch").is_null() & pl.col("build_status").is_in(["s3_channel", "local_1hz"]))["m"].to_numpy().astype(np.int64)
round_old = np.isin(rm, old_built_m)
print("new matches", new_m.size, "| new rounds", int(round_new.sum()), "| v3-built rounds", int(round_old.sum()), "| all rounds", rm.size)

diff = np.zeros(NS + 1, dtype=np.int64)
np.add.at(diff, off[:-1][round_new], 1)
np.add.at(diff, off[1:][round_new], -1)
exp_mask = np.cumsum(diff[:-1]) > 0
print("steps inside new rounds", int(exp_mask.sum()))

a16 = np.load(V3 + "loadout_step_i16.npy", mmap_mode="r")
b16 = np.load(V4 + "loadout_step_i16.npy", mmap_mode="r")
ae = np.load(V3 + "loadout_equip_i32.npy", mmap_mode="r")
be = np.load(V4 + "loadout_equip_i32.npy", mmap_mode="r")
assert a16.shape == b16.shape == (NS, 2, 19) and ae.shape == be.shape == (NS, 2)
B = 2_000_000
n_changed = n_changed_outside = n_new_nonzero = 0
for s0 in range(0, NS, B):
    s1 = min(NS, s0 + B)
    a = np.asarray(a16[s0:s1])
    b = np.asarray(b16[s0:s1])
    eb = np.asarray(be[s0:s1])
    c = (a != b).any(axis=(1, 2)) | (np.asarray(ae[s0:s1]) != eb).any(axis=1)
    ex = exp_mask[s0:s1]
    n_changed += int(c.sum())
    n_changed_outside += int((c & ~ex).sum())
    n_new_nonzero += int((((b != 0).any(axis=(1, 2)) | (eb != 0).any(axis=1)) & ex).sum())
    del a, b, eb
print("steps changed v3->v4", n_changed, "| changed outside new rounds", n_changed_outside,
      "| nonzero steps inside new rounds", n_new_nonzero)

conf = {"chunks": 0, "info_roster_conflicts": 0, "rows_unrostered": 0, "rows_bad_side": 0, "rows_pre_freeze": 0,
        "rows_off_grid": 0, "rows_beyond_round_steps": 0, "rows_no_export_round": 0, "duplicates_dropped": 0,
        "target_agg_rows_written": 0, "matches_found": 0}
target_line = None
for line in open(LOGS + "v4_s3_channel.log", encoding="utf-8"):
    mm = re.search(r"(\{.*\})\s*$", line)
    if mm and '"chunk"' in mm.group(1):
        dd = json.loads(mm.group(1))
        conf["chunks"] += 1
        for k in conf:
            if k != "chunks" and k in dd:
                conf[k] += dd[k]
    if "TARGET keys" in line:
        target_line = line.strip()
print("S3 log totals", conf)
print("S3 target line:", target_line)

cov_tbl = new.group_by(["v4_batch", "build_status"]).agg([
    pl.len().alias("matches"), (pl.col("rows") > 0).sum().alias("matches_with_rows"),
    pl.col("rounds_with_rows").sum().alias("rounds_with_rows"),
    pl.col("steps_with_rows").sum().alias("steps_with_rows"), pl.col("rows").sum().alias("rows")]).sort("v4_batch")
print(cov_tbl)


def freeze_stats(rmask, label):
    rr = np.nonzero(rmask)[0]
    sf = off[rr]
    hp_ct = np.asarray(b16[sf, 0, 0]).astype(np.int64)
    hp_t = np.asarray(b16[sf, 1, 0]).astype(np.int64)
    both = (hp_ct > 0) & (hp_t > 0)
    res = {"rounds": int(rr.size), "both_sides_hp_gt0": int(both.sum()),
           "both_sides_hp_zero": int(((hp_ct == 0) & (hp_t == 0)).sum()),
           "any_side_hp_gt500": int(((hp_ct > 500) | (hp_t > 500)).sum()),
           "mean_hp_ct_when_both_gt0": round(float(hp_ct[both].mean()), 1) if both.any() else None,
           "mean_hp_t_when_both_gt0": round(float(hp_t[both].mean()), 1) if both.any() else None}
    print("freeze-step sanity", label, res)
    return res


fs_new = freeze_stats(round_new, "new rounds (v4)")
fs_old = freeze_stats(round_old, "v3-built rounds (comparison)")
out = {"new_matches": int(new_m.size), "new_rounds": int(round_new.sum()), "steps_changed_v3_to_v4": n_changed,
       "steps_changed_outside_new_rounds": n_changed_outside, "nonzero_steps_inside_new_rounds": n_new_nonzero,
       "s3_log_totals": conf, "s3_target_line": target_line, "coverage_by_batch": cov_tbl.to_dicts(),
       "freeze_sanity_new": fs_new, "freeze_sanity_v3_built": fs_old}
json.dump(out, open(V4S + "verify_v4.json", "w"), indent=1, default=str)
print("verify written", V4S + "verify_v4.json")
