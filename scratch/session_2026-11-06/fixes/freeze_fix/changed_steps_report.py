"""Old and new values for every changed step of the 283 confirmed freeze-tick rounds (read-only on v3 and v4).
Writes freeze_fix/changed_steps_v3_v4.parquet (one row per changed step, side and field) and a per-round summary CSV.
"""
import json
import numpy as np
import polars as pl

V2 = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
V3 = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
V4 = "/home/sagemaker-user/adam/builder/final/export_clean_v4/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
keys = json.load(open(V2 + "match_keys.json"))
kid = {k: i for i, k in enumerate(keys)}
off = np.load(V2 + "offsets.npy").astype(np.int64)
rf = np.load(V2 + "round_freeze.npy").astype(np.int64)
rm = np.load(V2 + "round_match.npy").astype(np.int64)
rn = np.load(V2 + "round_num.npy").astype(np.int64)
FIELDS19 = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
            "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy"]
fx = pl.read_csv(FX + "override_rounds.csv").filter(pl.col("status") == "fixed").select("match_key", "round")
ridx = []
for mk, rd in fx.rows():
    idx = np.nonzero((rm == kid[mk]) & (rn == int(rd)))[0]
    assert idx.size == 1
    ridx.append(int(idx[0]))
ridx = np.array(sorted(ridx), dtype=np.int64)
a16 = np.load(V3 + "loadout_step_i16.npy", mmap_mode="r")
b16 = np.load(V4 + "loadout_step_i16.npy", mmap_mode="r")
ae = np.load(V3 + "loadout_equip_i32.npy", mmap_mode="r")
be = np.load(V4 + "loadout_equip_i32.npy", mmap_mode="r")
rows = []
per_round = []
for r in ridx.tolist():
    s0, s1 = int(off[r]), int(off[r + 1])
    steps = np.arange(s0, s1, dtype=np.int64)
    x = np.asarray(a16[s0:s1]).astype(np.int64)
    y = np.asarray(b16[s0:s1]).astype(np.int64)
    xe = np.asarray(ae[s0:s1]).astype(np.int64)
    ye = np.asarray(be[s0:s1]).astype(np.int64)
    ch = (x != y).any(axis=2) | (xe != ye)
    n_changed = 0
    for si, sd in zip(*np.nonzero(ch)):
        step = int(steps[si])
        k = step - s0
        base = {"match_key": keys[int(rm[r])], "round": int(rn[r]), "step": step, "k": int(k), "tick": int(rf[r] + 64 * k),
                "side": "CT" if sd == 0 else "T"}
        for fi, fname in enumerate(FIELDS19):
            if x[si, sd, fi] != y[si, sd, fi]:
                rows.append({**base, "field": fname, "old": int(x[si, sd, fi]), "new": int(y[si, sd, fi])})
        if xe[si, sd] != ye[si, sd]:
            rows.append({**base, "field": "equip", "old": int(xe[si, sd]), "new": int(ye[si, sd])})
        n_changed += 1
    per_round.append({"match_key": keys[int(rm[r])], "round": int(rn[r]), "steps": s1 - s0, "steps_changed": n_changed})
D = pl.DataFrame(rows)
D.write_parquet(FX + "changed_steps_v3_v4.parquet")
pl.DataFrame(per_round).write_csv(FX + "changed_steps_per_round.csv")
summary = {"rounds": len(ridx), "steps": int(sum(p["steps"] for p in per_round)), "steps_changed": int(sum(p["steps_changed"] for p in per_round)),
           "field_rows": D.height, "rounds_with_changes": sum(1 for p in per_round if p["steps_changed"] > 0)}
json.dump(summary, open(FX + "changed_steps_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
print("DONE", flush=True)
