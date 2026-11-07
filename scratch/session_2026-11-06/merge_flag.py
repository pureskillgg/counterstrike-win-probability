"""Merge the loadout parts into export_clean/loadout_round.npy, with a per-round flag for duplicated player ids at freeze.

Flag rule: a round is flagged (loadout_flag = 1) when any player_id appears more than once in the status rows at the freeze
tick. In those rounds the side attribution from the status rows is ambiguous, so the 15 features per side are set to 0.
The alive counts are kept as recorded, so rounds with a player missing from the record (fewer than 5 alive) stay unflagged.
Writes export_clean/loadout_round.npy, loadout_flag.npy, loadout_names.json, loadout_report.json. Nothing goes to S3.
"""
import glob
import json
import os

import numpy as np
import polars as pl

BUILDER = os.path.expanduser("~/adam/builder")
EXP = BUILDER + "/final/export_clean"
PARTS = BUILDER + "/final/loadout_parts"
CT, T = 3, 2
FEATS = ["alive", "hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
         "sniper_other", "smg", "equip", "money"]

parts = sorted(glob.glob(f"{PARTS}/bucket_c*.parquet")) + [f"{PARTS}/laptop.parquet"]
A = pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed")
keys = json.load(open(EXP + "/match_keys.json"))
rm = np.load(EXP + "/round_match.npy")
rn = np.load(EXP + "/round_num.npy")
grid = pl.DataFrame({"match_key": [keys[i] for i in rm.tolist()], "round": rn.astype(np.int64)}).with_row_index("ridx")

names = []
J = grid
for side, code in (("CT", CT), ("T", T)):
    S = A.filter(pl.col("side") == code).select(["match_key", "round"] + FEATS)
    S = S.rename({f: f"{side}_{f}" for f in FEATS}).with_columns(pl.col("round").cast(pl.Int64))
    J = J.join(S, on=["match_key", "round"], how="left")
    names += [f"{side}_{f}" for f in FEATS]
dup = (A.group_by(["match_key", "round"]).agg(pl.col("dup_rows").sum().alias("dup_rows"))
       .with_columns(pl.col("round").cast(pl.Int64)))
J = J.join(dup, on=["match_key", "round"], how="left").with_columns(pl.col("dup_rows").fill_null(0))
both = J.select((pl.col("CT_alive").is_not_null() & pl.col("T_alive").is_not_null()).alias("ok"))["ok"]
flag = (J["dup_rows"].to_numpy() > 0)
print("export rounds", J.height, "| both sides present", int(both.sum()), "| flagged duplicate-id rounds", int(flag.sum()), flush=True)

out = np.zeros((J.height, len(names)), dtype=np.float32)
ridx = J["ridx"].to_numpy()
for j, nm in enumerate(names):
    col = J[nm].to_numpy().astype(np.float32)
    out[ridx, j] = np.nan_to_num(col, nan=0.0)
out[ridx[flag], :] = 0.0
flag_out = np.zeros(J.height, dtype=np.uint8)
flag_out[ridx] = flag.astype(np.uint8)

np.save(EXP + "/loadout_round.npy", out)
np.save(EXP + "/loadout_flag.npy", flag_out)
json.dump(names, open(EXP + "/loadout_names.json", "w"))
alive_ct = out[:, names.index("CT_alive")]
alive_t = out[:, names.index("T_alive")]
unflagged = flag_out == 0
rep = {"rounds": int(J.height), "flagged_duplicate_id": int(flag_out.sum()),
       "unflagged_both_5": int((unflagged & (alive_ct == 5) & (alive_t == 5)).sum()),
       "unflagged_fewer_than_5_one_side": int((unflagged & ((alive_ct < 5) | (alive_t < 5))).sum()),
       "unflagged_more_than_5_one_side": int((unflagged & ((alive_ct > 5) | (alive_t > 5))).sum())}
json.dump(rep, open(EXP + "/loadout_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1), flush=True)
