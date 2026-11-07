"""Does player_id_fixed or player_controller_id separate the players who share player_id at freeze? Read-only."""
import glob
import json
import os

import numpy as np
import polars as pl

E = os.path.expanduser("~/adam/builder/final/export_clean")
keys = json.load(open(E + "/match_keys.json"))
rm = np.load(E + "/round_match.npy")
rn = np.load(E + "/round_num.npy")
fl = np.load(E + "/loadout_flag.npy")
fr = pl.read_parquet(os.path.expanduser("~/adam/builder/final/clean_rounds.parquet"),
                     columns=["match_key", "round", "freeze"]).unique(["match_key", "round"])
flag_rounds = {}
for i in np.where(fl == 1)[0]:
    flag_rounds.setdefault(int(rm[i]), []).append(int(rn[i]))
root = "/mnt/sagemaker-nvme/chk"
tot = {"rounds": 0, "pid_dup_rounds": 0, "fixed_dup_rounds": 0, "ctrl_dup_rounds": 0, "fixed_dup_ids": 0}
for mid, rs in list(flag_rounds.items())[:12]:
    mk = keys[mid]
    folder = mk.rsplit("/", 1)[0]
    ps_path = f"{root}/{folder}/player_status"
    pi_path = f"{root}/{folder}/player_info"
    if not (os.path.exists(ps_path) and os.path.exists(pi_path)):
        continue
    ps = pl.read_parquet(ps_path).with_columns(pl.col("round").cast(pl.Int64))
    pi = pl.read_parquet(pi_path).with_columns(pl.col("round").cast(pl.Int64))
    for r in rs:
        fz = fr.filter((pl.col("match_key") == mk) & (pl.col("round") == r))["freeze"][0]
        f = ps.filter((pl.col("round") == r) & (pl.col("tick") == fz) & (pl.col("health") > 0))
        tot["rounds"] += 1
        if f.group_by("player_id").len().filter(pl.col("len") > 1).height:
            tot["pid_dup_rounds"] += 1
        if f.group_by("player_id_fixed").len().filter(pl.col("len") > 1).height:
            tot["fixed_dup_rounds"] += 1
        if f.group_by("player_controller_id").len().filter(pl.col("len") > 1).height:
            tot["ctrl_dup_rounds"] += 1
        ri = pi.filter(pl.col("round") == r)
        tot["fixed_dup_ids"] += ri.group_by("player_id_fixed").len().filter(pl.col("len") > 1).height
print("sample", tot, flush=True)
if tot["rounds"]:
    sample_mk = None
    for mid, rs in list(flag_rounds.items())[:12]:
        mk = keys[mid]
        folder = mk.rsplit("/", 1)[0]
        if os.path.exists(f"{root}/{folder}/player_status"):
            sample_mk = (mk, rs[0]); break
    mk, r = sample_mk
    folder = mk.rsplit("/", 1)[0]
    ps = pl.read_parquet(f"{root}/{folder}/player_status").with_columns(pl.col("round").cast(pl.Int64))
    pi = pl.read_parquet(f"{root}/{folder}/player_info").with_columns(pl.col("round").cast(pl.Int64))
    fz = fr.filter((pl.col("match_key") == mk) & (pl.col("round") == r))["freeze"][0]
    f = ps.filter((pl.col("round") == r) & (pl.col("tick") == fz) & (pl.col("health") > 0))
    print("example", mk[-12:], "round", r, flush=True)
    print(f.select("player_id", "player_id_fixed", "player_controller_id", "money", "inv_primary").sort("player_id").to_pandas().to_string()
          if False else f.select("player_id", "player_id_fixed", "player_controller_id", "money", "inv_primary").sort("player_id"), flush=True)
    print("roster", pi.filter(pl.col("round") == r).select("player_id", "player_id_fixed", "team_code").sort("player_id"), flush=True)
