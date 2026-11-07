"""Read-only: are flagged duplicate-id rows identical records? Is a short-roster player recorded elsewhere in its match?"""
import json, os
import boto3
import numpy as np
import polars as pl
from botocore.config import Config

B = "pureskillgg-csgo-production-sagemaker-0"
cli = boto3.client("s3", config=Config(request_checksum_calculation="when_required", response_checksum_validation="when_required"))
E = os.path.expanduser("~/adam/builder/final/export_clean")
keys = json.load(open(E + "/match_keys.json"))
rm = np.load(E + "/round_match.npy")
rn = np.load(E + "/round_num.npy")
X = np.load(E + "/loadout_round.npy", mmap_mode="r")
names = json.load(open(E + "/loadout_names.json"))
fl = np.load(E + "/loadout_flag.npy")
ca = np.asarray(X[:, names.index("CT_alive")])
ta = np.asarray(X[:, names.index("T_alive")])
fr = pl.read_parquet(os.path.expanduser("~/adam/builder/final/clean_rounds.parquet"),
                     columns=["match_key", "round", "freeze"]).unique(["match_key", "round"])


def matches_with(mask, n):
    out = {}
    for i in np.where(mask)[0]:
        out.setdefault(int(rm[i]), []).append(int(rn[i]))
    return list(out.items())[:n]


dup_matches = matches_with(fl == 1, 6)
short_matches = matches_with((fl == 0) & ((ca < 5) | (ta < 5)), 6)
root = "/mnt/sagemaker-nvme/chk"
cols = ["health", "armor", "has_helmet", "has_defuser", "money", "equipment_value_calc", "inv_primary", "is_defusing"]


def pull(mk):
    folder = mk.rsplit("/", 1)[0]
    out = {}
    for name in ["player_status", "player_info"]:
        d = f"{root}/{folder}/{name}"
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if not os.path.exists(d):
            cli.download_file(B, f"data/{folder}/{name}", d)
        out[name] = pl.read_parquet(d).with_columns(pl.col("round").cast(pl.Int64))
    return out


same = diff = 0
print("== flagged duplicate-id rounds: identical records or different players?", flush=True)
for mid, rs in dup_matches:
    mk = keys[mid]
    d = pull(mk)
    for r in rs[:4]:
        fz = fr.filter((pl.col("match_key") == mk) & (pl.col("round") == r))["freeze"][0]
        f = d["player_status"].filter((pl.col("round") == r) & (pl.col("tick") == fz))
        dups = f.group_by("player_id").len().filter(pl.col("len") > 1)["player_id"].to_list()
        for pid in dups:
            rows = f.filter(pl.col("player_id") == pid).select(cols).unique()
            if rows.height == 1:
                same += 1
            else:
                diff += 1
        if dups and r == rs[0]:
            roster = d["player_info"].filter((pl.col("round") == r) & (pl.col("player_id").is_in(dups))) \
                .select("player_id", "team_code").sort("player_id").rows()
            print(mk[-12:], "round", r, "dup ids", dups, "roster (id, team)", roster[:8], flush=True)
print("dup ids checked:", same + diff, "| identical rows:", same, "| different rows:", diff, flush=True)

print("== short-roster rounds: is the missing player recorded elsewhere in the same match?", flush=True)
for mid, rs in short_matches:
    mk = keys[mid]
    d = pull(mk)
    pi = d["player_info"]
    ps = d["player_status"]
    r = rs[0]
    rost = set(pi.filter(pl.col("round") == r)["player_id"].to_list())
    all_ids = set(pi["player_id"].to_list())
    rsize = pi.group_by("round").agg(pl.col("player_id").n_unique().alias("n"))["n"].to_list()
    at_freeze = set(ps.filter(pl.col("round") == r)["player_id"].to_list())
    print(mk[-12:], "round", r, "roster", len(rost), "| distinct ids in match", len(all_ids),
          "| roster sizes across the match", sorted(set(rsize)),
          "| ids in match but not in this roster", sorted(all_ids - rost)[:6],
          "| of those seen in this round's status", sorted((all_ids - rost) & at_freeze), flush=True)
