"""Why are some rounds not 5 alive at freeze (fewer, and more than 5)? Read-only diagnostic.

Re-fetches a few raw matches from S3 (download_file, to NVMe scratch) and reads the laptop status_1hz for laptop rounds.
Compares, for each problem round: the player_info roster per side, the status rows at the freeze tick, and the status rows
one second before and after. Writes nothing to S3 and nothing to the export.
"""
import json
import os
import shutil

import boto3
import numpy as np
import polars as pl
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
BUILDER = os.path.expanduser("~/adam/builder")
FIN = BUILDER + "/final"
EXP = FIN + "/export_clean"
SCR = "/mnt/sagemaker-nvme/diag"
LAPTOP_STATUS = BUILDER + "/laptop_status/status_1hz"
T, CT = 2, 3


def export_rounds():
    keys = json.load(open(EXP + "/match_keys.json"))
    rm = np.load(EXP + "/round_match.npy")
    rn = np.load(EXP + "/round_num.npy")
    X = np.load(EXP + "/loadout_round.npy")
    names = json.load(open(EXP + "/loadout_names.json"))
    K = pl.DataFrame({"match_key": [keys[i] for i in rm.tolist()], "round": rn.astype(np.int64),
                      "CT_alive": X[:, names.index("CT_alive")].astype(np.int64),
                      "T_alive": X[:, names.index("T_alive")].astype(np.int64)})
    src = (pl.read_parquet(FIN + "/clean_rounds.parquet", columns=["match_key", "round", "src"])
           .unique(["match_key", "round"]))
    return K.join(src, on=["match_key", "round"], how="left")


def fetch(cli, folder, name, root):
    dst = os.path.join(root, folder, name)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cli.download_file(BUCKET, f"data/{folder}/{name}", dst, Config=TransferConfig(max_concurrency=4))
    return dst


def describe_bucket_match(cli, mk, rounds_here):
    folder = mk.rsplit("/", 1)[0]
    root = f"{SCR}/{abs(hash(mk)) % 10**9}"
    ps_path = fetch(cli, folder, "player_status", root)
    pi_path = fetch(cli, folder, "player_info", root)
    ps = pl.read_parquet(ps_path).with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    pi = pl.read_parquet(pi_path).with_columns(pl.col("round").cast(pl.Int64))
    out = []
    for r, label in rounds_here:
        fz = pl.read_parquet(FIN + "/clean_rounds.parquet", columns=["match_key", "round", "freeze"]).filter(
            (pl.col("match_key") == mk) & (pl.col("round") == r))["freeze"][0]
        roster = pi.filter(pl.col("round") == r).select("player_id", "team_code").unique("player_id", keep="last")
        side_count = roster.group_by("team_code").len().rows()
        at_fz = ps.filter((pl.col("round") == r) & (pl.col("tick") == fz))
        near = {d: ps.filter((pl.col("round") == r) & (pl.col("tick") == fz + d)).select("player_id", "health")
                for d in (-64, 64)}
        alive_fz = at_fz.filter(pl.col("health") > 0)
        ids_fz = set(at_fz["player_id"].to_list())
        roster_ids = set(roster["player_id"].to_list())
        missing_at_fz = sorted(roster_ids - ids_fz)
        rows_in_round = ps.filter(pl.col("round") == r)
        out.append({
            "case": label, "match": mk[-40:], "round": r, "freeze": fz,
            "roster_by_team": side_count, "roster_n": roster.height,
            "status_rows_at_freeze": at_fz.height, "alive_at_freeze": alive_fz.height,
            "roster_ids_missing_at_freeze": missing_at_fz,
            "missing_ids_seen_at_freeze-1s": sorted(set(missing_at_fz) & set(near[-64]["player_id"].to_list())),
            "missing_ids_seen_at_freeze+1s": sorted(set(missing_at_fz) & set(near[64]["player_id"].to_list())),
            "status_ids_not_in_roster": sorted(set(ids_fz) - roster_ids),
            "distinct_ids_in_round": rows_in_round["player_id"].n_unique(),
            "status_rows_per_tick_at_freeze_hp0": at_fz.filter(pl.col("health") <= 0).height,
        })
    return out


def describe_laptop_round(mk, r, label):
    files = sorted(__import__("glob").glob(f"{LAPTOP_STATUS}/*.parquet"))
    st = pl.scan_parquet(files).filter((pl.col("match_key") == mk) & (pl.col("round") == r)).select(
        "second", "player_id", "side", "health").collect()
    s0 = st.filter(pl.col("second") == 0)
    s1 = st.filter(pl.col("second") == 1)
    return {"case": label, "match": mk[-40:], "round": r,
            "second0_rows": s0.height, "second0_by_side": s0.group_by("side").len().rows(),
            "second0_hp0": s0.filter(pl.col("health") <= 0).height,
            "second1_rows": s1.height, "second1_by_side": s1.group_by("side").len().rows(),
            "ids_seen_only_at_second1": sorted(set(s1["player_id"].to_list()) - set(s0["player_id"].to_list()))[:10]}


def main():
    K = export_rounds()
    low = K.filter((pl.col("CT_alive") < 5) | (pl.col("T_alive") < 5))
    high = K.filter((pl.col("CT_alive") > 5) | (pl.col("T_alive") > 5))
    print("low rounds", low.height, "| by src", low.group_by("src").len().rows(), flush=True)
    print("high rounds", high.height, "| by src", high.group_by("src").len().rows(), flush=True)
    print("low: CT<5", int((K["CT_alive"] < 5).sum()), "T<5", int((K["T_alive"] < 5).sum()), flush=True)
    print("CT alive distribution at freeze:", K.group_by("CT_alive").len().sort("CT_alive").rows(), flush=True)
    print("T alive distribution at freeze:", K.group_by("T_alive").len().sort("T_alive").rows(), flush=True)
    print("high-side detail, CT>5:", K.filter(pl.col("CT_alive") > 5).group_by("CT_alive").len().sort("CT_alive").rows(),
          "T>5:", K.filter(pl.col("T_alive") > 5).group_by("T_alive").len().sort("T_alive").rows(), flush=True)

    cli = boto3.client("s3", config=Config(max_pool_connections=16, retries={"max_attempts": 5},
                                           request_checksum_calculation="when_required",
                                           response_checksum_validation="when_required"))
    bl = low.filter(pl.col("src") == "bucket").sort(["match_key", "round"]).head(60)
    bh = high.filter(pl.col("src") == "bucket").sort(["match_key", "round"]).head(60)
    lap_l = low.filter(pl.col("src") == "laptop").sort(["match_key", "round"]).head(6)
    lap_h = high.filter(pl.col("src") == "laptop").sort(["match_key", "round"]).head(6)

    def group(df):
        d = {}
        for row in df.iter_rows(named=True):
            d.setdefault(row["match_key"], []).append((row["round"], f"CT{row['CT_alive']}_T{row['T_alive']}"))
        return list(d.items())[:6]

    print("\n=== bucket, fewer than 5 alive ===", flush=True)
    for mk, rounds in group(bl):
        for rec in describe_bucket_match(cli, mk, rounds):
            print(json.dumps(rec, default=str), flush=True)
    print("\n=== bucket, more than 5 alive ===", flush=True)
    for mk, rounds in group(bh):
        for rec in describe_bucket_match(cli, mk, rounds):
            print(json.dumps(rec, default=str), flush=True)
    print("\n=== laptop, fewer than 5 alive ===", flush=True)
    for row in lap_l.iter_rows(named=True):
        print(json.dumps(describe_laptop_round(row["match_key"], row["round"], f"CT{row['CT_alive']}_T{row['T_alive']}")), flush=True)
    print("\n=== laptop, more than 5 alive ===", flush=True)
    for row in lap_h.iter_rows(named=True):
        print(json.dumps(describe_laptop_round(row["match_key"], row["round"], f"CT{row['CT_alive']}_T{row['T_alive']}")), flush=True)
    shutil.rmtree(SCR, ignore_errors=True)


if __name__ == "__main__":
    main()
