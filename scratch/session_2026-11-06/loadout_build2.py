"""Per-round loadouts at the freeze tick, per side, aligned to export_clean.

Sources (no S3 writes):
  bucket months : raw per-match player_status + player_info from S3 (download_file to NVMe scratch), rows at
                  tick == freeze_end, side from player_info (team_code 2 = T, 3 = CT).
  laptop months : laptop status_1hz (copied to the space), rows at second == 0 (= freeze_end), side from the file.
Per side, per round: alive, hp, armor_n, armor_sum, helmet_n, kit_n, defusing, ak, m4, awp, rifle, sniper_other, smg,
equip (sum of equipment_value_calc), money (sum of money). Weapon id sets as in wp_items.py.

Writes final/loadout_parts/*.parquet and export_clean/loadout_round.npy (N_rounds, 2 x 15) float32, CT block then T block,
plus loadout_names.json and loadout_report.json. Nothing is written to S3.
Column count: FEATS below has 15 names per side, so this version writes 2 x 15 = 30 columns, as the line above says.
The export_clean_v2 loadout_round.npy is 2 x 22 = 44 columns (186,329,568 bytes = 1,058,690 x 44 x 4 + 128 header bytes;
its loadout_names.json lists 22 per side). The 7 extra names per side (heavy, no_primary, flash, smoke, he, fire, decoy) are not
in this FEATS list, so that export file was not written by this version of the script.

  ~/venv-dsdk/bin/python loadout_build2.py bucket | laptop | merge
"""
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import polars as pl
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
BUILDER = os.path.expanduser("~/adam/builder")
FIN = BUILDER + "/final"
EXP = FIN + "/export_clean"
PARTS = FIN + "/loadout_parts"
SCR = "/mnt/sagemaker-nvme/loadout"
LAPTOP_STATUS = BUILDER + "/laptop_status/status_1hz"
T, CT = 2, 3
AK, M4, AWP = (7,), (16, 60), (9,)
RIFLE = (7, 8, 10, 13, 16, 39, 60)
SNIPER_OTHER = (11, 38, 40)
SMG = (17, 19, 23, 24, 26, 33, 34)
FEATS = ["alive", "hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
         "sniper_other", "smg", "equip", "money"]
MK_RE = r"(csds/\d{4}/\d{2}/\d{2}/[^/]+)/[^/]+$"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def side_agg(df):
    return df.group_by(["match_key", "round", "side"]).agg(
        pl.len().alias("alive"),
        pl.col("health").sum().alias("hp"),
        (pl.col("armor") > 0).cast(pl.Int32).sum().alias("armor_n"),
        pl.col("armor").sum().alias("armor_sum"),
        pl.col("has_helmet").cast(pl.Int32).sum().alias("helmet_n"),
        pl.col("has_defuser").cast(pl.Int32).sum().alias("kit_n"),
        pl.col("is_defusing").cast(pl.Int32).sum().alias("defusing"),
        pl.col("inv_primary").is_in(list(AK)).cast(pl.Int32).sum().alias("ak"),
        pl.col("inv_primary").is_in(list(M4)).cast(pl.Int32).sum().alias("m4"),
        pl.col("inv_primary").is_in(list(AWP)).cast(pl.Int32).sum().alias("awp"),
        pl.col("inv_primary").is_in(list(RIFLE)).cast(pl.Int32).sum().alias("rifle"),
        pl.col("inv_primary").is_in(list(SNIPER_OTHER)).cast(pl.Int32).sum().alias("sniper_other"),
        pl.col("inv_primary").is_in(list(SMG)).cast(pl.Int32).sum().alias("smg"),
        pl.col("equipment_value_calc").sum().alias("equip"),
        pl.col("money").sum().alias("money"),
    )


def wanted_rounds():
    cr = pl.read_parquet(FIN + "/clean_rounds.parquet", columns=["match_key", "round", "freeze", "use", "src"])
    return cr.filter(pl.col("use")).unique(["match_key", "round"])


def fetch_file(args):
    cli, folder, name, root = args
    dst = os.path.join(root, folder, name)
    if os.path.exists(dst):
        return "ok"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        cli.download_file(BUCKET, f"data/{folder}/{name}", dst + ".part", Config=TransferConfig(max_concurrency=4))
    except Exception as e:
        if os.path.exists(dst + ".part"):
            os.remove(dst + ".part")
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        return "missing" if code in ("404", "NoSuchKey", "NotFound") else "failed"
    os.replace(dst + ".part", dst)
    return "ok"


def stage_bucket():
    os.makedirs(PARTS, exist_ok=True)
    cli = boto3.client("s3", config=Config(max_pool_connections=64, retries={"max_attempts": 8},
                                           request_checksum_calculation="when_required",
                                           response_checksum_validation="when_required"))
    wr = wanted_rounds().filter(pl.col("src") == "bucket")
    bkeys = set(wr["match_key"].unique().to_list())
    log("bucket usable matches", len(bkeys), "rounds", wr.height)
    for c in [f"c{i:03d}" for i in range(68)]:
        mk = pl.read_parquet(f"{BUILDER}/data/{c}/matches.parquet", columns=["match_key"])["match_key"].to_list()
        keys = [m for m in mk if m in bkeys]
        out = f"{PARTS}/bucket_{c}.parquet"
        if not keys or os.path.exists(out):
            continue
        t0 = time.time()
        root = f"{SCR}/{c}"
        jobs = []
        for m in keys:
            folder = m.rsplit("/", 1)[0]
            jobs += [(cli, folder, "player_status", root), (cli, folder, "player_info", root)]
        with ThreadPoolExecutor(32) as ex:
            res = list(ex.map(fetch_file, jobs))
        miss = {k: res.count(k) for k in ("ok", "missing", "failed")}
        ps_files = glob.glob(f"{root}/csds/*/*/*/*/player_status")
        pi_files = glob.glob(f"{root}/csds/*/*/*/*/player_info")
        rt = wr.filter(pl.col("match_key").is_in(keys)).select(
            "match_key", pl.col("round").cast(pl.Int64), pl.col("freeze").cast(pl.Int64)).lazy()
        ps = (pl.scan_parquet(ps_files, include_file_paths="src", extra_columns="ignore")
              .with_columns(pl.col("src").str.extract(MK_RE, 1).add(pl.lit("/csds")).alias("match_key"),
                            pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
              .join(rt, on=["match_key", "round"], how="inner")
              .filter(pl.col("tick") == pl.col("freeze"), pl.col("health") > 0))
        pi = (pl.scan_parquet(pi_files, include_file_paths="src", extra_columns="ignore")
              .with_columns(pl.col("src").str.extract(MK_RE, 1).add(pl.lit("/csds")).alias("match_key"),
                            pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
              .select("match_key", "round", "player_id", "team_code")
              .unique(["match_key", "round", "player_id"], keep="last"))
        df = (ps.with_columns(pl.col("player_id").cast(pl.Int64))
              .join(pi, on=["match_key", "round", "player_id"], how="left")
              .with_columns(pl.col("team_code").cast(pl.Int64).alias("side"))
              .filter(pl.col("side").is_in([T, CT])).collect())
        agg = side_agg(df)
        agg.write_parquet(out)
        log(c, "matches", len(keys), "fetch", miss, "player-rows", df.height, "agg rows", agg.height,
            f"secs={time.time() - t0:.0f}")
        shutil.rmtree(root, ignore_errors=True)


def stage_laptop():
    os.makedirs(PARTS, exist_ok=True)
    wr = wanted_rounds().filter(pl.col("src") == "laptop").select("match_key", pl.col("round").cast(pl.Int64))
    log("laptop usable rounds", wr.height)
    files = sorted(glob.glob(f"{LAPTOP_STATUS}/*.parquet"))
    st = (pl.scan_parquet(files)
          .filter(pl.col("second") == 0, pl.col("health") > 0)
          .with_columns(pl.col("round").cast(pl.Int64))
          .join(wr.lazy(), on=["match_key", "round"], how="inner")
          .select("match_key", "round", "side", "health", "armor", "has_helmet", "has_defuser", "is_defusing",
                  "money", "equipment_value_calc", "inv_primary"))
    df = st.collect()
    if df["side"].dtype == pl.String:
        df = df.with_columns(pl.when(pl.col("side") == "CT").then(CT).when(pl.col("side") == "T").then(T)
                             .otherwise(None).cast(pl.Int64).alias("side"))
    df = df.with_columns(pl.col("side").cast(pl.Int64))
    log("laptop status rows at freeze", df.height, "| side values", sorted(df["side"].unique().to_list()))
    side_agg(df).write_parquet(f"{PARTS}/laptop.parquet")
    log("laptop part written")


def stage_merge():
    parts = sorted(glob.glob(f"{PARTS}/bucket_c*.parquet")) + [f"{PARTS}/laptop.parquet"]
    A = pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed")
    dup = A.group_by(["match_key", "round", "side"]).len().filter(pl.col("len") > 1).height
    log("aggregated rows", A.height, "| duplicate (match, round, side):", dup)
    keys = json.load(open(EXP + "/match_keys.json"))
    rm = np.load(EXP + "/round_match.npy")
    rn = np.load(EXP + "/round_num.npy")
    grid = pl.DataFrame({"match_key": [keys[i] for i in rm.tolist()],
                         "round": rn.astype(np.int64)}).with_row_index("ridx")
    names = []
    J = grid
    for side, code in (("CT", CT), ("T", T)):
        S = A.filter(pl.col("side") == code).select(["match_key", "round"] + FEATS)
        S = S.rename({f: f"{side}_{f}" for f in FEATS}).with_columns(pl.col("round").cast(pl.Int64))
        J = J.join(S, on=["match_key", "round"], how="left")
        names += [f"{side}_{f}" for f in FEATS]
    both = J.select((pl.col("CT_alive").is_not_null() & pl.col("T_alive").is_not_null()).alias("ok"))["ok"]
    log("export rounds", J.height, "| with both sides", int(both.sum()), "| missing", int((~both).sum()))
    out = np.zeros((J.height, len(names)), dtype=np.float32)
    ridx = J["ridx"].to_numpy()
    for j, nm in enumerate(names):
        col = J[nm].to_numpy().astype(np.float32)
        out[ridx, j] = np.nan_to_num(col, nan=0.0)
    np.save(EXP + "/loadout_round.npy", out)
    json.dump(names, open(EXP + "/loadout_names.json", "w"))
    json.dump({"rounds": J.height, "both_sides": int(both.sum()), "missing": int((~both).sum()), "dup": dup},
              open(EXP + "/loadout_report.json", "w"), indent=1)
    log("wrote loadout_round.npy", out.shape, f"{out.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    {"bucket": stage_bucket, "laptop": stage_laptop, "merge": stage_merge}[sys.argv[1]]()
