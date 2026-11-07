"""Per-round loadouts at the freeze tick, per side, aligned to export_clean.

Sources (no S3 writes):
  bucket months : raw per-match player_status + player_info from S3 (GET via download_file to NVMe scratch),
                  rows at tick == freeze_end, side from player_info (team_code 2 = T, 3 = CT).
  laptop months : laptop status_1hz (copied to the space), rows at second == 0 (= freeze_end), side in the file.
Per side, per round: alive, hp, armor_n, armor_sum, helmet_n, kit_n, defusing, ak, m4, awp, rifle, sniper_other, smg,
equip (sum of equipment_value_calc), money (sum of money). Same weapon id sets as wp_items.py.

Writes final/loadout_parts/*.parquet and export_clean/loadout_round.npy (N_rounds, 2 x 15) float32, CT block then T block,
plus export_clean/loadout_names.json. Nothing is written to S3.

  ~/venv-dsdk/bin/python loadout_build.py bucket | laptop | merge
"""
import glob
import json
import os
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
    """df: one row per living player at the freeze tick, with match_key, round, side and the status fields."""
    return df.group_by(["match_key", "round", "side"]).agg(
        pl.len().alias("alive"),
        pl.col("health").sum().alias("hp"),
        (pl.col("armor") > 0).sum().alias("armor_n"),
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
        if not keys:
            continue
        out = f"{PARTS}/bucket_{c}.parquet"
        if os.path.exists(out):
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
        rt = wr.filter(pl.col("match_key").is_in(keys)).select("match_key", "round", "freeze")
        ps = (pl.scan_parquet(ps_files, include_file_paths="src", extra_columns="ignore")
              .with_columns(pl.col("src").str.extract(MK_RE, 1).add(pl.lit("/csds")).alias("match_key"),
                            pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
              .join(rt.lazy().with_columns(pl.col("round").cast(pl.Int64), pl.col("freeze").cast(pl.Int64)),
                    on=["match_key", "round"], how="inner")
              .filter(pl.col("tick") == pl.col("freeze")).filter(pl.col("health") > 0))
        pi = (pl.scan_parquet(pi_files, include_file_paths="src", extra_columns="ignore")
              .with_columns(pl.col("src").str.extract(MK_RE, 1).add(pl.lit("/csds")).alias("match_key"),
                            pl.col("round").cast(pl.Int64))
              .select("match_key", "round", "player_id", "team_code").unique(["match_key", "round", "player_id"], keep="last"))
        df = (ps.join(pi, on=["match_key", "round", "player_id"], how="left")
              .with_columns(pl.col("team_code").alias("side"))
              .filter(pl.col("side").is_in([T, CT])).collect())
        agg = side_agg(df)
        agg.write_parquet(out)
        log(c, "matches", len(keys), "fetch", miss, "rows", df.height, "agg rows", agg.height, f"secs={time.time() - t0:.0f}")
        # keep NVMe small: the raw files of a finished chunk are no longer needed
        for d in glob.glob(f"{root}/csds/*/*/*/*"):
            shutil.rmtree(d, ignore_errors=True) if False else None
        os.system(f"rm -rf {root}")


def stage_laptop():
    os.makedirs(PARTS, exist_ok=True)
    wr = wanted_rounds().filter(pl.col("src") == "laptop")
    log("laptop usable rounds", wr.height)
    files = sorted(glob.glob(f"{LAPTOP_STATUS}/*.parquet"))
    st = (pl.scan_parquet(files)
          .filter(pl.col("second") == 0).filter(pl.col("health") > 0)
          .select("match_key", "round", "side", "health", "armor", "has_helmet", "has_defuser", "is_defusing",
                  "money", "equipment_value_calc", "inv_primary")
          .with_columns(pl.col("round").cast(pl.Int64)))
    wr2 = wr.select("match_key", pl.col("round").cast(pl.Int64))
    df = st.join(wr2.lazy(), on=["match_key", "round"], how="inner").collect()
    log("laptop status rows at freeze", df.height, "| side values", df["side"].unique().to_list())
    side_agg(df.with_columns(pl.col("side").cast(pl.Int64))).write_parquet(f"{PARTS}/laptop.parquet")
    log("laptop parts written")


def stage_merge():
    parts = sorted(glob.glob(f"{PARTS}/bucket_c*.parquet")) + [f"{PARTS}/laptop.parquet"]
    A = pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed")
    # one row per (match, round, side)
    dup = A.group_by(["match_key", "round", "side"]).len().filter(pl.col("len") > 1).height
    log("aggregated rows", A.height, "| duplicate (match, round, side):", dup)
    wide = A.with_columns(pl.when(pl.col("side") == CT).then(pl.lit("CT")).otherwise(pl.lit("T")).alias("sd"))
    W = wide.select(["match_key", "round", "sd"] + FEATS)
    W = W.pivot(on="sd", index=["match_key", "round"], values=FEATS)
    names = [f"{s}_{f}" for s in ("CT", "T") for f in FEATS]
    keys = json.load(open(EXP + "/match_keys.json"))
    rm = np.load(EXP + "/round_match.npy")
    rn = np.load(EXP + "/round_num.npy")
    grid = pl.DataFrame({"match_id": rm.astype(np.int64), "round": rn.astype(np.int64)}).with_columns(
        pl.col("match_id").map_elements(lambda i: keys[i], return_dtype=pl.String).alias("match_key"),
        pl.arange(0, rm.shape[0], eager=True).cast(pl.Int64).alias("ridx"))
    J = grid.join(W.with_columns(pl.col("round").cast(pl.Int64)), on=["match_key", "round"], how="left")
    have = J.select(pl.col("CT_alive").is_not_null() & pl.col("T_alive").is_not_null()).to_series()
    log("export rounds", J.height, "| with both sides", int(have.sum()), "| missing", int((~have).sum()))
    out = np.zeros((J.height, 2 * len(FEATS)), dtype=np.float32)
    for j, nm in enumerate(names):
        col = J[f"{nm}"].to_numpy() if f"{nm}" in J.columns else None
        if col is not None:
            out[:, j] = np.nan_to_num(col.astype(np.float32), nan=0.0)
    np.save(EXP + "/loadout_round.npy", out)
    json.dump(names, open(EXP + "/loadout_names.json", "w"))
    json.dump({"rounds": J.height, "both_sides": int(have.sum()), "missing": int((~have).sum()), "dup": dup},
              open(EXP + "/loadout_report.json", "w"), indent=1)
    log("wrote loadout_round.npy", out.shape, f"{out.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    import shutil  # noqa: F401  (kept for the per-chunk cleanup path)
    stage = sys.argv[1]
    {"bucket": stage_bucket, "laptop": stage_laptop, "merge": stage_merge}[stage]()
