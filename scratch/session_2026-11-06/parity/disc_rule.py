"""Mid-round disconnect and bot-takeover rule, AWS window (Oct 9 to Oct 25), active duty, non-wingman matches.

For each round: a player_disconnect or bot_takeover event counts as mid-round when its tick is in [freeze, end_eff).
Events before freeze or at/after end_eff are pre or post. Reads S3 (get_object only) and rounds.parquet on the space.
Writes disc_rule_summary.json and disc_rule_rounds.parquet next to this script. Writes nothing to S3.
"""
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import polars as pl
import pyarrow.parquet as pq
from botocore.config import Config

BUCKET = "pureskillgg-csgo-production-sagemaker-0"
COMB = os.path.expanduser("~/adam/status/data/combined")
HERE = os.path.expanduser("~/adam/status_scripts/parity")
POOL = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]
LO, HI = "2026/10/09", "2026/10/25"
s3 = boto3.client("s3", config=Config(max_pool_connections=128, retries={"max_attempts": 5}))


def get_events(mk):
    folder = mk.rsplit("/", 1)[0]
    rows, err = [], 0
    for ch in ("player_disconnect", "bot_takeover"):
        try:
            body = s3.get_object(Bucket=BUCKET, Key=f"data/{folder}/{ch}")["Body"].read()
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code not in ("NoSuchKey", "404"):
                err += 1
            continue
        try:
            t = pq.read_table(io.BytesIO(body), columns=["round", "tick"]).to_pylist()
        except Exception:
            err += 1
            continue
        for r in t:
            rows.append((mk, ch, r["round"], r["tick"]))
    return rows, err


def main():
    H = pl.read_parquet(os.path.expanduser("~/adam/status_scripts/parity/per_match_headers_space.parquet"))
    H = H.filter(pl.col("status") == "ok")
    H = H.with_columns(pl.Series("day", ["/".join(k.split("/")[1:4]) for k in H["match_key"].to_list()]))
    H = H.filter((pl.col("day") >= LO) & (pl.col("day") <= HI) & (~pl.col("is_wingman")) & pl.col("map_name").is_in(POOL))
    mks = H["match_key"].to_list()
    print("matches in scope:", len(mks), flush=True)

    with ThreadPoolExecutor(64) as ex:
        res = list(ex.map(get_events, mks))
    ev = [r for rows, _ in res for r in rows]
    errs = sum(e for _, e in res)
    E = pl.DataFrame(ev, schema=["match_key", "ch", "round", "tick"], orient="row").with_columns(
        pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    print("events:", E.height, "read errors:", errs, flush=True)
    print("events by channel:", E.group_by("ch").agg(pl.len().alias("n")).rows(), flush=True)

    R = pl.read_parquet(os.path.join(COMB, "rounds.parquet"),
                        columns=["match_key", "round", "era", "n_t", "n_ct", "use", "round_status", "freeze", "end_eff",
                                 "disc_or_bot", "dup_keeper", "in_scope", "channels_complete", "roster_ok",
                                 "player_count_norm"])
    R = R.filter((pl.col("era") == "2026") & pl.col("match_key").is_in(mks)).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("freeze").cast(pl.Int64), pl.col("end_eff").cast(pl.Int64))
    print("rounds in scope:", R.height, flush=True)

    EV = E.join(R.select(["match_key", "round", "freeze", "end_eff"]), on=["match_key", "round"], how="inner")
    EV = EV.with_columns(
        ((pl.col("tick") >= pl.col("freeze")) & (pl.col("tick") < pl.col("end_eff"))).alias("mid"),
        (pl.col("tick") < pl.col("freeze")).alias("pre"),
    )
    flags = EV.group_by(["match_key", "round"]).agg(
        ((pl.col("ch") == "player_disconnect") & pl.col("mid")).any().alias("mid_disc"),
        ((pl.col("ch") == "bot_takeover") & pl.col("mid")).any().alias("mid_bot"),
        ((pl.col("ch") == "player_disconnect") & pl.col("pre")).any().alias("pre_disc"),
        ((pl.col("ch") == "player_disconnect") & ~pl.col("mid") & ~pl.col("pre")).any().alias("post_disc"),
        ((pl.col("ch") == "bot_takeover") & ~pl.col("mid")).any().alias("nonmid_bot"),
        pl.len().alias("n_events"),
    )
    RF = R.join(flags, on=["match_key", "round"], how="left").with_columns(
        [pl.col(c).fill_null(False) for c in ["mid_disc", "mid_bot", "pre_disc", "post_disc", "nonmid_bot"]])

    # validation: every disc_or_bot round should have an event; no event round should lack the flag
    val_missing = RF.filter(pl.col("disc_or_bot") & (pl.col("n_events").fill_null(0) == 0)).height
    val_extra = RF.filter((~pl.col("disc_or_bot")) & (pl.col("n_events").fill_null(0) > 0)).height
    print("validation: disc_or_bot rounds with no event:", val_missing, "| event rounds without flag:", val_extra, flush=True)

    std = ((pl.col("n_t") == 5) & (pl.col("n_ct") <= 5)) | ((pl.col("n_ct") == 5) & (pl.col("n_t") <= 5))
    OK_STATUS = ["ok", "player_count != 10", "disconnect or bot takeover"]
    clean = pl.col("dup_keeper") & pl.col("in_scope") & pl.col("channels_complete")
    keep_new = clean & std & pl.col("round_status").is_in(OK_STATUS) & ~pl.col("mid_disc") & ~pl.col("mid_bot")
    RF = RF.with_columns(keep_new.alias("keep_new"), std.alias("std"),
                         ((pl.col("n_t") == 5) & (pl.col("n_ct") == 5)).alias("fv"))

    out = {}
    out["matches"] = len(mks)
    out["rounds_in_scope"] = RF.height
    out["validation_disc_no_event"] = val_missing
    out["validation_event_no_flag"] = val_extra
    out["baseline_use_rounds"] = RF.filter(pl.col("use")).height
    out["baseline_use_matches"] = RF.filter(pl.col("use"))["match_key"].n_unique()
    out["new_keep_rounds_all_patterns"] = RF.filter(pl.col("keep_new")).height
    out["new_keep_matches_all_patterns"] = RF.filter(pl.col("keep_new"))["match_key"].n_unique()
    out["new_keep_rounds_5v5"] = RF.filter(pl.col("keep_new") & pl.col("fv")).height
    out["new_keep_matches_5v5"] = RF.filter(pl.col("keep_new") & pl.col("fv"))["match_key"].n_unique()
    out["new_keep_rounds_by_status"] = RF.filter(pl.col("keep_new")).group_by("round_status").agg(pl.len().alias("n")).rows()
    out["standard_clean_rounds_not_kept_by_reason"] = RF.filter(clean & std & ~pl.col("keep_new")).with_columns(
        pl.when(pl.col("mid_bot")).then(pl.lit("mid_bot"))
        .when(pl.col("mid_disc")).then(pl.lit("mid_disc"))
        .otherwise(pl.col("round_status")).alias("reason")).group_by("reason").agg(pl.len().alias("n")).sort("n", descending=True).rows()
    out["disc_or_bot_rounds_timing"] = RF.filter(pl.col("disc_or_bot")).select(
        pl.col("mid_disc").sum().alias("mid_disc"), pl.col("mid_bot").sum().alias("mid_bot"),
        pl.col("pre_disc").sum().alias("pre_disc"), pl.col("post_disc").sum().alias("post_disc"),
        pl.col("nonmid_bot").sum().alias("nonmid_bot"), pl.len().alias("rounds")).rows()
    out["player_count_not10_rounds_roster_ok"] = RF.filter(pl.col("round_status") == "player_count != 10").group_by("roster_ok").agg(pl.len().alias("n")).rows()
    out["player_count_not10_keep_new"] = RF.filter((pl.col("round_status") == "player_count != 10") & pl.col("keep_new")).height

    with open(os.path.join(HERE, "disc_rule_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    RF.write_parquet(os.path.join(HERE, "disc_rule_rounds.parquet"))
    for k, v in out.items():
        print(k, ":", v, flush=True)
    print("wrote disc_rule_summary.json", flush=True)


if __name__ == "__main__":
    main()
