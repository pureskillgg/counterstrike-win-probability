"""Breakdown of the non-use rounds kept by the disconnect rule (keep_new and not use), AWS window, active duty, non-wingman.

Re-reads disconnect and bot-takeover events from S3 (get_object only), uses disc_rule_rounds.parquet. Writes nothing to S3.
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
HERE = os.path.expanduser("~/adam/status_scripts/parity")
s3 = boto3.client("s3", config=Config(max_pool_connections=128, retries={"max_attempts": 5}))


def get_events(mk):
    folder = mk.rsplit("/", 1)[0]
    rows = []
    for ch in ("player_disconnect", "bot_takeover"):
        try:
            body = s3.get_object(Bucket=BUCKET, Key=f"data/{folder}/{ch}")["Body"].read()
        except Exception:
            continue
        for r in pq.read_table(io.BytesIO(body), columns=["round", "tick"]).to_pylist():
            rows.append((mk, ch, r["round"], r["tick"]))
    return rows


def main():
    RF = pl.read_parquet(os.path.join(HERE, "disc_rule_rounds.parquet"))
    K = RF.filter(pl.col("keep_new") & (~pl.col("use")))
    mks = K["match_key"].unique().to_list()
    print("kept non-use rounds:", K.height, "matches:", len(mks), flush=True)
    print("by round_status:", K.group_by("round_status").agg(pl.len().alias("n")).rows(), flush=True)
    print("by side pattern (n_t,n_ct) top:", K.group_by(["n_t", "n_ct"]).agg(pl.len().alias("n")).sort("n", descending=True).head(8).rows(), flush=True)

    with ThreadPoolExecutor(64) as ex:
        ev = [r for rows in ex.map(get_events, mks) for r in rows]
    E = pl.DataFrame(ev, schema=["match_key", "ch", "round", "tick"], orient="row").with_columns(
        pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    E = E.join(K.select(["match_key", "round", "freeze", "end_eff"]), on=["match_key", "round"], how="inner")
    E = E.with_columns(
        pl.when(pl.col("tick") < pl.col("freeze")).then(pl.lit("pre"))
        .when(pl.col("tick") >= pl.col("end_eff")).then(pl.lit("post"))
        .otherwise(pl.lit("mid")).alias("when"))

    cls = E.group_by(["match_key", "round", "ch", "when"]).agg(pl.len().alias("n"), pl.col("tick").n_unique().alias("ticks"))
    per = cls.group_by(["match_key", "round"]).agg(
        ((pl.col("ch") == "player_disconnect") & (pl.col("when") == "pre")).any().alias("pre_disc"),
        ((pl.col("ch") == "player_disconnect") & (pl.col("when") == "post")).any().alias("post_disc"),
        ((pl.col("ch") == "bot_takeover") & (pl.col("when") == "pre")).any().alias("pre_bot"),
        ((pl.col("ch") == "bot_takeover") & (pl.col("when") == "post")).any().alias("post_bot"),
        ((pl.col("ch") == "player_disconnect") & (pl.col("n") >= 2)).any().alias("multi_disc_same_event"),
    )
    KB = K.join(per, on=["match_key", "round"], how="left").with_columns(
        [pl.col(c).fill_null(False) for c in ["pre_disc", "post_disc", "pre_bot", "post_bot", "multi_disc_same_event"]])
    print("event timing among kept non-use rounds (rounds):", KB.select(
        pl.col("pre_disc").sum().alias("pre_disc"), pl.col("post_disc").sum().alias("post_disc"),
        pl.col("pre_bot").sum().alias("pre_bot"), pl.col("post_bot").sum().alias("post_bot"),
        pl.col("multi_disc_same_event").sum().alias("multi_disc_same_event")).rows(), flush=True)

    # persistence: a bot takeover in an earlier round of the same match
    bot_rounds = E.filter(pl.col("ch") == "bot_takeover").select(["match_key", "round"]).unique().rename({"round": "bot_round"})
    first_bot = bot_rounds.group_by("match_key").agg(pl.col("bot_round").min().alias("first_bot_round"))
    KB = KB.join(first_bot, on="match_key", how="left").with_columns(
        (pl.col("first_bot_round").is_not_null() & (pl.col("first_bot_round") < pl.col("round"))).alias("earlier_bot_in_match"))
    print("kept non-use rounds with a bot takeover in an earlier round of the same match:",
          KB.filter(pl.col("earlier_bot_in_match")).height, flush=True)
    print("of which the kept round itself has no event:", KB.filter(pl.col("earlier_bot_in_match") & ~pl.col("pre_disc")
          & ~pl.col("post_disc") & ~pl.col("pre_bot") & ~pl.col("post_bot")).height, flush=True)

    pc = KB.filter(pl.col("round_status") == "player_count != 10")
    print("player-count rounds kept by player_count_norm:", pc.group_by("player_count_norm").agg(pl.len().alias("n")).sort("player_count_norm").rows(), flush=True)
    print("player-count rounds kept by roster_ok:", pc.group_by("roster_ok").agg(pl.len().alias("n")).rows(), flush=True)
    dc = KB.filter(pl.col("round_status") == "disconnect or bot takeover")
    print("disconnect-or-bot kept rounds: pre_disc", dc.filter(pl.col("pre_disc")).height, "post_disc only",
          dc.filter(pl.col("post_disc") & ~pl.col("pre_disc")).height, "bot events (pre or post)",
          dc.filter(pl.col("pre_bot") | pl.col("post_bot")).height, flush=True)

    out = {"kept_nonuse_rounds": K.height, "matches": len(mks),
           "earlier_bot_in_match": KB.filter(pl.col("earlier_bot_in_match")).height,
           "pre_disc_rounds": KB.filter(pl.col("pre_disc")).height,
           "post_disc_rounds": KB.filter(pl.col("post_disc")).height,
           "pre_bot_rounds": KB.filter(pl.col("pre_bot")).height,
           "post_bot_rounds": KB.filter(pl.col("post_bot")).height,
           "multi_disc_same_event_rounds": KB.filter(pl.col("multi_disc_same_event")).height}
    with open(os.path.join(HERE, "disc_breakdown_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote disc_breakdown_summary.json", flush=True)


if __name__ == "__main__":
    main()
