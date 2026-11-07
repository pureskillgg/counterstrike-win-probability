"""A12: flashbang detonation minus throw difference by match day, split by null-grenade_id matches.
Shows whether the +62 excess (detonations above throws) sits in the batch where grenade_id is null.
Output goes to stdout; the caller redirects to logs/swarm2/A12_flash_by_day.log.
"""
from pathlib import Path

import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"


def folder(ch):
    hits = sorted(p for p in TOME.iterdir() if p.is_dir() and p.name.startswith(f"basic_{ch}."))
    if len(hits) != 1:
        raise SystemExit(f"{ch}: expected one folder, got {[h.name for h in hits]}")
    return hits[0]


def scan(ch):
    fs = sorted(str(p) for p in folder(ch).glob("dataframe_*"))
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


def main():
    gs = scan("grenade_state").select(["match_key", "event_type", "grenade_id", "round"]).collect()
    fire = scan("weapon_fire").select(["match_key", "round", "weapon_name"]).filter(
        pl.col("weapon_name") == "flashbang").collect()
    det = gs.filter(pl.col("event_type") == "flashbang_detonate")
    # per-match null flag: a match is null-batch when every detonation row has null grenade_id
    per_match = det.group_by("match_key").agg(
        pl.col("grenade_id").null_count().alias("nulls"), pl.len().alias("n"))
    per_match = per_match.with_columns(null_batch=pl.col("nulls") == pl.col("n"))
    day_of = lambda s: s.str.split("/").list.get(3)  # noqa: E731
    d_all = det.group_by(["match_key", "round"]).agg(pl.len().alias("det"))
    t_all = fire.group_by(["match_key", "round"]).agg(pl.len().alias("thr"))
    j = d_all.join(t_all, on=["match_key", "round"], how="full", coalesce=True).with_columns(
        pl.col("det").fill_null(0).cast(pl.Int64), pl.col("thr").fill_null(0).cast(pl.Int64),
    ).join(per_match.select(["match_key", "null_batch"]), on="match_key", how="left").with_columns(
        day=day_of(pl.col("match_key")), diff=pl.col("det") - pl.col("thr"))
    print("## by day: detonations, throws, difference, match-rounds with diff > 0")
    out = (j.group_by("day").agg(
        pl.col("det").sum().alias("det"), pl.col("thr").sum().alias("thr"),
        pl.col("diff").sum().alias("diff"), (pl.col("diff") > 0).sum().alias("rounds_above"),
        pl.col("null_batch").first().alias("null_batch_match")).sort("day"))
    print(out)
    nb = j.filter(pl.col("null_batch") == True)  # noqa: E712
    nn = j.filter(pl.col("null_batch") == False)  # noqa: E712
    print(f"\nnull-id batch matches: rounds={nb.height:,} det={nb['det'].sum():,} thr={nb['thr'].sum():,} "
          f"diff={nb['diff'].sum():+,}")
    print(f"populated-id matches:  rounds={nn.height:,} det={nn['det'].sum():,} thr={nn['thr'].sum():,} "
          f"diff={nn['diff'].sum():+,}")
    print(f"total: det={j['det'].sum():,} thr={j['thr'].sum():,} diff={j['diff'].sum():+,}")


if __name__ == "__main__":
    main()
