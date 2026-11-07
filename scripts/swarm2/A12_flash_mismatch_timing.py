"""A12: test whether flashbang detonation-minus-throw round mismatches come from events after round_end.

Hypothesis (ASSUMPTION, untested before this script): a grenade thrown near round end can detonate after
round_end, so the detonation and the throw land in different rounds, which gives mismatches of both signs.
Test: count detonations and throws whose tick is after the round's round_end tick, and see whether they
account for the per-round mismatches.
Output goes to stdout; the caller redirects to logs/swarm2/A12_flash_mismatch_timing.log.
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
    gs = scan("grenade_state").select(["match_key", "round", "tick", "event_type"]).filter(
        pl.col("event_type") == "flashbang_detonate").collect()
    fire = scan("weapon_fire").select(["match_key", "round", "tick", "weapon_name"]).filter(
        pl.col("weapon_name") == "flashbang").collect()
    re_ = scan("round_end").select(["match_key", "round", "tick"]).collect()
    print(f"round_end rows: {re_.height:,}; distinct (match,round): "
          f"{re_.select(['match_key', 'round']).n_unique():,}; rows per (match,round) max: "
          f"{re_.group_by(['match_key', 'round']).agg(pl.len()).select(pl.col('len').max()).item()}")
    end = re_.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("end_tick")).with_columns(
        pl.col("end_tick").cast(pl.Int64))

    det = gs.with_columns(pl.col("tick").cast(pl.Int64)).join(end, on=["match_key", "round"], how="left")
    thr = fire.with_columns(pl.col("tick").cast(pl.Int64)).join(end, on=["match_key", "round"], how="left")
    print(f"detonations with no round_end row in their round: {det.filter(pl.col('end_tick').is_null()).height:,}; "
          f"throws with no round_end row: {thr.filter(pl.col('end_tick').is_null()).height:,}")
    det_after = det.filter(pl.col("tick") > pl.col("end_tick")).height
    thr_after = thr.filter(pl.col("tick") > pl.col("end_tick")).height
    print(f"flashbang detonations after round_end tick: {det_after:,} of {det.height:,}")
    print(f"flashbang throws after round_end tick: {thr_after:,} of {thr.height:,}")

    dpr = det.group_by(["match_key", "round"]).agg(pl.len().cast(pl.Int64).alias("det"),
                                                   (pl.col("tick") > pl.col("end_tick")).sum().cast(pl.Int64).alias("det_after"))
    tpr = thr.group_by(["match_key", "round"]).agg(pl.len().cast(pl.Int64).alias("thr"),
                                                   (pl.col("tick") > pl.col("end_tick")).sum().cast(pl.Int64).alias("thr_after"))
    j = dpr.join(tpr, on=["match_key", "round"], how="full", coalesce=True).with_columns(
        pl.col("det").fill_null(0), pl.col("thr").fill_null(0),
        pl.col("det_after").fill_null(0), pl.col("thr_after").fill_null(0)).with_columns(
        diff=pl.col("det") - pl.col("thr"))
    mis = j.filter(pl.col("diff") != 0)
    up = mis.filter(pl.col("diff") > 0)
    dn = mis.filter(pl.col("diff") < 0)
    print(f"\nmatch-rounds with a mismatch: {mis.height:,} (above {up.height:,}, below {dn.height:,})")
    print(f"sum of positive diffs: {up['diff'].sum():+,}; sum of negative diffs: {dn['diff'].sum():+,}; "
          f"net {mis['diff'].sum():+,}")
    print(f"mismatch rounds where the mismatch equals the after-round_end count on the same side:")
    up_match = up.filter(pl.col("diff") == pl.col("det_after")).height
    dn_match = dn.filter((-pl.col("diff")) == pl.col("thr_after")).height
    print(f"  above-side: detonations after round_end equal the excess in {up_match:,} of {up.height:,} rounds")
    print(f"  below-side: throws after round_end equal the deficit in {dn_match:,} of {dn.height:,} rounds")
    print(f"  above-side rounds with any detonation after round_end: {up.filter(pl.col('det_after') > 0).height:,}")
    print(f"  below-side rounds with any throw after round_end: {dn.filter(pl.col('thr_after') > 0).height:,}")


if __name__ == "__main__":
    main()
