"""A12: where do the null grenade_id detonation rows sit? Matches, rounds, dates, and whether they are mixed
with non-null rows in the same match.
Covers flashbang_detonate, smokegrenade_detonate and hegrenade_detonate.
Output goes to stdout; the caller redirects to logs/swarm2/A12_null_grenade_id.log.
"""
from pathlib import Path

import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
EVENTS = ["flashbang_detonate", "smokegrenade_detonate", "hegrenade_detonate"]


def folder(ch):
    hits = sorted(p for p in TOME.iterdir() if p.is_dir() and p.name.startswith(f"basic_{ch}."))
    if len(hits) != 1:
        raise SystemExit(f"{ch}: expected one folder, got {[h.name for h in hits]}")
    return hits[0]


def main():
    fs = sorted(str(p) for p in folder("grenade_state").glob("dataframe_*"))
    gs = pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed").select(
        ["match_key", "round", "tick", "event_type", "entity_id", "grenade_id"]).collect()
    det = gs.filter(pl.col("event_type").is_in(EVENTS)).with_columns(
        date=pl.col("match_key").str.slice(5, 10),  # csds/2026/10/DD/... -> "2026/10" (see below)
        day=pl.col("match_key").str.split("/").list.get(3),
        is_null=pl.col("grenade_id").is_null(),
    )
    print(f"detonation rows (3 kinds): {det.height:,}; null grenade_id: {det.filter(pl.col('is_null')).height:,}")

    print("\n## per event_type: null rows, matches with any null row, matches with null rows and non-null rows")
    for ev in EVENTS:
        sub = det.filter(pl.col("event_type") == ev)
        nul = sub.filter(pl.col("is_null"))
        nn = sub.filter(~pl.col("is_null"))
        m_null = nul["match_key"].n_unique()
        m_mixed = len(set(nul["match_key"].unique().to_list()) & set(nn["match_key"].unique().to_list()))
        print(f"  {ev:<22} null={nul.height:>6,} ({100 * nul.height / max(sub.height, 1):5.2f}%) "
              f"matches_with_null={m_null:,} matches_with_null_and_nonnull={m_mixed:,}")

    fl = det.filter(pl.col("event_type") == "flashbang_detonate")
    print("\n## flashbang null rows by day (match_key date segment)")
    by_day = (fl.group_by("day").agg(pl.len().alias("rows"), pl.col("is_null").sum().alias("null_rows"),
                                     pl.col("match_key").n_unique().alias("matches"))
              .with_columns((pl.col("null_rows") / pl.col("rows") * 100).round(2).alias("null_pct"))
              .sort("day"))
    print(by_day)

    # rounds: all-null vs mixed
    rr = fl.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("is_null").sum().alias("k"))
    all_null = rr.filter(pl.col("k") == pl.col("n")).height
    mixed = rr.filter((pl.col("k") > 0) & (pl.col("k") < pl.col("n"))).height
    none_null = rr.filter(pl.col("k") == 0).height
    print(f"\nflashbang match-rounds: all detonations null-id={all_null:,}; mixed={mixed:,}; no null-id={none_null:,}")
    tot_null_rounds_rows = rr.filter(pl.col("k") > 0)["n"].sum()
    print(f"rows in match-rounds that contain any null-id row: {tot_null_rounds_rows:,} of {fl.height:,}")


if __name__ == "__main__":
    main()
