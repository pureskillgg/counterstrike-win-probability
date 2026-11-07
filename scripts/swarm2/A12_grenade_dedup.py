"""A12: are the grenade_state detonation rows one-per-grenade? Compare detonations with throws.

Source tables (2026 tome, read-only):
  grenade_state rows with event_type in {flashbang_detonate, smokegrenade_detonate, hegrenade_detonate}
  weapon_fire rows with weapon_name in {flashbang, smokegrenade, hegrenade} (throws)
Checks:
  1. detonation rows vs throw rows, in total and per (match_key, round)
  2. per-round difference distribution (equal, detonations above throws, below)
  3. null grenade_id rows: do they share (match_key, round, entity_id) with a non-null row
  4. (match_key, round, entity_id) groups with more than one detonation: tick gaps
  5. tick_throw null rate on detonation rows
Output goes to stdout; the caller redirects to logs/swarm2/A12_grenade_dedup.log.
"""
from pathlib import Path

import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
KINDS = {
    "flashbang": ("flashbang_detonate", "flashbang"),
    "smokegrenade": ("smokegrenade_detonate", "smokegrenade"),
    "hegrenade": ("hegrenade_detonate", "hegrenade"),
}


def folder(ch):
    hits = sorted(p for p in TOME.iterdir() if p.is_dir() and p.name.startswith(f"basic_{ch}."))
    if len(hits) != 1:
        raise SystemExit(f"{ch}: expected one folder, got {[h.name for h in hits]}")
    return hits[0]


def scan(ch):
    fs = sorted(str(p) for p in folder(ch).glob("dataframe_*"))
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


def main():
    gs = scan("grenade_state").select(
        ["match_key", "round", "tick", "event_type", "entity_id", "grenade_id", "tick_throw"]).collect()
    fire = scan("weapon_fire").select(["match_key", "round", "tick", "player_id", "weapon_name"]).filter(
        pl.col("weapon_name").is_in(["flashbang", "smokegrenade", "hegrenade"])).collect()
    print(f"grenade_state rows: {gs.height:,}; throw rows (weapon_fire flashbang/smoke/HE): {fire.height:,}")

    for kind, (det_ev, thr_name) in KINDS.items():
        print(f"\n## {kind}")
        det = gs.filter(pl.col("event_type") == det_ev)
        thr = fire.filter(pl.col("weapon_name") == thr_name)
        print(f"  detonation rows = {det.height:,}; throw rows = {thr.height:,}; "
              f"detonations - throws = {det.height - thr.height:+,}")

        dpr = det.group_by(["match_key", "round"]).agg(pl.len().alias("det"))
        tpr = thr.group_by(["match_key", "round"]).agg(pl.len().alias("thr"))
        both = dpr.join(tpr, on=["match_key", "round"], how="full", coalesce=True).with_columns(
            pl.col("det").fill_null(0).cast(pl.Int64), pl.col("thr").fill_null(0).cast(pl.Int64))
        both = both.with_columns((pl.col("det") - pl.col("thr")).alias("diff"))
        n_rounds = both.height
        eq = both.filter(pl.col("diff") == 0).height
        pos = both.filter(pl.col("diff") > 0)
        neg = both.filter(pl.col("diff") < 0)
        print(f"  match-rounds with any detonation or throw: {n_rounds:,}; equal counts: {eq:,}; "
              f"detonations above throws: {pos.height:,} (sum {pos['diff'].sum():,}); "
              f"detonations below throws: {neg.height:,} (sum {neg['diff'].sum():,})")
        print("  largest positive diffs (match, round, det, thr):")
        for r in pos.sort("diff", descending=True).head(5).iter_rows(named=True):
            print(f"    {r['match_key']} r{r['round']} det={r['det']} thr={r['thr']}")

        # null grenade_id rows
        nul = det.filter(pl.col("grenade_id").is_null())
        nn = det.filter(pl.col("grenade_id").is_not_null())
        print(f"  null grenade_id detonations: {nul.height:,} ({100 * nul.height / max(det.height, 1):.2f}%)")
        share = nul.join(nn.select(["match_key", "round", "entity_id"]).unique(),
                         on=["match_key", "round", "entity_id"], how="inner")
        print(f"  null-id rows whose (match,round,entity_id) appears on a non-null detonation: {share.height:,}")
        share_tick = nul.join(nn.select(["match_key", "round", "entity_id", "tick"]),
                              on=["match_key", "round", "entity_id", "tick"], how="inner")
        print(f"  null-id rows with the same (match,round,entity_id,tick) on a non-null detonation: {share_tick.height:,}")

        # entity reuse groups
        g = det.sort(["match_key", "round", "entity_id", "tick"]).with_columns(
            (pl.col("tick") - pl.col("tick").shift(1).over(["match_key", "round", "entity_id"])).alias("gap"))
        multi = g.filter(pl.col("gap").is_not_null())
        print(f"  second-or-later detonation rows on the same (match,round,entity_id): {multi.height:,}")
        if multi.height:
            q = multi["gap"].quantile
            print(f"  tick gap to previous detonation on same entity: min={multi['gap'].min()} "
                  f"p05={q(0.05)} median={multi['gap'].median()} max={multi['gap'].max()}; "
                  f"gaps <= 64 ticks: {multi.filter(pl.col('gap') <= 64).height:,}")

        tn = det.filter(pl.col("tick_throw").is_null()).height
        print(f"  tick_throw null on detonation rows: {tn:,} ({100 * tn / max(det.height, 1):.2f}%)")


if __name__ == "__main__":
    main()
