"""Recover the victim side for 2024 deaths whose side is null in deaths.parquet, then rerun the alive-count gate on the same seeded sample.

Diagnosis (gate_2024_residual.py, gate_2024_vanish_check.py): the gate counts a death toward a side only when that side is known, so a
death with a null side is missing from the expected count and the gate sees one player too few. The raw death table has no team code
for those events (player_team_code and player_id_fixed are null), but the victim's position is in it.

Recovery: for each null-side death, find the raw player_death row at the same round and tick with the same (x, y), then the
player_vector row at that round and tick with the same (x_pos, y_pos); its team_code is the victim's side (2 = T, 3 = CT).
Deaths that cannot be matched stay null.

Writes data/diagnostics/gate_2024_side_recovery.json.
Run in the cs2gpu environment (polars, pyarrow):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/gate_2024_side_recovery.py
"""
import glob
import json
import os
import random
import sys
import time

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from gate_2024_residual import checked_table  # noqa: E402

ROOT = "G:/pureskill/csds-cs2"
RAW = "E:/csds_extracted/csds"
OUT = os.path.join(ROOT, "data", "diagnostics", "gate_2024_side_recovery.json")


RAW_FALLBACK = "E:/csds_recovered_20261011/csds"


def raw_dir(mk):
    p = mk.split("/")
    for root in (RAW, RAW_FALLBACK):
        d = os.path.join(root, p[1], p[2], p[3], p[4])
        if os.path.isdir(d):
            return d
    return os.path.join(RAW, p[1], p[2], p[3], p[4])


def recover_sides(mk, nulls):
    """nulls: rows (round, tick) with a null side for this match. Returns {(round, tick, player_id): side or None}."""
    d = raw_dir(mk)
    out = {}
    if not os.path.isdir(d):
        return {k: None for k in nulls}
    de = pq.read_table(os.path.join(d, "player_death"), columns=["round", "tick", "player_id", "player_x_pos", "player_y_pos"]).to_pylist()
    pv = pq.read_table(os.path.join(d, "player_vector"), columns=["round", "tick", "x_pos", "y_pos", "team_code"]).to_pylist()
    by_rt_death = {}
    for r in de:
        by_rt_death.setdefault((r["round"], r["tick"]), []).append(r)
    by_rt_pv = {}
    for r in pv:
        by_rt_pv.setdefault((r["round"], r["tick"]), []).append(r)
    for (rd, tick, pid) in nulls:
        cand = by_rt_death.get((rd, tick), [])
        side = None
        for c in cand:
            xy = (c["player_x_pos"], c["player_y_pos"])
            if xy[0] is None or xy[1] is None:
                continue
            hits = [p for p in by_rt_pv.get((rd, tick), []) if p["x_pos"] is not None and abs(p["x_pos"] - xy[0]) < 0.01
                    and abs(p["y_pos"] - xy[1]) < 0.01 and p["team_code"] in (2, 3)]
            teams = {h["team_code"] for h in hits}
            if len(teams) == 1:
                side = teams.pop()
                break
        out[(rd, tick, pid)] = side
    return out


def main():
    t0 = time.time()
    days = sorted(glob.glob(ROOT + "/data/positions_1hz_2024/2024-*.parquet"))
    pos = pl.concat([pl.read_parquet(d, columns=["match_key", "round", "tick", "second", "player_id", "side"]) for d in days])
    per_match = pos.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n"))
    five = sorted(per_match.filter(pl.col("n") == 10)["match_key"].to_list())
    random.seed(20241)
    sample = random.sample(five, 100)
    P = pos.filter(pl.col("match_key").is_in(sample))
    fe = (P.with_columns((pl.col("tick") - 64 * pl.col("second")).alias("fe"))
            .group_by(["match_key", "round"]).agg(pl.col("fe").min().alias("fe")))
    D = pl.read_parquet(ROOT + "/data/combined/deaths.parquet").filter(pl.col("match_key").is_in(sample))
    D = D.join(fe, on=["match_key", "round"], how="inner").with_columns(
        ((pl.col("tick") - pl.col("fe") + 63) // 64).clip(lower_bound=0).cast(pl.Int64).alias("second"),
        pl.col("side").cast(pl.Int64).alias("victim_side"))
    snap = P.select(["match_key", "round", "second"]).unique()
    positions = P.select(["match_key", "round", "second", pl.col("side").cast(pl.Int64), "player_id"])

    before = checked_table(positions, D.select(["match_key", "round", "second", "victim_side"]), snap, "ceil")
    agree_before = 1 - before.filter(pl.col("diff") != 0).height / before.height

    null_rows = D.filter(pl.col("victim_side").is_null())
    print(f"deaths in sample {D.height:,}; null side {null_rows.height:,} in {null_rows['match_key'].n_unique()} matches", flush=True)
    recovered = {}
    for mk in null_rows["match_key"].unique().to_list():
        sub = null_rows.filter(pl.col("match_key") == mk).select(["round", "tick", "player_id"]).rows()
        nulls = [(int(r), int(t), int(p) if p is not None else -1) for r, t, p in sub]
        res = recover_sides(mk, [(r, t, p) for r, t, p in nulls])
        for (r, t, p), s in res.items():
            recovered[(mk, r, t, p)] = s
    n_rec = sum(1 for v in recovered.values() if v is not None)
    print(f"recovered sides for {n_rec:,} of {len(recovered):,} null-side deaths ({time.time() - t0:.0f} s)", flush=True)

    rec_df = pl.DataFrame({"match_key": [k[0] for k in recovered], "round": [k[1] for k in recovered], "tick": [k[2] for k in recovered],
                           "player_id": [k[3] for k in recovered], "side_rec": [recovered[k] for k in recovered]},
                          schema_overrides={"round": pl.Int64, "tick": pl.Int64, "player_id": pl.Int64, "side_rec": pl.Int64})
    D2 = D.with_columns(pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).join(
        rec_df, on=["match_key", "round", "tick", "player_id"], how="left")
    D2 = D2.with_columns(pl.coalesce([pl.col("victim_side"), pl.col("side_rec")]).alias("victim_side"))
    after = checked_table(positions, D2.select(["match_key", "round", "second", "victim_side"]), snap, "ceil")
    agree_after = 1 - after.filter(pl.col("diff") != 0).height / after.height
    bad_after = after.filter(pl.col("diff") != 0)
    # Exclusion test: do the rounds with an unknown death side (plus the roster-defect rounds) explain every remaining mismatch?
    R = pl.read_parquet(ROOT + "/data/combined/rounds.parquet", columns=["match_key", "round", "missing_side"]).with_columns(
        pl.col("round").cast(pl.Int64))
    null_rounds = null_rows.select(["match_key", pl.col("round").cast(pl.Int64)]).unique()
    excl = null_rounds.join(R, on=["match_key", "round"], how="left")
    defect_rounds = R.filter(pl.col("missing_side").is_not_null() & pl.col("match_key").is_in(sample)).select(["match_key", "round"])
    excluded = pl.concat([null_rounds, defect_rounds]).unique()
    ex = before.with_columns(pl.col("round").cast(pl.Int64)).join(excluded.with_columns(pl.lit(True).alias("_x")),
                                                                  on=["match_key", "round"], how="left")
    kept = ex.filter(pl.col("_x").is_null())
    agree_kept = 1 - kept.filter(pl.col("diff") != 0).height / kept.height
    res = {"sample_matches": len(sample), "agree_before": agree_before, "agree_after": agree_after,
           "rounds_with_null_side_death": null_rounds.height, "rounds_roster_defect": defect_rounds.height,
           "rounds_with_null_side_death_that_are_roster_defect": int(excl.filter(pl.col("missing_side").is_not_null()).height),
           "rounds_excluded_union": excluded.height, "agree_on_kept_rounds": agree_kept,
           "mismatch_pairs_on_kept_rounds": kept.filter(pl.col("diff") != 0).height,
           "kept_pairs": kept.height,
           "null_side_deaths": null_rows.height, "recovered": n_rec, "unrecovered": len(recovered) - n_rec,
           "mismatch_pairs_after": bad_after.height, "pairs": after.height,
           "mismatch_rounds_after": bad_after.select(["match_key", "round"]).unique().height,
           "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
