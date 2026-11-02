"""Decisive check on the 2024 residual: for each player who vanishes from positions_1hz mid-round in a NON-roster-defect round, read the
original 2024 tome (E: copy; the G: copy under source_2024 is partial) and ask three questions:

  raw_rows_after   does the raw player_vector still have rows for this player after the positions vanish? (yes: positions dropped them)
  raw_death        does the raw player_death table record this player's death in the round? and at which tick?
  health_zero      does the raw player_status show health 0 for this player at or before the vanish?
  in_deaths_table  does data/combined/deaths.parquet contain that death (same match, round, tick)?

Reading the answers:
  raw rows after the vanish                 -> positions extraction dropped rows (extraction bug)
  raw death recorded, not in deaths.parquet  -> deaths table missing events (deaths extraction bug)
  no raw death, health above zero           -> the player is gone for another reason (disconnect, bot takeover); report separately
Writes data/diagnostics/gate_2024_vanish_check.json.
Run in the cs2gpu environment (polars, pyarrow):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/gate_2024_vanish_check.py
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
RAW = "E:/csds_final/csds/csds_tail_recovered_20261011/csds"
OUT = os.path.join(ROOT, "data", "diagnostics", "gate_2024_vanish_check.json")
MAX_CASES = 60


def raw_dir(mk):
    p = mk.split("/")  # csds/2024/02/02/<id>/csds
    return os.path.join(RAW, p[1], p[2], p[3], p[4])


def read(d, table, cols, filt=None):
    f = os.path.join(d, table)
    if not os.path.exists(f):
        return None
    t = pl.from_arrow(pq.read_table(f, columns=[c for c in cols if c in pq.read_schema(f).names]))
    return t.filter(filt) if filt is not None else t


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
    deaths = D.select(["match_key", "round", "second", "victim_side"])
    snap = P.select(["match_key", "round", "second"]).unique()
    positions = P.select(["match_key", "round", "second", pl.col("side").cast(pl.Int64), "player_id"])
    chk = checked_table(positions, deaths, snap, "ceil")
    R = pl.read_parquet(ROOT + "/data/combined/rounds.parquet", columns=["match_key", "round", "missing_side"])
    clean_bad = (chk.filter(pl.col("diff") != 0).select(["match_key", "round"]).unique()
                 .join(R, on=["match_key", "round"], how="left").filter(pl.col("missing_side").is_null()))
    print(f"non-defect bad rounds: {clean_bad.height}", flush=True)

    # Vanished players: rounds where the player's last positions row is before the round's last positions row, in a clean bad round.
    rounds_keys = clean_bad.select(["match_key", "round"]).with_columns(pl.col("round").cast(pl.Int64))
    lastrow = (P.group_by(["match_key", "round", "player_id"]).agg(pl.col("tick").max().alias("pos_last_tick"),
                                                                    pl.col("tick").min().alias("pos_first_tick"))
               .with_columns(pl.col("round").cast(pl.Int64)))
    lastrow = lastrow.join(rounds_keys, on=["match_key", "round"], how="inner")
    roundend = lastrow.group_by(["match_key", "round"]).agg(pl.col("pos_last_tick").max().alias("round_last_tick"))
    vanished = (lastrow.join(roundend, on=["match_key", "round"], how="left")
                .filter(pl.col("pos_last_tick") < pl.col("round_last_tick")).sort(["match_key", "round", "player_id"]))
    print(f"vanished players in clean bad rounds: {vanished.height}", flush=True)
    cases = vanished.head(MAX_CASES).to_dicts()

    out = []
    for c in cases:
        mk, rd, pid, last = c["match_key"], int(c["round"]), int(c["player_id"]), int(c["pos_last_tick"])
        d = raw_dir(mk)
        if not os.path.isdir(d):
            out.append({"match_key": mk, "round": rd, "player_id": pid, "raw": "missing"})
            continue
        pv = read(d, "player_vector", ["tick", "round", "player_id"], (pl.col("round") == rd) & (pl.col("player_id") == pid))
        st = read(d, "player_status", ["tick", "round", "player_id", "health"], (pl.col("round") == rd) & (pl.col("player_id") == pid))
        de = read(d, "player_death", ["round", "tick", "player_id"], (pl.col("round") == rd) & (pl.col("player_id") == pid))
        raw_after = int(pv.filter(pl.col("tick") > last).height) if pv is not None else None
        raw_last = int(pv["tick"].max()) if pv is not None and pv.height else None
        death_ticks = de["tick"].to_list() if de is not None else []
        health_zero_at_or_before = None
        if st is not None and st.height:
            z = st.filter((pl.col("health") <= 0) & (pl.col("tick") <= last + 64))
            health_zero_at_or_before = bool(z.height > 0)
        dt = deaths.filter(pl.col("match_key") == mk)  # deaths table (by second); compare by match and round through tick below
        in_deaths = None
        if death_ticks:
            dd = D.filter((pl.col("match_key") == mk) & (pl.col("round") == rd) & (pl.col("tick").is_in(death_ticks)))
            in_deaths = bool(dd.height > 0)
        out.append({"match_key": mk, "round": rd, "player_id": pid, "pos_last_tick": last,
                    "raw_rows_after_vanish": raw_after, "raw_last_tick": raw_last,
                    "raw_death_ticks": [int(x) for x in death_ticks], "in_deaths_table": in_deaths,
                    "health_zero_at_or_before_vanish": health_zero_at_or_before})

    cat = {"raw_rows_after_vanish": 0, "raw_death_missing_from_deaths_table": 0, "raw_death_in_deaths_table": 0,
           "no_raw_death_health_not_zero": 0, "unclassified": 0, "raw_missing": 0}
    for o in out:
        if o.get("raw") == "missing":
            cat["raw_missing"] += 1
        elif (o.get("raw_rows_after_vanish") or 0) > 0:
            cat["raw_rows_after_vanish"] += 1
        elif o.get("raw_death_ticks") and o.get("in_deaths_table") is False:
            cat["raw_death_missing_from_deaths_table"] += 1
        elif o.get("raw_death_ticks") and o.get("in_deaths_table"):
            cat["raw_death_in_deaths_table"] += 1
        elif not o.get("raw_death_ticks") and o.get("health_zero_at_or_before_vanish") is False:
            cat["no_raw_death_health_not_zero"] += 1
        else:
            cat["unclassified"] += 1
    res = {"sample_matches": len(sample), "clean_bad_rounds": clean_bad.height, "vanished_players": vanished.height,
           "cases_checked": len(out), "categories": cat, "cases": out, "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
