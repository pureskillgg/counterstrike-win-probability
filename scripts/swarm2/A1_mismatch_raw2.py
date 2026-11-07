# A1: for 4 mismatch matches (one per day 01/02/03 plus last), check raw E: round events vs tick span and score updates.
import os, glob, json
import polars as pl
import pyarrow.parquet as pq
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
E = r"E:\csds_final\csds\2026\10"
m = pl.read_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_round_end_per_match.csv", infer_schema_length=0).with_columns(
    pl.col("S").cast(pl.Int64), pl.col("rows").cast(pl.Int64), pl.col("distinct_rounds").cast(pl.Int64), pl.col("max_round").cast(pl.Int64))
mis = m.filter(pl.col("distinct_rounds") != pl.col("S")).sort("key")
sel = []
for day in ["01", "02", "03"]:
    c = mis.filter(pl.col("key").str.split("/").list.get(3) == day)
    sel.append(c.row(0, named=True))
sel.append(mis.row(-1, named=True))
res = []
for r in sel:
    key = r["key"]; dd, mid = key.split("/")[3], key.split("/")[4]
    base = os.path.join(E, dd, mid)
    rre = pl.read_parquet(os.path.join(base, "round_end"))
    tick_schema = pq.read_schema(os.path.join(base, "tick")).names
    tick = pl.read_parquet(os.path.join(base, "tick"), columns=[c for c in ["tick", "round", "game_phase", "round_phase"] if c in tick_schema]) if "tick" in tick_schema else None
    rsu = pl.read_parquet(os.path.join(base, "score_update"))
    rst = pl.read_parquet(os.path.join(base, "round_start"))
    rs_state = pl.read_parquet(os.path.join(base, "round_state")) if os.path.exists(os.path.join(base, "round_state")) else None
    hdr = pl.read_parquet(os.path.join(base, "header"), columns=[c for c in ["t_starters_score_final", "ct_starters_score_final", "max_rounds", "match_date"] if c in pq.read_schema(os.path.join(base, "header")).names])
    tmax = tick.get_column("tick").max() if tick is not None else None
    tick_round_max = tick.get_column("round").max() if tick is not None and "round" in tick.columns else None
    print("== ", key[11:], "S", r["S"], "| raw round_end rows", rre.height, "max round", rre.get_column("round").max(),
          "last round_end tick", rre.get_column("tick").max(), "| raw tick max", tmax, "| tick.round max", tick_round_max)
    print("   tick cols", tick_schema[:12], "round_start rows", rst.height, "max", rst.get_column("round").max(),
          "| round_state max", None if rs_state is None else rs_state.get_column("round").max(), "| score_update last", rsu.tail(2).select(["round","team_code","old_score","new_score"]).rows(),
          "| header", hdr.rows())
    res.append(dict(key=key, S=r["S"], round_end_rows=rre.height, round_end_max=rre.get_column("round").max(), last_round_end_tick=rre.get_column("tick").max(), raw_tick_max=tmax, tick_round_max=tick_round_max))
json.dump(res, open(r"G:\pureskill\csds-cs2\logs\swarm2\A1_mismatch_raw2.json", "w"), indent=1, default=str)
