"""X12 step 1: place_name (callout) at the 1 Hz snapshot rows of the existing state sample.

For each of the 150 sampled matches, read player_status from E: (columns tick, round, player_id,
place_name, is_in_bomb_zone), one match at a time, and keep the (round, tick, player_id) rows that
are in data\\state_sample\\players_1hz.parquet. Writes data\\state_sample\\place_names.parquet.

Read only from E:. Writes nothing under E:. Does not edit existing files.

Run from the project root:
  C:\\Users\\13144\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\exp\\X12_extract_place_names.py
"""

import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
import pyarrow.parquet as pq

ROOT = Path(r"G:\pureskill\csds-cs2")
SAMPLE = ROOT / "data" / "state_sample"
LOG_PATH = ROOT / "logs" / "exp" / "X12_extract.log"
OUT_PATH = SAMPLE / "place_names.parquet"
E_CSDS = r"E:\csds_final\csds"
PS_COLS = ["tick", "round", "player_id", "place_name", "is_in_bomb_zone"]

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_logf = open(LOG_PATH, "w", encoding="utf-8")


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s, flush=True)
    _logf.write(s + "\n")
    _logf.flush()


def e_dir(mk):
    parts = mk.split("/")
    return os.path.join(E_CSDS, parts[1], parts[2], parts[3], parts[4])


def clean_name_expr():
    return pl.col("place_name").is_not_null() & (pl.col("place_name").str.strip_chars() != "")


def main():
    t_all = time.perf_counter()
    p = pl.read_parquet(SAMPLE / "players_1hz.parquet",
                        columns=["match_key", "round", "tick", "player_id"])
    p = p.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                       pl.col("player_id").cast(pl.Int64))
    rounds = pl.read_parquet(SAMPLE / "rounds.parquet",
                             columns=["match_key", "round", "map_name"]).with_columns(
        pl.col("round").cast(pl.Int64))
    p = p.join(rounds, on=["match_key", "round"], how="left")
    n_dup = p.height - p.unique(["match_key", "round", "tick", "player_id"]).height
    log(f"players_1hz snapshot rows {p.height}; duplicate keys {n_dup}; "
        f"rows without a rounds.parquet map {p.filter(pl.col('map_name').is_null()).height}")
    by_match = p.partition_by("match_key", as_dict=True)
    matches = sorted(k[0] for k in by_match.keys())
    log(f"matches {len(matches)}")

    parts, timings = [], []
    missing_status_total = 0
    dup_status_total = 0
    for i, mk in enumerate(matches, 1):
        path = os.path.join(e_dir(mk), "player_status")
        t0 = time.perf_counter()
        tbl = pq.read_table(path, columns=PS_COLS)
        t_read = time.perf_counter() - t0
        st = pl.from_arrow(tbl).with_columns(
            pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64),
            pl.col("player_id").cast(pl.Int64), pl.col("place_name").cast(pl.String),
            pl.col("is_in_bomb_zone").cast(pl.Boolean))
        n_st = st.height
        st_dup = n_st - st.unique(["round", "tick", "player_id"]).height
        dup_status_total += st_dup
        if st_dup:
            st = st.unique(["round", "tick", "player_id"], keep="first", maintain_order=True)
        keys = by_match[(mk,)].select(["round", "tick", "player_id"])
        j = keys.join(st, on=["round", "tick", "player_id"], how="left")
        miss = j.filter(pl.col("is_in_bomb_zone").is_null()).height
        missing_status_total += miss
        j = j.with_columns(pl.lit(mk).alias("match_key"))
        parts.append(j.select(["match_key", "round", "tick", "player_id", "place_name",
                               "is_in_bomb_zone"]))
        wall = time.perf_counter() - t0
        timings.append({"match_key": mk, "file_bytes": os.path.getsize(path), "status_rows": n_st,
                        "keys": keys.height, "missing_status": miss, "read_s": round(t_read, 3),
                        "wall_s": round(wall, 3)})
        if i % 25 == 0 or i == len(matches):
            log(f"  {i}/{len(matches)} matches; last read {t_read:.2f}s wall {wall:.2f}s")

    out = pl.concat(parts, how="vertical").with_columns(
        pl.col("is_in_bomb_zone").fill_null(False))
    out = out.sort(["match_key", "round", "tick", "player_id"])
    out.write_parquet(OUT_PATH, compression="zstd")
    log(f"wrote {OUT_PATH} rows {out.height} size {os.path.getsize(OUT_PATH) / 1e6:.2f} MB")
    log(f"status rows missing for a snapshot key: {missing_status_total}; "
        f"duplicate player_status keys dropped: {dup_status_total}")

    m = out.join(rounds, on=["match_key", "round"], how="left")
    n = m.height
    null_or_empty_all = n - m.filter(clean_name_expr()).height
    log(f"== place_name counts (snapshot rows {n}) ==")
    log(f"null or empty place_name: {null_or_empty_all} ({100 * null_or_empty_all / n:.3f}%)")
    per_map = m.group_by("map_name").agg(
        pl.len().alias("rows"),
        pl.col("place_name").filter(clean_name_expr()).n_unique().alias("distinct_place_name"),
        (~clean_name_expr()).sum().alias("null_or_empty"),
        pl.col("is_in_bomb_zone").sum().alias("bomb_zone_rows"),
    ).sort("rows", descending=True)
    log("per map:")
    for r in per_map.iter_rows(named=True):
        share = 100 * r["null_or_empty"] / r["rows"]
        log(f"  {r['map_name']}: rows {r['rows']}, distinct place_name {r['distinct_place_name']}, "
            f"null/empty {r['null_or_empty']} ({share:.3f}%), bomb-zone rows {r['bomb_zone_rows']}")
    top_maps = per_map["map_name"].to_list()[:2]
    for mp in top_maps:
        top = (m.filter((pl.col("map_name") == mp) & clean_name_expr())
               .group_by("place_name").len().sort("len", descending=True).head(15))
        log(f"top 15 place_name on {mp} (snapshot rows):")
        for r in top.iter_rows(named=True):
            log(f"  {r['place_name']!r}: {r['len']}")

    # bomb-zone callouts per map (for the site-zone step)
    bz = (m.filter(pl.col("is_in_bomb_zone") & clean_name_expr())
          .group_by(["map_name", "place_name"]).len().sort(["map_name", "len"], descending=[False, True]))
    log("bomb-zone place_name (is_in_bomb_zone true), top 6 per map:")
    for mp in per_map["map_name"].to_list():
        sub = bz.filter(pl.col("map_name") == mp).head(6)
        log(f"  {mp}: " + ", ".join(f"{r['place_name']!r}={r['len']}" for r in sub.iter_rows(named=True)))

    total_bytes = sum(t["file_bytes"] for t in timings)
    total_read = sum(t["read_s"] for t in timings)
    total_wall = sum(t["wall_s"] for t in timings)
    log(f"E: player_status bytes read: {total_bytes / 1e6:.1f} MB over {len(timings)} files; "
        f"read time {total_read:.1f} s (mean {total_read / len(timings):.2f} s/file, "
        f"max {max(t['read_s'] for t in timings):.2f} s); wall per match mean "
        f"{total_wall / len(timings):.2f} s")
    with open(ROOT / "logs" / "exp" / "X12_extract_timing.csv", "w", encoding="utf-8") as fh:
        fh.write("match_key,file_bytes,status_rows,keys,missing_status,read_s,wall_s\n")
        for t in timings:
            fh.write(f"{t['match_key']},{t['file_bytes']},{t['status_rows']},{t['keys']},"
                     f"{t['missing_status']},{t['read_s']},{t['wall_s']}\n")
    log(f"total wall {time.perf_counter() - t_all:.1f} s")


if __name__ == "__main__":
    main()
