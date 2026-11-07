"""verify D5_10, step 1: independent per-round and per-match build from the raw tomes (read-only on G:).

Builds, for the 2024 tome and the 2026 tome, one row per (match_key, round) with:
  winner (round_end.winner_team_code, first row by tick), round_end row count, distinct winners,
  freeze-end scores (round_state round_freeze_end, first row by tick, t_score / ct_score and raw fields),
  bomb outcome per round (bomb_planted / bomb_defused / bomb_exploded and planter side),
  and per match: header finals, max_rounds (2026 header only), regime markers (counts of
  round_announce_match_start, cs_win_panel_round, cs_win_panel_match, round_announce_last_round_half).

Written only to the scratchpad (not to E:, not into the project). Column projection on every scan.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_build.py
"""
import glob
import re
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10")
OUT.mkdir(parents=True, exist_ok=True)

ROUND_EV = ["round_freeze_end", "round_start", "round_end"]
MATCH_EV = ["round_announce_match_start", "cs_win_panel_round", "cs_win_panel_match",
            "round_announce_last_round_half", "round_announce_match_point"]
BOMB_EV = ["bomb_planted", "bomb_defused", "bomb_exploded"]


def files(base, suf, ch):
    return sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))


def scan(base, suf, ch, want):
    """Lazy scan of one channel, projected to the wanted columns that exist in the file schema."""
    fs = files(base, suf, ch)
    if not fs:
        return None, []
    have = set(pl.read_parquet_schema(fs[0]).keys())
    cols = [c for c in want if c in have]
    return pl.concat([pl.scan_parquet(f).select(cols) for f in fs], how="vertical_relaxed"), cols


def build(year):
    base, suf = TOMES[year]
    print(f"\n==== build {year} ({base}) ====")

    # ---------- header ----------
    hf = sorted(glob.glob(str(base / "header" / "dataframe_*")))
    hcols_want = ["key", "match_date", "map_name", "match_type", "t_starters_score_final",
                  "ct_starters_score_final", "max_rounds"]
    hdr = pl.concat([pl.scan_parquet(f).select([c for c in hcols_want if c in pl.read_parquet_schema(f)])
                     for f in hf], how="vertical_relaxed").collect()
    hdr = hdr.rename({"key": "match_key"})
    if "max_rounds" not in hdr.columns:
        hdr = hdr.with_columns(pl.lit(None, dtype=pl.Int64).alias("max_rounds"))
    print(f"header: rows {hdr.height}, unique keys {hdr['match_key'].n_unique()}, "
          f"max_rounds column present in header: {'max_rounds' in set(pl.read_parquet_schema(hf[0]).keys())}")

    # day from the match key (regex checked against every key)
    pat = re.compile(r"csds/(\d{4})/(\d{2})/(\d{2})/")
    keys = hdr["match_key"].to_list()
    miss = [k for k in keys if pat.search(k) is None]
    print(f"match keys not matching csds/YYYY/MM/DD/ pattern: {len(miss)}; example key: {keys[0]}")
    hdr = hdr.with_columns(
        day=pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1),
        ym=pl.col("match_key").str.extract(r"csds/(\d{4}/\d{2})/", 1))

    # ---------- round_end ----------
    re_lf, _ = scan(base, suf, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    RE_raw = re_lf.collect()
    RE = (RE_raw.sort(["match_key", "round", "tick"])
          .group_by(["match_key", "round"]).agg(
              pl.col("winner_team_code").first().alias("winner"),
              pl.col("tick").first().alias("end_tick"),
              pl.len().alias("n_end_rows"),
              pl.col("winner_team_code").n_unique().alias("n_winner_distinct"),
              pl.col("player_count").first().alias("player_count")))
    print(f"round_end: rows {RE_raw.height}; match-round pairs {RE.height}; "
          f"pairs with >1 row {int((RE['n_end_rows'] > 1).sum())}; "
          f"winner values {sorted(RE['winner'].unique().drop_nulls().to_list())}")

    # ---------- round_state ----------
    rs_want = ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]
    rs_lf, rs_cols = scan(base, suf, "round_state", rs_want)
    RS = rs_lf.filter(pl.col("event_type").is_in(ROUND_EV + MATCH_EV)).collect()
    print(f"round_state (selected event types): rows {RS.height}; columns used {rs_cols}")
    FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
          .group_by(["match_key", "round"]).agg(
              pl.col("tick").first().alias("fe_tick"),
              pl.col("t_score").first().alias("fe_t"),
              pl.col("ct_score").first().alias("fe_ct"),
              (pl.col("t_score_raw").first() if "t_score_raw" in rs_cols else pl.lit(None, dtype=pl.Int64)).alias("fe_t_raw"),
              (pl.col("ct_score_raw").first() if "ct_score_raw" in rs_cols else pl.lit(None, dtype=pl.Int64)).alias("fe_ct_raw"),
              pl.len().alias("n_fe_rows"),
              pl.col("t_score").n_unique().alias("n_fe_t_distinct"),
              pl.col("ct_score").n_unique().alias("n_fe_ct_distinct")))
    print(f"freeze-end: match-round pairs {FE.height}; pairs with >1 freeze row {int((FE['n_fe_rows'] > 1).sum())}; "
          f"pairs with >1 distinct t_score {int((FE['n_fe_t_distinct'] > 1).sum())}")
    cnt_rs = (RS.filter(pl.col("event_type").is_in(ROUND_EV))
              .group_by(["match_key", "round", "event_type"]).agg(pl.len().alias("n"))
              .pivot(on="event_type", index=["match_key", "round"], values="n", aggregate_function="sum")
              .fill_null(0))
    cnt_rs = cnt_rs.rename({c: f"n_{c}" for c in cnt_rs.columns if c not in ("match_key", "round")})
    MC = (RS.filter(pl.col("event_type").is_in(MATCH_EV))
          .group_by(["match_key", "event_type"]).agg(pl.len().alias("n"))
          .pivot(on="event_type", index="match_key", values="n", aggregate_function="sum").fill_null(0))
    MC = MC.rename({c: f"n_{c}" for c in MC.columns if c != "match_key"})
    for c in MATCH_EV:
        if f"n_{c}" not in MC.columns:
            MC = MC.with_columns(pl.lit(0).alias(f"n_{c}"))

    # ---------- bomb_state ----------
    bs_lf, bs_cols = scan(base, suf, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
    BS = bs_lf.filter(pl.col("event_type").is_in(BOMB_EV)).collect()
    print(f"bomb_state bomb rows: {BS.height}; by type {BS.group_by('event_type').len().sort('event_type').to_dicts()}")
    BOMBP = (BS.group_by(["match_key", "round"]).agg(
        pl.col("event_type").filter(pl.col("event_type") == "bomb_planted").len().alias("n_planted"),
        pl.col("event_type").filter(pl.col("event_type") == "bomb_defused").len().alias("n_defused"),
        pl.col("event_type").filter(pl.col("event_type") == "bomb_exploded").len().alias("n_exploded"),
        pl.col("player_team_code").filter(pl.col("event_type") == "bomb_planted").first().alias("planter_side")))

    # ---------- player_count of round_start rows? (not needed) ----------

    # ---------- assemble rounds ----------
    R = (RE.join(FE, on=["match_key", "round"], how="full", coalesce=True)
         .join(cnt_rs, on=["match_key", "round"], how="left")
         .join(BOMBP, on=["match_key", "round"], how="left")
         .with_columns(year=pl.lit(year)))
    R = R.with_columns(
        n_planted=pl.col("n_planted").fill_null(0), n_defused=pl.col("n_defused").fill_null(0),
        n_exploded=pl.col("n_exploded").fill_null(0)).sort(["match_key", "round"])

    # ---------- assemble matches ----------
    mk_rounds = RE.group_by("match_key").agg(
        pl.len().alias("n_round_end_pairs"),
        pl.col("round").max().alias("max_round_end"),
        pl.col("round").min().alias("min_round_end"),
        (pl.col("n_end_rows") > 1).sum().alias("n_dup_end_pairs"))
    M = (hdr.join(mk_rounds, on="match_key", how="left")
         .join(MC, on="match_key", how="left"))
    for c in MATCH_EV:
        M = M.with_columns(pl.col(f"n_{c}").fill_null(0))
    M = M.with_columns(year=pl.lit(year))

    pq.write_table(R.to_arrow(), str(OUT / f"rounds_{year}.parquet"))
    pq.write_table(M.to_arrow(), str(OUT / f"matches_{year}.parquet"))
    print(f"wrote rounds_{year}.parquet rows {R.height}; matches_{year}.parquet rows {M.height}")
    print(f"matches with round_announce_match_start > 0: {M.filter(pl.col('n_round_announce_match_start') > 0).height}")
    print(f"matches with cs_win_panel_round > 0: {M.filter(pl.col('n_cs_win_panel_round') > 0).height}")
    print(f"matches with round_announce_last_round_half > 0: {M.filter(pl.col('n_round_announce_last_round_half') > 0).height}")
    print(f"matches with max_rounds non-null: {M.filter(pl.col('max_rounds').is_not_null()).height}")


for y in ["2024", "2026"]:
    build(y)
print("\nbuild complete")
