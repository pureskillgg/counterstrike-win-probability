"""M2 probe: per-match coverage of round-level channels in both eras, keyset vs rows, roster size check."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")
ERAS = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}


def read(base, suf, ch, cols=None):
    fd = base / f"basic_{ch}{suf}"
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    if not files:
        return None
    frames = []
    for f in files:
        sch = pl.read_parquet_schema(f)
        if not sch:
            continue
        use = [c for c in (cols or sch.keys()) if c in sch]
        frames.append(pl.read_parquet(f, columns=use))
    if not frames:
        return None
    return pl.concat(frames, how="vertical_relaxed")


def keyset(base, suf, ch):
    fd = base / f"basic_{ch}{suf}"
    files = sorted(glob.glob(str(fd / "keyset_*")))
    if not files:
        return None
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


for era, (base, suf) in ERAS.items():
    hdr = pl.concat([pl.read_parquet(f, columns=["key", "match_date"]) for f in
                     sorted(glob.glob(str(base / "header" / "dataframe_*")))])
    all_m = set(hdr["key"].to_list())
    print(f"===== {era}: header matches {len(all_m)}")
    for ch in ["round_end", "round_start", "round_state", "player_info", "player_death", "bomb_state",
               "player_disconnect", "bot_takeover"]:
        df = read(base, suf, ch, ["match_key"])
        ks = keyset(base, suf, ch)
        if df is None:
            print(f"  {ch}: no data pages")
            continue
        have = set(df["match_key"].unique().to_list())
        ks_n = None if ks is None else len(ks)
        print(f"  {ch:<18} rows={df.height:>7} matches_with_rows={len(have):>5} keyset={ks_n} "
              f"zero_row_matches={len(all_m - have):>5} rows_outside_header={len(have - all_m)} "
              f"keyset_minus_header={None if ks is None else len(ks - all_m)} "
              f"header_minus_keyset={None if ks is None else len(all_m - ks)}")
    # roster check: distinct player_id per round in player_info, grouped by round_end player_count
    re_ = read(base, suf, "round_end", ["match_key", "round", "player_count"])
    pi = read(base, suf, "player_info", ["match_key", "round", "player_id", "team_code"])
    if pi is not None:
        per = pi.group_by(["match_key", "round"]).agg(
            pl.col("player_id").n_unique().alias("np"),
            (pl.col("team_code") == 2).sum().alias("nt"),
            (pl.col("team_code") == 3).sum().alias("nct"))
        j = re_.join(per, on=["match_key", "round"], how="left")
        print("  player_info roster (np, nt, nct) by round_end player_count:")
        print(j.group_by(["player_count"]).agg(
            pl.len().alias("rounds"),
            pl.col("np").is_null().sum().alias("no_pi"),
            ((pl.col("np") == 10) & (pl.col("nt") == 5) & (pl.col("nct") == 5)).sum().alias("five_v_five_pi"),
        ).sort("rounds", descending=True).head(8))
    # round_start coverage at round level
    rs = read(base, suf, "round_start", ["match_key", "round"])
    if rs is not None and re_ is not None:
        rr = re_.select(["match_key", "round"]).unique()
        rsu = rs.select(["match_key", "round"]).unique()
        print(f"  rounds in round_end: {rr.height}; with round_start row: {rsu.height}; "
              f"round_start rows also in round_end: {rsu.join(rr, on=['match_key','round']).height}")
        per_m = rr.group_by("match_key").agg(pl.len().alias("nr"))
        per_s = rsu.group_by("match_key").agg(pl.len().alias("ns"))
        mm = per_m.join(per_s, on="match_key", how="left").with_columns(pl.col("ns").fill_null(0))
        print("  matches with 0 round_start rows:", mm.filter(pl.col("ns") == 0).height,
              " matches with some but not all:", mm.filter((pl.col("ns") > 0) & (pl.col("ns") < pl.col("nr"))).height,
              " matches with all:", mm.filter(pl.col("ns") == pl.col("nr")).height)
    print()
