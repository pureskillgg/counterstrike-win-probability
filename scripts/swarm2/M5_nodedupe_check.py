"""M5: does duplicate removal change the per-map batch difference? (read-only on the tome)

Recomputes the core-map round-level CT% by batch from clean 5v5 non-wingman rounds WITHOUT the
duplicate removal, and reports how many duplicate-group members fall in each batch and map.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M5_nodedupe_check.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUFFIX = ".2026-09-18,2026-10-08"
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M5")
CORE = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]


def batch_label(d):
    if d == "2026-10-08":
        return "b1008"
    if d is not None and "2026-10-01" <= d <= "2026-10-07":
        return "b1001_1007"
    return "other"


def main():
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    hdr = (pl.scan_parquet(sorted(glob.glob(str(TOME / "header" / "dataframe_*"))))
           .select(["key", "match_date", "map_name", "is_wingman"]).collect()
           .rename({"key": "match_key"}))
    hdr = hdr.with_columns(pl.col("match_date").str.slice(0, 10).map_elements(
        batch_label, return_dtype=pl.String).alias("batch"))
    re = (pl.scan_parquet(sorted(glob.glob(str(TOME / f"basic_round_end{SUFFIX}" / "dataframe_*"))))
          .select(["match_key", "round", "winner_team_code", "player_count"]).collect())
    j = re.join(hdr, on="match_key", how="inner").filter((~pl.col("is_wingman")) & (pl.col("player_count") == 10))
    j = j.with_columns((pl.col("winner_team_code") == 3).cast(pl.Int8).alias("ct_win"))
    j = j.filter(pl.col("batch").is_in(["b1001_1007", "b1008"]) & pl.col("map_name").is_in(CORE))
    dedupe = pl.read_parquet(SCRATCH / "M5_dup_groups.parquet")
    removed = set(dedupe["match_key"].to_list())
    j = j.with_columns(pl.col("match_key").is_in(list(removed)).alias("dup_removed"))
    print("all clean core rounds (no dedupe):", j.height, "matches", j["match_key"].n_unique())
    print("duplicate copies (lowest key kept elsewhere) among core rounds:")
    print(j.filter(pl.col("dup_removed")).group_by(["map_name", "batch"]).agg(
        pl.col("match_key").n_unique().alias("matches")).sort(["map_name", "batch"]))
    rows = []
    for m in CORE:
        for lab, sub in [("with_dupes", j.filter(pl.col("map_name") == m)),
                         ("dedupe", j.filter((pl.col("map_name") == m) & (~pl.col("dup_removed"))))]:
            a = sub.filter(pl.col("batch") == "b1001_1007")
            b = sub.filter(pl.col("batch") == "b1008")
            pa = float(a["ct_win"].mean())
            pb = float(b["ct_win"].mean())
            rows.append({"map": m, "version": lab, "ct_1001_1007": round(100 * pa, 2),
                         "ct_1008": round(100 * pb, 2), "diff_pp": round(100 * (pb - pa), 2),
                         "matches_1001_1007": a["match_key"].n_unique(), "matches_1008": b["match_key"].n_unique()})
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
