"""M5 loader: clean 5v5 round table for the map x batch question (read-only on the tome).

Reads only header, round_end and round_state projections from
tomes\\2026-10-01_to_08\\tome\\csds. Writes one intermediate parquet to the session scratchpad
(not under the project, not under E:).

Definitions (ASSUMPTIONS are marked where they are choices):
  - Clean round: header is_wingman False, max_rounds 24 (non-wingman), round_end player_count == 10.
    This matches the 5v5 filter used in notes\\eda\\priors_by_map_round.md.
  - Outcome: ct_win = round_end.winner_team_code == 3 (side-based; verified in the EDA notes).
  - Batch: header match_date day. 2026-10-08 is 'b1008'; 2026-10-01..10-07 is 'b1001_1007';
    any other date is 'other' (ASSUMPTION: follows notes\\eda\\batch_effect_2026-10-08.md).
  - Duplicate matches: groups with an identical round_end signature (round, tick, winner, reason
    for every round). Within a group the lowest match_key is kept (same rule as bakeoff.py).
    ASSUMPTION: round_end signature alone is used; it is compared with the bakeoff count in the log.
  - Score lead: ct_score - t_score at round_freeze_end (team-based, see priors notes), clipped to +/-4.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M5_load.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M5")
OUT_PQ = SCRATCH / "M5_rounds.parquet"
OUT_DUP = SCRATCH / "M5_dup_groups.parquet"
CT_SIDE = 3


def files(ch):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no pages for {ch}")
    return fs


def batch_label(d):
    if d == "2026-10-08":
        return "b1008"
    if d is not None and "2026-10-01" <= d <= "2026-10-07":
        return "b1001_1007"
    return "other"


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    hdr = (pl.scan_parquet(files("header"))
           .select(["key", "match_id", "match_date", "map_name", "is_wingman", "max_rounds",
                    "platform", "match_type"])
           .collect())
    hdr = hdr.with_columns(
        pl.col("match_date").str.slice(0, 10).alias("match_day"),
    ).with_columns(
        pl.col("match_day").map_elements(batch_label, return_dtype=pl.String).alias("batch"),
    )
    print(f"header rows {hdr.height}; unique keys {hdr['key'].n_unique()}")
    print("batch counts (header level):")
    print(hdr.group_by("batch").agg(pl.len().alias("matches")).sort("batch"))

    re = (pl.scan_parquet(files("round_end"))
          .select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
          .collect())
    print(f"round_end rows {re.height}")

    rs = (pl.scan_parquet(files("round_state"))
          .filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "t_score", "ct_score"])
          .unique(subset=["match_key", "round"], keep="first")
          .collect())
    print(f"freeze-end rows (unique match, round) {rs.height}")

    # Duplicate detection on round_end signature (all matches, before any round filter).
    sig = (re.sort(["match_key", "round"])
           .group_by("match_key")
           .agg(pl.struct(["round", "tick", "winner_team_code", "win_reason_code"]).alias("rounds"),
                pl.len().alias("n_rounds"))
           .with_columns(pl.col("rounds").map_elements(lambda s: str(sorted(
               [(x["round"], x["tick"], x["winner_team_code"], x["win_reason_code"]) for x in s])),
               return_dtype=pl.String).alias("sig"))
           .drop("rounds"))
    grp = (sig.group_by("sig")
           .agg(pl.col("match_key").sort().alias("members"), pl.len().alias("size"))
           .filter(pl.col("size") > 1))
    n_groups = grp.height
    n_members = int(grp["size"].sum()) if n_groups else 0
    n_removed = n_members - n_groups
    print(f"round_end-signature duplicate groups (all matches, no filter): {n_groups}; "
          f"members {n_members}; removable (keep lowest) {n_removed}")
    dup_rows = []
    for row in grp.iter_rows(named=True):
        mem = sorted(row["members"])
        for k in mem[1:]:
            dup_rows.append({"match_key": k, "dup_of": mem[0], "group_size": row["size"]})
    dup_df = pl.DataFrame(dup_rows, schema={"match_key": pl.String, "dup_of": pl.String,
                                            "group_size": pl.Int64}) if dup_rows else pl.DataFrame(
        schema={"match_key": pl.String, "dup_of": pl.String, "group_size": pl.Int64})
    dup_df.write_parquet(OUT_DUP)
    removed_keys = set(dup_df["match_key"].to_list())

    # Join header to rounds. round_end.match_key == header.key (verified by the EDA notes).
    j = (re.join(hdr.rename({"key": "match_key"}), on="match_key", how="inner")
         .join(rs, on=["match_key", "round"], how="left"))
    print(f"round_end rows joined to header: {j.height} (header miss: {re.height - j.height})")

    clean = j.filter((~pl.col("is_wingman")) & (pl.col("player_count") == 10))
    print(f"clean 5v5 non-wingman rounds (before dedupe): {clean.height}; matches {clean['match_key'].n_unique()}")
    clean = clean.with_columns(
        pl.col("match_key").is_in(list(removed_keys)).alias("dup_removed"))
    kept = clean.filter(~pl.col("dup_removed"))
    print(f"after dedupe (lowest key kept): rounds {kept.height}; matches {kept['match_key'].n_unique()}")

    kept = kept.with_columns(
        (pl.col("winner_team_code") == CT_SIDE).cast(pl.Int8).alias("ct_win"),
        (pl.col("ct_score") - pl.col("t_score")).clip(-4, 4).alias("score_lead_ct"),
        pl.col("round").alias("round_no"),
    ).select(["match_key", "match_id", "match_day", "batch", "map_name", "round_no", "ct_win",
              "winner_team_code", "win_reason_code", "player_count", "max_rounds", "platform",
              "match_type", "tick", "score_lead_ct", "t_score", "ct_score"])
    kept.write_parquet(OUT_PQ)
    print(f"wrote {OUT_PQ} rows {kept.height}")
    print("rounds with no freeze-end score:", kept.filter(pl.col("score_lead_ct").is_null()).height)
    print("missing map_name:", kept.filter(pl.col("map_name").is_null()).height)


if __name__ == "__main__":
    main()
