import glob
import json
import random

import polars as pl

RAW_CHECK = "/mnt/sagemaker-nvme/raw_check"
ROUNDS = "/home/sagemaker-user/adam/status/data/combined/rounds.parquet"
PAGES = sorted(glob.glob("/mnt/sagemaker-nvme/parity/c*/dataframe_*"))
KEY = ["round", "tick", "player_id"]

# three match keys drawn with random.Random(7).sample over the 118 per_match entries with V_rows > 0
MATCH_KEYS = [
    "csds/2026/10/21/m_47360c7367d0788ec8c5e842/csds",
    "csds/2026/10/17/m_1caf727ba135e50429bfabf7/csds",
    "csds/2026/10/21/m_72b83f4542327f39379d7253/csds",
]


def derive_raw(folder_dir):
    rs = pl.read_parquet(f"{folder_dir}/round_state")
    re_ = pl.read_parquet(f"{folder_dir}/round_end")
    ps = pl.read_parquet(f"{folder_dir}/player_status")

    freeze = (
        rs.filter(pl.col("event_type") == "round_freeze_end")
        .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
        .group_by("round")
        .agg(pl.col("tick").min().alias("freeze"))
    )
    end = (
        re_.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
        .group_by("round")
        .agg(pl.col("tick").min().alias("end"))
    )
    both = freeze.join(end, on="round", how="inner")

    p_all = ps.select(["round", "tick", "player_id", "health"]).with_columns(
        pl.col("round").cast(pl.Int64),
        pl.col("tick").cast(pl.Int64),
        pl.col("player_id").cast(pl.Int64),
        pl.col("health").cast(pl.Int64),
    )
    joined = p_all.join(both, on="round", how="inner")
    window = joined.filter(
        (pl.col("tick") >= pl.col("freeze"))
        & (pl.col("tick") < pl.col("end"))
        & (((pl.col("tick") - pl.col("freeze")) % 64) == 0)
        & (pl.col("health") > 0)
    )
    pre_dedupe = window.height
    conflicts = (
        window.group_by(KEY)
        .agg(pl.col("health").n_unique().alias("nu"))
        .filter(pl.col("nu") > 1)
        .height
    )
    deduped = window.select(["round", "tick", "player_id", "health"]).unique(
        subset=KEY, keep="first", maintain_order=True
    )
    info = {
        "n_round_state_rows": rs.height,
        "n_round_end_rows": re_.height,
        "n_player_status_rows": ps.height,
        "rounds_with_freeze": freeze.height,
        "rounds_with_end": end.height,
        "rounds_with_both": both.height,
        "window_rows_pre_dedupe": pre_dedupe,
        "dedupe_key_groups_with_conflicting_health": conflicts,
        "raw_rows_after_dedupe": deduped.height,
    }
    return deduped, both, info


def load_tome(match_key):
    hits = []
    for pg in PAGES:
        n = pl.scan_parquet(pg).filter(pl.col("match_key") == match_key).select(pl.len()).collect().item()
        if n:
            hits.append((pg, n))
    frames = []
    for pg, _ in hits:
        frames.append(
            pl.read_parquet(pg)
            .filter(pl.col("match_key") == match_key)
            .select(["tick", "round", "player_id", "health"])
            .with_columns(
                pl.col("tick").cast(pl.Int64),
                pl.col("round").cast(pl.Int64),
                pl.col("player_id").cast(pl.Int64),
                pl.col("health").cast(pl.Int64),
            )
        )
    tome = pl.concat(frames) if frames else pl.DataFrame(schema={"tick": pl.Int64, "round": pl.Int64, "player_id": pl.Int64, "health": pl.Int64})
    return tome, hits


def main():
    rounds = pl.read_parquet(ROUNDS).select(["match_key", "round", "use"])
    rounds = rounds.with_columns(pl.col("round").cast(pl.Int64))

    random_order = MATCH_KEYS
    out = {}
    for idx, mk in enumerate(random_order, start=1):
        folder = mk.rsplit("/", 1)[0]
        d = f"{RAW_CHECK}/m{idx}"
        print("=" * 100)
        print(f"m{idx} match_key={mk}")
        print(f"folder={folder}")

        raw_all, both, info = derive_raw(d)
        print("raw derive info:", json.dumps(info))

        use_rounds = rounds.filter((pl.col("match_key") == mk) & pl.col("use")).select("round")
        rounds_for_match = rounds.filter(pl.col("match_key") == mk)
        print(
            "rounds.parquet rows for match:",
            rounds_for_match.height,
            "use=True:",
            rounds_for_match.filter(pl.col("use")).height,
            "use=False:",
            rounds_for_match.filter(~pl.col("use")).height,
        )

        raw_rounds_set = set(raw_all["round"].unique().to_list())
        raw_use = raw_all.join(use_rounds, on="round", how="semi")
        raw_dropped_by_use = raw_all.height - raw_use.height
        print("raw rows dropped by use filter:", raw_dropped_by_use, "raw rounds total:", len(raw_rounds_set))

        tome_all, hits = load_tome(mk)
        print("tome pages with match:", hits)
        print("tome rows for match (pre-use):", tome_all.height)
        tome = tome_all.join(use_rounds, on="round", how="semi")
        print("tome rows after use filter:", tome.height)
        tome_dups = tome.height - tome.select(KEY).unique().height
        print("tome duplicate key rows (post-use):", tome_dups)

        raw_only = raw_use.join(tome.select(KEY).unique(), on=KEY, how="anti")
        tome_only = tome.join(raw_use.select(KEY).unique(), on=KEY, how="anti")
        shared = raw_use.join(tome.unique(subset=KEY, keep="first", maintain_order=True), on=KEY, how="inner", suffix="_tome")
        mism = shared.filter(pl.col("health") != pl.col("health_tome"))

        # diagnostics for tome-only rows: is the row on the raw grid and inside a raw freeze..end window
        tome_only_diag = (
            tome_only.join(both, on="round", how="left")
            .with_columns(
                pl.col("freeze").is_not_null().alias("round_has_raw_freeze_and_end"),
                (
                    (pl.col("tick") >= pl.col("freeze"))
                    & (pl.col("tick") < pl.col("end"))
                ).fill_null(False).alias("in_raw_window"),
                (((pl.col("tick") - pl.col("freeze")) % 64) == 0).fill_null(False).alias("on_64_grid"),
            )
        )
        diag_counts = (
            tome_only_diag.group_by(["round_has_raw_freeze_and_end", "in_raw_window", "on_64_grid"])
            .len()
            .sort(["round_has_raw_freeze_and_end", "in_raw_window", "on_64_grid"])
        )
        print("tome-only breakdown (round_has_raw_freeze_and_end, in_raw_window, on_64_grid):")
        print(diag_counts)

        summary = {
            "match_key": mk,
            "folder": folder,
            "raw_rows": raw_use.height,
            "tome_rows": tome.height,
            "raw_only_rows": raw_only.height,
            "tome_only_rows": tome_only.height,
            "health_mismatch_rows": mism.height,
            "tome_dup_key_rows": tome_dups,
            "raw_dedupe_conflicts": info["dedupe_key_groups_with_conflicting_health"],
        }
        print("SUMMARY", json.dumps(summary))
        out[f"m{idx}"] = summary

        raw_use.write_parquet(f"{RAW_CHECK}/derived_m{idx}.parquet")
        tome.write_parquet(f"{RAW_CHECK}/tome_m{idx}.parquet")

        with pl.Config(tbl_rows=12, tbl_cols=12, fmt_str_lengths=60):
            if raw_only.height:
                print("FIRST RAW-ONLY ROWS:")
                print(raw_only.sort(KEY).head(10))
            if tome_only.height:
                print("FIRST TOME-ONLY ROWS:")
                print(tome_only.sort(KEY).head(10))
            if mism.height:
                print("FIRST HEALTH MISMATCHES:")
                print(mism.sort(KEY).head(10))

    print("=" * 100)
    print("ALL", json.dumps(out))


if __name__ == "__main__":
    main()
