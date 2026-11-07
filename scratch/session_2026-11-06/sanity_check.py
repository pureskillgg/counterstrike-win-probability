import polars as pl

RAW_CHECK = "/mnt/sagemaker-nvme/raw_check"
KEY = ["round", "tick", "player_id"]

for idx in (1, 2, 3):
    raw = pl.read_parquet(f"{RAW_CHECK}/derived_m{idx}.parquet")
    tome = pl.read_parquet(f"{RAW_CHECK}/tome_m{idx}.parquet")
    shared = raw.join(tome, on=KEY, how="inner", suffix="_tome")
    diff = shared.filter(pl.col("health") != pl.col("health_tome"))
    print(f"m{idx}: raw={raw.height} tome={tome.height} shared={shared.height} health_diff={diff.height}")
    print(f"  raw health min/max={raw['health'].min()}/{raw['health'].max()} tome health min/max={tome['health'].min()}/{tome['health'].max()}")
    print(f"  raw rounds={raw['round'].n_unique()} tome rounds={tome['round'].n_unique()}")
    # mutation test: drop one tome row and alter one health value; the checks must notice both
    t_drop = tome.slice(1)
    lo = t_drop.join(raw.select(KEY), on=KEY, how="anti").height
    ro = raw.join(t_drop.select(KEY), on=KEY, how="anti").height
    t_mut = tome.with_columns(
        pl.when(pl.int_range(pl.len()) == 5).then(pl.col("health") - 1).otherwise(pl.col("health")).alias("health")
    )
    mm = raw.join(t_mut, on=KEY, how="inner", suffix="_tome").filter(pl.col("health") != pl.col("health_tome")).height
    print(f"  mutation test: tome_only after drop={lo} (expect 0, row dropped from tome -> raw_only={ro} expect 1), health mismatch after mutation={mm} (expect 1)")

# pre-use-filter comparison (no use restriction), for information only
rounds = pl.read_parquet("/home/sagemaker-user/adam/status/data/combined/rounds.parquet").select(["match_key", "round", "use"])
import json
import glob

for idx, page_match in ((1, "csds/2026/10/21/24193878-5c8d-429d-b536-e4af4a2a2538/csds"),
                        (2, "csds/2026/10/17/78871936-76dd-4b02-b27c-f267edb977a4/csds"),
                        (3, "csds/2026/10/21/38adf222-ba2e-4ee6-9cd3-1f99a1741e78/csds")):
    raw_all = pl.read_parquet(f"{RAW_CHECK}/derived_m{idx}.parquet")
    use_rounds = rounds.filter((pl.col("match_key") == page_match) & pl.col("use")).select(pl.col("round").cast(pl.Int64))
    dropped_rounds = rounds.filter((pl.col("match_key") == page_match) & (~pl.col("use"))).select(pl.col("round").cast(pl.Int64))
    tome_pre = pl.read_parquet(f"{RAW_CHECK}/tome_m{idx}.parquet").join(use_rounds, on="round", how="semi")
    # tome_m{idx} is already use-filtered; rebuild pre-use tome from the cached page instead
    pages = sorted(glob.glob("/mnt/sagemaker-nvme/parity/c*/dataframe_*"))
    frames = []
    for pg in pages:
        n = pl.scan_parquet(pg).filter(pl.col("match_key") == page_match).select(pl.len()).collect().item()
        if n:
            frames.append(pl.read_parquet(pg).filter(pl.col("match_key") == page_match)
                          .select(["tick", "round", "player_id", "health"])
                          .with_columns(pl.col("tick").cast(pl.Int64), pl.col("round").cast(pl.Int64),
                                        pl.col("player_id").cast(pl.Int64), pl.col("health").cast(pl.Int64)))
    tome_all = pl.concat(frames)
    raw_pre_dropped = raw_all  # raw derived after use filter; recompute pre-use below
    print(f"m{idx} pre-use: tome_all={tome_all.height}; rounds with use=False={dropped_rounds['round'].to_list()}")
    # raw pre-use: rebuild with the same derivation as raw_vs_tome_check.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("chk", f"{RAW_CHECK}/raw_vs_tome_check.py")
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    raw_pre, _, _ = chk.derive_raw(f"{RAW_CHECK}/m{idx}")
    lo = tome_all.join(raw_pre.select(KEY), on=KEY, how="anti")
    ro = raw_pre.join(tome_all.select(KEY).unique(), on=KEY, how="anti")
    sh = raw_pre.join(tome_all.unique(subset=KEY, keep="first"), on=KEY, how="inner", suffix="_tome")
    hd = sh.filter(pl.col("health") != pl.col("health_tome")).height
    print(f"  pre-use: raw={raw_pre.height} tome={tome_all.height} raw_only={ro.height} tome_only={lo.height} health_diff={hd}")
    if lo.height:
        print("  tome_only by round:", lo.group_by("round").len().sort("round").head(30))
