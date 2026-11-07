import glob, os
import polars as pl
pi_files = sorted(glob.glob("E:/csds_final/csds/2026/09/*/*/player_info"))
sample = pi_files[::max(1, len(pi_files) // 40)][:40]
rows = []
freeze_multi = 0
for pif in sample:
    d = os.path.dirname(pif)
    mk = d.replace("\\", "/").split("/")[-1][:8]
    pi = pl.read_parquet(pif)
    ros = pi.filter(pl.col("team_code").is_in([2, 3])).group_by(["round", "team_code"]).agg(pl.col("player_id").n_unique().alias("n"))
    rr = ros.pivot(on="team_code", index="round", values="n").fill_null(0)
    bad = rr.filter(~((pl.col("3") == 5) & (pl.col("2") == 5)))
    hdr = pl.read_parquet(os.path.join(d, "header")) if os.path.exists(os.path.join(d, "header")) else None
    wing = None; mapn = None; maxr = None
    if hdr is not None and hdr.height:
        h = hdr.row(0, named=True)
        wing = h.get("is_wingman"); mapn = h.get("map_name"); maxr = h.get("max_rounds")
    rs = pl.read_parquet(os.path.join(d, "round_state"), columns=["round", "tick", "event_type"])
    fe = rs.filter(pl.col("event_type") == "round_freeze_end").group_by("round").agg(pl.len().alias("n"))
    freeze_multi += fe.filter(pl.col("n") > 1).height
    rows.append((mk, rr.height, bad.height, wing, mapn, maxr, sorted(set(zip(bad["3"].to_list(), bad["2"].to_list())))[:4]))
for r in sorted(rows, key=lambda x: -x[2]):
    if r[2] > 0:
        print("match", r[0], "rounds", r[1], "non-5v5 rounds", r[2], "is_wingman", r[3], "map", r[4], "max_rounds", r[5], "(CT,T) seen", r[6])
print("matches sampled", len(rows), "with any non-5v5 round", sum(1 for r in rows if r[2] > 0))
print("matches with 5v5 every round", sum(1 for r in rows if r[2] == 0))
print("rounds with more than one round_freeze_end event (sampled matches):", freeze_multi)
