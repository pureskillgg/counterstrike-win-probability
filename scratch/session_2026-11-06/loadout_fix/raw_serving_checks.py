import glob, os
import polars as pl
pi_files = sorted(glob.glob("E:/csds_final/csds/2026/09/*/*/player_info"))
print("September 2026 match folders found", len(pi_files))
sample = pi_files[::max(1, len(pi_files) // 40)][:40]
print("matches sampled", len(sample))
team_codes = {}
roster_hist = {}
multi_team = 0
rounds_total = 0
rounds_bad = 0
bad_examples = []
st_rows = 0; st_dup = 0; st_health0 = 0; st_fake = 0; st_pid_ne_fixed = 0
st_missing_roster = 0; st_roster_join_rows = 0
grid_rounds = 0; rounds_no_frz = 0; rounds_zero_grid_rows = 0; grid_rows_at_frz = 0; rounds_frz_row_missing = 0
side_other = 0
for pif in sample:
    d = os.path.dirname(pif)
    pi = pl.read_parquet(pif)
    for k, v in pi["team_code"].value_counts().iter_rows():
        team_codes[k] = team_codes.get(k, 0) + v
    mt = pi.group_by(["round", "player_id"]).agg(pl.col("team_code").n_unique().alias("n_team")).filter(pl.col("n_team") > 1).height
    multi_team += mt
    side_other += pi.filter(~pl.col("team_code").is_in([2, 3])).height
    ros = pi.filter(pl.col("team_code").is_in([2, 3])).group_by(["round", "team_code"]).agg(pl.col("player_id").n_unique().alias("n"))
    rr = ros.pivot(on="team_code", index="round", values="n").fill_null(0)
    for r in rr.iter_rows(named=True):
        rounds_total += 1
        ct, t = r.get("3", 0), r.get("2", 0)
        key = (int(ct), int(t))
        roster_hist[key] = roster_hist.get(key, 0) + 1
        if not (ct == 5 and t == 5):
            rounds_bad += 1
            if len(bad_examples) < 6: bad_examples.append((os.path.basename(os.path.dirname(d)), d.split("/")[-1][:8], r["round"], ct, t))
    ps = pl.read_parquet(os.path.join(d, "player_status"), columns=["tick", "round", "player_id", "health", "is_fake_player", "player_id_fixed"])
    st_rows += ps.height
    st_dup += ps.group_by(["round", "tick", "player_id"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1).height
    st_health0 += ps.filter(pl.col("health") == 0).height
    st_fake += ps.filter(pl.col("is_fake_player") == True).height
    st_pid_ne_fixed += ps.filter(pl.col("player_id") != pl.col("player_id_fixed")).height
    rs = pl.read_parquet(os.path.join(d, "round_state"), columns=["round", "tick", "event_type"])
    frz = rs.filter(pl.col("event_type") == "round_freeze_end").group_by("round").agg(pl.col("tick").min().alias("frz"))
    fz = dict(frz.iter_rows())
    rounds_in_ps = ps["round"].unique().to_list()
    pj = ps.join(pi.filter(pl.col("team_code").is_in([2, 3])).select(["round", "player_id"]).unique().with_columns(pl.lit(1).alias("rostered")), on=["round", "player_id"], how="left")
    st_missing_roster += pj.filter(pl.col("rostered").is_null()).height
    st_roster_join_rows += pj.height
    for r in rounds_in_ps:
        grid_rounds += 1
        f = fz.get(r)
        if f is None:
            rounds_no_frz += 1; continue
        sub = ps.filter(pl.col("round") == r)
        g = sub.filter(((pl.col("tick") - f) % 64 == 0) & (pl.col("tick") >= f))
        if g.height == 0:
            rounds_zero_grid_rows += 1
        if sub.filter(pl.col("tick") == f).height == 0:
            rounds_frz_row_missing += 1
print("team_code value counts over sampled player_info rows", team_codes)
print("player_info rows with team_code not in (2,3):", side_other)
print("(round, player_id) with more than one team_code:", multi_team)
print("rounds seen in roster", rounds_total, "rounds with CT or T roster != 5:", rounds_bad)
print("roster (CT,T) histogram:", sorted(roster_hist.items(), key=lambda kv: -kv[1])[:10])
print("bad-roster examples (match, round, CT, T):", bad_examples)
print("player_status rows", st_rows, "| duplicate (round,tick,player) keys", st_dup, "| health==0 rows", st_health0, "| is_fake_player rows", st_fake, "| player_id != player_id_fixed rows", st_pid_ne_fixed)
print("status rows with no roster row for (round, player_id):", st_missing_roster, "of", st_roster_join_rows)
print("rounds with status:", grid_rounds, "| no freeze_end in round_state:", rounds_no_frz, "| no status row on freeze grid (tick>=frz, tick-frz%64==0):", rounds_zero_grid_rows, "| no status row exactly at freeze_end tick:", rounds_frz_row_missing)
