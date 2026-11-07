# verify_A2_4 step 04: what else round 1 would need in build 9842 (player_info, player_death, bomb rows, freeze_end),
# checked in the G: staging_2024_full raw copy for all 1957 build-9842 matches, and in the tome.
# Also: match-end win-panel round label vs header total vs last round_end label (tail completeness check).
# Read-only. Output: stdout only.
import os, json, collections
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
G_ROOT = r"G:\pureskill\csds-cs2\staging_2024_full\csds\2024\02"


def chan_frame(name, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + name]
    assert len(d) == 1, (name, d)
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect()
             for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]


hdir = os.path.join(TOME, "header")
hm = json.load(open(os.path.join(hdir, "tome"), encoding="utf8"))
H = pl.concat([pl.read_parquet(os.path.join(hdir, p["dataframe"]["dataframe"]))
               .select(["key", "build_num", "t_starters_score_final", "ct_starters_score_final"])
               for p in hm["pages"]], how="vertical_relaxed")
rows = H.to_dicts()
BUILD = {r["key"]: r["build_num"] for r in rows}
HTOT = {r["key"]: (r["t_starters_score_final"] or 0) + (r["ct_starters_score_final"] or 0) for r in rows}

# (1) tome: round label of the match-end win panel and max labels
st = chan_frame("round_state", ["round", "event_type", "match_key"])
cwp = st.filter(pl.col("event_type") == "cs_win_panel_match").group_by("match_key").agg(
    pl.col("round").min().alias("cwp_round"), pl.len().alias("n_cwp"))
rmax = st.group_by("match_key").agg(pl.col("round").max().alias("state_max"))
re_t = chan_frame("round_end", ["round", "match_key"]).group_by("match_key").agg(pl.col("round").max().alias("re_max"),
                                                                                  pl.len().alias("n_re"))
T = pl.DataFrame({"match_key": list(BUILD.keys()), "build": list(BUILD.values()), "htot": list(HTOT.values())})
T = T.join(cwp, on="match_key", how="left").join(rmax, on="match_key", how="left").join(re_t, on="match_key", how="left")
print("== (1) tome tail check, per build")
for b in sorted(set(BUILD.values())):
    s = T.filter(pl.col("build") == b)
    print(f"build {b}: n={s.height}; n_cwp per match {s.group_by('n_cwp').len().sort('n_cwp').rows()}")
    print(f"   cwp_round - htot: {s.with_columns((pl.col('cwp_round') - pl.col('htot')).alias('d')).group_by('d').len().sort('d').rows()}")
    print(f"   state_max - re_max: {s.with_columns((pl.col('state_max') - pl.col('re_max')).alias('d')).group_by('d').len().sort('d').rows()}")
    print(f"   htot - re_max: {s.with_columns((pl.col('htot') - pl.col('re_max')).alias('d')).group_by('d').len().sort('d').rows()}")
    print(f"   cwp_round - re_max: {s.with_columns((pl.col('cwp_round') - pl.col('re_max')).alias('d')).group_by('d').len().sort('d').rows()}")

# (2) round-1 channel presence for build 9842 in the G staging copy (all matches with a file)
print("\n== (2) G staging_2024_full: round-1 rows by channel, build 9842 (matches with file / with round-1 rows)")
chans = ["round_state", "round_start", "round_end", "player_info", "player_death", "bomb_state", "weapon_fire", "player_spawn"]
cnt = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r["build_num"] != 9842:
        continue
    key = r["key"]
    dd, mid = key.split("/")[3], key.split("/")[4]
    for ch in chans:
        f = os.path.join(G_ROOT, dd, mid, ch)
        if not os.path.isfile(f):
            continue
        cnt[ch][0] += 1
        df = pl.read_parquet(f, columns=["round"])
        if (df["round"] == 1).any():
            cnt[ch][1] += 1
for ch in chans:
    print(f"  {ch:<13} files {cnt[ch][0]:>5}; matches with a round-1 row {cnt[ch][1]:>5}")

# (3) tome player_info round coverage (5v5 check input) for build 9842 round 1
pi = chan_frame("player_info", ["match_key", "round", "player_id", "team_code"])
p1 = pi.filter(pl.col("round") == 1)["match_key"].unique().to_list()
print("\n== (3) tome player_info round-1 matches:", collections.Counter(BUILD.get(m) for m in p1))
print("DONE feasibility")
