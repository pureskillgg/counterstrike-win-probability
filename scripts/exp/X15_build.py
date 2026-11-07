# X15 build: per-second CS2 win-probability dataset from the 1 Hz sample, plus death events.
import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import time
import polars as pl

BASE = "G:/pureskill/csds-cs2/data/state_sample/"
SCR = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad/"
os.makedirs(SCR, exist_ok=True)

# Verified mapping from notes/stats/W1.md Table 0 (inv_primary code -> class)
RIFLE = [7, 8, 10, 13, 16, 39, 60]            # ak47, aug, famas, galilar, m4a1, sg556, m4a1_silencer
AWP = [9]                                     # awp
SMG = [17, 19, 23, 24, 26, 33, 34]            # mac10, p90, mp5sd, ump45, bizon, mp7, mp9
OTHER = [11, 14, 25, 27, 28, 29, 35, 38, 40]  # g3sg1, m249, xm1014, mag7, negev, sawedoff, nova, scar20, ssg08
RIFLE_AWP = RIFLE + AWP
KNOWN = set(RIFLE + AWP + SMG + OTHER + [0])
KEYS = ["match_key", "round", "seconds_since_live"]

t0 = time.time()
p_raw = pl.read_parquet(BASE + "players_1hz.parquet")
r_raw = pl.read_parquet(BASE + "rounds.parquet")
print("loaded players", p_raw.shape, "rounds", r_raw.shape, flush=True)

print("unknown inv_primary codes:", sorted(set(p_raw["inv_primary"].unique().to_list()) - KNOWN))
print("team_code values:", p_raw["team_code"].unique().sort().to_list())
print("overtime player rows (round>24):", p_raw.filter(pl.col("round") > 24).height)

EX_P = pl.col("match_key").str.contains("22e9c3e7") & (pl.col("round") == 16)
print("excluded 4-T round player rows:", p_raw.filter(EX_P).height)

p = (p_raw.filter((~EX_P) & (pl.col("round") <= 24))
     .with_columns([
         pl.col("round").cast(pl.Int32),
         pl.col("seconds_since_live").cast(pl.Int32),
         pl.col("inv_primary").cast(pl.Int32),
         pl.col("inv_primary").is_in(RIFLE_AWP).cast(pl.Int32).alias("is_ra"),
         pl.col("inv_primary").is_in(AWP).cast(pl.Int32).alias("is_awp"),
         (pl.col("armor") > 0).cast(pl.Int32).alias("has_armor"),
     ]))
print("kept player rows", p.height, "matches", p["match_key"].n_unique(), flush=True)

dup = p.group_by(["match_key", "round", "player_id", "seconds_since_live"]).len().filter(pl.col("len") > 1).height
print("duplicate (match,round,player,sec) keys:", dup)

EX_R = pl.col("match_key").str.contains("22e9c3e7") & (pl.col("round") == 16)
r = (r_raw.filter((~EX_R) & (pl.col("round") <= 24))
     .with_columns(pl.col("round").cast(pl.Int32)))
print("kept rounds", r.height, "matches", r["match_key"].n_unique())

# ---- checks on round table ----
print("round == round_number share:", (r["round"] == r["round_number"]).mean())
print("plant_tick null:", r["plant_tick"].is_null().sum(), " plant_tick<=0:", (r["plant_tick"] <= 0).sum(),
      " rounds with plant:", (r["plant_tick"] > 0).sum())
rs = r.sort(["match_key", "round"])
pre = (rs["ct_score"] + rs["t_score"] == rs["round"] - 1).mean()
post = (rs["ct_score"] + rs["t_score"] == rs["round"]).mean()
print("score check: share ct+t == round-1 (pre-round)", pre, " share == round (post-round)", post)
use_pre_fix = post > pre
print("score correction applied (post-round scores detected):", use_pre_fix)

# ---- tick grid checks ----
pc = p.select(["match_key", "round", "seconds_since_live", "tick"]).unique()
pc = pc.join(r.select(["match_key", "round", "live_start_tick", "round_end_tick"]), on=["match_key", "round"], how="left")
print("rows with tick != live_start + 64*sec:",
      pc.filter(pl.col("tick") != pl.col("live_start_tick") + 64 * pl.col("seconds_since_live")).height)
print("rows with tick > round_end_tick:", pc.filter(pl.col("tick") > pl.col("round_end_tick")).height)
fs_chk = p.group_by(["match_key", "round"]).agg(
    pl.col("seconds_since_live").min().alias("fs"),
    pl.col("seconds_since_live").max().alias("mx"),
    pl.col("seconds_since_live").n_unique().alias("nu"))
print("rounds with min sec != 0:", fs_chk.filter(pl.col("fs") != 0).height,
      " rounds with gaps in sec:", fs_chk.filter(pl.col("nu") != pl.col("mx") + 1).height, " of", fs_chk.height)

# ---- team aggregates per tick ----
agg = p.group_by(KEYS + ["team_code"]).agg([
    pl.len().alias("n"),
    pl.col("equipment_value_calc").cast(pl.Float64).sum().alias("eq"),
    pl.col("is_ra").sum().cast(pl.Float64).alias("ra"),
    pl.col("is_awp").sum().cast(pl.Float64).alias("awp"),
    pl.col("has_armor").sum().cast(pl.Float64).alias("arm"),
    pl.col("health").cast(pl.Float64).sum().alias("hp"),
])
DYN = ["n", "eq", "ra", "awp", "arm", "hp"]
ct = agg.filter(pl.col("team_code") == 3).drop("team_code").rename({c: "ct_" + c for c in DYN})
tt = agg.filter(pl.col("team_code") == 2).drop("team_code").rename({c: "t_" + c for c in DYN})
grid = p.group_by(KEYS).agg(pl.col("tick").max().alias("tick"))
grid = grid.join(ct, on=KEYS, how="left").join(tt, on=KEYS, how="left")
dyn_cols = ["ct_" + c for c in DYN] + ["t_" + c for c in DYN]
grid = grid.with_columns([pl.col(c).fill_null(0) for c in dyn_cols])
print("grid rows (ticks with >=1 alive player):", grid.height, flush=True)

# ---- static freeze-end equipment ----
fs = p.group_by(["match_key", "round"]).agg(pl.col("seconds_since_live").min().alias("fs"))
fe = agg.join(fs, on=["match_key", "round"]).filter(pl.col("seconds_since_live") == pl.col("fs"))
fe_ct = fe.filter(pl.col("team_code") == 3).select(
    ["match_key", "round", pl.col("eq").alias("ct_eq_fe"), pl.col("n").alias("ct_n_fe")])
fe_t = fe.filter(pl.col("team_code") == 2).select(
    ["match_key", "round", pl.col("eq").alias("t_eq_fe"), pl.col("n").alias("t_n_fe")])
print("freeze-end alive counts CT:", fe_ct.group_by("ct_n_fe").len().sort("ct_n_fe").to_dicts())
print("freeze-end alive counts T :", fe_t.group_by("t_n_fe").len().sort("t_n_fe").to_dicts())
grid = grid.join(fe_ct, on=["match_key", "round"], how="left").join(fe_t, on=["match_key", "round"], how="left")
grid = grid.with_columns([pl.col("ct_eq_fe").fill_null(0), pl.col("t_eq_fe").fill_null(0)])

# ---- rounds info and base features ----
rr = r.select(["match_key", "round", "ct_win", "round_number", "ct_score", "t_score", "plant_tick",
               "round_end_tick", "live_start_tick", "split"])
g = grid.join(rr, on=["match_key", "round"], how="left")
if use_pre_fix:
    score_lead_expr = pl.col("ct_score") - pl.col("t_score") - (2 * pl.col("ct_win") - 1)
else:
    score_lead_expr = pl.col("ct_score") - pl.col("t_score")
planted = pl.col("plant_tick").is_not_null() & (pl.col("plant_tick") > 0) & (pl.col("tick") >= pl.col("plant_tick"))
g = g.with_columns([
    pl.col("ct_n").cast(pl.Float64).alias("alive_ct"),
    pl.col("t_n").cast(pl.Float64).alias("alive_t"),
    (pl.col("ct_n") - pl.col("t_n")).cast(pl.Float64).alias("ct_alive_minus_t"),
    planted.cast(pl.Float64).alias("bomb_planted"),
    pl.when(planted).then((pl.col("tick") - pl.col("plant_tick")) / 64.0).otherwise(-1.0).alias("seconds_since_plant"),
    pl.col("seconds_since_live").cast(pl.Float64),
    pl.col("round_number").cast(pl.Float64),
    score_lead_expr.cast(pl.Float64).alias("score_lead"),
    pl.col("round_number").is_in([1, 13]).cast(pl.Float64).alias("pistol"),
    pl.col("ct_win").cast(pl.Float64).alias("y"),
])
print("planted grid rows", g.filter(pl.col("bomb_planted") == 1).height)
print("rows with tick != live_start + 64*sec:",
      g.filter(pl.col("tick") != pl.col("live_start_tick") + 64 * pl.col("seconds_since_live")).height)
print("ct_win share of grid rows:", round(g["y"].mean(), 4), " grid rounds:", g.select(["match_key", "round"]).unique().height)
print("grid rows with null ct_win:", g.filter(pl.col("y").is_null()).height)

g = g.sort(["match_key", "round", "seconds_since_live"])
g.write_parquet(SCR + "X15_grid.parquet")
print("wrote grid", g.shape, flush=True)

# ---- death events: present at t, absent at t+1, snapshot at t+1 exists in same round ----
pres = p.select(["match_key", "round", "player_id", "seconds_since_live", "team_code", "inv_primary"])
pa = pres.with_columns((pl.col("seconds_since_live") + 1).alias("sec1"))
ps = pres.select(["match_key", "round", "player_id", pl.col("seconds_since_live").alias("sec1")])
cand = pa.join(ps, on=["match_key", "round", "player_id", "sec1"], how="anti")
print("candidate absences (present t, absent t+1 any):", cand.height)
gk = g.select(["match_key", "round", pl.col("seconds_since_live").cast(pl.Int32).alias("sec1")])
deaths = cand.join(gk, on=["match_key", "round", "sec1"], how="semi")
print("deaths with snapshot at t+1 in same round:", deaths.height,
      " dropped (no t+1 snapshot, round-end boundary):", cand.height - deaths.height)
deaths = deaths.with_columns(
    pl.when(pl.col("inv_primary") == 9).then(pl.lit("AWP"))
    .when(pl.col("inv_primary").is_in(RIFLE)).then(pl.lit("rifle"))
    .when(pl.col("inv_primary") == 0).then(pl.lit("pistol_only"))
    .otherwise(pl.lit("other_primary")).alias("victim_class"),
)
print("death victim class counts:", deaths.group_by("victim_class").len().sort("victim_class").to_dicts())
print("death victim team counts:", deaths.group_by("team_code").len().sort("team_code").to_dicts())
deaths.write_parquet(SCR + "X15_deaths.parquet")
print("build done in", round(time.time() - t0, 1), "s", flush=True)
