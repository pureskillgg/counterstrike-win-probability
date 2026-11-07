# N6_verify: teammate-share variants under the spatial one-to-one matching (tests why 0.4536 is not reproduced)
import glob, json
import numpy as np
import polars as pl
BASE = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
OUT = "G:/pureskill/csds-cs2/data/viz/N6_verify_team_variants_results.json"
def files(ch): return sorted(glob.glob(BASE + "/basic_" + ch + ".COMBINED/dataframe_*"))
def lf26(ch, cols=None):
    lf = pl.scan_parquet(files(ch)).filter(pl.col("match_key").str.starts_with("csds/2026/"))
    return lf.select(cols) if cols else lf
PRIM_L = sorted({7, 8, 10, 13, 16, 39, 60, 17, 19, 23, 24, 26, 33, 34, 9, 11, 38, 40})
re_all = lf26("round_end", ["match_key", "round", "tick", "player_count"]).collect()
clean = re_all.filter(pl.col("player_count") == 10).select(["match_key", "round", pl.col("tick").alias("round_end_tick")])
st = lf26("round_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type") == "round_start").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("round_start_tick")).collect()
clean = clean.join(st, on=["match_key", "round"], how="left")
drops = lf26("item_dropped", ["match_key", "round", "tick", "player_id", "def_index", "reason", "player_team_code", "x_pos", "y_pos"]).collect()
picks = lf26("item_pickup", ["match_key", "round", "tick", "player_id", "def_index", "player_team_code", "player_x_pos", "player_y_pos"]).collect()
d = drops.join(clean, on=["match_key", "round"], how="inner").filter(pl.col("def_index").is_in(PRIM_L) & (pl.col("tick") >= pl.col("round_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick"))).with_row_index("drid")
p = picks.join(clean, on=["match_key", "round"], how="inner").filter(pl.col("def_index").is_in(PRIM_L) & (pl.col("tick") >= pl.col("round_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick"))).with_row_index("pkid")
def gk(df): return pl.concat_str([pl.col("match_key"), pl.col("round").cast(pl.Utf8), pl.col("def_index").cast(pl.Utf8)], separator="|")
ev_d = d.select([gk(d).alias("g"), pl.col("tick"), pl.lit(1).cast(pl.Int8).alias("kind"), pl.col("drid").cast(pl.Int64).alias("did"), pl.lit(-1).cast(pl.Int64).alias("pid"), pl.col("x_pos").alias("ex"), pl.col("y_pos").alias("ey")])
ev_p = p.select([gk(p).alias("g"), pl.col("tick"), pl.lit(0).cast(pl.Int8).alias("kind"), pl.lit(-1).cast(pl.Int64).alias("did"), pl.col("pkid").cast(pl.Int64).alias("pid"), pl.col("player_x_pos").alias("ex"), pl.col("player_y_pos").alias("ey")])
ev = pl.concat([ev_d, ev_p]).sort(["g", "tick", "kind"])
G = ev["g"].to_list(); K = ev["kind"].to_list(); D = ev["did"].to_list(); P = ev["pid"].to_list(); EX = ev["ex"].to_numpy(); EY = ev["ey"].to_numpy()
avail = []; prev = None; op, od = [], []
for i in range(len(G)):
    if G[i] != prev:
        avail = []; prev = G[i]
    if K[i] == 1:
        avail.append((D[i], EX[i], EY[i]))
    elif avail:
        px, py = EX[i], EY[i]; best = 0; bd = None
        for j, (did, x, y) in enumerate(avail):
            dist = 1e18 if (np.isnan(x) or np.isnan(y) or np.isnan(px) or np.isnan(py)) else (x-px)**2 + (y-py)**2
            if bd is None or dist < bd: bd = dist; best = j
        op.append(P[i]); od.append(avail.pop(best)[0])
pairs = pl.DataFrame({"pkid": op, "drid": od}, schema={"pkid": pl.Int64, "drid": pl.Int64})
dd = d.select(["drid", "reason", pl.col("player_team_code").alias("dteam"), pl.col("player_id").alias("dpid")])
pp = p.select(["pkid", pl.col("player_team_code").alias("pteam"), pl.col("player_id").alias("ppid")])
J = pairs.join(dd, on="drid").join(pp, on="pkid")
known = J.filter(pl.col("dteam").is_not_null() & pl.col("pteam").is_not_null())
team = known.filter(pl.col("dteam") == pl.col("pteam"))
self_ = known.filter(pl.col("dpid") == pl.col("ppid"))
team_nonself = team.filter(pl.col("dpid") != pl.col("ppid"))
known_nonself = known.filter(pl.col("dpid") != pl.col("ppid"))
res = {
  "spatial_r2_total": J.height,
  "teammate_share_known": round(team.height / known.height, 4), "n_known": known.height,
  "enemy_share_known": round(1 - team.height / known.height, 4),
  "self_repickups_known": self_.height,
  "teammate_share_excluding_self_repickups": round(team_nonself.height / known_nonself.height, 4), "n_known_nonself": known_nonself.height,
  "teammate_share_by_reason": {r: round(known.filter((pl.col("reason")==r) & (pl.col("dteam")==pl.col("pteam"))).height / known.filter(pl.col("reason")==r).height, 4) for r in ["death","manual","replaced"]},
  "teammate_share_all_r2_null_counted_as_enemy": round(team.height / J.height, 4),
  "claimed_teammate_share": 0.4536,
}
print(json.dumps(res, indent=1))
json.dump(res, open(OUT, "w"), indent=1)
