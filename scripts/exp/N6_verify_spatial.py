# N6_verify spatial tie-break test: one-to-one matching where each pickup takes the nearest (2D) earlier unmatched drop
# of the same def_index in the same round (window = round_start..round_end). Tests whether a different tie-break
# explains the claimed reason split and teammate share. Total matched count is invariant to tie-break.
import glob, json
import numpy as np
import polars as pl

BASE = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
OUT = "G:/pureskill/csds-cs2/data/viz/N6_verify_spatial_results.json"

def files(ch):
    return sorted(glob.glob(BASE + "/basic_" + ch + ".COMBINED/dataframe_*"))

def lf26(ch, cols=None):
    lf = pl.scan_parquet(files(ch)).filter(pl.col("match_key").str.starts_with("csds/2026/"))
    if cols:
        lf = lf.select(cols)
    return lf

PRIM_L = sorted({7, 8, 10, 13, 16, 39, 60, 17, 19, 23, 24, 26, 33, 34, 9, 11, 38, 40})

re_all = lf26("round_end", ["match_key", "round", "tick", "player_count"]).collect()
clean = re_all.filter(pl.col("player_count") == 10).select(["match_key", "round", pl.col("tick").alias("round_end_tick")])
N_CLEAN = clean.height
st = lf26("round_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type") == "round_start").group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("round_start_tick")).collect()
clean = clean.join(st, on=["match_key", "round"], how="left")

drops = lf26("item_dropped", ["match_key", "round", "tick", "player_id", "def_index", "reason", "player_team_code",
                              "x_pos", "y_pos"]).collect()
picks = lf26("item_pickup", ["match_key", "round", "tick", "player_id", "def_index",
                             "player_team_code", "player_x_pos", "player_y_pos"]).collect()
d = drops.join(clean, on=["match_key", "round"], how="inner").filter(
    (pl.col("def_index").is_in(PRIM_L)) & (pl.col("tick") >= pl.col("round_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))
p = picks.join(clean, on=["match_key", "round"], how="inner").filter(
    (pl.col("def_index").is_in(PRIM_L)) & (pl.col("tick") >= pl.col("round_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))
d = d.with_row_index("drid")
p = p.with_row_index("pkid")
print("primary drops in window:", d.height, " primary pickups in window:", p.height)

# event table sorted by group then tick; pickups before drops at equal tick
def gk(df):
    return pl.concat_str([pl.col("match_key"), pl.col("round").cast(pl.Utf8), pl.col("def_index").cast(pl.Utf8)], separator="|")
ev_d = d.select([gk(d).alias("g"), pl.col("tick"), pl.lit(1).cast(pl.Int8).alias("kind"), pl.col("drid").cast(pl.Int64).alias("did"),
                 pl.lit(-1).cast(pl.Int64).alias("pid"), pl.col("x_pos").alias("ex"), pl.col("y_pos").alias("ey")])
ev_p = p.select([gk(p).alias("g"), pl.col("tick"), pl.lit(0).cast(pl.Int8).alias("kind"), pl.lit(-1).cast(pl.Int64).alias("did"),
                 pl.col("pkid").cast(pl.Int64).alias("pid"), pl.col("player_x_pos").alias("ex"), pl.col("player_y_pos").alias("ey")])
ev = pl.concat([ev_d, ev_p]).sort(["g", "tick", "kind"])
G = ev["g"].to_list(); K = ev["kind"].to_list(); D = ev["did"].to_list(); P = ev["pid"].to_list()
EX = ev["ex"].to_numpy(); EY = ev["ey"].to_numpy()

avail = []  # list of (drid, x, y)
prev = None
op, od = [], []
for i in range(len(G)):
    g = G[i]
    if g != prev:
        avail = []; prev = g
    if K[i] == 1:
        avail.append((D[i], EX[i], EY[i]))
    elif avail:
        px, py = EX[i], EY[i]
        best = None; bd = None
        for j, (did, x, y) in enumerate(avail):
            if x is None or y is None or px is None or py is None or np.isnan(x) or np.isnan(y) or np.isnan(px) or np.isnan(py):
                dist = 1e18
            else:
                dist = (x - px) ** 2 + (y - py) ** 2
            if bd is None or dist < bd:
                bd = dist; best = j
        did = avail.pop(best)[0]
        op.append(P[i]); od.append(did)

pairs = pl.DataFrame({"pkid": op, "drid": od}, schema={"pkid": pl.Int64, "drid": pl.Int64})
print("spatial one-to-one linked:", pairs.height)

dd = d.select(["drid", "reason", pl.col("player_team_code").alias("dteam"), "tick"]).rename({"tick": "dtick"})
pp = p.select(["pkid", pl.col("player_team_code").alias("pteam"), "tick"]).rename({"tick": "ptick"})
J = pairs.join(dd, on="drid", how="inner").join(pp, on="pkid", how="inner")
res = {"N_clean": N_CLEAN, "n_drops_primary_window": d.height, "n_linked_spatial": J.height,
       "share_linked_spatial": round(J.height / d.height, 4)}
for r in ["death", "manual", "replaced"]:
    sub = J.filter(pl.col("reason") == r)
    n_r = d.filter(pl.col("reason") == r).height
    res["share_" + r] = round(sub.height / n_r, 4)
    res["n_" + r] = sub.height
known = J.filter(pl.col("dteam").is_not_null() & pl.col("pteam").is_not_null())
team = known.filter(pl.col("dteam") == pl.col("pteam")).height
res["teammate_share_known_spatial"] = round(team / known.height, 4)
res["n_known_spatial"] = known.height
res["median_delay_s_spatial"] = round(float(((J["ptick"] - J["dtick"]) / 64.0).median()), 3)
print(json.dumps(res, indent=1))
json.dump(res, open(OUT, "w"), indent=1, default=str)
