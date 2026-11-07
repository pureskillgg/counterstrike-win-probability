# N6_verify extra checks: 23-28% acquisition range, post-round decomposition, buy-phase timing, team-code consistency.
import glob, json
from collections import deque
import numpy as np
import polars as pl

BASE = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
OUT = "G:/pureskill/csds-cs2/data/viz/N6_verify_extra_results.json"

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
    pl.col("event_type").is_in(["round_freeze_end", "round_start"])).collect()
rs_all = (st.filter(pl.col("event_type") == "round_start").group_by(["match_key", "round"])
          .agg(pl.col("tick").min().alias("round_start_tick")))
fz_all = (st.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"])
          .agg(pl.col("tick").min().alias("freeze_end_tick")))
# next round start (round+1) for each round, all 2026 rounds
nxt = rs_all.with_columns((pl.col("round") - 1).alias("round")).rename({"round_start_tick": "next_round_start_tick"})
clean = clean.join(rs_all, on=["match_key", "round"], how="left").join(fz_all, on=["match_key", "round"], how="left") \
             .join(nxt, on=["match_key", "round"], how="left")

drops_all = lf26("item_dropped", ["match_key", "round", "tick", "player_id", "entity_id", "def_index", "item",
                                  "reason", "player_team_code"]).collect()
picks_all = lf26("item_pickup", ["match_key", "round", "tick", "player_id", "def_index", "item",
                                 "player_team_code"]).collect()
res = {"N_clean": N_CLEAN}

# ---- (a) all-def drops by reason per round, in-window ----
bj = clean.select(["match_key", "round", "round_start_tick", "round_end_tick"])
dw = drops_all.join(bj, on=["match_key", "round"], how="inner").filter(
    (pl.col("tick") >= pl.col("round_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))
by_reason_all = dw.group_by("reason").len().sort("reason")
res["a_drops_all_defs_by_reason_window"] = {r: {"n": n, "per_round": round(n / N_CLEAN, 4)} for r, n in by_reason_all.rows()}
print("(a) drops all defs by reason (window):", res["a_drops_all_defs_by_reason_window"])

# ---- window-level pickups (all defs) by freeze / live ----
pw = picks_all.join(bj, on=["match_key", "round"], how="inner").filter(
    (pl.col("tick") >= pl.col("round_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))
pw = pw.join(fz_all, on=["match_key", "round"], how="left")
pw_fz = pw.with_columns((pl.col("tick") < pl.col("freeze_end_tick")).fill_null(False).alias("in_freeze"))
print("(a2) pickups all defs in window:", pw.height, " in freeze:", pw_fz.filter(pl.col("in_freeze")).height)

# ---- linking helper (FIFO, one-to-one, same match/round/def) ----
def link(dc, pc):
    def gkey(df):
        return pl.concat_str([pl.col("match_key"), pl.col("round").cast(pl.Utf8), pl.col("def_index").cast(pl.Utf8)], separator="|")
    dc = dc.with_row_index("drid").with_columns(pl.col("drid").cast(pl.Int64))
    pc = pc.with_row_index("pkid").with_columns(pl.col("pkid").cast(pl.Int64))
    ev_d = dc.select([gkey(dc).alias("g"), pl.col("tick"), pl.lit(1).cast(pl.Int8).alias("kind"),
                      pl.col("drid").alias("did"), pl.lit(-1).cast(pl.Int64).alias("pid")])
    ev_p = pc.select([gkey(pc).alias("g"), pl.col("tick"), pl.lit(0).cast(pl.Int8).alias("kind"),
                      pl.lit(-1).cast(pl.Int64).alias("did"), pl.col("pkid").alias("pid")])
    ev = pl.concat([ev_d, ev_p]).sort(["g", "tick", "kind"])
    G = ev["g"].to_list(); K = ev["kind"].to_list(); D = ev["did"].to_list(); P = ev["pid"].to_list()
    avail = deque(); prev = None; op = []; od = []
    for g, k, d, p in zip(G, K, D, P):
        if g != prev:
            avail.clear(); prev = g
        if k == 1:
            avail.append(d)
        elif avail:
            op.append(p); od.append(avail.popleft())
    return pl.DataFrame({"pkid": op, "drid": od}, schema={"pkid": pl.Int64, "drid": pl.Int64}), dc, pc

# primary, window
dcp = dw.filter(pl.col("def_index").is_in(PRIM_L))
pcp = pw.filter(pl.col("def_index").is_in(PRIM_L))
pairs, dcp_i, pcp_i = link(dcp, pcp)
n_pick_prim = pcp.height
n_linked_prim = pairs.height
# upper bound: pickups with ANY earlier same-round same-def drop (no one-to-one)
dmin = dcp.group_by(["match_key", "round", "def_index"]).agg(pl.col("tick").min().alias("min_drop_tick"))
pu = pcp.join(dmin, on=["match_key", "round", "def_index"], how="left").filter(
    pl.col("min_drop_tick").is_not_null() & (pl.col("min_drop_tick") < pl.col("tick")))
res["b_primary_pickups_window"] = n_pick_prim
res["b_one_to_one_linked_primary"] = n_linked_prim
res["b_one_to_one_share_of_primary_pickups"] = round(n_linked_prim / n_pick_prim, 4)
res["b_upper_bound_pickups_with_any_earlier_same_def_drop"] = pu.height
res["b_upper_bound_share_of_primary_pickups"] = round(pu.height / n_pick_prim, 4)
print("(b) primary pickups:", n_pick_prim, " one-to-one:", n_linked_prim, round(n_linked_prim/n_pick_prim, 4),
      " upper-bound any-earlier:", pu.height, round(pu.height/n_pick_prim, 4))

# unlinked primary pickups: freeze timing (buy-phase proxy)
linked_pk = set(pairs["pkid"].to_list())
unl = pcp_i.filter(~pl.col("pkid").is_in(list(linked_pk))).join(fz_all, on=["match_key", "round"], how="left")
unl_fz = unl.filter(pl.col("tick") < pl.col("freeze_end_tick")).height
res["c_unlinked_primary_pickups"] = unl.height
res["c_unlinked_primary_pickups_in_freeze"] = unl_fz
res["c_unlinked_primary_in_freeze_share"] = round(unl_fz / unl.height, 4) if unl.height else None
print("(c) unlinked primary pickups:", unl.height, " in freeze:", unl_fz, round(unl_fz/unl.height, 4))

# ---- (d) post-round decomposition under tag-based (no window) ----
dc_all_prim = drops_all.join(clean.select(["match_key", "round", "round_start_tick", "round_end_tick", "next_round_start_tick"]),
                             on=["match_key", "round"], how="inner").filter(pl.col("def_index").is_in(PRIM_L))
pc_all_prim = picks_all.join(clean.select(["match_key", "round", "round_start_tick", "round_end_tick", "next_round_start_tick"]),
                             on=["match_key", "round"], how="inner").filter(pl.col("def_index").is_in(PRIM_L))
pairs_n, dn, pn = link(dc_all_prim, pc_all_prim)
dn = dn.join(pairs_n, on="drid", how="left").with_columns(
    ((pl.col("tick") >= pl.col("round_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick"))).alias("drop_in_window"))
pn = pn.join(pairs_n, on="pkid", how="left").with_columns(
    ((pl.col("tick") >= pl.col("round_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick"))).alias("pick_in_window"))
dn = dn.join(pn.select(["pkid", "tick", "pick_in_window"]).rename({"tick": "pick_tick"}), on="pkid", how="left")
dn_linked = dn.filter(pl.col("pkid").is_not_null())
cat = dn_linked.with_columns(
    pl.when(pl.col("drop_in_window") & pl.col("pick_in_window")).then(pl.lit("in_in"))
    .when(pl.col("drop_in_window") & ~pl.col("pick_in_window")).then(pl.lit("in_drop_post_round_pick"))
    .when(~pl.col("drop_in_window") & ~pl.col("pick_in_window")).then(pl.lit("post_drop_post_pick"))
    .otherwise(pl.lit("post_drop_in_pick")).alias("cat"))
cat_counts = cat.group_by("cat").len().sort("cat").rows()
res["d_no_window_linked_primary_by_category"] = {c: n for c, n in cat_counts}
res["d_no_window_linked_primary_total"] = pairs_n.height
res["d_no_window_drops_primary_total"] = dc_all_prim.height
res["d_post_round_primary_drops_excluded_by_window"] = dc_all_prim.filter(~((pl.col("tick") >= pl.col("round_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))).height
print("(d) no-window linked primary by category:", cat_counts, " total linked:", pairs_n.height)

# between-round check: are post-round pickups before the next round start?
post_p = pc_all_prim.filter(pl.col("tick") > pl.col("round_end_tick"))
bet = post_p.filter(pl.col("next_round_start_tick").is_not_null() & (pl.col("tick") < pl.col("next_round_start_tick"))).height
res["d2_post_round_primary_pickups"] = post_p.height
res["d2_post_round_primary_pickups_before_next_round_start"] = bet
res["d2_share_before_next_round_start"] = round(bet / post_p.height, 4) if post_p.height else None
print("(d2) post-round primary pickups:", post_p.height, " before next round_start:", bet)

# ---- (e) team-code consistency within (match, round, player_id) ----
tc = pl.concat([drops_all.join(bj, on=["match_key", "round"], how="inner").filter(
                    (pl.col("tick") >= pl.col("round_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))
                .select(["match_key", "round", "player_id", "player_team_code"]),
                pw.select(["match_key", "round", "player_id", "player_team_code"])]).drop_nulls()
per_pid = tc.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").n_unique().alias("n_codes"))
res["e_pid_round_groups"] = per_pid.height
res["e_pid_round_groups_with_multiple_team_codes"] = per_pid.filter(pl.col("n_codes") > 1).height
print("(e) (match,round,player_id) groups:", per_pid.height, " with >1 team code:", per_pid.filter(pl.col("n_codes") > 1).height)

# ---- (f) teammate share in one-to-one primary pairs, excluding null team, by candidate definition ----
pdr = dcp_i.select(["drid", "player_team_code", "reason"]).rename({"player_team_code": "dteam"})
ppk = pcp_i.select(["pkid", "player_team_code"]).rename({"player_team_code": "pteam"})
pp = pairs.join(pdr, on="drid").join(ppk, on="pkid")
known = pp.filter(pl.col("dteam").is_not_null() & pl.col("pteam").is_not_null())
team = known.filter(pl.col("dteam") == pl.col("pteam")).height
res["f_teammate_share_known_fifo"] = round(team / known.height, 4)
res["f_enemy_share_known_fifo"] = round(1 - team / known.height, 4)
res["f_n_known_fifo"] = known.height
res["f_teammate_share_all_r2_incl_null_as_enemy"] = round(team / pp.height, 4)
print("(f) teammate share known:", round(team/known.height, 4), " enemy:", round(1-team/known.height, 4),
      " all r2 incl null:", round(team/pp.height, 4), " n_known:", known.height)

json.dump(res, open(OUT, "w"), indent=1, default=str)
print("wrote", OUT)
