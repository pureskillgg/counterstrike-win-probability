"""Score candidate side rules for the freeze-tick fix on kill consistency.

A kill between two rostered players on opposite sides is consistent; same side is a team kill (rare).
Rules compared on the 283 fixed rounds of the 23 matches:
  R_roster : first team_code per (round, player) from player_info   (what v2/v3 used)
  R_pv     : team_code per (round, player) from player_vector (mode; conflicts counted)
  R_place  : the override in freeze_fix/override_rounds.csv (place_name at freeze, earlier fix)
"""
import json
import os
import numpy as np
import polars as pl

RAW = "/home/sagemaker-user/adam/builder/raw_cache_bucket/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
inv = json.load(open(FX + "inv_matches.json"))
ov = pl.read_csv(FX + "override_rounds.csv", infer_schema_length=0).with_columns(pl.col("round").cast(pl.Int64))
print("override rows", ov.height, "status", ov["status"].unique().to_list())

res = {"R_roster": [0, 0, 0], "R_pv": [0, 0, 0], "R_place": [0, 0, 0]}  # opp, same, unknown
conflicts = []
side_rows = {"R_roster": 0, "R_pv": 0}
per_round_rows = []

for key in inv:
    d = RAW + os.path.dirname(key) + "/"
    pv = pl.read_parquet(d + "player_vector", columns=["round", "player_id", "team_code"])
    pinfo = pl.read_parquet(d + "player_info", columns=["round", "player_id", "team_code"])
    pdth = pl.read_parquet(d + "player_death", columns=["round", "player_id", "attacker_id"])

    pv = pv.filter(pl.col("team_code").is_not_null() & pl.col("player_id").is_not_null())
    tpv = (pv.group_by(["round", "player_id", "team_code"]).len())
    nconf = tpv.group_by(["round", "player_id"]).len().filter(pl.col("len") > 1)
    if nconf.height:
        conflicts.append((key, nconf.height))
    rpv = (tpv.sort("len", descending=True).unique(subset=["round", "player_id"], keep="first")
           .select(["round", "player_id", "team_code"]))
    rroster = pinfo.unique(subset=["round", "player_id"], keep="first").select(["round", "player_id", "team_code"])

    ovk = ov.filter((pl.col("match_key") == key) & (pl.col("status") == "fixed"))
    rounds_fixed = sorted(set(int(r) for r in ovk["round"].to_list()))
    for r in rounds_fixed:
        dr = pdth.filter(pl.col("round") == r)
        mp = {"pv": {}, "roster": {}, "place": {}}
        for pid, tc in rpv.filter(pl.col("round") == r).select(["player_id", "team_code"]).iter_rows():
            mp["pv"][int(pid)] = int(tc)
        for pid, tc in rroster.filter(pl.col("round") == r).select(["player_id", "team_code"]).iter_rows():
            mp["roster"][int(pid)] = int(tc)
        row = ovk.filter(pl.col("round") == r).row(0, named=True)
        for k, v in json.loads(row["override"]).items():
            mp["place"][int(k)] = int(v)
        # side rows: players whose override side differs from roster and from player_vector
        for name, rule in (("R_roster", mp["roster"]), ("R_pv", mp["pv"])):
            diff = sum(1 for pid, tc in mp["place"].items() if rule.get(pid) not in (None, tc))
            side_rows[name] += diff
        per_round_rows.append((key, r, len(mp["place"]),
                               sum(1 for pid, tc in mp["place"].items() if mp["pv"].get(pid) not in (None, tc)),
                               sum(1 for pid, tc in mp["place"].items() if mp["roster"].get(pid) not in (None, tc))))
        for att, vic in dr.select(["attacker_id", "player_id"]).iter_rows():
            if att is None or vic is None or att == vic:
                continue
            for name, rule in (("R_roster", mp["roster"]), ("R_pv", mp["pv"]), ("R_place", mp["place"])):
                a, v = rule.get(int(att)), rule.get(int(vic))
                if a is None or v is None:
                    res[name][2] += 1
                elif a != v:
                    res[name][0] += 1
                else:
                    res[name][1] += 1

print("player_vector team conflicts (round,player) across 23 matches:", sum(c for _, c in conflicts), "in", len(conflicts), "matches")
for name, (opp, same, unk) in res.items():
    tot = opp + same
    print(f"{name}: kills opposite={opp} same-team={same} unknown={unk} consistency={opp / tot if tot else float('nan'):.4f}")
print("side rows differing from override: roster", side_rows["R_roster"], " player_vector", side_rows["R_pv"])
pr = pl.DataFrame(per_round_rows, schema=["match_key", "round", "n_players", "diff_pv", "diff_roster"], orient="row")
print("rounds:", pr.height, "rounds where override != pv:", pr.filter(pl.col("diff_pv") > 0).height,
      "override != roster:", pr.filter(pl.col("diff_roster") > 0).height)
pr.write_csv(FX + "side_rule_check.csv")
