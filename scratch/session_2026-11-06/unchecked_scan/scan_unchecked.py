"""Freeze-tick defect scan for the 7,064 unchecked matches, from the local E: raw tree.

Per round (export round index from round_match / round_num; freeze tick from round_freeze):
  dup  = a player_id with more than one distinct team_code in that round's roster (player_info)
  unro = a player alive at the freeze tick (player_status, health > 0) whose player_id has no roster row
Read-only on E: and on the scratch folder inputs. Writes scan_unchecked_defects.csv and scan_unchecked_summary.json.
"""
import csv
import json
import os
import time
import numpy as np
import polars as pl

D = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/unchecked_scan/"
E = "E:/csds_final/csds/"
keys = json.load(open(D + "match_keys.json"))
kid = {k: i for i, k in enumerate(keys)}
rm = np.load(D + "round_match.npy").astype(np.int64)
rn = np.load(D + "round_num.npy").astype(np.int64)
rf = np.load(D + "round_freeze.npy").astype(np.int64)
order = np.argsort(rm, kind="stable")
bounds = np.concatenate([[0], np.cumsum(np.bincount(rm, minlength=len(keys)))])
targets = [r["match_key"] for r in csv.DictReader(open(D + "undetermined_matches.csv"))]
print("targets", len(targets), flush=True)

T0 = time.time()
rows = []
summary = {"matches": len(targets), "matches_folder_missing": 0, "matches_read": 0, "rounds_checked": 0,
           "rounds_no_status_at_freeze": 0, "rounds_dup": 0, "rounds_unro": 0, "rounds_dup_and_unro": 0,
           "rounds_defect": 0, "matches_with_defect": 0, "alive_ct_hist": {}, "alive_n_hist": {}}
for n, mk in enumerate(targets):
    mid = kid[mk]
    idx = order[bounds[mid]:bounds[mid + 1]]
    folder = E + mk[len("csds/"):].rsplit("/", 1)[0] + "/"
    if not os.path.exists(folder + "player_info") or not os.path.exists(folder + "player_status"):
        summary["matches_folder_missing"] += 1
        continue
    summary["matches_read"] += 1
    ip = pl.read_parquet(folder + "player_info", columns=["round", "player_id", "team_code"]).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    fz = pl.DataFrame({"round": [int(rn[i]) for i in idx], "tick": [int(rf[i]) for i in idx]},
                      schema={"round": pl.Int64, "tick": pl.Int64})
    st = pl.read_parquet(folder + "player_status", columns=["tick", "round", "player_id", "health"]).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    alive = st.join(fz, on=["round", "tick"], how="inner").filter(pl.col("health") > 0).select("round", "player_id").unique()
    alive_by = {}
    for r, pid in alive.iter_rows():
        alive_by.setdefault(int(r), set()).add(int(pid))
    roster = {}
    for r, pid, tc in ip.drop_nulls(["round", "player_id"]).iter_rows():
        roster.setdefault(int(r), {}).setdefault(int(pid), set())
        if tc is not None:
            roster[int(r)][int(pid)].add(int(tc))
    for i in idx:
        r = int(rn[i])
        summary["rounds_checked"] += 1
        if r not in alive_by:
            summary["rounds_no_status_at_freeze"] += 1
            continue
        ro = roster.get(r, {})
        dup = sorted(p for p, codes in ro.items() if len(codes) > 1)
        unro = sorted(p for p in alive_by[r] if p not in ro)
        n_alive = len(alive_by[r])
        summary["alive_n_hist"][str(n_alive)] = summary["alive_n_hist"].get(str(n_alive), 0) + 1
        if dup or unro:
            summary["rounds_defect"] += 1
            summary["rounds_dup"] += 1 if dup else 0
            summary["rounds_unro"] += 1 if unro else 0
            summary["rounds_dup_and_unro"] += 1 if (dup and unro) else 0
            rows.append({"match_key": mk, "round": r, "round_index": int(i), "freeze_tick": int(rf[i]),
                         "n_alive": n_alive, "dup_ids": " ".join(map(str, dup)), "unro_ids": " ".join(map(str, unro))})
    if any(x["match_key"] == mk for x in rows[-200:]):
        summary["matches_with_defect"] += 1
    if n % 250 == 0:
        print(n, "matches", round(time.time() - T0), "s", "defect rounds so far", summary["rounds_defect"], flush=True)

with open(D + "scan_unchecked_defects.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["match_key", "round", "round_index", "freeze_tick", "n_alive", "dup_ids", "unro_ids"])
    w.writeheader()
    w.writerows(rows)
summary["matches_with_defect"] = len(set(r["match_key"] for r in rows))
json.dump(summary, open(D + "scan_unchecked_summary.json", "w"), indent=1)
print("SCAN_DONE", json.dumps({k: v for k, v in summary.items() if not k.endswith("hist")}), round(time.time() - T0), "s", flush=True)
