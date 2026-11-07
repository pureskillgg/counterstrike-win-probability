"""M4 build: rebuild the bake-off state table (2026 tome) and attach map, planter and site fields.

Reuses the bake-off's own helpers (fv.read, keyset_keys, build_states, make_splits, constants) by
loading scripts/model/bakeoff.py by path. Its main() is not called. The filter and state loop below
is a copy of bakeoff.main() sections 1 to 5, so the kept counts can be checked against the bake-off:
expected 1,521 matches, 28,683 rounds, 234,265 states.

Outputs go to the scratchpad (not under E: or the project tree):
  states_2026.parquet  one row per state, with map_name, batch flag, plant fields, split tags
  rounds_2026.parquet  one row per kept round, with map, outcome, planter position, site code
  plants_all_2026.parquet  every bomb_planted row in every header match (for site-code checks)
"""
import collections
import importlib.util
import os

import polars as pl

BK = r"G:\pureskill\csds-cs2\scripts\model\bakeoff.py"
SCR = (r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
       r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M4")
os.makedirs(SCR, exist_ok=True)

spec = importlib.util.spec_from_file_location("bakeoff_mod", BK)
bk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bk)
fv = bk.fv
T, CT, START, TICK_RATE = bk.T, bk.CT, bk.START, bk.TICK_RATE
WIN_BOMB_EXPLODED, WIN_BOMB_DEFUSED = 1, 7


def main():
    hdr = fv.read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate", "map_name"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_matches = set(H)
    print(f"header matches: {len(all_matches)}")

    RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                               "player_count"])
    RST = fv.read("round_start", ["match_key", "round", "tick"])
    RS = fv.read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code",
                                  "weapon_name", "attacker_id"])
    BS = fv.read("bomb_state", ["match_key", "round", "tick", "event_type", "site_code", "player_id",
                                "player_team_code", "player_x_pos", "player_y_pos", "player_z_pos"])
    DC = fv.read("player_disconnect", ["match_key", "round", "tick"])
    BT = fv.read("bot_takeover", ["match_key", "round", "tick"])
    TC = fv.read("team_change", ["match_key", "round", "tick"])
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
    frames = {"round_end": RE, "round_start": RST, "round_state": RS, "player_death": DE,
              "player_info": PI, "bomb_state": BS, "team_change": TC,
              "player_disconnect": DC, "bot_takeover": BT}

    # ---- 1. match-level channel check (bakeoff section 1)
    missing_by_match = collections.defaultdict(list)
    for ch in bk.DENSE + bk.SPARSE:
        df = frames[ch]
        ks = bk.keyset_keys(ch)
        have = set(df["match_key"].unique().to_list())
        no_keyset = all_matches - ks
        zero_rows = all_matches - have
        dense = ch in bk.DENSE
        miss = no_keyset | (zero_rows if dense else set())
        for m in miss:
            missing_by_match[m].append(ch)
    kept_matches = all_matches - set(missing_by_match)
    print(f"matches dropped for a missing channel: {len(missing_by_match)}; kept: {len(kept_matches)}")

    # ---- duplicate signature (bakeoff section 1)
    sig_parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]),
                                               int(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]),
                                               str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]),
                                               str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in sig_parts.items()}

    # ---- 2. round filters (bakeoff section 2)
    rd = {}
    for x in RE.iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            rd[(m, int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                        "pc": x["player_count"]}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True)
                if x["match_key"] in kept_matches}
    live = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        k = (m, int(x["round"]))
        if k in live:
            continue
        live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        players[(m, int(x["round"]))][x["player_id"]] = x["team_code"]
    disc_rounds = set()
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            if x["match_key"] in kept_matches:
                disc_rounds.add((x["match_key"], int(x["round"])))
    plants = collections.defaultdict(list)
    plant_rows = collections.defaultdict(list)
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        if x["match_key"] in kept_matches:
            k = (x["match_key"], int(x["round"]))
            plants[k].append(int(x["tick"]))
            plant_rows[k].append(x)
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            deaths[(m, int(x["round"]))].append({"tick": int(x["tick"]), "pid": x["player_id"],
                                                 "team": x["player_team_code"],
                                                 "weapon": x["weapon_name"],
                                                 "attacker": x["attacker_id"]})
    reasons = collections.Counter()
    survivors = []
    for k in sorted(rd):
        rr = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if rr["pc"] != 10:
            reasons["round_end.player_count != 10"] += 1
            continue
        if not (len(pl_) == 10 and n_t == START and n_ct == START):
            reasons["not exactly 5 T and 5 CT at start"] += 1
            continue
        if k in disc_rounds:
            reasons["disconnect or bot takeover"] += 1
            continue
        if k not in rst_keys:
            reasons["no round_start row"] += 1
            continue
        if k not in live:
            reasons["no live start"] += 1
            continue
        if live[k]["t"] is None or live[k]["ct"] is None:
            reasons["null score at live start"] += 1
            continue
        survivors.append(k)
    pre_final = {k[0] for k in survivors}
    sig_groups = collections.defaultdict(list)
    for m in sorted(kept_matches):
        sig_groups[sigs.get(m, ())].append(m)
    dup_drop = set()
    for g in [g for g in sig_groups.values() if len(g) > 1]:
        dup_drop.update(g[1:])
    dup_rounds = sum(1 for k in survivors if k[0] in dup_drop)
    reasons["duplicate copy of a kept match"] += dup_rounds
    survivors = [k for k in survivors if k[0] not in dup_drop]
    print(f"matches with rounds before dedupe: {len(pre_final)}; duplicate matches removed: "
          f"{len(dup_drop)}")
    for name, n in reasons.most_common():
        print(f"  dropped: {name}: {n}")

    # ---- 2b. states (bakeoff section 2, build_states is the bake-off function itself)
    diag = collections.Counter()
    state_rows, rounds_ok = [], []
    for k in survivors:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        local = collections.Counter()
        out = bk.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                              H[k[0]]["max_rounds"], local)
        if out is None:
            reasons["death with unresolved victim side in live window"] += 1
            continue
        diag.update(local)
        state_rows.extend(out)
        rounds_ok.append(k)
    S = pl.DataFrame(state_rows)
    kept_final = {k[0] for k in rounds_ok}
    print(f"kept matches: {len(kept_final)}; kept rounds: {len(rounds_ok)}; states: {S.height}")
    print("expected (bake-off report): 1521 matches, 28683 rounds, 234265 states")

    # ---- 5. splits (same function and seed as the bake-off)
    sets = bk.make_splits(kept_final, H)
    prim_split = {m: part for part in ("train", "val", "test") for m in sets[bk.PRIMARY][part]}
    chro_split = {m: part for part in ("train", "val", "test") for m in sets[bk.CHRONO][part]}
    print("primary split matches:", {p: len(sets[bk.PRIMARY][p]) for p in ("train", "val", "test")})
    print("chrono split matches:", {p: len(sets[bk.CHRONO][p]) for p in ("train", "val", "test")})

    # ---- 3. attach map, batch and split tags to states
    meta = pl.DataFrame({"match_key": list(kept_final)})
    meta = meta.with_columns(
        pl.col("match_key").map_elements(lambda m: H[m]["map_name"], return_dtype=pl.Utf8).alias("map_name"),
        pl.col("match_key").map_elements(lambda m: H[m]["match_date"], return_dtype=pl.Utf8).alias("match_date"),
        pl.col("match_key").map_elements(lambda m: prim_split[m], return_dtype=pl.Utf8).alias("split_primary"),
        pl.col("match_key").map_elements(lambda m: chro_split[m], return_dtype=pl.Utf8).alias("split_chrono"),
    )
    meta = meta.with_columns((pl.col("match_date").str.starts_with("2026-10-08")).alias("batch_1008"))

    # ---- planter fields per kept round: the plant row whose tick equals the bake-off plant tick
    round_rows = []
    for k in rounds_ok:
        rr = rd[k]
        frz, end = live[k]["tick"], rr["end"]
        in_win = [t for t in plants.get(k, []) if frz <= t <= end]
        ptick = min(in_win) if in_win else None
        row = {"match_key": k[0], "round": k[1], "freeze_tick": frz, "end_tick": end,
               "winner": rr["winner"], "win_reason_code": None,
               "plant_tick": ptick, "plant_site_code": None, "plant_x": None, "plant_y": None,
               "plant_z": None, "planter_team": None, "plant_rows_in_round": len(plants.get(k, []))}
        if ptick is not None:
            prow = [x for x in plant_rows[k] if int(x["tick"]) == ptick][0]
            row.update({"plant_site_code": prow["site_code"], "plant_x": prow["player_x_pos"],
                        "plant_y": prow["player_y_pos"], "plant_z": prow["player_z_pos"],
                        "planter_team": prow["player_team_code"]})
        round_rows.append(row)
    R = pl.DataFrame(round_rows)
    wr = RE.select(["match_key", "round", "win_reason_code"]).with_columns(pl.col("round").cast(pl.Int64))
    R = R.with_columns(pl.col("round").cast(pl.Int64)).join(wr, on=["match_key", "round"], how="left",
                                                           suffix="_re")
    R = R.with_columns(pl.col("win_reason_code_re").alias("win_reason_code")).drop("win_reason_code_re")
    R = R.join(meta, on="match_key", how="left")
    R = R.with_columns((pl.col("winner") == CT).cast(pl.Int8).alias("ct_win"))
    R.write_parquet(os.path.join(SCR, "rounds_2026.parquet"))
    print(f"rounds table: {R.height} rows; planted rounds: {int(R['plant_tick'].is_not_null().sum())}")

    # ---- state table with plant fields for states after the plant
    rd_plant = R.select(["match_key", "round", "plant_tick", "plant_site_code", "plant_x", "plant_y",
                         "plant_z", "map_name"]).with_columns(pl.col("round").cast(pl.Int64))
    S2 = S.with_columns(pl.col("round").cast(pl.Int64)).join(rd_plant, on=["match_key", "round"],
                                                            how="left")
    S2 = S2.join(meta, on="match_key", how="left")
    S2.write_parquet(os.path.join(SCR, "states_2026.parquet"))
    print(f"states table: {S2.height} rows, columns: {S2.columns}")

    # ---- every bomb_planted row in every header match (kept or not), for site-code checks
    pa_rows = []
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        m = x["match_key"]
        pa_rows.append({"match_key": m, "round": int(x["round"]), "tick": int(x["tick"]),
                        "site_code": x["site_code"], "planter_team": x["player_team_code"],
                        "x": x["player_x_pos"], "y": x["player_y_pos"], "z": x["player_z_pos"],
                        "map_name": H[m]["map_name"] if m in H else None,
                        "match_date": H[m]["match_date"] if m in H else None,
                        "kept": m in kept_final})
    P = pl.DataFrame(pa_rows)
    P.write_parquet(os.path.join(SCR, "plants_all_2026.parquet"))
    print(f"plant rows (all header matches): {P.height}; kept: {int(P['kept'].sum())}")
    print("diagnostics:", dict(diag))


if __name__ == "__main__":
    main()
