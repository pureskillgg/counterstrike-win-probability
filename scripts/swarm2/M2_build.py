"""M2 build: round-level and state-level tables with map_name, per era.

Ports the match filter, round filters, duplicate-match removal and state builder of
scripts/model/bakeoff.py (2026 kept set). The 2026 run must reproduce the bake-off
counts: 1,521 matches, 28,683 rounds, 234,265 states. The 2024 run uses the same rules with
these ASSUMPTIONS, because the 2024 tome differs:
  - full roster is player_count == 20 (2024 counts both sides; 2026 uses 10). ASSUMPTION.
  - no team_change channel in the 2024 tome, so it is not in the dense check.
  - no max_rounds in the 2024 header. Set to 24 for the pistol and overtime flags only (not
    used by the map analysis). ASSUMPTION.
  - round_start is still required, as in the bake-off.

Outputs (scratchpad, not the project):
  M2_rounds_<era>.parquet   one row per round_end row in a header match, with status and flags
  M2_states_<era>.parquet   states for kept rounds only

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M2_build.py
"""

import collections
import glob
import sys
import time
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("G:/pureskill/csds-cs2")
SCRATCH = Path("C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/"
               "e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
T, CT, START, TICK_RATE = 2, 3, 5, 64
DENSE_2026 = ["round_end", "round_start", "round_state", "player_death", "player_info",
              "bomb_state", "team_change"]
DENSE_2024 = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state"]
SPARSE = ["player_disconnect", "bot_takeover"]

ERAS = {
    "2026": {"base": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
             "suf": ".2026-09-18,2026-10-08", "full_pc": 10, "dense": DENSE_2026, "max_rounds_col": True},
    "2024": {"base": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
             "suf": ".2023-12-28,2024-02-25", "full_pc": 20, "dense": DENSE_2024, "max_rounds_col": False},
}
ASSUMED_MAX_ROUNDS_2024 = 24

SCHEMAS = {
    "round_end": {"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64,
                  "winner_team_code": pl.Int64, "win_reason_code": pl.Int64, "player_count": pl.Int64},
    "round_start": {"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64},
    "round_state": {"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64,
                    "event_type": pl.String, "t_score": pl.Int64, "ct_score": pl.Int64},
    "player_death": {"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64,
                     "player_id": pl.Int64, "player_team_code": pl.Int64},
    "player_info": {"match_key": pl.String, "round": pl.Int64, "player_id": pl.Int64,
                    "team_code": pl.Int64},
    "bomb_state": {"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64, "event_type": pl.String},
    "team_change": {"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64},
    "player_disconnect": {"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64},
    "bot_takeover": {"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64},
}


def pages(base, suf, ch):
    return sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))


def read(base, suf, ch):
    cols = list(SCHEMAS[ch].keys())
    frames = []
    for f in pages(base, suf, ch):
        sch = pl.read_parquet_schema(f)
        if not sch:
            continue
        missing = [c for c in cols if c not in sch]
        if missing:
            raise SystemExit(f"{ch}: missing columns {missing} in {f}")
        frames.append(pl.read_parquet(f, columns=cols))
    if not frames:
        return pl.DataFrame(schema=SCHEMAS[ch])
    out = pl.concat(frames, how="vertical_relaxed")
    return out.cast({k: v for k, v in SCHEMAS[ch].items() if out.schema[k] != v}, strict=False)


def keyset_keys(base, suf, ch):
    files = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "keyset_*")))
    if not files:
        raise SystemExit(f"no keyset files for {ch}")
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


def build_states(key, rr, deaths, plants, players, max_rounds, diag):
    """Ported unchanged from scripts/model/bakeoff.py build_states."""
    frz, end = rr["freeze"], rr["end"]
    win, unresolved = [], 0
    for d in deaths:
        if d["tick"] < frz:
            diag["deaths_pre_live"] += 1
            continue
        if d["tick"] > end:
            diag["deaths_post_end"] += 1
            continue
        side = d["team"]
        if side is None:
            diag["in_window_null_team"] += 1
            side = players.get(d["pid"])
            if side is not None:
                diag["in_window_null_team_imputed"] += 1
        if side not in (T, CT):
            unresolved += 1
            continue
        win.append({**d, "side": side})
    if unresolved:
        return None
    diag["deaths_in_window"] += len(win)
    in_win = [t for t in plants if frz <= t <= end]
    diag["plants_all"] += len(plants)
    diag["plants_in_window"] += len(in_win)
    diag["plants_after_end"] += sum(1 for t in plants if t > end)
    diag["plants_before_live"] += sum(1 for t in plants if t < frz)
    if len(plants) > 1:
        diag["rounds_with_multiple_plants"] += 1
    plant_tick = min(in_win) if in_win else None
    alive = {T: START, CT: START}
    states = [{"tick": frz, "alive_t": START, "alive_ct": START, "kind": "live_start"}]
    seen = set()
    for d in win:
        if d["pid"] in seen:
            diag["in_window_duplicate_victim_skipped"] += 1
            continue
        seen.add(d["pid"])
        alive[d["side"]] -= 1
        states.append({"tick": d["tick"], "alive_t": alive[T], "alive_ct": alive[CT], "kind": "death"})
    out = []
    for s in states:
        planted = int(plant_tick is not None and plant_tick <= s["tick"])
        out.append({
            "match_key": key[0], "round": key[1], "tick": s["tick"], "kind": s["kind"],
            "alive_ct": s["alive_ct"], "alive_t": s["alive_t"],
            "ct_alive_minus_t": s["alive_ct"] - s["alive_t"],
            "bomb_planted": planted,
            "seconds_since_round_live": (s["tick"] - frz) / TICK_RATE,
            "seconds_since_plant": ((s["tick"] - plant_tick) / TICK_RATE) if planted else -1.0,
            "round_number": key[1],
            "ct_score_minus_t_score": rr["ct_score"] - rr["t_score"],
            "is_pistol_round": int(key[1] in (1, max_rounds // 2 + 1)),
            "is_overtime": int(key[1] > max_rounds),
            "label_ct_win": int(rr["winner"] == CT),
        })
    return out


def build_era(name, cfg):
    t0 = time.time()
    base, suf, full_pc = cfg["base"], cfg["suf"], cfg["full_pc"]
    print(f"\n######## ERA {name} ########")
    print(f"base: {base}")
    hcols = ["key", "match_date", "map_name"] + (["max_rounds"] if cfg["max_rounds_col"] else [])
    hdr = pl.concat([pl.read_parquet(f, columns=hcols) for f in sorted(glob.glob(str(base / "header" / "dataframe_*")))],
                    how="vertical_relaxed")
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    if not cfg["max_rounds_col"]:
        for r in H.values():
            r["max_rounds"] = ASSUMED_MAX_ROUNDS_2024
    all_matches = set(H)
    print(f"header matches: {len(all_matches)}; map_name nulls: {sum(1 for r in H.values() if r['map_name'] is None)}")

    frames = {ch: read(base, suf, ch) for ch in SCHEMAS}
    RE, RST, RS = frames["round_end"], frames["round_start"], frames["round_state"]
    DE, BS, PI = frames["player_death"], frames["bomb_state"], frames["player_info"]
    DC, BT = frames["player_disconnect"], frames["bot_takeover"]
    TC = frames["team_change"]
    print(f"round_end rows {RE.height}; unique (match, round) {RE.unique(['match_key', 'round']).height}")

    # ---- 1. channel presence (bake-off rule)
    missing_by_match = collections.defaultdict(list)
    chan_summary = []
    for ch in cfg["dense"] + SPARSE:
        df = frames[ch]
        ks = keyset_keys(base, suf, ch)
        have = set(df["match_key"].unique().to_list())
        zero_rows = all_matches - have
        dense = ch in cfg["dense"]
        no_keyset = all_matches - ks
        miss = no_keyset | (zero_rows if dense else set())
        for m in miss:
            missing_by_match[m].append(ch)
        chan_summary.append((ch, "dense" if dense else "sparse", df.height, len(zero_rows), len(no_keyset),
                             len(miss)))
    for row in chan_summary:
        print(f"  {row[0]:<20} {row[1]:<6} rows={row[2]:>8} zero_row_matches={row[3]:>5} "
              f"not_in_keyset={row[4]:>5} counted_missing={row[5]:>5}")
    dropped_matches = set(missing_by_match)
    kept_matches = all_matches - dropped_matches
    print(f"matches dropped for a missing channel: {len(dropped_matches)}; kept at this stage: {len(kept_matches)}")

    # ---- duplicate signatures (bake-off)
    sig_parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in sig_parts.items()}

    # ---- 2. per-round lookups restricted to kept matches (bake-off)
    rd_all = {}
    for x in RE.iter_rows(named=True):
        m = x["match_key"]
        if m in all_matches:
            rd_all[(m, int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                            "pc": x["player_count"]}
    rd = {k: v for k, v in rd_all.items() if k[0] in kept_matches}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True)
                if x["match_key"] in all_matches}
    live = {}
    live_dups = 0
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        k = (m, int(x["round"]))
        if k in live:
            live_dups += 1
            continue
        live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    live_all = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        live_all.setdefault((x["match_key"], int(x["round"])), {"t": x["t_score"], "ct": x["ct_score"]})
    players = collections.defaultdict(dict)
    pi_dup_rows = 0
    for x in PI.iter_rows(named=True):
        m = x["match_key"]
        if m not in kept_matches:
            continue
        k = (m, int(x["round"]))
        if x["player_id"] in players[k]:
            pi_dup_rows += 1
        players[k][x["player_id"]] = x["team_code"]
    players_all = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        players_all[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
    disc_rounds = set()
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            if x["match_key"] in kept_matches:
                disc_rounds.add((x["match_key"], int(x["round"])))
    plants = collections.defaultdict(list)
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        if x["match_key"] in kept_matches:
            plants[(x["match_key"], int(x["round"]))].append(int(x["tick"]))
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        m = x["match_key"]
        if m in kept_matches:
            deaths[(m, int(x["round"]))].append({"tick": int(x["tick"]), "pid": x["player_id"],
                                                 "team": x["player_team_code"]})
    print(f"live-start rows (kept matches): {len(live)}; duplicate freeze rows skipped: {live_dups}; "
          f"player_info duplicate rows: {pi_dup_rows}")

    # bomb outcome per round for the side-vs-winner check (all matches)
    outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    bomb_side = {}
    for x in outcome.iter_rows(named=True):
        bomb_side[(x["match_key"], int(x["round"]))] = 3 if "bomb_defused" in x["evs"] else 2
    planted_rounds = {(x["match_key"], int(x["round"])) for x in
                      BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True)}

    # ---- round filters (bake-off order)
    reasons = collections.Counter()
    status = {}
    survivors = []
    for k in sorted(rd):
        rr = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if rr["pc"] != full_pc:
            reasons["round_end.player_count != full roster"] += 1
            status[k] = "player_count"
            continue
        if not (len(pl_) == 10 and n_t == START and n_ct == START):
            reasons["not exactly 5 T and 5 CT at start (player_info)"] += 1
            status[k] = "roster_5v5"
            continue
        if k in disc_rounds:
            reasons["disconnect or bot takeover event in round"] += 1
            status[k] = "disconnect"
            continue
        if k not in rst_keys:
            reasons["no round_start row"] += 1
            status[k] = "no_round_start"
            continue
        if k not in live:
            reasons["no round_freeze_end (live start) row"] += 1
            status[k] = "no_freeze"
            continue
        if live[k]["t"] is None or live[k]["ct"] is None:
            reasons["null score at live start"] += 1
            status[k] = "null_score"
            continue
        survivors.append(k)

    # ---- duplicate matches (bake-off: keep lowest key per identical signature group)
    sig_groups = collections.defaultdict(list)
    for m in sorted(kept_matches):
        sig_groups[sigs.get(m, ())].append(m)
    dedup_groups = [g for g in sig_groups.values() if len(g) > 1]
    dup_drop = set()
    for g in dedup_groups:
        dup_drop.update(g[1:])
    dup_rounds = sum(1 for k in survivors if k[0] in dup_drop)
    reasons["duplicate copy of a kept match (identical rounds, deaths, players)"] += dup_rounds
    for k in survivors:
        if k[0] in dup_drop:
            status[k] = "duplicate_match"
    survivors = [k for k in survivors if k[0] not in dup_drop]
    print(f"duplicate groups among channel-kept matches: {len(dedup_groups)}; matches removed as copies: "
          f"{len(dup_drop)}; rounds removed: {dup_rounds}")

    diag = collections.Counter()
    state_rows = []
    rounds_ok = []
    for k in survivors:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        local = collections.Counter()
        out = build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                           H[k[0]]["max_rounds"], local)
        if out is None:
            reasons["death with unresolved victim side in live window"] += 1
            status[k] = "unresolved_victim_side"
            continue
        diag.update(local)
        state_rows.extend(out)
        rounds_ok.append(k)
        status[k] = "kept"
    kept_final_matches = {k[0] for k in rounds_ok}
    print("round filter reasons (kept matches):")
    for n, c in reasons.most_common():
        print(f"  dropped: {n}: {c}")
    print(f"rounds in kept matches (after roster/player_count filters): {len(rd)}")
    print(f"rounds kept: {len(rounds_ok)} in {len(kept_final_matches)} matches")
    S = pl.DataFrame(state_rows)
    print(f"states: {S.height} (live start {int((S['kind'] == 'live_start').sum())}, "
          f"death {int((S['kind'] == 'death').sum())})")
    print(f"diagnostics: {dict(diag)}")

    # ---- round-level table for all round_end rows in header matches
    recs = []
    for k, v in rd_all.items():
        m = k[0]
        h = H[m]
        lv = live_all.get(k)
        st = status.get(k, "not_in_kept_match" if m not in kept_matches else "dropped_other")
        if m in dropped_matches:
            st = "dropped_channel"
        recs.append({
            "era": name, "match_key": m, "round": k[1], "map_name": h["map_name"],
            "match_date": h["match_date"], "max_rounds": h["max_rounds"],
            "winner": v["winner"], "player_count": v["pc"],
            "label_ct_win": int(v["winner"] == CT),
            "t_score_live": None if lv is None else lv["t"],
            "ct_score_live": None if lv is None else lv["ct"],
            "score_lead_live": None if lv is None or lv["t"] is None or lv["ct"] is None else lv["ct"] - lv["t"],
            "status": st,
            "kept": st == "kept",
            "bomb_side": bomb_side.get(k),
            "bomb_mismatch": None if k not in bomb_side else int(bomb_side[k] != v["winner"]),
            "has_plant": int(k in planted_rounds),
        })
    R = pl.DataFrame(recs)
    print(f"round table rows: {R.height}; status counts: {R.group_by('status').agg(pl.len()).sort('status').rows()}")
    S = S.with_columns(pl.lit(name).alias("era"),
                       pl.col("match_key").map_elements(lambda m: H[m]["map_name"], return_dtype=pl.String)
                       .alias("map_name"),
                       pl.col("match_key").map_elements(lambda m: H[m]["match_date"], return_dtype=pl.String)
                       .alias("match_date"))
    print(f"build time {time.time() - t0:.1f}s")
    return R, S, {"matches_kept_channels": len(kept_matches), "matches_final": len(kept_final_matches),
                  "rounds_kept": len(rounds_ok), "states": S.height, "reasons": dict(reasons)}


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, cfg in ERAS.items():
        R, S, info = build_era(name, cfg)
        R.write_parquet(SCRATCH / f"M2_rounds_{name}.parquet")
        S.write_parquet(SCRATCH / f"M2_states_{name}.parquet")
        summary[name] = info
    print("\n== summary ==")
    for k, v in summary.items():
        print(k, v)
    print(f"wrote parquet to {SCRATCH}")


if __name__ == "__main__":
    main()
