"""M3 build: rebuild the bake-off kept set on the 2026-10-01..08 tome and attach map_name.

Reuses by path (no edits to the originals):
  scripts/model/feature_validation.py  -> fv.read, fv.TOME, fv.TICK_RATE
  scripts/model/bakeoff.py             -> bk.keyset_keys, bk.build_states, bk.make_splits,
                                          bk.FEATS, bk.MONO, bk.DENSE, bk.SPARSE, bk.PRIMARY, bk.CHRONO
The channel check, duplicate-match removal and round filters are ported from bakeoff.main()
sections 1 and 2 with the same logic. map_name comes from the header tome (ASSUMPTION: it is the
played map; the header is the only map source in the tome).

Output (scratchpad, not the project tree):
  M3_states.parquet   one row per state, FEATS + label + map_name + primary/chrono split
  M3_matches.parquet  one row per kept match
Checks against logs/bakeoff.log: 1,521 matches, 28,683 rounds, 234,265 states,
primary split train/val/test 1095/122/304 matches and 167,379/19,170/47,716 states.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M3_build.py
"""

import collections
import importlib.util
import sys
import time
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fv = load_module("feature_validation", ROOT / "scripts" / "model" / "feature_validation.py")
bk = load_module("bakeoff_m3", ROOT / "scripts" / "model" / "bakeoff.py")
T, CT, START = fv.T, fv.CT, fv.START


def main():
    t0 = time.time()
    hdr = fv.read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate",
                             "t_starters_score_final", "ct_starters_score_final", "map_name"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_matches = set(H)
    assert set(collections.Counter(r["tick_rate"] for r in H.values())) == {fv.TICK_RATE}
    print(f"header matches: {len(all_matches)}; map_name nulls: "
          f"{sum(1 for r in H.values() if r['map_name'] is None)}")

    RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                               "player_count"])
    RST = fv.read("round_start", ["match_key", "round", "tick"])
    RS = fv.read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code",
                                  "weapon_name", "attacker_id"])
    BS = fv.read("bomb_state", ["match_key", "round", "tick", "event_type"])
    DC = fv.read("player_disconnect", ["match_key", "round", "tick"])
    BT = fv.read("bot_takeover", ["match_key", "round", "tick"])
    TC = fv.read("team_change", ["match_key", "round", "tick"])
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
    frames = {"round_end": RE, "round_start": RST, "round_state": RS, "player_death": DE,
              "player_info": PI, "bomb_state": BS, "team_change": TC,
              "player_disconnect": DC, "bot_takeover": BT}

    # ---- 1. match-level channel check (ported)
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
    dropped_matches = set(missing_by_match)
    kept_matches = all_matches - dropped_matches
    print(f"matches dropped for a missing channel: {len(dropped_matches)}; kept at this stage: "
          f"{len(kept_matches)}")

    # ---- 2. round filters (ported)
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
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        if x["match_key"] in kept_matches:
            plants[(x["match_key"], int(x["round"]))].append(int(x["tick"]))
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
            reasons["not exactly 5 T and 5 CT at start (player_info)"] += 1
            continue
        if k in disc_rounds:
            reasons["disconnect or bot takeover event in round"] += 1
            continue
        if k not in rst_keys:
            reasons["no round_start row"] += 1
            continue
        if k not in live:
            reasons["no round_freeze_end (live start) row"] += 1
            continue
        if live[k]["t"] is None or live[k]["ct"] is None:
            reasons["null score at live start"] += 1
            continue
        survivors.append(k)

    # ---- duplicate matches (ported)
    pre_final = {k[0] for k in survivors}
    pre_sets = bk.make_splits(pre_final, H)
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
    sig_groups = collections.defaultdict(list)
    for m in sorted(kept_matches):
        sig_groups[sigs.get(m, ())].append(m)
    dedup_groups = [g for g in sig_groups.values() if len(g) > 1]
    dup_drop = set()
    for g in dedup_groups:
        dup_drop.update(g[1:])
    dup_rounds = sum(1 for k in survivors if k[0] in dup_drop)
    reasons["duplicate copy of a kept match (identical rounds, deaths, players)"] += dup_rounds
    survivors = [k for k in survivors if k[0] not in dup_drop]
    print(f"duplicate groups: {len(dedup_groups)}; duplicate matches removed: {len(dup_drop)}; "
          f"rounds removed: {dup_rounds}")
    for name, n in reasons.most_common():
        print(f"  dropped: {name}: {n}")

    # ---- states (ported via bakeoff.build_states)
    diag = collections.Counter()
    state_rows = []
    rounds_ok = []
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
    S = pl.DataFrame(state_rows).with_row_index("row_idx")
    kept_final_matches = {k[0] for k in rounds_ok}
    print(f"rounds kept: {len(rounds_ok)} in {len(kept_final_matches)} matches")
    print(f"states: {S.height}; live start {int((S['kind'] == 'live_start').sum())}; "
          f"death {int((S['kind'] == 'death').sum())}")

    # ---- map_name and splits
    mk = sorted(kept_final_matches)
    match_tbl = pl.DataFrame({
        "match_key": mk,
        "map_name": [H[m]["map_name"] for m in mk],
        "match_date": [H[m]["match_date"] for m in mk],
        "max_rounds": [H[m]["max_rounds"] for m in mk],
    })
    sets = bk.make_splits(kept_final_matches, H)
    for name in (bk.PRIMARY, bk.CHRONO):
        part_of = {}
        for part in ("train", "val", "test"):
            for m in sets[name][part]:
                part_of[m] = part
        match_tbl = match_tbl.with_columns(
            pl.col("match_key").replace_strict(part_of, default=None).alias(f"split_{name}"))
    S = S.join(match_tbl.select(["match_key", "map_name", "split_primary", "split_chronological"]),
               on="match_key", how="left").sort("row_idx")
    assert S["map_name"].null_count() == 0, "a state has no map_name"
    assert S["split_primary"].null_count() == 0, "a state has no primary split"

    n_rounds_by_match = {}
    for k in rounds_ok:
        n_rounds_by_match[k[0]] = n_rounds_by_match.get(k[0], 0) + 1
    n_states_by_match = S.group_by("match_key").agg(pl.len().alias("n_states"))
    match_tbl = match_tbl.with_columns(
        pl.col("match_key").replace_strict(n_rounds_by_match, default=0).alias("n_rounds")
    ).join(n_states_by_match, on="match_key", how="left")

    # ---- checks against the bake-off log
    print("\n== checks against logs/bakeoff.log ==")
    print(f"matches {len(kept_final_matches)} (expect 1521); rounds {len(rounds_ok)} "
          f"(expect 28683); states {S.height} (expect 234265)")
    for name in (bk.PRIMARY,):
        for part in ("train", "val", "test"):
            sub = S.filter(pl.col("split_primary") == part)
            print(f"  primary {part}: matches {sub['match_key'].n_unique()}, states {sub.height}")
    print("  expect primary train 1095 matches / 167379 states; val 122 / 19170; test 304 / 47716")

    print("\n== map_name in the kept set ==")
    by_map = (match_tbl.group_by("map_name")
              .agg(pl.len().alias("matches"), pl.col("n_rounds").sum().alias("rounds"),
                   pl.col("n_states").sum().alias("states"))
              .sort("matches", descending=True))
    ct_by_map = S.group_by("map_name").agg(pl.col("label_ct_win").mean().alias("ct_share_states"))
    by_map = by_map.join(ct_by_map, on="map_name", how="left").sort("matches", descending=True)
    print(by_map)
    print(f"maps in kept set: {by_map.height}")

    S.write_parquet(SCRATCH / "M3_states.parquet")
    match_tbl.write_parquet(SCRATCH / "M3_matches.parquet")
    print(f"\nwrote {SCRATCH / 'M3_states.parquet'} and {SCRATCH / 'M3_matches.parquet'}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
