"""1 Hz in-round state sample (CS2 round win-probability EDA).

Question: does in-round state (health, armor, equipment, weapons, utility, positions) predict the
round winner beyond the coarse event state the bake-off uses? This script only builds the sample.
It fits no model.

Steps
  1. Kept set. The bake-off filters (scripts/model/bakeoff.py main(), steps 1-2 and duplicate
     removal) are ported here, since main() is not importable. Round states come from
     bakeoff.build_states, imported read-only from that file. Counts are checked against
     logs/bakeoff.log and the run stops if they differ.
  2. Sample. Matches the bake-off keeps, days 01-07 first (day = the <DD> folder of the match key).
     random.Random(0).sample of the sorted pool, 150 matches, or all of them if fewer.
     Match-level 80/20 split: random.Random(0).shuffle of the sorted sample, first 20% test.
  3. Clean rounds: the bake-off round filters, applied to the sampled matches.
  4. Snapshots. Per clean round, one per second of live play: ticks freeze_end + 64k with
     tick < round_end tick.
  5. Raw reads. For each sampled match, player_status and player_vector only, column-projected,
     from E:\\csds_final\\csds\\2026\\10\\<DD>\\<id>\\<channel>. Read-only. One read per file, timed.
     Rows are kept for players present (alive) at each snapshot tick.
  6. Output (under G only): data\\state_sample\\players_1hz.parquet and rounds.parquet.
  7. Validation: health and armor ranges, alive count per side against player_death, at most
     5 per side, row counts, sizes and time spent reading E:. Written to logs\\eda.

Run from the project root:
  pixi run python -I -X utf8 scripts\\eda\\extract_state_1hz.py
"""

import bisect
import collections
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
OUT_DIR = ROOT / "data" / "state_sample"
LOG_DIR = ROOT / "logs" / "eda"
E_CSDS = r"E:\csds_final\csds"

SEED = 0
N_SAMPLE = 150
TEST_FRAC = 0.20
TICK_STEP = 64
START = 5
T, CT = 2, 3
DAYS = {"01", "02", "03", "04", "05", "06", "07"}
MATCH_KEY_RE = re.compile(r"^csds/2026/10/(\d{2})/([0-9a-f-]{36})/csds$")

INV_COLS = ["inv_primary", "inv_secondary", "inv_flashbang", "inv_smokegrenade", "inv_hegrenade",
            "inv_molotov", "inv_incgrenade", "inv_decoy", "inv_c4"]
STATUS_COLS = ["tick", "round", "player_id", "health", "armor", "has_helmet", "has_defuser",
               "money", "equipment_value_calc", "is_defusing"] + INV_COLS
VECTOR_COLS = ["tick", "round", "player_id", "team_code", "weapon_code", "x_pos", "y_pos", "z_pos"]
PLAYER_OUT_COLS = ["match_key", "round", "tick", "seconds_since_live", "split", "player_id",
                   "team_code", "health", "armor", "has_helmet", "has_defuser", "money",
                   "equipment_value_calc"] + INV_COLS + ["weapon_code", "x_pos", "y_pos", "z_pos",
                                                         "is_defusing"]
PLAYER_CASTS = {"round": pl.Int32, "tick": pl.Int64, "seconds_since_live": pl.Int32,
                "player_id": pl.Int32, "team_code": pl.Int8, "health": pl.Int16, "armor": pl.Int16,
                "has_helmet": pl.Boolean, "has_defuser": pl.Boolean, "money": pl.Int32,
                "equipment_value_calc": pl.Int32, "weapon_code": pl.Int32, "x_pos": pl.Float32,
                "y_pos": pl.Float32, "z_pos": pl.Float32, "is_defusing": pl.Boolean,
                **{c: pl.Int16 for c in INV_COLS}}
JOIN_CASTS = {"round": pl.Int64, "tick": pl.Int64, "player_id": pl.Int64}

# Port of the bake-off's filter logic imports the bake-off module (read only). Its main() is not run.
_spec = importlib.util.spec_from_file_location("bakeoff", ROOT / "scripts" / "model" / "bakeoff.py")
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)
fv = bk.fv
assert fv.T == T and fv.CT == CT and fv.START == START and fv.TICK_RATE == TICK_STEP


def match_day(mk):
    return MATCH_KEY_RE.match(mk).group(1)


def e_dir(mk):
    parts = mk.split("/")
    return os.path.join(E_CSDS, parts[1], parts[2], parts[3], parts[4])


def bakeoff_kept():
    """Port of bakeoff.py main() steps 1-2 and duplicate removal. Returns the kept rounds."""
    print("== 1. Kept set (bake-off port) ==")
    hdr = fv.read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate",
                             "map_name", "t_starters_score_final", "ct_starters_score_final"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_matches = set(H)
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

    # channel filter
    dropped = set()
    for ch in bk.DENSE + bk.SPARSE:
        df = frames[ch]
        ks = bk.keyset_keys(ch)
        have = set(df["match_key"].unique().to_list())
        dense = ch in bk.DENSE
        miss = (all_matches - ks) | ((all_matches - have) if dense else set())
        dropped |= miss
    kept_matches = all_matches - dropped
    print(f"matches: header {len(all_matches)}; kept after channel filter {len(kept_matches)}")

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
        if x["match_key"] in kept_matches:
            players[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
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

    # round filters, same order as the bake-off
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

    # duplicate matches: identical round_end, player_death and player_info content under two keys
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
    dup_matches_with_rounds = len({k[0] for k in survivors if k[0] in dup_drop})
    reasons["duplicate copy of a kept match (identical rounds, deaths, players)"] += dup_rounds
    survivors = [k for k in survivors if k[0] not in dup_drop]

    # live-window state check (bake-off build_states; unresolved victim side drops the round)
    rounds_ok = []
    n_states = 0
    for k in survivors:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        out = bk.build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                              H[k[0]]["max_rounds"], collections.Counter())
        if out is None:
            reasons["death with unresolved victim side in live window"] += 1
            continue
        n_states += len(out)
        rounds_ok.append(k)
    kept_final = {k[0] for k in rounds_ok}
    rounds_in_kept = len(rd)

    got = {"channel_kept": len(kept_matches), "dup_matches_removed": dup_matches_with_rounds,
           "rounds_in_kept_matches": rounds_in_kept, "kept_rounds": len(rounds_ok),
           "kept_matches": len(kept_final), "states": n_states}
    expect = {"channel_kept": 1709, "dup_matches_removed": 47, "rounds_in_kept_matches": 35502,
              "kept_rounds": 28683, "kept_matches": 1521, "states": 234265}
    for key in expect:
        print(f"  {key}: {got[key]} (bake-off log {expect[key]})")
    assert got == expect, f"kept set does not reproduce bakeoff.log: {got}"
    print("  reproduces logs/bakeoff.log counts: VERIFIED")
    for name, n in reasons.most_common():
        print(f"  dropped round: {name}: {n}")
    return {"H": H, "rd": rd, "live": live, "plants": plants, "players": players,
            "deaths": deaths, "rounds_ok": rounds_ok, "kept_final": kept_final, "reasons": reasons,
            "got": got}


def choose_matches(kept_final):
    pool_all = sorted(kept_final)
    pool = [m for m in pool_all if match_day(m) in DAYS]
    if len(pool) >= N_SAMPLE:
        chosen = sorted(random.Random(SEED).sample(pool, N_SAMPLE))
    else:
        chosen = list(pool)
    shuffled = list(chosen)
    random.Random(SEED).shuffle(shuffled)
    n_test = round(TEST_FRAC * len(chosen))
    test = set(shuffled[:n_test])
    split = {m: ("test" if m in test else "train") for m in chosen}
    day_counts = collections.Counter(match_day(m) for m in pool_all)
    print("== 2. Sample ==")
    print(f"kept matches: {len(pool_all)}; kept in days 01-07: {len(pool)}; "
          f"kept by day: {dict(sorted(day_counts.items()))}")
    print(f"chosen: {len(chosen)} (seed {SEED}); test {n_test}, train {len(chosen) - n_test}")
    return chosen, split, {"pool_days_01_07": len(pool), "kept_matches": len(pool_all),
                           "day_counts_kept": dict(sorted(day_counts.items()))}


def build_rounds_and_snaps(kept, chosen, split):
    H, rd, live, plants, players, deaths = (kept[x] for x in
                                             ("H", "rd", "live", "plants", "players", "deaths"))
    chosen_set = set(chosen)
    round_rows, snap_rows, dtimes = [], [], {}
    bad_winner = 0
    for k in kept["rounds_ok"]:
        m, r = k
        if m not in chosen_set:
            continue
        fz = live[k]["tick"]
        end = rd[k]["end"]
        winner = rd[k]["winner"]
        if winner not in (T, CT):
            bad_winner += 1
        in_win = [t for t in plants.get(k, []) if fz <= t <= end]
        plant = min(in_win) if in_win else None
        round_rows.append({"match_key": m, "round": r, "ct_win": int(winner == CT),
                           "map_name": H[m]["map_name"], "round_number": r,
                           "ct_score": int(live[k]["ct"]), "t_score": int(live[k]["t"]),
                           "plant_tick": plant, "round_end_tick": end,
                           "split": split[m], "live_start_tick": fz})
        for t in range(fz, end, TICK_STEP):
            snap_rows.append((m, r, t, (t - fz) // TICK_STEP))
        # death-based alive ticks for validation (first death per victim in window, bake-off rule)
        seen = set()
        tT, tCT = [], []
        for d in deaths.get(k, []):
            if d["tick"] < fz or d["tick"] > end:
                continue
            if d["pid"] in seen:
                continue
            seen.add(d["pid"])
            side = d["team"] if d["team"] is not None else players.get(k, {}).get(d["pid"])
            if side == T:
                tT.append(d["tick"])
            elif side == CT:
                tCT.append(d["tick"])
        dtimes[k] = (sorted(tT), sorted(tCT))
    assert bad_winner == 0, "winner_team_code outside {2, 3} in a kept round"
    rounds_df = pl.DataFrame(round_rows).sort(["match_key", "round"])
    snaps = pl.DataFrame(snap_rows, schema={"match_key": pl.String, "round": pl.Int64,
                                            "tick": pl.Int64, "seconds_since_live": pl.Int64},
                         orient="row")
    return rounds_df, snaps, dtimes


def read_e(mk, channel, cols, timings):
    path = os.path.join(e_dir(mk), channel)
    t0 = time.perf_counter()
    tbl = pq.read_table(path, columns=cols)
    df = pl.from_arrow(tbl)
    dt = time.perf_counter() - t0
    timings.append({"match_key": mk, "channel": channel, "file_bytes": os.path.getsize(path),
                    "rows": df.height, "read_s": round(dt, 4)})
    return df


def process_match(mk, snaps_m, split_label, timings, diag):
    st = read_e(mk, "player_status", STATUS_COLS, timings).with_columns(
        [pl.col(c).cast(JOIN_CASTS[c]) if c in JOIN_CASTS else pl.col(c) for c in STATUS_COLS])
    vec = read_e(mk, "player_vector", VECTOR_COLS, timings).with_columns(
        [pl.col(c).cast(JOIN_CASTS[c]) if c in JOIN_CASTS else pl.col(c) for c in VECTOR_COLS])
    keys = snaps_m.select(["match_key", "round", "tick", "seconds_since_live"])
    diag["status_rows_read"] += st.height
    diag["vector_rows_read"] += vec.height
    st_dup = st.height - st.unique(["round", "tick", "player_id"]).height
    vec_dup = vec.height - vec.unique(["round", "tick", "player_id"]).height
    diag["status_dup_rows"] += st_dup
    diag["vector_dup_rows"] += vec_dup
    if st_dup:
        st = st.unique(["round", "tick", "player_id"], keep="first", maintain_order=True)
    if vec_dup:
        vec = vec.unique(["round", "tick", "player_id"], keep="first", maintain_order=True)
    st_s = st.join(keys, on=["round", "tick"], how="inner")
    vec_s = vec.join(keys.select(["round", "tick"]), on=["round", "tick"], how="inner")
    key3 = ["round", "tick", "player_id"]
    diag["status_rows_at_snapshots"] += st_s.height
    diag["vector_rows_at_snapshots"] += vec_s.height
    diag["status_rows_without_vector_row"] += st_s.join(vec_s.select(key3), on=key3,
                                                        how="anti").height
    diag["vector_rows_without_status_row"] += vec_s.join(st_s.select(key3), on=key3,
                                                        how="anti").height
    joined = st_s.join(vec_s.select(key3 + ["team_code", "weapon_code", "x_pos", "y_pos", "z_pos"]),
                       on=key3, how="left")
    null_team = joined.filter(pl.col("team_code").is_null()).height
    diag["rows_null_team_dropped"] += null_team
    joined = joined.filter(pl.col("team_code").is_in([T, CT]))
    joined = joined.with_columns(pl.lit(split_label).alias("split"))
    joined = joined.select(PLAYER_OUT_COLS)
    return joined.cast(PLAYER_CASTS)


def e_totals(timings):
    return {"files": len(timings), "bytes": int(sum(t["file_bytes"] for t in timings)),
            "read_s_total": round(float(sum(t["read_s"] for t in timings)), 3),
            "read_s_mean_per_file": round(float(sum(t["read_s"] for t in timings) / max(len(timings), 1)), 4),
            "rows": int(sum(t["rows"] for t in timings))}


def validate(out, snaps_all, dtimes, timings, diag):
    print("== 6. Validation ==")
    res = {}
    # (1) ranges on every output row
    rng = out.select([pl.col("health").min().alias("h_min"), pl.col("health").max().alias("h_max"),
                      pl.col("armor").min().alias("a_min"), pl.col("armor").max().alias("a_max"),
                      (pl.col("health").lt(0) | pl.col("health").gt(100)).sum().alias("h_out"),
                      (pl.col("armor").lt(0) | pl.col("armor").gt(100)).sum().alias("a_out"),
                      pl.len().alias("rows")]).row(0, named=True)
    res["1_ranges"] = {k: (int(v) if v is not None else None) for k, v in rng.items()}
    print(f"(1) health {rng['h_min']}..{rng['h_max']}, rows outside 0-100: {rng['h_out']}; "
          f"armor {rng['a_min']}..{rng['a_max']}, rows outside 0-100: {rng['a_out']}; rows {rng['rows']}")

    # per-snapshot counts per side from the table
    cnt = out.group_by(["match_key", "round", "tick"]).agg(
        (pl.col("team_code") == T).sum().alias("nT"),
        (pl.col("team_code") == CT).sum().alias("nCT"),
        pl.len().alias("n")).sort(["match_key", "round", "tick"])
    cmap = {(r[0], r[1], r[2]): (r[3], r[4], r[5]) for r in cnt.iter_rows()}

    # (2) alive count vs player_death, over every snapshot.
    # Inclusive rule: a victim is dead from its death tick on (deaths with tick <= snapshot tick).
    # Strict rule: the victim is still counted at its death tick (deaths with tick < snapshot tick).
    n_snap = snaps_all.height
    ok_incl = ok_strict = 0
    over5 = 0
    max_side = 0
    mismatch = []
    cat = collections.Counter()
    k_zero = collections.Counter()
    gap_by_round = collections.Counter()
    snap_with_death_tick = 0
    strict_vs_incl_differ = 0
    start_ok = 0
    n_start = 0
    for m, r, t, k in snaps_all.iter_rows():
        tT, tCT = dtimes[(m, r)]
        e_t = START - bisect.bisect_right(tT, t)
        e_ct = START - bisect.bisect_right(tCT, t)
        s_t = START - bisect.bisect_left(tT, t)
        s_ct = START - bisect.bisect_left(tCT, t)
        if (e_t, e_ct) != (s_t, s_ct):
            strict_vs_incl_differ += 1
        nT, nCT, n = cmap.get((m, r, t), (0, 0, 0))
        if n == 0:
            k_zero["snapshots_with_no_alive_rows"] += 1
        max_side = max(max_side, nT, nCT)
        over5 += int(nT > 5 or nCT > 5)
        if k == 0:
            n_start += 1
            start_ok += int((nT, nCT) == (START, START))
        if (nT, nCT) == (e_t, e_ct):
            ok_incl += 1
        else:
            gap_by_round[(m, r)] += 1
            if (nT, nCT) == (s_t, s_ct):
                cat["table lists the victim at its death tick (strict rule only)"] += 1
            elif nT < e_t or nCT < e_ct:
                cat["table has fewer alive than the death count"
                    + (" (both sides fewer)" if nT < e_t and nCT < e_ct else "")] += 1
            else:
                cat["table has more alive than the death count"] += 1
            if len(mismatch) < 25:
                mismatch.append({"match_key": m, "round": r, "tick": t, "k": k, "table_T": nT,
                                 "table_CT": nCT, "death_based_T": e_t, "death_based_CT": e_ct})
        if (nT, nCT) == (s_t, s_ct):
            ok_strict += 1
        if (bisect.bisect_left(tT, t) < bisect.bisect_right(tT, t)
                or bisect.bisect_left(tCT, t) < bisect.bisect_right(tCT, t)):
            snap_with_death_tick += 1
    res["2_alive_check"] = {
        "snapshots": n_snap,
        "agree_inclusive_rule": ok_incl,
        "agree_inclusive_share": round(ok_incl / n_snap, 6),
        "agree_strict_rule": ok_strict,
        "agree_strict_share": round(ok_strict / n_snap, 6),
        "snapshots_where_rules_differ": strict_vs_incl_differ,
        "snapshots_with_a_death_at_that_tick": snap_with_death_tick,
        "differing_by_reason": dict(cat),
        "differing_rounds": sorted({f"{m}|r{r}" for (m, r) in gap_by_round}),
        "first_mismatches_inclusive": mismatch,
        "live_start_5v5": {"ok": start_ok, "of": n_start},
    }
    print(f"(2) snapshots {n_snap}: agree (inclusive rule) {ok_incl} "
          f"({100 * ok_incl / n_snap:.4f}%); agree (strict rule) {ok_strict} "
          f"({100 * ok_strict / n_snap:.4f}%); rules differ on {strict_vs_incl_differ} snapshots, "
          f"{snap_with_death_tick} snapshots have a death at that tick; live-start 5v5 {start_ok} of {n_start}")
    print(f"    differing snapshots by reason: {dict(cat)}; rounds: {len(gap_by_round)}")
    # (3) at most 5 per side
    res["3_max_per_side"] = {"max_players_one_side_one_snapshot": int(max_side),
                             "snapshots_over_5_on_a_side": int(over5)}
    print(f"(3) max alive on one side in one snapshot: {max_side}; snapshots with more than 5: {over5}")
    res["snapshots_with_no_rows"] = dict(k_zero)
    res["diag"] = dict(diag)
    print(f"    diagnostics: {dict(diag)}")
    res["4_e_io"] = e_totals(timings)
    print(f"(4) E reads: {res['4_e_io']}")
    return res, gap_by_round


def main():
    t_all = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    kept = bakeoff_kept()
    chosen, split, sel_info = choose_matches(kept["kept_final"])
    rounds_df, snaps, dtimes = build_rounds_and_snaps(kept, chosen, split)
    print("== 3. Clean rounds and snapshots ==")
    print(f"clean rounds in sample: {rounds_df.height} in {rounds_df['match_key'].n_unique()} matches; "
          f"snapshots (1 per live second): {snaps.height}")
    print(f"rounds by split: {rounds_df.group_by('split').len().sort('split').rows()}")
    print(f"snapshots by split: {snaps.join(rounds_df.select(['match_key', 'round', 'split']), on=['match_key', 'round']).group_by('split').len().sort('split').rows()}")

    print("== 4. Raw reads (player_status, player_vector only) ==")
    timings, diag = [], collections.Counter()
    parts = []
    snaps_by_match = {}
    for mk, sub in snaps.group_by("match_key", maintain_order=True):
        snaps_by_match[mk[0]] = sub.with_columns(
            pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    split_of = {m: split[m] for m in chosen}
    done = 0
    for mk in chosen:
        t0 = time.perf_counter()
        part = process_match(mk, snaps_by_match[mk], split_of[mk], timings, diag)
        parts.append(part)
        done += 1
        if done % 25 == 0 or done == len(chosen):
            print(f"  {done}/{len(chosen)} matches; last match {mk[-45:]} wall {time.perf_counter() - t0:.2f}s")
    out = pl.concat(parts, how="vertical")
    out = out.join(rounds_df.select(["match_key", "round"]).with_columns(
        pl.col("round").cast(pl.Int32)), on=["match_key", "round"], how="inner")
    out = out.sort(["match_key", "round", "tick", "player_id"])
    print(f"  output rows: {out.height}")

    # every snapshot is counted in validation, including any with no rows in the table
    snaps_all = snaps.sort(["match_key", "round", "tick"])
    rounds_df = rounds_df.with_columns(
        pl.col("round").cast(pl.Int64), pl.col("round_number").cast(pl.Int64))
    players_path = OUT_DIR / "players_1hz.parquet"
    rounds_path = OUT_DIR / "rounds.parquet"

    # validation
    val, gap_by_round = validate(
        out, snaps_all.select(["match_key", "round", "tick", "seconds_since_live"])
        .rename({"seconds_since_live": "k"}).with_columns(pl.col("round").cast(pl.Int64)),
        dtimes, timings, diag)
    gaps = [gap_by_round.get((m, r), 0) for m, r in rounds_df.select(["match_key", "round"]).iter_rows()]
    rounds_df = rounds_df.with_columns(pl.Series("alive_gap_snapshots", gaps, dtype=pl.Int32))
    rounds_out = rounds_df.select(["match_key", "round", "ct_win", "map_name", "round_number",
                                   "ct_score", "t_score", "plant_tick", "round_end_tick", "split",
                                   "live_start_tick", "alive_gap_snapshots"])
    out.write_parquet(players_path, compression="zstd")
    rounds_out.write_parquet(rounds_path, compression="zstd")
    val["5_outputs"] = {
        "rounds_with_alive_gap": int((rounds_out["alive_gap_snapshots"] > 0).sum()),
        "players_1hz_rows": out.height,
        "players_1hz_bytes": os.path.getsize(players_path),
        "rounds_rows": rounds_out.height,
        "rounds_bytes": os.path.getsize(rounds_path),
        "rounds_plant_null": int(rounds_out["plant_tick"].is_null().sum()),
        "matches_sampled": len(chosen),
        "split_matches": {s: sum(1 for m in chosen if split[m] == s) for s in ("train", "test")},
        "split_rounds": dict(collections.Counter(rounds_out["split"].to_list())),
        "ct_win_share": round(float(rounds_out["ct_win"].mean()), 4),
    }
    val["selection"] = sel_info
    val["kept_check"] = kept["got"]
    val["kept_reasons"] = dict(kept["reasons"])
    val["wall_s"] = round(time.perf_counter() - t_all, 1)
    with open(LOG_DIR / "state_1hz_validation.json", "w", encoding="utf-8") as fh:
        json.dump(val, fh, indent=1, default=str)
    with open(LOG_DIR / "state_1hz_e_timing.csv", "w", encoding="utf-8") as fh:
        fh.write("match_key,channel,file_bytes,rows,read_s\n")
        for t in timings:
            fh.write(f"{t['match_key']},{t['channel']},{t['file_bytes']},{t['rows']},{t['read_s']}\n")
    print(f"players_1hz: {players_path} ({os.path.getsize(players_path) / 1e6:.2f} MB, "
          f"{out.height} rows); rounds: {rounds_path} ({rounds_out.height} rows)")
    print(f"done in {val['wall_s']}s; validation written to logs\\eda\\state_1hz_validation.json")


if __name__ == "__main__":
    main()
