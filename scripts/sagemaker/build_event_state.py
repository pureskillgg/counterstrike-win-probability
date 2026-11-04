#!/usr/bin/env python3
"""Event-only state table (10 features), split across worker processes.

Same rules as scripts/model/bakeoff.py on the space:
  - match-level channel check (dense and sparse channels, keysets)
  - duplicate matches: identical event signature under two keys, keep the first key in sorted order
  - round filters: 10 players at round end, exactly 5 T and 5 CT at the start, no disconnect or takeover in the round,
    round_start and round_freeze_end rows present, no null score at live start
  - build_states, copied unchanged from bakeoff.py at run time (it is the function that defines the 10 features)

Differences from bakeoff.py, for speed only:
  - duplicates are removed before the build, not after it, so the kept rounds are the same
  - work is split by match across processes; states are written to parquet parts, not held in memory
  - a round with a null max_rounds is skipped and counted (bakeoff.py would fail on it)
  - rounds in removed duplicate matches are not counted in the reasons table

Usage (JupyterLab terminal, foreground):
  ~/venv-dsdk/bin/python build_event_state.py --scratch /mnt/sagemaker-nvme/event_state --shards 16 --workers 16
Test first:
  ~/venv-dsdk/bin/python build_event_state.py --scratch /mnt/sagemaker-nvme/event_state_test --limit 400 --shards 4 --workers 4
"""
import argparse
import ast
import collections
import hashlib
import json
import shutil
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

PIPE = Path("/home/sagemaker-user/pipe/scripts/model")
sys.path.insert(0, str(PIPE))
import feature_validation as fv  # noqa: E402  (read(), TOME, SUFFIXES, T, CT, START, TICK_RATE)

T, CT, START, TICK_RATE = fv.T, fv.CT, fv.START, fv.TICK_RATE
DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state", "team_change"]
SPARSE = ["player_disconnect", "bot_takeover"]
USED = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state",
        "player_disconnect", "bot_takeover"]
COLS = {
    "header": ["key", "match_id", "match_date", "max_rounds", "tick_rate", "t_starters_score_final",
               "ct_starters_score_final"],
    "round_end": ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"],
    "round_start": ["match_key", "round", "tick"],
    "round_state": ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"],
    "player_death": ["match_key", "round", "tick", "player_id", "player_team_code", "weapon_name", "attacker_id"],
    "player_info": ["match_key", "round", "player_id", "team_code"],
    "bomb_state": ["match_key", "round", "tick", "event_type"],
    "team_change": ["match_key", "round", "tick"],
    "player_disconnect": ["match_key", "round", "tick"],
    "bot_takeover": ["match_key", "round", "tick"],
}
FEATURES = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live",
            "seconds_since_plant", "round_number", "ct_score_minus_t_score", "is_pistol_round", "is_overtime"]
STATE_SCHEMA = pa.schema([
    ("match_key", pa.string()), ("round", pa.int64()), ("tick", pa.int64()), ("kind", pa.string()),
    ("alive_ct", pa.int64()), ("alive_t", pa.int64()), ("ct_alive_minus_t", pa.int64()),
    ("bomb_planted", pa.int64()), ("seconds_since_round_live", pa.float64()), ("seconds_since_plant", pa.float64()),
    ("round_number", pa.int64()), ("ct_score_minus_t_score", pa.int64()), ("is_pistol_round", pa.int64()),
    ("is_overtime", pa.int64()), ("label_ct_win", pa.int64()),
])
FLUSH = 200_000


def load_build_states():
    """build_states exactly as written in bakeoff.py (parsed from the file, not retyped)."""
    tree = ast.parse((PIPE / "bakeoff.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_states")
    ns = {"T": T, "CT": CT, "START": START, "TICK_RATE": TICK_RATE}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "bakeoff.build_states", "exec"), ns)
    return ns["build_states"]


def keyset_keys(ch):
    files = sorted(f for s in fv.SUFFIXES for f in (fv.TOME / f"basic_{ch}{s}").glob("keyset_*"))
    if not files:
        raise SystemExit(f"no keyset files for {ch}")
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


def work(job):
    shard_dir, shard_id, max_rounds, part_path = job
    build = load_build_states()
    fr = {ch: pl.read_parquet(f"{shard_dir}/{ch}.parquet") for ch in USED}
    RE, RST, RS = fr["round_end"], fr["round_start"], fr["round_state"]
    DE, PI, BS = fr["player_death"], fr["player_info"], fr["bomb_state"]
    DC, BT = fr["player_disconnect"], fr["bot_takeover"]

    rd = {}
    for x in RE.iter_rows(named=True):
        rd[(x["match_key"], int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                                 "pc": x["player_count"]}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True)}
    live = {}
    for x in RS.iter_rows(named=True):  # already filtered to round_freeze_end
        k = (x["match_key"], int(x["round"]))
        if k not in live:  # first freeze_end wins, as in bakeoff.py
            live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        players[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
    disc = set()
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            disc.add((x["match_key"], int(x["round"])))
    plants = collections.defaultdict(list)
    for x in BS.iter_rows(named=True):  # already filtered to bomb_planted
        plants[(x["match_key"], int(x["round"]))].append(int(x["tick"]))
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        deaths[(x["match_key"], int(x["round"]))].append({
            "tick": int(x["tick"]), "pid": x["player_id"], "team": x["player_team_code"],
            "weapon": x["weapon_name"], "attacker": x["attacker_id"]})

    reasons, diag = collections.Counter(), collections.Counter()
    writer = pq.ParquetWriter(part_path, STATE_SCHEMA, compression="zstd")
    buf, n_states, rounds_ok = [], 0, 0
    for k in sorted(rd):
        base = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if base["pc"] != 10:
            reasons["round_end.player_count != 10"] += 1
            continue
        if not (len(pl_) == 10 and n_t == START and n_ct == START):
            reasons["not exactly 5 T and 5 CT at start (player_info)"] += 1
            continue
        if k in disc:
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
        mr = max_rounds.get(k[0])
        if mr is None:
            reasons["max_rounds null in header"] += 1
            continue
        rr = {**base, "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        local = collections.Counter()
        out = build(k, rr, deaths.get(k, []), plants.get(k, []), pl_, mr, local)
        if out is None:
            reasons["death with unresolved victim side in live window"] += 1
            continue
        diag.update(local)
        rounds_ok += 1
        buf.extend(out)
        if len(buf) >= FLUSH:
            writer.write_table(pa.Table.from_pylist(buf, schema=STATE_SCHEMA))
            n_states += len(buf)
            buf = []
    if buf:
        writer.write_table(pa.Table.from_pylist(buf, schema=STATE_SCHEMA))
        n_states += len(buf)
    writer.close()
    return {"shard": shard_id, "rounds_in_shard": len(rd), "rounds_ok": rounds_ok, "states": n_states,
            "reasons": dict(reasons), "diag": dict(diag), "live_dup_rows": len(RS) - len(live)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/mnt/sagemaker-nvme/event_state")
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="test only: first N kept matches")
    ap.add_argument("--exclude_json", default="", help="JSON with a match_keys list to leave out (holdout B)")
    ap.add_argument("--s3_out", default="", help="optional s3 prefix under adam/, e.g. s3://bucket/adam/tomes/states/")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.scratch)
    shutil.rmtree(out, ignore_errors=True)
    (out / "shards").mkdir(parents=True)
    (out / "parts").mkdir(parents=True)

    hdr = fv.read("header", COLS["header"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_matches = set(H)
    frames = {ch: fv.read(ch, COLS[ch]) for ch in DENSE + SPARSE}
    print(f"loaded {len(frames)} channels and {len(H)} headers in {time.time() - t0:.0f}s", flush=True)

    # 1. match-level channel check, as bakeoff.py
    missing = collections.defaultdict(list)
    for ch in DENSE + SPARSE:
        ks = keyset_keys(ch)
        have = set(frames[ch]["match_key"].unique().to_list())
        miss = (all_matches - ks) | ((all_matches - have) if ch in DENSE else set())
        for m in miss:
            missing[m].append(ch)
    kept = sorted(all_matches - set(missing))
    print(f"channel check: kept {len(kept)} of {len(all_matches)} matches", flush=True)
    n_excluded = 0
    if a.exclude_json:
        excl = set(json.load(open(a.exclude_json))["match_keys"])
        n_excluded = sum(1 for m in kept if m in excl)
        kept = [m for m in kept if m not in excl]
        print(f"excluded {n_excluded} matches listed in {a.exclude_json}; {len(kept)} remain", flush=True)
    if a.limit:
        kept = kept[: a.limit]
    kept_s = pl.Series("k", kept)

    use = {}
    for ch in USED:
        df = frames[ch].filter(pl.col("match_key").is_in(kept_s))
        if ch == "round_state":
            df = df.filter(pl.col("event_type") == "round_freeze_end")
        if ch == "bomb_state":
            df = df.filter(pl.col("event_type") == "bomb_planted")
        use[ch] = df
    del frames

    # 2. duplicate matches: identical event signature under two keys (bakeoff rule, over kept matches)
    sig_parts = collections.defaultdict(list)
    for x in use["round_end"].iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), int(x["winner_team_code"]))))
    for x in use["player_death"].iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    for x in use["player_info"].iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    sig = {m: hashlib.sha1(repr(tuple(sorted(v))).encode()).hexdigest() for m, v in sig_parts.items()}
    groups = collections.defaultdict(list)
    for m in kept:
        groups[sig.get(m, "EMPTY")].append(m)
    dup_drop, n_dup_groups = set(), 0
    for g in groups.values():
        if len(g) > 1:
            n_dup_groups += 1
            dup_drop.update(sorted(g)[1:])
    final = [m for m in kept if m not in dup_drop]
    print(f"duplicates: {n_dup_groups} groups; removed {len(dup_drop)} matches; building {len(final)} matches",
          flush=True)

    # 3. shard by match and write each shard's inputs to NVMe scratch
    shard_keys = [[] for _ in range(a.shards)]
    for i, m in enumerate(final):
        shard_keys[i % a.shards].append(m)
    jobs = []
    for s in range(a.shards):
        sd = out / "shards" / f"s{s:02d}"
        sd.mkdir(parents=True, exist_ok=True)
        ks = pl.Series("k", shard_keys[s])
        for ch in USED:
            use[ch].filter(pl.col("match_key").is_in(ks)).write_parquet(sd / f"{ch}.parquet")
        mr = {m: H[m]["max_rounds"] for m in shard_keys[s]}
        jobs.append((str(sd), s, mr, str(out / "parts" / f"part_{s:02d}.parquet")))
    print(f"shards written in {time.time() - t0:.0f}s; building on {a.workers} workers", flush=True)

    # 4. build in parallel
    with get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(work, jobs, chunksize=1)

    reasons, diag = collections.Counter(), collections.Counter()
    rounds_in, rounds_ok, states = 0, 0, 0
    for r in res:
        reasons.update(r["reasons"])
        diag.update(r["diag"])
        rounds_in += r["rounds_in_shard"]
        rounds_ok += r["rounds_ok"]
        states += r["states"]
    summary = {
        "limit": a.limit, "matches_channel_kept": len(kept), "matches_in_all_headers": len(all_matches),
        "duplicate_groups": n_dup_groups, "duplicate_matches_removed": len(dup_drop),
        "matches_built": len(final), "rounds_in_built_matches": rounds_in, "rounds_kept": rounds_ok,
        "states": states, "features": FEATURES, "dropped_reasons": dict(reasons), "diagnostics": dict(diag),
        "seconds": round(time.time() - t0),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("matches_channel_kept", "duplicate_matches_removed", "matches_built",
                                              "rounds_kept", "states", "seconds")}, indent=1), flush=True)
    print("dropped reasons:", dict(reasons), flush=True)
    if a.s3_out:
        subprocess.run(["aws", "s3", "cp", str(out / "parts"), a.s3_out, "--recursive"], check=True)
        subprocess.run(["aws", "s3", "cp", str(out / "summary.json"), a.s3_out + "summary.json"], check=True)
        print(f"copied to {a.s3_out}", flush=True)


if __name__ == "__main__":
    main()
