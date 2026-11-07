"""A10 verification of the 1 Hz state sample (read-only for E:; writes only under logs\\swarm2).

Part A  invariants on data\\state_sample\\players_1hz.parquet and rounds.parquet
        (snapshot grid, live window, before round_end, seconds_since_live origin).
Part B  independent recomputation from raw E: channels for the 150 sampled matches:
        live start and round end from round_state / round_end, snapshot presence from
        player_status (team from player_vector), death-based alive counts from player_death
        (inclusive and strict rules), roster presence at the live start, and an
        elimination-end check (losing side distinct in-window victims).
Nothing in the extractor is imported. Run:
  <pixi python> -I -X utf8 scripts\\swarm2\\A10_verify_sample.py
"""
import bisect
import collections
import json
import os
import sys
import time

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"G:\pureskill\csds-cs2"
E_CSDS = r"E:\csds_final\csds"
ROUNDS = os.path.join(ROOT, "data", "state_sample", "rounds.parquet")
PLAYERS = os.path.join(ROOT, "data", "state_sample", "players_1hz.parquet")
OUT_JSON = os.path.join(ROOT, "logs", "swarm2", "A10_verify_sample.json")
TICK = 64
START = 5
T, CT = 2, 3

res = {}
t0 = time.perf_counter()


def e_dir(mk):
    p = mk.split("/")
    return os.path.join(E_CSDS, p[1], p[2], p[3], p[4])


def rd(mk, ch, cols):
    path = os.path.join(e_dir(mk), ch)
    names = pq.ParquetDataset(path).schema.names
    use = [c for c in cols if c in names]
    df = pl.from_arrow(pq.read_table(path, columns=use))
    return df


rounds = pl.read_parquet(ROUNDS)
out = pl.read_parquet(PLAYERS).with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))

# ---------------- Part A: invariants on the output files ----------------
print("== Part A: invariants on output files ==")
A = {}
A["rounds_rows"] = rounds.height
A["rounds_unique_key"] = rounds.select(["match_key", "round"]).unique().height
A["players_rows"] = out.height
A["players_unique_key"] = out.select(["match_key", "round", "tick", "player_id"]).unique().height
A["matches_in_rounds"] = rounds["match_key"].n_unique()
A["matches_in_players"] = out["match_key"].n_unique()
A["live_start_ge_round_end"] = int((rounds["live_start_tick"] >= rounds["round_end_tick"]).sum())
A["ticks_not_multiple_of_64_from_live_start_in_players"] = int(
    out.join(rounds.select(["match_key", "round", "live_start_tick"]), on=["match_key", "round"], how="inner")
    .filter(((pl.col("tick") - pl.col("live_start_tick")) % TICK) != 0).height)
A["rows_with_tick_below_live_start"] = int(
    out.join(rounds.select(["match_key", "round", "live_start_tick"]), on=["match_key", "round"], how="inner")
    .filter(pl.col("tick") < pl.col("live_start_tick")).height)
A["rows_with_tick_at_or_after_round_end"] = int(
    out.join(rounds.select(["match_key", "round", "round_end_tick"]), on=["match_key", "round"], how="inner")
    .filter(pl.col("tick") >= pl.col("round_end_tick")).height)
A["rows_outside_clean_rounds"] = int(
    out.join(rounds.select(["match_key", "round"]), on=["match_key", "round"], how="anti").height)
sec = out.join(rounds.select(["match_key", "round", "live_start_tick"]), on=["match_key", "round"], how="inner")
sec = sec.with_columns(((pl.col("tick") - pl.col("live_start_tick")) // TICK).alias("k_from_tick"))
A["seconds_since_live_mismatch_vs_tick_formula"] = int(
    sec.filter(pl.col("seconds_since_live").cast(pl.Int64) != pl.col("k_from_tick")).height)
A["seconds_since_live_max"] = int(out["seconds_since_live"].max())
A["seconds_since_live_min"] = int(out["seconds_since_live"].min())

# expected snapshot grid: ticks freeze_end + 64k with tick < round_end (bake-off live window)
exp_rows = []
for m, r, fz, end in rounds.select(["match_key", "round", "live_start_tick", "round_end_tick"]).iter_rows():
    n = (end - fz + TICK - 1) // TICK
    for k in range(n):
        exp_rows.append((m, r, fz + TICK * k, k))
exp = pl.DataFrame(exp_rows, schema={"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64,
                                     "k": pl.Int64}, orient="row")
A["expected_snapshots_from_grid"] = exp.height
A["snapshots_with_no_rows"] = int(exp.join(out.select(["match_key", "round", "tick"]).unique(),
                                           on=["match_key", "round", "tick"], how="anti").height)
A["rows_not_on_expected_grid"] = int(out.select(["match_key", "round", "tick"]).unique().join(
    exp.select(["match_key", "round", "tick"]), on=["match_key", "round", "tick"], how="anti").height)
A["snapshots_in_players_distinct"] = int(out.select(["match_key", "round", "tick"]).unique().height)
per_snap = out.group_by(["match_key", "round", "tick"]).agg(
    (pl.col("team_code") == T).sum().alias("nT"), (pl.col("team_code") == CT).sum().alias("nCT"))
A["snapshots_with_more_than_5_on_a_side"] = int(
    per_snap.filter((pl.col("nT") > 5) | (pl.col("nCT") > 5)).height)
A["snapshot_k0_5v5_count"] = int(per_snap.join(exp.filter(pl.col("k") == 0).select(["match_key", "round", "tick"]),
                                              on=["match_key", "round", "tick"], how="inner")
                                 .filter((pl.col("nT") == START) & (pl.col("nCT") == START)).height)
A["snapshot_k0_count"] = int(exp.filter(pl.col("k") == 0).height)
res["A_invariants"] = A
for k, v in A.items():
    print(f"  {k}: {v}")

# ---------------- Part B: raw recomputation for the sampled matches ----------------
print("\n== Part B: raw recomputation from E: (150 matches) ==")
matches = sorted(rounds["match_key"].unique().to_list())
B = {"matches": len(matches), "rounds_checked": 0, "fz_end_mismatch_rounds": [],
     "fz_multiplicity_gt1": [], "end_multiplicity_ne1": [], "second_vs_tick_maxabs": 0.0,
     "status_dup_rows": 0, "vector_dup_rows": 0, "presence_key_mismatch_rows": 0,
     "snapshots": 0, "agree_incl": 0, "agree_strict": 0, "agree_incl_by_round_mismatch": {},
     "strict_only_mismatch_snapshots": 0, "snap_with_death_at_tick": 0,
     "live_start_roster_absent": [], "live_start_presence_not_5v5_rounds": [],
     "round_flags": [], "elim_check": {"elim_rounds": 0, "losing_victims_eq5": 0,
                                       "losing_victims_ne5": []},
     "rounds_ct_win_mismatch": 0, "rounds_score_mismatch": 0}
agree_by_round = collections.Counter()
snap_detail = []

for mk in matches:
    t_m = time.perf_counter()
    rm = rounds.filter(pl.col("match_key") == mk)
    rs = rd(mk, "round_state", ["round", "tick", "event_type", "t_score", "ct_score", "second"])
    re_ = rd(mk, "round_end", ["round", "tick", "winner_team_code", "win_reason_code", "player_count",
                               "second"])
    pi = rd(mk, "player_info", ["round", "player_id", "team_code"])
    de = rd(mk, "player_death", ["round", "tick", "player_id", "player_team_code", "second"])
    st = rd(mk, "player_status", ["round", "tick", "player_id"])
    ve = rd(mk, "player_vector", ["round", "tick", "player_id", "team_code"])

    # duplicate checks (status and vector keys)
    B["status_dup_rows"] += st.height - st.unique(["round", "tick", "player_id"]).height
    B["vector_dup_rows"] += ve.height - ve.unique(["round", "tick", "player_id"]).height

    # seconds check: second == tick / 64 on the event rows used here
    for df in (rs, re_, de):
        if df.height and "second" in df.columns:
            d = (df["second"].cast(pl.Float64) - df["tick"].cast(pl.Float64) / TICK).abs().max()
            if d is not None:
                B["second_vs_tick_maxabs"] = max(B["second_vs_tick_maxabs"], float(d))

    fz_rows = rs.filter(pl.col("event_type") == "round_freeze_end").group_by("round").agg(
        pl.len().alias("n"), pl.col("tick").min().alias("fz"))
    end_rows = re_.group_by("round").agg(pl.len().alias("n"), pl.col("tick").min().alias("end"))
    fz_map = {r: (n, f) for r, n, f in fz_rows.select(["round", "n", "fz"]).iter_rows()}
    end_map = {r: (n, e) for r, n, e in end_rows.select(["round", "n", "end"]).iter_rows()}
    for r, n, f in fz_rows.select(["round", "n", "fz"]).iter_rows():
        if n > 1:
            B["fz_multiplicity_gt1"].append([mk, r, n])
    for r, n, e in end_rows.select(["round", "n", "end"]).iter_rows():
        if n != 1:
            B["end_multiplicity_ne1"].append([mk, r, n])

    # per-round checks against rounds.parquet
    rm_rows = rm.select(["round", "live_start_tick", "round_end_tick", "ct_win", "ct_score", "t_score"]).iter_rows(named=True)
    snap_list = []
    round_info = {}
    for row in rm_rows:
        r = row["round"]
        B["rounds_checked"] += 1
        fz = fz_map.get(r, (0, None))[1]
        end = end_map.get(r, (0, None))[1]
        if fz != row["live_start_tick"] or end != row["round_end_tick"]:
            B["fz_end_mismatch_rounds"].append([mk, r, fz, end, row["live_start_tick"], row["round_end_tick"]])
        winner = re_.filter(pl.col("round") == r)["winner_team_code"][0]
        if int(winner == CT) != int(row["ct_win"]):
            B["rounds_ct_win_mismatch"] += 1
        fz_sc = rs.filter((pl.col("round") == r) & (pl.col("event_type") == "round_freeze_end")).sort("tick")
        if fz_sc.height and (int(fz_sc["t_score"][0]) != int(row["t_score"]) or
                             int(fz_sc["ct_score"][0]) != int(row["ct_score"])):
            B["rounds_score_mismatch"] += 1
        round_info[r] = {"fz": int(row["live_start_tick"]), "end": int(row["round_end_tick"]),
                         "winner": int(winner)}
        for t in range(int(row["live_start_tick"]), int(row["round_end_tick"]), TICK):
            snap_list.append((r, t, (t - int(row["live_start_tick"])) // TICK))
    if not snap_list:
        continue
    snaps = pl.DataFrame(snap_list, schema={"round": pl.Int64, "tick": pl.Int64, "k": pl.Int64}, orient="row")
    B["snapshots"] += snaps.height

    # presence and team at snapshot ticks (raw)
    st_s = st.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).join(
        snaps, on=["round", "tick"], how="inner")
    ve_s = ve.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).join(
        snaps.select(["round", "tick"]), on=["round", "tick"], how="inner")
    pres = st_s.join(ve_s.select(["round", "tick", "player_id", "team_code"]),
                     on=["round", "tick", "player_id"], how="left").filter(
        pl.col("team_code").is_in([T, CT]))
    cnt = pres.group_by(["round", "tick"]).agg((pl.col("team_code") == T).sum().alias("nT"),
                                               (pl.col("team_code") == CT).sum().alias("nCT"))
    cnt = snaps.join(cnt, on=["round", "tick"], how="left").with_columns(
        pl.col("nT").fill_null(0), pl.col("nCT").fill_null(0))

    # compare presence keys with the extractor output for this match
    out_m = out.filter(pl.col("match_key") == mk).select(["round", "tick", "player_id"])
    a = pres.select(["round", "tick", "player_id"]).join(out_m, on=["round", "tick", "player_id"], how="anti").height
    b = out_m.join(pres.select(["round", "tick", "player_id"]), on=["round", "tick", "player_id"], how="anti").height
    B["presence_key_mismatch_rows"] += a + b

    # death-based alive counts, from raw player_death (first death per victim in window)
    seen_side = {}
    dtimes = {}
    for r, info in round_info.items():
        d_r = de.filter(pl.col("round") == r).sort("tick")
        roster = {pid: team for pid, team in pi.filter(pl.col("round") == r).select(["player_id", "team_code"]).iter_rows()}
        tT, tCT, seen = [], [], set()
        for tick, pid, team in d_r.select(["tick", "player_id", "player_team_code"]).iter_rows():
            if tick < info["fz"] or tick > info["end"]:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            side = team if team is not None else roster.get(pid)
            if side == T:
                tT.append(tick)
            elif side == CT:
                tCT.append(tick)
        tT.sort()
        tCT.sort()
        dtimes[r] = (tT, tCT)
        # elimination check: winner decides the losing side
        if info["winner"] in (T, CT):
            lose = CT if info["winner"] == T else T
            n_lose = len(tT) if lose == T else len(tCT)
            reason = int(re_.filter(pl.col("round") == r)["win_reason_code"][0])
            if reason in (8, 9):
                B["elim_check"]["elim_rounds"] += 1
                if n_lose == START:
                    B["elim_check"]["losing_victims_eq5"] += 1
                else:
                    B["elim_check"]["losing_victims_ne5"].append([mk, r, reason, n_lose,
                                                                  "losing_side_T" if lose == T else "losing_side_CT"])

    # live-start presence against the round roster
    for r, info in round_info.items():
        roster_t = {pid for pid, team in pi.filter(pl.col("round") == r).select(["player_id", "team_code"]).iter_rows() if team == T}
        roster_ct = {pid for pid, team in pi.filter(pl.col("round") == r).select(["player_id", "team_code"]).iter_rows() if team == CT}
        at_fz = set(st.filter((pl.col("round") == r) & (pl.col("tick") == info["fz"]))["player_id"].to_list())
        pt, pct = len(at_fz & roster_t), len(at_fz & roster_ct)
        if pt != START or pct != START:
            B["live_start_presence_not_5v5_rounds"].append([mk, r, pt, pct, sorted(roster_t - at_fz), sorted(roster_ct - at_fz)])

    # agreement per snapshot
    cnt_rows = cnt.select(["round", "tick", "k", "nT", "nCT"]).iter_rows()
    for r, t, k, nT, nCT in cnt_rows:
        tT, tCT = dtimes.get(r, ([], []))
        e_t = START - bisect.bisect_right(tT, t)
        e_ct = START - bisect.bisect_right(tCT, t)
        s_t = START - bisect.bisect_left(tT, t)
        s_ct = START - bisect.bisect_left(tCT, t)
        ok_i = (nT, nCT) == (e_t, e_ct)
        ok_s = (nT, nCT) == (s_t, s_ct)
        B["agree_incl"] += int(ok_i)
        B["agree_strict"] += int(ok_s)
        if not ok_i:
            agree_by_round[(mk, r)] += 1
            snap_detail.append([mk, r, t, k, nT, nCT, e_t, e_ct])
        if (e_t, e_ct) != (s_t, s_ct):
            B["snap_with_death_at_tick"] += 1
            if ok_i != ok_s:
                pass
        if not ok_s and ok_i:
            pass
    if time.perf_counter() - t_m > 5:
        print(f"  {mk[-45:]} took {time.perf_counter() - t_m:.1f}s")

B["agree_incl_by_round_mismatch"] = {f"{m}|r{r}": c for (m, r), c in agree_by_round.items()}
B["strict_agree_share"] = round(B["agree_strict"] / max(B["snapshots"], 1), 6)
B["incl_agree_share"] = round(B["agree_incl"] / max(B["snapshots"], 1), 6)
B["snapshot_mismatch_detail_first25"] = snap_detail[:25]
B["snapshot_mismatch_count_incl"] = B["snapshots"] - B["agree_incl"]
B["snapshot_mismatch_count_strict"] = B["snapshots"] - B["agree_strict"]
B["wall_s"] = round(time.perf_counter() - t0, 1)
res["B_raw_recompute"] = B
for k, v in B.items():
    if k in ("snapshot_mismatch_detail_first25",):
        print(f"  {k}: {v[:9]}")
    elif isinstance(v, list) and len(v) > 12:
        print(f"  {k}: count {len(v)}; first {v[:5]}")
    else:
        print(f"  {k}: {v}")

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, default=str)
print(f"\nwritten {OUT_JSON}; wall {time.perf_counter() - t0:.1f}s")
