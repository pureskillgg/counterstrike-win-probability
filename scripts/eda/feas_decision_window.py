"""Feasibility Task A: decision window and movement start (CS2 round win-probability EDA).

Read-only on the tome and on data\\state_sample. Writes only to stdout (tee'd to logs\\feas) and to
logs\\feas\\feas_decision_window_rounds.csv (one row per sampled round).

Definitions (see notes\\eda\\decision_window.md for the full write-up)
  freeze   = tick of the first round_state row with event_type round_freeze_end (= rounds.live_start_tick).
             The freezetime_ended_inferred row is compared to it, not used.
  end      = round_end.tick (= rounds.round_end_tick).
  reason   = round_end.win_reason_code, named with WIN_REASON parsed from scripts\\eda\\batch_effect.py.
  decision, by reason:
    CTs_Win (8) / Terrorists_Win (9)  -> elimination: last player_death of the losing side inside
                                         [freeze, end]. Victim side = player_death.player_team_code,
                                         else player_info.team_code for that round and player.
    Bomb_Defused (7)                  -> bomb_state.bomb_defused (the completion event; the bomb_defuse
                                         channel only has bomb_begin_defuse rows). Earliest in window.
    Target_Bombed (1)                 -> bomb_state.bomb_exploded. Earliest in window.
    anything else (12 Target_Saved, surrender, hostage) -> decision = end.
  Movement start: first 1 Hz snapshot (seconds_since_live >= 1) where some player alive at freeze and
  at the snapshot has Euclidean distance > 50 in (x_pos, y_pos) from the freeze snapshot.

Run from the repo root:
  pixi run python -I -X utf8 scripts\\eda\\feas_decision_window.py
"""

import ast
import collections
import glob
import math
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SUFFIX = ".2026-09-18,2026-10-08"
SAMPLE = os.path.join(ROOT, "data", "state_sample")
OUT_CSV = os.path.join(ROOT, "logs", "feas", "feas_decision_window_rounds.csv")

T, CT = 2, 3
MOVE_RADIUS = 50.0  # from the task. ASSUMPTION: units are CS2 world units; distance is 2D Euclidean.
KEYS = ["match_key", "round"]


def read_folder(folder, cols=None):
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    if not files:
        raise SystemExit(f"no dataframe files in {folder}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def read_channel(ch, cols=None):
    return read_folder(os.path.join(TOME, f"basic_{ch}{SUFFIX}"), cols)


def load_win_reason():
    """WIN_REASON is taken from scripts\\eda\\batch_effect.py by parsing its AST (the module is not run)."""
    path = os.path.join(ROOT, "scripts", "eda", "batch_effect.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "WIN_REASON" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("WIN_REASON not found in batch_effect.py")


def pct(x, q):
    return float(np.percentile(np.asarray(x, dtype=float), q))


def fmt_sec(x):
    return f"{x:.3f}"


def main():
    WR = load_win_reason()
    print(f"WIN_REASON (from scripts/eda/batch_effect.py): {WR}")

    # ------------------------------------------------------------------ sample
    rounds = pl.read_parquet(os.path.join(SAMPLE, "rounds.parquet")).with_columns(
        pl.col("round").cast(pl.Int64), pl.col("round_number").cast(pl.Int64),
        pl.col("live_start_tick").cast(pl.Int64), pl.col("round_end_tick").cast(pl.Int64),
        pl.col("alive_gap_snapshots").cast(pl.Int64))
    n_rounds = rounds.height
    matches = rounds["match_key"].unique().to_list()
    match_set = set(matches)
    print("== 0. Sample ==")
    print(f"rounds={n_rounds} unique (match_key, round)={rounds.select(KEYS).unique().height} "
          f"matches={len(match_set)}")
    print("split counts (rounds):", rounds.group_by("split").len().sort("split").rows())

    hdr_files = read_folder(os.path.join(TOME, "header"), ["key", "tick_rate", "max_rounds"])
    hdr = hdr_files.filter(pl.col("key").is_in(matches))
    print(f"header rows for sample matches: {hdr.height}; tick_rate counts: "
          f"{hdr.group_by('tick_rate').len().sort('tick_rate').rows()}")
    TICK = int(hdr["tick_rate"].unique().to_list()[0]) if hdr.height else 64
    assert hdr.height == len(match_set) and hdr["tick_rate"].unique().len() == 1, "tick rate not uniform"
    print(f"TICK per second used for seconds conversions: {TICK} (from header tick_rate)")

    # ------------------------------------------------------------------ round_end
    RE = (read_channel("round_end", ["match_key", "round", "tick", "winner_team_code",
                                     "win_reason_code", "player_count"])
          .filter(pl.col("match_key").is_in(matches))
          .with_columns(pl.col("round").cast(pl.Int64)))
    print(f"round_end rows for sample matches: {RE.height}; duplicate (match,round): "
          f"{RE.group_by(KEYS).len().filter(pl.col('len') > 1).height}")
    RE = RE.rename({"tick": "rend_tick"})
    rd = rounds.join(RE, on=KEYS, how="left")
    print(f"rounds with no round_end row: {rd['rend_tick'].null_count()}; "
          f"round_end.tick != rounds.round_end_tick: "
          f"{rd.filter(pl.col('rend_tick') != pl.col('round_end_tick')).height}")
    print(f"null winner_team_code: {rd['winner_team_code'].null_count()}; "
          f"null win_reason_code: {rd['win_reason_code'].null_count()}")
    bad_side = rd.filter(((pl.col("win_reason_code") == 8) & (pl.col("winner_team_code") != CT))
                         | ((pl.col("win_reason_code") == 9) & (pl.col("winner_team_code") != T))).height
    print(f"rounds where win_reason 8 is not a CT win or 9 is not a T win: {bad_side}")

    # ------------------------------------------------------------------ round_state (freeze, inferred)
    RS = (read_channel("round_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("match_key").is_in(matches)
                  & pl.col("event_type").is_in(["round_freeze_end", "freezetime_ended_inferred"]))
          .with_columns(pl.col("round").cast(pl.Int64)))
    fz = (RS.filter(pl.col("event_type") == "round_freeze_end").group_by(KEYS)
          .agg(pl.col("tick").min().alias("fz_tick"), pl.len().alias("n_fz")))
    inf = (RS.filter(pl.col("event_type") == "freezetime_ended_inferred").group_by(KEYS)
           .agg(pl.col("tick").min().alias("inf_tick"), pl.len().alias("n_inf")))
    rd = rd.join(fz, on=KEYS, how="left").join(inf, on=KEYS, how="left")
    print("== 1. Freeze ==")
    print(f"rounds with no round_freeze_end row: {rd.filter(pl.col('fz_tick').is_null()).height}; "
          f"rounds with >1 round_freeze_end row: {rd.filter(pl.col('n_fz') > 1).height}")
    mism = rd.filter(pl.col("fz_tick") != pl.col("live_start_tick")).height
    print(f"rounds.live_start_tick != round_freeze_end tick: {mism}")
    print(f"rounds with no freezetime_ended_inferred row: {rd.filter(pl.col('inf_tick').is_null()).height}; "
          f"rounds with >1 such row: {rd.filter(pl.col('n_inf') > 1).height}")
    diff_rows = rd.filter(pl.col("inf_tick").is_not_null()).with_columns(
        (pl.col("inf_tick") - pl.col("fz_tick")).alias("inf_minus_freeze"))
    diffs = diff_rows["inf_minus_freeze"].to_list()
    if diffs:
        cnt = collections.Counter(diffs)
        print(f"freezetime_ended_inferred minus freeze (ticks), n={len(diffs)}: "
              f"min {min(diffs)} median {float(np.median(diffs)):.1f} max {max(diffs)}; "
              f"exactly 0: {cnt.get(0, 0)}; nonzero: {len(diffs) - cnt.get(0, 0)}")
        print("  most common differences (ticks: count): "
              + ", ".join(f"{k}: {v}" for k, v in cnt.most_common(12)))

    # ------------------------------------------------------------------ player_info teams
    PI = (read_channel("player_info", ["match_key", "round", "player_id", "team_code"])
          .filter(pl.col("match_key").is_in(matches)).with_columns(pl.col("round").cast(pl.Int64)))
    pi_team, pi_conflict = {}, 0
    for m, r, pid, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
        if tc is None:
            continue
        k3 = (m, r, pid)
        if k3 in pi_team and pi_team[k3] != tc:
            pi_conflict += 1
        pi_team.setdefault(k3, tc)
    print(f"player_info (round, player) with non-null team: {len(pi_team)}; conflicting team rows: {pi_conflict}")

    # ------------------------------------------------------------------ player_death
    DE = (read_channel("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
          .filter(pl.col("match_key").is_in(matches)).with_columns(pl.col("round").cast(pl.Int64)))
    info = {}
    for row in rd.iter_rows(named=True):
        info[(row["match_key"], row["round"])] = row

    deaths = collections.defaultdict(list)
    null_all = null_win = null_win_res = null_win_unres = null_post = 0
    win_deaths = pre_deaths = post_deaths = unknown_round = sample_round_deaths = 0
    for m, r, tick, pid, raw in DE.select(["match_key", "round", "tick", "player_id",
                                           "player_team_code"]).iter_rows():
        k = (m, r)
        if k not in info:
            unknown_round += 1
            continue
        sample_round_deaths += 1
        if raw is None:
            null_all += 1
        f, e = info[k]["live_start_tick"], info[k]["round_end_tick"]
        if tick > e and raw is None:
            null_post += 1
        if tick < f:
            pre_deaths += 1
            continue
        if tick > e:
            post_deaths += 1
            continue
        win_deaths += 1
        side = raw
        if raw is None:
            null_win += 1
            side = pi_team.get((m, r, pid))
            if side is not None:
                null_win_res += 1
            else:
                null_win_unres += 1
        deaths[k].append((tick, pid, side, raw is None))
    print("== 2. Deaths ==")
    print(f"player_death rows in sample matches: {DE.height}; in the 2673 clean sample rounds: "
          f"{sample_round_deaths}; null player_team_code in those rounds (all ticks): {null_all} "
          f"(of which after round_end: {null_post})")
    print(f"deaths in live window [freeze, end]: {win_deaths}; before freeze: {pre_deaths}; "
          f"after end: {post_deaths}; rows in sample matches but not a clean sample round: {unknown_round}")
    print(f"null team inside window: {null_win}; resolved from player_info: {null_win_res}; "
          f"still unresolved (side None): {null_win_unres}")
    print(f"player_info conflicts while resolving: {pi_conflict}")

    # ------------------------------------------------------------------ bomb events
    BS = (read_channel("bomb_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("match_key").is_in(matches)
                  & pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
          .with_columns(pl.col("round").cast(pl.Int64)))
    BD = (read_channel("bomb_defuse", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("match_key").is_in(matches)).with_columns(pl.col("round").cast(pl.Int64)))
    exploded, defused, begin_def = (collections.defaultdict(list) for _ in range(3))
    for m, r, tick, et in BS.select(["match_key", "round", "tick", "event_type"]).iter_rows():
        (exploded if et == "bomb_exploded" else defused)[(m, r)].append(tick)
    for m, r, tick in BD.select(["match_key", "round", "tick"]).iter_rows():
        begin_def[(m, r)].append(tick)
    print("== 3. Bomb events ==")
    print(f"bomb_exploded rows in sample matches (incl. non-clean rounds): {sum(len(v) for v in exploded.values())} "
          f"(rounds {len(exploded)}; rounds with >1: {sum(1 for v in exploded.values() if len(v) > 1)})")
    print(f"bomb_defused rows in sample matches (incl. non-clean rounds): {sum(len(v) for v in defused.values())} "
          f"(rounds {len(defused)}; rounds with >1: {sum(1 for v in defused.values() if len(v) > 1)})")
    print(f"bomb_begin_defuse rows in sample matches (incl. non-clean rounds): {sum(len(v) for v in begin_def.values())} "
          f"(rounds {len(begin_def)})")

    # ------------------------------------------------------------------ decision tick
    rows = []
    issue = collections.Counter()
    n_losing_victims = collections.Counter()
    for k in sorted(info):
        rr = info[k]
        f, e, code, w = rr["live_start_tick"], rr["round_end_tick"], rr["win_reason_code"], \
            rr["winner_team_code"]
        name = WR.get(code, "UNMAPPED")
        src, dtype, dec, nlv = None, None, None, None
        if code in (8, 9):
            dtype = "elimination"
            losing = T if w == CT else CT
            cands = [d for d in deaths.get(k, []) if d[2] == losing]
            victims = {d[1] for d in cands}
            nlv = len(victims)
            n_losing_victims[nlv] += 1
            if cands:
                dec = max(d[0] for d in cands)
                src = "last_losing_side_death"
            else:
                dec, src = e, "fallback_round_end (no losing-side death in window)"
                issue["elimination without losing-side death in window"] += 1
        elif code == 7:
            dtype = "bomb_defused"
            ticks = [t for t in defused.get(k, []) if f <= t <= e]
            if ticks:
                dec, src = min(ticks), "bomb_defused event"
            else:
                dec, src = e, "fallback_round_end (no bomb_defused in window)"
                issue["bomb_defused code without event in window"] += 1
        elif code == 1:
            dtype = "bomb_exploded"
            ticks = [t for t in exploded.get(k, []) if f <= t <= e]
            if ticks:
                dec, src = min(ticks), "bomb_exploded event"
            else:
                dec, src = e, "fallback_round_end (no bomb_exploded in window)"
                issue["Target_Bombed code without event in window"] += 1
        else:
            dtype, dec, src = "time_out_or_other", e, "round_end (time-out or other)"
        if dec < f:
            issue["decision before freeze"] += 1
        if dec > e:
            issue["decision after end"] += 1
        rows.append({"match_key": k[0], "round": k[1], "win_reason_code": int(code),
                     "win_reason_name": name, "winner_team_code": int(w), "decision_type": dtype,
                     "decision_source": src, "freeze_tick": int(f), "end_tick": int(e),
                     "decision_tick": int(dec), "n_losing_victims_in_window": nlv})
    DEC = pl.DataFrame(rows, schema={
        "match_key": pl.String, "round": pl.Int64, "win_reason_code": pl.Int64,
        "win_reason_name": pl.String, "winner_team_code": pl.Int64, "decision_type": pl.String,
        "decision_source": pl.String, "freeze_tick": pl.Int64, "end_tick": pl.Int64,
        "decision_tick": pl.Int64, "n_losing_victims_in_window": pl.Int64})
    print("== 4. Decision tick ==")
    print("decision counts by win_reason_code:")
    for row in DEC.group_by(["win_reason_code", "win_reason_name", "decision_type"]).len() \
            .sort("win_reason_code").iter_rows():
        print(f"  {row[0]} {row[1]} -> {row[2]}: {row[3]} ({row[3] / n_rounds:.4f})")
    print("decision counts by decision_type:")
    for row in DEC.group_by("decision_type").len().sort("decision_type").iter_rows():
        print(f"  {row[0]}: {row[1]} ({row[1] / n_rounds:.4f})")
    print("decision_source counts:")
    for row in DEC.group_by("decision_source").len().sort("decision_source").iter_rows():
        print(f"  {row[0]}: {row[1]}")
    print(f"elimination rounds by distinct losing-side victims in window: {dict(sorted(n_losing_victims.items()))}")
    print(f"issues: {dict(issue) if issue else 'none'}")
    bomb_pos = collections.Counter()
    for k in info:
        code, e = info[k]["win_reason_code"], info[k]["round_end_tick"]
        for name, evs in (("bomb_exploded", exploded), ("bomb_defused", defused)):
            ts = evs.get(k, [])
            if ts:
                pos = "same tick as round_end" if min(ts) == e else (
                    "after round_end" if min(ts) > e else "before round_end")
                bomb_pos[(name, code, pos)] += 1
    print("clean rounds with a bomb event, by (event, round_end code, position vs round_end):")
    for key_, v in sorted(bomb_pos.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        print(f"  {key_[0]} | code {key_[1]} | {key_[2]}: {v}")
    print(f"decision_tick == end_tick: {DEC.filter(pl.col('decision_tick') == pl.col('end_tick')).height}")
    print("decision - freeze in seconds, by type (median, p10, p90):")
    for t in ["elimination", "bomb_defused", "bomb_exploded", "time_out_or_other"]:
        sub = DEC.filter(pl.col("decision_type") == t).with_columns(
            ((pl.col("decision_tick") - pl.col("freeze_tick")) / TICK).alias("s"))
        if sub.height:
            s = sub["s"].to_list()
            print(f"  {t}: n={len(s)} median {fmt_sec(float(np.median(s)))} "
                  f"p10 {fmt_sec(pct(s, 10))} p90 {fmt_sec(pct(s, 90))}")

    # bomb_defused completion vs begin_defuse cross-check
    chk = [(k, min(d for d in defused.get(k, []) if info[k]["live_start_tick"] <= d <= info[k]["round_end_tick"])) for k in info
           if info[k]["win_reason_code"] == 7 and any(info[k]["live_start_tick"] <= d <= info[k]["round_end_tick"] for d in defused.get(k, []))]
    lag = []
    begin_after = 0
    no_begin = 0
    for k, dt in chk:
        bs = [t for t in begin_def.get(k, []) if t <= dt]
        if not bs:
            no_begin += 1
            continue
        lag.append((dt - max(bs)) / TICK)
        if max(bs) > dt:
            begin_after += 1
    if lag:
        print(f"cross-check bomb_defused vs last bomb_begin_defuse at or before it: n={len(lag)}, "
              f"median lag {float(np.median(lag)):.2f} s, p10 {pct(lag, 10):.2f}, p90 {pct(lag, 90):.2f}; "
              f"no begin row before it: {no_begin}")
    print(f"defuse rounds with begin_defuse row after completion tick (not used): {begin_after}")

    # ------------------------------------------------------------------ 1 Hz snapshots
    P = (pl.read_parquet(os.path.join(SAMPLE, "players_1hz.parquet"),
                         columns=["match_key", "round", "tick", "seconds_since_live", "player_id",
                                  "team_code", "x_pos", "y_pos"])
         .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                       pl.col("seconds_since_live").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),
                       pl.col("x_pos").cast(pl.Float64), pl.col("y_pos").cast(pl.Float64)))
    n_rows = P.height
    slices_all = P.select(KEYS + ["tick"]).unique().height
    print("== 5. Snapshots after decision ==")
    print(f"players_1hz rows: {n_rows}; distinct (round, tick) slices: {slices_all}; "
          f"rounds with any row: {P.select(KEYS).unique().height}")
    PD = P.join(DEC.select(KEYS + ["decision_type", "decision_tick", "freeze_tick", "end_tick"]),
                on=KEYS, how="left")
    print(f"rows with no decision match (should be 0): {PD['decision_tick'].null_count()}")
    post = PD.filter(pl.col("tick") > pl.col("decision_tick"))
    n_post = post.height
    slices_post = post.select(KEYS + ["tick"]).unique().height
    rounds_post = post.select(KEYS).unique().height
    print("note: the extraction keeps ticks < round_end only, so no 1 Hz row can follow round_end; "
          "post-round time (round_end to round_officially_ended, see feas_probe_decision_detail.py) is absent.")
    print(f"rows with tick > decision: {n_post} = {n_post / n_rows:.4f} of rows")
    print(f"slices with tick > decision: {slices_post} = {slices_post / slices_all:.4f} of slices")
    print(f"rounds with >=1 post-decision row: {rounds_post} = {rounds_post / n_rounds:.4f} of rounds")
    per_round = (post.group_by(KEYS + ["decision_type"])
                 .agg(pl.len().alias("post_rows"),
                      ((pl.col("tick").max() - pl.col("decision_tick").first()) / TICK).alias("span_s"),
                      ((pl.col("tick") - pl.col("decision_tick")) / TICK).median().alias("med_row_s")))
    if per_round.height:
        span = per_round["span_s"].to_list()
        rowsec = post.with_columns(((pl.col("tick") - pl.col("decision_tick")) / TICK).alias("s"))["s"].to_list()
        print(f"median seconds after decision per round, using last post-decision tick (rounds with >=1): "
              f"{float(np.median(span)):.3f} s (p10 {pct(span, 10):.3f}, p90 {pct(span, 90):.3f})")
        print(f"median seconds after decision, all post-decision rows: {float(np.median(rowsec)):.3f} s; "
              f"median post rows per round with >=1: {float(np.median(per_round['post_rows'].to_list())):.1f}")
    print("by decision type (rounds, rounds with post rows, post rows, share of type's rows):")
    for t in ["elimination", "bomb_defused", "bomb_exploded", "time_out_or_other"]:
        sub = PD.filter(pl.col("decision_type") == t)
        n_t_rounds = DEC.filter(pl.col("decision_type") == t).height
        n_t_rows = sub.height
        t_post = post.filter(pl.col("decision_type") == t)
        n_t_post = t_post.height
        r_t = t_post.select(KEYS).unique().height
        print(f"  {t}: rounds {n_t_rounds}; rounds with post {r_t}; post rows {n_t_post} "
              f"of {n_t_rows} ({(n_t_post / n_t_rows if n_t_rows else float('nan')):.4f})")

    # ------------------------------------------------------------------ movement start
    freeze_rows = P.filter(pl.col("seconds_since_live") == 0)
    fz_round = freeze_rows.select(KEYS).unique().height
    fz_off = freeze_rows.join(rounds.select(KEYS + ["live_start_tick"]), on=KEYS, how="left") \
        .filter(pl.col("tick") != pl.col("live_start_tick")).height
    print("== 6. Movement start ==")
    print(f"rounds with a freeze snapshot (seconds_since_live == 0): {fz_round} of {n_rounds}; "
          f"freeze rows with tick != live_start_tick: {fz_off}")
    base = freeze_rows.select(KEYS + ["player_id", "x_pos", "y_pos"]).rename({"x_pos": "x0", "y_pos": "y0"})
    cur = P.filter(pl.col("seconds_since_live") >= 1).join(base, on=KEYS + ["player_id"], how="inner")
    cur = cur.with_columns(((pl.col("x_pos") - pl.col("x0")) ** 2
                            + (pl.col("y_pos") - pl.col("y0")) ** 2).sqrt().alias("dist"))
    print(f"player pairs (alive at freeze and at a later snapshot): {cur.height}")
    k1 = cur.filter(pl.col("seconds_since_live") == 1)
    d1 = k1["dist"].to_numpy()
    print(f"at the 1 s snapshot: player pairs {len(d1)}; share moved > {MOVE_RADIUS:g} units: "
          f"{float(np.mean(d1 > MOVE_RADIUS)):.4f}; distance median {float(np.median(d1)):.1f}, "
          f"p10 {pct(d1, 10):.1f}, p90 {pct(d1, 90):.1f} units")
    movers = k1.group_by(KEYS).agg((pl.col("dist") > MOVE_RADIUS).sum().alias("movers"),
                                   pl.len().alias("pairs"))
    mv = movers["movers"].to_list()
    print(f"at the 1 s snapshot, players moved > {MOVE_RADIUS:g} per round: median {float(np.median(mv)):.1f} "
          f"of {float(np.median(movers['pairs'].to_list())):.0f}; rounds with all pairs moved: "
          f"{sum(1 for a, b in zip(mv, movers['pairs'].to_list()) if a == b)}; rounds with none: "
          f"{sum(1 for a in mv if a == 0)}")
    moved = (cur.filter(pl.col("dist") > MOVE_RADIUS).group_by(KEYS)
             .agg(pl.col("seconds_since_live").min().alias("k_first")))
    k_first = moved["k_first"].to_list()
    n_moved = len(k_first)
    print(f"rounds with any movement > {MOVE_RADIUS:g} units at some snapshot: {n_moved} of {n_rounds} "
          f"({n_moved / n_rounds:.4f}); rounds with no detected movement: {n_rounds - n_moved}")
    print(f"rounds whose first detected movement is at the 1 s snapshot (movement in first second): "
          f"{sum(1 for k in k_first if k == 1)} ({sum(1 for k in k_first if k == 1) / n_rounds:.4f} of rounds)")
    if k_first:
        print(f"first-movement seconds after freeze (rounds with movement, integer 1 Hz resolution): "
              f"median {float(np.median(k_first)):.2f}, p10 {pct(k_first, 10):.2f}, p90 {pct(k_first, 90):.2f}, "
              f"mean {float(np.mean(k_first)):.3f}, max {max(k_first)}")
        dist = collections.Counter(k_first)
        print("first-movement distribution (seconds: rounds): "
              + ", ".join(f"{k}: {dist[k]}" for k in sorted(dist)[:15]))
    print("reminder: a first detection at k means movement in (k-1, k] seconds; resolution is 1 s")

    # ------------------------------------------------------------------ supplementary footstep proxy
    FS = (read_channel("player_footstep", ["match_key", "round", "tick", "player_id"])
          .filter(pl.col("match_key").is_in(matches)).with_columns(pl.col("round").cast(pl.Int64)))
    first_fs, fs_before, fs_after = {}, 0, 0
    fs_first_sec_players = set()
    for m, r, t, pid in FS.select(["match_key", "round", "tick", "player_id"]).iter_rows():
        k = (m, r)
        if k not in info:
            continue
        f, e = info[k]["live_start_tick"], info[k]["round_end_tick"]
        if t < f:
            fs_before += 1
            continue
        if t > e:
            fs_after += 1
            continue
        if t <= f + TICK:
            fs_first_sec_players.add((m, r, pid))
        if k not in first_fs or t < first_fs[k]:
            first_fs[k] = t
    print("== 6b. Supplementary, not part of the 1 Hz spec: first player_footstep after freeze ==")
    fs_secs = [(first_fs[k] - info[k]["live_start_tick"]) / TICK for k in info if k in first_fs]
    print(f"clean rounds with a footstep in [freeze, end]: {len(fs_secs)} of {n_rounds}; "
          f"footsteps before freeze: {fs_before}; after end: {fs_after}")
    if fs_secs:
        print(f"first footstep seconds after freeze: median {float(np.median(fs_secs)):.3f}, "
              f"p10 {pct(fs_secs, 10):.3f}, p90 {pct(fs_secs, 90):.3f}; share <= 1.0 s: "
              f"{float(np.mean(np.asarray(fs_secs) <= 1.0)):.4f}")
    pairs1 = cur.filter(pl.col("seconds_since_live") == 1).select(KEYS + ["player_id", "dist"]).iter_rows()
    mv_fs = mv_n = st_fs = st_n = 0
    for m, r, pid, dist_ in pairs1:
        has = (m, r, pid) in fs_first_sec_players
        if dist_ > MOVE_RADIUS:
            mv_n += 1
            mv_fs += has
        else:
            st_n += 1
            st_fs += has
    print(f"validation at player level (1 s snapshot): pairs moved > {MOVE_RADIUS:g} units: {mv_n}, with a footstep "
          f"in the first second: {mv_fs} ({mv_fs / max(mv_n, 1):.4f}); pairs not moved: {st_n}, with a footstep "
          f"in the first second: {st_fs} ({st_fs / max(st_n, 1):.4f})")

    # ------------------------------------------------------------------ live vs decision window
    print("== 7. Live window vs decision window ==")
    W = DEC.with_columns(
        ((pl.col("end_tick") - pl.col("freeze_tick")) / TICK).alias("live_s"),
        ((pl.col("decision_tick") - pl.col("freeze_tick")) / TICK).alias("dec_s"),
        ((pl.col("end_tick") - pl.col("decision_tick")) / TICK).alias("removed_s"))
    live_total = W["live_s"].sum()
    dec_total = W["dec_s"].sum()
    removed_total = W["removed_s"].sum()
    print(f"total live seconds (freeze to end): {live_total:.2f}")
    print(f"total decision-window seconds (freeze to decision): {dec_total:.2f}")
    print(f"removed seconds (decision to end): {removed_total:.2f} = {removed_total / live_total:.4f} of live")
    share_per_round = [r / l for r, l in zip(W["removed_s"].to_list(), W["live_s"].to_list()) if l > 0]
    print(f"per-round share of live seconds removed: median {float(np.median(share_per_round)):.4f}, "
          f"p10 {pct(share_per_round, 10):.4f}, p90 {pct(share_per_round, 90):.4f}; "
          f"rounds with removed > 0: {sum(1 for x in W['removed_s'].to_list() if x > 0)}")
    for t in ["elimination", "bomb_defused", "bomb_exploded", "time_out_or_other"]:
        sub = W.filter(pl.col("decision_type") == t)
        if sub.height:
            print(f"  {t}: rounds {sub.height}; live {sub['live_s'].sum():.2f} s; removed "
                  f"{sub['removed_s'].sum():.2f} s ({sub['removed_s'].sum() / sub['live_s'].sum():.4f})")
    # 1 Hz slices in each window, observed in players_1hz and on the grid
    obs_live = PD.filter(pl.col("tick") < pl.col("end_tick")).select(KEYS + ["tick"]).unique().height
    obs_dec = PD.filter(pl.col("tick") <= pl.col("decision_tick")).select(KEYS + ["tick"]).unique().height
    grid = sum(len(range(f, e, TICK)) for f, e in zip(W["freeze_tick"].to_list(), W["end_tick"].to_list()))
    gap = int(rounds["alive_gap_snapshots"].sum())
    print(f"1 Hz slices, live window: observed {obs_live}; expected grid (freeze + 64k < end) {grid}; "
          f"observed == grid: {obs_live == grid}")
    print(f"alive_gap_snapshots (extraction's alive-count validation mismatches, not missing slices): "
          f"sum {gap} in {int((rounds['alive_gap_snapshots'] > 0).sum())} round(s)")
    print(f"1 Hz slices, decision window (tick <= decision): observed {obs_dec} "
          f"= {obs_dec / obs_live:.4f} of observed live slices; removed "
          f"{obs_live - obs_dec} = {(obs_live - obs_dec) / obs_live:.4f}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    W.write_csv(OUT_CSV)
    print(f"wrote {OUT_CSV} ({W.height} rows)")


if __name__ == "__main__":
    main()
