"""Verifier for the decision-window feasibility report (Task A).

Recomputes the claims in notes\\eda\\decision_window.md with separate code. It does not import or
run feas_decision_window.py. Read-only on the tome, data\\state_sample and every existing output.
The only file written is logs\\feas\\verify_decision.txt.

Checks
  1. Win-reason names: tome round_end.win_reason_message per code, and every WIN_REASON literal
     under scripts\\ (parsed with ast, not run).
  2. Five random clean rounds (random.Random(0) over the sorted (match_key, round) list), recomputed
     in plain Python from raw rows and compared with logs\\feas\\feas_decision_window_rounds.csv.
  3. Full-table comparison with that CSV, and impossible-count checks over all clean rounds.
  4. The unexplained 4-victim elimination round (match m_7392c5c6..., round 16).
  5. Radius and dimension sensitivity of the movement result, and the 1 s continuity check.

ASSUMPTIONS, checked here and not taken from the report: tick rate 64 (from the header); radius 50
is applied in 2D on (x_pos, y_pos), with 3D shown for comparison; the 1 Hz grid is freeze + 64k.

Run from the repo root:
  pixi run python -I -X utf8 scripts\\eda\\verify_feas_decision.py
"""

import ast
import bisect
import collections
import glob
import math
import os
import random
import re
import sys
import traceback

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SUFFIX = ".2026-09-18,2026-10-08"
SAMPLE = os.path.join(ROOT, "data", "state_sample")
REPORT_CSV = os.path.join(ROOT, "logs", "feas", "feas_decision_window_rounds.csv")
LOG = os.path.join(ROOT, "logs", "feas", "verify_decision.txt")
WR_SOURCES = ["scripts/eda/batch_effect.py", "scripts/eda/batch_5v5.py", "scripts/eda/flow_checks.py",
              "scripts/eda/flow_profile.py", "scripts/model/feature_validation.py"]
WR_REF_SOURCE = "scripts/eda/batch_effect.py"
RADIUS = 50.0  # ASSUMPTION (from the task): world units, 2D Euclidean on (x_pos, y_pos)
RADII = [10.0, 25.0, 50.0, 100.0, 200.0]
TICK = 64  # checked against the header tick_rate
T, CT = 2, 3
KEYS = ["match_key", "round"]
SEED = 0
N_PICK = 5
ANOM_MATCH = "csds/2026/10/07/m_7392c5c6d537a57c2e735e98/csds"
ANOM_ROUND = 16
TYPE_NAMES = {"elim": "elimination", "defused": "bomb_defused", "exploded": "bomb_exploded",
              "other": "time_out_or_other"}

OUT = []
FAILS = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    OUT.append(line)
    print(line)


def section(title):
    say("")
    say(f"== {title} ==")


def check(label, bad_count):
    """Record one check. A bad_count of 0 is a pass."""
    status = "PASS" if bad_count == 0 else "FAIL"
    say(f"[{status}] {label}: {bad_count}")
    if bad_count:
        FAILS.append(f"{label} = {bad_count}")
    return bad_count


def norm(s):
    """Lowercase, drop the tome's '#SFUI_Notice_' localisation prefix, then drop punctuation."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^#sfui_notice_", "", str(s).lower()))


def read_folder(folder, cols=None):
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    if not files:
        raise SystemExit(f"no dataframe files in {folder}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def channel(ch, cols=None):
    return read_folder(os.path.join(TOME, f"basic_{ch}{SUFFIX}"), cols)


def channel_columns(ch):
    files = sorted(glob.glob(os.path.join(TOME, f"basic_{ch}{SUFFIX}", "dataframe_*")))
    return pl.read_parquet(files[0], n_rows=0).columns


def win_reason_literals():
    found = {}
    for rel in WR_SOURCES:
        path = os.path.join(ROOT, *rel.split("/"))
        tree = ast.parse(open(path, encoding="utf-8").read())
        found[rel] = "no WIN_REASON assignment"
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "WIN_REASON" for t in node.targets):
                try:
                    found[rel] = ast.literal_eval(node.value)
                except ValueError:
                    found[rel] = "non-literal"
                break
    return found


def dist2(a, b):
    return math.hypot(a["x_pos"] - b["x_pos"], a["y_pos"] - b["y_pos"])


def dist3(a, b):
    return math.sqrt((a["x_pos"] - b["x_pos"]) ** 2 + (a["y_pos"] - b["y_pos"]) ** 2
                     + (a["z_pos"] - b["z_pos"]) ** 2)


def sample_round(m, r, base, RS, DE, BS, PL, csv):
    sel = lambda df: df.filter((pl.col("match_key") == m) & (pl.col("round") == r))  # noqa: E731
    row = list(sel(base).iter_rows(named=True))[0]
    rs_rows = list(sel(RS).iter_rows(named=True))
    de_rows = list(sel(DE).iter_rows(named=True))
    bs_rows = list(sel(BS).iter_rows(named=True))
    pl_rows = list(sel(PL).iter_rows(named=True))
    live, end = row["live_start_tick"], row["round_end_tick"]
    freeze = min(x["tick"] for x in rs_rows if x["event_type"] == "round_freeze_end")
    code, winner = row["win_reason_code"], row["winner_team_code"]
    losing = T if winner == CT else CT
    say(f"-- {m} round {r}")
    say(f"   win_reason_code {code} ({row['win_reason_message']}), winner {winner}, "
        f"player_count {row['player_count']}")
    say(f"   freeze {freeze}, round_end {end}, freeze to round_end {(end - freeze) / TICK:.3f} s "
        f"(live_start_tick {live})")
    in_w = [d for d in de_rows if freeze <= d["tick"] <= end]
    losers = [d for d in in_w if d["player_team_code"] == losing]
    victims = {d["player_id"] for d in losers}
    if code in (8, 9):
        dtype = "elimination"
        dec = max((d["tick"] for d in losers), default=end)
    elif code == 7:
        dtype = "bomb_defused"
        dec = min((x["tick"] for x in bs_rows if x["event_type"] == "bomb_defused"
                   and freeze <= x["tick"] <= end), default=end)
    elif code == 1:
        dtype = "bomb_exploded"
        dec = min((x["tick"] for x in bs_rows if x["event_type"] == "bomb_exploded"
                   and freeze <= x["tick"] <= end), default=end)
    else:
        dtype = "time_out_or_other"
        dec = end
    f0 = {x["player_id"]: x for x in pl_rows if x["seconds_since_live"] == 0}
    alive_losing = sum(1 for x in f0.values() if x["team_code"] == losing)
    team_mis = sum(1 for d in losers if d["player_id"] in f0
                   and f0[d["player_id"]]["team_code"] != d["player_team_code"])
    say(f"   decision {dtype} at tick {dec} = {(dec - freeze) / TICK:.3f} s after freeze; "
        f"equals round_end: {dec == end}")
    say(f"   deaths: all {len(de_rows)}, in window {len(in_w)}, losing side in window {len(losers)}, "
        f"distinct losing victims {len(victims)}, losing players alive at freeze {alive_losing}, "
        f"team mismatch vs freeze slice {team_mis}")
    post_rows = [x for x in pl_rows if x["tick"] > dec]
    grid_bad = [x for x in pl_rows if x["tick"] - live != TICK * x["seconds_since_live"]]
    slices = len({x["tick"] for x in pl_rows})
    say(f"   1 Hz rows {len(pl_rows)}, slices {slices} (grid {len(range(live, end, TICK))}), "
        f"rows after decision {len(post_rows)}, rows off grid {len(grid_bad)}")
    k1 = [x for x in pl_rows if x["seconds_since_live"] == 1 and x["player_id"] in f0]
    d2 = [dist2(x, f0[x["player_id"]]) for x in k1]
    d3 = [dist3(x, f0[x["player_id"]]) for x in k1]
    say(f"   at 1 s: pairs {len(d2)}, moved > {RADIUS:g} (2D) {sum(1 for v in d2 if v > RADIUS)}, "
        f"moved > {RADIUS:g} (3D) {sum(1 for v in d3 if v > RADIUS)}, zero displacement (2D) "
        f"{sum(1 for v in d2 if v == 0)}")
    first = None
    cand = sorted((x for x in pl_rows if x["seconds_since_live"] >= 1 and x["player_id"] in f0),
                  key=lambda z: z["seconds_since_live"])
    for x in cand:
        if dist2(x, f0[x["player_id"]]) > RADIUS:
            first = x["seconds_since_live"]
            break
    say(f"   first detection (2D > {RADIUS:g}): k = {first} s after freeze")
    c = list(sel(csv).iter_rows(named=True))[0]
    checks = {
        "freeze": c["freeze_tick"] == freeze,
        "end": c["end_tick"] == end,
        "decision_tick": c["decision_tick"] == dec,
        "decision_type": c["decision_type"] == dtype,
        "win_reason_code": c["win_reason_code"] == code,
        "losing_victims": dtype != "elimination" or c["n_losing_victims_in_window"] == len(victims),
    }
    bad = [k for k, v in checks.items() if not v]
    say("   vs report CSV: " + ("all six fields OK" if not bad else f"DIFF in {bad}"))
    if bad:
        FAILS.append(f"sample round {m} {r} differs from CSV in {bad}")


def main():
    section("0. Inputs and column names")
    rounds = pl.read_parquet(os.path.join(SAMPLE, "rounds.parquet"))
    say("rounds.parquet columns:", rounds.columns)
    rounds = rounds.with_columns(
        pl.col("match_key").cast(pl.String), pl.col("round").cast(pl.Int64),
        pl.col("live_start_tick").cast(pl.Int64), pl.col("round_end_tick").cast(pl.Int64))
    n_rounds = rounds.height
    keys_sorted = sorted(rounds.select(KEYS).rows())
    matches = sorted({k[0] for k in keys_sorted})
    check("duplicate (match_key, round) in rounds.parquet", n_rounds - len(set(keys_sorted)))
    say("clean rounds:", n_rounds, "| matches:", len(matches))

    hdr = read_folder(os.path.join(TOME, "header"), ["key", "tick_rate"]).filter(pl.col("key").is_in(matches))
    say("header tick_rate values (sample matches):", sorted({int(x) for x in hdr["tick_rate"].to_list()}))
    check("sample matches without a header row, or with tick_rate != 64",
          len(matches) - hdr["key"].n_unique() + sum(1 for x in hdr["tick_rate"].to_list() if int(x) != TICK))
    for ch in ["round_end", "round_state", "player_death", "bomb_state"]:
        say(f"{ch} columns:", channel_columns(ch))
    say("players_1hz.parquet columns:",
        pl.read_parquet(os.path.join(SAMPLE, "players_1hz.parquet"), n_rows=0).columns)

    # ------------------------------------------------------------ 1. win-reason names
    section("1. Win-reason names: source and agreement with the tome")
    lit = win_reason_literals()
    ref = lit[WR_REF_SOURCE]
    for rel, d in lit.items():
        if isinstance(d, dict):
            shared = set(d) & set(ref)
            disagree = sorted(k for k in shared if d[k] != ref[k])
            only_here = sorted(set(d) - set(ref))
            only_ref = sorted(set(ref) - set(d))
            say(f"{rel}: {len(d)} codes; disagree with batch_effect.py on shared codes {disagree}; "
                f"codes only here {only_here}; codes only in batch_effect.py {only_ref}")
        else:
            say(f"{rel}: {d}")
    check("WIN_REASON literals that are missing or that disagree with batch_effect.py on a shared code",
          sum(1 for d in lit.values() if not isinstance(d, dict)
              or any(d[k] != ref[k] for k in set(d) & set(ref))))
    RE = (channel("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                "win_reason_message", "player_count"])
          .filter(pl.col("match_key").is_in(matches)).with_columns(pl.col("round").cast(pl.Int64)))
    check("round_end duplicate (match_key, round) rows", RE.group_by(KEYS).len().filter(pl.col("len") > 1).height)
    say("tome round_end rows for sample matches (clean and non-clean rounds):")
    bad_names = 0
    for code, msg, winner, n in RE.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len() \
            .sort(["win_reason_code", "winner_team_code"]).iter_rows():
        name = ref.get(code, "UNMAPPED")
        ok = norm(name) == norm(msg)
        bad_names += 0 if ok else 1
        say(f"  code {code} | message {msg!r} | winner {winner} | rows {n} | WIN_REASON {name!r} | agree {ok}")
    check("codes whose WIN_REASON name does not match tome win_reason_message", bad_names)

    # ------------------------------------------------------------ 2. freeze, end, winner
    RS = (channel("round_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("match_key").is_in(matches)).with_columns(pl.col("round").cast(pl.Int64)))
    say("round_state event_type counts (sample matches):", RS.group_by("event_type").len().sort("event_type").rows())
    fz = RS.filter(pl.col("event_type") == "round_freeze_end").group_by(KEYS).agg(
        pl.col("tick").min().alias("freeze"), pl.len().alias("n_freeze"))
    inf = RS.filter(pl.col("event_type") == "freezetime_ended_inferred").group_by(KEYS).agg(
        pl.col("tick").min().alias("inferred"))
    roe = RS.filter(pl.col("event_type") == "round_officially_ended").group_by(KEYS).agg(
        pl.col("tick").min().alias("officially_ended"))
    rc = RE.select(KEYS + ["tick", "winner_team_code", "win_reason_code", "win_reason_message",
                           "player_count"]).rename({"tick": "rend"})
    base = (rounds.join(rc, on=KEYS, how="left").join(fz, on=KEYS, how="left")
            .join(inf, on=KEYS, how="left").join(roe, on=KEYS, how="left"))

    section("2. Freeze, end and winner (2,673 clean rounds)")
    check("rounds with no round_end row", base.filter(pl.col("rend").is_null()).height)
    check("round_end.tick != rounds.round_end_tick", base.filter(pl.col("rend") != pl.col("round_end_tick")).height)
    check("rounds with null win_reason_code", base.filter(pl.col("win_reason_code").is_null()).height)
    check("rounds with no round_freeze_end row", base.filter(pl.col("freeze").is_null()).height)
    check("rounds with more than one round_freeze_end row", base.filter(pl.col("n_freeze") > 1).height)
    check("round_freeze_end tick != rounds.live_start_tick",
          base.filter(pl.col("freeze") != pl.col("live_start_tick")).height)
    check("winner side contradicts code (8 needs CT=3, 9 needs T=2)", base.filter(
        ((pl.col("win_reason_code") == 8) & (pl.col("winner_team_code") != CT))
        | ((pl.col("win_reason_code") == 9) & (pl.col("winner_team_code") != T))).height)
    check("ct_win label contradicts winner_team_code", base.filter(
        pl.col("ct_win").cast(pl.Int64) != (pl.col("winner_team_code") == CT).cast(pl.Int64)).height)
    say("clean-round code x winner:", base.group_by(["win_reason_code", "winner_team_code"]).len()
        .sort(["win_reason_code", "winner_team_code"]).rows())
    say("clean-round player_count:", base.group_by("player_count").len().sort("player_count").rows())
    b = base.with_columns((pl.col("inferred") - pl.col("freeze")).alias("inf_minus_fz"),
                          (pl.col("officially_ended") - pl.col("rend")).alias("off_minus_end"))
    dd = b["inf_minus_fz"].drop_nulls().to_list()
    cnt = collections.Counter(dd)
    neg = {k: cnt[k] for k in sorted(cnt) if k < 0}
    say(f"freezetime_ended_inferred minus round_freeze_end (ticks): n={len(dd)}, median {np.median(dd):.1f}, "
        f"+1 in {cnt.get(1, 0)} rounds, 0 in {cnt.get(0, 0)}, negative in {sum(neg.values())} rounds {neg}")
    oo = b["off_minus_end"].drop_nulls().to_list()
    if oo:
        say(f"round_officially_ended minus round_end (ticks): n={len(oo)} (missing {b.height - len(oo)}), "
            f"median {np.median(oo):.1f} ({np.median(oo) / TICK:.2f} s), min {min(oo)}, max {max(oo)}")
    else:
        say("round_officially_ended: no rows for clean rounds")

    # ------------------------------------------------------------ 3. deaths and 1 Hz
    PL = (pl.read_parquet(os.path.join(SAMPLE, "players_1hz.parquet"))
          .filter(pl.col("match_key").is_in(matches))
          .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                        pl.col("seconds_since_live").cast(pl.Int64),
                        pl.col("player_id").cast(pl.Int64, strict=False),
                        pl.col("team_code").cast(pl.Int64, strict=False),
                        pl.col("x_pos").cast(pl.Float64), pl.col("y_pos").cast(pl.Float64),
                        pl.col("z_pos").cast(pl.Float64)))

    section("3. Deaths")
    DE = (channel("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
          .filter(pl.col("match_key").is_in(matches))
          .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64),
                        pl.col("player_id").cast(pl.Int64, strict=False),
                        pl.col("player_team_code").cast(pl.Int64, strict=False)))
    DEc = DE.join(base.select(KEYS + ["live_start_tick", "round_end_tick", "winner_team_code",
                                      "win_reason_code"]), on=KEYS, how="inner")
    say("player_death rows (sample matches):", DE.height, "| in clean rounds:", DEc.height,
        "| outside clean rounds:", DE.height - DEc.height)
    check("clean-round deaths with null player_team_code", DEc.filter(pl.col("player_team_code").is_null()).height)
    DEc = DEc.with_columns(
        pl.when(pl.col("tick") < pl.col("live_start_tick")).then(pl.lit("before_freeze"))
        .when(pl.col("tick") > pl.col("round_end_tick")).then(pl.lit("after_end"))
        .otherwise(pl.lit("in_window")).alias("where"))
    say("clean-round deaths by position:", DEc.group_by("where").len().sort("where").rows())
    check("deaths before freeze (negative time relative to freeze)",
          DEc.filter(pl.col("where") == "before_freeze").height)
    win_deaths = DEc.filter(pl.col("where") == "in_window")
    check("same player dies twice inside the window", win_deaths.group_by(KEYS + ["player_id"]).len()
          .filter(pl.col("len") > 1).height)
    check("rounds with more in-window deaths than 10 players",
          win_deaths.group_by(KEYS).len().filter(pl.col("len") > 10).height)
    ae = DEc.filter(pl.col("where") == "after_end")
    say(f"after-end deaths (not in the report's text): {ae.height} in {ae.select(KEYS).unique().height} rounds; "
        f"by player_team_code {ae.group_by('player_team_code').len().sort('player_team_code').rows()}; "
        f"by win code {ae.group_by('win_reason_code').len().sort('win_reason_code').rows()}")
    ae_rows = list(ae.iter_rows(named=True))
    for wc in sorted({x["win_reason_code"] for x in ae_rows}):
        g = [x["tick"] - x["round_end_tick"] for x in ae_rows if x["win_reason_code"] == wc]
        say(f"  after-end deaths, win code {wc}: n {len(g)}, ticks after round_end min {min(g)}, "
            f"median {np.median(g):.0f}, max {max(g)}")
    lose_after = ae.filter(((pl.col("winner_team_code") == CT) & (pl.col("player_team_code") == T))
                           | ((pl.col("winner_team_code") == T) & (pl.col("player_team_code") == CT)))
    say("losing-side deaths after round_end, by win code:",
        lose_after.group_by("win_reason_code").len().sort("win_reason_code").rows())
    check("elimination rounds with a losing-side death after round_end (report says 0)",
          lose_after.filter(pl.col("win_reason_code").is_in([8, 9])).select(KEYS).unique().height)
    fz_slice = PL.filter(pl.col("seconds_since_live") == 0).select(KEYS + ["player_id", "team_code"]) \
        .unique(subset=KEYS + ["player_id"])
    tc = win_deaths.select(KEYS + ["player_id", "player_team_code"]).join(fz_slice, on=KEYS + ["player_id"], how="left")
    check("in-window death rows whose player is missing from the 1 Hz freeze slice",
          tc.filter(pl.col("team_code").is_null()).height)
    check("in-window death rows where player_team_code != 1 Hz team_code at freeze",
          tc.filter(pl.col("team_code").is_not_null() & (pl.col("team_code") != pl.col("player_team_code"))).height)

    # ------------------------------------------------------------ 4. bombs
    section("4. Bomb events")
    BS = (channel("bomb_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("match_key").is_in(matches))
          .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)))
    say("bomb_state event types (sample matches):", BS.group_by("event_type").len().sort("event_type").rows())
    BSc = BS.join(base.select(KEYS + ["live_start_tick", "round_end_tick", "win_reason_code"]), on=KEYS, how="inner")
    say("bomb_state rows in clean rounds by (event_type, code):",
        BSc.group_by(["event_type", "win_reason_code"]).len().sort(["event_type", "win_reason_code"]).rows())
    bomb_post = BSc.filter(pl.col("tick") > pl.col("round_end_tick"))
    say("bomb_state rows after round_end in clean rounds by (event_type, code):",
        bomb_post.group_by(["event_type", "win_reason_code"]).len().sort(["event_type", "win_reason_code"]).rows())
    bomb_in = BSc.filter((pl.col("tick") >= pl.col("live_start_tick")) & (pl.col("tick") <= pl.col("round_end_tick")))
    defused_t = bomb_in.filter(pl.col("event_type") == "bomb_defused").group_by(KEYS).agg(
        pl.col("tick").min().alias("defused_t"), pl.len().alias("n_def"))
    exploded_t = bomb_in.filter(pl.col("event_type") == "bomb_exploded").group_by(KEYS).agg(
        pl.col("tick").min().alias("exploded_t"), pl.len().alias("n_exp"))
    say("null player_team_code rows in all sample-match deaths:",
        DE.filter(pl.col("player_team_code").is_null()).height, "(report says 51, all outside clean rounds)")
    bd_types = channel("bomb_defuse", ["match_key", "round", "tick", "event_type"]) \
        .filter(pl.col("match_key").is_in(matches)).group_by("event_type").len().rows()
    say("bomb_defuse channel event types (sample matches):", bd_types, "(report says only bomb_begin_defuse)")
    ex9 = [x["tick"] - x["round_end_tick"] for x in bomb_post.filter(
        (pl.col("event_type") == "bomb_exploded") & (pl.col("win_reason_code") == 9)).iter_rows(named=True)]
    if ex9:
        say(f"bomb_exploded after round_end in code 9 rounds: n {len(ex9)}, ticks after round_end min {min(ex9)}, "
            f"median {np.median(ex9):.0f}, max {max(ex9)} (report says up to 444)")
    check("rounds with more than one bomb_defused in window", defused_t.filter(pl.col("n_def") > 1).height)
    check("rounds with more than one bomb_exploded in window", exploded_t.filter(pl.col("n_exp") > 1).height)

    # ------------------------------------------------------------ 5. decision tick
    section("5. Decision tick, recomputed for all clean rounds")
    lose = (win_deaths
            .with_columns(pl.when(pl.col("winner_team_code") == CT).then(pl.lit(T)).otherwise(pl.lit(CT))
                          .alias("losing"))
            .filter(pl.col("player_team_code") == pl.col("losing"))
            .group_by(KEYS).agg(pl.col("tick").max().alias("last_lose"),
                                pl.col("player_id").n_unique().alias("n_lose_victims")))
    D = (base.join(lose, on=KEYS, how="left").join(defused_t.select(KEYS + ["defused_t"]), on=KEYS, how="left")
         .join(exploded_t.select(KEYS + ["exploded_t"]), on=KEYS, how="left"))
    code = pl.col("win_reason_code")
    D = D.with_columns(
        pl.when(code.is_in([8, 9]) & pl.col("last_lose").is_not_null()).then(pl.col("last_lose"))
        .when((code == 7) & pl.col("defused_t").is_not_null()).then(pl.col("defused_t"))
        .when((code == 1) & pl.col("exploded_t").is_not_null()).then(pl.col("exploded_t"))
        .otherwise(pl.col("round_end_tick")).alias("dec_tick"),
        pl.when(code.is_in([8, 9])).then(pl.lit("elim"))
        .when(code == 7).then(pl.lit("defused"))
        .when(code == 1).then(pl.lit("exploded"))
        .otherwise(pl.lit("other")).alias("dtype"))
    check("elimination rounds with no losing-side death in window (fallback used)",
          D.filter((pl.col("dtype") == "elim") & pl.col("last_lose").is_null()).height)
    check("bomb_defused code rounds without bomb_defused in window",
          D.filter((code == 7) & pl.col("defused_t").is_null()).height)
    check("Target_Bombed code rounds without bomb_exploded in window",
          D.filter((code == 1) & pl.col("exploded_t").is_null()).height)
    check("decision tick before freeze", D.filter(pl.col("dec_tick") < pl.col("live_start_tick")).height)
    check("decision tick after round_end", D.filter(pl.col("dec_tick") > pl.col("round_end_tick")).height)
    tcounts = D.group_by("dtype").len().sort("dtype").rows()
    for t, n in tcounts:
        say(f"  {TYPE_NAMES[t]}: {n} ({n / n_rounds:.4f})")
    check("decision-type counts that do not sum to the round count",
          abs(sum(n for _, n in tcounts) - n_rounds))
    check("decision-type share above 1", sum(1 for _, n in tcounts if n / n_rounds > 1))
    say("code x decision type:", D.group_by(["win_reason_code", "dtype"]).len()
        .sort(["win_reason_code", "dtype"]).rows())
    say("decision tick == round_end_tick:", D.filter(pl.col("dec_tick") == pl.col("round_end_tick")).height,
        "of", n_rounds)
    for t in ["elim", "defused", "exploded", "other"]:
        s = [(d - f) / TICK for d, f in D.filter(pl.col("dtype") == t)
             .select(["dec_tick", "live_start_tick"]).iter_rows()]
        if s:
            say(f"  {TYPE_NAMES[t]} decision minus freeze (s): n {len(s)}, median {np.median(s):.3f}, "
                f"p10 {np.percentile(s, 10):.3f}, p90 {np.percentile(s, 90):.3f}")
    trip = list(D.select(["live_start_tick", "round_end_tick", "dec_tick"]).iter_rows())
    live_s = sum((e - f) / TICK for f, e, _ in trip)
    removed = [(e - d) / TICK for f, e, d in trip]
    shares = [(e - d) / (e - f) for f, e, d in trip if e > f]
    say(f"live seconds {live_s:.2f}; removed seconds {sum(removed):.2f} = {sum(removed) / live_s:.4f} of live; "
        f"per-round share removed: median {np.median(shares):.4f}, max {max(shares):.4f}")
    check("rounds with round_end <= live_start_tick", sum(1 for f, e, _ in trip if e <= f))
    check("negative removed time", sum(1 for x in removed if x < 0))
    check("per-round removed share outside [0, 1]", sum(1 for s in shares if s < 0 or s > 1))

    alive_T = PL.filter((pl.col("seconds_since_live") == 0) & (pl.col("team_code") == T)).group_by(KEYS).len() \
        .rename({"len": "alive_T"})
    alive_CT = PL.filter((pl.col("seconds_since_live") == 0) & (pl.col("team_code") == CT)).group_by(KEYS).len() \
        .rename({"len": "alive_CT"})
    E = D.join(alive_T, on=KEYS, how="left").join(alive_CT, on=KEYS, how="left") \
        .filter(pl.col("dtype") == "elim")
    E = E.with_columns(pl.when(pl.col("winner_team_code") == CT).then(pl.col("alive_T"))
                       .otherwise(pl.col("alive_CT")).alias("alive_losing"))
    say("elimination rounds by distinct losing-side victims in window:",
        E.group_by("n_lose_victims").len().sort("n_lose_victims").rows())
    mm = E.filter(pl.col("n_lose_victims") != pl.col("alive_losing"))
    check("elimination rounds where distinct losing victims != losing players alive at freeze", mm.height)
    say("  mismatched rounds:", mm.select(KEYS + ["win_reason_code", "n_lose_victims", "alive_losing"]).rows())
    alive_all = alive_T.join(alive_CT, on=KEYS, how="full", coalesce=True).with_columns(
        (pl.col("alive_T").fill_null(0) + pl.col("alive_CT").fill_null(0)).alias("alive_all"))
    vic_all = win_deaths.group_by(KEYS).agg(pl.col("player_id").n_unique().alias("victims_all"))
    cmp_all = vic_all.join(alive_all.select(KEYS + ["alive_all"]), on=KEYS, how="left")
    check("rounds where distinct in-window victims (both sides) exceed players alive at freeze",
          cmp_all.filter(pl.col("victims_all") > pl.col("alive_all")).height)

    # ------------------------------------------------------------ 6. grid and post-decision
    section("6. 1 Hz grid, post-decision rows and impossible counts")
    PLc = PL.join(base.select(KEYS + ["live_start_tick", "round_end_tick"]), on=KEYS, how="inner")
    say("players_1hz rows in clean rounds:", PLc.height, "| all sample-match rows:", PL.height)
    check("1 Hz rows with seconds_since_live < 0", PLc.filter(pl.col("seconds_since_live") < 0).height)
    check("1 Hz rows with tick < live_start_tick", PLc.filter(pl.col("tick") < pl.col("live_start_tick")).height)
    check("1 Hz rows with tick >= round_end_tick", PLc.filter(pl.col("tick") >= pl.col("round_end_tick")).height)
    check("1 Hz rows where tick != live_start_tick + 64 * seconds_since_live", PLc.filter(
        (pl.col("tick") - pl.col("live_start_tick")) != TICK * pl.col("seconds_since_live")).height)
    check("duplicate (round, tick, player_id) rows",
          PLc.group_by(KEYS + ["tick", "player_id"]).len().filter(pl.col("len") > 1).height)
    check("1 Hz slices with more than 10 player rows",
          PLc.group_by(KEYS + ["tick"]).len().filter(pl.col("len") > 10).height)
    fz_counts = PLc.filter(pl.col("seconds_since_live") == 0).group_by(KEYS).len().rename({"len": "n_fz"})
    fz_bad = base.select(KEYS + ["player_count"]).join(fz_counts, on=KEYS, how="left") \
        .filter(pl.col("n_fz").fill_null(0) != pl.col("player_count")).height
    check("rounds whose freeze slice has a different player count from player_count", fz_bad)
    death_map = {(m, r): sorted(ts) for m, r, ts in DEc.filter(pl.col("where") != "before_freeze")
                 .group_by(KEYS).agg(pl.col("tick")).iter_rows()}
    pc_map = {(m, r): pc for m, r, pc in base.select(KEYS + ["player_count"]).iter_rows()}
    mism_rounds = collections.Counter()
    n_slices = mism_slices = 0
    for m, r, t, n in PLc.group_by(KEYS + ["tick"]).len().rename({"len": "n"}).iter_rows():
        n_slices += 1
        dts = death_map.get((m, r), [])
        base_pc = pc_map[(m, r)]
        lo = base_pc - bisect.bisect_right(dts, t)  # a death exactly at t may already be counted
        hi = base_pc - bisect.bisect_left(dts, t)
        if not (lo <= n <= hi):
            mism_slices += 1
            mism_rounds[(m, r)] += 1
    say(f"1 Hz slices whose row count is outside [player_count - deaths at or before t, "
        f"player_count - deaths before t]: {mism_slices} of {n_slices} slices, in {len(mism_rounds)} rounds")
    for (m, r), c in mism_rounds.most_common(10):
        say(f"  {m} round {r}: {c} slices")
    check("1 Hz slices whose player rows contradict death events and player_count", mism_slices)
    gap_rows = base.filter(pl.col("alive_gap_snapshots") > 0).select(KEYS + ["alive_gap_snapshots"]).iter_rows()
    say("rounds with alive_gap_snapshots > 0 (report says 9 in 1 round):", list(gap_rows))
    obs = {(m, r): n for m, r, n in PLc.group_by(KEYS).agg(pl.col("tick").n_unique().alias("n")).iter_rows()}
    grid_exp = {(m, r): len(range(f, e, TICK))
                for m, r, f, e in base.select(KEYS + ["live_start_tick", "round_end_tick"]).iter_rows()}
    check("rounds whose observed 1 Hz slice count differs from freeze + 64k < round_end grid",
          sum(1 for k, v in grid_exp.items() if obs.get(k, 0) != v))
    say(f"grid slices expected {sum(grid_exp.values())}, observed {sum(obs.values())}")
    PD = PLc.join(D.select(KEYS + ["dec_tick"]), on=KEYS, how="inner")
    check("players_1hz rows with tick > decision tick (report says 0)",
          PD.filter(pl.col("tick") > pl.col("dec_tick")).height)

    # ------------------------------------------------------------ 7. movement
    section("7. Movement start (radius 50 in 2D is an ASSUMPTION; sensitivity shown)")
    f0 = PL.filter(pl.col("seconds_since_live") == 0).select(KEYS + ["player_id", "x_pos", "y_pos", "z_pos"]) \
        .unique(subset=KEYS + ["player_id"]) \
        .rename({"x_pos": "x0", "y_pos": "y0", "z_pos": "z0"})
    pr = PL.filter(pl.col("seconds_since_live") >= 1).join(f0, on=KEYS + ["player_id"], how="inner").with_columns(
        ((pl.col("x_pos") - pl.col("x0")) ** 2 + (pl.col("y_pos") - pl.col("y0")) ** 2).sqrt().alias("d2"),
        ((pl.col("x_pos") - pl.col("x0")) ** 2 + (pl.col("y_pos") - pl.col("y0")) ** 2
         + (pl.col("z_pos") - pl.col("z0")) ** 2).sqrt().alias("d3"))
    k1 = pr.filter(pl.col("seconds_since_live") == 1)
    say("pairs alive at freeze and at 1 s:", k1.height)
    d2_all = k1["d2"].to_numpy()
    say(f"1 s displacement (2D): median {np.median(d2_all):.1f}, p10 {np.percentile(d2_all, 10):.1f}, "
        f"p90 {np.percentile(d2_all, 90):.1f} (report: 192.9, 0.0, 209.5)")
    for tc_, nm in [(T, "T"), (CT, "CT")]:
        kk = k1.filter(pl.col("team_code") == tc_)
        say(f"  {nm}: pairs {kk.height}, share moved > {RADIUS:g} (2D) {kk.filter(pl.col('d2') > RADIUS).height / kk.height:.4f}"
            f" (report: T 78.9%, CT 84.4%)")
    say(f"at 1 s: share moved > {RADIUS:g} (2D) {k1.filter(pl.col('d2') > RADIUS).height / k1.height:.4f}; "
        f"share moved > {RADIUS:g} (3D) {k1.filter(pl.col('d3') > RADIUS).height / k1.height:.4f}; "
        f"share exactly zero (2D) {k1.filter(pl.col('d2') == 0).height / k1.height:.4f}")
    for rad in RADII:
        kf = pr.filter(pl.col("d2") > rad).group_by(KEYS).agg(pl.col("seconds_since_live").min().alias("k"))
        say(f"radius {rad:g} (2D): rounds with any detection {kf.height} of {n_rounds}; "
            f"detected at k=1 {kf.filter(pl.col('k') == 1).height}; "
            f"pair share at 1 s {k1.filter(pl.col('d2') > rad).height / k1.height:.4f}")
    kf50 = pr.filter(pl.col("d2") > RADIUS).group_by(KEYS).agg(pl.col("seconds_since_live").min().alias("k"))
    say("first-detection seconds at radius 50 (2D): ", kf50.group_by("k").len().sort("k").rows())
    check("rounds with no detected movement at radius 50 (report says 0)", n_rounds - kf50.height)
    check("rounds whose first detection at radius 50 is not at k=1 (report says 0)",
          kf50.filter(pl.col("k") != 1).height)
    movers = k1.filter(pl.col("d2") > RADIUS).group_by(KEYS).len().rename({"len": "movers"})
    say("movers at 1 s per round: median", float(np.median(movers["movers"].to_list())),
        "| rounds with all pairs moving:", k1.group_by(KEYS).len().rename({"len": "pairs"}).join(
            movers, on=KEYS, how="left").with_columns(pl.col("movers").fill_null(0)).filter(
            pl.col("movers") == pl.col("pairs")).height)
    cont = PL.sort(KEYS + ["player_id", "seconds_since_live"]).with_columns(
        (pl.col("x_pos") - pl.col("x_pos").shift(1).over(KEYS + ["player_id"])).alias("dx"),
        (pl.col("y_pos") - pl.col("y_pos").shift(1).over(KEYS + ["player_id"])).alias("dy"),
        (pl.col("seconds_since_live") - pl.col("seconds_since_live").shift(1).over(KEYS + ["player_id"])).alias("dk"))
    steps = cont.filter(pl.col("dk") == 1).with_columns(
        (pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt().alias("step"))["step"].drop_nulls().to_numpy()
    say(f"consecutive 1 s steps: n {len(steps)}, median {np.median(steps):.1f}, p99 {np.percentile(steps, 99):.1f}, "
        f"max {steps.max():.1f}, steps above 260: {int((steps > 260).sum())}, above 300: {int((steps > 300).sum())}")

    section("8. Rounds where freezetime_ended_inferred is before round_freeze_end")
    kmap = {(m, r): k for m, r, k in kf50.iter_rows()}
    for m, r, f_, inf_, off in b.filter(pl.col("inf_minus_fz") < 0).select(
            KEYS + ["freeze", "inferred", "inf_minus_fz"]).iter_rows():
        say(f"  {m} round {r}: freezetime_ended_inferred - round_freeze_end = {off} ticks; "
            f"first 1 Hz detection k = {kmap.get((m, r))}")

    # ------------------------------------------------------------ 9. report CSV, full table
    section("9. Full-table comparison with logs\\feas\\feas_decision_window_rounds.csv")
    csv = pl.read_csv(REPORT_CSV).with_columns(pl.col("match_key").cast(pl.String), pl.col("round").cast(pl.Int64))
    say("CSV columns:", csv.columns, "| rows:", csv.height)
    cs = csv.select(KEYS + [pl.col("freeze_tick").alias("c_freeze"), pl.col("end_tick").alias("c_end"),
                            pl.col("decision_tick").alias("c_dec"), pl.col("decision_type").alias("c_type"),
                            pl.col("n_losing_victims_in_window").alias("c_nlv"),
                            pl.col("win_reason_code").alias("c_code")])
    J = D.select(KEYS + ["freeze", "round_end_tick", "dec_tick", "dtype", "n_lose_victims", "win_reason_code"]) \
        .join(cs, on=KEYS, how="inner")
    check("rows in the report CSV or clean rounds that do not match one-to-one", abs(n_rounds - J.height)
          + abs(csv.height - J.height))
    rows_ = list(J.iter_rows(named=True))
    check("CSV freeze_tick != recomputed freeze", sum(1 for x in rows_ if x["c_freeze"] != x["freeze"]))
    check("CSV end_tick != round_end_tick", sum(1 for x in rows_ if x["c_end"] != x["round_end_tick"]))
    check("CSV decision_tick != recomputed decision tick", sum(1 for x in rows_ if x["c_dec"] != x["dec_tick"]))
    check("CSV decision_type != recomputed decision type",
          sum(1 for x in rows_ if x["c_type"] != TYPE_NAMES[x["dtype"]]))
    check("CSV win_reason_code != tome code", sum(1 for x in rows_ if x["c_code"] != x["win_reason_code"]))
    check("CSV losing victims != recomputed (elimination rounds)",
          sum(1 for x in rows_ if x["dtype"] == "elim" and x["c_nlv"] != x["n_lose_victims"]))

    # ------------------------------------------------------------ 10. five random rounds
    section(f"10. Five random clean rounds (random.Random({SEED}) over sorted (match_key, round))")
    rng = random.Random(SEED)
    idx = rng.sample(range(len(keys_sorted)), N_PICK)
    say("indices:", idx)
    for i in idx:
        m, r = keys_sorted[i]
        sample_round(m, r, base, RS, DE, BS, PL, csv)

    # ------------------------------------------------------------ 11. anomaly
    section("11. Anomaly: 4 distinct losing-side victims (match m_7392c5c6..., round 16)")
    arow = base.filter((pl.col("match_key") == ANOM_MATCH) & (pl.col("round") == ANOM_ROUND))
    if arow.height == 0:
        say("anomaly round not found among clean rounds")
    else:
        ar = list(arow.iter_rows(named=True))[0]
        live_a, end_a = ar["live_start_tick"], ar["round_end_tick"]
        say(f"  code {ar['win_reason_code']} ({ar['win_reason_message']}), winner {ar['winner_team_code']}, "
            f"freeze {ar['freeze']}, round_end {end_a}, player_count {ar['player_count']}")
        ad = list(DEc.filter((pl.col("match_key") == ANOM_MATCH) & (pl.col("round") == ANOM_ROUND))
                  .sort("tick").iter_rows(named=True))
        for x in ad:
            say(f"  death tick {x['tick']} | vs round_end {x['tick'] - end_a:+d} | vs freeze "
                f"{x['tick'] - live_a:+d} | player {x['player_id']} | team {x['player_team_code']} | {x['where']}")
        ap = PL.filter((pl.col("match_key") == ANOM_MATCH) & (pl.col("round") == ANOM_ROUND))
        fz_rows = list(ap.filter(pl.col("seconds_since_live") == 0).iter_rows(named=True))
        T_fz = sorted(x["player_id"] for x in fz_rows if x["team_code"] == T)
        CT_fz = sorted(x["player_id"] for x in fz_rows if x["team_code"] == CT)
        T_vic = sorted({x["player_id"] for x in ad if x["where"] == "in_window" and x["player_team_code"] == T})
        say(f"  T alive at freeze ({len(T_fz)}): {T_fz}")
        say(f"  CT alive at freeze ({len(CT_fz)}): {CT_fz}")
        say(f"  T in-window victims ({len(T_vic)}): {T_vic}")
        say(f"  T alive at freeze with no in-window death row: {sorted(set(T_fz) - set(T_vic))}")
        last_t = max(x["tick"] for x in ap.iter_rows(named=True) if x["tick"] < end_a)
        T_last = sorted(x["player_id"] for x in ap.iter_rows(named=True)
                        if x["tick"] == last_t and x["team_code"] == T)
        say(f"  last 1 Hz slice before round_end: tick {last_t} ({end_a - last_t} ticks before), "
            f"T ids {T_last}")
        T_any = sorted({x["player_id"] for x in ap.iter_rows(named=True) if x["team_code"] == T})
        T_no_death = sorted(set(T_any) - set(T_vic))
        say(f"  T players in any 1 Hz slice of this round: {T_any}; with no in-window death row: {T_no_death}")
        for pid in T_no_death:
            rec = list(DE.filter((pl.col("match_key") == ANOM_MATCH) & (pl.col("round") == ANOM_ROUND)
                                 & (pl.col("player_id") == pid)).iter_rows(named=True))
            say(f"  death rows for T player {pid} in this round (any tick): {rec}")
            first_seen = min(x["tick"] for x in ap.iter_rows(named=True)
                             if x["team_code"] == T and x["player_id"] == pid)
            last_seen = max(x["tick"] for x in ap.iter_rows(named=True)
                            if x["team_code"] == T and x["player_id"] == pid)
            say(f"  T player {pid} first 1 Hz tick {first_seen} ({first_seen - live_a:+d} vs freeze), "
                f"last {last_seen} ({last_seen - end_a:+d} vs round_end); "
                f"upper bound on decision error if this player died last: {end_a - last_seen} ticks")
        check("anomaly round: T players with no in-window death row", len(T_no_death))

    section("Summary")
    say("failed checks:", len(FAILS))
    for f_ in FAILS:
        say("  -", f_)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say("EXCEPTION:")
        say(traceback.format_exc())
        raise
    finally:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(OUT) + "\n")
        print(f"wrote {LOG}")
