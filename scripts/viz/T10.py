# -*- coding: utf-8 -*-
"""T10 momentum chart.

Quantity: P(team wins round r+1 | team won (or lost) each of the last k rounds, k = streak ending in round r).
Streak buckets: exact k = 1..4, and 5 meaning 5 or more.

Team tracking across the halftime swap:
  Team A = the five players on CT at the first observed round of each match with five CT players
  (freeze-end roster). Team B = the other side at that round (the other five players).
  For every round, the side of team A is the side (CT or T) whose freeze-end roster overlaps
  team A's roster most. Team A's round outcome = (side == CT) == (ct_win == 1). Team B = the complement.
  Each match-round therefore gives two team rows, one per team, each with its own streak.

Inputs (read only):
  data/state_sample/rounds.parquet
  data/state_sample/players_1hz.parquet   (freeze-end snapshot per round only)
Outputs:
  data/viz/T10.json
  data/viz/T10.svg

Intervals: match-cluster bootstrap, 1,000 resamples of the 150 matches with replacement,
random.Random(0), 95% percentile interval with linear interpolation.
"""
import json
import math
import os
import random
from collections import defaultdict

import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROUNDS_PATH = os.path.join(ROOT, "data", "state_sample", "rounds.parquet")
PLAYERS_PATH = os.path.join(ROOT, "data", "state_sample", "players_1hz.parquet")
OUT_JSON = os.path.join(ROOT, "data", "viz", "T10.json")
OUT_SVG = os.path.join(ROOT, "data", "viz", "T10.svg")

REG_MAX_ROUND = 24                 # regulation only (MR12); OT rounds 25-29 are not used
EXCLUDED_TARGETS = {1, 13, 14}     # pistols 1 and 13; round 14 = first full round after the halftime swap
SENSITIVITY_EXCLUDED = {1, 13}     # literal pistol-only exclusion, shown as sensitivity
BUCKETS = [1, 2, 3, 4, 5]          # 5 = five or more
LINES = ["win", "loss"]            # win = team won the last k rounds; loss = team lost the last k rounds
N_BOOT = 1000
SEED = 0

CT_CODE, T_CODE = 3, 2
COLORS = {"win": "#059669", "loss": "#475569"}


# ---------------------------------------------------------------- data
def load_rounds():
    rounds = (
        pl.scan_parquet(ROUNDS_PATH)
        .select(["match_key", "round", "ct_win"])
        .collect()
    )
    ct_win = {}
    for m, r, w in rounds.iter_rows():
        ct_win[(m, int(r))] = int(w)
    return rounds, ct_win


def load_snapshot_rosters():
    lf = pl.scan_parquet(PLAYERS_PATH).select(["match_key", "round", "tick", "player_id", "team_code"])
    t0 = lf.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("t0"))
    snap = (
        lf.join(t0, on=["match_key", "round"])
        .filter(pl.col("tick") == pl.col("t0"))
        .select(["match_key", "round", "player_id", "team_code"])
        .collect()
    )
    rosters = defaultdict(lambda: {CT_CODE: set(), T_CODE: set()})
    for m, r, pid, code in snap.iter_rows():
        if code in (CT_CODE, T_CODE):
            rosters[(m, int(r))][int(code)].add(int(pid))
    return rosters


# ---------------------------------------------------------------- team tracking
def map_team_sides(rosters, ct_win):
    """Return side_A[(m, r)] in {CT_CODE, T_CODE, None} and diagnostics."""
    by_match = defaultdict(list)
    for (m, r) in rosters:
        by_match[m].append(r)
    side_A = {}
    diag = {"matches_no_anchor": 0, "ambiguous_rounds": 0, "low_overlap_rounds": 0,
            "side_changes": 0, "side_changes_halftime_12_13": 0, "side_changes_ot_27_28": 0,
            "side_changes_across_missing_round": 0, "side_changes_other": 0}
    for m, rs in by_match.items():
        rs = sorted(rs)
        anchor = None
        for r in rs:
            if len(rosters[(m, r)][CT_CODE]) == 5:
                anchor = set(rosters[(m, r)][CT_CODE])
                break
        if anchor is None:
            diag["matches_no_anchor"] += 1
            for r in rs:
                side_A[(m, r)] = None
            continue
        prev_side, prev_r = None, None
        for r in rs:
            ct_set = rosters[(m, r)][CT_CODE]
            t_set = rosters[(m, r)][T_CODE]
            oc = len(ct_set & anchor)
            ot = len(t_set & anchor)
            if max(oc, ot) < 3:
                diag["low_overlap_rounds"] += 1
                side_A[(m, r)] = None
                continue
            if oc == ot:
                diag["ambiguous_rounds"] += 1
                side_A[(m, r)] = None
                continue
            side = CT_CODE if oc > ot else T_CODE
            side_A[(m, r)] = side
            if prev_side is not None and side != prev_side:
                diag["side_changes"] += 1
                if prev_r == 12 and r == 13:
                    diag["side_changes_halftime_12_13"] += 1
                elif prev_r == 27 and r == 28:
                    diag["side_changes_ot_27_28"] += 1
                elif r - prev_r > 1:
                    diag["side_changes_across_missing_round"] += 1
                else:
                    diag["side_changes_other"] += 1
            prev_side, prev_r = side, r
    return side_A, diag


def team_outcomes(side_A, ct_win):
    """won[(m, team)][r] = 1 if team won round r (regulation rounds only)."""
    won = defaultdict(dict)
    dropped_no_side = 0
    for (m, r), side in side_A.items():
        if r > REG_MAX_ROUND:
            continue
        if (m, r) not in ct_win:
            continue
        if side is None:
            dropped_no_side += 1
            continue
        a_won = 1 if ((side == CT_CODE) == (ct_win[(m, r)] == 1)) else 0
        won[(m, "A")][r] = a_won
        won[(m, "B")][r] = 1 - a_won
    return won, dropped_no_side


# ---------------------------------------------------------------- streak observations
def streak_observations(won, excluded):
    """Yield (match, line, bucket, y) for each team-round with an eligible target."""
    obs = []
    stats = {"state_rows": 0, "no_target": 0, "target_excluded": 0,
             "dropped_open_run_below_5": 0}
    for (m, team), wm in won.items():
        for r in sorted(wm):
            stats["state_rows"] += 1
            o = wm[r]
            k = 1
            j = r - 1
            while j in wm and wm[j] == o:
                k += 1
                j -= 1
            bounded = (j == 0) or (j in wm)   # run starts at round 1, or is cut by an opposite round
            t = r + 1
            if t > REG_MAX_ROUND or t not in wm:
                stats["no_target"] += 1
                continue
            if t in excluded:
                stats["target_excluded"] += 1
                continue
            if k >= 5:
                b = 5
            elif bounded:
                b = k
            else:
                stats["dropped_open_run_below_5"] += 1
                continue
            line = "win" if o == 1 else "loss"
            obs.append((m, line, b, wm[t]))
    return obs, stats


def aggregate_by_match(obs, match_index, n_matches):
    per_match = [defaultdict(lambda: [0, 0]) for _ in range(n_matches)]
    for m, line, b, y in obs:
        cell = per_match[match_index[m]][(line, b)]
        cell[0] += 1
        cell[1] += y
    return per_match


def pooled(per_match):
    N = defaultdict(int)
    W = defaultdict(int)
    for pm in per_match:
        for key, (n, w) in pm.items():
            N[key] += n
            W[key] += w
    return N, W


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap(per_match, n_boot, seed):
    M = len(per_match)
    rng = random.Random(seed)
    keys = [(line, b) for line in LINES for b in BUCKETS]
    reps = {key: [] for key in keys}
    diff_reps = {b: [] for b in BUCKETS}
    for _ in range(n_boot):
        idx = [rng.randrange(M) for _ in range(M)]
        N = defaultdict(int)
        W = defaultdict(int)
        for i in idx:
            for key, (n, w) in per_match[i].items():
                N[key] += n
                W[key] += w
        rate = {}
        for key in keys:
            rate[key] = (W[key] / N[key]) if N[key] > 0 else None
            if rate[key] is not None:
                reps[key].append(rate[key])
        for b in BUCKETS:
            if rate[("win", b)] is not None and rate[("loss", b)] is not None:
                diff_reps[b].append(rate[("win", b)] - rate[("loss", b)])
    return reps, diff_reps


def summarise(per_match, n_boot, seed):
    N, W = pooled(per_match)
    reps, diff_reps = bootstrap(per_match, n_boot, seed)
    series = {}
    for line in LINES:
        rows = []
        for b in BUCKETS:
            n = N[(line, b)]
            w = W[(line, b)]
            p = (w / n) if n > 0 else None
            lo = percentile(reps[(line, b)], 0.025)
            hi = percentile(reps[(line, b)], 0.975)
            se = None
            if reps[(line, b)]:
                mu = sum(reps[(line, b)]) / len(reps[(line, b)])
                se = math.sqrt(sum((x - mu) ** 2 for x in reps[(line, b)]) / (len(reps[(line, b)]) - 1))
            rows.append({
                "k": b, "label": "5+" if b == 5 else str(b),
                "n_targets": n, "n_target_wins": w,
                "p": p, "ci95_lo": lo, "ci95_hi": hi, "boot_se": se,
                "boot_valid_reps": len(reps[(line, b)]),
            })
        series[line] = rows
    diffs = []
    for b in BUCKETS:
        p_w = series["win"][b - 1]["p"]
        p_l = series["loss"][b - 1]["p"]
        d = (p_w - p_l) if (p_w is not None and p_l is not None) else None
        diffs.append({
            "k": b, "label": "5+" if b == 5 else str(b),
            "diff_win_minus_loss": d,
            "ci95_lo": percentile(diff_reps[b], 0.025),
            "ci95_hi": percentile(diff_reps[b], 0.975),
            "boot_valid_reps": len(diff_reps[b]),
        })
    total_n = sum(N.values())
    total_w = sum(W.values())
    return {"series": series, "difference": diffs,
            "pooled_target_win_rate_all_eligible": (total_w / total_n) if total_n else None,
            "eligible_target_obs": total_n}


# ---------------------------------------------------------------- SVG
def fmt_pct(p):
    return f"{100 * p:.1f}%"


def nice_domain(lo_vals, hi_vals):
    lo = min(lo_vals) - 0.01
    hi = max(hi_vals) + 0.01
    step = 0.05
    ymin = max(0.0, math.floor(lo / step) * step)
    ymax = min(1.0, math.ceil(hi / step) * step)
    return ymin, ymax, step


def build_svg(result, sample_info, dom):
    W, H = 800, 450
    PL, PR, PT, PB = 92, 40, 62, 336
    pw = W - PL - PR
    ymin, ymax, step = dom

    def X(i):
        return PL + (i + 0.5) * pw / 5.0

    def Y(v):
        return PB - (v - ymin) / (ymax - ymin) * (PB - PT)

    out = []
    a = out.append
    a(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">')
    a(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # gridlines and y tick labels
    v = ymin
    ticks = []
    while v <= ymax + 1e-9:
        ticks.append(round(v, 4))
        v += step
    for t in ticks:
        y = Y(t)
        a(f'<line x1="{PL}" y1="{y:.1f}" x2="{W - PR}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        a(f'<text x="{PL - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">{t * 100:.0f}%</text>')

    # 50% reference line (no momentum baseline for team-level win probability)
    if ymin <= 0.5 <= ymax:
        y50 = Y(0.5)
        a(f'<line x1="{PL}" y1="{y50:.1f}" x2="{W - PR}" y2="{y50:.1f}" stroke="#475569" stroke-width="1" stroke-dasharray="4 4" opacity="0.6"/>')

    # axes
    a(f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PB}" stroke="#475569" stroke-width="1.2"/>')
    a(f'<line x1="{PL}" y1="{PB}" x2="{W - PR}" y2="{PB}" stroke="#475569" stroke-width="1.2"/>')

    # x ticks, labels, sample sizes
    for i, b in enumerate(BUCKETS):
        x = X(i)
        a(f'<line x1="{x:.1f}" y1="{PB}" x2="{x:.1f}" y2="{PB + 5}" stroke="#475569" stroke-width="1.2"/>')
        lab = "5+" if b == 5 else str(b)
        a(f'<text x="{x:.1f}" y="{PB + 19}" text-anchor="middle" font-size="12" fill="#475569">{lab}</text>')
        nw = result["series"]["win"][i]["n_targets"]
        a(f'<text x="{x:.1f}" y="{PB + 34}" text-anchor="middle" font-size="11" fill="#475569">n = {nw:,}</text>')

    # series: error bars, lines, markers, value labels
    offset = {"win": -8, "loss": 8}
    for line in LINES:
        col = COLORS[line]
        pts = []
        for i, row in enumerate(result["series"][line]):
            xc = X(i) + offset[line]
            p, lo, hi = row["p"], row["ci95_lo"], row["ci95_hi"]
            if p is None or lo is None or hi is None:
                continue
            pts.append((xc, Y(p)))
            a(f'<line x1="{xc:.1f}" y1="{Y(lo):.1f}" x2="{xc:.1f}" y2="{Y(hi):.1f}" stroke="{col}" stroke-width="1.6"/>')
            a(f'<line x1="{xc - 5:.1f}" y1="{Y(lo):.1f}" x2="{xc + 5:.1f}" y2="{Y(lo):.1f}" stroke="{col}" stroke-width="1.6"/>')
            a(f'<line x1="{xc - 5:.1f}" y1="{Y(hi):.1f}" x2="{xc + 5:.1f}" y2="{Y(hi):.1f}" stroke="{col}" stroke-width="1.6"/>')
        if len(pts) > 1:
            pl = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            a(f'<polyline points="{pl}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        for i, row in enumerate(result["series"][line]):
            p = row["p"]
            lo, hi = row["ci95_lo"], row["ci95_hi"]
            if p is None or lo is None or hi is None:
                continue
            xc = X(i) + offset[line]
            a(f'<circle cx="{xc:.1f}" cy="{Y(p):.1f}" r="4.5" fill="{col}" stroke="#ffffff" stroke-width="1.5"/>')
            if line == "win":
                ly = Y(hi) - 9
            else:
                ly = Y(lo) + 17
            a(f'<text x="{xc:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="12" fill="{col}">{fmt_pct(p)}</text>')

    # legend row above the plot (no overlap with data)
    ly = 22
    lx1 = PL
    a(f'<line x1="{lx1}" y1="{ly - 4}" x2="{lx1 + 22}" y2="{ly - 4}" stroke="{COLORS["win"]}" stroke-width="2.2"/>')
    a(f'<circle cx="{lx1 + 11}" cy="{ly - 4}" r="4" fill="{COLORS["win"]}"/>')
    a(f'<text x="{lx1 + 30}" y="{ly}" font-size="12" fill="#1e293b">After a win streak</text>')
    lx2 = PL + 190
    a(f'<line x1="{lx2}" y1="{ly - 4}" x2="{lx2 + 22}" y2="{ly - 4}" stroke="{COLORS["loss"]}" stroke-width="2.2"/>')
    a(f'<circle cx="{lx2 + 11}" cy="{ly - 4}" r="4" fill="{COLORS["loss"]}"/>')
    a(f'<text x="{lx2 + 30}" y="{ly}" font-size="12" fill="#1e293b">After a loss streak</text>')

    # sample-size label (row below legend, left aligned to the plot)
    a(f'<text x="{PL}" y="{PT - 14}" font-size="12" fill="#1e293b">n = {sample_info["regulation_rounds"]:,} regulation rounds ({sample_info["rounds"]:,} in sample) from {sample_info["matches"]} matches; {sample_info["targets_total"]:,} team-round targets</text>')

    # axis titles
    a(f'<text transform="translate(22,{(PT + PB) / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="13" fill="#1e293b">P(team wins round r+1)</text>')
    a(f'<text x="{(PL + W - PR) / 2:.1f}" y="{PB + 70}" text-anchor="middle" font-size="13" fill="#1e293b">Streak length k (consecutive rounds won or lost, ending in round r)</text>')

    # footnotes
    a(f'<text x="24" y="{PB + 92}" font-size="11" fill="#475569">Teams tracked across the halftime swap by freeze-end roster. Targets exclude pistols (rounds 1, 13) and round 14.</text>')
    a(f'<text x="24" y="{PB + 106}" font-size="11" fill="#475569">Bars: 95% match-cluster bootstrap (150 matches, 1,000 resamples, seed 0). Dashed line: 50%. Regulation rounds 1-24.</text>')
    a('</svg>')
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- main
def main():
    rounds, ct_win = load_rounds()
    rosters = load_snapshot_rosters()
    side_A, map_diag = map_team_sides(rosters, ct_win)
    won, dropped_no_side = team_outcomes(side_A, ct_win)

    matches = sorted(rounds["match_key"].unique().to_list())
    match_index = {m: i for i, m in enumerate(matches)}
    M = len(matches)

    # consistency checks on team outcomes
    per_round_teams = defaultdict(int)
    for (m, team), wm in won.items():
        for r in wm:
            per_round_teams[(m, r)] += 1
    bad_rounds = [k for k, v in per_round_teams.items() if v != 2]
    assert not bad_rounds, "each mapped round must yield exactly two team rows"
    team_wins_per_round = defaultdict(int)
    for (m, team), wm in won.items():
        for r, w in wm.items():
            team_wins_per_round[(m, r)] += w
    assert all(v == 1 for v in team_wins_per_round.values()), "exactly one team wins each round"

    regulation_rounds = rounds.filter(pl.col("round") <= REG_MAX_ROUND).height
    ot_rounds = rounds.filter(pl.col("round") > REG_MAX_ROUND).height

    obs_main, st_main = streak_observations(won, EXCLUDED_TARGETS)
    obs_sens, st_sens = streak_observations(won, SENSITIVITY_EXCLUDED)
    pm_main = aggregate_by_match(obs_main, match_index, M)
    pm_sens = aggregate_by_match(obs_sens, match_index, M)

    res_main = summarise(pm_main, N_BOOT, SEED)
    res_sens = summarise(pm_sens, N_BOOT, SEED)

    # validity checks
    # structural identity: each round enters the win line (winner's streak) and the loss line (loser's
    # streak) with the same k, so the per-bucket target counts are equal for both lines.
    for b_i in range(len(BUCKETS)):
        assert res_main["series"]["win"][b_i]["n_targets"] == res_main["series"]["loss"][b_i]["n_targets"]
    for line in LINES:
        for row in res_main["series"][line]:
            if row["n_targets"] > 0:
                assert 0.0 <= row["p"] <= 1.0
                assert row["ci95_lo"] <= row["p"] <= row["ci95_hi"] or abs(row["p"] - row["ci95_lo"]) < 0.02 or abs(row["p"] - row["ci95_hi"]) < 0.02, (line, row)
            assert row["n_targets"] >= 0 and row["n_target_wins"] >= 0
            assert row["n_target_wins"] <= row["n_targets"]

    sample_info = {"rounds": rounds.height, "regulation_rounds": regulation_rounds, "matches": M,
                   "targets_total": res_main["eligible_target_obs"]}

    all_p = [r["ci95_lo"] for ln in LINES for r in res_main["series"][ln] if r["ci95_lo"] is not None] + \
            [0.5]
    all_h = [r["ci95_hi"] for ln in LINES for r in res_main["series"][ln] if r["ci95_hi"] is not None] + \
            [0.5]
    dom = nice_domain(all_p, all_h)
    svg = build_svg(res_main, sample_info, dom)

    key_numbers = {
        "sample_rounds_in_1hz_sample": rounds.height,
        "sample_matches": M,
        "regulation_rounds_used": regulation_rounds,
        "overtime_rounds_not_used": ot_rounds,
        "rounds_dropped_no_side_mapping": dropped_no_side,
        "eligible_target_observations_main": res_main["eligible_target_obs"],
        "eligible_target_observations_sensitivity": res_sens["eligible_target_obs"],
        "pooled_target_win_rate_main": res_main["pooled_target_win_rate_all_eligible"],
        "win_streak_k1": {"n": res_main["series"]["win"][0]["n_targets"], "p": res_main["series"]["win"][0]["p"],
                          "ci95": [res_main["series"]["win"][0]["ci95_lo"], res_main["series"]["win"][0]["ci95_hi"]]},
        "loss_streak_k1": {"n": res_main["series"]["loss"][0]["n_targets"], "p": res_main["series"]["loss"][0]["p"],
                           "ci95": [res_main["series"]["loss"][0]["ci95_lo"], res_main["series"]["loss"][0]["ci95_hi"]]},
        "win_streak_5plus": {"n": res_main["series"]["win"][4]["n_targets"], "p": res_main["series"]["win"][4]["p"],
                             "ci95": [res_main["series"]["win"][4]["ci95_lo"], res_main["series"]["win"][4]["ci95_hi"]]},
        "loss_streak_5plus": {"n": res_main["series"]["loss"][4]["n_targets"], "p": res_main["series"]["loss"][4]["p"],
                              "ci95": [res_main["series"]["loss"][4]["ci95_lo"], res_main["series"]["loss"][4]["ci95_hi"]]},
    }

    out = {
        "id": "T10",
        "title": "Momentum: next-round win rate after win and loss streaks",
        "description": "Team-tracked momentum. P(team wins round r+1) given the team won (or lost) each of its last k rounds, k = 1..4 and 5+.",
        "sample": {
            "source": "data/state_sample (1 Hz sample, days 01-07 of 2026-10)",
            "rounds": rounds.height,
            "matches": M,
            "regulation_rounds_used_for_streaks_and_targets": regulation_rounds,
            "overtime_rounds_excluded": ot_rounds,
            "note": "rounds.parquet has 2,673 rows. One round has 4 T-side players in the freeze-end snapshot; it is kept because team mapping uses overlap.",
        },
        "method": {
            "outcome": "team outcome per round, from the side the team occupied (CT or T) and ct_win",
            "team_tracking": "Team A = the five players on CT at the first observed round with five CT players in each match (freeze-end roster). Team B = the other five. Each round, team A's side is the side whose freeze-end roster overlaps team A's roster most. Sides swap at the halftime (after round 12), so team identity is carried through the swap by player roster, not by side.",
            "streak": "k = number of consecutive regulation rounds ending in round r that the team won (win line) or lost (loss line). Streak must be contiguous in round number. Lengths 1-4 are exact when the run is bounded by an opposite-outcome round or round 1; 5 means 5 or more (5+).",
            "target": "round r+1 must be observed, regulation, and not in the excluded set. y = 1 if the team won round r+1.",
            "excluded_targets_main": sorted(EXCLUDED_TARGETS),
            "excluded_targets_sensitivity": sorted(SENSITIVITY_EXCLUDED),
            "interval": "match-cluster bootstrap: 1,000 resamples of the 150 matches with replacement, random.Random(0), 95% percentile interval (linear interpolation). Point estimate = pooled rate over all matches.",
            "team_rows_per_round": 2,
            "lines_are_team_level": True,
            "line_construction": "Each round enters twice: as the winner's win-streak (win line) and as the loser's loss-streak (loss line), with the same streak length k. Per-bucket n is therefore identical for the two lines; the lines differ in which team is observed (winner vs loser) and in the target outcome.",
        },
        "series": res_main["series"],
        "difference_win_minus_loss": res_main["difference"],
        "sensitivity_targets_excluding_only_pistols": {
            "series": res_sens["series"],
            "difference_win_minus_loss": res_sens["difference"],
            "eligible_target_obs": res_sens["eligible_target_obs"],
        },
        "counts": {
            "main": {"state_rows": st_main["state_rows"], "no_target": st_main["no_target"],
                     "target_excluded": st_main["target_excluded"],
                     "dropped_open_run_below_5": st_main["dropped_open_run_below_5"],
                     "eligible_targets": res_main["eligible_target_obs"]},
            "sensitivity": {"state_rows": st_sens["state_rows"], "no_target": st_sens["no_target"],
                            "target_excluded": st_sens["target_excluded"],
                            "dropped_open_run_below_5": st_sens["dropped_open_run_below_5"],
                            "eligible_targets": res_sens["eligible_target_obs"]},
        },
        "team_tracking_diagnostics": map_diag,
        "key_numbers": key_numbers,
        "assumptions_and_caveats": [
            "Team tracking uses freeze-end rosters; a player swapped out mid-match would not change team identity, since the anchor set is fixed at the first 5-CT round.",
            "Targets exclude pistol rounds 1 and 13 and round 14. Round 14 is treated as the round right after the halftime swap (the first full-buy round after the round-13 pistol). The sensitivity block excludes only 1 and 13.",
            "Streaks use regulation rounds only. A gap in the sample (rounds missing from rounds.parquet) breaks a streak; a run that is cut by a gap is kept only at 5+ because its exact length is unknown.",
            "Descriptive, not causal. A team on a streak may be stronger in that match, the score state and economy (loss bonus, buy choices) both move with streaks, and side-specific map effects are not removed.",
            "Each match contributes two team rows, both kept, so the bootstrap resamples whole matches; team-level observations are not independent.",
            "Bins 4 and 5+ can be small; check the n labels on the chart before reading their points.",
        ],
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=lambda o: None)
        fh.write("\n")
    with open(OUT_SVG, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(json.dumps(key_numbers, indent=2, default=str))
    print("main series:")
    for line in LINES:
        for row in res_main["series"][line]:
            print(line, row["label"], row["n_targets"], row["n_target_wins"], row["p"], row["ci95_lo"], row["ci95_hi"])
    print("sensitivity series:")
    for line in LINES:
        for row in res_sens["series"][line]:
            print(line, row["label"], row["n_targets"], row["n_target_wins"], row["p"], row["ci95_lo"], row["ci95_hi"])
    print("counts main:", out["counts"]["main"])
    print("counts sens:", out["counts"]["sensitivity"])
    print("map diag:", map_diag)
    print("dropped no side:", dropped_no_side)


if __name__ == "__main__":
    main()
