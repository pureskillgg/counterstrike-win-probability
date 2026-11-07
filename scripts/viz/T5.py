#!/usr/bin/env python3
"""T5: plant timing and CT win rate by plant-time bin (CS2 csds, 2026-10-01 to 07).

Writes data/viz/T5.json and data/viz/T5.svg under the repo root.

Definitions
- Live start: round_freeze_end tick (verified equal to rounds.parquet live_start_tick).
- Round end: basic_round_end tick (verified equal to rounds.parquet round_end_tick).
- Plant: basic_bomb_state event_type == bomb_planted. A plant counts only when
  live_start <= tick < round_end. Plants at or after round end are excluded.
- CT win: winner_team_code == 3 (equivalently rounds.parquet ct_win).

Intervals: match-cluster bootstrap (resample matches with replacement), 1000 resamples,
random.Random(0) from the standard library, 2.5th and 97.5th percentiles
(linear interpolation between order statistics).

Only the standard library and polars (for reading parquet) are used.
"""
import glob
import json
import math
import os
import random
import statistics
from collections import Counter

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SAMPLE_DIR = os.path.join(ROOT, "data", "state_sample")
TOME_CSDS = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
OUT_JSON = os.path.join(ROOT, "data", "viz", "T5.json")
OUT_SVG = os.path.join(ROOT, "data", "viz", "T5.svg")

TICK_RATE = 64
BIN_S = 10
N_BINS = 12  # 0-120 s displayed; anything later is counted as overflow
DISPLAY_MAX_S = N_BINS * BIN_S
N_BOOT = 1000
SEED = 0
LOW_N = 30

CT = "#2563eb"
NEU = "#475569"
GRID = "#e2e8f0"


# ---------------------------------------------------------------- loading

def channel_files(channel):
    pattern = os.path.join(TOME_CSDS, channel + ".*", "dataframe_*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return files


def load_planted():
    df = (
        pl.scan_parquet(channel_files("basic_bomb_state"))
        .filter(pl.col("event_type") == "bomb_planted")
        .group_by(["match_key", "round"])
        .agg(pl.col("tick").min().alias("bp_tick"), pl.len().alias("n_events"))
        .collect()
    )
    planted = {}
    for r in df.to_dicts():
        planted[(r["match_key"], r["round"])] = (r["bp_tick"], r["n_events"])
    return planted, int(df["n_events"].max()) if df.height else 0


def load_freeze_end():
    df = (
        pl.scan_parquet(channel_files("basic_round_state"))
        .filter(pl.col("event_type") == "round_freeze_end")
        .group_by(["match_key", "round"])
        .agg(pl.col("tick").min().alias("fe_tick"), pl.len().alias("n_events"))
        .collect()
    )
    fe = {(r["match_key"], r["round"]): r["fe_tick"] for r in df.to_dicts()}
    return fe, int(df["n_events"].max()) if df.height else 0


def load_round_end():
    df = (
        pl.scan_parquet(channel_files("basic_round_end"))
        .select(["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
        .collect()
    )
    return df.to_dicts()


def load_sample_rounds():
    df = pl.read_parquet(
        os.path.join(SAMPLE_DIR, "rounds.parquet"),
        columns=[
            "match_key",
            "round",
            "ct_win",
            "live_start_tick",
            "round_end_tick",
            "plant_tick",
        ],
    )
    return df.to_dicts()


# ---------------------------------------------------------------- populations

def sample_population(fe, re_by_key):
    rows = []
    checks = Counter()
    for r in load_sample_rounds():
        key = (r["match_key"], r["round"])
        checks["sample_rounds"] += 1
        re = re_by_key.get(key)
        if re is None:
            checks["sample_round_without_round_end"] += 1
            continue
        if fe.get(key) != r["live_start_tick"]:
            checks["live_start_mismatch_vs_freeze_end"] += 1
        if re["tick"] != r["round_end_tick"]:
            checks["round_end_mismatch_vs_round_end_channel"] += 1
        ct = 1 if re["winner_team_code"] == 3 else 0
        if ct != int(r["ct_win"]):
            checks["ct_win_mismatch_vs_winner_team_code"] += 1
        if re["player_count"] != 10:
            checks["sample_round_player_count_not_10"] += 1
        rows.append(
            {
                "match_key": r["match_key"],
                "round": r["round"],
                "ct_win": int(r["ct_win"]),
                "live": int(r["live_start_tick"]),
                "end": int(r["round_end_tick"]),
                "win_reason": re["win_reason_code"],
                "plant_tick_sample": r["plant_tick"],
            }
        )
    return rows, checks


def tome_population(fe):
    rows = []
    checks = Counter()
    for r in load_round_end():
        checks["round_end_rows"] += 1
        if r["player_count"] != 10:
            checks["excluded_player_count_not_10"] += 1
            continue
        if r["winner_team_code"] is None:
            checks["excluded_no_winner"] += 1
            continue
        key = (r["match_key"], r["round"])
        live = fe.get(key)
        if live is None:
            checks["excluded_no_round_freeze_end"] += 1
            continue
        rows.append(
            {
                "match_key": r["match_key"],
                "round": r["round"],
                "ct_win": 1 if r["winner_team_code"] == 3 else 0,
                "live": int(live),
                "end": int(r["tick"]),
                "win_reason": r["win_reason_code"],
                "plant_tick_sample": None,
            }
        )
    checks["tome_clean_rounds"] = len(rows)
    return rows, checks


# ---------------------------------------------------------------- per-round records

def build_records(rows, planted):
    recs = []
    for r in rows:
        key = (r["match_key"], r["round"])
        bp = planted.get(key)
        bp_tick = bp[0] if bp else None
        live, end = r["live"], r["end"]
        in_win = bp_tick is not None and live <= bp_tick < end
        after_end = bp_tick is not None and bp_tick >= end
        before_live = bp_tick is not None and bp_tick < live
        plant_s = (bp_tick - live) / TICK_RATE if in_win else None
        recs.append(
            {
                "match": r["match_key"],
                "round": r["round"],
                "ct_win": r["ct_win"],
                "in_win": in_win,
                "after_end": after_end,
                "before_live": before_live,
                "bp_tick": bp_tick,
                "n_planted_events": bp[1] if bp else 0,
                "plant_s": plant_s,
                "win_reason": r.get("win_reason"),
                "bin": int(plant_s // BIN_S) if in_win else None,
                "plant_tick_sample": r.get("plant_tick_sample"),
            }
        )
    return recs


def match_table(recs):
    agg = {}
    for x in recs:
        m = agg.get(x["match"])
        if m is None:
            m = agg[x["match"]] = {
                "rounds": 0,
                "plants": 0,
                "ct_plant": 0,
                "no_rounds": 0,
                "no_ct": 0,
                "n": [0] * N_BINS,
                "k": [0] * N_BINS,
            }
        m["rounds"] += 1
        if x["in_win"]:
            m["plants"] += 1
            m["ct_plant"] += x["ct_win"]
            if x["bin"] < N_BINS:
                m["n"][x["bin"]] += 1
                m["k"][x["bin"]] += x["ct_win"]
        else:
            m["no_rounds"] += 1
            m["no_ct"] += x["ct_win"]
    return list(agg.values())


def totals(mtab, weights):
    R = P = N0 = K0 = KP = 0
    nb = [0] * N_BINS
    kb = [0] * N_BINS
    for m, w in zip(mtab, weights):
        if w == 0:
            continue
        R += w * m["rounds"]
        P += w * m["plants"]
        N0 += w * m["no_rounds"]
        K0 += w * m["no_ct"]
        KP += w * m["ct_plant"]
        for b in range(N_BINS):
            if m["n"][b]:
                nb[b] += w * m["n"][b]
                kb[b] += w * m["k"][b]
    return {
        "rounds": R,
        "plants": P,
        "no_rounds": N0,
        "no_ct": K0,
        "ct_plant": KP,
        "n": nb,
        "k": kb,
    }


def pct(vals, q):
    v = sorted(x for x in vals if x is not None and not math.isnan(x))
    if not v:
        return None
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def ci95(vals):
    return [pct(vals, 0.025), pct(vals, 0.975)]


def bootstrap(mtab):
    rng = random.Random(SEED)
    M = len(mtab)
    rep_rate = []
    rep_nopl = []
    rep_share = [[] for _ in range(N_BINS)]
    rep_ct = [[] for _ in range(N_BINS)]
    invalid = [0] * N_BINS
    for _ in range(N_BOOT):
        counts = Counter(rng.randrange(M) for _ in range(M))
        w = [0] * M
        for idx, c in counts.items():
            w[idx] = c
        t = totals(mtab, w)
        rep_rate.append(t["plants"] / t["rounds"])
        rep_nopl.append(t["no_ct"] / t["no_rounds"] if t["no_rounds"] else None)
        for b in range(N_BINS):
            rep_share[b].append(t["n"][b] / t["rounds"])
            if t["n"][b]:
                rep_ct[b].append(t["k"][b] / t["n"][b])
            else:
                invalid[b] += 1
    return {
        "plant_rate": rep_rate,
        "no_plant_ct": rep_nopl,
        "share": rep_share,
        "ct": rep_ct,
        "invalid": invalid,
    }


def summarise(recs, mtab):
    M = len(mtab)
    pe = totals(mtab, [1] * M)
    rep = bootstrap(mtab)
    R = pe["rounds"]
    P = pe["plants"]
    plant_times = [x["plant_s"] for x in recs if x["in_win"]]
    after = [x for x in recs if x["after_end"]]
    ct_all = sum(x["ct_win"] for x in recs)
    no_rate = pe["no_ct"] / pe["no_rounds"]
    clean_no = [x for x in recs if not x["in_win"] and not x["after_end"]]
    kn = {
        "rounds": R,
        "matches": M,
        "plants_in_live_window": P,
        "plant_rate": P / R,
        "plant_rate_ci95": ci95(rep["plant_rate"]),
        "no_plant_rounds": pe["no_rounds"],
        "plant_time_mean_s": statistics.fmean(plant_times),
        "plant_time_median_s": statistics.median(plant_times),
        "plant_time_min_s": min(plant_times),
        "plant_time_max_s": max(plant_times),
        "plants_in_0_120s_bins": sum(pe["n"]),
        "plants_overflow_over_120s": P - sum(pe["n"]),
        "ct_win_rate_all_rounds": ct_all / R,
        "ct_win_rate_plant_rounds": pe["ct_plant"] / P,
        "ct_win_rate_no_plant_rounds": no_rate,
        "ct_win_rate_no_plant_rounds_ci95": ci95(rep["no_plant_ct"]),
        "bomb_planted_after_round_end": len(after),
        "bomb_planted_after_round_end_ct_wins": sum(x["ct_win"] for x in after),
        "bomb_planted_after_round_end_win_reason_counts": {
            str(k): v for k, v in sorted(Counter(x["win_reason"] for x in after).items())
        },
        "ct_win_rate_no_plant_excluding_late_plant_rounds": (
            sum(x["ct_win"] for x in clean_no) / len(clean_no)
        ),
        "no_plant_rounds_excluding_late_plant_rounds": len(clean_no),
        "plant_rate_if_after_end_plants_counted": (P + len(after)) / R,
        "bomb_planted_before_live_start": sum(1 for x in recs if x["before_live"]),
        "max_planted_events_per_round": max((x["n_planted_events"] for x in recs), default=0),
    }
    # CT win rate among plant rounds, bootstrapped over matches
    rep_plant_ct = []
    rng = random.Random(SEED)
    for _ in range(N_BOOT):
        counts = Counter(rng.randrange(M) for _ in range(M))
        w = [0] * M
        for idx, c in counts.items():
            w[idx] = c
        t = totals(mtab, w)
        rep_plant_ct.append(t["ct_plant"] / t["plants"])
    kn["ct_win_rate_plant_rounds_ci95"] = ci95(rep_plant_ct)

    bins = []
    for b in range(N_BINS):
        n = pe["n"][b]
        k = pe["k"][b]
        bins.append(
            {
                "bin": f"{b * BIN_S}-{(b + 1) * BIN_S}",
                "start_s": b * BIN_S,
                "end_s": (b + 1) * BIN_S,
                "plants": n,
                "share_of_all_rounds": n / R,
                "share_of_all_rounds_ci95": ci95(rep["share"][b]),
                "ct_wins": k,
                "ct_win_rate": (k / n) if n else None,
                "ct_win_rate_ci95": ci95(rep["ct"][b]),
                "low_n": n < LOW_N,
                "bootstrap_replicates_with_zero_plants": rep["invalid"][b],
            }
        )
    return {"key_numbers": kn, "bins": bins}


def run_population(name, rows, planted, checks):
    recs = build_records(rows, planted)
    checks = dict(checks)
    # internal consistency checks
    mism_in = 0
    mism_out = 0
    for x in recs:
        ps = x["plant_tick_sample"]
        if x["in_win"]:
            if ps is None or int(ps) != x["bp_tick"]:
                mism_in += 1
        elif ps is not None:
            mism_out += 1
    if any(r["plant_tick_sample"] is not None for r in rows):
        checks["plant_tick_vs_bomb_state_in_window_mismatch"] = mism_in
        checks["plant_tick_set_outside_window_mismatch"] = mism_out
    bad_bin = sum(1 for x in recs if x["in_win"] and not (0 <= x["bin"]))
    checks["negative_bin_index"] = bad_bin
    mtab = match_table(recs)
    out = summarise(recs, mtab)
    out["name"] = name
    out["checks"] = checks
    return out


# ---------------------------------------------------------------- SVG

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_text(x, y, s, size=12, anchor="start", fill=NEU, rotate=None, weight=None):
    extra = ""
    if rotate is not None:
        extra += f' transform="rotate({rotate} {x:.2f} {y:.2f})"'
    if weight:
        extra += f' font-weight="{weight}"'
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}"{extra}>{esc(s)}</text>'
    )


def svg_line(x1, y1, x2, y2, stroke, width=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width}"{d}/>'
    )


def svg_rect(x, y, w, h, fill, opacity=None):
    op = f' fill-opacity="{opacity}"' if opacity is not None else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{max(h, 0):.2f}" fill="{fill}"{op}/>'


def svg_circle(cx, cy, r, fill, stroke, width=1.5):
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def nice_step(maxv):
    for step in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5):
        if maxv / step <= 6:
            return step
    return 1.0


def fmt_pct(v, step):
    dec = 0 if abs(step * 100 - round(step * 100)) < 1e-9 else 1
    return f"{v * 100:.{dec}f}%"


def build_svg(main):
    kn = main["key_numbers"]
    bins = main["bins"]
    W, H = 800, 450
    Y_TOP, Y_BOT = 92.0, 350.0
    LX0, LX1 = 72.0, 364.0
    RX0, RX1 = 484.0, 776.0
    out = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
        svg_rect(0, 0, W, H, "#ffffff"),
        '<g font-family="sans-serif">',
    ]

    def xpos(sec, x0, x1):
        return x0 + sec / DISPLAY_MAX_S * (x1 - x0)

    # ----- left panel: share of all rounds per bin, histogram bars with CI
    hi_max = max(b["share_of_all_rounds_ci95"][1] for b in bins)
    step = nice_step(hi_max)
    ymax = math.ceil(hi_max * 1.15 / step) * step

    def yl(v):
        return Y_BOT - v / ymax * (Y_BOT - Y_TOP)

    k = 0
    while k * step <= ymax + 1e-9:
        v = k * step
        out.append(svg_line(LX0, yl(v), LX1, yl(v), GRID, 1))
        out.append(svg_text(LX0 - 8, yl(v) + 4, fmt_pct(v, step), anchor="end"))
        k += 1
    bar_w = (LX1 - LX0) / N_BINS * 0.72
    for b in bins:
        cx = xpos(b["start_s"] + BIN_S / 2, LX0, LX1)
        s = b["share_of_all_rounds"]
        lo, hi = b["share_of_all_rounds_ci95"]
        y_top = yl(s)
        out.append(svg_rect(cx - bar_w / 2, y_top, bar_w, Y_BOT - y_top, NEU, 0.85))
        out.append(svg_line(cx, yl(lo), cx, yl(hi), NEU, 1.2))
        out.append(svg_line(cx - 4, yl(hi), cx + 4, yl(hi), NEU, 1.2))
        out.append(svg_line(cx - 4, yl(lo), cx + 4, yl(lo), NEU, 1.2))
        out.append(svg_text(cx, yl(hi) - 6, f"{b['plants']:,}", size=11, anchor="middle"))
    out.append(svg_line(LX0, Y_TOP, LX0, Y_BOT, NEU, 1))
    out.append(svg_line(LX0, Y_BOT, LX1, Y_BOT, NEU, 1))

    # ----- right panel: CT win rate per bin, with CI and no-plant reference
    def yr(v):
        return Y_BOT - v * (Y_BOT - Y_TOP)

    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(svg_line(RX0, yr(v), RX1, yr(v), GRID, 1))
        out.append(svg_text(RX0 - 8, yr(v) + 4, f"{v * 100:.0f}%", anchor="end"))
    nop_lo, nop_hi = kn["ct_win_rate_no_plant_rounds_ci95"]
    nop = kn["ct_win_rate_no_plant_rounds"]
    out.append(svg_rect(RX0, yr(nop_hi), RX1 - RX0, yr(nop_lo) - yr(nop_hi), NEU, 0.12))
    out.append(svg_line(RX0, yr(nop), RX1, yr(nop), NEU, 1.5, dash="6,4"))
    for b in bins:
        cx = xpos(b["start_s"] + BIN_S / 2, RX0, RX1)
        if b["ct_win_rate"] is not None:
            lo, hi = b["ct_win_rate_ci95"]
            out.append(svg_line(cx, yr(lo), cx, yr(hi), CT, 1.4))
            out.append(svg_line(cx - 4, yr(hi), cx + 4, yr(hi), CT, 1.4))
            out.append(svg_line(cx - 4, yr(lo), cx + 4, yr(lo), CT, 1.4))
            fill = "#ffffff" if b["low_n"] else CT
            out.append(svg_circle(cx, yr(b["ct_win_rate"]), 4.5, fill, CT, 1.8))
        out.append(svg_text(cx, Y_BOT + 36, f"{b['plants']:,}", size=11, anchor="middle"))
    out.append(svg_line(RX0, Y_TOP, RX0, Y_BOT, NEU, 1))
    out.append(svg_line(RX0, Y_BOT, RX1, Y_BOT, NEU, 1))

    # ----- x axes: ticks and labels
    for s in range(0, DISPLAY_MAX_S + 1, 20):
        for x0, x1 in ((LX0, LX1), (RX0, RX1)):
            x = xpos(s, x0, x1)
            out.append(svg_line(x, Y_BOT, x, Y_BOT + 4, NEU, 1))
            out.append(svg_text(x, Y_BOT + 18, f"{s}", anchor="middle"))
    out.append(svg_text((LX0 + LX1) / 2, 404, "Seconds from live start to bomb plant", size=13, anchor="middle"))
    out.append(svg_text((RX0 + RX1) / 2, 404, "Seconds from live start to bomb plant", size=13, anchor="middle"))

    # ----- y axis titles
    out.append(svg_text(18, (Y_TOP + Y_BOT) / 2, "Share of all rounds", size=13, anchor="middle", rotate=-90))
    out.append(svg_text(428, (Y_TOP + Y_BOT) / 2, "CT win rate", size=13, anchor="middle", rotate=-90))

    # ----- headers and legend
    n_rounds = kn["rounds"]
    n_matches = kn["matches"]
    P = kn["plants_in_live_window"]
    pr_lo, pr_hi = kn["plant_rate_ci95"]
    out.append(svg_text(LX0, 30, f"n = {n_rounds:,} rounds, {n_matches} matches"))
    out.append(svg_text(LX0, 46, f"{P:,} plants in live window = {kn['plant_rate'] * 100:.1f}% of rounds"))
    out.append(svg_text(LX0, 62, f"95% CI {pr_lo * 100:.1f}% to {pr_hi * 100:.1f}%"))

    out.append(svg_text(RX0, 30, f"n = {P:,} plant rounds, 10 s bins"))
    out.append(svg_circle(RX0 + 6, 42, 4, CT, CT, 1.5))
    out.append(svg_text(RX0 + 16, 46, "CT win rate by plant bin, 95% CI"))
    out.append(svg_line(RX0, 58, RX0 + 12, 58, NEU, 1.5, dash="6,4"))
    out.append(
        svg_text(
            RX0 + 16,
            62,
            f"No plant in live window (n = {kn['no_plant_rounds']:,})",
        )
    )

    # ----- footer
    n_excl = kn["bomb_planted_after_round_end"]
    out.append(
        svg_text(
            16,
            428,
            "Source: 1 Hz sample, 150 matches. Hollow markers: fewer than 30 plants. "
            "Plant window: freeze end to round end.",
            size=11,
        )
    )
    out.append(
        svg_text(
            16,
            442,
            f"95% intervals: match-cluster bootstrap, 1,000 resamples, seed 0. "
            f"{n_excl} plants logged after round end are excluded.",
            size=11,
        )
    )
    out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- main

def main():
    planted, max_events = load_planted()
    fe, fe_max_events = load_freeze_end()
    re_rows = load_round_end()
    re_by_key = {(r["match_key"], r["round"]): r for r in re_rows}

    sample_rows, sample_checks = sample_population(fe, re_by_key)
    sample_checks["round_end_channel_duplicate_keys"] = len(re_rows) - len(re_by_key)
    sample_checks["bomb_planted_max_events_per_round"] = max_events
    sample_checks["round_freeze_end_max_events_per_round"] = fe_max_events
    primary = run_population("primary_1hz_sample", sample_rows, planted, sample_checks)

    tome_rows, tome_checks = tome_population(fe)
    sensitivity = run_population("sensitivity_full_tome_clean_rounds", tome_rows, planted, tome_checks)

    result = {
        "id": "T5",
        "title": "Plant timing and CT win rate by plant-time bin",
        "definitions": {
            "live_start": "basic_round_state round_freeze_end tick (min per match and round)",
            "round_end": "basic_round_end tick",
            "plant": "basic_bomb_state bomb_planted tick with live_start <= tick < round_end",
            "ct_win": "round_end winner_team_code == 3",
            "bins": f"{BIN_S} s bins of seconds from live start, {DISPLAY_MAX_S} s displayed",
            "interval": f"95% percentile, match-cluster bootstrap, {N_BOOT} resamples, random.Random({SEED})",
            "low_n_threshold": LOW_N,
        },
        "primary": primary,
        "sensitivity_full_tome": sensitivity,
        "caveats": [
            "Sample rounds with a bomb_planted event at or after round_end are excluded as specified (counts and win_reason_code split in primary.key_numbers). A plant after the round has ended is physically odd and suggests a timestamp or round-end logging artifact; the cause was not verified.",
            "Plant-rate denominator is all 2,673 clean sample rounds; excluded late plants count as no plant.",
            "The 1 Hz sample is a subset of the full 2026 tome; the sensitivity block is not independent of the primary result.",
            "Bootstrap percentile intervals are not bias-corrected; low-n bins have wide intervals and some resamples can drop a bin entirely.",
        ],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
    svg = build_svg(primary)
    with open(OUT_SVG, "w", encoding="utf-8", newline="\n") as f:
        f.write(svg)

    kn = primary["key_numbers"]
    print("primary rounds", kn["rounds"], "plants", kn["plants_in_live_window"], "rate", round(kn["plant_rate"], 4))
    print("plant rate ci", [round(x, 4) for x in kn["plant_rate_ci95"]])
    print("after-end plants", kn["bomb_planted_after_round_end"], "before live", kn["bomb_planted_before_live_start"])
    print("no-plant CT rate", round(kn["ct_win_rate_no_plant_rounds"], 4), [round(x, 4) for x in kn["ct_win_rate_no_plant_rounds_ci95"]])
    for b in primary["bins"]:
        print(
            b["bin"],
            b["plants"],
            round(b["share_of_all_rounds"], 4),
            b["ct_win_rate"] if b["ct_win_rate"] is None else round(b["ct_win_rate"], 4),
            [None if x is None else round(x, 4) for x in b["ct_win_rate_ci95"]],
        )
    print("checks", primary["checks"])
    skn = sensitivity["key_numbers"]
    print("tome rounds", skn["rounds"], "matches", skn["matches"], "plants", skn["plants_in_live_window"], "rate", round(skn["plant_rate"], 4))
    print("tome checks", sensitivity["checks"])


if __name__ == "__main__":
    main()
