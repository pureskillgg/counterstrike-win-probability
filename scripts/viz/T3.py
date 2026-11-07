"""T3: kills versus shots by weapon (top 20 weapon groups), 2026-10 tome sample.

Shots come from basic_weapon_fire, kills from basic_player_death (enemy attackers only),
both restricted to rounds with player_count == 10 in basic_round_end and to events at or
before that round's end tick. Intervals are a match-cluster bootstrap (resample matches
with replacement, 1000 resamples, numpy default_rng(0), 95% percentile intervals).

Writes data/viz/T3.json and data/viz/T3.svg.
"""
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
JSON_OUT = ROOT / "data" / "viz" / "T3.json"
SVG_OUT = ROOT / "data" / "viz" / "T3.svg"

TOP_N = 20
N_BOOT = 1000
SEED = 0
CLEAN_PLAYER_COUNT = 10
T_CODE, CT_CODE = 2, 3

CT_COL = "#2563eb"  # not used for series (CT/T colours reserved for sides)
T_COL = "#d97706"
NEUTRAL = "#475569"
GRID = "#e2e8f0"
ACCENT = "#059669"
TEXT = "#334155"

KNIFE_GROUP = "knife (all variants)"
INCENDIARY_GROUP = "molotov + incendiary"
NON_FIREARM = {KNIFE_GROUP, INCENDIARY_GROUP, "flashbang", "smokegrenade", "hegrenade"}

DISPLAY = {
    "ak47": "AK-47",
    "m4a1_silencer": "M4A1-S",
    "m4a1": "M4A1",
    "glock": "Glock-18",
    "mac10": "MAC-10",
    "galilar": "Galil AR",
    "usp_silencer": "USP-S",
    "mp9": "MP9",
    "flashbang": "flashbang",
    "smokegrenade": "smoke",
    "hegrenade": "HE grenade",
    "deagle": "Desert Eagle",
    "mp7": "MP7",
    "famas": "FAMAS",
    "awp": "AWP",
    "tec9": "Tec-9",
    "negev": "Negev",
    "fiveseven": "Five-SeveN",
    "p90": "P90",
    KNIFE_GROUP: "knife (all)",
    INCENDIARY_GROUP: "molotov/incendiary",
}


def scan(channel, cols):
    pattern = (TOME / f"basic_{channel}.*" / "dataframe_*").as_posix()
    return pl.scan_parquet(pattern).select(cols).collect()


def fire_group(name):
    if name.startswith("knife") or name == "bayonet":
        return KNIFE_GROUP
    if name in ("molotov", "incgrenade"):
        return INCENDIARY_GROUP
    return name


def kill_group(name):
    if name.startswith("knife") or name == "bayonet":
        return KNIFE_GROUP
    if name in ("molotov", "incgrenade", "inferno"):
        # death events credit both molotov and incendiary fire damage to "inferno"
        return INCENDIARY_GROUP
    if name == "m4a1_silencer_off":
        return "m4a1_silencer"
    if name == "usp_silencer_off":
        return "usp_silencer"
    return name


def main():
    # ---- rounds -------------------------------------------------------------
    rounds = scan("round_end", ["match_key", "round", "tick", "player_count"])
    n_matches_tome = rounds["match_key"].n_unique()
    rounds_key = rounds.select(
        ["match_key", "round", pl.col("tick").alias("end_tick"), "player_count"]
    )
    assert rounds_key.select(["match_key", "round"]).is_unique().all()
    clean_keys = rounds_key.filter(pl.col("player_count") == CLEAN_PLAYER_COUNT)
    n_rounds_all = rounds.height
    n_clean_rounds = clean_keys.height
    # bootstrap universe: matches that contribute at least one clean 5v5 round
    all_matches = sorted(clean_keys["match_key"].unique().to_list())
    n_matches = len(all_matches)
    match_index = {mk: i for i, mk in enumerate(all_matches)}
    n_matches_clean = n_matches

    # ---- shots (weapon_fire) ------------------------------------------------
    fires = scan("weapon_fire", ["match_key", "round", "tick", "weapon_name"])
    n_fire_all = fires.height
    fj = fires.join(rounds_key, on=["match_key", "round"], how="left")
    n_fire_no_round = fj.filter(pl.col("player_count").is_null()).height
    fj_clean_round = fj.filter(pl.col("player_count") == CLEAN_PLAYER_COUNT)
    n_fire_clean_round = fj_clean_round.height
    fire_clean = fj_clean_round.filter(pl.col("tick") <= pl.col("end_tick"))
    n_fire_after_end = n_fire_clean_round - fire_clean.height
    fire_agg = fire_clean.group_by(["match_key", "weapon_name"]).len()
    shots_raw = defaultdict(int)
    shot_group_total = defaultdict(int)
    for mk, wn, n in fire_agg.iter_rows():
        g = fire_group(wn)
        shots_raw[(mk, g)] += n
        shot_group_total[g] += n
    n_shots_clean = fire_clean.height
    shot_tot_all = defaultdict(int)  # sensitivity: including post-round-end shots
    for wn, n in fj_clean_round.group_by("weapon_name").len().iter_rows():
        shot_tot_all[fire_group(wn)] += n

    # ---- kills (player_death) -----------------------------------------------
    deaths = scan(
        "player_death",
        ["match_key", "round", "tick", "player_id_fixed", "attacker_id_fixed",
         "weapon_name", "player_team_code", "attacker_team_code"],
    )
    n_death_all = deaths.height
    dj_all = deaths.join(rounds_key, on=["match_key", "round"], how="left")
    n_death_no_round = dj_all.filter(pl.col("player_count").is_null()).height
    dj_all = dj_all.filter(pl.col("player_count") == CLEAN_PLAYER_COUNT)
    n_death_clean_round = dj_all.height
    dj = dj_all.filter(pl.col("tick") <= pl.col("end_tick"))
    n_death_after_end = n_death_clean_round - dj.height
    n_deaths_clean = dj.height

    pt = pl.col("player_team_code")
    at = pl.col("attacker_team_code")
    both = (pt.is_in([T_CODE, CT_CODE]) & at.is_in([T_CODE, CT_CODE])).fill_null(False)
    self_kill = (pl.col("attacker_id_fixed") == pl.col("player_id_fixed")).fill_null(False)
    same_team = both & (pt == at).fill_null(False)
    enemy = both & (pt != at).fill_null(False) & (~self_kill)
    n_enemy_kills = dj.filter(enemy).height
    n_self_kills = dj.filter(self_kill).height
    n_team_kills = dj.filter(same_team & (~self_kill)).height
    n_no_attacker_team = dj.filter((~both) & (~self_kill)).height
    assert n_enemy_kills + n_self_kills + n_team_kills + n_no_attacker_team == dj.height
    kill_tot_all = defaultdict(int)  # sensitivity: including post-round-end kills
    for wn, n in dj_all.filter(enemy).group_by("weapon_name").len().iter_rows():
        kill_tot_all[kill_group(wn)] += n

    enemy_rows = dj.filter(enemy).group_by(["match_key", "weapon_name"]).len()
    kills_raw = defaultdict(int)
    kill_group_total = defaultdict(int)
    for mk, wn, n in enemy_rows.iter_rows():
        g = kill_group(wn)
        kills_raw[(mk, g)] += n
        kill_group_total[g] += n

    # kills whose weapon has no fire group (world, bomb, trigger, etc.) or is outside top-N
    top_groups = [g for g, _ in sorted(shot_group_total.items(), key=lambda kv: -kv[1])[:TOP_N]]
    top_set = set(top_groups)
    kills_outside_top = sum(v for g, v in kill_group_total.items() if g not in top_set)
    kills_in_top = sum(kill_group_total.get(g, 0) for g in top_groups)
    no_fire_kill_names = {}
    for mk, wn, n in enemy_rows.iter_rows():
        g = kill_group(wn)
        if g not in shot_group_total:
            no_fire_kill_names[wn] = no_fire_kill_names.get(wn, 0) + n

    # ---- per-match matrices -------------------------------------------------
    M = n_matches
    G = len(top_groups)
    gi = {g: j for j, g in enumerate(top_groups)}
    S = np.zeros((M, G), dtype=np.float64)
    K = np.zeros((M, G), dtype=np.float64)
    for (mk, g), v in shots_raw.items():
        if g in gi:
            S[match_index[mk], gi[g]] += v
    for (mk, g), v in kills_raw.items():
        if g in gi:
            K[match_index[mk], gi[g]] += v

    shots_obs = S.sum(axis=0)
    kills_obs = K.sum(axis=0)
    assert np.all(shots_obs > 0)

    # ---- match-cluster bootstrap --------------------------------------------
    rng = np.random.default_rng(SEED)
    idx_all = rng.integers(0, M, size=(N_BOOT, M))
    boot_k = np.empty((N_BOOT, G))
    boot_s = np.empty((N_BOOT, G))
    for b in range(N_BOOT):
        rows = idx_all[b]
        boot_k[b] = K[rows].sum(axis=0)
        boot_s[b] = S[rows].sum(axis=0)
    boot_r = np.where(boot_s > 0, boot_k / np.where(boot_s > 0, boot_s, 1.0), np.nan)
    k_lo, k_hi = np.percentile(boot_k, [2.5, 97.5], axis=0)
    r_lo, r_hi = np.nanpercentile(boot_r, [2.5, 97.5], axis=0)
    ratio_obs = kills_obs / shots_obs

    series = []
    for j, g in enumerate(top_groups):
        series.append({
            "group": g,
            "label": DISPLAY.get(g, g),
            "is_firearm": g not in NON_FIREARM,
            "shots": int(shots_obs[j]),
            "kills": int(kills_obs[j]),
            "kills_per_shot": float(ratio_obs[j]),
            "kills_ci95": [float(k_lo[j]), float(k_hi[j])],
            "kills_ci95_clipped_at_zero": bool(k_lo[j] <= 0),
            "kills_per_shot_ci95": [float(r_lo[j]), float(r_hi[j])],
            "sensitivity_incl_post_round_end_events": {
                "shots": int(shot_tot_all[g]),
                "kills": int(kill_tot_all[g]),
                "kills_per_shot": (kill_tot_all[g] / shot_tot_all[g]) if shot_tot_all[g] else None,
            },
        })

    firearm_idx = [j for j, g in enumerate(top_groups) if g not in NON_FIREARM]
    nonfirearm_idx = [j for j, g in enumerate(top_groups) if g in NON_FIREARM]
    fa_shots = int(shots_obs[firearm_idx].sum())
    fa_kills = int(kills_obs[firearm_idx].sum())
    nf_shots = int(shots_obs[nonfirearm_idx].sum())
    nf_kills = int(kills_obs[nonfirearm_idx].sum())

    key_numbers = {
        "n_rounds_clean_5v5": n_clean_rounds,
        "n_matches_in_bootstrap_universe": n_matches,
        "n_matches_with_clean_rounds": n_matches_clean,
        "n_matches_in_tome": n_matches_tome,
        "n_rounds_all_player_counts": n_rounds_all,
        "n_shots_clean_rounds_all_weapons": n_shots_clean,
        "n_deaths_clean_rounds": n_deaths_clean,
        "n_enemy_kills_clean_rounds": n_enemy_kills,
        "n_team_kills_excluded": n_team_kills,
        "n_self_kills_excluded": n_self_kills,
        "n_deaths_unknown_attacker_or_team_excluded": n_no_attacker_team,
        "n_kills_plotted_top20": kills_in_top,
        "n_kills_not_in_top20_or_no_shot_group": kills_outside_top,
        "kills_without_shot_group_by_name": no_fire_kill_names,
        "top20_firearm_shots": fa_shots,
        "top20_firearm_kills": fa_kills,
        "top20_firearm_kills_per_shot": fa_kills / fa_shots,
        "top20_nonfirearm_shots": nf_shots,
        "top20_nonfirearm_kills": nf_kills,
        "top20_nonfirearm_kills_per_shot": nf_kills / nf_shots,
        "fire_rows_all": n_fire_all,
        "fire_rows_without_round_end_match": n_fire_no_round,
        "fire_rows_after_round_end_tick_dropped": n_fire_after_end,
        "death_rows_all": n_death_all,
        "death_rows_without_round_end_match": n_death_no_round,
        "death_rows_after_round_end_tick_dropped": n_death_after_end,
    }

    payload = {
        "id": "T3",
        "title": "Kills versus shots by weapon",
        "source": "tomes/2026-10-01_to_08 basic_weapon_fire, basic_player_death, basic_round_end",
        "filter": "round_end.player_count == 10; events at or before round_end tick",
        "kill_definition": "player_death with attacker and victim on opposite sides (team codes 2 or 3); suicides, team kills and world deaths excluded",
        "grouping": {
            "knife": "all basic knife* names and bayonet grouped",
            "molotov_incendiary": "fire: molotov + incgrenade; death: molotov + incgrenade + inferno (inferno is the shared fire-damage credit)",
            "silencer_off": "death names m4a1_silencer_off and usp_silencer_off merged into m4a1_silencer and usp_silencer",
        },
        "bootstrap": {
            "unit": "match (cluster); universe = matches with at least one clean 5v5 round",
            "n_clusters": n_matches,
            "n_resamples": N_BOOT,
            "seed": SEED,
            "method": "numpy default_rng(0).integers, resample matches with replacement, 95% percentile interval",
            "ci_on": "kills count (vertical bars) and kills per shot (in JSON)",
        },
        "series": series,
        "key_numbers": key_numbers,
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    svg = build_svg(series, n_clean_rounds, n_matches)
    SVG_OUT.write_text(svg, encoding="utf-8")
    print(json.dumps({"n_rounds": n_clean_rounds, "n_matches": n_matches,
                      "series": [(s["label"], s["shots"], s["kills"],
                                  round(s["kills_per_shot"], 5)) for s in series]}, indent=1))
    print(json.dumps(key_numbers, indent=1))


# ---- SVG --------------------------------------------------------------------
W, H = 800, 450
L, R, T, B = 92.0, 780.0, 22.0, 384.0
X_MIN, X_MAX = 3e4, 3e6
Y_MIN, Y_MAX = 0.1, 1e5
CHAR_W = 6.3  # rough advance width at 11 px sans-serif
LABEL_FS = 11


def px(v):
    return L + (math.log10(v) - math.log10(X_MIN)) / (math.log10(X_MAX) - math.log10(X_MIN)) * (R - L)


def py(v):
    v = min(max(v, Y_MIN), Y_MAX)
    return B - (math.log10(v) - math.log10(Y_MIN)) / (math.log10(Y_MAX) - math.log10(Y_MIN)) * (B - T)


def fmt_int(v):
    return f"{int(round(v)):,}"


def text_box(anchor, tx, ty, text):
    w = CHAR_W * len(text)
    if anchor == "start":
        x0, x1 = tx, tx + w
    elif anchor == "end":
        x0, x1 = tx - w, tx
    else:
        x0, x1 = tx - w / 2, tx + w / 2
    return (x0, ty - 9, x1, ty + 3)


def boxes_overlap(a, b, pad=1.0):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def build_svg(series, n_rounds, n_matches):
    els = []
    add = els.append
    add(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">')
    add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # grid and ticks
    x_ticks = [3e4, 1e5, 3e5, 1e6, 3e6]
    x_labels = ["30,000", "100,000", "300,000", "1,000,000", "3,000,000"]
    y_ticks = [0.1, 1, 10, 100, 1e3, 1e4, 1e5]
    y_labels = ["0.1", "1", "10", "100", "1,000", "10,000", "100,000"]
    for xv in x_ticks:
        x = px(xv)
        add(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{B}" stroke="{GRID}" stroke-width="1"/>')
    for yv in y_ticks:
        y = py(yv)
        add(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')

    # reference lines: 1 kill per shot (dashed), 0.1 and 0.01 kills per shot (dotted)
    def seg(x0, y0, x1, y1, dash, width=1.4):
        add(f'<line x1="{px(x0):.1f}" y1="{py(y0):.1f}" x2="{px(x1):.1f}" y2="{py(y1):.1f}" '
            f'stroke="{NEUTRAL}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    seg(3e4, 3e4, 1e5, 1e5, "6,4")
    seg(3e4, 3e3, 1e6, 1e5, "2,3", 1.0)
    seg(3e4, 300, 3e6, 3e4, "2,3", 1.0)
    diag_label = ("1 kill per shot", px(1e5) + 6, T + 14)
    add(f'<text x="{diag_label[1]:.1f}" y="{diag_label[2]:.1f}" font-size="{LABEL_FS}" fill="{TEXT}">'
        f'{diag_label[0]}</text>')

    # axes
    add(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{NEUTRAL}" stroke-width="1.2"/>')
    add(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="{NEUTRAL}" stroke-width="1.2"/>')
    for xv, lab in zip(x_ticks, x_labels):
        anchor = "end" if xv == X_MAX else "middle"
        add(f'<text x="{px(xv):.1f}" y="{B + 17:.1f}" font-size="12" text-anchor="{anchor}" fill="{TEXT}">{lab}</text>')
    for yv, lab in zip(y_ticks, y_labels):
        add(f'<text x="{L - 8:.1f}" y="{py(yv) + 4:.1f}" font-size="12" text-anchor="end" fill="{TEXT}">{lab}</text>')
    add(f'<text x="{(L + R) / 2:.1f}" y="{H - 10}" font-size="13" text-anchor="middle" fill="{TEXT}">'
        f'Shots fired (weapon_fire events, log scale)</text>')
    ycx = (T + B) / 2
    add(f'<text transform="translate(20,{ycx:.1f}) rotate(-90)" font-size="13" text-anchor="middle" '
        f'fill="{TEXT}">Kills (enemy kills, log scale)</text>')

    # legend (bottom-right inside plot) and sample-size label (bottom-left inside plot)
    lg_x0, lg_y0, lg_x1, lg_y1 = 610.0, 282.0, 776.0, 376.0
    legend_box = (lg_x0, lg_y0, lg_x1, lg_y1)
    add(f'<rect x="{lg_x0}" y="{lg_y0}" width="{lg_x1 - lg_x0}" height="{lg_y1 - lg_y0}" '
        f'fill="#ffffff" stroke="{GRID}" stroke-width="1"/>')
    rows = [
        ("circle", NEUTRAL, "firearm"),
        ("diamond", ACCENT, "knife or grenade"),
        ("bar", NEUTRAL, "95% CI, match bootstrap"),
        ("dash", NEUTRAL, "1 kill per shot"),
        ("dot", NEUTRAL, "0.1 and 0.01 per shot"),
    ]
    for i, (kind, col, text) in enumerate(rows):
        cy = lg_y0 + 13 + i * 18.5
        cx = lg_x0 + 12
        if kind == "circle":
            add(f'<circle cx="{cx}" cy="{cy:.1f}" r="4.5" fill="{col}"/>')
        elif kind == "diamond":
            add(f'<path d="M{cx} {cy - 5:.1f} L{cx + 5} {cy:.1f} L{cx} {cy + 5:.1f} L{cx - 5} {cy:.1f} Z" fill="{col}"/>')
        elif kind == "bar":
            add(f'<line x1="{cx}" y1="{cy - 5:.1f}" x2="{cx}" y2="{cy + 5:.1f}" stroke="{col}" stroke-width="1.4"/>')
            add(f'<line x1="{cx - 3}" y1="{cy - 5:.1f}" x2="{cx + 3}" y2="{cy - 5:.1f}" stroke="{col}" stroke-width="1.4"/>')
            add(f'<line x1="{cx - 3}" y1="{cy + 5:.1f}" x2="{cx + 3}" y2="{cy + 5:.1f}" stroke="{col}" stroke-width="1.4"/>')
        elif kind == "dash":
            add(f'<line x1="{cx - 7}" y1="{cy:.1f}" x2="{cx + 7}" y2="{cy:.1f}" stroke="{col}" stroke-width="1.4" stroke-dasharray="6,4"/>')
        else:
            add(f'<line x1="{cx - 7}" y1="{cy:.1f}" x2="{cx + 7}" y2="{cy:.1f}" stroke="{col}" stroke-width="1" stroke-dasharray="2,3"/>')
        add(f'<text x="{cx + 14}" y="{cy + 4:.1f}" font-size="11" fill="{TEXT}">{text}</text>')

    n_text = f"n = {fmt_int(n_rounds)} clean 5v5 rounds, {fmt_int(n_matches)} matches"
    n_box = (L, 4.0, L + 7.0 * len(n_text), 17.0)  # top margin, clear of error bars
    add(f'<text x="{L:.1f}" y="14.0" font-size="12" fill="{TEXT}">{html.escape(n_text)}</text>')

    # points, error bars, labels
    pts = []
    for s in series:
        x = px(s["shots"])
        y = py(s["kills"])
        lo = max(s["kills_ci95"][0], Y_MIN)
        hi = max(s["kills_ci95"][1], Y_MIN)
        col = NEUTRAL if s["is_firearm"] else ACCENT
        pts.append({"s": s, "x": x, "y": y, "lo": py(lo), "hi": py(hi), "col": col,
                    "firearm": s["is_firearm"]})

    fixed = [("diag", diag_label[0], diag_label[1], diag_label[2], "start")]
    fixed_boxes = [text_box("start", diag_label[1], diag_label[2], diag_label[0]),
                   legend_box, n_box]
    marker_boxes = {i: (p["x"] - 6, p["y"] - 6, p["x"] + 6, p["y"] + 6) for i, p in enumerate(pts)}

    for p in pts:
        y_top, y_bot = min(p["lo"], p["hi"]), max(p["lo"], p["hi"])
        add(f'<line x1="{p["x"]:.1f}" y1="{y_top:.1f}" x2="{p["x"]:.1f}" y2="{y_bot:.1f}" '
            f'stroke="{p["col"]}" stroke-width="1.3" opacity="0.7"/>')
        for yy in (y_top, y_bot):
            add(f'<line x1="{p["x"] - 3:.1f}" y1="{yy:.1f}" x2="{p["x"] + 3:.1f}" y2="{yy:.1f}" '
                f'stroke="{p["col"]}" stroke-width="1.3" opacity="0.7"/>')

    for p in pts:
        if p["firearm"]:
            add(f'<circle cx="{p["x"]:.1f}" cy="{p["y"]:.1f}" r="4.5" fill="{p["col"]}" stroke="#ffffff" stroke-width="0.8"/>')
        else:
            x, y = p["x"], p["y"]
            add(f'<path d="M{x:.1f} {y - 6:.1f} L{x + 6:.1f} {y:.1f} L{x:.1f} {y + 6:.1f} L{x - 6:.1f} {y:.1f} Z" '
                f'fill="{p["col"]}" stroke="#ffffff" stroke-width="0.8"/>')

    candidates = [(8, 4, "start"), (-8, 4, "end"), (0, -10, "middle"), (0, 18, "middle"),
                  (8, -6, "start"), (-8, -6, "end"), (8, 15, "start"), (-8, 15, "end"),
                  (0, -22, "middle"), (0, 30, "middle"), (14, 4, "start"), (-14, 4, "end")]
    placed = list(fixed_boxes)
    label_out = []
    forced = []
    order = sorted(range(len(pts)), key=lambda i: -pts[i]["s"]["kills"])
    for i in order:
        p = pts[i]
        chosen = None
        text = p["s"]["label"]
        for dx, dy, anchor in candidates:
            tx, ty = p["x"] + dx, p["y"] + dy
            box = text_box(anchor, tx, ty, text)
            if box[0] < L + 2 or box[2] > R - 2 or box[1] < T + 2 or box[3] > B - 2:
                continue
            if any(boxes_overlap(box, q) for q in placed):
                continue
            if any(boxes_overlap(box, marker_boxes[j]) for j in marker_boxes if j != i):
                continue
            chosen = (tx, ty, anchor, box)
            break
        if chosen is None:
            dx, dy, anchor = candidates[0]
            tx, ty = p["x"] + dx, p["y"] + dy
            chosen = (tx, ty, anchor, text_box(anchor, tx, ty, text))
            forced.append(text)
        placed.append(chosen[3])
        label_out.append((chosen[0], chosen[1], chosen[2], text))

    for tx, ty, anchor, text in label_out:
        add(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="{LABEL_FS}" text-anchor="{anchor}" fill="{TEXT}">'
            f'{html.escape(text)}</text>')

    add("</svg>")
    if forced:
        print("labels placed on fallback position:", forced)
    return "\n".join(els) + "\n"


if __name__ == "__main__":
    main()
