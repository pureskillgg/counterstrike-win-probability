"""T2: kills by weapon, top 15 weapons, stacked by attacker side (CT / T).

Reads (read-only) from the 2026 tomes:
  tomes/2026-10-01_to_08/tome/csds/basic_round_end*/dataframe_*
  tomes/2026-10-01_to_08/tome/csds/basic_player_death*/dataframe_*
Writes:
  data/viz/T2.json
  data/viz/T2.svg

Run with:
  "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 "G:/pureskill/csds-cs2/scripts/viz/T2.py"

Kill rules (per player_death row, restricted to clean 5v5 rounds where round_end.player_count == 10):
  - world death: weapon in {world, worldent, planted_c4, trigger_hurt}, or attacker_id null / 65535 -> excluded
  - suicide: attacker_id == player_id (raw slot ids) -> excluded
  - team kill: attacker side == victim side -> excluded
  - attacker side: attacker_team_code (2 = T, 3 = CT); if null, the side of the same slot in the same
    round from other rows; if still unknown, the opposite of the victim side (assumption, flagged)
  - rows with no attacker side and no victim side -> excluded and counted

Interval: match-cluster bootstrap (resample matches with replacement, 1000 resamples, seed 0),
percentile 95% interval with linear interpolation.
"""
import glob
import json
import math
import os
import random
from collections import Counter
from xml.sax.saxutils import escape

import polars as pl

REPO = "G:/pureskill/csds-cs2"
TOME_ROOT = os.path.join(REPO, "tomes", "2026-10-01_to_08", "tome", "csds")
OUT_JSON = os.path.join(REPO, "data", "viz", "T2.json")
OUT_SVG = os.path.join(REPO, "data", "viz", "T2.svg")

TEAM_T = 2
TEAM_CT = 3
CLEAN_PLAYER_COUNT = 10
WORLD_WEAPONS = {"world", "worldent", "planted_c4", "trigger_hurt"}
WORLD_SENTINEL = 65535
TOP_N = 15
N_BOOT = 1000
SEED = 0

DISPLAY = {
    "ak47": "AK-47",
    "m4a1": "M4A4",
    "m4a1_silencer": "M4A1-S",
    "awp": "AWP",
    "usp_silencer": "USP-S",
    "glock": "Glock-18",
    "deagle": "Desert Eagle",
    "galilar": "Galil AR",
    "mac10": "MAC-10",
    "mp9": "MP9",
    "famas": "FAMAS",
    "mp7": "MP7",
    "tec9": "Tec-9",
    "ssg08": "SSG 08",
    "fiveseven": "Five-SeveN",
    "hegrenade": "HE grenade",
    "p250": "P250",
    "xm1014": "XM1014",
    "hkp2000": "P2000",
    "elite": "Dual Berettas",
    "inferno": "Molotov/incendiary",
    "incgrenade": "Incendiary grenade",
    "molotov": "Molotov",
    "flashbang": "Flashbang",
    "smokegrenade": "Smoke grenade",
    "taser": "Zeus (taser)",
    "knife": "Knife (all variants)",
}


def tome_files(prefix):
    pattern = os.path.join(TOME_ROOT, prefix + "*", "dataframe_*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("no dataframe files for channel " + prefix)
    return files


def clean_weapon(raw):
    """Name cleaning: lower-case, fold *_off silencer variants into the base weapon,
    fold all knife variants (knife, knife_t, knife_*, bayonet) into 'knife'."""
    name = raw.strip().lower()
    if name.endswith("_off"):
        name = name[: -len("_off")]
    if name == "knife_t" or name == "bayonet" or name.startswith("knife"):
        return "knife"
    return name


def pct(values, p):
    s = sorted(values)
    n = len(s)
    pos = p * (n - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def fmt_int(n):
    return f"{n:,}"


def fmt_k(v):
    if v >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


def build_svg(rows, n_text):
    W, H = 800, 450
    x0, x1 = 170.0, 600.0
    top_y = 58.0
    row_h = 20.0
    bar_h = 13.0
    axis_y = top_y + len(rows) * row_h + 6.0
    vmax = max(r["count_hi"] for r in rows)
    step = 20000
    for s in (1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000):
        if vmax / s <= 8:
            step = s
            break
    axis_max = math.ceil(vmax * 1.02 / step) * step

    def X(v):
        return x0 + (x1 - x0) * v / axis_max

    out = []
    a = out.append
    a(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">')
    a(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # legend
    a('<rect x="170" y="14" width="10" height="10" fill="#2563eb"/>')
    a('<text x="185" y="23" font-size="12" fill="#475569">CT attacker</text>')
    a('<rect x="275" y="14" width="10" height="10" fill="#d97706"/>')
    a('<text x="290" y="23" font-size="12" fill="#475569">T attacker</text>')
    a('<line x1="375" y1="19" x2="395" y2="19" stroke="#475569" stroke-width="1.5"/>')
    a('<line x1="375" y1="15" x2="375" y2="23" stroke="#475569" stroke-width="1.5"/>')
    a('<line x1="395" y1="15" x2="395" y2="23" stroke="#475569" stroke-width="1.5"/>')
    a('<text x="402" y="23" font-size="12" fill="#475569">whisker: 95% CI of kill count (match bootstrap)</text>')

    # column headers
    a('<text x="680" y="46" text-anchor="end" font-size="12" fill="#475569">Share of kills</text>')
    a('<text x="790" y="46" text-anchor="end" font-size="12" fill="#475569">Share 95% CI</text>')

    # gridlines and x ticks
    t = 0
    while t <= axis_max + 1e-9:
        xx = X(t)
        if t > 0:
            a(f'<line x1="{xx:.1f}" y1="{top_y - 4:.1f}" x2="{xx:.1f}" y2="{axis_y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        a(f'<text x="{xx:.1f}" y="{axis_y + 16:.1f}" text-anchor="middle" font-size="12" fill="#475569">{fmt_k(t)}</text>')
        t += step

    # axis lines
    a(f'<line x1="{x0:.1f}" y1="{top_y - 4:.1f}" x2="{x0:.1f}" y2="{axis_y:.1f}" stroke="#475569" stroke-width="1"/>')
    a(f'<line x1="{x0:.1f}" y1="{axis_y:.1f}" x2="{x1:.1f}" y2="{axis_y:.1f}" stroke="#475569" stroke-width="1"/>')

    # bars
    for i, r in enumerate(rows):
        yc = top_y + 10.0 + i * row_h
        yb = yc - bar_h / 2
        xct = X(r["ct"])
        xtot = X(r["total"])
        a(f'<rect x="{X(0):.1f}" y="{yb:.1f}" width="{xct - X(0):.1f}" height="{bar_h:.1f}" fill="#2563eb"/>')
        a(f'<rect x="{xct:.1f}" y="{yb:.1f}" width="{xtot - xct:.1f}" height="{bar_h:.1f}" fill="#d97706"/>')
        xl = X(r["count_lo"])
        xh = X(r["count_hi"])
        a(f'<line x1="{xl:.1f}" y1="{yc:.1f}" x2="{xh:.1f}" y2="{yc:.1f}" stroke="#475569" stroke-width="1.5"/>')
        for xx in (xl, xh):
            a(f'<line x1="{xx:.1f}" y1="{yc - 4:.1f}" x2="{xx:.1f}" y2="{yc + 4:.1f}" stroke="#475569" stroke-width="1.5"/>')
        a(f'<text x="160" y="{yc + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">{escape(r["label"])}</text>')
        a(f'<text x="680" y="{yc + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">{r["share"] * 100:.1f}%</text>')
        a(f'<text x="790" y="{yc + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">[{r["share_lo"] * 100:.1f}, {r["share_hi"] * 100:.1f}]</text>')

    # axis titles
    a(f'<text x="{(x0 + x1) / 2:.1f}" y="{axis_y + 36:.1f}" text-anchor="middle" font-size="13" fill="#475569">Kills (count), stacked by attacker side</text>')
    a(f'<text x="18" y="{(top_y + axis_y) / 2:.1f}" text-anchor="middle" font-size="13" fill="#475569" transform="rotate(-90 18 {(top_y + axis_y) / 2:.1f})">Weapon</text>')

    # sample size
    a(f'<text x="20" y="436" font-size="12" fill="#475569">{escape(n_text)}</text>')
    a("</svg>")
    return "\n".join(out) + "\n"


def main():
    # 1. clean 5v5 rounds from round_end
    re_files = tome_files("basic_round_end")
    rounds = (
        pl.scan_parquet(re_files)
        .select(["match_key", "round", "tick", "player_count"])
        .collect()
    )
    round_rows = rounds.height
    dup_round_rows = round_rows - rounds.unique(["match_key", "round"]).height
    clean_rounds = rounds.filter(pl.col("player_count") == CLEAN_PLAYER_COUNT)
    clean_round_n = clean_rounds.height
    clusters = sorted(clean_rounds["match_key"].unique().to_list())
    M = len(clusters)
    cluster_index = {mk: i for i, mk in enumerate(clusters)}
    rnd = rounds.select(["match_key", "round", pl.col("tick").alias("round_end_tick"), "player_count"])

    # 2. player_death rows (only the columns needed)
    pd_files = tome_files("basic_player_death")
    deaths = (
        pl.scan_parquet(pd_files)
        .select([
            "match_key", "round", "tick", "player_id", "attacker_id", "weapon_name",
            "player_team_code", "attacker_team_code",
        ])
        .collect()
    )
    death_rows_total = deaths.height
    joined = deaths.join(rnd, on=["match_key", "round"], how="left")
    unmatched_round = joined.filter(pl.col("player_count").is_null()).height
    after_round_end = joined.filter(
        pl.col("player_count").is_not_null() & (pl.col("tick") > pl.col("round_end_tick"))
    ).height
    clean = joined.filter(pl.col("player_count") == CLEAN_PLAYER_COUNT)
    clean_rows = clean.select([
        "match_key", "round", "tick", "round_end_tick", "player_id", "attacker_id", "weapon_name",
        "player_team_code", "attacker_team_code",
    ]).to_dicts()
    death_rows_clean = len(clean_rows)

    # 3. per (match, round, raw slot) -> team, from victim and attacker rows
    team_at = {}
    slot_team_conflicts = 0
    for r in clean_rows:
        key_round = (r["match_key"], r["round"])
        for slot, team in (
            (r["player_id"], r["player_team_code"]),
            (r["attacker_id"], r["attacker_team_code"]),
        ):
            if slot is None or slot == WORLD_SENTINEL or team not in (TEAM_T, TEAM_CT):
                continue
            key = (key_round, slot)
            prev = team_at.get(key)
            if prev is None:
                team_at[key] = team
            elif prev != team:
                slot_team_conflicts += 1

    # 4. classify each clean death row
    stats = Counter()
    kills = []  # (match_key, weapon_clean, attacker_side)
    side_source = Counter()
    for r in clean_rows:
        mk, rd = r["match_key"], r["round"]
        if r["weapon_name"] in WORLD_WEAPONS or r["attacker_id"] is None or r["attacker_id"] == WORLD_SENTINEL:
            stats["excl_world"] += 1
            continue
        if r["attacker_id"] == r["player_id"]:
            stats["excl_suicide"] += 1
            continue
        key_round = (mk, rd)
        a_team = r["attacker_team_code"]
        v_team = r["player_team_code"]
        if a_team in (TEAM_T, TEAM_CT):
            side_source["attacker_row"] += 1
        else:
            a_team = team_at.get((key_round, r["attacker_id"]))
            if a_team is not None:
                side_source["round_slot_map"] += 1
        if v_team not in (TEAM_T, TEAM_CT):
            v_team = team_at.get((key_round, r["player_id"]))
        if a_team is None:
            if v_team is None:
                stats["excl_unattributed"] += 1
                continue
            a_team = TEAM_T if v_team == TEAM_CT else TEAM_CT
            side_source["inferred_opposite_of_victim"] += 1
        if v_team is not None and a_team == v_team:
            stats["excl_team_kill"] += 1
            continue
        if v_team is None:
            stats["team_check_unavailable"] += 1
        # strictly after round_end: the final kill of an elimination round shares the round_end tick
        late = r["tick"] > r["round_end_tick"]
        if late:
            stats["kept_after_round_end_tick"] += 1
        kills.append((mk, clean_weapon(r["weapon_name"]), a_team, late))
        stats["kept"] += 1

    K = len(kills)

    # 5. weapon totals and CT/T split
    ct_count = Counter()
    t_count = Counter()
    for mk, w, side, _late in kills:
        if side == TEAM_CT:
            ct_count[w] += 1
        else:
            t_count[w] += 1
    totals = Counter()
    for w in set(list(ct_count.keys()) + list(t_count.keys())):
        totals[w] = ct_count[w] + t_count[w]
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [w for w, _ in ranked[:TOP_N]]
    top_set = set(top)

    # 6. per-match tallies for the match-cluster bootstrap
    mat_total = [0] * M
    mat_w = {w: [0] * M for w in top}
    for mk, w, side, _late in kills:
        i = cluster_index.get(mk)
        if i is None:
            continue
        mat_total[i] += 1
        if w in top_set:
            mat_w[w][i] += 1

    # 7. bootstrap: resample matches with replacement, 1000 resamples, seed 0
    rng = random.Random(SEED)
    boot_total = []
    boot_w = {w: [] for w in top}
    for _ in range(N_BOOT):
        idx = [rng.randrange(M) for _ in range(M)]
        boot_total.append(sum(mat_total[i] for i in idx))
        for w in top:
            arr = mat_w[w]
            boot_w[w].append(sum(arr[i] for i in idx))

    # 8. assemble results
    top15 = []
    for rank, w in enumerate(top, start=1):
        total = totals[w]
        share = total / K
        shares_b = [boot_w[w][b] / boot_total[b] for b in range(N_BOOT) if boot_total[b] > 0]
        s_lo = pct(shares_b, 0.025)
        s_hi = pct(shares_b, 0.975)
        c_lo = pct(boot_w[w], 0.025)
        c_hi = pct(boot_w[w], 0.975)
        top15.append({
            "rank": rank,
            "weapon_clean": w,
            "label": DISPLAY.get(w, w),
            "ct": ct_count[w],
            "t": t_count[w],
            "total": total,
            "ct_share_of_weapon": ct_count[w] / total,
            "share": share,
            "share_ci95": [s_lo, s_hi],
            "count_ci95": [c_lo, c_hi],
            "share_point_inside_ci": bool(s_lo <= share <= s_hi),
        })

    all_weapons = [
        {"weapon_clean": w, "ct": ct_count[w], "t": t_count[w], "total": tot, "share": tot / K}
        for w, tot in ranked
    ]

    # sensitivity: drop kills strictly after the round_end tick (post-round kills)
    sens_total = Counter()
    for mk, w, side, late in kills:
        if not late:
            sens_total[w] += 1
    K_sens = sum(sens_total.values())
    sensitivity = {
        "rule": "drop kills with tick > round_end tick (post-round kills, within clean 5v5 rounds)",
        "kills_kept": K_sens,
        "kills_dropped": K - K_sens,
        "top15": [
            {
                "weapon_clean": x["weapon_clean"],
                "label": x["label"],
                "total_primary": x["total"],
                "total_sensitivity": sens_total[x["weapon_clean"]],
                "share_primary": x["share"],
                "share_sensitivity": sens_total[x["weapon_clean"]] / K_sens,
            }
            for x in top15
        ],
    }

    # sanity checks
    checks = {
        "no_negative_counts": all(x["ct"] >= 0 and x["t"] >= 0 and x["total"] >= 0 for x in top15),
        "intervals_ordered": all(x["share_ci95"][0] <= x["share_ci95"][1] and x["count_ci95"][0] <= x["count_ci95"][1] for x in top15),
        "ct_plus_t_equals_total": all(x["ct"] + x["t"] == x["total"] for x in top15),
        "all_weapons_sum_to_kept": sum(x["total"] for x in all_weapons) == K,
        "points_inside_share_ci": all(x["share_point_inside_ci"] for x in top15),
        "top15_shares_sum": round(sum(x["share"] for x in top15), 6),
    }

    waterfall = {
        "round_end_rows": round_rows,
        "round_end_duplicate_keys": dup_round_rows,
        "player_death_rows_total": death_rows_total,
        "player_death_rows_no_round_end_match": unmatched_round,
        "player_death_rows_after_round_end_tick_in_matched_rounds": after_round_end,
        "player_death_rows_in_clean_5v5_rounds": death_rows_clean,
        "excluded_world": stats["excl_world"],
        "excluded_suicide": stats["excl_suicide"],
        "excluded_team_kill": stats["excl_team_kill"],
        "excluded_unattributed_side": stats["excl_unattributed"],
        "team_check_unavailable_kept": stats["team_check_unavailable"],
        "kills_kept": K,
        "kills_kept_strictly_after_round_end_tick": stats["kept_after_round_end_tick"],
    }

    result = {
        "id": "T2",
        "chart": "Kills by weapon, top 15 weapons, stacked by attacker side",
        "source": {
            "tomes": "tomes/2026-10-01_to_08/tome/csds/basic_player_death* and basic_round_end*",
            "rows_used": "player_death rows joined to round_end on (match_key, round); clean = round_end.player_count == 10",
            "weapon_column": "basic_player_death.weapon_name",
            "team_codes": "2 = T, 3 = CT",
            "slot_columns": "player_id / attacker_id (raw slot ids; consistent per round with the _fixed columns)",
        },
        "name_cleaning": {
            "lower_case_and_strip": True,
            "fold_off_variants": {"m4a1_silencer_off": "m4a1_silencer", "usp_silencer_off": "usp_silencer"},
            "fold_knife_variants_to": "knife (knife, knife_t, knife_*, bayonet)",
            "excluded_world_names": sorted(WORLD_WEAPONS),
            "display_labels": DISPLAY,
        },
        "n": {
            "matches": M,
            "clean_5v5_rounds": clean_round_n,
            "kills_kept": K,
        },
        "waterfall": waterfall,
        "side_source_counts": dict(side_source),
        "slot_team_conflicts_per_round": slot_team_conflicts,
        "bootstrap": {
            "unit": "match (cluster)",
            "clusters": M,
            "resamples": N_BOOT,
            "seed": SEED,
            "method": "resample matches with replacement; percentile 2.5 and 97.5 with linear interpolation",
            "statistics": "share = weapon kills / all kept kills (all weapons); count = weapon kills",
        },
        "top15": top15,
        "all_weapons": all_weapons,
        "sensitivity": sensitivity,
        "checks": checks,
        "assumptions": [
            "Kill side for rows with null attacker_team_code is taken from the same round slot, else opposite the victim side (flagged in side_source_counts).",
            "Team-kill test is skipped when the victim side is unknown (team_check_unavailable_kept).",
            "Kills are counted per player_death row; no assist credit.",
            "Deaths with tick after round_end tick are counted (see waterfall) because the spec does not exclude them for player_death.",
            "Clean 5v5 rounds include overtime rounds (round numbers above 24) where round_end.player_count == 10.",
        ],
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    n_text = (
        f"n = {fmt_int(K)} kills; {fmt_int(clean_round_n)} clean 5v5 rounds; {fmt_int(M)} matches. "
        f"Bootstrap: 1,000 match resamples, seed 0."
    )
    svg_rows = []
    for x in top15:
        svg_rows.append({
            "label": x["label"],
            "ct": x["ct"],
            "t": x["t"],
            "total": x["total"],
            "share": x["share"],
            "share_lo": x["share_ci95"][0],
            "share_hi": x["share_ci95"][1],
            "count_lo": x["count_ci95"][0],
            "count_hi": x["count_ci95"][1],
        })
    svg = build_svg(svg_rows, n_text)
    with open(OUT_SVG, "w", encoding="utf-8") as fh:
        fh.write(svg)

    print("clean rounds:", clean_round_n, "matches:", M, "kills kept:", K)
    print("waterfall:", json.dumps(waterfall))
    print("side source:", dict(side_source), "slot conflicts:", slot_team_conflicts)
    for x in top15:
        print(
            f"{x['rank']:>2} {x['label']:<20} total={x['total']:>6} CT={x['ct']:>6} T={x['t']:>6} "
            f"share={x['share'] * 100:6.2f}% CI=[{x['share_ci95'][0] * 100:6.2f},{x['share_ci95'][1] * 100:6.2f}] "
            f"count CI=[{x['count_ci95'][0]:.0f},{x['count_ci95'][1]:.0f}]"
        )
    print("checks:", json.dumps(checks))
    print("wrote", OUT_JSON, "and", OUT_SVG)


if __name__ == "__main__":
    main()
