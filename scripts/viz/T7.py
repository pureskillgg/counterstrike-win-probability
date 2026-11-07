"""T7 - Kill density by map (victim positions), 2026 tome, 2026-10-01 to 08.

Maps: the two maps with the most clean 5v5 rounds (round_end.player_count == 10).
Map name: header.map_name, joined on header.key == round_end.match_key.
Kill: a player_death row in a clean round where attacker_id is present, is not the
victim (attacker_id != player_id), and both team codes are present and differ
(2 = T, 3 = CT). Victim position = player_x_pos / player_y_pos (the player_* fields
are the victim). Enemy kills with a missing victim position are dropped and counted.
Grid: 40 x 40 square bins. Each map's own window is a square centred on its kill positions
(side = 1.02 x the larger of its x and y ranges); both panels use the larger of the two
window sides, so the cell size is identical. Colour = kill count per cell, linear scale,
one maximum shared by both panels.
Intervals: match-cluster bootstrap. Matches on each map are resampled with replacement,
1,000 resamples, random.Random(0) (a fresh generator per map). 95% percentile intervals
(linear interpolation) for kills per round and for the share of kills in the top 10% of
cells (the cell set is fixed from the point estimate).
No radar image is available, so the panels are in raw world coordinates (game units).

Writes data/viz/T7.json and data/viz/T7.svg under the repo root.
Run: G:/pureskill/csds-cs2/.pixi/envs/default/python.exe -I -X utf8 scripts/viz/T7.py
"""
import json
import math
import random

import polars as pl

ROOT = 'G:/pureskill/csds-cs2'
TOME = ROOT + '/tomes/2026-10-01_to_08/tome/csds/'
HEADER = TOME + 'header/dataframe_00000'
ROUND_END = TOME + 'basic_round_end.2026-09-18,2026-10-08/dataframe_00000'
PLAYER_DEATH = TOME + 'basic_player_death.2026-09-18,2026-10-08/dataframe_00000'
ROUND_STATE = TOME + 'basic_round_state.2026-09-18,2026-10-08/dataframe_00000'
OUT_JSON = ROOT + '/data/viz/T7.json'
OUT_SVG = ROOT + '/data/viz/T7.svg'

N_BOOT = 1000
SEED = 0
CLEAN = 10
G = 40
TOP_FRAC = 0.10
N_MAPS = 2
REG_MAX_ROUND = 24

NEUTRAL = '#475569'
ACCENT = '#059669'
# Single-hue sequential ramp (ColorBrewer Blues).
STOPS = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
         '#4292c6', '#2171b5', '#08519c', '#08306b']

# SVG geometry (viewBox 0 0 800 450)
BOX_S = 280
BOX_Y = 84
PANEL_X = [84, 432]
CB_X = 740
CB_W = 12


def pct(vals, q):
    """Linear-interpolation percentile, q in [0, 1]."""
    s = sorted(vals)
    n = len(s)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def load():
    hdr = (pl.scan_parquet(HEADER).select(['key', 'map_name', 'tick_rate'])
           .rename({'key': 'match_key'}).collect())
    assert hdr['match_key'].n_unique() == hdr.height
    assert set(hdr['tick_rate'].unique().to_list()) == {64}
    re_ = (pl.scan_parquet(ROUND_END)
           .select(['match_key', 'round', 'player_count', 'winner_team_code']).collect())
    assert re_.height - re_.unique(subset=['match_key', 'round']).height == 0
    clean = (re_.filter(pl.col('player_count') == CLEAN)
             .join(hdr, on='match_key', how='left'))
    assert clean['map_name'].null_count() == 0
    pdth = (pl.scan_parquet(PLAYER_DEATH)
            .select(['match_key', 'round', 'tick', 'player_id', 'attacker_id',
                     'player_team_code', 'attacker_team_code',
                     'player_x_pos', 'player_y_pos']).collect())
    return hdr, re_, clean, pdth


def map_rounds(clean):
    return (clean.group_by('map_name')
            .agg(pl.len().alias('rounds'), pl.col('match_key').n_unique().alias('matches'))
            .sort(['rounds', 'map_name'], descending=[True, False]))


def classify(d):
    return d.with_columns(
        pl.when(pl.col('attacker_id').is_null()).then(pl.lit('no_attacker'))
        .when(pl.col('attacker_id') == pl.col('player_id')).then(pl.lit('self'))
        .when(pl.col('attacker_team_code').is_null() | pl.col('player_team_code').is_null())
        .then(pl.lit('team_unknown'))
        .when(pl.col('attacker_team_code') == pl.col('player_team_code'))
        .then(pl.lit('team_kill'))
        .otherwise(pl.lit('enemy')).alias('cls'))


def geometry(xs, ys, side=None):
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    own = max(xmax - xmin, ymax - ymin) * 1.02
    side = own if side is None else side
    assert side >= own
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return {'x0': cx - side / 2.0, 'y0': cy - side / 2.0, 'side': side,
            'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}


def bin_idx(v, lo, side):
    k = int(math.floor((v - lo) / side * G))
    return min(max(k, 0), G - 1)


def process_map(name, clean, kills, side):
    mc = clean.filter(pl.col('map_name') == name)
    matches = sorted(mc['match_key'].unique().to_list())
    m_idx = {m: k for k, m in enumerate(matches)}
    M = len(matches)
    rounds_pm = [0] * M
    for mk in mc['match_key'].to_list():
        rounds_pm[m_idx[mk]] += 1
    R = mc.height
    assert sum(rounds_pm) == R

    mk_k = kills.filter(pl.col('map_name') == name)
    keys = mk_k['match_key'].to_list()
    xs = mk_k['player_x_pos'].to_list()
    ys = mk_k['player_y_pos'].to_list()
    teams = mk_k['player_team_code'].to_list()
    secs = [(t - f) / 64.0 for t, f in zip(mk_k['tick'].to_list(),
                                           mk_k['freeze_end_tick'].to_list())]
    n_kills = len(xs)
    geo = geometry(xs, ys, side)
    cw = geo['side'] / G

    counts = [[0] * G for _ in range(G)]
    cell_of = []
    for x, y in zip(xs, ys):
        i = bin_idx(x, geo['x0'], geo['side'])
        j = bin_idx(y, geo['y0'], geo['side'])
        counts[j][i] += 1
        cell_of.append(j * G + i)
    grid_sum = sum(sum(row) for row in counts)
    assert grid_sum == n_kills

    flat = [(counts[j][i], i, j) for j in range(G) for i in range(G)]
    ranked = sorted(flat, key=lambda t: (-t[0], t[2], t[1]))
    n_top = int(round(TOP_FRAC * G * G))
    top_cells = {j * G + i for _, i, j in ranked[:n_top]}
    top_cut = ranked[n_top - 1][0]
    cells_at_cut = sum(1 for c, _, _ in flat if c == top_cut)

    kills_pm = [0] * M
    top_pm = [0] * M
    for mk, cell in zip(keys, cell_of):
        k = m_idx[mk]
        kills_pm[k] += 1
        if cell in top_cells:
            top_pm[k] += 1
    assert sum(kills_pm) == n_kills

    kpr = n_kills / R
    top_inside = sum(top_pm)
    top_share = top_inside / n_kills

    rng = random.Random(SEED)
    kpr_boot = []
    top_boot = []
    for _ in range(N_BOOT):
        kk = rr = tt = 0
        for _m in range(M):
            k = rng.randrange(M)
            kk += kills_pm[k]
            rr += rounds_pm[k]
            tt += top_pm[k]
        kpr_boot.append(kk / rr)
        top_boot.append(tt / kk)

    kpr_lo, kpr_hi = pct(kpr_boot, 0.025), pct(kpr_boot, 0.975)
    top_lo, top_hi = pct(top_boot, 0.025), pct(top_boot, 0.975)
    kpr_se = statistics_stdev(kpr_boot)

    cmax, hi_i, hi_j = ranked[0]
    hot_cell = hi_j * G + hi_i
    hot_k = [k for k, c in enumerate(cell_of) if c == hot_cell]
    hot_teams = {}
    for k in hot_k:
        key = str(teams[k])
        hot_teams[key] = hot_teams.get(key, 0) + 1
    team_counts = {}
    for t in teams:
        team_counts[str(t)] = team_counts.get(str(t), 0) + 1
    hot_secs = [secs[k] for k in hot_k]
    assert len(hot_k) == cmax
    return {
        'map_name': name,
        'clean_rounds': R,
        'matches': M,
        'kills': n_kills,
        'kills_per_round': round(kpr, 6),
        'kills_per_round_ci95': [round(kpr_lo, 6), round(kpr_hi, 6)],
        'kills_per_round_boot_se': round(kpr_se, 6),
        'kills_per_round_point_inside_ci': bool(kpr_lo <= kpr <= kpr_hi),
        'top_decile_cells': n_top,
        'top_decile_share_of_kills': round(top_share, 6),
        'top_decile_share_ci95': [round(top_lo, 6), round(top_hi, 6)],
        'top_decile_point_inside_ci': bool(top_lo <= top_share <= top_hi),
        'top_decile_cut_count': top_cut,
        'top_decile_cells_tied_at_cut': cells_at_cut,
        'max_cell': cmax,
        'max_cell_index_i_j': [hi_i, hi_j],
        'max_cell_center_world': [round(geo['x0'] + (hi_i + 0.5) * cw, 1),
                                  round(geo['y0'] + (hi_j + 0.5) * cw, 1)],
        'empty_cells': sum(1 for c, _, _ in flat if c == 0),
        'victim_team_counts_all_kills': team_counts,
        'median_seconds_after_freeze_end_all_kills': round(pct(secs, 0.5), 2),
        'hot_cell': {
            'kills': len(hot_k),
            'distinct_matches': len({keys[k] for k in hot_k}),
            'victim_team_counts': hot_teams,
            'median_seconds_after_freeze_end': round(pct(hot_secs, 0.5), 2),
            'share_of_kills_within_first_15s': round(
                sum(1 for s in hot_secs if s < 15.0) / len(hot_secs), 4),
            'center_world': [round(geo['x0'] + (hi_i + 0.5) * cw, 1),
                             round(geo['y0'] + (hi_j + 0.5) * cw, 1)],
        },
        'grid': {
            'bins_per_axis': G,
            'x0': round(geo['x0'], 3), 'y0': round(geo['y0'], 3),
            'side_world_units': round(geo['side'], 3),
            'cell_world_units': round(cw, 3),
            'data_xmin': round(geo['xmin'], 3), 'data_xmax': round(geo['xmax'], 3),
            'data_ymin': round(geo['ymin'], 3), 'data_ymax': round(geo['ymax'], 3),
            'counts_rows_bottom_to_top': 'counts[j][i]: j = y bin (0 = lowest y), i = x bin (0 = lowest x)',
        },
        'counts': counts,
        'grid_sum_equals_kills': bool(grid_sum == n_kills),
        'counts_nonnegative': bool(all(c >= 0 for row in counts for c in row)),
    }


def statistics_stdev(vals):
    n = len(vals)
    mu = sum(vals) / n
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))


def nice_ticks(lo, span, max_ticks=6):
    hi = lo + span
    ticks = []
    for step in (100, 200, 250, 500, 1000, 2000, 2500, 5000):
        first = math.ceil(lo / step) * step
        ticks = []
        v = first
        while v <= hi:
            ticks.append(v)
            v += step
        if len(ticks) <= max_ticks:
            return ticks
    return ticks


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, s, size=12, anchor='start', fill=NEUTRAL, weight=None, extra=''):
    w = f' font-weight="{weight}"' if weight else ''
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}"{w}{extra}>{esc(s)}</text>')


def hex_rgb(h):
    return tuple(int(h[k:k + 2], 16) for k in (1, 3, 5))


def ramp(t):
    t = min(max(t, 0.0), 1.0)
    pos = t * (len(STOPS) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(STOPS) - 1)
    f = pos - lo
    a, b = hex_rgb(STOPS[lo]), hex_rgb(STOPS[hi])
    return '#%02x%02x%02x' % tuple(int(round(a[c] + (b[c] - a[c]) * f)) for c in range(3))


def gradient_def():
    n = len(STOPS)
    stops = ''.join(f'<stop offset="{k / (n - 1):.4f}" stop-color="{STOPS[n - 1 - k]}"/>'
                    for k in range(n))
    return ('<defs><linearGradient id="cbar" x1="0" y1="0" x2="0" y2="1">'
            + stops + '</linearGradient></defs>')


def draw_panel(o, bx, res, vmax, y_title=True):
    S = BOX_S
    by = BOX_Y
    geo = res['grid']
    cw = S / G
    n_rounds = res['clean_rounds']
    n_matches = res['matches']

    o.append(text(bx, 26, res['map_name'], 13, weight='bold', fill='#1e293b'))
    o.append(text(bx, 42, f"n = {n_rounds:,} rounds ({n_matches} matches), "
                          f"{res['kills']:,} kills", 12))
    lo, hi = res['kills_per_round_ci95']
    o.append(text(bx, 57, f"Kills per round {res['kills_per_round']:.2f} "
                          f"[{lo:.2f}, {hi:.2f}]", 11))
    tlo, thi = res['top_decile_share_ci95']
    o.append(text(bx, 72, f"Top 10% of cells: {res['top_decile_share_of_kills'] * 100:.1f}% "
                          f"of kills [{tlo * 100:.1f}, {thi * 100:.1f}]", 11, fill=ACCENT))

    o.append(f'<rect x="{bx}" y="{by}" width="{S}" height="{S}" fill="#ffffff"/>')
    counts = res['counts']
    for j in range(G):
        row = counts[j]
        for i in range(G):
            c = row[i]
            if c > 0:
                x = bx + i * cw
                y = by + (G - 1 - j) * cw
                o.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw:.2f}" '
                         f'height="{cw:.2f}" fill="{ramp(c / vmax)}"/>')

    # Axis lines
    o.append(f'<line x1="{bx}" y1="{by + S}" x2="{bx + S}" y2="{by + S}" '
             f'stroke="{NEUTRAL}" stroke-width="1"/>')
    o.append(f'<line x1="{bx}" y1="{by}" x2="{bx}" y2="{by + S}" '
             f'stroke="{NEUTRAL}" stroke-width="1"/>')

    x0, y0, side = geo['x0'], geo['y0'], geo['side_world_units']
    for v in nice_ticks(x0, side):
        px = bx + (v - x0) / side * S
        o.append(f'<line x1="{px:.1f}" y1="{by + S}" x2="{px:.1f}" y2="{by + S + 4}" '
                 f'stroke="{NEUTRAL}" stroke-width="1"/>')
        o.append(text(px, by + S + 17, f'{v:,.0f}', 12, 'middle'))
    for v in nice_ticks(y0, side):
        py = by + S - (v - y0) / side * S
        o.append(f'<line x1="{bx - 4}" y1="{py:.1f}" x2="{bx}" y2="{py:.1f}" '
                 f'stroke="{NEUTRAL}" stroke-width="1"/>')
        o.append(text(bx - 7, py + 4, f'{v:,.0f}', 12, 'end'))

    o.append(text(bx + S / 2, by + S + 36, 'Victim x position (game units)', 13, 'middle'))
    if y_title:
        ymid = by + S / 2
        o.append(f'<text x="18" y="{ymid:.1f}" font-size="13" text-anchor="middle" '
                 f'fill="{NEUTRAL}" transform="rotate(-90 18 {ymid:.1f})">'
                 f'Victim y position (game units)</text>')


def draw_colorbar(o, vmax):
    x, y, w, h = CB_X, BOX_Y, CB_W, BOX_S
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="url(#cbar)" '
             f'stroke="{NEUTRAL}" stroke-width="0.5"/>')
    for v in (0, vmax / 2.0, float(vmax)):
        ty = y + (1 - v / vmax) * h
        o.append(f'<line x1="{x + w}" y1="{ty:.1f}" x2="{x + w + 4}" y2="{ty:.1f}" '
                 f'stroke="{NEUTRAL}" stroke-width="1"/>')
        o.append(text(x + w + 7, ty + 4, f'{int(round(v)):,}', 11))
    o.append(text(x, y - 26, 'Kills', 11))
    o.append(text(x, y - 12, 'per cell', 11))


def build_svg(results, vmax):
    o = []
    o.append('<svg viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg" '
             'font-family="sans-serif">')
    o.append('<rect x="0" y="0" width="800" height="450" fill="#ffffff"/>')
    o.append(gradient_def())
    for k, res in enumerate(results):
        draw_panel(o, PANEL_X[k], res, vmax, y_title=(k == 0))
    draw_colorbar(o, vmax)
    o.append(text(18, 416, 'No radar image is available for these maps: panels show raw CS2 '
                           'world coordinates (game units, y up).', 10))
    o.append(text(18, 430, 'Both panels share one colour scale (linear, kills per cell) and one world-unit '
                           'scale. Kills = enemy kills, clean 5v5 rounds.', 10))
    o.append(text(18, 444, 'Brackets: 95% match-cluster bootstrap CI (1,000 resamples, seed 0). '
                           'Source: 2026 tome, 2026-10-01 to 08.', 10))
    o.append('</svg>')
    return '\n'.join(o) + '\n'


def main():
    hdr, re_, clean, pdth = load()

    ranking = map_rounds(clean)
    top_maps = ranking['map_name'].to_list()[:N_MAPS]
    reg_rank = map_rounds(clean.filter(pl.col('round') <= REG_MAX_ROUND))
    reg_top = reg_rank['map_name'].to_list()[:N_MAPS]

    n_all = pdth.height
    n_dup = n_all - pdth.unique(subset=['match_key', 'round', 'tick', 'player_id']).height
    dc = pdth.join(clean.select(['match_key', 'round', 'map_name']),
                   on=['match_key', 'round'], how='inner')
    dc = classify(dc).with_columns(
        (pl.col('player_x_pos').is_not_null() & pl.col('player_y_pos').is_not_null())
        .alias('has_pos'))
    fe = (pl.scan_parquet(ROUND_STATE).select(['match_key', 'round', 'event_type', 'tick'])
          .filter(pl.col('event_type') == 'round_freeze_end')
          .group_by(['match_key', 'round']).agg(pl.col('tick').min().alias('freeze_end_tick'))
          .collect())
    dc = dc.join(fe, on=['match_key', 'round'], how='left')
    cls_counts = {str(k): int(v) for k, v in dc.group_by('cls').len().iter_rows()}
    enemy_nopos = dc.filter((pl.col('cls') == 'enemy') & ~pl.col('has_pos')).height
    kills = dc.filter((pl.col('cls') == 'enemy') & pl.col('has_pos'))
    kills_no_fe = kills.filter(pl.col('freeze_end_tick').is_null()).height
    assert kills_no_fe == 0

    own_side = {}
    for name in top_maps:
        mk_k = kills.filter(pl.col('map_name') == name)
        own_side[name] = geometry(mk_k['player_x_pos'].to_list(),
                                  mk_k['player_y_pos'].to_list())['side']
    shared_side = max(own_side.values())
    results = [process_map(name, clean, kills, shared_side) for name in top_maps]
    for r in results:
        r['grid']['own_window_side_world_units'] = round(own_side[r['map_name']], 3)
    vmax = max(r['max_cell'] for r in results)

    svg = build_svg(results, vmax)
    assert 'nan' not in svg.lower() and 'inf' not in svg.lower().replace('info', '')
    with open(OUT_SVG, 'w', encoding='utf-8', newline='\n') as f:
        f.write(svg)

    checks = {
        'round_end_rows': re_.height,
        'clean_round_rows_all_round_numbers': clean.height,
        'clean_round_rows_regulation_1_24': clean.filter(pl.col('round') <= REG_MAX_ROUND).height,
        'clean_round_max_round_number': int(clean['round'].max()),
        'player_death_rows_total': n_all,
        'player_death_exact_duplicates': n_dup,
        'player_death_rows_in_clean_rounds': dc.height,
        'player_death_rows_outside_clean_rounds': n_all - dc.height,
        'classification_counts_in_clean_rounds': cls_counts,
        'enemy_kills_missing_victim_position_dropped': enemy_nopos,
        'kills_missing_freeze_end_tick': kills_no_fe,
        'map_rank_top5_clean_rounds': [
            {'map_name': r['map_name'], 'rounds': int(r['rounds']), 'matches': int(r['matches'])}
            for r in ranking.head(5).iter_rows(named=True)],
        'map_top2_regulation_rounds_only': reg_top,
        'map_top2_matches_all_rounds_ranking': top_maps,
        'shared_colour_scale_vmax': vmax,
        'shared_window_side_world_units': round(shared_side, 3),
        'own_window_side_world_units': {k: round(v, 3) for k, v in own_side.items()},
        'bootstrap_resamples': N_BOOT,
        'bootstrap_seed': SEED,
        'bootstrap_unit': 'match (whole matches resampled with replacement, per map)',
        'all_counts_nonnegative': all(r['counts_nonnegative'] for r in results),
        'all_grid_sums_equal_kills': all(r['grid_sum_equals_kills'] for r in results),
        'all_ci_ordered': all(r['kills_per_round_ci95'][0] <= r['kills_per_round_ci95'][1]
                              and r['top_decile_share_ci95'][0] <= r['top_decile_share_ci95'][1]
                              for r in results),
        'all_point_inside_ci': all(r['kills_per_round_point_inside_ci']
                                   and r['top_decile_point_inside_ci'] for r in results),
        'svg_bytes': len(svg.encode('utf-8')),
    }

    maps_out = []
    for r in results:
        maps_out.append(r)

    result = {
        'id': 'T7',
        'title': 'Kill density by map (victim positions), two maps with most rounds, 2026 tome',
        'definition': {
            'population': 'player_death rows in clean 5v5 rounds (round_end.player_count == 10)',
            'kill': 'attacker_id present, attacker_id != player_id, both team codes present '
                    'and different (2 = T, 3 = CT)',
            'position': 'victim position: player_x_pos, player_y_pos (world units, no radar)',
            'maps': top_maps,
            'map_rule': 'two maps with most clean 5v5 rounds',
            'grid': '40 x 40 square bins, counts per cell, linear colour scale; both panels use '
                    'the same world-unit window side (the larger own window), each centred on '
                    'its own kills',
            'shared_colour_scale': True,
            'tome_dir': TOME,
        },
        'bootstrap': {
            'method': 'match-cluster (resample whole matches with replacement, per map)',
            'resamples': N_BOOT,
            'seed': SEED,
            'rng': 'random.Random(seed), fresh generator per map',
            'interval': '95% percentile (2.5th to 97.5th, linear interpolation)',
            'statistics': ['kills per clean round', 'share of kills in top 10% of cells '
                           '(cell set fixed from point estimate)'],
        },
        'maps': maps_out,
        'chart': {
            'svg': OUT_SVG,
            'viewbox': '0 0 800 450',
            'panels': [r['map_name'] for r in results],
            'colour_ramp': STOPS,
            'vmax': vmax,
            'note': 'No radar image; raw world coordinates; y up.',
        },
        'checks': checks,
    }
    with open(OUT_JSON, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)

    for r in results:
        print('%s: rounds=%d matches=%d kills=%d kpr=%.4f CI [%.4f, %.4f] '
              'top10 share=%.4f CI [%.4f, %.4f] max_cell=%d at %s centre=%s'
              % (r['map_name'], r['clean_rounds'], r['matches'], r['kills'],
                 r['kills_per_round'], r['kills_per_round_ci95'][0],
                 r['kills_per_round_ci95'][1], r['top_decile_share_of_kills'],
                 r['top_decile_share_ci95'][0], r['top_decile_share_ci95'][1],
                 r['max_cell'], r['max_cell_index_i_j'], r['max_cell_center_world']))
    print('vmax', vmax)
    print('checks', json.dumps(checks))


if __name__ == '__main__':
    main()
