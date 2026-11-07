"""T1 - CT round win rate by round number (1-24), 2026 tome.

Population: round_end rows of the 2026-10-01..08 tome with player_count == 10 (clean 5v5).
CT win: winner_team_code == 3 (side code 2 = T, 3 = CT).
Intervals: match-cluster bootstrap. Whole matches are resampled with replacement, 1,000
resamples, random.Random(0). The 2.5th and 97.5th percentiles (linear interpolation)
give the 95% interval. One resample draw is shared by all rounds, so within-match
correlation across rounds is preserved.

Writes data/viz/T1.json and data/viz/T1.svg under the repo root.
Run: G:/pureskill/csds-cs2/.pixi/envs/default/python.exe -I -X utf8 scripts/viz/T1.py
"""
import json
import math
import random
import statistics
from collections import Counter

import polars as pl

ROOT = 'G:/pureskill/csds-cs2'
ROUND_END = (ROOT + '/tomes/2026-10-01_to_08/tome/csds/'
             'basic_round_end.2026-09-18,2026-10-08/dataframe_00000')
STATE_ROUNDS = ROOT + '/data/state_sample/rounds.parquet'
OUT_JSON = ROOT + '/data/viz/T1.json'
OUT_SVG = ROOT + '/data/viz/T1.svg'

N_BOOT = 1000
SEED = 0
CLEAN = 10
R_LO, R_HI = 1, 24
PISTOL_ROUNDS = (1, 13)
HALFTIME_AFTER = 12

CT_BLUE = '#2563eb'
NEUTRAL = '#475569'
GRID = '#e2e8f0'
ACCENT = '#059669'


def pct(vals, q):
    """Linear-interpolation percentile, q in [0, 1]."""
    s = sorted(vals)
    n = len(s)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def svg_text(x, y, s, size=12, anchor='start', fill=NEUTRAL):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}">{s}</text>')


def build_svg(per_round, n_total, m_total, m_frame, ymin, ymax, ystep):
    W, H = 800, 450
    X0, X1 = 80.0, 776.0
    Y0, Y1 = 52.0, 350.0
    BW = (X1 - X0) / 24.0

    def xc(r):
        return X0 + (r - 0.5) * BW

    def yv(v):
        return Y1 - (v - ymin) / (ymax - ymin) * (Y1 - Y0)

    o = []
    o.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'font-family="sans-serif">')
    o.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # Legend (top row)
    ly = 22.0
    o.append(f'<line x1="80" y1="{ly}" x2="100" y2="{ly}" stroke="{CT_BLUE}" stroke-width="1.5"/>')
    o.append(f'<circle cx="90" cy="{ly}" r="3.5" fill="{CT_BLUE}"/>')
    o.append(svg_text(106, ly + 4, 'CT win rate (95% CI)', 12, fill=CT_BLUE))
    o.append(f'<circle cx="262" cy="{ly}" r="5" fill="{ACCENT}"/>')
    o.append(svg_text(274, ly + 4, 'Pistol round (1, 13)', 12, fill=ACCENT))
    o.append(f'<line x1="424" y1="{ly}" x2="444" y2="{ly}" stroke="{NEUTRAL}" '
             f'stroke-width="1" stroke-dasharray="4 3"/>')
    o.append(svg_text(450, ly + 4, '50% parity', 12))
    o.append(f'<line x1="534" y1="{ly - 6}" x2="534" y2="{ly + 6}" stroke="{NEUTRAL}" '
             f'stroke-width="1.5" stroke-dasharray="6 4"/>')
    o.append(svg_text(546, ly + 4, 'halftime', 12))

    # Pistol-round column shading
    for r in PISTOL_ROUNDS:
        o.append(f'<rect x="{X0 + (r - 1) * BW:.1f}" y="{Y0}" width="{BW:.1f}" '
                 f'height="{Y1 - Y0}" fill="{ACCENT}" fill-opacity="0.08"/>')

    # Horizontal grid and y ticks
    k0 = int(round(ymin / ystep))
    k1 = int(round(ymax / ystep))
    ticks = [round(k * ystep, 4) for k in range(k0, k1 + 1)]
    for v in ticks:
        y = yv(v)
        o.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<line x1="{X0 - 4}" y1="{y:.1f}" x2="{X0}" y2="{y:.1f}" stroke="{NEUTRAL}" stroke-width="1"/>')
        o.append(svg_text(X0 - 8, y + 4, f'{v * 100:.0f}%', 12, 'end'))

    # 50% parity reference
    y50 = yv(0.5)
    o.append(f'<line x1="{X0}" y1="{y50:.1f}" x2="{X1}" y2="{y50:.1f}" stroke="{NEUTRAL}" '
             f'stroke-width="1" stroke-dasharray="4 3"/>')

    # Halftime line (between round 12 and round 13)
    xh = X0 + HALFTIME_AFTER * BW
    o.append(f'<line x1="{xh:.1f}" y1="{Y0}" x2="{xh:.1f}" y2="{Y1}" stroke="{NEUTRAL}" '
             f'stroke-width="1.5" stroke-dasharray="6 4"/>')
    o.append(svg_text(xh + 6, Y1 - 8, 'halftime: sides swap', 11))

    # Axes
    o.append(f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="{NEUTRAL}" stroke-width="1"/>')
    o.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="{NEUTRAL}" stroke-width="1"/>')

    # Error bars (95% match-cluster bootstrap)
    for row in per_round:
        x = xc(row['round'])
        ylo = yv(row['ci95_low'])
        yhi = yv(row['ci95_high'])
        o.append(f'<line x1="{x:.1f}" y1="{yhi:.1f}" x2="{x:.1f}" y2="{ylo:.1f}" '
                 f'stroke="{CT_BLUE}" stroke-width="1.5" stroke-opacity="0.85"/>')
        for yy in (ylo, yhi):
            o.append(f'<line x1="{x - 4:.1f}" y1="{yy:.1f}" x2="{x + 4:.1f}" y2="{yy:.1f}" '
                     f'stroke="{CT_BLUE}" stroke-width="1.5" stroke-opacity="0.85"/>')

    # Connector through point estimates
    pts = ' '.join(f'{xc(row["round"]):.1f},{yv(row["ct_win_rate"]):.1f}' for row in per_round)
    o.append(f'<polyline points="{pts}" fill="none" stroke="{CT_BLUE}" '
             f'stroke-width="1.2" stroke-opacity="0.5"/>')

    # Point estimates and pistol labels
    for row in per_round:
        r = row['round']
        x = xc(r)
        yest = yv(row['ct_win_rate'])
        if r in PISTOL_ROUNDS:
            o.append(f'<circle cx="{x:.1f}" cy="{yest:.1f}" r="5.5" fill="{ACCENT}" '
                     f'stroke="#ffffff" stroke-width="1"/>')
            o.append(svg_text(x, yv(row['ci95_high']) - 7, 'pistol', 10, 'middle', ACCENT))
        else:
            o.append(f'<circle cx="{x:.1f}" cy="{yest:.1f}" r="3.5" fill="{CT_BLUE}" '
                     f'stroke="#ffffff" stroke-width="1"/>')

    # X ticks, round labels, n row
    for row in per_round:
        r = row['round']
        x = xc(r)
        o.append(f'<line x1="{x:.1f}" y1="{Y1}" x2="{x:.1f}" y2="{Y1 + 4}" stroke="{NEUTRAL}" stroke-width="1"/>')
        o.append(svg_text(x, Y1 + 17, str(r), 12, 'middle'))
        o.append(svg_text(x, Y1 + 36, f"{row['n_rounds']:,}", 10, 'middle'))
    o.append(svg_text(X0 - 8, Y1 + 36, 'n rounds', 10, 'end'))

    # Axis titles
    o.append(svg_text((X0 + X1) / 2, Y1 + 60, 'Round number (MR12: sides swap after round 12)', 13, 'middle'))
    ymid = (Y0 + Y1) / 2
    o.append(f'<text x="22" y="{ymid:.1f}" font-size="13" text-anchor="middle" fill="{NEUTRAL}" '
             f'transform="rotate(-90 22 {ymid:.1f})">CT round win rate</text>')

    # Sample size and method footer
    o.append(svg_text(X0 + 8, Y0 + 16,
                      f'n = {n_total:,} rounds (rounds 1-24, player_count = 10)', 11))
    o.append(svg_text(X0, 434,
                      f'Source: 2026 tome, {m_frame:,} matches ({m_total:,} with a clean round 1-24). '
                      'Whiskers: 95% match-cluster bootstrap CI, 1,000 resamples, seed 0.', 10))
    o.append('</svg>')
    return '\n'.join(o) + '\n'


def sample_check(round_end_df):
    s = pl.read_parquet(STATE_ROUNDS).select(['match_key', 'round', 'ct_win'])
    j = s.join(round_end_df.select(['match_key', 'round', 'winner_team_code']),
               on=['match_key', 'round'], how='left')
    missing = j.filter(pl.col('winner_team_code').is_null()).height
    agree = j.filter(
        pl.col('winner_team_code').is_not_null()
        & ((pl.col('ct_win') == 1) == (pl.col('winner_team_code') == 3))
    ).height
    return {'sample_rounds': s.height,
            'sample_rounds_missing_in_tome': missing,
            'sample_ct_win_agrees_with_tome': agree}


def main():
    df = pl.scan_parquet(ROUND_END).select(
        ['match_key', 'round', 'winner_team_code', 'win_reason_code', 'player_count']
    ).collect()

    checks = {}
    checks['round_end_rows'] = df.height
    checks['round_end_duplicate_match_round_rows'] = df.height - df.unique(
        subset=['match_key', 'round']).height
    checks['winner_codes_present'] = sorted(df['winner_team_code'].unique().to_list())
    checks['round_min'] = int(df['round'].min())
    checks['round_max'] = int(df['round'].max())
    assert checks['round_end_duplicate_match_round_rows'] == 0
    assert set(checks['winner_codes_present']) <= {2, 3}
    checks.update(sample_check(df))

    matches = sorted(df['match_key'].unique().to_list())
    m_idx = {m: i for i, m in enumerate(matches)}
    M = len(matches)
    checks['matches_in_frame'] = M

    clean = df.filter(pl.col('player_count') == CLEAN)
    raw_1_24 = df.filter(pl.col('round').is_between(R_LO, R_HI))
    clean_1_24 = clean.filter(pl.col('round').is_between(R_LO, R_HI))

    max_round = int(clean['round'].max())
    R = max_round + 1

    # per_match[i] = list of (round, ct_win) for clean rounds of match i
    per_match = [[] for _ in range(M)]
    for mk, rd, wt in clean.select(['match_key', 'round', 'winner_team_code']).iter_rows():
        per_match[m_idx[mk]].append((int(rd), 1 if wt == 3 else 0))

    # Point estimates
    n_r = [0] * R
    w_r = [0] * R
    for lst in per_match:
        for r, ct in lst:
            n_r[r] += 1
            w_r[r] += ct
    N1 = sum(n_r[R_LO:R_HI + 1])
    W1 = sum(w_r[R_LO:R_HI + 1])
    NA = sum(n_r)
    WA = sum(w_r)

    # Match-cluster bootstrap
    rng = random.Random(SEED)
    boot = {r: [] for r in range(R_LO, R_HI + 1)}
    boot_1_24 = []
    boot_all = []
    skipped = Counter()
    for _ in range(N_BOOT):
        mult = Counter(rng.randrange(M) for _ in range(M))
        n = [0] * R
        w = [0] * R
        for mi, k in mult.items():
            for r, ct in per_match[mi]:
                n[r] += k
                w[r] += k * ct
        for r in range(R_LO, R_HI + 1):
            if n[r] > 0:
                boot[r].append(w[r] / n[r])
            else:
                skipped[r] += 1
        boot_1_24.append(sum(w[R_LO:R_HI + 1]) / sum(n[R_LO:R_HI + 1]))
        boot_all.append(sum(w) / sum(n))

    matches_1_24 = len({mk for mk, rd in clean_1_24.select(['match_key', 'round']).iter_rows()})
    raw_by_round = dict(
        raw_1_24.group_by('round').len().iter_rows()
    )

    per_round = []
    for r in range(R_LO, R_HI + 1):
        rate = w_r[r] / n_r[r]
        lo = pct(boot[r], 0.025)
        hi = pct(boot[r], 0.975)
        assert n_r[r] >= 0 and 0.0 <= rate <= 1.0
        assert lo <= rate + 1e-12 and rate - 1e-12 <= hi, (r, lo, rate, hi)
        per_round.append({
            'round': r,
            'n_rounds': n_r[r],
            'n_matches': n_r[r],
            'ct_wins': w_r[r],
            't_wins': n_r[r] - w_r[r],
            'ct_win_rate': round(rate, 6),
            'ci95_low': round(lo, 6),
            'ci95_high': round(hi, 6),
            'boot_se': round(statistics.stdev(boot[r]), 6),
            'boot_resamples_used': len(boot[r]),
            'raw_rows_all_player_counts': int(raw_by_round.get(r, 0)),
            'is_pistol': r in PISTOL_ROUNDS,
        })

    ov1_lo = pct(boot_1_24, 0.025)
    ov1_hi = pct(boot_1_24, 0.975)
    ova_lo = pct(boot_all, 0.025)
    ova_hi = pct(boot_all, 0.975)

    los = [row['ci95_low'] for row in per_round]
    his = [row['ci95_high'] for row in per_round]
    ystep = 0.05
    ymin = round(math.floor((min(min(los), 0.5) - 0.01) / ystep) * ystep, 4)
    ymax = round(math.ceil((max(max(his), 0.5) + 0.01) / ystep) * ystep, 4)
    if (ymax - ymin) / ystep > 8:
        ystep = 0.10
        ymin = round(math.floor(ymin / ystep) * ystep, 4)
        ymax = round(math.ceil(ymax / ystep) * ystep, 4)

    checks['bootstrap_resamples'] = N_BOOT
    checks['bootstrap_skipped_zero_n_by_round'] = dict(skipped)
    checks['sum_per_round_n_equals_overall_1_24'] = sum(row['n_rounds'] for row in per_round) == N1
    checks['clean_rounds_excluded_1_24_non10'] = int(raw_1_24.height - clean_1_24.height)
    checks['pistol_rounds_check'] = [row['round'] for row in per_round if row['is_pistol']]

    svg = build_svg(per_round, N1, matches_1_24, M, ymin, ymax, ystep)
    with open(OUT_SVG, 'w', encoding='utf-8', newline='\n') as f:
        f.write(svg)

    result = {
        'id': 'T1',
        'title': 'CT round win rate by round number, rounds 1-24, 2026 tome',
        'definition': {
            'population': 'round_end rows, player_count == 10 (clean 5v5)',
            'outcome': 'CT win = winner_team_code == 3',
            'rounds_plotted': [R_LO, R_HI],
            'frame_matches': M,
            'tome': ROUND_END,
        },
        'bootstrap': {
            'method': 'match-cluster (resample whole matches with replacement)',
            'resamples': N_BOOT,
            'seed': SEED,
            'rng': 'random.Random(seed)',
            'interval': '95% percentile (2.5th to 97.5th, linear interpolation)',
            'shared_draw_across_rounds': True,
        },
        'overall_rounds_1_24': {
            'n_rounds': N1,
            'n_matches_with_a_round_in_1_24': matches_1_24,
            'ct_wins': W1,
            'ct_win_rate': round(W1 / N1, 6),
            'ci95_low': round(ov1_lo, 6),
            'ci95_high': round(ov1_hi, 6),
        },
        'overall_all_clean_rounds_incl_overtime': {
            'n_rounds': NA,
            'ct_wins': WA,
            'ct_win_rate': round(WA / NA, 6),
            'ci95_low': round(ova_lo, 6),
            'ci95_high': round(ova_hi, 6),
            'note': 'includes round numbers above 24 (overtime); not plotted',
        },
        'per_round': per_round,
        'highlights': {
            'pistol_round_1': next(r for r in per_round if r['round'] == 1),
            'pistol_round_13': next(r for r in per_round if r['round'] == 13),
            'halftime_line_after_round': HALFTIME_AFTER,
        },
        'chart': {
            'svg': OUT_SVG,
            'y_domain': [ymin, ymax],
            'y_step': ystep,
            'viewbox': '0 0 800 450',
        },
        'checks': checks,
    }
    with open(OUT_JSON, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)

    print('CT win rate rounds 1-24: %d/%d = %.4f  CI [%.4f, %.4f]'
          % (W1, N1, W1 / N1, ov1_lo, ov1_hi))
    print('CT win rate all clean rounds: %d/%d = %.4f  CI [%.4f, %.4f]'
          % (WA, NA, WA / NA, ova_lo, ova_hi))
    for row in per_round:
        print('R%02d n=%5d ct=%5d rate=%.4f CI [%.4f, %.4f]'
              % (row['round'], row['n_rounds'], row['ct_wins'], row['ct_win_rate'],
                 row['ci95_low'], row['ci95_high']))
    print('y domain', ymin, ymax, 'step', ystep)
    print('checks', json.dumps(checks))


if __name__ == '__main__':
    main()
