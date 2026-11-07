"""Independent recomputation of the N5 SMG vs rifle armor claim (verifier, id verify_N5).

Reads only:
  G:/pureskill/csds-cs2/data/state_sample/players_1hz.parquet
  G:/pureskill/csds-cs2/data/state_sample/rounds.parquet
  G:/pureskill/csds-cs2/tomes/combined_test/tome/csds/basic_weapon_fire.COMBINED (code->name crosstab only)
Writes only: G:/pureskill/csds-cs2/data/viz/verify_N5_smg_armor_results.json
"""
import glob
import json
import os

import numpy as np
import polars as pl

P = 'G:/pureskill/csds-cs2/data/state_sample/players_1hz.parquet'
R = 'G:/pureskill/csds-cs2/data/state_sample/rounds.parquet'
WF = 'G:/pureskill/csds-cs2/tomes/combined_test/tome/csds/basic_weapon_fire.COMBINED/dataframe_*'
OUT = 'G:/pureskill/csds-cs2/data/viz/verify_N5_smg_armor_results.json'
if os.path.exists(OUT):
    raise SystemExit('refusing to overwrite existing ' + OUT)

B = 500      # match-cluster bootstrap resamples
SEED = 0     # numpy default_rng seed
PISTOL_ROUNDS = [1, 13]

# inv_primary (primary inventory item def index) -> class.
# Names confirmed by the weapon_fire player_weapon_code -> weapon_name crosstab in the event tome
# (dominant name per code); classes follow the CS2 weapon taxonomy (ASSUMPTION: standard grouping).
PRIMARY_NAME = {
    0: 'none', 7: 'ak47', 8: 'aug', 9: 'awp', 10: 'famas', 11: 'g3sg1', 13: 'galilar', 14: 'm249',
    16: 'm4a1', 17: 'mac10', 19: 'p90', 23: 'mp5sd', 24: 'ump45', 25: 'xm1014', 26: 'bizon',
    27: 'mag7', 28: 'negev', 29: 'sawedoff', 33: 'mp7', 34: 'mp9', 35: 'nova', 38: 'scar20',
    39: 'sg556', 40: 'ssg08', 60: 'm4a1_silencer',
}
PRIMARY_CLASS = {
    0: 'none',
    7: 'rifle', 8: 'rifle', 10: 'rifle', 13: 'rifle', 16: 'rifle', 39: 'rifle', 60: 'rifle',
    17: 'smg', 19: 'smg', 23: 'smg', 24: 'smg', 26: 'smg', 33: 'smg', 34: 'smg',
    9: 'sniper', 11: 'sniper', 38: 'sniper', 40: 'sniper',
    14: 'shotgun_mg', 28: 'shotgun_mg', 25: 'shotgun_mg', 27: 'shotgun_mg', 29: 'shotgun_mg', 35: 'shotgun_mg',
}
# pistol item def indices (secondary slot), used only for the weapon_code sensitivity
PISTOL_DEF = {1, 2, 3, 4, 30, 32, 36, 61, 63, 64}
CLASSES = ['rifle', 'smg', 'sniper', 'shotgun_mg', 'none']

out = {'analysis_id': 'verify_N5', 'assumptions': [], 'checks': {}, 'tables': {}, 'bootstrap': {},
       'claim_comparison': []}

# ---------------------------------------------------------------- weapon code evidence
wf = pl.concat([pl.read_parquet(f, columns=['player_weapon_code', 'weapon_name']) for f in sorted(glob.glob(WF))],
               how='vertical_relaxed')
wf_ct = (wf.drop_nulls('player_weapon_code')
           .group_by(['player_weapon_code', 'weapon_name']).agg(pl.len().alias('n'))
           .sort(['player_weapon_code', 'n'], descending=[False, True]))
dominant = {}
for row in wf_ct.iter_rows(named=True):
    dominant.setdefault(row['player_weapon_code'], (row['weapon_name'], row['n']))
out['checks']['weapon_fire_dominant_name_for_codes'] = {
    str(k): v[0] for k, v in sorted(dominant.items()) if k in PRIMARY_NAME or k in PISTOL_DEF}
for code in sorted(PRIMARY_NAME):
    if code in dominant:
        assert dominant[code][0] == PRIMARY_NAME[code] or code in (0,), (code, dominant[code], PRIMARY_NAME[code])

# ---------------------------------------------------------------- load first live second
lf = pl.scan_parquet(P)
s0 = (lf.filter(pl.col('seconds_since_live') == 0)
        .select(['match_key', 'round', 'player_id', 'team_code', 'tick', 'armor', 'has_helmet', 'money',
                 'inv_primary', 'inv_secondary', 'weapon_code'])
        .collect())
out['checks']['sec0_rows'] = s0.height
out['checks']['sec0_dup_match_round_player'] = s0.height - s0.select(['match_key', 'round', 'player_id']).unique().height
out['checks']['sec0_distinct_matches'] = s0['match_key'].n_unique()
out['checks']['sec0_team_code_values'] = sorted(s0['team_code'].unique().to_list())

prim_vals = sorted(s0['inv_primary'].unique().to_list())
unmapped = [v for v in prim_vals if v not in PRIMARY_CLASS]
out['checks']['inv_primary_values_sec0'] = prim_vals
out['checks']['inv_primary_unmapped'] = unmapped
assert not unmapped, unmapped

# ---------------------------------------------------------------- round-level clean set
rd = pl.read_parquet(R, columns=['match_key', 'round', 'ct_win', 'live_start_tick', 'alive_gap_snapshots'])
pr = (s0.group_by(['match_key', 'round'])
        .agg([pl.len().alias('n'),
              (pl.col('team_code') == 2).sum().alias('nT'),
              (pl.col('team_code') == 3).sum().alias('nCT')]))
clean_rounds = pr.filter((pl.col('n') == 10) & (pl.col('nT') == 5) & (pl.col('nCT') == 5)).select(['match_key', 'round'])
out['checks']['rounds_sec0_total'] = pr.height
out['checks']['rounds_sec0_by_size'] = {str(k): v for k, v in
    pr.group_by('n').agg(pl.len().alias('r')).sort('n').iter_rows()}
out['checks']['clean_rounds_5v5_at_sec0'] = clean_rounds.height
out['checks']['clean_player_rounds_sec0'] = clean_rounds.height * 10
cmp_alive = (rd.join(clean_rounds, on=['match_key', 'round'], how='inner')
               .filter(pl.col('alive_gap_snapshots') == 0).height)
out['checks']['rounds_parquet_alive_gap0_matches_clean_5v5'] = (cmp_alive == clean_rounds.height)

# join sanity: sec0 tick equals live_start_tick; team 3 (CT) should win iff ct_win == 1
j = s0.join(rd.select(['match_key', 'round', 'ct_win', 'live_start_tick']), on=['match_key', 'round'], how='left')
out['checks']['sec0_rows_unmatched_to_rounds'] = int(j['ct_win'].is_null().sum())
out['checks']['sec0_tick_equals_live_start_tick_share'] = float(
    (j['tick'] == j['live_start_tick']).mean())
ct_side = (j['team_code'] == 3).cast(pl.Int8)
out['checks']['team3_is_CT_agreement_with_ct_win_share'] = float((ct_side == j['ct_win'].cast(pl.Int8)).mean())

d = s0.join(clean_rounds, on=['match_key', 'round'], how='inner')
d = d.join(rd.select(['match_key', 'round', 'ct_win']), on=['match_key', 'round'], how='left')
assert d.height == clean_rounds.height * 10
out['checks']['clean_player_rounds'] = d.height
out['checks']['clean_matches'] = d['match_key'].n_unique()

# ---------------------------------------------------------------- arrays
mk = d['match_key'].to_numpy()
matches = np.unique(mk)
midx = np.searchsorted(matches, mk)
M = len(matches)
round_no = d['round'].to_numpy()
team = d['team_code'].to_numpy()
armor = d['armor'].to_numpy()
helmet = d['has_helmet'].to_numpy().astype(bool)
money = d['money'].to_numpy().astype(float)
inv_p = d['inv_primary'].to_numpy()
wcode = d['weapon_code'].to_numpy()
pistol = np.isin(round_no, PISTOL_ROUNDS)
side_T = team == 2
side_CT = team == 3
armor_ge1 = (armor >= 1).astype(float)
cls = np.array([PRIMARY_CLASS[v] for v in inv_p])

out['checks']['matches_in_clean_set'] = int(M)
out['checks']['pistol_round_player_rows_clean'] = int(pistol.sum())
out['checks']['pistol_rounds_clean'] = int(d.filter(pl.col('round').is_in(PISTOL_ROUNDS)).select(['match_key', 'round']).unique().height)
out['checks']['pistol_rounds_by_round_number'] = {str(r): int(((round_no == r) & (team == 2)).sum() // 5) for r in PISTOL_ROUNDS}

# primary-weapon class counts (player-rounds) and mapping as used
out['mapping_used'] = {str(k): {'name': PRIMARY_NAME[k], 'class': PRIMARY_CLASS[k],
                                'player_rows_sec0': int((inv_p == k).sum())}
                       for k in sorted(PRIMARY_CLASS) if (inv_p == k).sum() > 0}

# ---------------------------------------------------------------- bootstrap setup (match clusters)
rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, M, size=(B, M))  # one draw shared by every statistic (paired)


def per_match(mask, y):
    num = np.bincount(midx[mask], weights=y[mask].astype(float), minlength=M)
    den = np.bincount(midx[mask], minlength=M).astype(float)
    return num, den


def boot_share(mask, y):
    num, den = per_match(mask, y)
    nb = num[boot_idx].sum(axis=1)
    db = den[boot_idx].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(db > 0, nb / np.where(db > 0, db, 1.0), np.nan)


def ci(a):
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def row(mask, label):
    n = int(mask.sum())
    if n == 0:
        return {'label': label, 'player_rounds': 0}
    res = {
        'label': label,
        'player_rounds': n,
        'rounds': int(d.filter(pl.Series(mask)).select(['match_key', 'round']).unique().height),
        'matches': int(len(np.unique(midx[mask]))),
        'armor_ge1_share': float(armor_ge1[mask].mean()),
        'helmet_share': float(helmet[mask].mean()),
        'mean_armor': float(armor[mask].mean()),
        'mean_money': float(money[mask].mean()),
    }
    b = boot_share(mask, armor_ge1)
    res['armor_ge1_share_ci95_match_cluster'] = ci(b)
    return res


def build_table(extra_mask, tag):
    t = {}
    for side_name, side_mask in [('ALL', np.ones(len(d), bool)), ('T', side_T), ('CT', side_CT)]:
        t[side_name] = {}
        for c in CLASSES:
            m = side_mask & (cls == c) & extra_mask
            t[side_name][c] = row(m, f'{side_name}|{c}|{tag}')
    return t


ALL_ROWS = np.ones(len(d), bool)
NO_PISTOL = ~pistol

out['tables']['all_rounds'] = build_table(ALL_ROWS, 'all_rounds')
out['tables']['excl_pistol_rounds'] = build_table(NO_PISTOL, 'excl_pistol_rounds')

# pistol-only players, same table restricted to class none (already in the table above)

# ---------------------------------------------------------------- SMG vs rifle comparisons
def diff_block(mask_a, mask_b, label):
    a_share = boot_share(mask_a, armor_ge1)
    b_share = boot_share(mask_b, armor_ge1)
    point_a = float(armor_ge1[mask_a].mean())
    point_b = float(armor_ge1[mask_b].mean())
    diffs = a_share - b_share
    return {'label': label, 'smg_share': point_a, 'rifle_share': point_b,
            'smg_minus_rifle_point': point_a - point_b,
            'smg_minus_rifle_ci95_match_cluster': ci(diffs),
            'smg_n': int(mask_a.sum()), 'rifle_n': int(mask_b.sum())}


cmp_blocks = {}
for tag, extra in [('all_rounds', ALL_ROWS), ('excl_pistol_rounds', NO_PISTOL)]:
    cmp_blocks[tag] = {}
    for side_name, side_mask in [('ALL', np.ones(len(d), bool)), ('T', side_T), ('CT', side_CT)]:
        cmp_blocks[tag][side_name] = diff_block(side_mask & (cls == 'smg') & extra,
                                                side_mask & (cls == 'rifle') & extra,
                                                f'{side_name}|smg_vs_rifle|{tag}')
out['smg_vs_rifle'] = cmp_blocks

# ---------------------------------------------------------------- sensitivity: weapon_code (active) instead of inv_primary
wc_cls = []
for code in wcode:
    if code in PISTOL_DEF:
        wc_cls.append('pistol_active')
    elif code in PRIMARY_CLASS and PRIMARY_CLASS[code] != 'none':
        wc_cls.append(PRIMARY_CLASS[code])
    else:
        wc_cls.append('other_active')
wc_cls = np.array(wc_cls)
out['sensitivity_weapon_code_active'] = {
    'active_class_counts': {k: int((wc_cls == k).sum()) for k in np.unique(wc_cls)},
    'smg_share_all': float(armor_ge1[wc_cls == 'smg'].mean()),
    'rifle_share_all': float(armor_ge1[wc_cls == 'rifle'].mean()),
    'smg_n': int((wc_cls == 'smg').sum()),
    'rifle_n': int((wc_cls == 'rifle').sum()),
}

# ---------------------------------------------------------------- claim comparison
claims = {
    'clean_player_rounds_sec0': (26720, out['checks']['clean_player_rounds']),
    'clean_rounds_sec0': (2672, out['checks']['clean_rounds_5v5_at_sec0']),
    'pistol_rounds_round1_or_13_in_clean_set': (270, out['checks']['pistol_rounds_clean']),
    'smg_armor_ge1_share_all_sides': (0.9469, out['tables']['all_rounds']['ALL']['smg']['armor_ge1_share']),
    'rifle_armor_ge1_share_all_sides': (0.9721, out['tables']['all_rounds']['ALL']['rifle']['armor_ge1_share']),
    'smg_minus_rifle_armor_share_all_sides': (-0.0252, cmp_blocks['all_rounds']['ALL']['smg_minus_rifle_point']),
    'smg_minus_rifle_armor_share_T': (-0.0162, cmp_blocks['all_rounds']['T']['smg_minus_rifle_point']),
    'smg_minus_rifle_armor_share_CT': (-0.0343, cmp_blocks['all_rounds']['CT']['smg_minus_rifle_point']),
}
for k, (claimed, mine) in claims.items():
    is_rate = isinstance(claimed, float) and abs(claimed) < 1.5
    if is_rate:
        ok = abs(mine - claimed) <= 0.005
    else:
        ok = abs(mine - claimed) <= 0.01 * abs(claimed)
    out['claim_comparison'].append({'name': k, 'claimed': claimed, 'recomputed': mine,
                                    'verdict': 'AGREE' if ok else 'DISAGREE'})

out['assumptions'] = [
    'ASSUMPTION: primary weapon = inv_primary at seconds_since_live == 0 (inventory slot), not weapon_code (active item).',
    'ASSUMPTION: inv_primary def indices map to the CS2 item definition names confirmed by weapon_fire crosstab; SSG 08 (40) is grouped as sniper.',
    'ASSUMPTION: pistol-only = inv_primary == 0.',
    'ASSUMPTION: clean round = exactly 5 T (team_code 2) and 5 CT (team_code 3) rows at sec0, matching rounds.parquet alive_gap_snapshots == 0.',
    'ASSUMPTION: pistol rounds = round 1 and round 13 (round column in players_1hz).',
    f'ASSUMPTION: match-cluster bootstrap, B={B} resamples of matches with replacement, numpy default_rng(seed={SEED}); percentile 95% CI; one draw shared across statistics.',
    'ASSUMPTION: armor >= 1 uses the armor column as recorded at sec0 (no post-round or later-second data used).',
]

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1, default=str)

# ---------------------------------------------------------------- print key numbers
print('CHECKS', json.dumps(out['checks'], indent=1, default=str))
print('MAPPING', json.dumps(out['mapping_used'], indent=1))
for tag in ['all_rounds', 'excl_pistol_rounds']:
    print('\n== TABLE', tag)
    for side in ['ALL', 'T', 'CT']:
        for c in CLASSES:
            r = out['tables'][tag][side][c]
            if r.get('player_rounds', 0) == 0:
                print(side, c, 'n=0')
                continue
            print(f"{side:3s} {c:10s} n={r['player_rounds']:6d} rounds={r['rounds']:5d} matches={r['matches']:3d} "
                  f"armor>=1={r['armor_ge1_share']:.4f} CI={[round(x, 4) for x in r['armor_ge1_share_ci95_match_cluster']]} "
                  f"helmet={r['helmet_share']:.4f} mean_armor={r['mean_armor']:.2f} mean_money={r['mean_money']:.0f}")
print('\n== SMG vs RIFLE')
for tag in cmp_blocks:
    for side in cmp_blocks[tag]:
        b = cmp_blocks[tag][side]
        print(tag, side, f"smg={b['smg_share']:.4f} rifle={b['rifle_share']:.4f} diff={b['smg_minus_rifle_point']:+.4f} "
              f"CI={[round(x, 4) for x in b['smg_minus_rifle_ci95_match_cluster']]} n_smg={b['smg_n']} n_rifle={b['rifle_n']}")
print('\n== WEAPON_CODE SENSITIVITY', json.dumps(out['sensitivity_weapon_code_active'], indent=1))
print('\n== CLAIM COMPARISON')
for c in out['claim_comparison']:
    print(c)
print('\nWROTE', OUT)
