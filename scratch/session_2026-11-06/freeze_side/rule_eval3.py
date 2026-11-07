import io, json, csv, collections, sys, boto3, time
import numpy as np
import polars as pl
from botocore.config import Config
T0 = time.time()
def log(*a): print('[%4.0fs]' % (time.time() - T0), *a, file=sys.stderr, flush=True)
s3 = boto3.client('s3', config=Config(retries={'max_attempts': 8, 'mode': 'standard'}))
B = 'pureskillgg-csgo-production-sagemaker-0'
FIX = '/home/sagemaker-user/adam/builder/fixes'
EXP = '/home/sagemaker-user/adam/builder/final/export_clean_v2/'
inv = json.load(open(FIX + '/freeze_fix/inv_matches.json'))
ovr = {}
for r in csv.DictReader(open(FIX + '/freeze_fix/override_rounds.csv')):
    ovr[(r['match_key'], int(r['round']))] = r
mk_all = json.load(open(EXP + 'match_keys.json'))
rmatch = np.load(EXP + 'round_match.npy').astype(np.int64)
rnum = np.load(EXP + 'round_num.npy').astype(np.int64)
idx = {k: i for i, k in enumerate(mk_all)}
def get(mk, name):
    folder = mk.rsplit('/', 1)[0]
    return pl.read_parquet(io.BytesIO(s3.get_object(Bucket=B, Key='data/' + folder + '/' + name)['Body'].read()))
def side_place(s):
    return 3 if s == 'CTSpawn' else (2 if s == 'TSpawn' else None)
tot = collections.Counter()
focus = []
for mk in inv:
    exp_rounds = set(int(x) for x in rnum[rmatch == idx[mk]])
    pi = get(mk, 'player_info')
    ps = get(mk, 'player_status').select('tick', 'round', 'player_id', 'player_controller_id', 'health', 'place_name', 'player_id_fixed').with_columns(pl.col('place_name').cast(pl.Utf8))
    pv = get(mk, 'player_vector').select('tick', 'round', 'player_id', 'team_code', 'x_pos', 'y_pos')
    pp = get(mk, 'player_personal')
    rs = get(mk, 'round_state').select('round', 'tick', 'event_type')
    dth = get(mk, 'player_death').select('round', 'tick', 'player_id', 'attacker_id', 'player_team_code', 'attacker_team_code', 'weapon_name')
    fzt = rs.filter(pl.col('event_type') == 'round_freeze_end').unique(subset=['round'], keep='first', maintain_order=True).select('round', pl.col('tick').alias('freeze'))
    # 1) identity uniqueness on the raw tables
    tot['status_fixed_dup_round_rows'] += ps.group_by(['round', 'tick', 'player_id_fixed']).len().filter(pl.col('len') > 1).height
    tot['info_fixed_multirow_round_pid'] += pi.group_by(['round', 'player_id_fixed']).len().filter(pl.col('len') > 1).height
    tot['personal_pawn_dup_round'] += pp.group_by(['round', 'player_id_pawn']).len().filter(pl.col('len') > 1).height
    tot['personal_fixed_dup_round'] += pp.group_by(['round', 'player_id_fixed']).len().filter(pl.col('len') > 1).height
    tot['player_name_rows'] += get(mk, 'player_name').height
    tot['personal_name_distinct'] += pp['name'].n_unique()
    # 2) info dup rows: which other fields differ between the two pid-0 rows
    dd = pi.filter(pl.col('player_id') == 0).group_by('round').agg(pl.col('team_code').n_unique().alias('k'), pl.len().alias('n')).filter(pl.col('n') > 1)
    tot['info_pid0_rounds_two_rows'] += dd.height
    for rr in dd['round'].to_list():
        rows = pi.filter((pl.col('round') == rr) & (pl.col('player_id') == 0)).select('wins', 'rank', 'radar_color_code', 'rank_raw', 'rank_platform', 'team_code').rows()
        if len(rows) == 2:
            other_same = all(rows[0][i] == rows[1][i] for i in range(5))
            tot['info_pid0_pair_other_fields_identical'] += int(other_same)
            tot['info_pid0_pair_other_fields_differ'] += int(not other_same)
    # 3) freeze-tick vector team nulls in export rounds, and the alive set
    alive = ps.join(fzt, on='round').filter((pl.col('tick') == pl.col('freeze')) & (pl.col('health') > 0))
    vf = pv.join(fzt, on='round').filter(pl.col('tick') == pl.col('freeze'))
    vmap = {(r_, p_): (t_, x_, y_) for r_, p_, t_, x_, y_ in vf.select('round', 'player_id', 'team_code', 'x_pos', 'y_pos').rows()}
    for r_, p_ in alive.select('round', 'player_id').rows():
        if r_ not in exp_rounds: continue
        tot['export_alive_rows_at_freeze'] += 1
        tot['export_alive_vector_team_null'] += int(vmap.get((r_, p_), (None,))[0] is None)
    # 4) round 19, match 27 (victim 0, killer 3 case) and same-side kills details
    codes = collections.defaultdict(list)
    for r_, p_, t_ in pi.select('round', 'player_id', 'team_code').rows(): codes[(r_, p_)].append(t_)
    for r_, fz in fzt.rows():
        if r_ not in exp_rounds: continue
        o = ovr.get((mk, r_))
        if o is None or o['status'] != 'fixed': continue
        A = {p_: pn for p_, pn in alive.filter(pl.col('round') == r_).select('player_id', 'place_name').rows()}
        ks = dth.filter((pl.col('round') == r_) & pl.col('attacker_id').is_not_null() & (pl.col('attacker_id') != 65535) & (pl.col('attacker_id') != pl.col('player_id')))
        for tk_, v_, k_, vt_, kt_, w_ in ks.select('tick', 'player_id', 'attacker_id', 'player_team_code', 'attacker_team_code', 'weapon_name').rows():
            if v_ in A and k_ in A and side_place(A[v_]) == side_place(A[k_]):
                focus.append((mk.split('/')[3], mk.split('/')[4][:8], r_, tk_, 'victim', v_, 'killer', k_, 'place', side_place(A[v_]), 'engine v/k', vt_, kt_, w_, 'roster v', codes.get((r_, v_)), 'roster k', codes.get((r_, k_)), 'dup', 0 in (v_, k_)))
        # pid 0 event team codes vs roster for this round
        tot['affected_rounds_seen'] += 1
    # 5) engine same-team kills in affected rounds: do they involve pid 0?
    for r_, fz in fzt.rows():
        if r_ not in exp_rounds: continue
        o = ovr.get((mk, r_))
        if o is None or o['status'] != 'fixed': continue
        ks = dth.filter((pl.col('round') == r_) & pl.col('attacker_team_code').is_not_null() & pl.col('player_team_code').is_not_null() & (pl.col('attacker_id') != 65535))
        for v_, k_, vt_, kt_ in ks.select('player_id', 'attacker_id', 'player_team_code', 'attacker_team_code').rows():
            if vt_ == kt_:
                tot['engine_same_team_in_affected'] += 1
                tot['engine_same_team_involving_pid0'] += int(v_ == 0 or k_ == 0)
            else:
                tot['engine_cross_team_in_affected'] += 1
    # 6) place 5v5 across all export rounds of this match
    for r_, fz in fzt.rows():
        if r_ not in exp_rounds: continue
        A = alive.filter(pl.col('round') == r_).select('place_name').to_series().to_list()
        if len(A) == 10:
            tot['export_round_10alive'] += 1
            tot['export_round_place_5v5'] += int(sum(1 for v in A if side_place(v) == 3) == 5 and sum(1 for v in A if side_place(v) == 2) == 5)
        tot['export_rounds_matched'] += 1
    log('done', mk.split('/')[3], mk.split('/')[4][:8], 'export rounds in match', len(exp_rounds))
log('TOTALS', dict(tot))
log('SAME-SIDE KILLS UNDER PLACE RULE IN 283 ROUNDS (n=%d)' % len(focus))
for f in focus: log('  ', f)
