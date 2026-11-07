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
for mk in inv:
    exp_rounds = set(int(x) for x in rnum[rmatch == idx[mk]])
    pi = get(mk, 'player_info').select('round', 'player_id', 'team_code', 'wins', 'rank', 'radar_color_code', 'rank_raw', 'rank_platform')
    ps = get(mk, 'player_status').select('tick', 'round', 'player_id', 'health', 'place_name').with_columns(pl.col('place_name').cast(pl.Utf8))
    rs = get(mk, 'round_state').select('round', 'tick', 'event_type')
    fzt = rs.filter(pl.col('event_type') == 'round_freeze_end').unique(subset=['round'], keep='first', maintain_order=True).select('round', pl.col('tick').alias('freeze'))
    info_pids = set(pi['player_id'].unique().to_list())
    status_pids = set(ps['player_id'].unique().to_list())
    tot['status_pids_absent_from_info_all_rounds'] += len(status_pids - info_pids)
    tot['matches'] += 1
    rows = collections.defaultdict(list)
    for r_, p_, t_, w_, rk, rc, rr, rp in pi.rows():
        rows[(r_, p_)].append((t_, w_, rk, rc, rr, rp))
    # signature of pid 0 from its single-row rounds, verified against place_name in those rounds
    alive = collections.defaultdict(dict)
    for r_, p_, pn in ps.join(fzt, on='round').filter((pl.col('tick') == pl.col('freeze')) & (pl.col('health') > 0)).select('round', 'player_id', 'place_name').rows():
        alive[r_][p_] = pn
    single_sig = collections.Counter()
    single_team_vs_place = collections.Counter()
    for (r_, p_), lst in rows.items():
        if p_ != 0 or len(lst) != 1: continue
        if r_ not in alive or 0 not in alive[r_]: continue
        sig = tuple(lst[0][1:6])
        single_sig[sig] += 1
        single_team_vs_place[int(lst[0][0] == side_place(alive[r_][0]))] += 1
    tot['pid0_single_rounds_with_place'] += sum(single_team_vs_place.values())
    tot['pid0_single_row_team_eq_place'] += single_team_vs_place[1]
    tot['pid0_single_row_team_ne_place'] += single_team_vs_place[0]
    tot['pid0_distinct_signatures_in_single_rounds'] += len(single_sig)
    sig0 = single_sig.most_common(1)[0][0] if single_sig else None
    if sig0 is not None: tot['matches_with_pid0_signature'] += 1
    # the non-p0 signature of the pid-0 pair must be stable over the match
    pairsig_other = collections.Counter()
    for (r_, p_), lst in rows.items():
        if p_ == 0 and len(lst) == 2:
            for row in lst:
                if sig0 is None or tuple(row[1:6]) != sig0:
                    pairsig_other[tuple(row[1:6])] += 1
    tot['matches_pair_nonp0_row_stable_sig'] += int(len(pairsig_other) == 1)
    for r_, fz in fzt.rows():
        if r_ not in exp_rounds: continue
        o = ovr.get((mk, r_))
        if o is None or o['status'] != 'fixed': continue
        lst = rows.get((r_, 0), [])
        A = alive.get(r_, {})
        if len(lst) != 2 or 0 not in A: continue
        u = [p for p in A if len(rows.get((r_, p), [])) == 0][0]
        tot['affected_rounds'] += 1
        if sig0 is None:
            tot['unresolved_no_pid0_signature'] += 1
            continue
        hit = [row for row in lst if tuple(row[1:6]) == sig0]
        if len(hit) != 1:
            tot['sig_hits_not_exactly_one'] += 1
            continue
        p0_team = hit[0][0]
        u_team = [row for row in lst if tuple(row[1:6]) != sig0][0][0]
        tot['sig_resolved'] += 1
        tot['sig_p0_eq_place'] += int(p0_team == side_place(A[0]))
        tot['sig_u_eq_place'] += int(u_team == side_place(A[u]))
        tot['sig_pair_both_eq_place'] += int(p0_team == side_place(A[0]) and u_team == side_place(A[u]))
    log('  match', mk.split('/')[3], mk.split('/')[4][:8], 'sig0', sig0 is not None, 'single rounds', sum(single_sig.values()), 'signatures', len(single_sig))
log('TOTALS', dict(tot))
