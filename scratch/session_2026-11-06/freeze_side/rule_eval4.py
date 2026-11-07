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
SIG = ['radar_color_code', 'rank', 'rank_raw', 'rank_platform']
tot = collections.Counter()
per_match = []
for mk in inv:
    exp_rounds = set(int(x) for x in rnum[rmatch == idx[mk]])
    pi = get(mk, 'player_info').select('round', 'player_id', 'team_code', 'wins', 'rank', 'radar_color_code', 'rank_raw', 'rank_platform', 'player_id_fixed')
    ps = get(mk, 'player_status').select('tick', 'round', 'player_id', 'health', 'place_name').with_columns(pl.col('place_name').cast(pl.Utf8))
    rs = get(mk, 'round_state').select('round', 'tick', 'event_type')
    fzt = rs.filter(pl.col('event_type') == 'round_freeze_end').unique(subset=['round'], keep='first', maintain_order=True).select('round', pl.col('tick').alias('freeze'))
    cnt_rows = pi.group_by(['round', 'player_id']).len()
    dupk = set(map(tuple, cnt_rows.filter(pl.col('len') > 1).select('round', 'player_id').rows()))
    singles = pi.filter(~pl.struct(['round', 'player_id']).is_in([{'round': a, 'player_id': b} for a, b in dupk])) if dupk else pi
    # per-pid constancy of signature fields across rounds (single-row rounds only)
    by_pid = collections.defaultdict(list)
    for row in singles.select('round', 'player_id', 'wins', *SIG).rows():
        by_pid[row[1]].append(row)
    const = collections.Counter()
    for p_, rows in by_pid.items():
        for i, f in enumerate(['wins'] + SIG):
            vals = set(r_[2 + i] for r_ in rows)
            const[f + ('_const' if len(vals) == 1 else '_varies')] += 1
    per_match.append((mk, dict(const)))
    # affected rounds: the two pid-0 rows and the unrostered slot
    alive = ps.join(fzt, on='round').filter((pl.col('tick') == pl.col('freeze')) & (pl.col('health') > 0)).select('round', 'player_id', 'place_name').rows()
    A = collections.defaultdict(dict)
    for r_, p_, pn in alive: A[r_][p_] = pn
    rows_by_round = collections.defaultdict(list)
    for r_, p_, t_, w_, rk, rc, rr, rp, pf in pi.select('round', 'player_id', 'team_code', 'wins', 'rank', 'radar_color_code', 'rank_raw', 'rank_platform', 'player_id_fixed').rows():
        rows_by_round[(r_, p_)].append((t_, w_, rk, rc, rr, rp, pf))
    for r_, fz in fzt.rows():
        if r_ not in exp_rounds: continue
        o = ovr.get((mk, r_))
        if o is None or o['status'] != 'fixed': continue
        if r_ not in A: continue
        p0 = 0
        u = [p for p in A[r_] if len(rows_by_round.get((r_, p), [])) == 0]
        if not u or len(rows_by_round.get((r_, p0), [])) != 2: continue
        u = u[0]
        # u signature from u's rows in other rounds (single-row rounds)
        usig = collections.Counter()
        for (rr_, pp_), lst in rows_by_round.items():
            if pp_ == u and rr_ != r_ and len(lst) == 1:
                usig[tuple(lst[0][3:6])] += 1
        p0sig = collections.Counter()
        for (rr_, pp_), lst in rows_by_round.items():
            if pp_ == p0 and rr_ != r_ and len(lst) == 1:
                p0sig[tuple(lst[0][3:6])] += 1
        pair = rows_by_round[(r_, p0)]
        tot['affected_rounds'] += 1
        tot['u_has_other_round_rows'] += int(sum(usig.values()) > 0)
        tot['p0_single_rows_other_rounds'] += int(sum(p0sig.values()) > 0)
        tot['u_sig_unique'] += int(len(usig) == 1)
        # field equality between the two pid-0 rows
        tot['pair_wins_differ'] += int(pair[0][1] != pair[1][1])
        tot['pair_rank_differ'] += int(pair[0][2] != pair[1][2])
        tot['pair_radar_differ'] += int(pair[0][3] != pair[1][3])
        tot['pair_rankraw_differ'] += int(pair[0][4] != pair[1][4])
        tot['pair_rankplat_differ'] += int(pair[0][5] != pair[1][5])
        tot['pair_fixed_differ'] += int(pair[0][6] != pair[1][6])
        tot['pair_radar_or_rank_differ'] += int(pair[0][2:6] != pair[1][2:6])
        # match rows to u's signature
        if usig:
            us = usig.most_common(1)[0][0]
            hit = [row for row in pair if tuple(row[3:6]) == us]
            tot['u_sig_matches_exactly_one_pair_row'] += int(len(hit) == 1)
            if len(hit) == 1:
                tot['u_side_from_sig_match'] += 1
                tot['u_side_from_sig_eq_place_u'] += int(hit[0][0] == side_place(A[r_][u]))
                tot['u_side_from_sig_eq_place_u_denom'] += 1
        # per-round sanity: place side of p0 among the two codes
        tot['p0_place_among_codes'] += int(side_place(A[r_][p0]) in set(row[0] for row in pair))
        log('  round', mk.split('/')[3], r_, 'u', u, 'pair', [(row[0], row[1], row[2], row[3], row[4], row[5], row[6]) for row in pair], 'u_sig', dict(usig), 'place p0', A[r_][p0], 'place u', A[r_][u]) if tot['affected_rounds'] <= 6 else None
log('CONSTANCY OF PER-PLAYER INFO FIELDS ACROSS ROUNDS (single-row rounds, per match, counted over players)')
agg = collections.Counter()
for mk, c in per_match:
    for k, v in c.items(): agg[k] += v
for k in sorted(agg): log('  %-22s %d' % (k, agg[k]))
log('AFFECTED-ROUND PAIR CHECKS')
for k in sorted(tot): log('  %-36s %d' % (k, tot[k]))
