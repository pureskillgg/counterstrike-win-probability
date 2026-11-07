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
dup_pid_raw = collections.Counter()
unro_per_match = {}
sig_kc = sig_ks = r4_kc_same_rounds = r4_ks_same_rounds = 0
sig_5v5 = sig_res = 0
for mk in inv:
    exp_rounds = set(int(x) for x in rnum[rmatch == idx[mk]])
    pi = get(mk, 'player_info').select('round', 'player_id', 'team_code', 'wins', 'rank', 'radar_color_code', 'rank_raw', 'rank_platform')
    ps = get(mk, 'player_status').select('tick', 'round', 'player_id', 'health', 'place_name').with_columns(pl.col('place_name').cast(pl.Utf8))
    rs = get(mk, 'round_state').select('round', 'tick', 'event_type')
    dth = get(mk, 'player_death').select('round', 'tick', 'player_id', 'attacker_id', 'player_team_code', 'attacker_team_code').with_columns(pl.col('attacker_id').cast(pl.Int64))
    fzt = rs.filter(pl.col('event_type') == 'round_freeze_end').unique(subset=['round'], keep='first', maintain_order=True).select('round', pl.col('tick').alias('freeze'))
    rows = collections.defaultdict(list)
    for r_, p_, t_, w_, rk, rc, rr, rp in pi.select('round', 'player_id', 'team_code', 'wins', 'rank', 'radar_color_code', 'rank_raw', 'rank_platform').rows():
        rows[(r_, p_)].append((t_, w_, rk, rc, rr, rp))
        if len(rows[(r_, p_)]) == 2: dup_pid_raw[p_] += 1
    info_pids = set(p for (_, p) in rows.keys())
    alive = collections.defaultdict(dict)
    for r_, p_, pn in ps.join(fzt, on='round').filter((pl.col('tick') == pl.col('freeze')) & (pl.col('health') > 0)).select('round', 'player_id', 'place_name').rows():
        alive[r_][p_] = pn
    status_pids = set(p for r_ in alive for p in alive[r_])
    unro_per_match[mk.split('/')[3] + '-' + mk.split('/')[4][:8]] = sorted(status_pids - info_pids)
    # pid-0 signature from single-row rounds
    single_sig = collections.Counter()
    for (r_, p_), lst in rows.items():
        if p_ == 0 and len(lst) == 1 and r_ in alive and 0 in alive[r_]:
            single_sig[tuple(lst[0][1:6])] += 1
    sig0 = single_sig.most_common(1)[0][0] if single_sig else None
    # kill data
    kills = collections.defaultdict(list)
    for r_, v_, k_, vt_, kt_ in dth.filter(pl.col('attacker_id').is_not_null() & (pl.col('attacker_id') != 65535) & (pl.col('attacker_id') != pl.col('player_id'))).select('round', 'player_id', 'attacker_id', 'player_team_code', 'attacker_team_code').rows():
        kills[r_].append((v_, k_))
    for r_, fz in fzt.rows():
        if r_ not in exp_rounds: continue
        o = ovr.get((mk, r_))
        if o is None or o['status'] != 'fixed': continue
        A = alive.get(r_, {})
        if 0 not in A or sig0 is None: continue
        lst = rows.get((r_, 0), [])
        if len(lst) != 2: continue
        hit = [row for row in lst if tuple(row[1:6]) == sig0]
        if len(hit) != 1: continue
        u = [p for p in A if len(rows.get((r_, p), [])) == 0]
        if len(u) != 1: continue
        u = u[0]
        # sides under the signature rule: slot 0 and u from the pair, others from roster
        M = {p: rows[(r_, p)][0][0] if rows.get((r_, p)) else None for p in A}
        M[0] = hit[0][0]
        M[u] = [row for row in lst if tuple(row[1:6]) != sig0][0][0]
        sig_res += 1
        if all(M.get(p) is not None for p in A):
            if sum(1 for p in A if M[p] == 3) == 5 and sum(1 for p in A if M[p] == 2) == 5: sig_5v5 += 1
        for v_, k_ in kills.get(r_, []):
            if v_ in A and k_ in A and M.get(v_) is not None and M.get(k_) is not None:
                sig_kc += 1; sig_ks += int(M[v_] == M[k_])
        # R4 (place) on the same rounds for comparison
        P = {p: side_place(A[p]) for p in A}
        for v_, k_ in kills.get(r_, []):
            if v_ in A and k_ in A and P.get(v_) is not None and P.get(k_) is not None:
                r4_kc_same_rounds += 1; r4_ks_same_rounds += int(P[v_] == P[k_])
        tot['sig_slot_eq_place_p0'] += int(M[0] == side_place(A[0]))
        tot['sig_slot_eq_place_u'] += int(M[u] == side_place(A[u]))
log('RAW dup pid distribution over all 443 pair-rounds (pid -> count)', dict(dup_pid_raw))
log('unrostered-in-info pid per match (status pid absent from player_info, all rounds)', unro_per_match)
log('signature rule on 91 resolved affected rounds: rounds', sig_res, '| 5v5', sig_5v5,
    '| kills checked', sig_kc, '| same-side', sig_ks, '| cross%', ('%.2f' % (100.0 * (sig_kc - sig_ks) / sig_kc)) if sig_kc else 'na')
log('place rule R4 on the SAME rounds: kills checked', r4_kc_same_rounds, '| same-side', r4_ks_same_rounds,
    '| cross%', ('%.2f' % (100.0 * (r4_kc_same_rounds - r4_ks_same_rounds) / r4_kc_same_rounds)) if r4_kc_same_rounds else 'na')
log('slot sides vs place on those rounds', dict(tot))
