import io, json, csv, collections, sys, boto3, time
import polars as pl
from botocore.config import Config
T0 = time.time()
def log(*a): print('[%4.0fs]' % (time.time() - T0), *a, file=sys.stderr, flush=True)
s3 = boto3.client('s3', config=Config(retries={'max_attempts': 8, 'mode': 'standard'}))
B = 'pureskillgg-csgo-production-sagemaker-0'
FIX = '/home/sagemaker-user/adam/builder/fixes'
inv = json.load(open(FIX + '/freeze_fix/inv_matches.json'))
ovr = {}
for r in csv.DictReader(open(FIX + '/freeze_fix/override_rounds.csv')):
    ovr[(r['match_key'], int(r['round']))] = r
def get(mk, name):
    folder = mk.rsplit('/', 1)[0]
    return pl.read_parquet(io.BytesIO(s3.get_object(Bucket=B, Key='data/' + folder + '/' + name)['Body'].read()))
def side_place(s):
    return 3 if s == 'CTSpawn' else (2 if s == 'TSpawn' else None)
RULES = ['R1_roster_first', 'R2_roster_first_comp', 'R3_vector_freeze', 'R4_place_freeze', 'R5_spawn_first',
         'R6_event_first', 'R7_override_place', 'R8_vector_for_colliding', 'R9_spawn_for_colliding', 'R10_event_for_colliding']
recs = []
eng = collections.Counter()
cnt = collections.Counter()
csv_rows = []
for mk in inv:
    pi = get(mk, 'player_info').select('round', 'player_id', 'team_code').with_columns(pl.col('round').cast(pl.Int64), pl.col('player_id').cast(pl.Int64))
    ps = get(mk, 'player_status').select('tick', 'round', 'player_id', 'health', 'place_name').with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64), pl.col('player_id').cast(pl.Int64), pl.col('place_name').cast(pl.Utf8))
    pv = get(mk, 'player_vector').select('tick', 'round', 'player_id', 'team_code').with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64), pl.col('player_id').cast(pl.Int64))
    rs = get(mk, 'round_state').select('round', 'tick', 'event_type').with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64))
    dth = get(mk, 'player_death').select('round', 'tick', 'player_id', 'attacker_id', 'player_team_code', 'attacker_team_code').with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64), pl.col('player_id').cast(pl.Int64), pl.col('attacker_id').cast(pl.Int64))
    spw = get(mk, 'player_spawn').select('round', 'tick', 'player_id', 'player_team_code').with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64), pl.col('player_id').cast(pl.Int64))
    fzt = rs.filter(pl.col('event_type') == 'round_freeze_end').unique(subset=['round'], keep='first', maintain_order=True).select('round', pl.col('tick').alias('freeze'))
    codes = collections.defaultdict(list)
    for r_, p_, t_ in pi.rows(): codes[(r_, p_)].append(t_)
    dup_all = collections.defaultdict(list)
    for (r_, p_), lst in codes.items():
        if len(set(lst)) > 1: dup_all[r_].append(p_)
    alive = collections.defaultdict(dict)
    for r_, p_, pn in ps.join(fzt, on='round').filter((pl.col('tick') == pl.col('freeze')) & (pl.col('health') > 0)).select('round', 'player_id', 'place_name').rows():
        alive[r_][p_] = pn
    vec = {(r_, p_): t_ for r_, p_, t_ in pv.join(fzt, on='round').filter(pl.col('tick') == pl.col('freeze')).select('round', 'player_id', 'team_code').rows()}
    spawn = {(r_, p_): t_ for r_, p_, t_ in spw.drop_nulls('player_team_code').sort('tick').group_by(['round', 'player_id'], maintain_order=True).first().select('round', 'player_id', 'player_team_code').rows()}
    evf = pl.concat([dth.select('round', 'tick', pl.col('player_id').alias('pid'), pl.col('player_team_code').alias('team')),
                     dth.filter(pl.col('attacker_id').is_not_null() & (pl.col('attacker_id') != 65535)).select('round', 'tick', pl.col('attacker_id').alias('pid'), pl.col('attacker_team_code').alias('team'))])
    event = {(r_, p_): t_ for r_, p_, t_ in evf.drop_nulls('team').sort('tick').group_by(['round', 'pid'], maintain_order=True).first().select('round', 'pid', 'team').rows()}
    kills = collections.defaultdict(list)
    for r_, v_, k_ in dth.filter(pl.col('attacker_id').is_not_null() & (pl.col('attacker_id') != 65535) & (pl.col('attacker_id') != pl.col('player_id'))).select('round', 'player_id', 'attacker_id').rows():
        kills[r_].append((v_, k_))
    tk = dth.filter(pl.col('attacker_team_code').is_not_null() & pl.col('player_team_code').is_not_null())
    eng['engine_pairs'] += tk.height
    eng['engine_same_team'] += tk.filter(pl.col('attacker_team_code') == pl.col('player_team_code')).height
    for r_, fz in fzt.rows():
        A = alive.get(r_)
        if not A:
            cnt['round_no_alive_at_freeze'] += 1
            continue
        cds = {p: codes.get((r_, p), []) for p in A}
        dup = [p for p in A if len(set(cds[p])) > 1]
        unro = [p for p in A if len(cds[p]) == 0]
        pat = (len(dup), len(unro), len(A))
        o = ovr.get((mk, r_))
        ofix = o is not None and o['status'] == 'fixed'
        cnt['rounds'] += 1
        if dup_all.get(r_) and not any(p in A for p in dup_all[r_]): cnt['dup_pair_round_dup_not_alive'] += 1
        R = {}
        R1 = {p: (cds[p][0] if cds[p] else None) for p in A}
        R['R1_roster_first'] = R1
        R2 = dict(R1)
        if pat[:2] == (1, 1):
            p0, u = dup[0], unro[0]
            s = set(cds[p0])
            if len(s) == 2:
                first = cds[p0][0]
                R2[p0] = first; R2[u] = [c for c in s if c != first][0]
        R['R2_roster_first_comp'] = R2
        R['R3_vector_freeze'] = {p: vec.get((r_, p)) for p in A}
        R['R4_place_freeze'] = {p: side_place(A[p]) for p in A}
        R['R5_spawn_first'] = {p: spawn.get((r_, p)) for p in A}
        R['R6_event_first'] = {p: event.get((r_, p)) for p in A}
        R7 = dict(R1)
        if pat == (1, 1, 10) and ofix:
            p0, u = dup[0], unro[0]
            s = set(cds[p0]); sp0 = side_place(A[p0])
            if sp0 in s and len(s) == 2:
                R7[p0] = sp0; R7[u] = [c for c in s if c != sp0][0]
            else:
                R7[p0] = None; R7[u] = None
        R['R7_override_place'] = R7
        for name, fld in [('R8_vector_for_colliding', vec), ('R9_spawn_for_colliding', spawn), ('R10_event_for_colliding', event)]:
            M = dict(R1)
            if pat[:2] == (1, 1) and len(set(cds[dup[0]])) == 2:
                p0, u = dup[0], unro[0]
                s = set(cds[p0]); x = fld.get((r_, p0))
                if x in s:
                    M[p0] = x; M[u] = [c for c in s if c != x][0]
                else:
                    M[p0] = None; M[u] = None
            R[name] = M
        affected = (pat == (1, 1, 10) and ofix)
        recs.append({'mk': mk, 'r': r_, 'fz': fz, 'A': A, 'cds': cds, 'dup': dup, 'unro': unro, 'pat': pat, 'R': R,
                     'kills': kills.get(r_, []), 'affected': affected, 'ofix': ofix, 'ovr': o})
        if o is not None:
            p0 = dup[0] if dup else None
            u = unro[0] if unro else None
            ovj = json.loads(o['override']) if o['override'] else {}
            csv_rows.append([mk, r_, fz, '%d-%d-%d' % pat, p0, ' '.join(str(c) for c in (cds[p0] if p0 is not None else [])), u,
                             (A.get(p0, '') if p0 is not None else ''), (A.get(u, '') if u is not None else ''),
                             (R1.get(p0) if p0 is not None else ''), (R2.get(u) if u is not None else ''),
                             (vec.get((r_, p0)) if p0 is not None else ''), (vec.get((r_, u)) if u is not None else ''),
                             (spawn.get((r_, p0)) if p0 is not None else ''), (spawn.get((r_, u)) if u is not None else ''),
                             (event.get((r_, p0)) if p0 is not None else ''), (event.get((r_, u)) if u is not None else ''),
                             (ovj.get(str(p0), '') if p0 is not None else ''), (ovj.get(str(u), '') if u is not None else ''),
                             o['status'], o['override'] or ''])
    log('done', mk.split('/')[3], mk.split('/')[4][:8], 'rounds so far', len(recs))
w = csv.writer(sys.stdout)
w.writerow(['match_key', 'round', 'freeze', 'pattern_dup_unro_alive', 'dup_pid', 'dup_codes_file_order', 'unro_pid',
            'place_dup', 'place_unro', 'roster_first_dup', 'roster_comp_unro', 'vector_team_freeze_dup', 'vector_team_freeze_unro',
            'spawn_team_dup', 'spawn_team_unro', 'first_event_team_dup', 'first_event_team_unro', 'override_side_dup', 'override_side_unro', 'override_status', 'override_json'])
w.writerows(csv_rows)
log('COUNTERS', dict(cnt))
log('ENGINE same-team kills among death events with both codes', dict(eng))
def metrics(sel):
    out = {}
    for name in RULES:
        n = res = f55 = kc = ks = 0
        for rec in recs:
            if not sel(rec): continue
            n += 1
            M = rec['R'][name]; A = rec['A']
            if all(M.get(p) is not None for p in A):
                res += 1
                if sum(1 for p in A if M[p] == 3) == 5 and sum(1 for p in A if M[p] == 2) == 5: f55 += 1
            for v, k in rec['kills']:
                if v in A and k in A and M.get(v) is not None and M.get(k) is not None:
                    kc += 1
                    if M[v] == M[k]: ks += 1
        out[name] = (n, res, f55, kc, ks)
    return out
def show(title, m):
    log('==', title)
    log('%-28s %6s %8s %8s %9s %9s %9s' % ('rule', 'rounds', 'resolved', '5v5', 'kills_ck', 'same_side', 'cross%'))
    for name in RULES:
        n, res, f55, kc, ks = m[name]
        cross = ('%.2f' % (100.0 * (kc - ks) / kc)) if kc else 'na'
        log('%-28s %6d %8d %8d %9d %9d %9s' % (name, n, res, f55, kc, ks, cross))
show('ALL rounds in 23 matches (rule applied to every round, no fallback)', metrics(lambda r: True))
show('AFFECTED 283 rounds (1 dup + 1 unrostered + 10 alive, override fixed)', metrics(lambda r: r['affected']))
show('NON-affected rounds', metrics(lambda r: not r['affected']))
log('== agreement with place_name at freeze on the 283 affected rounds (p0 = dup pid, u = unrostered pid)')
log('%-28s %8s %8s %8s | %8s %8s %8s' % ('rule', 'p0_agree', 'p0_dis', 'p0_none', 'u_agree', 'u_dis', 'u_none'))
for name in RULES:
    a0 = d0 = n0 = au = du = nu = 0
    for rec in recs:
        if not rec['affected']: continue
        p0, u = rec['dup'][0], rec['unro'][0]
        M = rec['R'][name]
        pp0 = side_place(rec['A'][p0]); ppu = side_place(rec['A'][u])
        if M.get(p0) is None: n0 += 1
        elif M[p0] == pp0: a0 += 1
        else: d0 += 1
        if M.get(u) is None: nu += 1
        elif M[u] == ppu: au += 1
        else: du += 1
    log('%-28s %8d %8d %8d | %8d %8d %8d' % (name, a0, d0, n0, au, du, nu))
aff = [r for r in recs if r['affected']]
log('affected rounds', len(aff), '| dup pid values', collections.Counter(r['dup'][0] for r in aff).most_common(3),
    '| unro pids', collections.Counter(r['unro'][0] for r in aff).most_common(10))
log('p0 place side among its two codes', sum(1 for r in aff if side_place(r['A'][r['dup'][0]]) in set(r['cds'][r['dup'][0]])), 'of', len(aff))
log('p0 distinct codes count', collections.Counter(len(set(r['cds'][r['dup'][0]])) for r in aff))
log('place_name values at freeze, alive rows of 283 rounds', collections.Counter(v for r in aff for v in r['A'].values()))
log('place_name values at freeze, alive rows of other rounds', collections.Counter(v for r in recs if not r['affected'] for v in r['A'].values()))
log('rounds with any unrostered alive player', sum(1 for r in recs if r['unro']), '| with any dup alive', sum(1 for r in recs if r['dup']))
log('rounds of 10 alive with place 5 CT / 5 T', sum(1 for r in recs if len(r['A']) == 10 and sum(1 for v in r['A'].values() if side_place(v) == 3) == 5 and sum(1 for v in r['A'].values() if side_place(v) == 2) == 5), 'of', len(recs))
log('R7 override mismatches vs override JSON', sum(1 for r in aff for p, s in json.loads(r['ovr']['override']).items() if r['R']['R7_override_place'].get(int(p)) != s))
