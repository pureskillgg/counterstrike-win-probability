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
export_rounds = {}
for mk in inv:
    m = idx[mk]
    export_rounds[mk] = set(int(x) for x in rnum[rmatch == m])
log('export rounds per investigated match, total', sum(len(v) for v in export_rounds.values()))
def get(mk, name):
    folder = mk.rsplit('/', 1)[0]
    return pl.read_parquet(io.BytesIO(s3.get_object(Bucket=B, Key='data/' + folder + '/' + name)['Body'].read()))
def side_place(s):
    return 3 if s == 'CTSpawn' else (2 if s == 'TSpawn' else None)
recs = []
cnt = collections.Counter()
att0 = collections.Counter()
dup_pid_all = collections.Counter()
for mk in inv:
    pi = get(mk, 'player_info').select('round', 'player_id', 'team_code').with_columns(pl.col('round').cast(pl.Int64), pl.col('player_id').cast(pl.Int64))
    ps = get(mk, 'player_status').select('tick', 'round', 'player_id', 'health', 'place_name').with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64), pl.col('player_id').cast(pl.Int64), pl.col('place_name').cast(pl.Utf8))
    pv = get(mk, 'player_vector').select('tick', 'round', 'player_id', 'team_code', 'x_pos', 'y_pos').with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64), pl.col('player_id').cast(pl.Int64))
    rs = get(mk, 'round_state').select('round', 'tick', 'event_type').with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64))
    dth = get(mk, 'player_death').select('round', 'tick', 'player_id', 'attacker_id', 'player_team_code', 'attacker_team_code', 'weapon_name').with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64), pl.col('player_id').cast(pl.Int64), pl.col('attacker_id').cast(pl.Int64))
    spw = get(mk, 'player_spawn').select('round', 'tick', 'player_id', 'player_team_code').with_columns(pl.col('round').cast(pl.Int64), pl.col('tick').cast(pl.Int64), pl.col('player_id').cast(pl.Int64))
    fzt = rs.filter(pl.col('event_type') == 'round_freeze_end').unique(subset=['round'], keep='first', maintain_order=True).select('round', pl.col('tick').alias('freeze'))
    codes = collections.defaultdict(list)
    for r_, p_, t_ in pi.rows(): codes[(r_, p_)].append(t_)
    for (r_, p_), lst in codes.items():
        if len(set(lst)) > 1 and r_ in export_rounds[mk]: dup_pid_all[p_] += 1
    # attacker_id 0 check
    z = dth.filter(pl.col('attacker_id') == 0)
    att0['attacker_id0_events_all'] += z.height
    for w_ in z['weapon_name'].to_list(): att0['w:' + str(w_)] += 1
    att0['attacker_null_events'] += dth.filter(pl.col('attacker_id').is_null()).height
    att0['attacker_65535_events'] += dth.filter(pl.col('attacker_id') == 65535).height
    alive = collections.defaultdict(dict)
    for r_, p_, pn in ps.join(fzt, on='round').filter((pl.col('tick') == pl.col('freeze')) & (pl.col('health') > 0)).select('round', 'player_id', 'place_name').rows():
        alive[r_][p_] = pn
    vecf = {(r_, p_): (t_, x_, y_) for r_, p_, t_, x_, y_ in pv.join(fzt, on='round').filter(pl.col('tick') == pl.col('freeze')).select('round', 'player_id', 'team_code', 'x_pos', 'y_pos').rows()}
    spawn = {(r_, p_): t_ for r_, p_, t_ in spw.drop_nulls('player_team_code').sort('tick').group_by(['round', 'player_id'], maintain_order=True).first().select('round', 'player_id', 'player_team_code').rows()}
    evf = pl.concat([dth.select('round', 'tick', pl.col('player_id').alias('pid'), pl.col('player_team_code').alias('team')),
                     dth.filter(pl.col('attacker_id').is_not_null() & (pl.col('attacker_id') != 65535)).select('round', 'tick', pl.col('attacker_id').alias('pid'), pl.col('attacker_team_code').alias('team'))])
    event = {(r_, p_): t_ for r_, p_, t_ in evf.drop_nulls('team').sort('tick').group_by(['round', 'pid'], maintain_order=True).first().select('round', 'pid', 'team').rows()}
    kills = collections.defaultdict(list)
    for r_, v_, k_, vt_, kt_ in dth.filter(pl.col('attacker_id').is_not_null() & (pl.col('attacker_id') != 65535) & (pl.col('attacker_id') != pl.col('player_id'))).select('round', 'player_id', 'attacker_id', 'player_team_code', 'attacker_team_code').rows():
        kills[r_].append((v_, k_, vt_, kt_))
    for r_, fz in fzt.rows():
        if r_ not in export_rounds[mk]:
            cnt['nonexport_rounds_skipped'] += 1
            continue
        A = alive.get(r_)
        if not A:
            cnt['export_round_no_alive'] += 1
            continue
        cds = {p: codes.get((r_, p), []) for p in A}
        dup = [p for p in A if len(set(cds[p])) > 1]
        unro = [p for p in A if len(cds[p]) == 0]
        pat = (len(dup), len(unro), len(A))
        o = ovr.get((mk, r_))
        ofix = o is not None and o['status'] == 'fixed'
        cnt['export_rounds'] += 1
        cnt['pattern %d-%d-%d' % pat] += 1
        recs.append({'mk': mk, 'r': r_, 'A': A, 'cds': cds, 'dup': dup, 'unro': unro, 'pat': pat, 'affected': ofix and pat == (1, 1, 10),
                     'vecf': {p: vecf.get((r_, p)) for p in A}, 'spawn': {p: spawn.get((r_, p)) for p in A},
                     'event': {p: event.get((r_, p)) for p in A}, 'kills': kills.get(r_, []), 'ovr': o})
log('export rounds analysed', len(recs))
for k in sorted(cnt): log('COUNT', k, cnt[k])
log('dup pids over export rounds (dup pid -> rounds)', dict(dup_pid_all))
log('attacker_id checks (all 23 matches)', dict(att0))
aff = [r for r in recs if r['affected']]
non = [r for r in recs if not r['affected']]
log('affected (283) =', len(aff), '| other export rounds =', len(non))
log('other export rounds with dup alive:', sum(1 for r in non if r['dup']), '| with unrostered alive:', sum(1 for r in non if r['unro']))
# ---- rule maps per round
def maps(r):
    A, cds, dup, unro = r['A'], r['cds'], r['dup'], r['unro']
    M = {}
    R1 = {p: (cds[p][0] if cds[p] else None) for p in A}
    M['R1_roster_first'] = R1
    R2 = dict(R1)
    if len(dup) == 1 and len(unro) == 1 and len(set(cds[dup[0]])) == 2:
        p0, u = dup[0], unro[0]; first = cds[p0][0]
        R2[p0] = first; R2[u] = [c for c in set(cds[p0]) if c != first][0]
    M['R2_roster_first_comp'] = R2
    M['R4_place_freeze'] = {p: side_place(A[p]) for p in A}
    M['R3_vector_freeze'] = {p: (r['vecf'][p][0] if r['vecf'][p] else None) for p in A}
    M['R5_spawn_first'] = {p: r['spawn'].get(p) for p in A}
    M['R6_event_first'] = {p: r['event'].get(p) for p in A}
    return M
def kill_stats(rs, name, only_involving=None):
    kc = ks = 0
    for r in rs:
        M = maps(r)[name]; A = r['A']
        for v, k, vt, kt in r['kills']:
            if v in A and k in A and M.get(v) is not None and M.get(k) is not None:
                if only_involving is not None and not (v in only_involving(r) or k in only_involving(r)): continue
                kc += 1
                if M[v] == M[k]: ks += 1
    return kc, ks
NAMES = ['R1_roster_first', 'R2_roster_first_comp', 'R3_vector_freeze', 'R4_place_freeze', 'R5_spawn_first', 'R6_event_first']
log('== KILL CONSISTENCY, export rounds only')
log('%-24s %-28s %8s %9s %9s' % ('subset', 'rule', 'kills_ck', 'same', 'cross%'))
for label, rs in [('export all', recs), ('affected 283', aff), ('other export', non)]:
    for name in NAMES:
        kc, ks = kill_stats(rs, name)
        log('%-24s %-28s %8d %9d %9s' % (label, name, kc, ks, ('%.2f' % (100.0 * (kc - ks) / kc)) if kc else 'na'))
# clean rounds (no dup, no unro) for apples-to-apples roster vs place
clean = [r for r in recs if not r['dup'] and not r['unro']]
log('clean export rounds (no dup, no unrostered)', len(clean))
for name in NAMES:
    kc, ks = kill_stats(clean, name)
    log('%-24s %-28s %8d %9d %9s' % ('clean export', name, kc, ks, ('%.2f' % (100.0 * (kc - ks) / kc)) if kc else 'na'))
# kills involving the two colliding slots in the 283 rounds
inv_slots = lambda r: set([r['dup'][0], r['unro'][0]]) if (r['dup'] and r['unro']) else set()
log('== kills involving the colliding pid 0 or unrostered slot, affected 283')
for name in NAMES:
    kc, ks = kill_stats(aff, name, only_involving=inv_slots)
    log('%-28s kills_ck %6d same %6d cross%% %s' % (name, kc, ks, ('%.2f' % (100.0 * (kc - ks) / kc)) if kc else 'na'))
# list same-side kills under place for the affected rounds
log('== same-side kills under R4 place rule in the 283 (mk-round-victim-killer-sides)')
shown = 0
for r in aff:
    M = maps(r)['R4_place_freeze']
    for v, k, vt, kt in r['kills']:
        if v in r['A'] and k in r['A'] and M.get(v) is not None and M.get(k) is not None and M[v] == M[k]:
            shown += 1
            if shown <= 12: log('  ', r['mk'].split('/')[3][:8], r['r'], 'victim', v, 'killer', k, 'side', M[v], 'engine team v/k', vt, kt, 'weapon?')
log('total same-side place kills in 283', shown)
# engine team codes: same-team fraction for death events with both codes, affected vs other
def eng(rs):
    t = s = 0
    for r in rs:
        for v, k, vt, kt in r['kills']:
            if vt is not None and kt is not None:
                t += 1; s += int(vt == kt)
    return t, s
log('engine same-team kills (affected 283):', eng(aff), '| other export:', eng(non))
# ---- field agreement for the colliding slots in the 283
log('== per-field agreement for pid0 (dup) and unrostered slot u, 283 rounds')
tab = collections.Counter()
for r in aff:
    p0, u = r['dup'][0], r['unro'][0]
    pl0 = side_place(r['A'][p0]); plu = side_place(r['A'][u])
    rf0 = r['cds'][p0][0]
    comp_u = [c for c in set(r['cds'][p0]) if c != rf0][0]
    v0 = r['vecf'][p0][0]; vu = r['vecf'][u][0] if r['vecf'][u] else None
    s0 = r['spawn'].get(p0); su = r['spawn'].get(u)
    e0 = r['event'].get(p0); eu = r['event'].get(u)
    tab['vec_p0 == roster_first_p0'] += int(v0 == rf0)
    tab['vec_p0 == place_p0'] += int(v0 == pl0)
    tab['vec_p0 null'] += int(v0 is None)
    tab['vec_u null'] += int(vu is None)
    tab['vec_u non-null'] += int(vu is not None)
    tab['spawn_p0 == roster_first_p0'] += int(s0 == rf0)
    tab['spawn_p0 == place_p0'] += int(s0 == pl0)
    tab['spawn_p0 null'] += int(s0 is None)
    tab['spawn_u non-null'] += int(su is not None)
    tab['event_p0 == place_p0'] += int(e0 == pl0)
    tab['event_p0 == roster_first_p0'] += int(e0 == rf0)
    tab['event_p0 null'] += int(e0 is None)
    tab['event_u non-null'] += int(eu is not None)
    tab['roster_first_p0 == place_p0'] += int(rf0 == pl0)
    tab['roster_comp_u == place_u'] += int(comp_u == plu)
    tab['place_p0 and place_u differ'] += int(pl0 != plu)
    tab['p0 has place'] += int(pl0 is not None)
    tab['u has place'] += int(plu is not None)
for k in sorted(tab): log('  %-34s %4d of %d' % (k, tab[k], len(aff)))
# positional check: freeze x,y of the 8 non-colliding players grouped by roster side; place p0 and u by nearest centroid
log('== positional side check at freeze (x,y centroid of 8 single-code players by roster side)')
pos = collections.Counter()
for r in aff:
    p0, u = r['dup'][0], r['unro'][0]
    ct = []; tt = []
    for p, info in r['vecf'].items():
        if p in (p0, u) or info is None: continue
        if r['cds'][p][0] == 3: ct.append((info[1], info[2]))
        elif r['cds'][p][0] == 2: tt.append((info[1], info[2]))
    if len(ct) < 2 or len(tt) < 2:
        pos['too_few_points'] += 1; continue
    cc = np.mean(np.array(ct), axis=0); tc = np.mean(np.array(tt), axis=0)
    def pside(pp):
        x = r['vecf'][pp]
        if x is None: return None
        d_ct = np.hypot(x[1] - cc[0], x[2] - cc[1]); d_t = np.hypot(x[1] - tc[0], x[2] - tc[1])
        return 3 if d_ct < d_t else 2
    pp0 = pside(p0); ppu = pside(u)
    pos['pos_p0 == place_p0'] += int(pp0 == side_place(r['A'][p0]))
    pos['pos_u == place_u'] += int(ppu == side_place(r['A'][u]))
    pos['pos_p0 == roster_first_p0'] += int(pp0 == r['cds'][p0][0])
    pos['centroid_gap_gt_300'] += int(np.hypot(cc[0] - tc[0], cc[1] - tc[1]) > 300)
    pos['rounds_checked'] += 1
for k in sorted(pos): log('  %-28s %d' % (k, pos[k]))
