import csv, os, json, collections
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
B = '/home/sagemaker-user/adam/builder'
RAW = B + '/raw_cache_bucket/'
F = B + '/fixes/fix332'
cands = list(csv.DictReader(open(F + '/candidates.csv')))
am = sorted(set(r['match_key'] for r in cands))
chan = ['player_vector', 'player_status', 'player_info', 'player_blind']
missing = []
for mk in am:
    folder = mk.rsplit('/', 1)[0]
    for ch in chan:
        if not os.path.exists(RAW + folder + '/' + ch):
            missing.append((mk, ch))
print('affected matches', len(am), 'missing channel files', len(missing), missing[:4])

ov = list(csv.DictReader(open(B + '/fixes/freeze_fix/override_rounds.csv')))
fixed = [r for r in ov if r['status'] == 'fixed']
by_m = collections.defaultdict(list)
for r in fixed:
    by_m[r['match_key']].append(r)
print('fixed matches', len(by_m))

tot = collections.Counter()
per_round = []
for mk, rows in by_m.items():
    folder = mk.rsplit('/', 1)[0]
    frz = {int(r['round']): int(r['freeze']) for r in rows}
    ticks = pa.array(sorted(set(frz.values())), type=pa.int64())
    def load(name, cols):
        t = pq.read_table(RAW + folder + '/' + name, columns=cols)
        t = t.filter(pc.is_in(t['tick'], value_set=ticks)) if 'tick' in cols else t
        return t.to_pylist()
    pv = load('player_vector', ['tick', 'round', 'player_id', 'team_code'])
    st = load('player_status', ['tick', 'round', 'player_id', 'health'])
    pi = pq.read_table(RAW + folder + '/player_info', columns=['round', 'player_id', 'team_code']).to_pylist()
    # roster keep-last per (round, player_id), in file order
    ro = {}
    for x in pi:
        ro[(x['round'], x['player_id'])] = x['team_code']
    # player_vector keep-first per (round, tick, player_id)
    pvd = {}
    for x in pv:
        k = (x['round'], x['tick'], x['player_id'])
        if k not in pvd:
            pvd[k] = x['team_code']
    std = {}
    for x in st:
        k = (x['round'], x['tick'], x['player_id'])
        if k not in std:
            std[k] = x['health']
    for r, f in frz.items():
        ovr = [x for x in rows if int(x['round']) == r][0]
        ovm = json.loads(ovr['override'])
        keep_side = {}
        for (rr, tt, pid), tc in pvd.items():
            if rr != r or tt != f:
                continue
            h = std.get((rr, tt, pid))
            if h is None or not (h > 0):
                continue
            rt = ro.get((rr, pid))
            side = rt if rt is not None else tc
            if side in (2, 3):
                keep_side[pid] = side
        n = len(keep_side)
        ct = sum(1 for v in keep_side.values() if v == 3)
        t_ = sum(1 for v in keep_side.values() if v == 2)
        tot['rounds'] += 1
        tot['alive_eq_10'] += (n == 10)
        tot['5v5'] += (ct == 5 and t_ == 5)
        agree = sum(1 for p, s in keep_side.items() if str(s) == str(ovm.get(str(p))))
        tot['pid_agree_with_override'] += agree
        tot['pid_keep_total'] += n
        tot['pid_mismatch'] += sum(1 for p, s in keep_side.items() if str(s) != str(ovm.get(str(p))))
        tot['override_pid_missing_in_keep'] += sum(1 for p in ovm if int(p) not in keep_side)
        per_round.append((mk, r, n, ct, t_, agree))
print(dict(tot))
bad = [x for x in per_round if not (x[2] == 10 and x[3] == 5 and x[4] == 5)]
print('rounds not 10 alive 5v5 under keep-script side:', len(bad))
for x in bad[:6]:
    print(' ', x)
