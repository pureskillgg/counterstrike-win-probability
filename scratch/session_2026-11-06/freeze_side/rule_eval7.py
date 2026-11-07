import io, json, csv, collections, sys, boto3, time
import numpy as np
import polars as pl
from botocore.config import Config
T0 = time.time()
def log(*a): print('[%4.0fs]' % (time.time() - T0), *a, file=sys.stderr, flush=True)
s3 = boto3.client('s3', config=Config(retries={'max_attempts': 8, 'mode': 'standard'}))
B = 'pureskillgg-csgo-production-sagemaker-0'
FIX = '/home/sagemaker-user/adam/builder/fixes'
E2 = '/home/sagemaker-user/adam/builder/final/export_clean_v2/'
E3 = '/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/'
E4 = '/home/sagemaker-user/adam/builder/final/export_clean_v4/'
inv = json.load(open(FIX + '/freeze_fix/inv_matches.json'))
ovr = {}
for r in csv.DictReader(open(FIX + '/freeze_fix/override_rounds.csv')):
    ovr[(r['match_key'], int(r['round']))] = r
mk_all = json.load(open(E2 + 'match_keys.json'))
rmatch = np.load(E2 + 'round_match.npy').astype(np.int64)
rnum = np.load(E2 + 'round_num.npy').astype(np.int64)
idx = {k: i for i, k in enumerate(mk_all)}
off = np.load(E3 + 'offsets.npy', mmap_mode='r')
lr2 = np.load(E2 + 'loadout_round.npy', mmap_mode='r')
lr4 = np.load(E4 + 'loadout_round.npy', mmap_mode='r')
ls4 = np.load(E4 + 'loadout_step_i16.npy', mmap_mode='r')
log('shapes', 'v2 loadout_round', lr2.shape, 'v4 loadout_round', lr4.shape, 'v4 loadout_step', ls4.shape, 'offsets', off.shape)
def get(mk, name):
    folder = mk.rsplit('/', 1)[0]
    return pl.read_parquet(io.BytesIO(s3.get_object(Bucket=B, Key='data/' + folder + '/' + name)['Body'].read()))
def side_place(s):
    return 3 if s == 'CTSpawn' else (2 if s == 'TSpawn' else None)
cnt = collections.Counter()
for mk in inv:
    ps = get(mk, 'player_status').select('tick', 'round', 'player_id', 'health', 'place_name').with_columns(pl.col('place_name').cast(pl.Utf8))
    pi = get(mk, 'player_info').select('round', 'player_id', 'team_code')
    rs = get(mk, 'round_state').select('round', 'tick', 'event_type')
    fzt = rs.filter(pl.col('event_type') == 'round_freeze_end').unique(subset=['round'], keep='first', maintain_order=True).select('round', pl.col('tick').alias('freeze'))
    codes = collections.defaultdict(list)
    for r_, p_, t_ in pi.rows(): codes[(r_, p_)].append(t_)
    fz = dict(fzt.rows())
    for (m_, rnd), o in [(k, v) for k, v in ovr.items() if k[0] == mk]:
        if o['status'] != 'fixed': continue
        r = int(np.nonzero((rmatch == idx[mk]) & (rnum == rnd))[0][0])
        f = fz[rnd]
        A = ps.filter((pl.col('round') == rnd) & (pl.col('tick') == f) & (pl.col('health') > 0)).select('player_id', 'health', 'place_name').rows()
        place_ct = sum(h for p, h, pn in A if side_place(pn) == 3)
        place_t = sum(h for p, h, pn in A if side_place(pn) == 2)
        # roster-first prediction (v3 rule): dup pid keeps first code, unrostered dropped
        rost_ct = rost_t = 0
        for p, h, pn in A:
            cs = codes.get((rnd, p), [])
            if not cs: continue
            c = cs[0]
            if c == 3: rost_ct += h
            elif c == 2: rost_t += h
        s0 = int(off[r])
        v4s_ct, v4s_t = int(ls4[s0, 0, 0]), int(ls4[s0, 1, 0])
        v4r_ct, v4r_t = int(lr4[r, 1]), int(lr4[r, 23])
        v2r_ct, v2r_t = int(lr2[r, 1]), int(lr2[r, 23])
        cnt['rounds'] += 1
        cnt['v4_step_CT_eq_place'] += int(v4s_ct == place_ct); cnt['v4_step_T_eq_place'] += int(v4s_t == place_t)
        cnt['v4_step_CT_eq_roster'] += int(v4s_ct == rost_ct); cnt['v4_step_T_eq_roster'] += int(v4s_t == rost_t)
        cnt['v4_round_CT_eq_place'] += int(v4r_ct == place_ct); cnt['v4_round_T_eq_place'] += int(v4r_t == place_t)
        cnt['v4_round_CT_eq_roster'] += int(v4r_ct == rost_ct); cnt['v4_round_T_eq_roster'] += int(v4r_t == rost_t)
        cnt['v2_round_CT_eq_place'] += int(v2r_ct == place_ct); cnt['v2_round_T_eq_place'] += int(v2r_t == place_t)
        cnt['v2_round_CT_eq_roster'] += int(v2r_ct == rost_ct); cnt['v2_round_T_eq_roster'] += int(v2r_t == rost_t)
        cnt['v4_step_rowsides_both_eq_place'] += int(v4s_ct == place_ct and v4s_t == place_t)
        cnt['v4_step_rowsides_both_eq_roster'] += int(v4s_ct == rost_ct and v4s_t == rost_t)
        cnt['v4_round_rowsides_both_eq_place'] += int(v4r_ct == place_ct and v4r_t == place_t)
        cnt['v2_round_rowsides_both_eq_place'] += int(v2r_ct == place_ct and v2r_t == place_t)
        cnt['v2_round_rowsides_both_eq_roster'] += int(v2r_ct == rost_ct and v2r_t == rost_t)
        cnt['place_ct_ne_rost_ct'] += int(place_ct != rost_ct)
log('CHECK of stored freeze hp (field 0 = hp) against place-based and roster-based sums, 283 fixed rounds')
for k in sorted(cnt): log('  %-34s %d' % (k, cnt[k]))
log('loadout_round field index 1 used as hp (index 0 = alive) per f1 layout; v4 round rows read at export index r')
