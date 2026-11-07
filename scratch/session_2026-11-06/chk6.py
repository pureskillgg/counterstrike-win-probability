import csv, json, collections
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
B = '/home/sagemaker-user/adam/builder'
RAW = B + '/raw_cache_bucket/'
mk = 'csds/2026/06/11/m_4eb4f067ee36910523878933/csds'
folder = mk.rsplit('/', 1)[0]
R, FZ = 2, 5366
pv_t = pq.read_table(RAW + folder + '/player_vector', columns=['tick', 'round', 'player_id', 'team_code'])
print('pv dtype team_code', pv_t.schema.field('team_code').type, 'pid', pv_t.schema.field('player_id').type)
pv = pv_t.filter(pc.and_(pc.equal(pv_t['tick'], FZ), pc.equal(pv_t['round'], R))).to_pylist()
st_t = pq.read_table(RAW + folder + '/player_status', columns=['tick', 'round', 'player_id', 'health'])
st = st_t.filter(pc.and_(pc.equal(st_t['tick'], FZ), pc.equal(st_t['round'], R))).to_pylist()
pi = pq.read_table(RAW + folder + '/player_info', columns=['round', 'player_id', 'team_code']).to_pylist()
rr = [x for x in pi if x['round'] == R]
print('pv at freeze', [(x['player_id'], x['team_code']) for x in pv])
print('status at freeze', [(x['player_id'], x['health']) for x in st])
print('roster rows r2', [(x['player_id'], x['team_code']) for x in rr])
ov = [x for x in csv.DictReader(open(B + '/fixes/freeze_fix/override_rounds.csv')) if x['match_key'] == mk and x['round'] == str(R)]
print('override', ov[0]['alive'], ov[0]['dups'], ov[0]['unrostered'], ov[0]['override'])
