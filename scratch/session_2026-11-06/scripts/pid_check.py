import sys, json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
base, split = sys.argv[1], sys.argv[2]
assign = json.load(open(split, encoding='utf-8'))['assign']
tv = [k for k, v in assign.items() if v in ('train', 'val')]
print('train+val match keys:', len(tv), '| test keys (excluded):', sum(1 for v in assign.values() if v == 'test'))
t = pq.read_table(f"{base}/deaths.parquet", columns=['match_key', 'player_id'], filters=[('match_key', 'in', tv)])
print('deaths rows (train+val):', t.num_rows)
print('distinct player_id:', pc.count_distinct(t['player_id']).as_py())
arr = t['player_id'].to_numpy()
print('nonpositive values:', int((arr <= 0).sum()))
pos = arr[arr > 0]
digits = np.floor(np.log10(pos)).astype(int) + 1
vals, cnts = np.unique(digits, return_counts=True)
print('digit-length histogram (digits: rows):', dict(zip(vals.tolist(), cnts.tolist())))
steam = ((arr >= 76561197960265728) & (arr < 76561202255233024)).sum()
print('rows in Steam64 individual-account range:', int(steam))
print('rows < 10000 (slot/index-like):', int((arr < 10000).sum()))
print('max bit length:', int(np.ceil(np.log2(arr.max()+1))) if arr.size else 0)
g = t.group_by('player_id').aggregate([('match_key', 'count_distinct')])
mc = g.column('match_key_count_distinct').to_numpy()
print('distinct players:', mc.size)
bins = {'1 match': int((mc == 1).sum()), '2-5 matches': int(((mc >= 2) & (mc <= 5)).sum()), '6-50 matches': int(((mc >= 6) & (mc <= 50)).sum()), '>50 matches': int((mc > 50).sum())}
print('players by number of train+val matches they appear in:', bins)
print('median matches per player:', float(np.median(mc)), '| max:', int(mc.max()))
pm = t.group_by('match_key').aggregate([('player_id', 'count_distinct')])
pmc = pm.column('player_id_count_distinct').to_numpy()
u, c = np.unique(pmc, return_counts=True)
print('distinct player_id per match (count: matches):', dict(zip(u.tolist(), c.tolist())))
