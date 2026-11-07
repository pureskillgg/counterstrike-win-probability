import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, json
from collections import Counter
F='/home/sagemaker-user/adam/builder/final'
pf=pq.ParquetFile(F+'/clean_grid.parquet')
print('grid rows', pf.metadata.num_rows, 'cols', pf.schema_arrow.names)
cols=[c for c in ['match_key','round'] if c in pf.schema_arrow.names]
print('using cols', cols)
cm=pq.read_table(F+'/clean_rounds.parquet', columns=['match_key','round','use']).to_pydict()
use=set((k,r) for k,r,u in zip(cm['match_key'],cm['round'],cm['use']) if u)
keys=json.load(open('/home/sagemaker-user/adam/builder/final/export_clean_v2/match_keys.json')); ks=set(keys)
exp_pairs=set()
import numpy as np
rm=np.load('/home/sagemaker-user/adam/builder/final/export_clean_v2/round_match.npy'); rn=np.load('/home/sagemaker-user/adam/builder/final/export_clean_v2/round_num.npy')
exp_pairs=set(zip([keys[i] for i in rm.tolist()], rn.tolist()))
dropped=use-exp_pairs
print('dropped pairs', len(dropped))
t=pq.read_table(F+'/clean_grid.parquet', columns=cols)
mk=t.column('match_key'); rr=t.column('round')
# pair membership via join on string key
pk=pc.binary_join_element_wise(mk.cast(pa.string()), rr.cast(pa.string()), '|')
dropped_str=set(f'{k}|{r}' for k,r in dropped)
pk_list=pk.to_pylist()
inD=sum(1 for x in pk_list if x in dropped_str)
print('grid rows in dropped pairs', inD, '| grid rows total', len(pk_list))
print('grid rows in export pairs', sum(1 for x in pk_list if x.rsplit('|',1)[0] in ks and (x.rsplit('|',1)[0], int(x.rsplit('|',1)[1])) in exp_pairs))
