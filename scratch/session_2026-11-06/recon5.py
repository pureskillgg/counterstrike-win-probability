import json, glob
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
F='/home/sagemaker-user/adam/builder/final'; E=F+'/export_clean_v2'; D='/home/sagemaker-user/adam/builder/data'
keys=json.load(open(E+'/match_keys.json')); ks=set(keys)
ms=np.load(E+'/match_split.npy')
names={0:'train',1:'val',2:'test'}
sm=json.load(open(F+'/split_manifest_v2.json'))
asg=sm['assign']
print('assign len', len(asg), '| assign split counts', Counter(asg.values()))
exp_split={k:names[int(ms[i])] for i,k in enumerate(keys)}
inA=[k for k in keys if k in asg]
print('export keys in assign', len(inA), '| export keys not in assign', len(keys)-len(inA))
mism=[k for k in inA if asg[k]!=exp_split[k]]
print('export vs assign split mismatches among shared keys', len(mism), mism[:2])
new_exp=[k for k in keys if k not in asg]
print('export keys not in assign by export split', Counter(exp_split[k] for k in new_exp))
out_asg=[k for k in asg if k not in ks]
print('assign keys not in export', len(out_asg), '| by assign split', Counter(asg[k] for k in out_asg))
hb=set(l.strip() for l in open(F+'/holdout_b_keys.txt', encoding='utf-8') if l.strip())
cm=pq.read_table(F+'/clean_matches.parquet', columns=['match_key','dup_keeper','in_scope','n_ok_rounds']).to_pydict()
usable=set(k for i,k in enumerate(cm['match_key']) if cm['dup_keeper'][i] and cm['in_scope'][i] and (cm['n_ok_rounds'][i] or 0)>0)
cmset=set(cm['match_key'])
c=Counter()
for k in out_asg:
    if k in hb: c['holdout_b']+=1
    elif k in usable: c['usable_not_export(empty47)']+=1
    elif k in cmset: c['in_clean_not_usable']+=1
    else: c['not_in_clean_nonholdout']+=1
print('assign-not-export breakdown', dict(c))
print('assign-not-export breakdown by split', Counter((asg[k], ('hb' if k in hb else ('usable' if k in usable else ('clean' if k in cmset else 'other')))) for k in out_asg))
# map names
mp={}
for cc in range(68):
    t=pq.read_table(f'{D}/c{cc:03d}/matches.parquet', columns=['match_key','map_name','match_date']).to_pydict()
    for k,m,d in zip(t['match_key'],t['map_name'],t['match_date']): mp[k]=(m,str(d)[:10])
lap=None
try:
    lt=pq.read_table(F+'/laptop_matches.parquet')
    print('laptop_matches cols', lt.schema.names[:20], 'rows', lt.num_rows)
except Exception as e:
    print('laptop read err', e)
absent_bucket=[k for k in out_asg if k in mp and k not in hb]
print('non-holdout absent (bucket universe) by split x map', Counter((asg[k], mp[k][0]) for k in absent_bucket if k not in usable).most_common(12))
print('holdout_b map counts', Counter(mp[k][0] for k in hb if k in mp).most_common(10))
