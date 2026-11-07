import json, glob, os
from collections import Counter, defaultdict
import pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc
D='/home/sagemaker-user/adam/builder/data'; F='/home/sagemaker-user/adam/builder/final'; E=F+'/export_clean_v2'
keys=json.load(open(E+'/match_keys.json')); ks=set(keys)
cm=pq.read_table(F+'/clean_matches.parquet').to_pydict()
rows={k:i for i,k in enumerate(cm['match_key'])}
usable=set(k for i,k in enumerate(cm['match_key']) if cm['dup_keeper'][i] and cm['in_scope'][i] and (cm['n_ok_rounds'][i] or 0)>0)
print('usable', len(usable), '| usable - export', len(usable-ks), '| export - usable', len(ks-usable))
pp=set()
for p in sorted(glob.glob(F+'/pos_parts/bucket_c*.parquet')):
    pp|=set(pq.read_table(p, columns=['match_key']).column('match_key').to_pylist())
print('usable - export == pos_parts keys absent from export:', (usable-ks)==(pp-ks), len(usable-ks))
# duplicate groups
nonkeep=[k for i,k in enumerate(cm['match_key']) if (not cm['dup_keeper'][i]) and cm['in_scope'][i] and (cm['n_ok_rounds'][i] or 0)>0]
print('in-scope ok non-keepers', len(nonkeep))
print('gkeeper columns sample', cm['gkeeper'][:2], cm['gkey'][:2], cm['ggroup'][:2])
kp=[cm['gkeeper'][rows[k]] for k in nonkeep]
print('non-keeper gkeeper is a key in clean_matches', sum(1 for g in kp if g in rows), 'of', len(kp))
print('non-keeper gkeeper is in export', sum(1 for g in kp if g in ks), '| gkeeper usable', sum(1 for g in kp if g in usable))
# groups: exactly one keeper per ggroup among usable?
grp=defaultdict(list)
for i,k in enumerate(cm['match_key']):
    if cm['in_scope'][i] and (cm['n_ok_rounds'][i] or 0)>0:
        grp[cm['ggroup'][i]].append((k, bool(cm['dup_keeper'][i])))
multi=sum(1 for g,v in grp.items() if sum(1 for _,kk in v if kk)!=1)
print('ok groups', len(grp), 'groups without exactly one keeper', multi)
