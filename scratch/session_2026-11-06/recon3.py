import json
from collections import Counter, defaultdict
import pyarrow.parquet as pq
D='/home/sagemaker-user/adam/builder/data'; F='/home/sagemaker-user/adam/builder/final'; E=F+'/export_clean_v2'
keys=json.load(open(E+'/match_keys.json')); ks=set(keys)
cm=pq.read_table(F+'/clean_matches.parquet').to_pydict()
n=len(cm['match_key'])
print('dup_keeper == gkeeper for all rows', all(bool(cm['dup_keeper'][i])==bool(cm['gkeeper'][i]) for i in range(n)))
usable=set(cm['match_key'][i] for i in range(n) if cm['dup_keeper'][i] and cm['in_scope'][i] and (cm['n_ok_rounds'][i] or 0)>0)
grp=defaultdict(list)
for i in range(n):
    grp[cm['gkey'][i]].append(i)
sizes=Counter(len(v) for v in grp.values())
print('dup groups by gkey', len(grp), 'size distribution', dict(sorted(sizes.items())))
bad_keep=0; lost=0; nonkeep_usable_group=0; nonkeep_total=0
for g,idx in grp.items():
    keepers=[cm['match_key'][i] for i in idx if cm['dup_keeper'][i]]
    if len(keepers)!=1: bad_keep+=1
    for i in idx:
        if not cm['dup_keeper'][i]:
            nonkeep_total+=1
            if cm['in_scope'][i] and (cm['n_ok_rounds'][i] or 0)>0:
                nonkeep_usable_group+=1
                if not any(cm['match_key'][j] in usable for j in idx if cm['dup_keeper'][j]):
                    lost+=1
print('groups without exactly one keeper', bad_keep)
print('non-keepers total', nonkeep_total, '| usable-looking non-keepers', nonkeep_usable_group, '| usable-looking non-keepers whose group has no usable keeper', lost)
keepers_usable=[cm['match_key'][i] for i in range(n) if cm['dup_keeper'][i] and cm['in_scope'][i] and (cm['n_ok_rounds'][i] or 0)>0]
print('usable keepers', len(keepers_usable), '| in export', sum(1 for k in keepers_usable if k in ks))
# the 1185 in-scope universe keys absent from clean_matches
chunk={}
for c in range(68):
    t=pq.read_table(f'{D}/c{c:03d}/matches.parquet', columns=['match_key','in_scope']).to_pydict()
    for k,s in zip(t['match_key'],t['in_scope']): chunk[k]=s
cmset=set(cm['match_key'])
absent=[k for k in chunk if k not in cmset]
print('chunk keys absent from clean_matches', len(absent), '| of which in_scope', sum(1 for k in absent if chunk[k]))
hb=set(l.strip() for l in open(F+'/holdout_b_keys.txt', encoding='utf-8') if l.strip())
print('absent & holdout_b', sum(1 for k in absent if k in hb), '| absent & not holdout_b', sum(1 for k in absent if k not in hb))
sm=json.load(open(F+'/split_manifest_v2.json'))
print('split_manifest_v2 type', type(sm).__name__, 'top keys', list(sm.keys())[:12] if isinstance(sm,dict) else len(sm))
