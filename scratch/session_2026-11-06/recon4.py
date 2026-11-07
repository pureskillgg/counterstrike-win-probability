import json
from collections import Counter, defaultdict
import pyarrow.parquet as pq
F='/home/sagemaker-user/adam/builder/final'; E=F+'/export_clean_v2'; D='/home/sagemaker-user/adam/builder/data'
keys=json.load(open(E+'/match_keys.json')); ks=set(keys)
cm=pq.read_table(F+'/clean_matches.parquet').to_pydict()
n=len(cm['match_key'])
idx_of={k:i for i,k in enumerate(cm['match_key'])}
usable=set(cm['match_key'][i] for i in range(n) if cm['dup_keeper'][i] and cm['in_scope'][i] and (cm['n_ok_rounds'][i] or 0)>0)
def show(i):
    return dict(key=cm['match_key'][i][-45:], src=cm['src'][i], keeper=cm['dup_keeper'][i], in_scope=cm['in_scope'][i], n_ok=cm['n_ok_rounds'][i], inexp=cm['match_key'][i] in ks)
# groups by gkey: bad ones
grp=defaultdict(list)
for i in range(n): grp[cm['gkey'][i]].append(i)
bad=[g for g,v in grp.items() if sum(1 for i in v if cm['dup_keeper'][i])!=1]
print('bad gkey groups', len(bad))
print('bad group keeper-count dist', Counter(sum(1 for i in grp[g] if cm['dup_keeper'][i]) for g in bad), '| sizes', Counter(len(grp[g]) for g in bad))
lost_rows=[]
for g,v in grp.items():
    for i in v:
        if (not cm['dup_keeper'][i]) and cm['in_scope'][i] and (cm['n_ok_rounds'][i] or 0)>0:
            if not any(cm['match_key'][j] in usable for j in v if cm['dup_keeper'][j]):
                lost_rows.append(i)
print('lost-looking rows', len(lost_rows))
for i in lost_rows[:6]:
    g=cm['gkey'][i]
    print('  row', show(i), '| group size', len(grp[g]), 'keepers', [show(j) for j in grp[g] if cm['dup_keeper'][j]][:2])
# check dup_key grouping
gk2=defaultdict(list)
for i in range(n): gk2[cm['dup_key'][i]].append(i)
print('dup_key groups', len(gk2), 'size dist', dict(sorted(Counter(len(v) for v in gk2.values()).items())))
bad2=[g for g,v in gk2.items() if sum(1 for i in v if cm['dup_keeper'][i])!=1]
print('dup_key groups without exactly one keeper', len(bad2))
print('dup_key sample', str(cm['dup_key'][0])[:80])
# keep ks check: in export and keeper
print('usable keepers in export', sum(1 for k in usable if k in ks), 'of', len(usable))
# manifest
sm=json.load(open(F+'/split_manifest_v2.json'))
print('manifest rule', str(sm.get('rule'))[:400])
print('manifest frozen_from', str(sm.get('frozen_from'))[:300])
print('manifest seed', sm.get('seed'), 'created_at', sm.get('created_at'))
asg=sm['assign']
print('assign type', type(asg).__name__, 'len', len(asg))
if isinstance(asg, dict):
    print('assign sample', list(asg.items())[:2])
h=sm.get('history')
print('history type', type(h).__name__, 'len', len(h) if h is not None else None)
print('history sample', str(h)[:600])
