import json
from collections import Counter, defaultdict
import pyarrow.parquet as pq
F='/home/sagemaker-user/adam/builder/final'; E=F+'/export_clean_v2'; D='/home/sagemaker-user/adam/builder/data'
keys=json.load(open(E+'/match_keys.json')); ks=set(keys)
cm=pq.read_table(F+'/clean_matches.parquet', columns=['match_key','dup_keeper','in_scope','n_ok_rounds','src','gkey']).to_pydict()
cmr={k:i for i,k in enumerate(cm['match_key'])}
usable=set(k for i,k in enumerate(cm['match_key']) if cm['dup_keeper'][i] and cm['in_scope'][i] and (cm['n_ok_rounds'][i] or 0)>0)
sm=json.load(open(F+'/split_manifest_v2.json')); asg=sm['assign']
hb=set(l.strip() for l in open(F+'/holdout_b_keys.txt', encoding='utf-8') if l.strip())
lt=pq.read_table(F+'/laptop_matches.parquet')
print('laptop cols', lt.schema.names)
lcols=[c for c in ['sig_hash','wseq','r0','map_name','match_date'] if c in lt.schema.names]
ld=lt.select(['match_key']+lcols).to_pydict()
info={}
for i,k in enumerate(ld['match_key']):
    info[k]={c:ld[c][i] for c in lcols}; info[k]['src']='laptop'
for c in range(68):
    t=pq.read_table(f'{D}/c{c:03d}/matches.parquet', columns=['match_key','sig_hash','wseq','r0','map_name','match_date']).to_pydict()
    for i,k in enumerate(t['match_key']):
        info[k]={'sig_hash':t['sig_hash'][i],'wseq':t['wseq'][i],'r0':t['r0'][i],'map_name':t['map_name'][i],'match_date':t['match_date'][i],'src':'bucket'}
print('info keys', len(info))
lost=[k for k in cm['match_key'] if (not cm['dup_keeper'][cmr[k]]) and cm['in_scope'][cmr[k]] and (cm['n_ok_rounds'][cmr[k]] or 0)>0 and k not in ks]
# restrict to groups with no keeper
from collections import defaultdict as dd
grp=dd(list)
for i,k in enumerate(cm['match_key']): grp[cm['gkey'][i]].append(k)
lost26=[k for k in lost if not any(cm['dup_keeper'][cmr[j]] for j in grp[cm['gkey'][cmr[k]]])]
print('lost-looking (no keeper in group)', len(lost26), '| src', Counter(cm['src'][cmr[k]] for k in lost26))
excl_test=set(k for k,v in asg.items() if v=='test' and k not in ks and k not in hb and k not in cmr)
print('excluded mirage-test-like keys (assign test, absent from clean, not holdout)', len(excl_test))
sig_of=lambda k: info.get(k,{}).get('sig_hash')
hb_sigs=set(sig_of(k) for k in hb if k in info) - {None}
ex_sigs=set(sig_of(k) for k in excl_test if k in info) - {None}
print('holdout_b with sig available', sum(1 for k in hb if k in info), '| excluded test with sig', sum(1 for k in excl_test if k in info))
print('lost26 with sig available', sum(1 for k in lost26 if k in info))
print('lost26 sig == any holdout_b sig', sum(1 for k in lost26 if sig_of(k) in hb_sigs))
print('lost26 sig == any excluded-test sig', sum(1 for k in lost26 if sig_of(k) in ex_sigs))
# sig groups across all known matches
sg=dd(list)
for k,v in info.items():
    if v.get('sig_hash') is not None: sg[v['sig_hash']].append(k)
print('lost26 sig group sizes', Counter(len(sg.get(sig_of(k),[])) for k in lost26))
print('lost26 partners in export', sum(1 for k in lost26 for p in sg.get(sig_of(k),[]) if p in ks and p!=k))
print('lost26 partners in holdout_b', sum(1 for k in lost26 for p in sg.get(sig_of(k),[]) if p in hb and p!=k))
print('lost26 partners in excluded test', sum(1 for k in lost26 for p in sg.get(sig_of(k),[]) if p in excl_test and p!=k))
print('lost26 sample', [(k[-40:], info.get(k,{}).get('map_name'), info.get(k,{}).get('match_date')) for k in lost26[:4]])
# the 2 unexplained train keys in assign
unexp=[k for k,v in asg.items() if v=='train' and k not in ks and k not in hb and k not in cmr]
print('unexplained assign train keys', len(unexp), unexp)
for k in unexp:
    print('  info', info.get(k), '| partners', [p for p in sg.get(sig_of(k),[]) if p!=k][:3] if sig_of(k) else None)
print('unexplained sig in export', [ (sig_of(k) in set(sig_of(p) for p in ks if p in info)) for k in unexp] if False else 'skip')

print('--- follow-up ---')
print('excluded test (878) map counts', Counter(info[k]['map_name'] for k in excl_test if k in info).most_common(10))
print('excluded test (878) with map available', sum(1 for k in excl_test if k in info))
print('holdout_b map counts', Counter(info[k]['map_name'] for k in hb if k in info).most_common(10))
exp_sigs=set(sig_of(p) for p in ks if p in info)
for k in unexp:
    s=sig_of(k)
    print('unexplained', k[-40:], 'partners in export', [p for p in sg.get(s,[]) if p in ks and p!=k], 'partners in holdout', [p for p in sg.get(s,[]) if p in hb])
print('mirage test total in manifest', sum(1 for k,v in asg.items() if v=='test' and info.get(k,{}).get('map_name')=='de_mirage'))
print('export mirage test matches', sum(1 for k in keys if info.get(k,{}).get('map_name')=='de_mirage' and asg.get(k)=='test'))

print('--- mirage residual ---')
mt=[k for k,v in asg.items() if v=='test' and info.get(k,{}).get('map_name')=='de_mirage']
resid=[k for k in mt if k not in ks and k not in hb and k not in excl_test]
print('mirage test keys', len(mt), 'residual (not export, not holdout, not excluded-test)', len(resid), [ (k[-40:], k in usable) for k in resid])
print('residual in usable (empty47)', [k in usable for k in resid], '| residual in clean_matches', [k in cmr for k in resid])
