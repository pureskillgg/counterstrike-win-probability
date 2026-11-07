import numpy as np, json, glob, os
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
E='/home/sagemaker-user/adam/builder/final/export_clean_v2'
F='/home/sagemaker-user/adam/builder/final'
keys=json.load(open(E+'/match_keys.json')); ks=set(keys)
cr=pq.read_table(F+'/clean_rounds.parquet', columns=['match_key','round','use','dup_keeper']).to_pydict()
from collections import defaultdict
use_by=defaultdict(int); tot_by=defaultdict(int)
for k,r,u in zip(cr['match_key'],cr['round'],cr['use']):
    tot_by[k]+=1
    if u: use_by[k]+=1
pp_keys=set()
for p in sorted(glob.glob(F+'/pos_parts/bucket_c*.parquet')):
    pp_keys |= set(pq.read_table(p, columns=['match_key']).column('match_key').to_pylist())
absent=pp_keys-ks
print('keys in pos_parts absent from export', len(absent))
print('absent keys: use rounds total', sum(use_by[k] for k in absent), '| all rounds in clean_rounds total', sum(tot_by[k] for k in absent))
print('absent keys with zero use rounds', sum(1 for k in absent if use_by[k]==0))
# dropped-round count per absent key: use rounds not in export => all use rounds of absent keys
# locate sentinel-filled steps
poff=np.load(E+'/pos_offsets.npy').astype(np.int64)
rm=np.load(E+'/round_match.npy').astype(np.int64)
rnum=np.load(E+'/round_num.npy').astype(np.int64)
K=rm*1024+rnum; order=np.argsort(K); Ks=K[order]
import pyarrow as pa2
ALLc=[]
pp_cov=np.zeros(int(poff[-1]),dtype=np.int32)
from pyarrow import compute as pc2
keys_arr=pa.array(keys)
for p in sorted(glob.glob(F+'/pos_parts/bucket_c*.parquet')):
    t=pq.read_table(p, columns=['match_key','round','tick','c_site_hi'])
    idx=pc2.index_in(t.column('match_key'), value_set=keys_arr)
    f=pc2.is_valid(idx).to_numpy(zero_copy_only=False)
    s=np.nonzero(f)[0]
    mi=idx.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)[s]
    rn=t.column('round').to_numpy(zero_copy_only=False)[s].astype(np.int64)
    tk=t.column('tick').to_numpy(zero_copy_only=False)[s].astype(np.int64)
    rk=mi*1024+rn
    pos=np.clip(np.searchsorted(Ks,rk),0,len(Ks)-1)
    ok=Ks[pos]==rk
    ridx=order[pos[ok]]
    d=tk[ok]-np.load(E+'/round_freeze.npy').astype(np.int64)[ridx] if False else None
    ALLc.append((ridx, tk[ok]))
rfz=np.load(E+'/round_freeze.npy').astype(np.int64)
cov=np.zeros(int(poff[-1]),dtype=np.int32)
for ridx,tk in ALLc:
    d=tk-rfz[ridx]
    pidx=poff[ridx]+d//128
    np.add.at(cov,pidx,1)
# bucket-src rounds
cm=pq.read_table(F+'/clean_matches.parquet', columns=['match_key','src']).to_pydict()
bset={k for k,s in zip(cm['match_key'],cm['src']) if s=='bucket'}
is_b=np.array([k in bset for k in keys])
rb=np.nonzero(is_b[rm])[0]
mask=np.zeros(int(poff[-1])+1,dtype=np.int64)
np.add.at(mask,poff[rb],1); np.add.at(mask,poff[rb+1],-1)
expm=np.cumsum(mask)[:-1]>0
miss=np.nonzero(expm & (cov==0))[0]
mr=np.searchsorted(poff,miss,side='right')-1
rel=miss-poff[mr]; ln=poff[mr+1]-poff[mr]
print('missing bucket steps', len(miss), 'rounds', len(np.unique(mr)))
print('missing steps that are the LAST step of their round', int((rel==ln-1).sum()))
print('missing steps in rounds with exactly 1 missing step', int(sum(1 for r,c in zip(*np.unique(mr,return_counts=True)) if c==1)))
print('rounds with missing steps: count of missing per round min/max', np.unique(mr,return_counts=True)[1].min(), np.unique(mr,return_counts=True)[1].max())
print('missing steps: rel index min/max', int(rel.min()), int(rel.max()), '| round length min/max of affected', int(ln.min()), int(ln.max()))
# tail check: are missing steps the trailing steps of each affected round?
tail_ok=0; per_round_missing=defaultdict(list)
for pos_,r_ in zip(miss,mr): per_round_missing[int(r_)].append(int(pos_-poff[r_]))
for r_,lst in per_round_missing.items():
    L=int(poff[r_+1]-poff[r_])
    if sorted(lst)==list(range(L-len(lst),L)): tail_ok+=1
print('affected rounds whose missing steps are exactly the trailing steps', tail_ok, 'of', len(per_round_missing))
print('pos_check_all bucket files', len(glob.glob('/home/sagemaker-user/adam/builder/final/pos_check_all/*.parquet')))
print('export_clean_v2 has verification.json', os.path.exists(E+'/verification.json'), '| export_clean has verification.json', os.path.exists('/home/sagemaker-user/adam/builder/final/export_clean/verification.json'))
