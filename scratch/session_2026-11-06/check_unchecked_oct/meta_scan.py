import json, glob, os
files=sorted(glob.glob('G:/pureskill/csds-cs2/models/*.meta.json'))+['C:/Users/<user>/sprint/explain83/train_meta.json','C:/Users/<user>/sprint/matrix_trainctr/meta.json','G:/pureskill/csds-cs2/data/combined/centre_fit_train_keys.json']
for f in files:
    try:
        m=json.load(open(f, encoding='utf-8'))
    except Exception as e:
        print(os.path.basename(f), 'ERR', e); continue
    def nfeat(o):
        for k in ['features','feats','feature_names','feats_all','inputs','input_features']:
            if isinstance(o,dict) and k in o and isinstance(o[k],list): return k, len(o[k])
        return None
    info={}
    if isinstance(m,dict):
        for k,v in m.items():
            if isinstance(v,list): info[k]=len(v)
            elif isinstance(v,(int,float,str)) and len(str(v))<60: info[k]=v
    print(os.path.basename(f), json.dumps(info)[:700])
