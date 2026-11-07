import polars as pl, json
S='C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
bk=set(rounds['match_key'].unique().to_list())
a=set(pl.read_csv(f'{S}/space/holdout_b_keys.csv', schema_overrides={'match_key':pl.Utf8})['match_key'].to_list())
b=set(l.strip() for l in open('G:/pureskill/csds-cs2/data/holdout_b/holdout_keys.txt', encoding='utf-8') if l.strip())
d=json.load(open('G:/pureskill/csds-cs2/data/combined/holdout_22_25_test_keys.json', encoding='utf-8'))
print('holdout_22_25_test_keys.json type', type(d).__name__, (list(d.keys())[:8] if isinstance(d,dict) else len(d)))
if isinstance(d,dict):
    for k,v in d.items():
        if isinstance(v,list): print(' ', k, 'list', len(v), str(v[:1])[:150])
        else: print(' ', k, str(v)[:150])
    cand=[v for v in d.values() if isinstance(v,list) and v and isinstance(v[0],str)]
    c=set(cand[0]) if cand else set()
else:
    c=set(d)
print('space holdout_b_keys', len(a), 'local holdout_keys.txt', len(b), 'equal', a==b)
print('json holdout list', len(c), 'equal to space', c==a, 'overlap', len(c & a))
print('holdout ∩ batch: space', len(a & bk), 'local txt', len(b & bk), 'json', len(c & bk))
