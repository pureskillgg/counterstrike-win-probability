import polars as pl, json, os
S='C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
batch=set(rounds['match_key'].unique().to_list())
for base in ['C:/Users/13144/sprint/matrix_trainctr','G:/pureskill/csds-cs2/data/sprint/matrix_trainctr']:
    p=os.path.join(base,'val_keys.parquet')
    print('==',p, os.path.getsize(p))
    v=pl.read_parquet(p)
    print('schema', v.schema, 'rows', v.height)
    print(v.head(3))
    col=[c for c in v.columns if 'key' in c.lower()]
    print('key cols', col)
    if col:
        ks=set(v[col[0]].cast(pl.Utf8).to_list())
        print('unique keys', len(ks), 'batch overlap', len(ks & batch))
    if 'match_key' in v.columns:
        pass
    m=json.load(open(os.path.join(base,'meta.json')))
    print('meta keys', list(m.keys()))
