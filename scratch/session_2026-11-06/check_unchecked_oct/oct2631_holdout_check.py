import polars as pl, collections, json
S='C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
bk=set(rounds['match_key'].unique().to_list())
hb=set(pl.read_csv(f'{S}/space/holdout_b_keys.csv', schema_overrides={'match_key':pl.Utf8})['match_key'].to_list())
def keys_of(path, col='match_key'):
    try:
        df=pl.read_parquet(path, columns=[col]) if path.endswith('.parquet') else None
    except Exception as e:
        return 'ERR '+str(e)[:200]
    return set(df[col].cast(pl.Utf8).to_list())
paths=['C:/Users/13144/sprint/holdout_oct26_31_keys.parquet','C:/Users/13144/sprint/holdout_oct26_31/keys.parquet']
for p in paths:
    ks=keys_of(p)
    if isinstance(ks,str): print(p, ks); continue
    dt=collections.Counter('/'.join(k.split('/')[1:4]) for k in ks)
    print(p, 'keys', len(ks), 'batch overlap', len(ks & bk), 'holdoutB overlap', len(ks & hb), 'date range', min(dt), max(dt), 'by date', dict(sorted(dt.items())))
