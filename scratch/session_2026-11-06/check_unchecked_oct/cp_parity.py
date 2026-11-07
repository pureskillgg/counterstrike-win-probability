import polars as pl, collections
S='C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
bk=set(rounds['match_key'].unique().to_list())
r2=rounds.select(['match_key','round_num','round_freeze','steps']).rename({'round_num':'round'}).with_columns(pl.col('round').cast(pl.Int64))
for path in ['G:/pureskill/csds-cs2/data/combined/cache_pos_2026.parquet','G:/pureskill/csds-cs2/data/combined/cache_pos_2026_trainctr.parquet']:
    df=pl.scan_parquet(path).select(['match_key','round','tick']).filter(pl.col('match_key').is_in(list(bk))).unique().collect()
    df=df.with_columns(pl.col('round').cast(pl.Int64)).join(r2,on=['match_key','round'],how='inner')
    df=df.with_columns(((pl.col('tick')-pl.col('round_freeze'))//64).alias('sec'))
    inr=df.filter((pl.col('sec')>=0)&(pl.col('sec')<pl.col('steps')))
    u=inr.select(['match_key','round','sec']).unique()
    par=collections.Counter((s%2) for s in u['sec'].to_list())
    per_round=u.group_by(['match_key','round']).agg(pl.len().alias('n')).join(r2,on=['match_key','round'])
    full=per_round.filter(pl.col('n')>=pl.col('steps')).height
    print(path.split('/')[-1], 'seconds', u.height, 'parity', dict(par), 'rounds with all steps', full, 'of', per_round.height, flush=True)
    # sample: show the first 12 seconds of one round
    k=per_round.filter(pl.col('n')<pl.col('steps')).head(1)
    if k.height:
        mk=k['match_key'][0]; rr=k['round'][0]
        print(' sample partial round', mk[-40:], rr, 'secs', sorted(u.filter((pl.col('match_key')==mk)&(pl.col('round')==rr))['sec'].to_list())[:20], 'steps', int(k['steps'][0]), flush=True)
