import polars as pl
B='csds/2026/10/21/01c62f0b-c6f9-4efd-979a-9a4ad1fd6880/csds'
rounds=pl.read_csv('C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct/batch_v4_rounds.csv').filter(pl.col('match_key')==B).filter(pl.col('round_num')<=3)
print(rounds)
for src in ['positions_1hz','status_1hz']:
    df=pl.scan_parquet(f'G:/pureskill/csds-cs2/data/{src}/2026-10-21.parquet').filter(pl.col('match_key')==B).filter(pl.col('round')<=3).select(['round','tick','second']).collect()
    g=df.group_by('round').agg([pl.col('second').min().alias('smin'),pl.col('second').max().alias('smax'),pl.col('second').n_unique().alias('nsec'),pl.col('tick').min().alias('tmin'),pl.col('tick').max().alias('tmax'),pl.len().alias('n')]).sort('round')
    print(src); print(g)
    print(df.sort(['round','second']).head(4))
