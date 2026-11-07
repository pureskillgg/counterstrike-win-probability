import polars as pl, glob, os, csv
S='C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
D='G:/pureskill/csds-cs2/data'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
bk=set(rounds['match_key'].unique().to_list())
# _plants: Oct 21-25 files only
hits=set(); keys_all=set()
for d in ['2026-10-21','2026-10-22','2026-10-23','2026-10-24','2026-10-25']:
    p=f'{D}/positions_1hz/_plants/{d}.parquet'
    if not os.path.exists(p): print('missing', p); continue
    ks=set(pl.scan_parquet(p).select('match_key').unique().collect()['match_key'].to_list())
    hits|=ks&bk; keys_all|=ks
print('_plants Oct21-25: keys in files', len(keys_all), 'batch keys present', len(hits), 'batch missing', len(bk-hits), 'non-batch keys', len(keys_all-bk), flush=True)
print('_plants schema', pl.scan_parquet(f'{D}/positions_1hz/_plants/2026-10-21.parquet').collect_schema().names(), flush=True)
# ratio test: cache_pos_2026 seconds vs positions_1hz seconds on OLD keys (Oct 5 file), distinct (key, round, second)
old=pl.scan_parquet(f'{D}/positions_1hz/2026-10-05.parquet').select(['match_key','round','second']).unique().collect()
oldk=set(old['match_key'].unique().to_list())
cp=pl.scan_parquet(f'{D}/combined/cache_pos_2026_before_oct14_31.parquet').select(['match_key','round','tick']).filter(pl.col('match_key').is_in(list(oldk))).unique().collect()
cp=cp.with_columns(pl.col('round').cast(pl.Int64))
# map tick to second using the first tick of round as freeze proxy: sec = (tick - min_tick_round)//64
g=cp.group_by(['match_key','round']).agg(pl.col('tick').min().alias('tmin'))
cp=cp.join(g,on=['match_key','round']).with_columns(((pl.col('tick')-pl.col('tmin'))//64).alias('sec'))
cps=cp.select(['match_key','round','sec']).unique()
old2=old.with_columns(pl.col('round').cast(pl.Int64)).rename({'second':'sec'})
common=set(cps['match_key'].unique().to_list()) & set(old2['match_key'].unique().to_list())
a=old2.filter(pl.col('match_key').is_in(list(common))).height
b=cps.filter(pl.col('match_key').is_in(list(common))).height
print('OLD keys in both (Oct 5 positions vs cache_pos_before):', len(common), 'positions seconds', a, 'cache_pos seconds', b, 'ratio', round(b/max(a,1),3), flush=True)
