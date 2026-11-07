import polars as pl, glob, os
S='C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
bk=set(rounds['match_key'].unique().to_list())
r2=rounds.select(['match_key','round_num','round_freeze','steps']).rename({'round_num':'round'}).with_columns(pl.col('round').cast(pl.Int64))
f='G:/pureskill/csds-cs2/data/combined/cache_pos_2026.parquet'
df=pl.scan_parquet(f).select(['match_key','round','tick']).filter(pl.col('match_key').is_in(list(bk))).unique().collect()
df=df.with_columns(pl.col('round').cast(pl.Int64)).join(r2, on=['match_key','round'], how='inner')
df=df.with_columns(((pl.col('tick')-pl.col('round_freeze'))//64).alias('sec'), ((pl.col('tick')-pl.col('round_freeze'))%64).alias('mod'))
inr=df.filter((pl.col('sec')>=0)&(pl.col('sec')<pl.col('steps')))
print('cache_pos_2026 batch matches', inr['match_key'].n_unique(), 'rounds', inr.select(['match_key','round']).unique().height, 'seconds', inr.select(['match_key','round','sec']).unique().height, 'tick mod64==0 share', round(inr.filter(pl.col('mod')==0).height/max(inr.height,1),4), flush=True)
# batch Oct21 keys missing from ext and oct23_31
b21={k for k in bk if k.startswith('csds/2026/10/21/')}
ext=set(pl.scan_parquet('G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026_ext.parquet').select('match_key').unique().collect()['match_key'].to_list())
o=set(pl.scan_parquet('G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026_oct23_31.parquet').select('match_key').unique().collect()['match_key'].to_list())
print('Oct21 batch keys', len(b21), 'missing from ext', sorted(b21-ext), 'missing from oct23_31', sorted(b21-o), flush=True)
# _plants check
pf=sorted(glob.glob('G:/pureskill/csds-cs2/data/positions_1hz/_plants/*.parquet'))
print('_plants files', len(pf), os.path.basename(pf[-1]) if pf else None)
hits=0; tot=0
for p in pf:
    if os.path.basename(p) < '2026-10-21.parquet' and os.path.basename(p)[:7] != '2026-10': continue
    ks=set(pl.scan_parquet(p).select('match_key').unique().collect()['match_key'].to_list())
    hits+=len(ks & bk); tot+=len(ks)
print('_plants batch-key overlap (Oct 21-25 files)', hits, 'of keys in those files', tot, flush=True)
