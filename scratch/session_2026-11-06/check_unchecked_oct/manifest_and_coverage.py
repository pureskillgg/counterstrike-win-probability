import polars as pl, json, os, collections
S='C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
bk=set(rounds['match_key'].unique().to_list())
D='G:/pureskill/csds-cs2/data/combined'
mans={'current':f'{D}/split_manifest.json','backup_pre_22_26 (candidate A)':f'{D}/_backup_pre_22_26_20261027/split_manifest.json','combined_backup_pre_1021 (candidate B?)':'G:/pureskill/csds-cs2/data/combined_backup_pre_1021/split_manifest.json','failed_extension':f'{D}/_failed_extension_22_25_20261027/split_manifest.json'}
for name,p in mans.items():
    if not os.path.exists(p): print(name,'MISSING'); continue
    m=json.load(open(p, encoding='utf-8'))
    a=m.get('assign',{})
    lab=collections.Counter(a.values())
    bl=collections.Counter(a[k] for k in bk if k in a)
    print(name, 'entries', len(a), 'labels', dict(lab), 'batch in manifest', sum(1 for k in bk if k in a), 'batch labels', dict(bl), 'created', m.get('created_at'), flush=True)
# coverage in derived caches
def cov(path, label):
    df=pl.scan_parquet(path).select(['match_key','round','second']).filter(pl.col('match_key').is_in(list(bk))).unique().collect()
    df=df.with_columns(pl.col('round').cast(pl.Int64), pl.col('second').cast(pl.Int64))
    j=rounds.select(['match_key','round_num','steps']).rename({'round_num':'round'}).with_columns(pl.col('round').cast(pl.Int64)).join(df.with_columns(pl.lit(1).alias('h')), on=['match_key','round'], how='left')
    inr=df.join(rounds.select(['match_key','round_num','steps']).rename({'round_num':'round'}).with_columns(pl.col('round').cast(pl.Int64)), on=['match_key','round'], how='inner').filter((pl.col('second')>=0)&(pl.col('second')<pl.col('steps')))
    rnd=inr.select(['match_key','round']).unique().height
    sec=inr.height
    mt=inr['match_key'].n_unique()
    print(label, 'matches', mt, 'rounds', rnd, 'seconds', sec, 'of batch rounds 54642 / seconds 3723828', flush=True)
cov(f'{D}/cache_snapshot_2026_ext.parquet','cache_snapshot_2026_ext')
cov(f'{D}/cache_snapshot_2026_oct23_31.parquet','cache_snapshot_2026_oct23_31')
cov('G:/pureskill/csds-cs2/data/combined/cache_pos_2026.parquet','cache_pos_2026 (current) [key=second]') if False else None
