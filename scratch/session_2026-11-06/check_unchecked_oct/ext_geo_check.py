import polars as pl, glob, os, json, collections
S='C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
bk=set(rounds['match_key'].unique().to_list())
ext='G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026_ext.parquet'
c=pl.scan_parquet(ext).select(['match_key','round','second','label_ct_win']).filter(pl.col('match_key').is_in(list(bk))).collect()
print('ext batch rows', c.height, 'matches', c['match_key'].n_unique(), 'rounds', c.select(['match_key','round']).unique().height, 'seconds', c.select(['match_key','round','second']).unique().height, 'label null', c['label_ct_win'].null_count())
bkeys_ext=set(c['match_key'].unique().to_list())
print('ext batch matches', len(bkeys_ext))
# geo grids
geo_files=sorted(glob.glob('G:/pureskill/csds-cs2/data/diagnostics/allmaps_geo/*.parquet'))
tot=collections.Counter()
for f in geo_files:
    m=os.path.basename(f).replace('.parquet','')
    g=pl.scan_parquet(f).select(['match_key','round','second']).filter(pl.col('match_key').is_in(list(bkeys_ext))).collect()
    tot[m]=(g['match_key'].n_unique(), g.height)
    print('geo', m, 'batch matches', g['match_key'].n_unique(), 'batch rows', g.height)
