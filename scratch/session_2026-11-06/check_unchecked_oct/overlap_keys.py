import polars as pl, collections
S='C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
P=pl.Config
batch=pl.read_csv(f'{S}/batch_v4_per_match.csv', schema_overrides={'match_key':pl.Utf8}) if False else pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8}).select(['match_key','match_split','date']).unique()
bk=set(batch['match_key'].to_list())
print('batch unique keys', len(bk))
def dt(k):
    p=k.split('/'); return '/'.join(p[1:4]) if len(p)>=4 else None
cache=pl.read_csv(f'{S}/space/cache_keys.csv', schema_overrides={'match_key':pl.Utf8,'match_date':pl.Utf8,'era':pl.Utf8})
ck=set(cache['match_key'].to_list())
print('cache keys', len(ck), 'batch in cache', len(bk&ck), 'batch not in cache', len(bk-ck))
bc=cache.filter(pl.col('match_key').is_in(list(bk)))
print('batch-in-cache by match_date', bc.group_by('match_date').agg(pl.len().alias('n')).sort('match_date').to_dicts()[:10])
print('batch-in-cache by era', bc.group_by('era').agg(pl.len().alias('n')).to_dicts())
# cache dates overall
cd=cache.with_columns(pl.col('match_key').str.slice(5,10).alias('ymd'))
print('cache date range', cache['match_date'].min(), cache['match_date'].max())
print('cache keys on Oct21-25', cache.filter(pl.col('match_key').str.contains(r'^csds/2026/10/2[1-5]/')).height)
print('cache keys on Oct26-31', cache.filter(pl.col('match_key').str.contains(r'^csds/2026/10/(2[6-9]|3[01])/')).height)
# manifests
for name in ['manifest_v2_labels.csv','manifest_v1_labels.csv']:
    m=pl.read_csv(f'{S}/space/{name}', schema_overrides={'match_key':pl.Utf8,'label':pl.Utf8})
    mb=m.filter(pl.col('match_key').is_in(list(bk)))
    print(name, 'batch in manifest', mb.height, 'label counts', mb.group_by('label').agg(pl.len().alias('n')).to_dicts(), 'manifest total', m.height, m.group_by('label').agg(pl.len().alias('n')).to_dicts())
    print('   batch-in-manifest by split x date', mb.with_columns(pl.col('match_key').str.slice(5,10).alias('ymd')).group_by(['label','ymd']).agg(pl.len().alias('n')).sort(['label','ymd']).to_dicts())
    print('   manifest keys on Oct 21-25 (any label)', m.filter(pl.col('match_key').str.contains(r'^csds/2026/10/2[1-5]/')).group_by('label').agg(pl.len().alias('n')).to_dicts())
    print('   manifest keys on Oct 26-31 (any label)', m.filter(pl.col('match_key').str.contains(r'^csds/2026/10/(2[6-9]|3[01])/')).group_by('label').agg(pl.len().alias('n')).to_dicts())
    if name=='manifest_v2_labels.csv':
        m2=m
# holdout B
hb=set(pl.read_csv(f'{S}/space/holdout_b_keys.csv', schema_overrides={'match_key':pl.Utf8})['match_key'].to_list())
print('holdout B keys', len(hb), 'batch overlap', len(hb & bk), 'holdout B in cache', len(hb & ck))
print('holdout B dates', collections.Counter(dt(k) for k in hb).most_common(10))
# frozen
fz=set(pl.read_csv(f'{S}/space/frozen_keys.csv', schema_overrides={'match_key':pl.Utf8})['match_key'].to_list())
print('frozen keys', len(fz), 'batch overlap', len(fz & bk), 'frozen in cache', len(fz & ck), 'frozen in holdoutB', len(fz & hb))
print('frozen dates', collections.Counter(dt(k) for k in fz).most_common(12))
