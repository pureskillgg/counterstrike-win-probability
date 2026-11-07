import polars as pl, json, collections
S='C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
bk=set(rounds['match_key'].unique().to_list())
fz=set(pl.read_csv(f'{S}/space/frozen_keys.csv', schema_overrides={'match_key':pl.Utf8})['match_key'].to_list())
hb=set(pl.read_csv(f'{S}/space/holdout_b_keys.csv', schema_overrides={'match_key':pl.Utf8})['match_key'].to_list())
man=pl.read_csv(f'{S}/space/manifest_v2_labels.csv', schema_overrides={'match_key':pl.Utf8,'label':pl.Utf8})
lab=dict(zip(man['match_key'].to_list(), man['label'].to_list()))
t21=set(k for k,v in lab.items() if v=='test' and k.startswith('csds/2026/10/2') and '/2026/10/2' in k)
print('manifest v2 test-labelled keys on Oct 21-25', len(t21), 'equal to Holdout B list', t21==hb, 'overlap', len(t21 & hb), flush=True)
def label_of(k):
    if k in fz: return 'test(frozen)'
    return lab.get(k,'NA')
for name, path in [('cache_snapshot_2026_oct23_31','G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026_oct23_31.parquet'),('cache_snapshot_2026','G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026.parquet')]:
    df=pl.scan_parquet(path).select(['match_key','label_ct_win']).collect()
    keys=df['match_key'].to_list(); labs=df['label_ct_win'].to_list()
    rows=collections.Counter(); keyset=collections.defaultdict(set)
    for k,l in zip(keys,labs):
        s=label_of(k)
        rows[s]+=1
        if k in bk: rows['batch:'+s]+=1
        if k in hb: rows['holdoutB:'+s]+=1
        if k in fz: rows['frozen_rows_total']+=1
    print(name, 'rows by retrain split (frozen overrides to test; else manifest label):', {k:v for k,v in sorted(rows.items())}, flush=True)
    print(name, 'null labels', df['label_ct_win'].null_count(), flush=True)
