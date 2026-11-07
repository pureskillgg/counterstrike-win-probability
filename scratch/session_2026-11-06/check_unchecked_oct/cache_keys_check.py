import polars as pl, json, collections, os, sys
S='C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
bk=set(rounds['match_key'].unique().to_list())
hb=set(pl.read_csv(f'{S}/space/holdout_b_keys.csv', schema_overrides={'match_key':pl.Utf8})['match_key'].to_list())
fz=set(pl.read_csv(f'{S}/space/frozen_keys.csv', schema_overrides={'match_key':pl.Utf8})['match_key'].to_list())
cf=set(json.load(open('G:/pureskill/csds-cs2/data/combined/centre_fit_train_keys.json', encoding='utf-8')))
files=[
 'G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026.parquet',
 'G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026_ext.parquet',
 'G:/pureskill/csds-cs2/data/combined/_backup_pre_22_26_20261027/cache_snapshot_2026_ext.parquet',
 'G:/pureskill/csds-cs2/data/combined/_failed_extension_22_25_20261027/cache_snapshot_2026_ext.parquet',
 'G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026_oct23_31.parquet',
 'G:/pureskill/csds-cs2/data/combined/cache_pos_2026.parquet',
 'G:/pureskill/csds-cs2/data/combined/cache_pos_2026_before_oct14_31.parquet',
 'G:/pureskill/csds-cs2/data/combined/cache_pos_2026_trainctr.parquet',
 'G:/pureskill/csds-cs2/data/combined/_joined_pos_2026.parquet',
 'G:/pureskill/csds-cs2/data/combined/event_feats_2026.parquet',
]
only=sys.argv[1:]
for f in files:
    if only and not any(o in f for o in only): continue
    if not os.path.exists(f): print('MISSING', f); continue
    try:
        ks=pl.scan_parquet(f).select('match_key').unique().collect()['match_key'].to_list()
    except Exception as e:
        print('ERR', f, e); continue
    s=set(ks)
    dates=collections.Counter('/'.join(k.split('/')[1:4]) for k in s if k and k.startswith('csds/'))
    b_dates=collections.Counter('/'.join(k.split('/')[1:4]) for k in (s & bk))
    out={
     'file': os.path.basename(os.path.dirname(f))+'/'+os.path.basename(f),
     'unique_keys': len(s),
     'batch_overlap': len(s & bk),
     'batch_overlap_by_date': dict(sorted(b_dates.items())),
     'holdoutB_overlap': len(s & hb),
     'frozen_overlap': len(s & fz),
     'centre_fit_overlap': len(s & cf),
     'keys_dated_2026_10_26_31': sum(v for d,v in dates.items() if d>='2026/10/26' and d<='2026/10/31'),
     'key_date_range': [min(dates) if dates else None, max(dates) if dates else None],
    }
    print(json.dumps(out), flush=True)
