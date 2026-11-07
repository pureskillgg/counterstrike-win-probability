import polars as pl, csv, collections, glob, os
S='C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
ex=pl.read_csv(f'{S}/s3_extra_oct21_25_keys.csv', schema_overrides={'match_key':pl.Utf8})['match_key'].to_list()
exs=set(ex)
D='G:/pureskill/csds-cs2/data'
for src in ['positions_1hz','status_1hz']:
    hits=set(); files=[]
    for f in sorted(glob.glob(f'{D}/{src}/2026-10-2[1-5].parquet')):
        ks=set(pl.scan_parquet(f).select('match_key').unique().collect()['match_key'].to_list())
        h=ks & exs
        hits|=h
        files.append((os.path.basename(f), len(ks), len(h)))
    print(src, 'files', files, 'extra-folder matches present locally', len(hits), flush=True)
# also check cache_pos current and ext for extras
for f in ['G:/pureskill/csds-cs2/data/combined/cache_pos_2026.parquet','G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026_oct23_31.parquet','G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026_ext.parquet']:
    ks=set(pl.scan_parquet(f).select('match_key').unique().collect()['match_key'].to_list())
    print(os.path.basename(f), 'extra-folder matches present', len(ks & exs), flush=True)
