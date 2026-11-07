import polars as pl, os, glob, json, re, sys
S='C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/check_unchecked_oct'
D='G:/pureskill/csds-cs2/data'
rounds=pl.read_csv(f'{S}/batch_v4_rounds.csv', schema_overrides={'match_key':pl.Utf8})
keys=rounds['match_key'].unique().to_list()
print('batch keys', len(keys), 'batch rounds', rounds.height, flush=True)
sources=sys.argv[1:] or ['positions_1hz','status_1hz','status_16hz','status_8hz']
DATE_RE=re.compile(r'(\d{4})-(\d{2})-(\d{2})')
summary={}
for src in sources:
    files=sorted(glob.glob(f'{D}/{src}/*.parquet'))
    frames=[]; file_rows=[]
    for f in files:
        stem=os.path.basename(f).replace('.parquet','')
        lf=pl.scan_parquet(f).select(['match_key','round','second','tick']).filter(pl.col('match_key').is_in(keys))
        df=lf.group_by(['match_key','round','second']).agg(pl.len().alias('n')).collect()
        if df.height==0: continue
        df=df.with_columns(pl.lit(stem).alias('file'))
        frames.append(df.select(['match_key','round','second','file']))
        file_rows.append((stem, df['match_key'].n_unique(), df.height))
    if not frames:
        summary[src]={'files_scanned':len(files),'files_with_batch_keys':0,'batch_matches_present':0}
        print(src,'no batch keys found', flush=True); continue
    allp=pl.concat(frames)
    files_per_key=allp.group_by('match_key').agg(pl.col('file').unique().sort().alias('files'))
    # distinct (key, round, second) across files
    dist=allp.select(['match_key','round','second']).unique()
    # join with v4 rounds: second must be in [0, steps)
    j=rounds.select(['match_key','round_num','steps']).rename({'round_num':'round'}).join(dist.with_columns(pl.col('round').cast(pl.Int64)), on=['match_key','round'], how='left')
    j=j.with_columns(pl.col('second').cast(pl.Int64))
    in_range=j.filter(pl.col('second').is_not_null() & (pl.col('second')>=0) & (pl.col('second')<pl.col('steps')))
    # rounds present (any second in range)
    rounds_present=in_range.select(['match_key','round']).unique().height
    secs_present=in_range.height
    out_range=dist.join(rounds.select(['match_key','round_num','steps']).rename({'round_num':'round'}).with_columns(pl.col('round').cast(pl.Int32)), on=['match_key','round'], how='left')
    out_of_range=out_range.filter(pl.col('steps').is_null() | (pl.col('second')>=pl.col('steps')) | (pl.col('second')<0)).height
    # matches present
    mp=set(allp['match_key'].unique().to_list())
    # per-round stats
    per_round=in_range.group_by(['match_key','round']).agg(pl.len().alias('secs_present'))
    rr=rounds.select(['match_key','round_num','steps']).rename({'round_num':'round'}).with_columns(pl.col('round').cast(pl.Int64)).join(per_round.with_columns(pl.col('round').cast(pl.Int64)), on=['match_key','round'], how='left').with_columns(pl.col('secs_present').fill_null(0))
    full_rounds=rr.filter(pl.col('secs_present')>=pl.col('steps')).height
    partial=rr.filter((pl.col('secs_present')>0)&(pl.col('secs_present')<pl.col('steps'))).height
    # per-match table
    pm=rr.group_by('match_key').agg([pl.col('steps').sum().alias('total_seconds'), pl.col('secs_present').sum().alias('seconds_present'), (pl.col('secs_present')>0).sum().alias('rounds_present'), pl.len().alias('rounds_total')])
    pm=pm.join(files_per_key, on='match_key', how='left')
    pm.write_parquet(f'{S}/present_{src}_per_match.parquet')
    rr.write_parquet(f'{S}/present_{src}_per_round.parquet')
    pm_dates=pm.with_columns(pl.col('files').list.len().alias('nfiles'))
    summary[src]={
        'files_scanned':len(files),
        'files_with_batch_keys':len(file_rows),
        'batch_matches_present':len(mp),
        'batch_matches_present_all_rounds':int((pm['rounds_present']==pm['rounds_total']).sum()),
        'batch_rounds_present_any_second':rounds_present,
        'batch_rounds_full_steps':full_rounds,
        'batch_rounds_partial':partial,
        'batch_seconds_present_in_v4_range':secs_present,
        'rows_outside_v4_second_range_or_round_unknown':out_of_range,
        'files_holding_batch_keys_sample':[x for x in file_rows[:3]],
    }
    print(src, json.dumps(summary[src]), flush=True)
    # distribution of which file dates contain each key
    fdates=pm.with_columns(pl.col('files').list.join(',').alias('fj'))
    print(' top file-set patterns:', fdates.group_by('fj').agg(pl.len().alias('n')).sort('n', descending=True).head(6).to_dicts(), flush=True)
json.dump(summary, open(f'{S}/presence_local_{"_".join(sources)}.json','w'), indent=1, default=str)
print('done', flush=True)
