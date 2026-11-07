import glob, pyarrow.parquet as pq, polars as pl
base = 'G:/pureskill/csds-cs2/tomes/combined_test/tome/csds/'
for ch in ['weapon_fire','item_pickup','item_dropped','item_equip','round_end','round_state','player_death']:
    files = sorted(glob.glob(base + f'basic_{ch}.COMBINED/dataframe_*'))
    print(ch, 'pages:', len(files))
    if files:
        print('  schema:', [f"{n}:{t}" for n, t in zip(pq.read_schema(files[0]).names, pq.read_schema(files[0]).types)][:40])
