import glob, polars as pl, json
base = 'G:/pureskill/csds-cs2/tomes/combined_test/tome/csds/'
def load(ch, cols):
    files = sorted(glob.glob(base + f'basic_{ch}.COMBINED/dataframe_*'))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how='diagonal_relaxed')
wf = load('weapon_fire', ['player_weapon_code','weapon_name'])
print("weapon_fire rows:", wf.height)
m_wf = wf.group_by(['player_weapon_code','weapon_name']).agg(pl.len().alias('n')).sort(['player_weapon_code','n'], descending=[False,True])
print("weapon_fire (code, name, n):")
print(m_wf.to_pandas().to_string(index=False) if False else m_wf.write_csv(None))
ip = load('item_pickup', ['def_index','item'])
print("item_pickup (def_index, item, n):")
print(ip.group_by(['def_index','item']).agg(pl.len().alias('n')).sort('def_index').write_csv(None))
idr = load('item_dropped', ['def_index','item'])
print("item_dropped (def_index, item, n):")
print(idr.group_by(['def_index','item']).agg(pl.len().alias('n')).sort('def_index').write_csv(None))
