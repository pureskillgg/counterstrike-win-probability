import os, sys, re, collections
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
import pyarrow as pa, pyarrow.parquet as pq
ROOT = 'G:/pureskill/csds-cs2/evac_2026-10-13/tomes'
FLAG = {'basic_item_equip', 'basic_grenade_state', 'basic_item_pickup', 'basic_item_dropped', 'basic_bomb_state', 'basic_round_state', 'basic_player_info', 'basic_weapon_fire', 'basic_player_death', 'basic_other_death', 'basic_player_spawn', 'basic_round_start'}
PATS = {'at_sign': re.compile(r'@'), 'steam_text': re.compile(r'STEAM_|\[U:1:|7656119\d{7}'), 'url': re.compile(r'https?://|www\.'), 'ipv4': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), 'winuser_path': re.compile(r'C:[\/]+Users', re.I)}
counts = collections.Counter(); str_cols = collections.Counter(); files = 0; errs = 0
for dp, dn, fn in os.walk(ROOT):
    for f in fn:
        p = os.path.join(dp, f)
        parts = p.replace(chr(92), '/').split('/')
        if 'csds' not in parts: continue
        seg = parts[parts.index('csds') + 1]
        table = seg.split('.')[0]
        if table not in FLAG: continue
        with open(p, 'rb') as fh:
            if fh.read(4) != b'PAR1': continue
        try:
            sch = pq.read_schema(p)
            scols = [fld.name for fld in sch if pa.types.is_string(fld.type) or pa.types.is_large_string(fld.type)]
            if not scols: continue
            t = pq.read_table(p, columns=scols)
        except Exception:
            errs += 1; continue
        files += 1
        for c in scols:
            str_cols[(table, c)] += 1
            for v in t.column(c).to_pylist():
                if not v: continue
                for k, rx in PATS.items():
                    if rx.search(v): counts[(table, c, k)] += 1
print('parquet files read', files, 'errors', errs)
print('string columns in flagged tables (table, column): files', dict(str_cols))
print('identifier-shaped value counts (table, column, pattern):', dict(counts) if counts else 'none')
