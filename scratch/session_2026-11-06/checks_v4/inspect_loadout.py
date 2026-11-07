import json, csv, collections
import numpy as np
V2 = '/home/sagemaker-user/adam/builder/final/export_clean_v2/'
V3 = '/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/'
V4 = '/home/sagemaker-user/adam/builder/final/export_clean_v4/'
for D in [V3, V4]:
    for nm in ['loadout_step_i16.npy', 'loadout_equip_i32.npy', 'loadout_round.npy', 'loadout_flag.npy', 'steps_alive.npy', 'round_match.npy', 'round_num.npy', 'round_freeze.npy', 'offsets.npy', 'match_split.npy']:
        try:
            a = np.load(D + nm, mmap_mode='r')
            print(D.split('/')[-2], nm, a.shape, a.dtype)
        except Exception as e:
            print(D.split('/')[-2], nm, 'ERR', type(e).__name__)
o2 = np.load(V2 + 'offsets.npy'); o3 = np.load(V3 + 'offsets.npy'); o4 = np.load(V4 + 'offsets.npy')
print('offsets v2==v3', np.array_equal(o2, o3), 'v2==v4', np.array_equal(o2, o4))
k2 = json.load(open(V2 + 'match_keys.json')); k3 = json.load(open(V3 + 'match_keys.json')); k4 = json.load(open(V4 + 'match_keys.json'))
print('match_keys v2==v3', k2 == k3, 'v2==v4', k2 == k4, 'counts', len(k2), len(k3), len(k4))
print('v4 built_match_keys length', len(json.load(open(V4 + 'built_match_keys.json'))), 'v3 built', len(json.load(open(V3 + 'built_match_keys.json'))))
for D in [V3, V4]:
    rows = list(csv.DictReader(open(D + 'match_loadout_status.csv', encoding='utf-8')))
    st = collections.Counter(r['build_status'] for r in rows)
    print(D.split('/')[-2], 'status rows', len(rows), 'status counts', dict(st), 'blank status', sum(1 for r in rows if not r['build_status'].strip()))
    print(D.split('/')[-2], 'columns', list(rows[0].keys()))
    print(D.split('/')[-2], 'match_key order equals match_keys.json', [r['match_key'] for r in rows] == (k4 if D == V4 else k3))
    print(D.split('/')[-2], 'v4_build_scope counts', dict(collections.Counter(r.get('v4_build_scope', '') for r in rows)))
    print(D.split('/')[-2], 'reason counts', dict(collections.Counter(r['reason'] for r in rows).most_common(12)))
for D in [V3, V4]:
    t = open(D + 'README.json', encoding='utf-8').read()
    hits = [t[max(0, i - 160):i + 80].replace('\n', ' ') for i in range(len(t)) if t.startswith('300', i)]
    print(D.split('/')[-2], 'README "300" hits', len(hits))
    for h in hits[:6]:
        print('   ...', h)
    names = json.load(open(D + 'loadout_names.json')) if (D + 'loadout_names.json') else None
