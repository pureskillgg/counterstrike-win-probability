import sys, json, os
root = "/g/pureskill/csds-cs2/source_2024"
assign = json.load(open(sys.argv[1], encoding='utf-8'))['assign']
keys = sorted(k for k, v in assign.items() if v == 'train' and k.startswith('csds/2024/'))
k = next((x for x in keys if os.path.isdir(os.path.join(root, *x.split("/")[:-1]))), keys[0])
d = os.path.join(root, *k.split('/')[:-1])
print('train match dir exists:', os.path.isdir(d))
names = sorted(os.listdir(d)) if os.path.isdir(d) else []
print('event files in that match folder:', len(names))
print('event names:', names)
for ev in ['player_info']:
    p = os.path.join(d, ev)
    if not os.path.exists(p):
        print(ev, 'missing'); continue
    with open(p, 'rb') as fh:
        head = fh.read(4096)
    print(ev, 'size', os.path.getsize(p), 'magic:', head[:4])
    if head[:1] == b'{' or head[:1] == b'[':
        line = head.split(b'\n')[0]
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                print('  top-level keys:', sorted(rec.keys()))
                for kk, vv in rec.items():
                    if isinstance(vv, list) and vv and isinstance(vv[0], dict):
                        print('  nested list', kk, 'item keys:', sorted(vv[0].keys()))
                    if isinstance(vv, dict):
                        print('  nested dict', kk, 'keys:', sorted(vv.keys())[:20])
        except Exception as e:
            print('  first-line parse failed:', type(e).__name__)
