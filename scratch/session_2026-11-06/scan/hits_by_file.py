import sys, json
sys.stdout.reconfigure(errors='replace', encoding='utf-8')
BS = chr(92)
want = set(sys.argv[2].split(','))
for line in open(sys.argv[1], encoding='utf-8', errors='replace'):
    if not line.startswith('{'):
        continue
    d = json.loads(line)
    h = d.get('hits', {})
    names = {n: v for c in h.values() for n, v in c.items() if n in want}
    if names:
        ex = {k: v for k, v in d.get('masked_examples', {}).items() if k in want}
        obj = d['obj'].replace(BS, '/')
        i = obj.find('s3mirror/')
        label = ('MIRROR:' + obj[i + 9:]) if i >= 0 else ('G:' + obj.split('csds-cs2/')[-1] if 'csds-cs2/' in obj else obj)
        print(label, names, ex if ex else '')
