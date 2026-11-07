import os, re, collections
F = '/home/sagemaker-user/adam/builder/fixes/fix332/'
pat = re.compile(r'(\d{4})[/_-](\d{2})[/_-](\d{2})')
def bucket(d):
    m = pat.search(d)
    if not m:
        return 'nodate'
    y, mo, dd = m.groups()
    if (y, mo) != ('2026', '10'):
        return y + mo
    dd = int(dd)
    if 21 <= dd <= 25:
        return 'oct21-25'
    if 26 <= dd <= 31:
        return 'oct26-31'
    return 'oct-other:%02d' % dd
def count_file(name, first_col=True):
    c = collections.Counter()
    n = 0
    with open(F + name, errors='replace') as fh:
        for line in fh:
            n += 1
            c[bucket(line.split('\t')[0].split(',')[0].strip())] += 1
    return n, dict(c)
print('dl_list.tsv', count_file('dl_list.tsv'))
print('dl_jobs.txt', count_file('dl_jobs.txt'))
print('recover_list.csv', count_file('recover_list.csv'))
print('s3_folders_2026_10.txt (dates in listing)', count_file('s3_folders_2026_10.txt'))
raw = os.listdir(F + 'raw')
c = collections.Counter(bucket(d) for d in raw)
print('raw dirs', len(raw), dict(c))
files = 0
for d in raw:
    files += len(os.listdir(os.path.join(F, 'raw', d)))
print('files under raw', files)
print('download.sh size', os.path.getsize(F + 'download.sh'))
