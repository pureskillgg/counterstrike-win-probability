import sys
s3 = {}
for line in open(sys.argv[1]):
    p = line.rstrip('\n').split('\t')
    if len(p) >= 3 and p[1].isdigit():
        s3[p[0][len('cs2data_gz/'):]] = (int(p[1]), p[2])
loc = {}
for line in open(sys.argv[2]):
    sz, _, rel = line.rstrip('\n').partition(' ')
    rel = rel[2:] if rel.startswith('./') else rel
    loc[rel] = int(sz)
same = [k for k in s3 if k in loc and loc[k] == s3[k][0]]
print('S3 objects', len(s3), '| local staged files', len(loc))
print('size-equal matches', len(same))
print('S3-only', [k for k in s3 if k not in loc][:20])
print('local-only', [k for k in loc if k not in s3][:20])
print('size-mismatch', [(k, s3[k][0], loc[k]) for k in s3 if k in loc and loc[k] != s3[k][0]][:20])
print('ETag sample (no multipart dash = single-part md5):', sum(1 for k in s3 if '-' not in s3[k][1]), 'single-part of', len(s3))
