import subprocess, re
REPO = r'C:\Users\<user>\Documents\pureskill\counterstrike-win-probability'
S = r'C:\Users\<user>\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad'
EMAIL = re.compile(rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
IPV4 = re.compile(rb'(?<![0-9.])(?:\d{1,3}\.){3}\d{1,3}(?![0-9.])')
rows = [l.rstrip('\n').split('\t') for l in open(S + r'\hist_blob_list.txt', encoding='utf-8') if l.strip()]
em = ip = 0
for commit, path in rows:
    blob = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=REPO, capture_output=True, check=True).stdout
    em += len(EMAIL.findall(blob)); ip += len(IPV4.findall(blob))
print('blobs:', len(rows), ' email-pattern hits:', em, ' ipv4-pattern hits:', ip)
