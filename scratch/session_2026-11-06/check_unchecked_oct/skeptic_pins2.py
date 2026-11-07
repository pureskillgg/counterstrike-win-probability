import sys
t=open(sys.argv[1], encoding="utf-8", errors="replace").read()
def show(anchor, before, after):
    i=t.find(anchor)
    print("=====", anchor, i)
    print(t[max(0,i-before):i+after].replace("\n"," | "))
    print()
show('"id": "425fca91', 0, 2400)
show('"id": "e53ed710', 0, 2000)
show('oct-2026-holdout-reuse-approved', 1400, 1100)
show('"id": "4edc74c3', 0, 1600)
